#!/usr/bin/env python3
"""Select a fresh token-local operand reader for the frozen SmolLM2 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from run_phase3_native_boundary import verify_sha256
from run_phase4_carry_sequence_boundary import value_sha256
from run_phase8_operand_reader_selection import (
    flatten_states_and_labels,
    passes_gate,
    reader_metrics,
    render_and_locate,
)
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.operand_reader import (
    fit_nearest_centroid_digit_reader,
)
from open_latent_interfaces.phase13_data import (
    build_phase13_examples,
    phase13_sha256,
)
from open_latent_interfaces.prefill import verify_decimal_digit_contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    args = parser.parse_args()
    if args.output.exists() or args.artifact_output.exists():
        raise SystemExit("refusing to overwrite reader result or artifact")

    config = json.loads(args.config.read_text())
    if str(args.output) != config["output"]:
        raise SystemExit("reader output differs from frozen path")
    if str(args.artifact_output) != config["artifact_output"]:
        raise SystemExit("reader artifact differs from frozen path")
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if runner_hash != config["runner_sha256"]:
        raise SystemExit("reader runner hash mismatch")
    for dependency, expected_hash in config["code_dependencies"].items():
        verify_sha256(Path(dependency), expected_hash)

    dataset_path = Path(config["dataset_config"])
    verify_sha256(dataset_path, config["dataset_config_sha256"])
    dataset_config = json.loads(dataset_path.read_text())
    if dataset_config.get("audit_authorized", False):
        raise SystemExit("reader selection may not use sealed audit data")
    onboarding_path = Path(config["onboarding_result"])
    verify_sha256(onboarding_path, config["onboarding_result_sha256"])
    if json.loads(onboarding_path.read_text()).get("passes") is not True:
        raise SystemExit("model onboarding did not pass")
    capability_path = Path(config["capability_result"])
    verify_sha256(capability_path, config["capability_result_sha256"])
    capability = json.loads(capability_path.read_text())
    if capability.get("status") != "exposed_fit_measurement":
        raise SystemExit("capability result has unexpected status")

    examples = build_phase13_examples(
        **dataset_config["dataset"]["parameters"]
    )
    observed_dataset_hash = phase13_sha256(examples)
    if observed_dataset_hash != dataset_config["dataset"]["sha256"]:
        raise SystemExit("Phase 13 dataset hash mismatch")
    split_examples = {
        split: [row for row in examples if row.split == split]
        for split in ("fit", "selection")
    }
    for split, rows in split_examples.items():
        if value_sha256([row.example_id for row in rows]) != config[
            f"{split}_examples_sha256"
        ]:
            raise SystemExit(f"{split} example hash mismatch")

    model_config = dataset_config["model"]
    if model_config != config["model"]:
        raise SystemExit("reader model differs from frozen model")
    if capability.get("model") != model_config:
        raise SystemExit("capability result used a different model")
    if capability.get("dataset_sha256") != observed_dataset_hash:
        raise SystemExit("capability result used a different dataset")

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rendered: dict[str, list[str]] = {}
    positions: dict[str, list[tuple[int, ...]]] = {}
    for split, rows in split_examples.items():
        prompts, token_positions, contract = render_and_locate(
            tokenizer,
            rows,
            dataset_config["assistant_prefix"],
        )
        if value_sha256(prompts) != config[
            f"{split}_rendered_prompts_sha256"
        ]:
            raise SystemExit(f"{split} rendered prompt hash mismatch")
        if value_sha256(contract) != config[
            f"{split}_token_contract_sha256"
        ]:
            raise SystemExit(f"{split} token contract mismatch")
        rendered[split] = prompts
        positions[split] = token_positions
    digit_token_ids = verify_decimal_digit_contract(
        tokenizer,
        rendered["fit"][0],
    )
    if value_sha256(digit_token_ids) != config["digit_token_ids_sha256"]:
        raise SystemExit("digit-token map hash mismatch")

    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        dtype=getattr(torch, args.dtype),
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("base model parameters must remain frozen")

    candidate_indices = config["candidate_hidden_state_indices"]
    if candidate_indices != sorted(set(candidate_indices)):
        raise SystemExit("candidate hidden-state indices must be unique and sorted")
    expected_hidden_state_count = config["expected_hidden_state_count"]
    if not candidate_indices or min(candidate_indices) < 1:
        raise SystemExit("candidate hidden-state indices must be block outputs")
    if max(candidate_indices) >= expected_hidden_state_count:
        raise SystemExit("candidate hidden-state index exceeds frozen model")

    capture = ActivationCapture(model, tokenizer, device=device)
    started = time.perf_counter()
    captured = {
        split: capture.capture_token_positions(
            rendered[split],
            positions[split],
            hidden_state_indices=candidate_indices,
            batch_size=config["base_model_batch_size"],
        )
        for split in ("fit", "selection")
    }

    candidates = []
    fitted = {}
    for hidden_index in candidate_indices:
        fit_states, fit_digits = flatten_states_and_labels(
            captured["fit"][hidden_index].values,
            split_examples["fit"],
        )
        reader, counts = fit_nearest_centroid_digit_reader(
            fit_states,
            fit_digits,
        )
        selection_states, _ = flatten_states_and_labels(
            captured["selection"][hidden_index].values,
            split_examples["selection"],
        )
        flat_predictions = reader.predict(selection_states).tolist()
        predictions = []
        cursor = 0
        for token_positions in positions["selection"]:
            width = len(token_positions)
            predictions.append(flat_predictions[cursor : cursor + width])
            cursor += width
        if cursor != len(flat_predictions):
            raise RuntimeError("selection predictions were not fully consumed")
        rotated_predictions = [
            [(digit + 1) % 10 for digit in row] for row in predictions
        ]
        target_metrics = reader_metrics(
            predictions,
            split_examples["selection"],
        )
        rotated_metrics = reader_metrics(
            rotated_predictions,
            split_examples["selection"],
        )
        passes = passes_gate(
            target_metrics,
            rotated_metrics,
            config["selection_rule"],
        )
        candidates.append(
            {
                "hidden_state_index": hidden_index,
                "fit_class_counts": counts.tolist(),
                "target": target_metrics,
                "rotated_label_control": rotated_metrics,
                "passes": passes,
            }
        )
        fitted[hidden_index] = (reader, counts)

    passing = [
        row["hidden_state_index"] for row in candidates if row["passes"]
    ]
    selected_index = min(passing) if passing else max(
        candidate_indices,
        key=lambda hidden_index: next(
            row["target"]["pair_accuracy"]
            for row in candidates
            if row["hidden_state_index"] == hidden_index
        ),
    )
    passes = bool(passing)
    selected_reader, selected_counts = fitted[selected_index]

    args.artifact_output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "digit_classes": selected_reader.classes.contiguous(),
            "digit_centroids": selected_reader.centroids.contiguous(),
            "fit_class_counts": selected_counts.contiguous(),
        },
        str(args.artifact_output),
        metadata={
            "schema_version": "oli.operand-digit-reader-tensors/v1",
            "model_id": model_config["id"],
            "model_revision": model_config["revision"],
            "hidden_state_index": str(selected_index),
            "source_phase": "phase13",
        },
    )
    artifact_hash = hashlib.sha256(args.artifact_output.read_bytes()).hexdigest()
    report = {
        "schema_version": "oli.phase13-smollm2-operand-reader-selection/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "selection_only",
        "model": model_config,
        "dataset_sha256": observed_dataset_hash,
        "candidate_hidden_state_indices": candidate_indices,
        "selection_rule": config["selection_rule"],
        "candidates": candidates,
        "selection": {
            "hidden_state_index": selected_index,
            "passes": passes,
        },
        "passes": passes,
        "artifact": {
            "path": str(args.artifact_output),
            "sha256": artifact_hash,
            "width": selected_reader.residual_width,
        },
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "runner_sha256": runner_hash,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Selection-only SmolLM2 token-local digit decoding under an "
            "external operand-token locator. The reader was fitted and "
            "selected only on fresh Phase 13 fit/selection data. No writer, "
            "integrated, development, audit, or cross-model tensor claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    print(f"wrote {args.artifact_output}")


if __name__ == "__main__":
    main()
