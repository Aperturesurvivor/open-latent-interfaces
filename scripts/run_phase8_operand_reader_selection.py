#!/usr/bin/env python3
"""Select a token-local operand digit reader for a frozen causal LM."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from run_phase4_carry_sequence_boundary import value_sha256, verify_sha256
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.operand_reader import (
    fit_nearest_centroid_digit_reader,
    locate_operand_digit_tokens,
    reconstruct_decimal_digits,
)
from open_latent_interfaces.phase7_data import (
    build_phase7_carry_quartets,
    phase7_carry_sha256,
)
from open_latent_interfaces.prefill import render_prefilled_chat


def render_and_locate(
    tokenizer: Any,
    examples: list[Any],
    assistant_prefix: str,
) -> tuple[list[str], list[tuple[int, ...]], list[list[Any]]]:
    rendered = []
    positions = []
    contract = []
    for example in examples:
        prompt = render_prefilled_chat(
            tokenizer,
            example.prompt,
            assistant_prefix=assistant_prefix,
        )
        located = locate_operand_digit_tokens(
            tokenizer,
            prompt,
            example.prompt,
            example.operand_a,
            example.operand_b,
        )
        rendered.append(prompt)
        positions.append(located.operand_a + located.operand_b)
        contract.append(
            [
                len(tokenizer(prompt)["input_ids"]),
                list(located.operand_a),
                list(located.operand_b),
            ]
        )
    return rendered, positions, contract


def flatten_states_and_labels(
    values: tuple[torch.Tensor, ...],
    examples: list[Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    states = []
    labels = []
    for selected, example in zip(values, examples, strict=True):
        digits = [int(value) for value in f"{example.operand_a}{example.operand_b}"]
        if selected.shape[0] != len(digits):
            raise ValueError("captured states do not align with operand digits")
        states.append(selected)
        labels.extend(digits)
    return torch.cat(states), torch.tensor(labels, dtype=torch.int64)


def reader_metrics(
    predictions: list[list[int]],
    examples: list[Any],
) -> dict[str, Any]:
    digit_correct = 0
    digit_count = 0
    operand_a_correct = 0
    operand_b_correct = 0
    pair_correct = 0
    rows = []
    for predicted, example in zip(predictions, examples, strict=True):
        a_width = len(str(example.operand_a))
        b_width = len(str(example.operand_b))
        if len(predicted) != a_width + b_width:
            raise ValueError("prediction width does not match operands")
        predicted_a_digits = predicted[:a_width]
        predicted_b_digits = predicted[a_width:]
        predicted_a = reconstruct_decimal_digits(predicted_a_digits)
        predicted_b = reconstruct_decimal_digits(predicted_b_digits)
        expected_digits = [
            int(value) for value in f"{example.operand_a}{example.operand_b}"
        ]
        digit_correct += sum(
            actual == expected
            for actual, expected in zip(
                predicted,
                expected_digits,
                strict=True,
            )
        )
        digit_count += len(expected_digits)
        a_exact = predicted_a == example.operand_a
        b_exact = predicted_b == example.operand_b
        operand_a_correct += a_exact
        operand_b_correct += b_exact
        pair_correct += a_exact and b_exact
        rows.append(
            {
                "example_id": example.example_id,
                "predicted_operand_a": predicted_a,
                "predicted_operand_b": predicted_b,
                "operand_a_exact": a_exact,
                "operand_b_exact": b_exact,
            }
        )
    n = len(examples)
    return {
        "n": n,
        "digit_count": digit_count,
        "digit_correct": digit_correct,
        "digit_accuracy": digit_correct / digit_count,
        "operand_a_correct": operand_a_correct,
        "operand_a_accuracy": operand_a_correct / n,
        "operand_b_correct": operand_b_correct,
        "operand_b_accuracy": operand_b_correct / n,
        "pair_correct": pair_correct,
        "pair_accuracy": pair_correct / n,
        "rows": rows,
    }


def passes_gate(
    target: dict[str, Any],
    rotated: dict[str, Any],
    rule: dict[str, float],
) -> bool:
    return (
        target["digit_accuracy"] >= rule["minimum_digit_accuracy"]
        and target["operand_a_accuracy"] >= rule["minimum_operand_a_accuracy"]
        and target["operand_b_accuracy"] >= rule["minimum_operand_b_accuracy"]
        and target["pair_accuracy"] >= rule["minimum_pair_accuracy"]
        and rotated["pair_accuracy"] <= rule["maximum_rotated_pair_accuracy"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()
    if args.output.exists() or args.artifact_output.exists():
        raise SystemExit("refusing to overwrite reader result or artifact")

    config = json.loads(args.config.read_text())
    if "runner_sha256" in config:
        verify_sha256(Path(__file__), config["runner_sha256"])
    dataset_path = Path(config["dataset_config"])
    verify_sha256(dataset_path, config["dataset_config_sha256"])
    dataset_config = json.loads(dataset_path.read_text())
    if dataset_config.get("audit_authorized", False):
        raise SystemExit("reader selection requires a sealed audit")
    examples = build_phase7_carry_quartets(
        **dataset_config["dataset"]["parameters"]
    )
    observed_dataset_hash = phase7_carry_sha256(examples)
    if observed_dataset_hash != dataset_config["dataset"]["sha256"]:
        raise SystemExit("Phase 8 dataset hash mismatch")
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
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rendered = {}
    positions = {}
    for split, rows in split_examples.items():
        prompts, token_positions, contract = render_and_locate(
            tokenizer,
            rows,
            dataset_config["assistant_prefix"],
        )
        if value_sha256(contract) != config[
            f"{split}_token_contract_sha256"
        ]:
            raise SystemExit(f"{split} token contract mismatch")
        rendered[split] = prompts
        positions[split] = token_positions

    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        torch_dtype=getattr(torch, args.dtype),
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    capture = ActivationCapture(model, tokenizer, device=device)
    candidate_indices = config["candidate_hidden_state_indices"]
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
            raise ValueError("selection predictions were not fully consumed")
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
        },
    )
    artifact_hash = hashlib.sha256(args.artifact_output.read_bytes()).hexdigest()
    report = {
        "schema_version": config.get(
            "result_schema_version",
            "oli.phase8-phi-operand-reader-selection/v1",
        ),
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
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Selection-only token-local latent digit decoding under an "
            "external operand-token locator. No integrated or audit claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    print(f"wrote {args.artifact_output}")


if __name__ == "__main__":
    main()
