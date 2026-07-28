#!/usr/bin/env python3
"""Refit and select Phi native digit prototypes on the wide distribution."""

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
from run_phase3_closed_loop_development import wrong_all_digits
from run_phase3_native_boundary import (
    predict_with_delta,
    prefix_prompts,
    render_examples,
    value_list_sha256,
    verify_sha256,
)
from run_phase3_prototype_selection import (
    fit_position_prototypes,
    prototype_delta,
    scale_and_gate,
)
from run_phase4_carry_sequence_boundary import value_sha256
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.evaluation import norm_match
from open_latent_interfaces.phase2_data import balanced_counterfactual_results
from open_latent_interfaces.phase3_data import (
    build_phase3_additions,
    phase3_addition_sha256,
)
from open_latent_interfaces.phase7_data import (
    build_phase7_carry_quartets,
    phase7_carry_sha256,
)
from open_latent_interfaces.prefill import verify_decimal_digit_contract


def evaluate_ids(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    delta: torch.Tensor,
    *,
    expected_ids: list[int],
    hidden_state_index: int,
    digit_token_ids: dict[int, int],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    logits = predict_with_delta(
        model,
        tokenizer,
        prompts,
        delta,
        hidden_state_index=hidden_state_index,
        batch_size=batch_size,
        device=device,
    )
    predicted = logits.argmax(dim=1).tolist()
    correct = sum(
        actual == expected
        for actual, expected in zip(predicted, expected_ids, strict=True)
    )
    return {
        "n": len(expected_ids),
        "correct": correct,
        "accuracy": correct / len(expected_ids),
        "digit_token_rate": sum(
            token_id in set(digit_token_ids.values()) for token_id in predicted
        )
        / len(predicted),
        "predicted_token_ids": predicted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prototype-output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()
    if args.output.exists() or args.prototype_output.exists():
        raise SystemExit("refusing to overwrite writer result or artifact")

    config = json.loads(args.config.read_text())
    paths = {
        "dataset_config": Path(config["dataset_config"]),
        "behavior_result": Path(config["behavior_result"]),
        "basis_artifact": Path(config["basis_artifact"]),
        "phase3_dataset_config": Path(config["phase3_dataset_config"]),
        "phase3_behavior_result": Path(config["phase3_behavior_result"]),
    }
    for name, path in paths.items():
        verify_sha256(path, config[f"{name}_sha256"])
    dataset_config = json.loads(paths["dataset_config"].read_text())
    behavior = json.loads(paths["behavior_result"].read_text())
    phase3_dataset_config = json.loads(
        paths["phase3_dataset_config"].read_text()
    )
    phase3_behavior = json.loads(paths["phase3_behavior_result"].read_text())
    if dataset_config.get("audit_authorized", False):
        raise SystemExit("writer selection requires a sealed audit")
    if not behavior["passes"]:
        raise SystemExit("wide-distribution behavior gate did not pass")
    examples = build_phase7_carry_quartets(
        **dataset_config["dataset"]["parameters"]
    )
    observed_dataset_hash = phase7_carry_sha256(examples)
    if observed_dataset_hash != dataset_config["dataset"]["sha256"]:
        raise SystemExit("Phase 9 dataset hash mismatch")
    exact_ids = sorted(
        row["example_id"]
        for row in behavior["rows"]
        if row["split"] == "fit" and row["exact"]
    )
    if value_sha256(exact_ids) != config["eligible_fit_examples_sha256"]:
        raise SystemExit("eligible fit example hash mismatch")
    exact_id_set = set(exact_ids)
    wide_fit = [
        row
        for row in examples
        if row.split == "fit" and row.example_id in exact_id_set
    ]
    phase3_examples = build_phase3_additions(
        **phase3_dataset_config["dataset"]["parameters"]
    )
    if phase3_addition_sha256(phase3_examples) != phase3_dataset_config[
        "dataset"
    ]["sha256"]:
        raise SystemExit("Phase 3 source dataset hash mismatch")
    if (
        phase3_dataset_config["model"] != dataset_config["model"]
        or not phase3_behavior["passes"]
    ):
        raise SystemExit("Phase 3 source model or behavior mismatch")
    phase3_exact_ids = sorted(
        row["example_id"]
        for row in phase3_behavior["rows"]
        if row["split"] == "fit" and row["exact"]
    )
    if value_sha256(phase3_exact_ids) != config[
        "phase3_eligible_fit_examples_sha256"
    ]:
        raise SystemExit("Phase 3 eligible fit example hash mismatch")
    phase3_exact_set = set(phase3_exact_ids)
    phase3_fit = [
        row
        for row in phase3_examples
        if row.split == "fit" and row.example_id in phase3_exact_set
    ]
    fit = phase3_fit + wide_fit
    selection = [row for row in examples if row.split == "selection"]
    if value_sha256([row.example_id for row in selection]) != config[
        "selection_examples_sha256"
    ]:
        raise SystemExit("selection example hash mismatch")
    targets = balanced_counterfactual_results(selection)
    if value_list_sha256(targets) != config["selection_targets_sha256"]:
        raise SystemExit("selection target hash mismatch")
    originals = [row.result for row in selection]
    wrong_targets = wrong_all_digits(targets)

    basis_tensors = load_file(str(paths["basis_artifact"]))
    bases = {
        0: basis_tensors["leading_basis"][: config["ranks"]["0"]].float(),
        1: basis_tensors["suffix_basis"][: config["ranks"]["1"]].float(),
        2: basis_tensors["suffix_basis"][: config["ranks"]["2"]].float(),
    }
    hidden_indices = {
        int(key): value for key, value in config["hidden_state_indices"].items()
    }
    model_config = dataset_config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rendered_phase3_fit = render_examples(
        tokenizer,
        phase3_fit,
        assistant_prefix=phase3_dataset_config["assistant_prefix"],
    )
    rendered_wide_fit = render_examples(
        tokenizer,
        wide_fit,
        assistant_prefix=dataset_config["assistant_prefix"],
    )
    rendered_fit = rendered_phase3_fit + rendered_wide_fit
    rendered_selection = render_examples(
        tokenizer,
        selection,
        assistant_prefix=dataset_config["assistant_prefix"],
    )
    digit_token_ids = verify_decimal_digit_contract(tokenizer, rendered_fit[0])

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
    started = time.perf_counter()

    prototypes = {}
    counts_by_position = {}
    fit_hashes = {}
    artifact_tensors = {}
    for position in range(3):
        prototype, counts, hashes = fit_position_prototypes(
            capture,
            examples=fit,
            rendered=rendered_fit,
            basis=bases[position],
            position=position,
            hidden_state_index=hidden_indices[position],
            batch_size=config["base_model_batch_size"],
        )
        allowed = range(1, 10) if position == 0 else range(10)
        if min(int(counts[digit]) for digit in allowed) < config[
            "minimum_fit_examples_per_digit"
        ]:
            raise SystemExit(f"insufficient fit support at position {position}")
        prototypes[position] = prototype
        counts_by_position[position] = counts
        fit_hashes[str(position)] = hashes
        key = (
            "leading_digit"
            if position == 0
            else ("tens_digit" if position == 1 else "ones_digit")
        )
        artifact_tensors[key] = prototype.contiguous()
        artifact_tensors[f"position_{position}_counts"] = counts.contiguous()

    args.prototype_output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        artifact_tensors,
        str(args.prototype_output),
        metadata={
            "schema_version": "oli.phase9-wide-writer-prototypes/v1",
            "model_id": model_config["id"],
            "model_revision": model_config["revision"],
            "basis_sha256": config["basis_artifact_sha256"],
        },
    )
    artifact_hash = hashlib.sha256(args.prototype_output.read_bytes()).hexdigest()

    rule = config["selection_rule"]
    positions = {}
    all_pass = True
    for position in range(3):
        hidden_index = hidden_indices[position]
        target_prompts = prefix_prompts(
            rendered_selection,
            targets,
            position=position,
        )
        identity_prompts = prefix_prompts(
            rendered_selection,
            originals,
            position=position,
        )
        target_states = capture.capture_last_token(
            target_prompts,
            hidden_state_indices=[hidden_index],
            batch_size=config["base_model_batch_size"],
        )[hidden_index].values.float()
        identity_states = capture.capture_last_token(
            identity_prompts,
            hidden_state_indices=[hidden_index],
            batch_size=config["base_model_batch_size"],
        )[hidden_index].values.float()
        target_logits = capture.next_token_logits(
            target_prompts,
            batch_size=config["base_model_batch_size"],
        )
        identity_logits = capture.next_token_logits(
            identity_prompts,
            batch_size=config["base_model_batch_size"],
        )
        target_expected = [
            digit_token_ids[int(str(value)[position])] for value in targets
        ]
        identity_expected = [
            digit_token_ids[int(str(value)[position])] for value in originals
        ]
        raw_target = prototype_delta(
            target_states,
            targets,
            prototypes[position],
            bases[position],
            position=position,
        )
        raw_identity = prototype_delta(
            identity_states,
            originals,
            prototypes[position],
            bases[position],
            position=position,
        )
        raw_wrong = prototype_delta(
            target_states,
            wrong_targets,
            prototypes[position],
            bases[position],
            position=position,
        )
        generator = torch.Generator().manual_seed(
            config["random_control_seed"] + position
        )
        random_coefficients = torch.randn(
            (len(selection), bases[position].shape[0]),
            generator=generator,
        )
        raw_random = random_coefficients @ bases[position]
        scale_rows = {}
        passing_scales = []
        for scale in config["scales"]:
            target_delta, _ = scale_and_gate(
                raw_target,
                target_states,
                target_logits,
                torch.tensor(target_expected),
                scale=scale,
                norm_cap=config["norm_cap"],
            )
            identity_delta, _ = scale_and_gate(
                raw_identity,
                identity_states,
                identity_logits,
                torch.tensor(identity_expected),
                scale=scale,
                norm_cap=config["norm_cap"],
            )
            norms = target_delta.norm(dim=1)
            wrong_delta = norm_match(raw_wrong, norms)
            random_delta = norm_match(raw_random, norms)
            metrics = {
                "target": evaluate_ids(
                    model,
                    tokenizer,
                    target_prompts,
                    target_delta,
                    expected_ids=target_expected,
                    hidden_state_index=hidden_index,
                    digit_token_ids=digit_token_ids,
                    batch_size=config["base_model_batch_size"],
                    device=device,
                ),
                "identity": evaluate_ids(
                    model,
                    tokenizer,
                    identity_prompts,
                    identity_delta,
                    expected_ids=identity_expected,
                    hidden_state_index=hidden_index,
                    digit_token_ids=digit_token_ids,
                    batch_size=config["base_model_batch_size"],
                    device=device,
                ),
                "wrong_digit_norm_matched": evaluate_ids(
                    model,
                    tokenizer,
                    target_prompts,
                    wrong_delta,
                    expected_ids=target_expected,
                    hidden_state_index=hidden_index,
                    digit_token_ids=digit_token_ids,
                    batch_size=config["base_model_batch_size"],
                    device=device,
                ),
                "random_norm_matched": evaluate_ids(
                    model,
                    tokenizer,
                    target_prompts,
                    random_delta,
                    expected_ids=target_expected,
                    hidden_state_index=hidden_index,
                    digit_token_ids=digit_token_ids,
                    batch_size=config["base_model_batch_size"],
                    device=device,
                ),
            }
            strongest_control = max(
                metrics["wrong_digit_norm_matched"]["accuracy"],
                metrics["random_norm_matched"]["accuracy"],
            )
            advantage = metrics["target"]["accuracy"] - strongest_control
            passes = (
                metrics["target"]["accuracy"]
                >= rule["minimum_target_accuracy"]
                and metrics["identity"]["accuracy"]
                >= rule["minimum_identity_accuracy"]
                and advantage >= rule["minimum_control_advantage"]
                and (
                    not rule["require_digit_token_rate"]
                    or metrics["target"]["digit_token_rate"] == 1.0
                )
            )
            metrics["gate"] = {
                "strongest_control_accuracy": strongest_control,
                "control_advantage": advantage,
                "passes": passes,
            }
            metrics["mean_target_relative_norm"] = float(
                (target_delta.norm(dim=1) / target_states.norm(dim=1)).mean()
            )
            scale_rows[str(scale)] = metrics
            if passes:
                passing_scales.append(float(scale))
        selected_scale = min(passing_scales) if passing_scales else max(
            config["scales"],
            key=lambda scale: (
                scale_rows[str(scale)]["target"]["accuracy"],
                scale_rows[str(scale)]["gate"]["control_advantage"],
                -float(scale),
            ),
        )
        position_passes = bool(passing_scales)
        all_pass &= position_passes
        positions[str(position)] = {
            "hidden_state_index": hidden_index,
            "rank": bases[position].shape[0],
            "fit_class_counts": counts_by_position[position].tolist(),
            "metrics": scale_rows,
            "selection": {
                "scale": selected_scale,
                "passes": position_passes,
            },
        }

    report = {
        "schema_version": "oli.phase9-phi-wide-writer-selection/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "selection_only",
        "model": model_config,
        "dataset_sha256": observed_dataset_hash,
        "behavior_result_sha256": config["behavior_result_sha256"],
        "eligible_fit_examples": len(fit),
        "fit_sources": {
            "phase3_exact": len(phase3_fit),
            "wide_exact": len(wide_fit),
            "phase3_eligible_fit_examples_sha256": config[
                "phase3_eligible_fit_examples_sha256"
            ],
            "wide_eligible_fit_examples_sha256": config[
                "eligible_fit_examples_sha256"
            ],
        },
        "eligible_fit_examples_sha256": config[
            "eligible_fit_examples_sha256"
        ],
        "basis_artifact_sha256": config["basis_artifact_sha256"],
        "fit_hashes": fit_hashes,
        "selection_targets_sha256": config["selection_targets_sha256"],
        "positions": positions,
        "passes": all_pass,
        "artifact": {
            "path": str(args.prototype_output),
            "sha256": artifact_hash,
        },
        "selection_rule": rule,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Selection-only wide-distribution prototype refit on fixed audited "
            "Phi bases. No new audit claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    print(f"wrote {args.prototype_output}")


if __name__ == "__main__":
    main()
