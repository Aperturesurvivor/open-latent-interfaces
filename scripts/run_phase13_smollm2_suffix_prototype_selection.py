#!/usr/bin/env python3
"""Fit and select SmolLM2-native suffix digit prototypes."""

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
from run_phase3_native_boundary import (
    predict_with_delta,
    prefix_prompts,
    render_examples,
    value_list_sha256,
    verify_sha256,
)
from run_phase4_carry_sequence_boundary import value_sha256
from run_phase9_leading_causal_compiler import evaluate_logits
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.evaluation import norm_match
from open_latent_interfaces.phase2_data import balanced_counterfactual_results
from open_latent_interfaces.phase13_data import (
    build_phase13_examples,
    phase13_sha256,
)
from open_latent_interfaces.prefill import verify_decimal_digit_contract


def rotate_position(values: list[int], *, position: int) -> list[int]:
    rotated = []
    for value in values:
        digits = list(str(value))
        if len(digits) != 3 or position not in (1, 2):
            raise ValueError("suffix rotation requires a three-digit value")
        digits[position] = str((int(digits[position]) + 1) % 10)
        rotated.append(int("".join(digits)))
    return rotated


def fit_digit_subspace(
    states: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if states.ndim != 2 or labels.shape != (states.shape[0],):
        raise ValueError("states and labels must align")
    centroids = torch.stack(
        [states[labels == digit].float().mean(dim=0) for digit in range(10)]
    )
    counts = torch.tensor(
        [int((labels == digit).sum()) for digit in range(10)],
        dtype=torch.int64,
    )
    centered = centroids - centroids.mean(dim=0, keepdim=True)
    _, _, basis = torch.linalg.svd(centered, full_matrices=False)
    return basis[:9].contiguous(), centroids.contiguous(), counts


def prototype_delta(
    states: torch.Tensor,
    desired_digits: torch.Tensor,
    centroids: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    current = states.float() @ basis.T
    desired = centroids[desired_digits].float() @ basis.T
    return (desired - current) @ basis


def scale_and_gate(
    raw_delta: torch.Tensor,
    states: torch.Tensor,
    base_logits: torch.Tensor,
    expected_ids: torch.Tensor,
    *,
    scale: float,
    norm_cap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    scaled = raw_delta * scale
    maximum = states.norm(dim=1).clamp_min(1e-12) * norm_cap
    factors = (
        maximum / scaled.norm(dim=1).clamp_min(1e-12)
    ).clamp(max=1.0)
    scaled = scaled * factors[:, None]
    hard_gate = base_logits.argmax(dim=1) == expected_ids
    scaled[hard_gate] = 0
    return scaled, hard_gate


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
        raise SystemExit("refusing to overwrite suffix result or artifact")

    config = json.loads(args.config.read_text())
    if str(args.output) != config["output"]:
        raise SystemExit("suffix result differs from frozen path")
    if str(args.artifact_output) != config["artifact_output"]:
        raise SystemExit("suffix artifact differs from frozen path")
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if runner_hash != config["runner_sha256"]:
        raise SystemExit("suffix runner hash mismatch")
    for dependency, expected_hash in config["code_dependencies"].items():
        verify_sha256(Path(dependency), expected_hash)

    dataset_path = Path(config["dataset_config"])
    verify_sha256(dataset_path, config["dataset_config_sha256"])
    dataset_config = json.loads(dataset_path.read_text())
    if dataset_config.get("audit_authorized", False):
        raise SystemExit("suffix selection may not use sealed audit data")
    for result_name in ("onboarding_result", "reader_result", "leading_result"):
        result_path = Path(config[result_name])
        verify_sha256(result_path, config[f"{result_name}_sha256"])
        result = json.loads(result_path.read_text())
        if result.get("passes", result.get("selection", {}).get("passes")) is not True:
            raise SystemExit(f"{result_name} did not pass")

    examples = build_phase13_examples(
        **dataset_config["dataset"]["parameters"]
    )
    dataset_hash = phase13_sha256(examples)
    if dataset_hash != dataset_config["dataset"]["sha256"]:
        raise SystemExit("Phase 13 dataset hash mismatch")
    fit = [row for row in examples if row.split == "fit"]
    selection = [row for row in examples if row.split == "selection"]
    for split, rows in (("fit", fit), ("selection", selection)):
        if value_sha256([row.example_id for row in rows]) != config[
            f"{split}_examples_sha256"
        ]:
            raise SystemExit(f"{split} example hash mismatch")
    targets = balanced_counterfactual_results(selection)
    if value_list_sha256(targets) != config["selection_targets_sha256"]:
        raise SystemExit("selection target hash mismatch")
    originals = [row.result for row in selection]

    model_config = dataset_config["model"]
    if model_config != config["model"]:
        raise SystemExit("suffix model differs from frozen model")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rendered_fit = render_examples(
        tokenizer,
        fit,
        assistant_prefix=dataset_config["assistant_prefix"],
    )
    rendered_selection = render_examples(
        tokenizer,
        selection,
        assistant_prefix=dataset_config["assistant_prefix"],
    )
    rendered_contract = {
        "fit": rendered_fit,
        "selection": rendered_selection,
    }
    if value_sha256(rendered_contract) != config[
        "rendered_prompts_sha256"
    ]:
        raise SystemExit("rendered prompt hash mismatch")
    digit_token_ids = verify_decimal_digit_contract(
        tokenizer,
        rendered_fit[0],
    )
    if value_sha256(digit_token_ids) != config["digit_token_ids_sha256"]:
        raise SystemExit("digit-token map hash mismatch")

    prompt_contract: dict[str, Any] = {}
    for position in (1, 2):
        wrong = rotate_position(targets, position=position)
        prompt_contract[str(position)] = {
            "fit": prefix_prompts(
                rendered_fit,
                [row.result for row in fit],
                position=position,
            ),
            "target": prefix_prompts(
                rendered_selection,
                targets,
                position=position,
            ),
            "identity": prefix_prompts(
                rendered_selection,
                originals,
                position=position,
            ),
            "target_digits": [
                int(str(value)[position]) for value in targets
            ],
            "identity_digits": [
                int(str(value)[position]) for value in originals
            ],
            "wrong_digits": [int(str(value)[position]) for value in wrong],
        }
    if value_sha256(prompt_contract) != config["prompt_contract_sha256"]:
        raise SystemExit("suffix prompt contract hash mismatch")

    hidden_state_index = config["hidden_state_index"]
    if not 1 <= hidden_state_index < config["expected_hidden_state_count"]:
        raise SystemExit("suffix boundary exceeds frozen model")
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
    capture = ActivationCapture(model, tokenizer, device=device)
    started = time.perf_counter()

    fitted: dict[int, dict[str, torch.Tensor]] = {}
    artifact_tensors = {}
    fit_hashes = {}
    for position in (1, 2):
        fit_prompts = prompt_contract[str(position)]["fit"]
        fit_states = capture.capture_last_token(
            fit_prompts,
            hidden_state_indices=[hidden_state_index],
            batch_size=config["base_model_batch_size"],
        )[hidden_state_index].values.float()
        fit_labels = torch.tensor(
            [int(str(row.result)[position]) for row in fit],
            dtype=torch.long,
        )
        basis, centroids, counts = fit_digit_subspace(
            fit_states,
            fit_labels,
        )
        if int(counts.min()) < config["minimum_fit_examples_per_digit"]:
            raise SystemExit(f"insufficient fit support at position {position}")
        fitted[position] = {
            "basis": basis,
            "centroids": centroids,
            "counts": counts,
        }
        name = "tens" if position == 1 else "ones"
        artifact_tensors[f"{name}_basis"] = basis.contiguous()
        artifact_tensors[f"{name}_centroids"] = centroids.contiguous()
        artifact_tensors[f"{name}_counts"] = counts.contiguous()
        fit_hashes[str(position)] = {
            "states": hashlib.sha256(
                fit_states.contiguous().numpy().tobytes()
            ).hexdigest(),
            "basis": hashlib.sha256(
                basis.contiguous().numpy().tobytes()
            ).hexdigest(),
            "centroids": hashlib.sha256(
                centroids.contiguous().numpy().tobytes()
            ).hexdigest(),
        }

    rule = config["selection_rule"]
    positions: dict[str, Any] = {}
    all_pass = True
    for position in (1, 2):
        contract = prompt_contract[str(position)]
        target_prompts = contract["target"]
        identity_prompts = contract["identity"]
        target_states = capture.capture_last_token(
            target_prompts,
            hidden_state_indices=[hidden_state_index],
            batch_size=config["base_model_batch_size"],
        )[hidden_state_index].values.float()
        identity_states = capture.capture_last_token(
            identity_prompts,
            hidden_state_indices=[hidden_state_index],
            batch_size=config["base_model_batch_size"],
        )[hidden_state_index].values.float()
        target_base = capture.next_token_logits(
            target_prompts,
            batch_size=config["base_model_batch_size"],
        )
        identity_base = capture.next_token_logits(
            identity_prompts,
            batch_size=config["base_model_batch_size"],
        )
        target_digits = torch.tensor(contract["target_digits"])
        identity_digits = torch.tensor(contract["identity_digits"])
        wrong_digits = torch.tensor(contract["wrong_digits"])
        target_expected = torch.tensor(
            [digit_token_ids[int(digit)] for digit in target_digits]
        )
        identity_expected = torch.tensor(
            [digit_token_ids[int(digit)] for digit in identity_digits]
        )
        wrong_expected = torch.tensor(
            [digit_token_ids[int(digit)] for digit in wrong_digits]
        )
        position_candidates = []
        passing = []
        logit_cache: dict[tuple[str, str], torch.Tensor] = {}

        def intervened_cached(
            condition: str,
            prompts: list[str],
            delta: torch.Tensor,
            cache: dict[tuple[str, str], torch.Tensor] = logit_cache,
        ) -> torch.Tensor:
            delta_hash = hashlib.sha256(
                delta.contiguous().numpy().tobytes()
            ).hexdigest()
            key = (condition, delta_hash)
            if key not in cache:
                cache[key] = predict_with_delta(
                    model,
                    tokenizer,
                    prompts,
                    delta,
                    hidden_state_index=hidden_state_index,
                    batch_size=config["base_model_batch_size"],
                    device=device,
                )
            return cache[key]

        full_basis = fitted[position]["basis"]
        centroids = fitted[position]["centroids"]
        for rank in config["ranks"]:
            basis = full_basis[:rank]
            raw_target = prototype_delta(
                target_states,
                target_digits,
                centroids,
                basis,
            )
            raw_identity = prototype_delta(
                identity_states,
                identity_digits,
                centroids,
                basis,
            )
            raw_wrong = prototype_delta(
                target_states,
                wrong_digits,
                centroids,
                basis,
            )
            generator = torch.Generator().manual_seed(
                config["random_control_seed"] + 100 * position + rank
            )
            raw_random = torch.randn(
                (len(selection), rank),
                generator=generator,
            ) @ basis
            for norm_cap in config["norm_caps"]:
                for scale in config["scales"]:
                    target_delta, target_gate = scale_and_gate(
                        raw_target,
                        target_states,
                        target_base,
                        target_expected,
                        scale=scale,
                        norm_cap=norm_cap,
                    )
                    identity_delta, identity_gate = scale_and_gate(
                        raw_identity,
                        identity_states,
                        identity_base,
                        identity_expected,
                        scale=scale,
                        norm_cap=norm_cap,
                    )
                    norms = target_delta.norm(dim=1)
                    wrong_delta = norm_match(raw_wrong, norms)
                    random_delta = norm_match(raw_random, norms)
                    target_logits = intervened_cached(
                        "target",
                        target_prompts,
                        target_delta,
                    )
                    identity_logits = intervened_cached(
                        "identity",
                        identity_prompts,
                        identity_delta,
                    )
                    wrong_logits = intervened_cached(
                        "wrong",
                        target_prompts,
                        wrong_delta,
                    )
                    random_logits = intervened_cached(
                        "random",
                        target_prompts,
                        random_delta,
                    )
                    metrics = {
                        "target": evaluate_logits(
                            target_logits,
                            target_expected,
                            digit_token_ids=digit_token_ids,
                        ),
                        "identity": evaluate_logits(
                            identity_logits,
                            identity_expected,
                            digit_token_ids=digit_token_ids,
                        ),
                        "wrong_digit_norm_matched": evaluate_logits(
                            wrong_logits,
                            target_expected,
                            digit_token_ids=digit_token_ids,
                        ),
                        "wrong_digit_alignment": evaluate_logits(
                            wrong_logits,
                            wrong_expected,
                            digit_token_ids=digit_token_ids,
                        ),
                        "random_norm_matched": evaluate_logits(
                            random_logits,
                            target_expected,
                            digit_token_ids=digit_token_ids,
                        ),
                    }
                    strongest_control = max(
                        metrics["wrong_digit_norm_matched"]["accuracy"],
                        metrics["random_norm_matched"]["accuracy"],
                    )
                    advantage = (
                        metrics["target"]["accuracy"] - strongest_control
                    )
                    mean_relative_norm = float(
                        (
                            norms
                            / target_states.norm(dim=1).clamp_min(1e-12)
                        ).mean()
                    )
                    passes = (
                        metrics["target"]["accuracy"]
                        >= rule["minimum_target_accuracy"]
                        and metrics["identity"]["accuracy"]
                        >= rule["minimum_identity_accuracy"]
                        and advantage >= rule["minimum_control_advantage"]
                        and mean_relative_norm
                        <= rule["maximum_mean_relative_norm"]
                        and (
                            not rule["require_digit_token_rate"]
                            or metrics["target"]["digit_token_rate"] == 1.0
                        )
                    )
                    row = {
                        "rank": rank,
                        "scale": scale,
                        "norm_cap": norm_cap,
                        "target_hard_gate_count": int(target_gate.sum()),
                        "identity_hard_gate_count": int(identity_gate.sum()),
                        "metrics": metrics,
                        "gate": {
                            "strongest_control_accuracy": strongest_control,
                            "control_advantage": advantage,
                            "mean_target_relative_norm": mean_relative_norm,
                            "passes": passes,
                        },
                    }
                    position_candidates.append(row)
                    if passes:
                        passing.append(row)
        selected = (
            min(
                passing,
                key=lambda row: (
                    row["gate"]["mean_target_relative_norm"],
                    row["rank"],
                    row["norm_cap"],
                    row["scale"],
                ),
            )
            if passing
            else max(
                position_candidates,
                key=lambda row: (
                    row["metrics"]["target"]["accuracy"],
                    row["metrics"]["identity"]["accuracy"],
                    row["gate"]["control_advantage"],
                    -row["gate"]["mean_target_relative_norm"],
                ),
            )
        )
        position_passes = bool(passing)
        all_pass = all_pass and position_passes
        positions[str(position)] = {
            "passes": position_passes,
            "selection": {
                "rank": selected["rank"],
                "scale": selected["scale"],
                "norm_cap": selected["norm_cap"],
                **selected["gate"],
            },
            "base": {
                "target": evaluate_logits(
                    target_base,
                    target_expected,
                    digit_token_ids=digit_token_ids,
                ),
                "identity": evaluate_logits(
                    identity_base,
                    identity_expected,
                    digit_token_ids=digit_token_ids,
                ),
            },
            "candidates": position_candidates,
            "unique_intervention_evaluations": len(logit_cache),
        }

    args.artifact_output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        artifact_tensors,
        str(args.artifact_output),
        metadata={
            "schema_version": "oli.phase13-smollm2-suffix-prototypes/v1",
            "model_id": model_config["id"],
            "model_revision": model_config["revision"],
            "hidden_state_index": str(hidden_state_index),
            "fit_split": "phase13-fit",
        },
    )
    artifact_hash = hashlib.sha256(args.artifact_output.read_bytes()).hexdigest()

    report = {
        "schema_version": "oli.phase13-smollm2-suffix-prototype-selection/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "selection_only",
        "model": model_config,
        "dataset_sha256": dataset_hash,
        "hidden_state_index": hidden_state_index,
        "fit_examples_sha256": config["fit_examples_sha256"],
        "selection_examples_sha256": config["selection_examples_sha256"],
        "selection_targets_sha256": config["selection_targets_sha256"],
        "basis_method": "svd_of_centered_fit_digit_centroids",
        "ranks": config["ranks"],
        "scales": config["scales"],
        "norm_caps": config["norm_caps"],
        "fit_tensor_sha256": fit_hashes,
        "positions": positions,
        "passes": all_pass,
        "selection_rule": rule,
        "artifact": {
            "path": str(args.artifact_output),
            "sha256": artifact_hash,
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
            "Selection-only SmolLM2-native suffix subspaces and prototypes "
            "fitted on fresh Phase 13 fit data and evaluated on selection. "
            "No integrated, development, audit, cognitive-feature, or "
            "model-general claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    print(f"wrote {args.artifact_output}")


if __name__ == "__main__":
    main()
