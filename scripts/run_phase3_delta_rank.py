#!/usr/bin/env python3
"""Fit and select low-rank native transport bases for frozen Phi-3.5."""

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
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.donors import (
    choose_position_donors,
    choose_prefix_donors,
)
from open_latent_interfaces.evaluation import norm_match, token_metrics
from open_latent_interfaces.phase2_data import balanced_counterfactual_results
from open_latent_interfaces.phase3_data import (
    build_phase3_additions,
    phase3_addition_sha256,
)
from open_latent_interfaces.prefill import verify_decimal_digit_contract


def rotate_all_digits(result: int) -> int:
    digits = [int(digit) for digit in str(result)]
    if len(digits) != 3:
        raise ValueError("rotation requires a three-digit result")
    return int(
        f"{digits[0] % 9 + 1}{(digits[1] + 1) % 10}{(digits[2] + 1) % 10}"
    )


def wrong_position_results(results: list[int], *, position: int) -> list[int]:
    wrong = []
    for result in results:
        digits = list(str(result))
        digit = int(digits[position])
        digits[position] = str(
            digit % 9 + 1 if position == 0 else (digit + 1) % 10
        )
        wrong.append(int("".join(digits)))
    return wrong


def capture_deltas(
    capture: ActivationCapture,
    *,
    recipients: list[Any],
    targets: list[int],
    donors: list[Any],
    rendered_recipients: list[str],
    rendered_donors: list[str],
    position: int,
    hidden_state_index: int,
    batch_size: int,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    recipient_prompts = prefix_prompts(
        rendered_recipients,
        targets,
        position=position,
    )
    donor_prompts = prefix_prompts(
        rendered_donors,
        [donor.result for donor in donors],
        position=position,
    )
    recipient = capture.capture_last_token(
        recipient_prompts,
        hidden_state_indices=[hidden_state_index],
        batch_size=batch_size,
    )[hidden_state_index].values
    donor = capture.capture_last_token(
        donor_prompts,
        hidden_state_indices=[hidden_state_index],
        batch_size=batch_size,
    )[hidden_state_index].values
    return recipient_prompts, recipient, donor - recipient


def reconstruct(delta: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return (delta.float() @ basis.T) @ basis


def evaluate_ranks(
    model: Any,
    tokenizer: Any,
    *,
    prompts: list[str],
    targets: list[int],
    recipient_states: torch.Tensor,
    targeted_delta: torch.Tensor,
    wrong_delta: torch.Tensor,
    basis: torch.Tensor,
    ranks: list[int],
    position: int,
    hidden_state_index: int,
    digit_token_ids: dict[int, int],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    expected = torch.tensor(
        [digit_token_ids[int(str(target)[position])] for target in targets]
    )
    digit_ids = set(digit_token_ids.values())
    conditions = (
        "base",
        "projected_targeted",
        "projected_wrong_digit_norm_matched",
        "shuffled_coefficients_norm_matched",
        "random_subspace_norm_matched",
    )
    results = {}
    for rank in ranks:
        rank_basis = basis[:rank]
        targeted = reconstruct(targeted_delta, rank_basis)
        wrong = reconstruct(wrong_delta, rank_basis)
        target_norms = targeted.norm(dim=1)
        rank_results = {}
        for condition_index, condition in enumerate(conditions):
            if condition == "base":
                delta = torch.zeros_like(targeted)
            elif condition == "projected_targeted":
                delta = targeted
            elif condition == "projected_wrong_digit_norm_matched":
                delta = norm_match(wrong, target_norms)
            elif condition == "shuffled_coefficients_norm_matched":
                shuffled = torch.cat((targeted[1:], targeted[:1]))
                delta = norm_match(shuffled, target_norms)
            else:
                generator = torch.Generator().manual_seed(
                    config["random_control_seed"]
                    + position * 1000
                    + rank * 10
                    + condition_index
                )
                coefficients = torch.randn(
                    (len(targets), rank),
                    generator=generator,
                )
                delta = norm_match(coefficients @ rank_basis, target_norms)
            logits = predict_with_delta(
                model,
                tokenizer,
                prompts,
                delta,
                hidden_state_index=hidden_state_index,
                batch_size=config["base_model_batch_size"],
                device=device,
            )
            predicted = logits.argmax(dim=1).tolist()
            metrics = token_metrics(logits, expected)
            metrics["mean_relative_norm"] = float(
                (delta.norm(dim=1) / recipient_states.norm(dim=1)).mean()
            )
            metrics["digit_token_rate"] = sum(
                token_id in digit_ids for token_id in predicted
            ) / len(predicted)
            metrics["predicted_token_ids"] = predicted
            rank_results[condition] = metrics
        results[str(rank)] = rank_results
    return results


def rank_passes(conditions: dict[str, Any], config: dict[str, Any]) -> bool:
    targeted = conditions["projected_targeted"]
    controls = [
        row["top1_exact"]
        for name, row in conditions.items()
        if name not in ("projected_targeted", "base")
    ]
    return (
        targeted["top1_exact"]
        >= config["selection_rule"]["minimum_target_accuracy"]
        and targeted["top1_exact"] - max(controls)
        >= config["selection_rule"]["minimum_control_advantage"]
        and targeted["mean_relative_norm"]
        <= config["selection_rule"]["maximum_relative_norm"]
        and targeted["digit_token_rate"] == 1.0
    )


def select_rank(
    results_by_position: dict[int, dict[str, dict[str, Any]]],
    ranks: list[int],
    config: dict[str, Any],
) -> tuple[int, bool]:
    passing = [
        rank
        for rank in ranks
        if all(
            rank_passes(results[str(rank)], config)
            for results in results_by_position.values()
        )
    ]
    if passing:
        return min(passing), True

    def score(rank: int) -> tuple[float, float, float, int]:
        rows = [
            results[str(rank)]
            for results in results_by_position.values()
        ]
        target_accuracies = [
            row["projected_targeted"]["top1_exact"] for row in rows
        ]
        advantages = []
        margins = []
        for row in rows:
            targeted = row["projected_targeted"]
            controls = [
                value["top1_exact"]
                for name, value in row.items()
                if name not in ("projected_targeted", "base")
            ]
            advantages.append(targeted["top1_exact"] - max(controls))
            margins.append(targeted["mean_target_margin"])
        return (
            min(target_accuracies),
            min(advantages),
            min(margins),
            -rank,
        )

    return max(ranks, key=score), False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--basis-output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()

    if args.output.exists() or args.basis_output.exists():
        raise SystemExit("refusing to overwrite rank result or basis artifact")
    config = json.loads(args.config.read_text())
    dataset_path = Path(config["dataset_config"])
    behavior_path = Path(config["behavior_result"])
    boundary_path = Path(config["boundary_result"])
    verify_sha256(dataset_path, config["dataset_config_sha256"])
    verify_sha256(behavior_path, config["behavior_result_sha256"])
    verify_sha256(boundary_path, config["boundary_result_sha256"])
    dataset_config = json.loads(dataset_path.read_text())
    behavior = json.loads(behavior_path.read_text())
    boundary = json.loads(boundary_path.read_text())
    if dataset_config.get("audit_authorized", False):
        raise SystemExit("rank selection requires a sealed audit")
    if not behavior["passes"] or not boundary["passes"]:
        raise SystemExit("required behavior or native-boundary gate failed")
    selected_indices = {
        int(position): row["selected_hidden_state_index"]
        for position, row in boundary["positions"].items()
    }
    if selected_indices != {
        int(key): value for key, value in config["hidden_state_indices"].items()
    }:
        raise SystemExit("configured boundaries differ from frozen source")

    examples = build_phase3_additions(**dataset_config["dataset"]["parameters"])
    observed_hash = phase3_addition_sha256(examples)
    if observed_hash != dataset_config["dataset"]["sha256"]:
        raise SystemExit("Phase 3 dataset hash mismatch")
    exact_fit_ids = {
        row["example_id"]
        for row in behavior["rows"]
        if row["split"] == "fit" and row["exact"]
    }
    fit = [
        example
        for example in examples
        if example.split == "fit" and example.example_id in exact_fit_ids
    ]
    selection = [example for example in examples if example.split == "selection"]
    targets_by_split = {
        "fit": [rotate_all_digits(example.result) for example in fit],
        "selection": balanced_counterfactual_results(selection),
    }
    observed_target_hashes = {
        split: value_list_sha256(targets)
        for split, targets in targets_by_split.items()
    }
    if observed_target_hashes != config["target_sha256"]:
        raise SystemExit("target assignment hash mismatch")

    assignments: dict[str, dict[int, dict[str, list[int]]]] = {}
    assignment_hashes = {}
    for split, recipients in (("fit", fit), ("selection", selection)):
        assignments[split] = {}
        targets = targets_by_split[split]
        for position in range(3):
            if position == 1:
                targeted = choose_prefix_donors(
                    fit,
                    recipients,
                    targets,
                    prefix_length=2,
                )
                wrong = choose_prefix_donors(
                    fit,
                    recipients,
                    wrong_position_results(targets, position=position),
                    prefix_length=2,
                )
            else:
                targeted = choose_position_donors(
                    fit,
                    recipients,
                    targets,
                    position=position,
                )
                wrong = choose_position_donors(
                    fit,
                    recipients,
                    targets,
                    position=position,
                    wrong_digit=True,
                )
            assignments[split][position] = {
                "targeted": targeted,
                "wrong": wrong,
            }
            assignment_hashes[f"{split}_{position}_targeted"] = value_list_sha256(
                [fit[index].example_id for index in targeted]
            )
            assignment_hashes[f"{split}_{position}_wrong"] = value_list_sha256(
                [fit[index].example_id for index in wrong]
            )
    if assignment_hashes != config["donor_assignment_sha256"]:
        raise SystemExit("donor assignment hash mismatch")

    device = torch.device(args.device)
    model_config = dataset_config["model"]
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
    digit_token_ids = verify_decimal_digit_contract(tokenizer, rendered_fit[0])
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

    fit_deltas = {}
    fit_state_hashes = {}
    for position in (0, 1):
        indices = assignments["fit"][position]["targeted"]
        _, states, delta = capture_deltas(
            capture,
            recipients=fit,
            targets=targets_by_split["fit"],
            donors=[fit[index] for index in indices],
            rendered_recipients=rendered_fit,
            rendered_donors=[rendered_fit[index] for index in indices],
            position=position,
            hidden_state_index=selected_indices[position],
            batch_size=config["base_model_batch_size"],
        )
        fit_deltas[position] = delta
        fit_state_hashes[str(position)] = {
            "recipient": hashlib.sha256(
                states.contiguous().numpy().tobytes()
            ).hexdigest(),
            "delta": hashlib.sha256(
                delta.contiguous().numpy().tobytes()
            ).hexdigest(),
        }
    _, _, leading_basis = torch.linalg.svd(
        fit_deltas[0].float(),
        full_matrices=False,
    )
    _, _, suffix_basis = torch.linalg.svd(
        fit_deltas[1].float(),
        full_matrices=False,
    )
    args.basis_output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "leading_basis": leading_basis.contiguous(),
            "suffix_basis": suffix_basis.contiguous(),
        },
        str(args.basis_output),
    )
    basis_hash = hashlib.sha256(args.basis_output.read_bytes()).hexdigest()

    selection_results = {}
    ranks = config["ranks"]
    for position in range(3):
        targeted_indices = assignments["selection"][position]["targeted"]
        wrong_indices = assignments["selection"][position]["wrong"]
        prompts, states, targeted_delta = capture_deltas(
            capture,
            recipients=selection,
            targets=targets_by_split["selection"],
            donors=[fit[index] for index in targeted_indices],
            rendered_recipients=rendered_selection,
            rendered_donors=[rendered_fit[index] for index in targeted_indices],
            position=position,
            hidden_state_index=selected_indices[position],
            batch_size=config["base_model_batch_size"],
        )
        _, _, wrong_delta = capture_deltas(
            capture,
            recipients=selection,
            targets=targets_by_split["selection"],
            donors=[fit[index] for index in wrong_indices],
            rendered_recipients=rendered_selection,
            rendered_donors=[rendered_fit[index] for index in wrong_indices],
            position=position,
            hidden_state_index=selected_indices[position],
            batch_size=config["base_model_batch_size"],
        )
        basis = leading_basis if position == 0 else suffix_basis
        selection_results[position] = evaluate_ranks(
            model,
            tokenizer,
            prompts=prompts,
            targets=targets_by_split["selection"],
            recipient_states=states,
            targeted_delta=targeted_delta,
            wrong_delta=wrong_delta,
            basis=basis,
            ranks=ranks,
            position=position,
            hidden_state_index=selected_indices[position],
            digit_token_ids=digit_token_ids,
            config=config,
            device=device,
        )

    leading_rank, leading_passes = select_rank(
        {0: selection_results[0]},
        ranks,
        config,
    )
    suffix_rank, suffix_passes = select_rank(
        {1: selection_results[1], 2: selection_results[2]},
        ranks,
        config,
    )
    report = {
        "schema_version": "oli.phase3-delta-rank/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "selection_only",
        "model": model_config,
        "dataset": {
            "sha256": observed_hash,
            "eligible_fit_examples": len(fit),
            "selection_examples": len(selection),
        },
        "sources": {
            "behavior_result_sha256": config["behavior_result_sha256"],
            "boundary_result_sha256": config["boundary_result_sha256"],
        },
        "targets": {
            "fit_scheme": "all_digits_cyclically_changed",
            "selection_scheme": "balanced_all_digits_changed",
            "sha256": observed_target_hashes,
        },
        "donor_assignment_sha256": assignment_hashes,
        "basis": {
            "method": "uncentered_svd_of_fit_native_deltas",
            "leading_fit_position": 0,
            "suffix_fit_position": 1,
            "cross_position_test": "suffix basis evaluated unchanged at position 2",
            "hidden_state_indices": selected_indices,
            "fit_state_sha256": fit_state_hashes,
            "path": str(args.basis_output),
            "sha256": basis_hash,
            "leading_shape": list(leading_basis.shape),
            "suffix_shape": list(suffix_basis.shape),
        },
        "selection": {
            "ranks": ranks,
            "selected_leading_rank": leading_rank,
            "leading_gate_passed": leading_passes,
            "selected_suffix_rank": suffix_rank,
            "suffix_gate_passed": suffix_passes,
            "positions": {
                str(position): results
                for position, results in selection_results.items()
            },
        },
        "passes": leading_passes and suffix_passes,
        "selection_rule": config["selection_rule"],
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Projected native deltas retain donor-dependent coefficients. "
            "This diagnoses causal output rank but is not donor-free."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
