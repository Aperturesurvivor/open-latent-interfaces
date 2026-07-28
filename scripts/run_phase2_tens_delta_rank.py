#!/usr/bin/env python3
"""Measure how much tens native-delta rank is needed for causal control."""

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
from run_phase1_conditional_transport_bridge import render_prompts, result_token_ids
from run_phase2_causal_adapter import result_list_sha256
from run_phase2_tens_native_boundary import (
    donor_assignments,
    id_list_sha256,
    predict_with_delta,
    rendered_donor_prompts,
)
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.evaluation import (
    norm_match,
    token_metrics,
)
from open_latent_interfaces.phase2_data import (
    balanced_counterfactual_results,
    build_phase2_additions,
    phase2_addition_sha256,
)


def verify_sha256(path: Path, expected: str) -> None:
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise SystemExit(f"hash mismatch for {path}")


def capture_deltas(
    capture: ActivationCapture,
    tokenizer: Any,
    *,
    examples: list[Any],
    targets: list[int],
    donors: list[Any],
    hidden_state_index: int,
    batch_size: int,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    recipient_prompts = [
        prompt + str(target)[0]
        for prompt, target in zip(
            render_prompts(tokenizer, examples),
            targets,
            strict=True,
        )
    ]
    donor_prompts = [
        prompt + str(donor.result)[0]
        for prompt, donor in zip(
            rendered_donor_prompts(
                tokenizer,
                donors,
                template_split=examples[0].split,
            ),
            donors,
            strict=True,
        )
    ]
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
    examples: list[Any],
    targets: list[int],
    recipient_states: torch.Tensor,
    targeted_delta: torch.Tensor,
    wrong_delta: torch.Tensor,
    basis: torch.Tensor,
    ranks: list[int],
    config: dict[str, Any],
    device: torch.device,
    include_outputs: bool,
) -> dict[str, dict[str, Any]]:
    expected = torch.tensor(
        [row[1] for row in result_token_ids(tokenizer, targets)]
    )
    conditions = (
        "base",
        "projected_targeted",
        "projected_wrong_tens_norm_matched",
        "shuffled_coefficients_norm_matched",
        "random_subspace_norm_matched",
    )
    results = {}
    for rank in ranks:
        rank_basis = basis[:rank]
        targeted = reconstruct(targeted_delta, rank_basis)
        wrong = reconstruct(wrong_delta, rank_basis)
        target_norms = targeted.norm(dim=1)
        layer = {}
        for condition_index, condition in enumerate(conditions):
            if condition == "base":
                delta = torch.zeros_like(targeted)
            elif condition == "projected_targeted":
                delta = targeted
            elif condition == "projected_wrong_tens_norm_matched":
                delta = norm_match(wrong, target_norms)
            elif condition == "shuffled_coefficients_norm_matched":
                shuffled = torch.cat((targeted[1:], targeted[:1]))
                delta = norm_match(shuffled, target_norms)
            else:
                generator = torch.Generator().manual_seed(
                    config["random_control_seed"] + rank * 10 + condition_index
                )
                coefficients = torch.randn(
                    (len(examples), rank),
                    generator=generator,
                )
                delta = norm_match(coefficients @ rank_basis, target_norms)
            logits = predict_with_delta(
                model,
                tokenizer,
                prompts,
                delta,
                hidden_state_index=config["hidden_state_index"],
                batch_size=config["base_model_batch_size"],
                device=device,
            )
            predicted = logits.argmax(dim=1).tolist()
            metrics = token_metrics(logits, expected)
            metrics["mean_relative_norm"] = float(
                (delta.norm(dim=1) / recipient_states.norm(dim=1)).mean()
            )
            metrics["digit_token_rate"] = sum(
                tokenizer.decode([int(token_id)]) in set("0123456789")
                for token_id in predicted
            ) / len(predicted)
            if include_outputs:
                metrics["outputs"] = [
                    {
                        "example_id": example.example_id,
                        "original_result": example.result,
                        "target_result": target,
                        "target_tens": int(str(target)[1]),
                        "predicted_token_id": int(token_id),
                        "predicted_text": tokenizer.decode([int(token_id)]),
                    }
                    for example, target, token_id in zip(
                        examples,
                        targets,
                        predicted,
                        strict=True,
                    )
                ]
            layer[condition] = metrics
        results[str(rank)] = layer
    return results


def rank_passes(
    conditions: dict[str, Any],
    *,
    config: dict[str, Any],
) -> bool:
    targeted = conditions["projected_targeted"]
    controls = [
        row["top1_exact"]
        for name, row in conditions.items()
        if name not in ("projected_targeted", "base")
    ]
    return (
        targeted["top1_exact"] >= config["selection_min_target_accuracy"]
        and targeted["top1_exact"] - max(controls)
        >= config["selection_min_control_advantage"]
        and targeted["mean_relative_norm"] <= config["selection_max_relative_norm"]
        and targeted["digit_token_rate"] == 1.0
    )


def select_rank(
    results: dict[str, dict[str, Any]],
    *,
    config: dict[str, Any],
) -> tuple[int, bool]:
    passing = [
        int(rank)
        for rank, conditions in results.items()
        if rank_passes(conditions, config=config)
    ]
    if passing:
        return min(passing), True

    def score(rank: str) -> tuple[float, float, float, int]:
        conditions = results[rank]
        targeted = conditions["projected_targeted"]
        controls = [
            row["top1_exact"]
            for name, row in conditions.items()
            if name not in ("projected_targeted", "base")
        ]
        return (
            targeted["top1_exact"],
            targeted["top1_exact"] - max(controls),
            targeted["mean_target_margin"],
            -int(rank),
        )

    return int(max(results, key=score)), False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--basis-output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source_path = Path(config["source_result"])
    dataset_config_path = Path(config["dataset_config"])
    verify_sha256(source_path, config["source_result_sha256"])
    verify_sha256(dataset_config_path, config["dataset_config_sha256"])
    source = json.loads(source_path.read_text())
    dataset_config = json.loads(dataset_config_path.read_text())
    examples = build_phase2_additions(**dataset_config["dataset"]["parameters"])
    if phase2_addition_sha256(examples) != source["dataset"]["sha256"]:
        raise SystemExit("dataset hash mismatch")
    fit = [example for example in examples if example.split == "fit"]
    selection = [example for example in examples if example.split == "selection"]
    development = [example for example in examples if example.split == "development"]
    split_examples = {
        "fit": fit,
        "selection": selection,
        "development": development,
    }
    split_targets = {
        split: balanced_counterfactual_results(rows)
        for split, rows in split_examples.items()
    }
    target_hashes = {
        split: result_list_sha256(targets)
        for split, targets in split_targets.items()
    }
    if target_hashes != config["target_sha256"]:
        raise SystemExit("counterfactual target hash mismatch")

    assignment_indices = {}
    assignment_hashes = {}
    for split, rows in split_examples.items():
        targeted, wrong = donor_assignments(
            fit,
            rows,
            split_targets[split],
        )
        assignment_indices[split] = {"targeted": targeted, "wrong": wrong}
        for label, indices in (("targeted", targeted), ("wrong_tens", wrong)):
            assignment_hashes[f"{split}_{label}"] = id_list_sha256(
                [fit[index].example_id for index in indices]
            )
    if assignment_hashes != config["donor_assignment_sha256"]:
        raise SystemExit("donor assignment hash mismatch")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        source["model"]["id"],
        revision=source["model"]["revision"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        source["model"]["id"],
        revision=source["model"]["revision"],
        torch_dtype=getattr(torch, args.dtype),
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    capture = ActivationCapture(model, tokenizer, device=device)
    started = time.perf_counter()
    hidden_index = config["hidden_state_index"]

    fit_prompts, fit_states, fit_delta = capture_deltas(
        capture,
        tokenizer,
        examples=fit,
        targets=split_targets["fit"],
        donors=[
            fit[index] for index in assignment_indices["fit"]["targeted"]
        ],
        hidden_state_index=hidden_index,
        batch_size=config["base_model_batch_size"],
    )
    del fit_prompts
    _, _, basis = torch.linalg.svd(
        fit_delta.float().cpu(),
        full_matrices=False,
    )
    ranks = config["ranks"]
    if max(ranks) > basis.shape[0]:
        raise SystemExit("requested rank exceeds fitted delta basis")
    args.basis_output.parent.mkdir(parents=True, exist_ok=True)
    save_file({"delta_basis": basis.contiguous()}, str(args.basis_output))
    basis_hash = hashlib.sha256(args.basis_output.read_bytes()).hexdigest()

    selection_prompts, selection_states, selection_delta = capture_deltas(
        capture,
        tokenizer,
        examples=selection,
        targets=split_targets["selection"],
        donors=[
            fit[index]
            for index in assignment_indices["selection"]["targeted"]
        ],
        hidden_state_index=hidden_index,
        batch_size=config["base_model_batch_size"],
    )
    _, _, selection_wrong_delta = capture_deltas(
        capture,
        tokenizer,
        examples=selection,
        targets=split_targets["selection"],
        donors=[
            fit[index] for index in assignment_indices["selection"]["wrong"]
        ],
        hidden_state_index=hidden_index,
        batch_size=config["base_model_batch_size"],
    )
    selection_results = evaluate_ranks(
        model,
        tokenizer,
        prompts=selection_prompts,
        examples=selection,
        targets=split_targets["selection"],
        recipient_states=selection_states,
        targeted_delta=selection_delta,
        wrong_delta=selection_wrong_delta,
        basis=basis,
        ranks=ranks,
        config=config,
        device=device,
        include_outputs=False,
    )
    selected_rank, selection_passed = select_rank(
        selection_results,
        config=config,
    )

    development_prompts, development_states, development_delta = capture_deltas(
        capture,
        tokenizer,
        examples=development,
        targets=split_targets["development"],
        donors=[
            fit[index]
            for index in assignment_indices["development"]["targeted"]
        ],
        hidden_state_index=hidden_index,
        batch_size=config["base_model_batch_size"],
    )
    _, _, development_wrong_delta = capture_deltas(
        capture,
        tokenizer,
        examples=development,
        targets=split_targets["development"],
        donors=[
            fit[index] for index in assignment_indices["development"]["wrong"]
        ],
        hidden_state_index=hidden_index,
        batch_size=config["base_model_batch_size"],
    )
    development_results = evaluate_ranks(
        model,
        tokenizer,
        prompts=development_prompts,
        examples=development,
        targets=split_targets["development"],
        recipient_states=development_states,
        targeted_delta=development_delta,
        wrong_delta=development_wrong_delta,
        basis=basis,
        ranks=[selected_rank],
        config=config,
        device=device,
        include_outputs=True,
    )
    report = {
        "schema_version": "oli.phase2-tens-delta-rank/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "development_only",
        "model": source["model"],
        "dataset": source["dataset"],
        "source": {
            "result": str(source_path),
            "result_sha256": config["source_result_sha256"],
        },
        "target_assignment": {
            "scheme": "balanced_all_digits_changed",
            "sha256": target_hashes,
        },
        "donors": {
            "pool": "fit",
            "prefix_length": 2,
            "assignment_sha256": assignment_hashes,
        },
        "basis": {
            "method": "uncentered_svd_of_fit_native_deltas",
            "hidden_state_index": hidden_index,
            "fit_examples": len(fit),
            "fit_states_sha256": hashlib.sha256(
                fit_states.contiguous().numpy().tobytes()
            ).hexdigest(),
            "fit_deltas_sha256": hashlib.sha256(
                fit_delta.contiguous().numpy().tobytes()
            ).hexdigest(),
            "path": str(args.basis_output),
            "sha256": basis_hash,
            "shape": list(basis.shape),
        },
        "selection": {
            "ranks": ranks,
            "selected_rank": selected_rank,
            "diagnostic_gate_passed": selection_passed,
            "conditions_by_rank": selection_results,
        },
        "development": {
            "rank": selected_rank,
            "conditions": development_results[str(selected_rank)],
        },
        "diagnostic_gate": {
            "min_target_accuracy": config["selection_min_target_accuracy"],
            "min_control_advantage": config["selection_min_control_advantage"],
            "max_relative_norm": config["selection_max_relative_norm"],
            "digit_token_rate": 1.0,
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
            "Projected native deltas retain donor-dependent coefficients; "
            "this diagnoses output rank but is not a donor-free writer or audit."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
