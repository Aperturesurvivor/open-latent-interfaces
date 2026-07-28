#!/usr/bin/env python3
"""Fit and evaluate donor-free native-coordinate tens prototypes."""

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
from run_phase2_tens_native_boundary import predict_with_delta
from safetensors.torch import load_file, save_file
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


def render_template(
    tokenizer: Any,
    examples: list[Any],
    template: str,
) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": template.format(
                        a=example.operand_a,
                        b=example.operand_b,
                    ),
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for example in examples
    ]


def fit_prototypes(
    capture: ActivationCapture,
    tokenizer: Any,
    examples: list[Any],
    basis: torch.Tensor,
    *,
    config: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    all_states = []
    all_results = []
    for template in config["fit_templates"]:
        prompts = [
            prompt + str(example.result)[0]
            for prompt, example in zip(
                render_template(tokenizer, examples, template),
                examples,
                strict=True,
            )
        ]
        states = capture.capture_last_token(
            prompts,
            hidden_state_indices=[config["hidden_state_index"]],
            batch_size=config["base_model_batch_size"],
        )[config["hidden_state_index"]].values.float()
        all_states.append(states)
        all_results.extend(example.result for example in examples)
    stacked = torch.cat(all_states)
    coordinates = stacked @ basis.T
    digit = torch.empty((10, basis.shape[0]))
    digit_counts = torch.empty(10, dtype=torch.long)
    for value in range(10):
        mask = torch.tensor(
            [int(str(result)[1]) == value for result in all_results]
        )
        digit[value] = coordinates[mask].mean(dim=0)
        digit_counts[value] = int(mask.sum())
    prefix = torch.empty((90, basis.shape[0]))
    prefix_counts = torch.empty(90, dtype=torch.long)
    for value in range(10, 100):
        mask = torch.tensor(
            [int(str(result)[:2]) == value for result in all_results]
        )
        if not bool(mask.any()):
            raise ValueError(f"missing fit prototype for prefix {value}")
        prefix[value - 10] = coordinates[mask].mean(dim=0)
        prefix_counts[value - 10] = int(mask.sum())
    hashes = {
        "states": hashlib.sha256(
            stacked.contiguous().numpy().tobytes()
        ).hexdigest(),
        "coordinates": hashlib.sha256(
            coordinates.contiguous().numpy().tobytes()
        ).hexdigest(),
    }
    return {
        "digit": digit,
        "digit_counts": digit_counts,
        "prefix": prefix,
        "prefix_counts": prefix_counts,
    }, hashes


def capture_context(
    capture: ActivationCapture,
    tokenizer: Any,
    examples: list[Any],
    results: list[int],
    *,
    config: dict[str, Any],
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    prompts = [
        prompt + str(result)[0]
        for prompt, result in zip(
            render_prompts(tokenizer, examples),
            results,
            strict=True,
        )
    ]
    states = capture.capture_last_token(
        prompts,
        hidden_state_indices=[config["hidden_state_index"]],
        batch_size=config["base_model_batch_size"],
    )[config["hidden_state_index"]].values.float()
    zeros = torch.zeros_like(states)
    base_logits = predict_with_delta(
        capture.model,
        tokenizer,
        prompts,
        zeros,
        hidden_state_index=config["hidden_state_index"],
        batch_size=config["base_model_batch_size"],
        device=capture.device,
    )
    return prompts, states, base_logits


def requested_ids(tokenizer: Any, results: list[int]) -> torch.Tensor:
    return torch.tensor([row[1] for row in result_token_ids(tokenizer, results)])


def prototype_delta(
    states: torch.Tensor,
    results: list[int],
    prototypes: dict[str, torch.Tensor],
    basis: torch.Tensor,
    *,
    method: str,
) -> torch.Tensor:
    if method == "digit":
        indices = torch.tensor([int(str(result)[1]) for result in results])
    elif method == "prefix":
        indices = torch.tensor([int(str(result)[:2]) - 10 for result in results])
    else:
        raise ValueError(f"unknown prototype method: {method}")
    desired = prototypes[method][indices]
    recipient = states @ basis.T
    return (desired - recipient) @ basis


def scale_and_gate(
    delta: torch.Tensor,
    states: torch.Tensor,
    base_logits: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    scale: float,
    norm_cap: float,
) -> torch.Tensor:
    scaled = delta * scale
    maximum = states.norm(dim=1) * norm_cap
    factor = (maximum / scaled.norm(dim=1).clamp_min(1e-12)).clamp(max=1.0)
    scaled = scaled * factor[:, None]
    already_correct = base_logits.argmax(dim=1) == target_ids
    scaled[already_correct] = 0
    return scaled


def evaluate_delta(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    examples: list[Any],
    results: list[int],
    states: torch.Tensor,
    delta: torch.Tensor,
    *,
    config: dict[str, Any],
    device: torch.device,
    include_outputs: bool,
) -> dict[str, Any]:
    expected = requested_ids(tokenizer, results)
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
        (delta.norm(dim=1) / states.norm(dim=1)).mean()
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
                "requested_result": result,
                "requested_tens": int(str(result)[1]),
                "predicted_token_id": int(token_id),
                "predicted_text": tokenizer.decode([int(token_id)]),
            }
            for example, result, token_id in zip(
                examples,
                results,
                predicted,
                strict=True,
            )
        ]
    return metrics


def selection_rows(
    model: Any,
    tokenizer: Any,
    *,
    examples: list[Any],
    targets: list[int],
    target_context: tuple[list[str], torch.Tensor, torch.Tensor],
    identity_context: tuple[list[str], torch.Tensor, torch.Tensor],
    prototypes: dict[str, torch.Tensor],
    basis: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    target_prompts, target_states, target_base = target_context
    identity_prompts, identity_states, identity_base = identity_context
    originals = [example.result for example in examples]
    target_ids = requested_ids(tokenizer, targets)
    identity_ids = requested_ids(tokenizer, originals)
    rows = []
    for method in config["methods"]:
        raw_target = prototype_delta(
            target_states,
            targets,
            prototypes,
            basis,
            method=method,
        )
        raw_identity = prototype_delta(
            identity_states,
            originals,
            prototypes,
            basis,
            method=method,
        )
        for scale in config["scales"]:
            target_delta = scale_and_gate(
                raw_target,
                target_states,
                target_base,
                target_ids,
                scale=scale,
                norm_cap=config["norm_cap"],
            )
            identity_delta = scale_and_gate(
                raw_identity,
                identity_states,
                identity_base,
                identity_ids,
                scale=scale,
                norm_cap=config["norm_cap"],
            )
            target = evaluate_delta(
                model,
                tokenizer,
                target_prompts,
                examples,
                targets,
                target_states,
                target_delta,
                config=config,
                device=device,
                include_outputs=False,
            )
            identity = evaluate_delta(
                model,
                tokenizer,
                identity_prompts,
                examples,
                originals,
                identity_states,
                identity_delta,
                config=config,
                device=device,
                include_outputs=False,
            )
            rows.append(
                {
                    "method": method,
                    "scale": scale,
                    "target_token_accuracy": target["top1_exact"],
                    "identity_token_accuracy": identity["top1_exact"],
                    "minimum_accuracy": min(
                        target["top1_exact"],
                        identity["top1_exact"],
                    ),
                    "mean_target_relative_norm": target["mean_relative_norm"],
                    "mean_identity_relative_norm": identity["mean_relative_norm"],
                    "target_mean_margin": target["mean_target_margin"],
                }
            )
    return rows


def select_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            row["minimum_accuracy"],
            row["target_token_accuracy"],
            row["identity_token_accuracy"],
            row["target_mean_margin"],
            -row["mean_target_relative_norm"],
            row["method"] == "digit",
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prototype-output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source_path = Path(config["source_result"])
    basis_path = Path(config["basis"])
    dataset_config_path = Path(config["dataset_config"])
    verify_sha256(source_path, config["source_result_sha256"])
    verify_sha256(basis_path, config["basis_sha256"])
    verify_sha256(dataset_config_path, config["dataset_config_sha256"])
    source = json.loads(source_path.read_text())
    dataset_config = json.loads(dataset_config_path.read_text())
    examples = build_phase2_additions(**dataset_config["dataset"]["parameters"])
    if phase2_addition_sha256(examples) != source["dataset"]["sha256"]:
        raise SystemExit("dataset hash mismatch")
    fit = [example for example in examples if example.split == "fit"]
    selection = [example for example in examples if example.split == "selection"]
    development = [example for example in examples if example.split == "development"]
    split_targets = {
        "selection": balanced_counterfactual_results(selection),
        "development": balanced_counterfactual_results(development),
    }
    target_hashes = {
        split: result_list_sha256(targets)
        for split, targets in split_targets.items()
    }
    if target_hashes != config["target_sha256"]:
        raise SystemExit("counterfactual target hash mismatch")

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
    basis = load_file(str(basis_path))["delta_basis"][: config["rank"]].float()
    prototypes, fit_hashes = fit_prototypes(
        capture,
        tokenizer,
        fit,
        basis,
        config=config,
    )
    args.prototype_output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: tensor.contiguous() for name, tensor in prototypes.items()},
        str(args.prototype_output),
    )
    prototype_hash = hashlib.sha256(args.prototype_output.read_bytes()).hexdigest()

    selection_target_context = capture_context(
        capture,
        tokenizer,
        selection,
        split_targets["selection"],
        config=config,
    )
    selection_identity_context = capture_context(
        capture,
        tokenizer,
        selection,
        [example.result for example in selection],
        config=config,
    )
    rows = selection_rows(
        model,
        tokenizer,
        examples=selection,
        targets=split_targets["selection"],
        target_context=selection_target_context,
        identity_context=selection_identity_context,
        prototypes=prototypes,
        basis=basis,
        config=config,
        device=device,
    )
    selected = select_candidate(rows)

    target_prompts, target_states, target_base = capture_context(
        capture,
        tokenizer,
        development,
        split_targets["development"],
        config=config,
    )
    originals = [example.result for example in development]
    identity_prompts, identity_states, identity_base = capture_context(
        capture,
        tokenizer,
        development,
        originals,
        config=config,
    )
    method = selected["method"]
    scale = selected["scale"]
    target_ids = requested_ids(tokenizer, split_targets["development"])
    identity_ids = requested_ids(tokenizer, originals)
    raw_target = prototype_delta(
        target_states,
        split_targets["development"],
        prototypes,
        basis,
        method=method,
    )
    targeted = scale_and_gate(
        raw_target,
        target_states,
        target_base,
        target_ids,
        scale=scale,
        norm_cap=config["norm_cap"],
    )
    raw_identity = prototype_delta(
        identity_states,
        originals,
        prototypes,
        basis,
        method=method,
    )
    identity_delta = scale_and_gate(
        raw_identity,
        identity_states,
        identity_base,
        identity_ids,
        scale=scale,
        norm_cap=config["norm_cap"],
    )
    wrong_results = []
    for target in split_targets["development"]:
        digits = list(str(target))
        digits[1] = str((int(digits[1]) + 1) % 10)
        wrong_results.append(int("".join(digits)))
    wrong = prototype_delta(
        target_states,
        wrong_results,
        prototypes,
        basis,
        method=method,
    )
    wrong = norm_match(wrong, targeted.norm(dim=1))
    shuffled_results = (
        split_targets["development"][1:]
        + split_targets["development"][:1]
    )
    shuffled = prototype_delta(
        target_states,
        shuffled_results,
        prototypes,
        basis,
        method=method,
    )
    shuffled = norm_match(shuffled, targeted.norm(dim=1))
    generator = torch.Generator().manual_seed(config["random_control_seed"])
    random_coefficients = torch.randn(
        (len(development), basis.shape[0]),
        generator=generator,
    )
    random_delta = norm_match(
        random_coefficients @ basis,
        targeted.norm(dim=1),
    )
    zeros = torch.zeros_like(targeted)
    development_conditions = {
        "base": evaluate_delta(
            model,
            tokenizer,
            target_prompts,
            development,
            split_targets["development"],
            target_states,
            zeros,
            config=config,
            device=device,
            include_outputs=True,
        ),
        "prototype_writer": evaluate_delta(
            model,
            tokenizer,
            target_prompts,
            development,
            split_targets["development"],
            target_states,
            targeted,
            config=config,
            device=device,
            include_outputs=True,
        ),
        "wrong_tens_norm_matched": evaluate_delta(
            model,
            tokenizer,
            target_prompts,
            development,
            split_targets["development"],
            target_states,
            wrong,
            config=config,
            device=device,
            include_outputs=True,
        ),
        "shuffled_target_norm_matched": evaluate_delta(
            model,
            tokenizer,
            target_prompts,
            development,
            split_targets["development"],
            target_states,
            shuffled,
            config=config,
            device=device,
            include_outputs=True,
        ),
        "random_subspace_norm_matched": evaluate_delta(
            model,
            tokenizer,
            target_prompts,
            development,
            split_targets["development"],
            target_states,
            random_delta,
            config=config,
            device=device,
            include_outputs=True,
        ),
        "identity_hard_gated": evaluate_delta(
            model,
            tokenizer,
            identity_prompts,
            development,
            originals,
            identity_states,
            identity_delta,
            config=config,
            device=device,
            include_outputs=True,
        ),
    }
    report = {
        "schema_version": "oli.phase2-tens-prototype-writer/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "development_only",
        "model": source["model"],
        "dataset": source["dataset"],
        "source": {
            "result": str(source_path),
            "result_sha256": config["source_result_sha256"],
            "basis": str(basis_path),
            "basis_sha256": config["basis_sha256"],
        },
        "target_assignment": {
            "scheme": "balanced_all_digits_changed",
            "sha256": target_hashes,
        },
        "fit": {
            "templates": config["fit_templates"],
            "examples_per_template": len(fit),
            "activation_sha256": fit_hashes,
            "prototype_path": str(args.prototype_output),
            "prototype_sha256": prototype_hash,
        },
        "selection": {
            "methods": config["methods"],
            "scales": config["scales"],
            "selected": selected,
            "candidates": rows,
        },
        "development": {"conditions": development_conditions},
        "diagnostic_gate": {
            "min_target_accuracy": config["min_target_accuracy"],
            "min_control_advantage": config["min_control_advantage"],
            "min_identity_accuracy": config["min_identity_accuracy"],
            "max_relative_norm": config["norm_cap"],
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
            "Teacher-forced donor-free tens writer using fit-derived prototypes; "
            "closed-loop composition and audit remain untested."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
