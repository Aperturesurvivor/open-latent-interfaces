#!/usr/bin/env python3
"""Evaluate closed-loop composition with a donor-free tens prototype implant."""

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
from run_phase2_tens_prototype_writer import prototype_delta
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.adapter import load_online_adapter
from open_latent_interfaces.capability import parse_first_integer
from open_latent_interfaces.evaluation import (
    norm_match,
    random_norm_matched,
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


def cap_and_gate(
    delta: torch.Tensor,
    states: torch.Tensor,
    base_logits: torch.Tensor,
    requested_ids: torch.Tensor,
    *,
    scale: float,
    norm_cap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    scaled = delta * scale
    maximum = states.norm(dim=1) * norm_cap
    factor = (maximum / scaled.norm(dim=1).clamp_min(1e-12)).clamp(max=1.0)
    scaled = scaled * factor[:, None]
    gate = base_logits.argmax(dim=1) == requested_ids
    scaled[gate] = 0
    return scaled, gate


def component_delta(
    step: int,
    states: torch.Tensor,
    results: list[int],
    *,
    adapters: dict[int, Any],
    prototypes_by_step: dict[int, dict[str, torch.Tensor]],
    basis: torch.Tensor,
) -> torch.Tensor:
    if step in prototypes_by_step:
        return prototype_delta(
            states,
            results,
            prototypes_by_step[step],
            basis,
            method="digit",
            answer_position=step,
        )
    digits = torch.tensor([int(str(result)[step]) for result in results])
    return adapters[step].predict(states, digits)


def evaluate_condition(
    condition: str,
    model: Any,
    tokenizer: Any,
    capture: ActivationCapture,
    *,
    examples: list[Any],
    targets: list[int],
    prompts: list[str],
    adapters: dict[int, Any],
    prototypes_by_step: dict[int, dict[str, torch.Tensor]],
    basis: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
    condition_index: int,
) -> dict[str, Any]:
    originals = [example.result for example in examples]
    target_ids = result_token_ids(tokenizer, targets)
    shuffled_targets = targets[1:] + targets[:1]
    prefixes = ["" for _ in examples]
    predicted_ids: list[list[int]] = [[] for _ in examples]
    accuracies = []
    norms = []
    gate_rates = []
    for step in range(3):
        hidden_index = config["hidden_state_indices"][step]
        step_prompts = [
            prompt + prefix for prompt, prefix in zip(prompts, prefixes, strict=True)
        ]
        states = capture.capture_last_token(
            step_prompts,
            hidden_state_indices=[hidden_index],
            batch_size=config["base_model_batch_size"],
        )[hidden_index].values
        base_logits = capture.next_token_logits(
            step_prompts,
            batch_size=config["base_model_batch_size"],
        )
        wanted = torch.tensor([row[step] for row in target_ids])
        raw_targeted = component_delta(
            step,
            states,
            targets,
            adapters=adapters,
            prototypes_by_step=prototypes_by_step,
            basis=basis,
        )
        targeted, targeted_gate = cap_and_gate(
            raw_targeted,
            states,
            base_logits,
            wanted,
            scale=config["scales"][step],
            norm_cap=config["norm_cap"],
        )
        if condition == "base":
            delta = torch.zeros_like(targeted)
            gate = targeted_gate
        elif condition == "hybrid":
            delta = targeted
            gate = targeted_gate
        elif condition == "identity_hard_gated":
            requested = torch.tensor(
                [
                    row[step]
                    for row in result_token_ids(tokenizer, originals)
                ]
            )
            raw = component_delta(
                step,
                states,
                originals,
                adapters=adapters,
                prototypes_by_step=prototypes_by_step,
                basis=basis,
            )
            delta, gate = cap_and_gate(
                raw,
                states,
                base_logits,
                requested,
                scale=config["scales"][step],
                norm_cap=config["norm_cap"],
            )
        elif condition == "shuffled_target_norm_matched":
            raw = component_delta(
                step,
                states,
                shuffled_targets,
                adapters=adapters,
                prototypes_by_step=prototypes_by_step,
                basis=basis,
            )
            scaled = raw * config["scales"][step]
            delta = norm_match(scaled, targeted.norm(dim=1))
            gate = torch.zeros(len(examples), dtype=torch.bool)
        elif condition == "shuffled_state_norm_matched":
            shuffled_states = torch.cat((states[1:], states[:1]))
            raw = component_delta(
                step,
                shuffled_states,
                targets,
                adapters=adapters,
                prototypes_by_step=prototypes_by_step,
                basis=basis,
            )
            scaled = raw * config["scales"][step]
            delta = norm_match(scaled, targeted.norm(dim=1))
            gate = torch.zeros(len(examples), dtype=torch.bool)
        else:
            delta = random_norm_matched(
                tuple(targeted.shape),
                targeted.norm(dim=1),
                seed=config["random_control_seed"] + condition_index * 10 + step,
            )
            gate = torch.zeros(len(examples), dtype=torch.bool)
        logits = predict_with_delta(
            model,
            tokenizer,
            step_prompts,
            delta,
            hidden_state_index=hidden_index,
            batch_size=config["base_model_batch_size"],
            device=device,
        )
        next_ids = logits.argmax(dim=1).tolist()
        accuracies.append(
            sum(
                actual == expected[step]
                for actual, expected in zip(next_ids, target_ids, strict=True)
            )
            / len(next_ids)
        )
        norms.append(float((delta.norm(dim=1) / states.norm(dim=1)).mean()))
        gate_rates.append(float(gate.float().mean()))
        for index, token_id in enumerate(next_ids):
            predicted_ids[index].append(int(token_id))
            prefixes[index] += tokenizer.decode([int(token_id)])
    text = [tokenizer.decode(row) for row in predicted_ids]
    parsed = [parse_first_integer(value) for value in text]
    return {
        "step_target_token_accuracy": accuracies,
        "target_full_result_accuracy": sum(
            value == target for value, target in zip(parsed, targets, strict=True)
        )
        / len(parsed),
        "original_full_result_accuracy": sum(
            value == original
            for value, original in zip(parsed, originals, strict=True)
        )
        / len(parsed),
        "parse_rate": sum(value is not None for value in parsed) / len(parsed),
        "mean_relative_norm_by_step": norms,
        "hard_gate_rate_by_step": gate_rates,
        "outputs": [
            {
                "example_id": example.example_id,
                "original_result": originals[index],
                "target_result": targets[index],
                "generated_text": text[index],
                "parsed": parsed[index],
                "predicted_token_ids": predicted_ids[index],
            }
            for index, example in enumerate(examples)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source_result_path = Path(config["source_adapter_result"])
    source_weights_path = Path(config["source_adapter_weights"])
    prototype_result_path = Path(config["prototype_result"])
    prototype_path = Path(config["prototypes"])
    ones_result_path = (
        Path(config["ones_prototype_result"])
        if "ones_prototype_result" in config
        else None
    )
    ones_prototype_path = (
        Path(config["ones_prototypes"])
        if "ones_prototypes" in config
        else None
    )
    if (ones_result_path is None) != (ones_prototype_path is None):
        raise SystemExit("ones prototype result and artifact must be configured together")
    basis_path = Path(config["basis"])
    dataset_config_path = Path(config["dataset_config"])
    for path, expected in (
        (source_result_path, config["source_adapter_result_sha256"]),
        (source_weights_path, config["source_adapter_weights_sha256"]),
        (prototype_result_path, config["prototype_result_sha256"]),
        (prototype_path, config["prototypes_sha256"]),
        (basis_path, config["basis_sha256"]),
        (dataset_config_path, config["dataset_config_sha256"]),
    ):
        verify_sha256(path, expected)
    if ones_result_path is not None and ones_prototype_path is not None:
        verify_sha256(
            ones_result_path,
            config["ones_prototype_result_sha256"],
        )
        verify_sha256(
            ones_prototype_path,
            config["ones_prototypes_sha256"],
        )
    source = json.loads(source_result_path.read_text())
    prototype_result = json.loads(prototype_result_path.read_text())
    dataset_config = json.loads(dataset_config_path.read_text())
    examples = build_phase2_additions(**dataset_config["dataset"]["parameters"])
    if phase2_addition_sha256(examples) != source["dataset"]["sha256"]:
        raise SystemExit("dataset hash mismatch")
    development = [example for example in examples if example.split == "development"]
    targets = balanced_counterfactual_results(development)
    if result_list_sha256(targets) != config["development_targets_sha256"]:
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
    adapter_steps = (0,) if ones_prototype_path is not None else (0, 2)
    adapters = {
        step: load_online_adapter(str(source_weights_path), step=step)
        for step in adapter_steps
    }
    prototype_tensors = load_file(str(prototype_path))
    prototypes_by_step = {
        1: {
            "digit": prototype_tensors["digit"].float(),
            "prefix": prototype_tensors["prefix"].float(),
        }
    }
    ones_result = None
    if ones_result_path is not None and ones_prototype_path is not None:
        ones_result = json.loads(ones_result_path.read_text())
        ones_tensors = load_file(str(ones_prototype_path))
        prototypes_by_step[2] = {
            "digit": ones_tensors["digit"].float(),
        }
    basis = load_file(str(basis_path))["delta_basis"][
        : config["prototype_rank"]
    ].float()
    if prototype_result["selection"]["selected"]["method"] != "digit":
        raise SystemExit("prototype source method mismatch")
    if prototype_result["selection"]["selected"]["scale"] != config["scales"][1]:
        raise SystemExit("prototype source scale mismatch")
    if ones_result is not None:
        if ones_result["selection"]["selected"]["method"] != "digit":
            raise SystemExit("ones prototype source method mismatch")
        if ones_result["selection"]["selected"]["scale"] != config["scales"][2]:
            raise SystemExit("ones prototype source scale mismatch")

    prompts = render_prompts(tokenizer, development)
    conditions = (
        "base",
        "hybrid",
        "identity_hard_gated",
        "shuffled_target_norm_matched",
        "shuffled_state_norm_matched",
        "random_norm_matched",
    )
    started = time.perf_counter()
    condition_results = {
        condition: evaluate_condition(
            condition,
            model,
            tokenizer,
            capture,
            examples=development,
            targets=targets,
            prompts=prompts,
            adapters=adapters,
            prototypes_by_step=prototypes_by_step,
            basis=basis,
            config=config,
            device=device,
            condition_index=index,
        )
        for index, condition in enumerate(conditions)
    }
    sources = {
        "adapter_result": str(source_result_path),
        "adapter_result_sha256": config["source_adapter_result_sha256"],
        "adapter_weights": str(source_weights_path),
        "adapter_weights_sha256": config["source_adapter_weights_sha256"],
        "prototype_result": str(prototype_result_path),
        "prototype_result_sha256": config["prototype_result_sha256"],
        "prototypes": str(prototype_path),
        "prototypes_sha256": config["prototypes_sha256"],
        "basis": str(basis_path),
        "basis_sha256": config["basis_sha256"],
    }
    if ones_result_path is not None and ones_prototype_path is not None:
        sources.update(
            {
                "ones_prototype_result": str(ones_result_path),
                "ones_prototype_result_sha256": config[
                    "ones_prototype_result_sha256"
                ],
                "ones_prototypes": str(ones_prototype_path),
                "ones_prototypes_sha256": config["ones_prototypes_sha256"],
            }
        )
    report = {
        "schema_version": (
            "oli.phase2-closed-loop-dual-prototype/v1"
            if ones_result is not None
            else "oli.phase2-closed-loop-hybrid/v1"
        ),
        "created_at": datetime.now(UTC).isoformat(),
        "status": "development_only",
        "model": source["model"],
        "dataset": source["dataset"],
        "sources": sources,
        "write": {
            "hidden_state_indices": config["hidden_state_indices"],
            "scales": config["scales"],
            "hard_gate_all_positions": True,
            "position_components": [
                "causal_adapter",
                "rank16_digit_prototype",
                (
                    "rank16_digit_prototype"
                    if ones_result is not None
                    else "causal_adapter"
                ),
            ],
        },
        "target_assignment": {
            "scheme": "balanced_all_digits_changed",
            "development_sha256": config["development_targets_sha256"],
        },
        "conditions": condition_results,
        "advancement_gate": source["advancement_gate"],
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Iterative closed-loop development composition; audit remains sealed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
