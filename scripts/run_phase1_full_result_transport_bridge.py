#!/usr/bin/env python3
"""Evaluate native transport conditioned on state and a complete result."""

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
from run_phase1_conditional_transport_bridge import (
    capture_deduplicated,
    predict_ids,
    render_prompts,
    result_token_ids,
    target_results,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.capability import parse_first_integer
from open_latent_interfaces.donors import choose_multi_donors
from open_latent_interfaces.evaluation import norm_match, random_norm_matched
from open_latent_interfaces.phase1_data import (
    build_phase1_additions,
    phase1_addition_sha256,
)
from open_latent_interfaces.typed_writer import (
    FullResultTransportDesign,
    FullResultTransportModel,
    build_full_result_transport_design,
)


def build_fit_designs(
    capture: ActivationCapture,
    examples: list[Any],
    prompts: list[str],
    *,
    config: dict[str, Any],
) -> tuple[list[FullResultTransportDesign], list[dict[str, Any]]]:
    donor_rows = choose_multi_donors(examples)
    if any(
        len(row) != config["donors_per_fit_recipient"] for row in donor_rows
    ):
        raise ValueError("unexpected multi-donor count")
    pairs = [
        (recipient_index, donor_index)
        for recipient_index, row in enumerate(donor_rows)
        for donor_index in row
    ]
    designs = []
    metadata = []
    hidden_index = config["hidden_state_index"]
    for step in range(3):
        recipient_inputs = [
            prompts[recipient_index] + str(examples[donor_index].result)[:step]
            for recipient_index, donor_index in pairs
        ]
        donor_inputs = [
            prompts[donor_index] + str(examples[donor_index].result)[:step]
            for _, donor_index in pairs
        ]
        recipient_states = capture_deduplicated(
            capture,
            recipient_inputs,
            hidden_index=hidden_index,
            batch_size=config["batch_size"],
        )
        donor_states = capture_deduplicated(
            capture,
            donor_inputs,
            hidden_index=hidden_index,
            batch_size=config["batch_size"],
        )
        deltas = donor_states - recipient_states
        results = torch.tensor(
            [examples[donor_index].result for _, donor_index in pairs]
        )
        design = build_full_result_transport_design(
            recipient_states,
            deltas,
            results,
            state_rank=config["state_rank"],
            max_transport_rank=max(config["transport_ranks"]),
        )
        designs.append(design)
        metadata.append(
            {
                "step": step,
                "training_pairs": len(pairs),
                "unique_recipient_contexts": len(set(recipient_inputs)),
                "unique_donor_contexts": len(set(donor_inputs)),
                "feature_count": design.features.shape[1],
                "mean_full_transport_relative_norm": float(
                    (deltas.norm(dim=1) / recipient_states.norm(dim=1)).mean()
                ),
            }
        )
    return designs, metadata


def select_models(
    model: Any,
    tokenizer: Any,
    capture: ActivationCapture,
    designs: list[FullResultTransportDesign],
    *,
    prompts: list[str],
    targets: list[int],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[list[FullResultTransportModel], list[dict[str, Any]]]:
    expected = result_token_ids(tokenizer, targets)
    target_tensor = torch.tensor(targets)
    hidden_index = config["hidden_state_index"]
    models = []
    selections = []
    for step, design in enumerate(designs):
        step_prompts = [
            prompt + str(result)[:step]
            for prompt, result in zip(prompts, targets, strict=True)
        ]
        states = capture.capture_last_token(
            step_prompts,
            hidden_state_indices=[hidden_index],
            batch_size=config["batch_size"],
        )[hidden_index].values
        candidates = []
        candidate_models = []
        for rank in config["transport_ranks"]:
            for ridge in config["ridge_values"]:
                fitted = design.fit(transport_rank=rank, ridge=ridge)
                raw = fitted.predict(states, target_tensor)
                for scale in config["scales"]:
                    delta = raw * scale
                    predicted = predict_ids(
                        model,
                        tokenizer,
                        step_prompts,
                        delta,
                        hidden_index=hidden_index,
                        batch_size=config["batch_size"],
                        device=device,
                    )
                    accuracy = sum(
                        actual == wanted[step]
                        for actual, wanted in zip(predicted, expected, strict=True)
                    ) / len(predicted)
                    candidates.append(
                        {
                            "transport_rank": rank,
                            "ridge": ridge,
                            "scale": scale,
                            "target_token_accuracy": accuracy,
                            "mean_relative_norm": float(
                                (delta.norm(dim=1) / states.norm(dim=1)).mean()
                            ),
                        }
                    )
                    candidate_models.append(fitted)
        selected_index = max(
            range(len(candidates)),
            key=lambda index: (
                candidates[index]["target_token_accuracy"],
                -candidates[index]["transport_rank"],
                -candidates[index]["mean_relative_norm"],
                candidates[index]["ridge"],
            ),
        )
        models.append(candidate_models[selected_index])
        selections.append(
            {
                "step": step,
                "selected": candidates[selected_index],
                "candidates": candidates,
            }
        )
    return models, selections


def condition_delta(
    condition: str,
    bridge: FullResultTransportModel,
    states: torch.Tensor,
    targets: list[int],
    originals: list[int],
    *,
    scale: float,
    seed: int,
) -> torch.Tensor:
    target_tensor = torch.tensor(targets)
    typed = bridge.predict(states, target_tensor) * scale
    if condition == "base":
        return torch.zeros_like(typed)
    if condition == "full_result_transport":
        return typed
    if condition == "same_result":
        return bridge.predict(states, torch.tensor(originals)) * scale
    if condition == "shuffled_result_norm_matched":
        shuffled = torch.tensor(targets[1:] + targets[:1])
        raw = bridge.predict(states, shuffled) * scale
        return norm_match(raw, typed.norm(dim=1))
    if condition == "shuffled_state_norm_matched":
        shuffled_states = torch.cat((states[1:], states[:1]))
        raw = bridge.predict(shuffled_states, target_tensor) * scale
        return norm_match(raw, typed.norm(dim=1))
    return random_norm_matched(
        tuple(typed.shape),
        typed.norm(dim=1),
        seed=seed,
    )


def evaluate_closed_loop(
    condition: str,
    model: Any,
    tokenizer: Any,
    capture: ActivationCapture,
    bridges: list[FullResultTransportModel],
    selections: list[dict[str, Any]],
    *,
    examples: list[Any],
    prompts: list[str],
    targets: list[int],
    config: dict[str, Any],
    device: torch.device,
    condition_index: int,
) -> dict[str, Any]:
    originals = [example.result for example in examples]
    expected = result_token_ids(tokenizer, targets)
    prefixes = ["" for _ in examples]
    predicted_ids: list[list[int]] = [[] for _ in examples]
    accuracies = []
    norms = []
    hidden_index = config["hidden_state_index"]
    for step, (bridge, selection) in enumerate(
        zip(bridges, selections, strict=True)
    ):
        step_prompts = [
            prompt + prefix for prompt, prefix in zip(prompts, prefixes, strict=True)
        ]
        states = capture.capture_last_token(
            step_prompts,
            hidden_state_indices=[hidden_index],
            batch_size=config["batch_size"],
        )[hidden_index].values
        delta = condition_delta(
            condition,
            bridge,
            states,
            targets,
            originals,
            scale=selection["selected"]["scale"],
            seed=config["random_seed"] + condition_index * 10 + step,
        )
        next_ids = predict_ids(
            model,
            tokenizer,
            step_prompts,
            delta,
            hidden_index=hidden_index,
            batch_size=config["batch_size"],
            device=device,
        )
        accuracies.append(
            sum(
                actual == wanted[step]
                for actual, wanted in zip(next_ids, expected, strict=True)
            )
            / len(next_ids)
        )
        norms.append(float((delta.norm(dim=1) / states.norm(dim=1)).mean()))
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


def evaluate_teacher_forced(
    condition: str,
    model: Any,
    tokenizer: Any,
    capture: ActivationCapture,
    bridges: list[FullResultTransportModel],
    selections: list[dict[str, Any]],
    *,
    examples: list[Any],
    prompts: list[str],
    targets: list[int],
    config: dict[str, Any],
    device: torch.device,
    condition_index: int,
) -> dict[str, Any]:
    originals = [example.result for example in examples]
    expected = result_token_ids(tokenizer, targets)
    accuracies = []
    norms = []
    hidden_index = config["hidden_state_index"]
    for step, (bridge, selection) in enumerate(
        zip(bridges, selections, strict=True)
    ):
        step_prompts = [
            prompt + str(result)[:step]
            for prompt, result in zip(prompts, targets, strict=True)
        ]
        states = capture.capture_last_token(
            step_prompts,
            hidden_state_indices=[hidden_index],
            batch_size=config["batch_size"],
        )[hidden_index].values
        delta = condition_delta(
            condition,
            bridge,
            states,
            targets,
            originals,
            scale=selection["selected"]["scale"],
            seed=config["random_seed"] + 1000 + condition_index * 10 + step,
        )
        next_ids = predict_ids(
            model,
            tokenizer,
            step_prompts,
            delta,
            hidden_index=hidden_index,
            batch_size=config["batch_size"],
            device=device,
        )
        accuracies.append(
            sum(
                actual == wanted[step]
                for actual, wanted in zip(next_ids, expected, strict=True)
            )
            / len(next_ids)
        )
        norms.append(float((delta.norm(dim=1) / states.norm(dim=1)).mean()))
    return {
        "step_target_token_accuracy": accuracies,
        "mean_relative_norm_by_step": norms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    prerequisite = Path(config["conditional_result"])
    if hashlib.sha256(prerequisite.read_bytes()).hexdigest() != config[
        "conditional_result_sha256"
    ]:
        raise SystemExit("conditional result hash mismatch")
    previous = json.loads(prerequisite.read_text())
    examples = build_phase1_additions()
    observed_hash = phase1_addition_sha256(examples)
    if observed_hash != previous["dataset"]["sha256"]:
        raise SystemExit("Phase 1 dataset hash mismatch")
    training = [example for example in examples if example.split == "train"]
    development = [
        example for example in examples if example.split == "development"
    ]
    fit_examples = training[: config["fit_examples"]]
    selection_examples = training[config["fit_examples"] :]
    if len(selection_examples) != config["selection_examples"]:
        raise ValueError("training fit/selection split mismatch")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        previous["model"]["id"], revision=previous["model"]["revision"]
    )
    model = AutoModelForCausalLM.from_pretrained(
        previous["model"]["id"],
        revision=previous["model"]["revision"],
        torch_dtype=getattr(torch, args.dtype),
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    capture = ActivationCapture(model, tokenizer, device=device)
    started = time.perf_counter()

    fit_prompts = render_prompts(tokenizer, fit_examples)
    designs, fit_metadata = build_fit_designs(
        capture,
        fit_examples,
        fit_prompts,
        config=config,
    )
    selection_prompts = render_prompts(tokenizer, selection_examples)
    selection_targets = target_results(selection_examples)
    bridges, selections = select_models(
        model,
        tokenizer,
        capture,
        designs,
        prompts=selection_prompts,
        targets=selection_targets,
        config=config,
        device=device,
    )

    development_prompts = render_prompts(tokenizer, development)
    development_targets = target_results(development)
    conditions = (
        "base",
        "full_result_transport",
        "same_result",
        "shuffled_result_norm_matched",
        "shuffled_state_norm_matched",
        "random_norm_matched",
    )
    closed_loop = {
        condition: evaluate_closed_loop(
            condition,
            model,
            tokenizer,
            capture,
            bridges,
            selections,
            examples=development,
            prompts=development_prompts,
            targets=development_targets,
            config=config,
            device=device,
            condition_index=index,
        )
        for index, condition in enumerate(conditions)
    }
    teacher_forced = {
        condition: evaluate_teacher_forced(
            condition,
            model,
            tokenizer,
            capture,
            bridges,
            selections,
            examples=development,
            prompts=development_prompts,
            targets=development_targets,
            config=config,
            device=device,
            condition_index=index,
        )
        for index, condition in enumerate(conditions)
    }
    report = {
        "schema_version": "oli.phase1-full-result-transport-bridge/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "development_only",
        "model": previous["model"],
        "dataset": {
            "sha256": observed_hash,
            "fit_examples": len(fit_examples),
            "fit_pairs": len(fit_examples)
            * config["donors_per_fit_recipient"],
            "selection_examples": len(selection_examples),
            "development_examples": len(development),
            "audit_examples_unopened": sum(
                example.split == "audit" for example in examples
            ),
        },
        "bridge": {
            "type": "ridge state/full-result interaction to reduced-rank transport",
            "hidden_state_index": config["hidden_state_index"],
            "decoder_block": config["hidden_state_index"] - 1,
            "state_rank": config["state_rank"],
            "fit": fit_metadata,
            "selection": selections,
            "inference_inputs": ["recipient native state", "complete target result"],
            "live_donor_required": False,
        },
        "closed_loop_conditions": closed_loop,
        "teacher_forced_conditions": teacher_forced,
        "prior_results": {
            "conditional_artifact": str(prerequisite),
            "conditional_sha256": config["conditional_result_sha256"],
            "next_digit_exact_accuracy": previous["conditions"][
                "conditional_transport"
            ]["target_full_result_accuracy"],
            "full_donor_exact_accuracy": previous["prior_results"][
                "full_donor_exact_accuracy"
            ],
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
            "Development-only full-result transport does not establish an "
            "audited or model-general deterministic graft."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
