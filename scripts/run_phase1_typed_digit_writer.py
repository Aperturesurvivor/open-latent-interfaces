#!/usr/bin/env python3
"""Fit, select, and evaluate a donor-free low-rank native digit writer."""

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
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.capability import parse_first_integer
from open_latent_interfaces.donors import choose_donors
from open_latent_interfaces.evaluation import norm_match, random_norm_matched
from open_latent_interfaces.interventions import intervened_next_token_logits
from open_latent_interfaces.phase1_data import (
    build_phase1_additions,
    phase1_addition_sha256,
)
from open_latent_interfaces.typed_writer import DigitSubspace, fit_digit_subspace


def render_prompts(tokenizer: Any, examples: list[Any]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": example.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for example in examples
    ]


def result_digits(examples: list[Any]) -> list[list[int]]:
    return [[int(character) for character in str(example.result)] for example in examples]


def target_results_for(examples: list[Any]) -> list[int]:
    targeted_indices, _ = choose_donors(examples)
    return [examples[index].result for index in targeted_indices]


def next_token_ids(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    hidden_state_index: int,
    deltas: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> list[int]:
    chunks = []
    for start in range(0, len(prompts), batch_size):
        chunks.append(
            intervened_next_token_logits(
                model,
                tokenizer,
                prompts[start : start + batch_size],
                hidden_state_index=hidden_state_index,
                deltas=deltas[start : start + batch_size],
                device=device,
            )
        )
    return torch.cat(chunks).argmax(dim=1).tolist()


def token_ids_for_results(tokenizer: Any, results: list[int]) -> list[list[int]]:
    ids = [
        tokenizer(str(result), add_special_tokens=False)["input_ids"]
        for result in results
    ]
    if any(len(row) != 3 for row in ids):
        raise ValueError("all results must tokenize to exactly three digit tokens")
    return ids


def select_hyperparameters(
    model: Any,
    tokenizer: Any,
    capture: ActivationCapture,
    *,
    selection_prompts: list[str],
    target_results: list[int],
    writers: list[DigitSubspace],
    config: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    hidden_index = config["hidden_state_index"]
    target_ids = token_ids_for_results(tokenizer, target_results)
    selections = []
    for step, writer in enumerate(writers):
        prompts = [
            prompt + str(result)[:step]
            for prompt, result in zip(
                selection_prompts, target_results, strict=True
            )
        ]
        states = capture.capture_last_token(
            prompts,
            hidden_state_indices=[hidden_index],
            batch_size=config["batch_size"],
        )[hidden_index].values
        target_digits = torch.tensor(
            [int(str(result)[step]) for result in target_results]
        )
        expected = [row[step] for row in target_ids]
        candidates = []
        for rank in config["ranks"]:
            if rank > writer.basis.shape[0]:
                continue
            for scale in config["scales"]:
                delta = writer.write_delta(
                    states,
                    target_digits,
                    rank=rank,
                    scale=scale,
                )
                predicted = next_token_ids(
                    model,
                    tokenizer,
                    prompts,
                    hidden_state_index=hidden_index,
                    deltas=delta,
                    batch_size=config["batch_size"],
                    device=device,
                )
                accuracy = sum(
                    actual == wanted
                    for actual, wanted in zip(predicted, expected, strict=True)
                ) / len(expected)
                relative_norm = float(
                    (delta.norm(dim=1) / states.norm(dim=1)).mean()
                )
                candidates.append(
                    {
                        "rank": rank,
                        "scale": scale,
                        "target_token_accuracy": accuracy,
                        "mean_relative_norm": relative_norm,
                    }
                )
        selected = max(
            candidates,
            key=lambda row: (
                row["target_token_accuracy"],
                -row["rank"],
                -row["mean_relative_norm"],
                -row["scale"],
            ),
        )
        selections.append({"step": step, "selected": selected, "candidates": candidates})
    return selections


def evaluate_condition(
    condition: str,
    model: Any,
    tokenizer: Any,
    capture: ActivationCapture,
    *,
    prompts: list[str],
    example_ids: list[str],
    original_results: list[int],
    target_results: list[int],
    writers: list[DigitSubspace],
    selections: list[dict[str, Any]],
    config: dict[str, Any],
    device: torch.device,
    condition_index: int,
) -> dict[str, Any]:
    hidden_index = config["hidden_state_index"]
    expected_ids = token_ids_for_results(tokenizer, target_results)
    shuffled_results = target_results[1:] + target_results[:1]
    prefixes = ["" for _ in prompts]
    predicted_ids: list[list[int]] = [[] for _ in prompts]
    step_accuracies = []
    step_relative_norms = []

    for step, (writer, selection) in enumerate(zip(writers, selections, strict=True)):
        recipient_inputs = [
            prompt + prefix for prompt, prefix in zip(prompts, prefixes, strict=True)
        ]
        states = capture.capture_last_token(
            recipient_inputs,
            hidden_state_indices=[hidden_index],
            batch_size=config["batch_size"],
        )[hidden_index].values
        selected = selection["selected"]
        target_digits = torch.tensor(
            [int(str(result)[step]) for result in target_results]
        )
        typed_delta = writer.write_delta(
            states,
            target_digits,
            rank=selected["rank"],
            scale=selected["scale"],
        )
        if condition == "base":
            delta = torch.zeros_like(typed_delta)
        elif condition == "typed_writer":
            delta = typed_delta
        elif condition == "same_digit":
            original_digits = torch.tensor(
                [int(str(result)[step]) for result in original_results]
            )
            delta = writer.write_delta(
                states,
                original_digits,
                rank=selected["rank"],
                scale=selected["scale"],
            )
        elif condition == "shuffled_target_norm_matched":
            shuffled_digits = torch.tensor(
                [int(str(result)[step]) for result in shuffled_results]
            )
            shuffled_delta = writer.write_delta(
                states,
                shuffled_digits,
                rank=selected["rank"],
                scale=selected["scale"],
            )
            delta = norm_match(shuffled_delta, typed_delta.norm(dim=1))
        else:
            delta = random_norm_matched(
                tuple(typed_delta.shape),
                typed_delta.norm(dim=1),
                seed=config["random_seed"] + condition_index * 10 + step,
            )
        next_ids = next_token_ids(
            model,
            tokenizer,
            recipient_inputs,
            hidden_state_index=hidden_index,
            deltas=delta,
            batch_size=config["batch_size"],
            device=device,
        )
        step_accuracies.append(
            sum(
                actual == wanted[step]
                for actual, wanted in zip(next_ids, expected_ids, strict=True)
            )
            / len(next_ids)
        )
        step_relative_norms.append(
            float((delta.norm(dim=1) / states.norm(dim=1)).mean())
        )
        for index, token_id in enumerate(next_ids):
            predicted_ids[index].append(int(token_id))
            prefixes[index] += tokenizer.decode([int(token_id)])

    generated_text = [tokenizer.decode(row) for row in predicted_ids]
    parsed = [parse_first_integer(text) for text in generated_text]
    return {
        "step_target_token_accuracy": step_accuracies,
        "target_full_result_accuracy": sum(
            value == target
            for value, target in zip(parsed, target_results, strict=True)
        )
        / len(parsed),
        "original_full_result_accuracy": sum(
            value == original
            for value, original in zip(parsed, original_results, strict=True)
        )
        / len(parsed),
        "parse_rate": sum(value is not None for value in parsed) / len(parsed),
        "mean_relative_norm_by_step": step_relative_norms,
        "outputs": [
            {
                "example_id": example_ids[index],
                "original_result": original_results[index],
                "target_result": target_results[index],
                "generated_text": generated_text[index],
                "parsed": parsed[index],
                "predicted_token_ids": predicted_ids[index],
            }
            for index in range(len(prompts))
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
    prerequisite = Path(config["stepwise_result"])
    if hashlib.sha256(prerequisite.read_bytes()).hexdigest() != config[
        "stepwise_result_sha256"
    ]:
        raise SystemExit("stepwise result hash mismatch")
    prerequisite_report = json.loads(prerequisite.read_text())
    model_config = prerequisite_report["model"]
    examples = build_phase1_additions()
    observed_hash = phase1_addition_sha256(examples)
    if observed_hash != prerequisite_report["dataset"]["sha256"]:
        raise SystemExit("Phase 1 dataset hash mismatch")
    training = [example for example in examples if example.split == "train"]
    development = [
        example for example in examples if example.split == "development"
    ]
    fit_count = config["fit_examples"]
    selection_count = config["selection_examples"]
    if fit_count + selection_count != len(training):
        raise ValueError("fit and selection counts must exhaust the training split")
    fit_examples = training[:fit_count]
    selection_examples = training[fit_count:]

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"], revision=model_config["revision"]
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    capture = ActivationCapture(model, tokenizer, device=device)
    hidden_index = config["hidden_state_index"]
    started = time.perf_counter()

    fit_prompts = render_prompts(tokenizer, fit_examples)
    fit_digits = result_digits(fit_examples)
    writers = []
    writer_metadata = []
    for step in range(3):
        prompts = [
            prompt + str(example.result)[:step]
            for prompt, example in zip(fit_prompts, fit_examples, strict=True)
        ]
        states = capture.capture_last_token(
            prompts,
            hidden_state_indices=[hidden_index],
            batch_size=config["batch_size"],
        )[hidden_index].values
        digits = torch.tensor([row[step] for row in fit_digits])
        writer = fit_digit_subspace(states, digits)
        writers.append(writer)
        singular_values = torch.linalg.svdvals(
            writer.centroids - writer.centroids.mean(dim=0)
        )
        writer_metadata.append(
            {
                "step": step,
                "classes": list(writer.classes),
                "available_rank": writer.basis.shape[0],
                "between_class_singular_values": singular_values.tolist(),
            }
        )

    selection_prompts = render_prompts(tokenizer, selection_examples)
    selection_targets = target_results_for(selection_examples)
    selections = select_hyperparameters(
        model,
        tokenizer,
        capture,
        selection_prompts=selection_prompts,
        target_results=selection_targets,
        writers=writers,
        config=config,
        device=device,
    )

    development_prompts = render_prompts(tokenizer, development)
    development_targets = target_results_for(development)
    original_results = [example.result for example in development]
    conditions = (
        "base",
        "typed_writer",
        "same_digit",
        "shuffled_target_norm_matched",
        "random_norm_matched",
    )
    condition_results = {
        condition: evaluate_condition(
            condition,
            model,
            tokenizer,
            capture,
            prompts=development_prompts,
            example_ids=[example.example_id for example in development],
            original_results=original_results,
            target_results=development_targets,
            writers=writers,
            selections=selections,
            config=config,
            device=device,
            condition_index=index,
        )
        for index, condition in enumerate(conditions)
    }

    report = {
        "schema_version": "oli.phase1-typed-digit-writer/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "development_only",
        "model": model_config,
        "dataset": {
            "sha256": observed_hash,
            "fit_examples": len(fit_examples),
            "selection_examples": len(selection_examples),
            "development_examples": len(development),
            "audit_examples_unopened": sum(
                example.split == "audit" for example in examples
            ),
        },
        "writer": {
            "type": "between-class centroid subspace replacement",
            "hidden_state_index": hidden_index,
            "decoder_block": hidden_index - 1,
            "fit": writer_metadata,
            "selection": selections,
            "inference_inputs": ["recipient native state", "desired next digit"],
            "live_donor_required": False,
        },
        "conditions": condition_results,
        "full_donor_upper_bound": {
            "artifact": str(prerequisite),
            "sha256": config["stepwise_result_sha256"],
            "target_full_result_accuracy": prerequisite_report["conditions"][
                "targeted_donor"
            ]["target_full_result_accuracy"],
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
            "A low-rank development result does not establish an audited or "
            "model-general deterministic graft."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
