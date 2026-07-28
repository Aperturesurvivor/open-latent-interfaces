#!/usr/bin/env python3
"""Fit and evaluate a low-rank paired native-state transport writer."""

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
from open_latent_interfaces.typed_writer import (
    TransportSubspace,
    fit_transport_subspace,
)


def render_prompts(tokenizer: Any, examples: list[Any]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": example.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for example in examples
    ]


def paired_targets(examples: list[Any]) -> tuple[list[int], list[int]]:
    indices, _ = choose_donors(examples)
    return indices, [examples[index].result for index in indices]


def result_token_ids(tokenizer: Any, results: list[int]) -> list[list[int]]:
    rows = [
        tokenizer(str(result), add_special_tokens=False)["input_ids"]
        for result in results
    ]
    if any(len(row) != 3 for row in rows):
        raise ValueError("all results must tokenize to exactly three digit tokens")
    return rows


def predict(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    delta: torch.Tensor,
    *,
    hidden_index: int,
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
                hidden_state_index=hidden_index,
                deltas=delta[start : start + batch_size],
                device=device,
            )
        )
    return torch.cat(chunks).argmax(dim=1).tolist()


def select_writers(
    model: Any,
    tokenizer: Any,
    capture: ActivationCapture,
    writers: list[TransportSubspace],
    *,
    prompts: list[str],
    target_results: list[int],
    config: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    expected = result_token_ids(tokenizer, target_results)
    hidden_index = config["hidden_state_index"]
    selections = []
    for step, writer in enumerate(writers):
        step_prompts = [
            prompt + str(result)[:step]
            for prompt, result in zip(prompts, target_results, strict=True)
        ]
        states = capture.capture_last_token(
            step_prompts,
            hidden_state_indices=[hidden_index],
            batch_size=config["batch_size"],
        )[hidden_index].values
        digits = torch.tensor([int(str(result)[step]) for result in target_results])
        candidates = []
        for rank in config["ranks"]:
            if rank > writer.basis.shape[0]:
                continue
            for scale in config["scales"]:
                delta = writer.write_delta(digits, rank=rank, scale=scale)
                predicted = predict(
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


def evaluate(
    condition: str,
    model: Any,
    tokenizer: Any,
    capture: ActivationCapture,
    writers: list[TransportSubspace],
    selections: list[dict[str, Any]],
    *,
    examples: list[Any],
    prompts: list[str],
    target_results: list[int],
    config: dict[str, Any],
    device: torch.device,
    condition_index: int,
) -> dict[str, Any]:
    original_results = [example.result for example in examples]
    expected = result_token_ids(tokenizer, target_results)
    shuffled = target_results[1:] + target_results[:1]
    prefixes = ["" for _ in examples]
    predicted_ids: list[list[int]] = [[] for _ in examples]
    accuracies = []
    relative_norms = []
    hidden_index = config["hidden_state_index"]
    for step, (writer, selection) in enumerate(zip(writers, selections, strict=True)):
        step_prompts = [
            prompt + prefix for prompt, prefix in zip(prompts, prefixes, strict=True)
        ]
        states = capture.capture_last_token(
            step_prompts,
            hidden_state_indices=[hidden_index],
            batch_size=config["batch_size"],
        )[hidden_index].values
        selected = selection["selected"]
        target_digits = torch.tensor(
            [int(str(result)[step]) for result in target_results]
        )
        typed = writer.write_delta(
            target_digits,
            rank=selected["rank"],
            scale=selected["scale"],
        )
        if condition == "base":
            delta = torch.zeros_like(typed)
        elif condition == "paired_transport":
            delta = typed
        elif condition == "same_digit":
            digits = torch.tensor(
                [int(str(result)[step]) for result in original_results]
            )
            delta = writer.write_delta(
                digits,
                rank=selected["rank"],
                scale=selected["scale"],
            )
        elif condition == "shuffled_target_norm_matched":
            digits = torch.tensor([int(str(result)[step]) for result in shuffled])
            raw = writer.write_delta(
                digits,
                rank=selected["rank"],
                scale=selected["scale"],
            )
            delta = norm_match(raw, typed.norm(dim=1))
        else:
            delta = random_norm_matched(
                tuple(typed.shape),
                typed.norm(dim=1),
                seed=config["random_seed"] + condition_index * 10 + step,
            )
        next_ids = predict(
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
        relative_norms.append(
            float((delta.norm(dim=1) / states.norm(dim=1)).mean())
        )
        for index, token_id in enumerate(next_ids):
            predicted_ids[index].append(int(token_id))
            prefixes[index] += tokenizer.decode([int(token_id)])
    text = [tokenizer.decode(row) for row in predicted_ids]
    parsed = [parse_first_integer(value) for value in text]
    return {
        "step_target_token_accuracy": accuracies,
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
        "mean_relative_norm_by_step": relative_norms,
        "outputs": [
            {
                "example_id": example.example_id,
                "original_result": original_results[index],
                "target_result": target_results[index],
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
    prerequisite = Path(config["typed_writer_result"])
    if hashlib.sha256(prerequisite.read_bytes()).hexdigest() != config[
        "typed_writer_result_sha256"
    ]:
        raise SystemExit("typed-writer result hash mismatch")
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
    hidden_index = config["hidden_state_index"]
    started = time.perf_counter()

    fit_prompts = render_prompts(tokenizer, fit_examples)
    fit_indices, fit_targets = paired_targets(fit_examples)
    writers = []
    fit_metadata = []
    for step in range(3):
        recipient_inputs = [
            prompt + str(target)[:step]
            for prompt, target in zip(fit_prompts, fit_targets, strict=True)
        ]
        donor_inputs = [
            fit_prompts[index] + str(fit_examples[index].result)[:step]
            for index in fit_indices
        ]
        recipient_states = capture.capture_last_token(
            recipient_inputs,
            hidden_state_indices=[hidden_index],
            batch_size=config["batch_size"],
        )[hidden_index].values
        donor_states = capture.capture_last_token(
            donor_inputs,
            hidden_state_indices=[hidden_index],
            batch_size=config["batch_size"],
        )[hidden_index].values
        digits = torch.tensor([int(str(result)[step]) for result in fit_targets])
        writer = fit_transport_subspace(donor_states - recipient_states, digits)
        writers.append(writer)
        fit_metadata.append(
            {
                "step": step,
                "classes": list(writer.classes),
                "available_rank": writer.basis.shape[0],
                "mean_full_transport_relative_norm": float(
                    (
                        (donor_states - recipient_states).norm(dim=1)
                        / recipient_states.norm(dim=1)
                    ).mean()
                ),
                "prototype_singular_values": torch.linalg.svdvals(
                    writer.class_deltas
                ).tolist(),
            }
        )

    selection_prompts = render_prompts(tokenizer, selection_examples)
    _, selection_targets = paired_targets(selection_examples)
    selections = select_writers(
        model,
        tokenizer,
        capture,
        writers,
        prompts=selection_prompts,
        target_results=selection_targets,
        config=config,
        device=device,
    )
    development_prompts = render_prompts(tokenizer, development)
    _, development_targets = paired_targets(development)
    conditions = (
        "base",
        "paired_transport",
        "same_digit",
        "shuffled_target_norm_matched",
        "random_norm_matched",
    )
    condition_results = {
        condition: evaluate(
            condition,
            model,
            tokenizer,
            capture,
            writers,
            selections,
            examples=development,
            prompts=development_prompts,
            target_results=development_targets,
            config=config,
            device=device,
            condition_index=index,
        )
        for index, condition in enumerate(conditions)
    }
    report = {
        "schema_version": "oli.phase1-paired-transport-writer/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "development_only",
        "model": previous["model"],
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
            "type": "class-conditioned paired transport subspace",
            "hidden_state_index": hidden_index,
            "decoder_block": hidden_index - 1,
            "fit": fit_metadata,
            "selection": selections,
            "inference_inputs": ["desired next digit"],
            "live_donor_required": False,
        },
        "conditions": condition_results,
        "prior_results": {
            "typed_writer_artifact": str(prerequisite),
            "typed_writer_sha256": config["typed_writer_result_sha256"],
            "typed_writer_exact_accuracy": previous["conditions"]["typed_writer"][
                "target_full_result_accuracy"
            ],
            "full_donor_exact_accuracy": previous["full_donor_upper_bound"][
                "target_full_result_accuracy"
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
            "Development-only mean paired transports do not establish an audited "
            "or model-general deterministic graft."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
