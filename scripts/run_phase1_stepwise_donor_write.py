#!/usr/bin/env python3
"""Closed-loop three-digit writing with a native donor patch at each step."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    probe_config_path = Path(config["probe_config"])
    if (
        hashlib.sha256(probe_config_path.read_bytes()).hexdigest()
        != config["probe_config_sha256"]
    ):
        raise SystemExit("probe config hash mismatch")
    selection_path = Path(config["selection_result"])
    if (
        hashlib.sha256(selection_path.read_bytes()).hexdigest()
        != config["selection_result_sha256"]
    ):
        raise SystemExit("donor selection result hash mismatch")
    probe_config = json.loads(probe_config_path.read_text())
    examples = build_phase1_additions(**probe_config["dataset"]["parameters"])
    observed_hash = phase1_addition_sha256(examples)
    if observed_hash != probe_config["dataset"]["sha256"]:
        raise SystemExit("Phase 1 dataset hash mismatch")
    development = [
        example for example in examples if example.split == "development"
    ]

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        probe_config["model"]["id"], revision=probe_config["model"]["revision"]
    )
    model = AutoModelForCausalLM.from_pretrained(
        probe_config["model"]["id"],
        revision=probe_config["model"]["revision"],
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": example.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for example in development
    ]
    targeted_indices, same_indices = choose_donors(development)
    shuffled_indices = targeted_indices[1:] + targeted_indices[:1]
    target_results = [development[index].result for index in targeted_indices]
    original_results = [example.result for example in development]
    target_token_ids = [
        tokenizer(str(result), add_special_tokens=False)["input_ids"]
        for result in target_results
    ]
    if any(len(token_ids) != 3 for token_ids in target_token_ids):
        raise ValueError("all target results must tokenize to exactly three digits")

    capture = ActivationCapture(model, tokenizer, device=device)
    hidden_index = config["hidden_state_index"]
    conditions = (
        "base",
        "targeted_donor",
        "same_leading_norm_matched",
        "shuffled_donor_norm_matched",
        "random_norm_matched",
    )
    condition_results = {}
    started = time.perf_counter()

    for condition_index, condition in enumerate(conditions):
        prefixes = ["" for _ in development]
        predicted_ids: list[list[int]] = [[] for _ in development]
        step_accuracies = []
        step_relative_norms = []
        for step in range(3):
            recipient_inputs = [
                prompt + prefix for prompt, prefix in zip(rendered, prefixes, strict=True)
            ]
            recipient_values = capture.capture_last_token(
                recipient_inputs,
                hidden_state_indices=[hidden_index],
                batch_size=config["batch_size"],
            )[hidden_index].values

            targeted_donor_inputs = [
                rendered[donor_index] + str(development[donor_index].result)[:step]
                for donor_index in targeted_indices
            ]
            targeted_donor_values = capture.capture_last_token(
                targeted_donor_inputs,
                hidden_state_indices=[hidden_index],
                batch_size=config["batch_size"],
            )[hidden_index].values
            targeted_delta = targeted_donor_values - recipient_values
            target_norms = targeted_delta.norm(dim=1)

            if condition == "base":
                delta = torch.zeros_like(targeted_delta)
            elif condition == "targeted_donor":
                delta = targeted_delta
            elif condition == "same_leading_norm_matched":
                donor_inputs = [
                    rendered[donor_index]
                    + str(development[donor_index].result)[:step]
                    for donor_index in same_indices
                ]
                donor_values = capture.capture_last_token(
                    donor_inputs,
                    hidden_state_indices=[hidden_index],
                    batch_size=config["batch_size"],
                )[hidden_index].values
                delta = norm_match(donor_values - recipient_values, target_norms)
            elif condition == "shuffled_donor_norm_matched":
                donor_inputs = [
                    rendered[donor_index]
                    + str(development[donor_index].result)[:step]
                    for donor_index in shuffled_indices
                ]
                donor_values = capture.capture_last_token(
                    donor_inputs,
                    hidden_state_indices=[hidden_index],
                    batch_size=config["batch_size"],
                )[hidden_index].values
                delta = norm_match(donor_values - recipient_values, target_norms)
            else:
                delta = random_norm_matched(
                    tuple(targeted_delta.shape),
                    target_norms,
                    seed=config["random_seed"] + condition_index * 10 + step,
                )

            logits_chunks = []
            for start in range(0, len(recipient_inputs), config["batch_size"]):
                logits_chunks.append(
                    intervened_next_token_logits(
                        model,
                        tokenizer,
                        recipient_inputs[start : start + config["batch_size"]],
                        hidden_state_index=hidden_index,
                        deltas=delta[start : start + config["batch_size"]],
                        device=device,
                    )
                )
            logits = torch.cat(logits_chunks)
            next_ids = logits.argmax(dim=1).tolist()
            expected_ids = torch.tensor(
                [token_ids[step] for token_ids in target_token_ids]
            )
            step_accuracies.append(
                float((torch.tensor(next_ids) == expected_ids).float().mean())
            )
            step_relative_norms.append(
                float(
                    (delta.norm(dim=1) / recipient_values.norm(dim=1)).mean()
                )
            )
            for index, token_id in enumerate(next_ids):
                predicted_ids[index].append(int(token_id))
                prefixes[index] += tokenizer.decode([int(token_id)])

        generated_text = [
            tokenizer.decode(token_ids) for token_ids in predicted_ids
        ]
        parsed = [parse_first_integer(text) for text in generated_text]
        condition_results[condition] = {
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
            "target_token_sequence_accuracy": sum(
                predicted == expected
                for predicted, expected in zip(
                    predicted_ids, target_token_ids, strict=True
                )
            )
            / len(predicted_ids),
            "parse_rate": sum(value is not None for value in parsed) / len(parsed),
            "mean_relative_norm_by_step": step_relative_norms,
            "outputs": [
                {
                    "example_id": example.example_id,
                    "original_result": original_results[index],
                    "target_result": target_results[index],
                    "generated_text": generated_text[index],
                    "parsed": parsed[index],
                    "predicted_token_ids": predicted_ids[index],
                }
                for index, example in enumerate(development)
            ],
        }

    report = {
        "schema_version": "oli.phase1-stepwise-native-write/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "development_only",
        "model": probe_config["model"],
        "dataset": {
            "sha256": observed_hash,
            "development_examples": len(development),
            "audit_examples_unopened": sum(
                example.split == "audit" for example in examples
            ),
        },
        "write": {
            "hidden_state_index": hidden_index,
            "decoder_block": hidden_index - 1,
            "steps": 3,
            "closed_loop": True,
            "target": "matched donor full three-digit result",
        },
        "conditions": condition_results,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Stepwise full residual replacement demonstrates a native sequential "
            "write path but is not a compact deterministic bridge and transfers "
            "many latent variables."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
