#!/usr/bin/env python3
"""Patch native donor residuals across layers and matched controls."""

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
from open_latent_interfaces.evaluation import norm_match, random_norm_matched
from open_latent_interfaces.interventions import (
    intervened_generate,
    intervened_next_token_logits,
)
from open_latent_interfaces.phase1_data import (
    build_phase1_additions,
    phase1_addition_sha256,
)


def donor_distance(recipient: Any, donor: Any) -> tuple[int, int, int]:
    return (
        abs((recipient.result % 100) - (donor.result % 100)),
        int(recipient.ones_carry != donor.ones_carry)
        + int(recipient.tens_carry != donor.tens_carry),
        abs(recipient.operand_a - donor.operand_a)
        + abs(recipient.operand_b - donor.operand_b),
    )


def choose_donors(examples: list[Any]) -> tuple[list[int], list[int]]:
    targeted = []
    same_leading = []
    for recipient_index, recipient in enumerate(examples):
        original_digit = int(str(recipient.result)[0])
        desired_digit = original_digit % 9 + 1
        target_candidates = [
            (index, donor)
            for index, donor in enumerate(examples)
            if int(str(donor.result)[0]) == desired_digit
        ]
        same_candidates = [
            (index, donor)
            for index, donor in enumerate(examples)
            if index != recipient_index and int(str(donor.result)[0]) == original_digit
        ]
        targeted.append(
            min(target_candidates, key=lambda item: donor_distance(recipient, item[1]))[
                0
            ]
        )
        same_leading.append(
            min(same_candidates, key=lambda item: donor_distance(recipient, item[1]))[
                0
            ]
        )
    return targeted, same_leading


def digit_metrics(
    logits: torch.Tensor,
    target_digits: torch.Tensor,
    original_digits: torch.Tensor,
    digit_token_ids: torch.Tensor,
) -> dict[str, float]:
    scores = logits[:, digit_token_ids]
    rows = torch.arange(len(logits))
    target_scores = scores[rows, target_digits]
    target_other = scores.clone()
    target_other[rows, target_digits] = -torch.inf
    original_scores = scores[rows, original_digits]
    original_other = scores.clone()
    original_other[rows, original_digits] = -torch.inf
    predictions = scores.argmax(dim=1)
    return {
        "target_digit_accuracy": float((predictions == target_digits).float().mean()),
        "target_mean_digit_margin": float(
            (target_scores - target_other.max(dim=1).values).mean()
        ),
        "original_digit_accuracy": float(
            (predictions == original_digits).float().mean()
        ),
        "original_mean_digit_margin": float(
            (original_scores - original_other.max(dim=1).values).mean()
        ),
        "mean_target_minus_original_logit": float(
            (target_scores - original_scores).mean()
        ),
    }


def run_logits(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    deltas: torch.Tensor,
    *,
    hidden_state_index: int,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
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
    return torch.cat(chunks)


def run_generation(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    deltas: torch.Tensor,
    *,
    hidden_state_index: int,
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    responses = []
    for start in range(0, len(prompts), batch_size):
        responses.extend(
            intervened_generate(
                model,
                tokenizer,
                prompts[start : start + batch_size],
                hidden_state_index=hidden_state_index,
                deltas=deltas[start : start + batch_size],
                device=device,
                max_new_tokens=max_new_tokens,
            )
        )
    return responses


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
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        probe_config["model"]["id"],
        revision=probe_config["model"]["revision"],
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": example.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for example in development
    ]
    targeted_indices, same_indices = choose_donors(development)
    targeted_index_tensor = torch.tensor(targeted_indices)
    same_index_tensor = torch.tensor(same_indices)
    shuffled_indices = targeted_indices[1:] + targeted_indices[:1]
    shuffled_index_tensor = torch.tensor(shuffled_indices)
    target_results = torch.tensor(
        [development[index].result for index in targeted_indices]
    )
    original_results = torch.tensor([example.result for example in development])
    target_digits = torch.tensor(
        [int(str(int(value))[0]) for value in target_results]
    )
    original_digits = torch.tensor(
        [int(str(int(value))[0]) for value in original_results]
    )
    digit_token_ids = torch.tensor(
        [
            int(tokenizer(str(digit), add_special_tokens=False)["input_ids"][0])
            for digit in range(10)
        ]
    )

    capture = ActivationCapture(model, tokenizer, device=device)
    started = time.perf_counter()
    captured = capture.capture_last_token(
        prompts,
        hidden_state_indices=config["hidden_state_indices"],
        batch_size=config["batch_size"],
    )
    layer_results = {}
    for hidden_index in config["hidden_state_indices"]:
        values = captured[hidden_index].values
        targeted_delta = values[targeted_index_tensor] - values
        targeted_norms = targeted_delta.norm(dim=1)
        same_native = values[same_index_tensor] - values
        same_norm = norm_match(same_native, targeted_norms)
        shuffled_native = values[shuffled_index_tensor] - values
        shuffled_norm = norm_match(shuffled_native, targeted_norms)
        random_delta = random_norm_matched(
            tuple(targeted_delta.shape),
            targeted_norms,
            seed=config["random_seed"] + hidden_index,
        )
        conditions = {
            "base": torch.zeros_like(targeted_delta),
            "targeted_donor": targeted_delta,
            "same_leading_native": same_native,
            "same_leading_norm_matched": same_norm,
            "shuffled_donor_norm_matched": shuffled_norm,
            "random_norm_matched": random_delta,
        }
        condition_results = {}
        for condition, delta in conditions.items():
            logits = run_logits(
                model,
                tokenizer,
                prompts,
                delta,
                hidden_state_index=hidden_index,
                device=device,
                batch_size=config["batch_size"],
            )
            responses = run_generation(
                model,
                tokenizer,
                prompts,
                delta,
                hidden_state_index=hidden_index,
                device=device,
                batch_size=config["batch_size"],
                max_new_tokens=config["max_new_tokens"],
            )
            parsed = [parse_first_integer(response) for response in responses]
            metrics = digit_metrics(
                logits,
                target_digits,
                original_digits,
                digit_token_ids,
            )
            metrics.update(
                {
                    "generated_target_exact": sum(
                        value == int(target)
                        for value, target in zip(parsed, target_results, strict=True)
                    )
                    / len(parsed),
                    "generated_original_exact": sum(
                        value == int(original)
                        for value, original in zip(
                            parsed, original_results, strict=True
                        )
                    )
                    / len(parsed),
                    "generation_parse_rate": sum(value is not None for value in parsed)
                    / len(parsed),
                }
            )
            condition_results[condition] = {
                "metrics": metrics,
                "mean_delta_norm": float(delta.norm(dim=1).mean()),
                "mean_relative_norm": float(
                    (delta.norm(dim=1) / values.norm(dim=1)).mean()
                ),
            }
        layer_results[str(hidden_index)] = {
            "decoder_block": hidden_index - 1,
            "conditions": condition_results,
        }

    donor_map = [
        {
            "recipient_id": recipient.example_id,
            "recipient_result": recipient.result,
            "target_donor_id": development[targeted_indices[index]].example_id,
            "target_donor_result": development[targeted_indices[index]].result,
            "same_leading_donor_id": development[same_indices[index]].example_id,
            "same_leading_donor_result": development[same_indices[index]].result,
        }
        for index, recipient in enumerate(development)
    ]
    report = {
        "schema_version": "oli.phase1-native-donor-patch/v1",
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
        "donor_selection": {
            "target_digit": "cyclic next leading digit 1-9",
            "matching": "suffix distance, carry mismatch count, operand distance",
            "template_matched": True,
            "map": donor_map,
        },
        "layers": layer_results,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Full residual replacement is an on-manifold donor diagnostic but "
            "transfers many variables at once. It localizes causal state without "
            "identifying a minimal typed write direction."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
