#!/usr/bin/env python3
"""Test whether a decoded tens-carry direction causally changes the hundreds digit."""

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
from open_latent_interfaces.evaluation import norm_match, random_norm_matched
from open_latent_interfaces.interventions import intervened_next_token_logits
from open_latent_interfaces.phase1_data import (
    build_phase1_additions,
    phase1_addition_sha256,
)
from open_latent_interfaces.probes import BinaryRidgeProbe, binary_metrics


def digit_metrics(
    logits: torch.Tensor,
    target_digits: torch.Tensor,
    original_digits: torch.Tensor,
    digit_token_ids: torch.Tensor,
) -> dict[str, float]:
    digit_scores = logits[:, digit_token_ids]
    rows = torch.arange(len(logits))
    target_scores = digit_scores[rows, target_digits]
    target_other = digit_scores.clone()
    target_other[rows, target_digits] = -torch.inf
    original_scores = digit_scores[rows, original_digits]
    original_other = digit_scores.clone()
    original_other[rows, original_digits] = -torch.inf
    predictions = digit_scores.argmax(dim=1)
    vocabulary_predictions = logits.argmax(dim=1)
    target_token_ids = digit_token_ids[target_digits]
    return {
        "counterfactual_digit_accuracy": float(
            (predictions == target_digits).float().mean()
        ),
        "counterfactual_vocabulary_top1": float(
            (vocabulary_predictions == target_token_ids).float().mean()
        ),
        "counterfactual_mean_digit_margin": float(
            (target_scores - target_other.max(dim=1).values).mean()
        ),
        "original_digit_accuracy": float(
            (predictions == original_digits).float().mean()
        ),
        "original_mean_digit_margin": float(
            (original_scores - original_other.max(dim=1).values).mean()
        ),
        "mean_counterfactual_minus_original_logit": float(
            (target_scores - original_scores).mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    probe_config_path = Path(config["probe_config"])
    observed_probe_config_hash = hashlib.sha256(
        probe_config_path.read_bytes()
    ).hexdigest()
    if observed_probe_config_hash != config["probe_config_sha256"]:
        raise SystemExit("probe config hash mismatch")
    probe_config = json.loads(probe_config_path.read_text())
    examples = build_phase1_additions(**probe_config["dataset"]["parameters"])
    observed_hash = phase1_addition_sha256(examples)
    if observed_hash != probe_config["dataset"]["sha256"]:
        raise SystemExit("Phase 1 dataset hash mismatch")
    train = [example for example in examples if example.split == "train"]
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

    def render(example: Any) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": example.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    all_examples = train + development
    all_prompts = [render(example) for example in all_examples]
    hidden_index = config["hidden_state_index"]
    capture = ActivationCapture(model, tokenizer, device=device)
    started = time.perf_counter()
    activations = capture.capture_last_token(
        all_prompts,
        hidden_state_indices=[hidden_index],
        batch_size=probe_config["batch_size"],
    )[hidden_index].values
    train_count = len(train)
    train_values = activations[:train_count]
    development_values = activations[train_count:]
    train_labels = torch.tensor([example.tens_carry for example in train])
    development_labels = torch.tensor(
        [example.tens_carry for example in development]
    )
    probe = BinaryRidgeProbe.fit(
        train_values, train_labels, l2=probe_config["probe_l2"]
    )
    shuffled_probe = BinaryRidgeProbe.fit(
        train_values,
        train_labels[torch.randperm(len(train_labels), generator=torch.Generator().manual_seed(
            config["random_seed"]
        ))],
        l2=probe_config["probe_l2"],
    )

    eligible_indices = []
    counterfactual_results = []
    for index, example in enumerate(development):
        desired_carry = 1 - example.tens_carry
        counterfactual = example.result + (desired_carry - example.tens_carry) * 100
        if 100 <= counterfactual <= 999:
            eligible_indices.append(index)
            counterfactual_results.append(counterfactual)
    selected = torch.tensor(eligible_indices)
    selected_values = development_values[selected]
    selected_labels = development_labels[selected]
    desired_labels = 1 - selected_labels
    selected_examples = [development[index] for index in eligible_indices]
    prompts = [render(example) for example in selected_examples]
    original_digits = torch.tensor(
        [int(str(example.result)[0]) for example in selected_examples]
    )
    target_digits = torch.tensor(
        [int(str(result)[0]) for result in counterfactual_results]
    )
    digit_token_ids = torch.tensor(
        [
            int(tokenizer(str(digit), add_special_tokens=False)["input_ids"][0])
            for digit in range(10)
        ]
    )
    base_logits = capture.next_token_logits(prompts, batch_size=probe_config["batch_size"])
    conditions = {
        "base": {
            "metrics": digit_metrics(
                base_logits,
                target_digits,
                original_digits,
                digit_token_ids,
            ),
            "mean_delta_norm": 0.0,
            "mean_relative_norm": 0.0,
        }
    }

    for strength in config["strengths"]:
        targeted = probe.minimal_label_shift(
            selected_values,
            desired_labels,
            margin=config["probe_margin"],
            strength=strength,
            max_relative_norm=config["max_relative_norm"],
        )
        shuffled_delta = shuffled_probe.minimal_label_shift(
            selected_values,
            desired_labels,
            margin=config["probe_margin"],
            strength=strength,
            max_relative_norm=None,
        )
        shuffled_delta = norm_match(shuffled_delta, targeted.norm(dim=1))
        random_delta = random_norm_matched(
            tuple(targeted.shape),
            targeted.norm(dim=1),
            seed=config["random_seed"] + int(strength * 100),
        )
        for name, delta in (
            ("targeted", targeted),
            ("shuffled_label", shuffled_delta),
            ("random", random_delta),
        ):
            logits = intervened_next_token_logits(
                model,
                tokenizer,
                prompts,
                hidden_state_index=hidden_index,
                deltas=delta,
                device=device,
            )
            shifted_probe_scores = probe.score(selected_values + delta)
            result = digit_metrics(
                logits,
                target_digits,
                original_digits,
                digit_token_ids,
            )
            result["desired_carry_probe_accuracy"] = float(
                ((shifted_probe_scores >= 0).long() == desired_labels)
                .float()
                .mean()
            )
            conditions[f"{name}|strength={strength:g}"] = {
                "metrics": result,
                "mean_delta_norm": float(delta.norm(dim=1).mean()),
                "mean_relative_norm": float(
                    (delta.norm(dim=1) / selected_values.norm(dim=1)).mean()
                ),
            }

    report = {
        "schema_version": "oli.phase1-carry-intervention/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "development_only",
        "model": probe_config["model"],
        "dataset": {
            "sha256": observed_hash,
            "eligible_development_examples": len(selected_examples),
            "audit_examples_unopened": sum(
                example.split == "audit" for example in examples
            ),
        },
        "candidate": {
            "variable": "tens_carry",
            "hidden_state_index": hidden_index,
            "decoder_block": hidden_index - 1,
            "probe_development_metrics": binary_metrics(
                probe.score(development_values), development_labels
            ),
        },
        "intervention": {
            "desired_state": "opposite tens-carry label",
            "expected_counterfactual": "result +/- 100",
            "probe_margin": config["probe_margin"],
            "max_relative_norm": config["max_relative_norm"],
            "strengths": config["strengths"],
        },
        "conditions": conditions,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Probe-directed residual shifts test one-step hundreds-digit effects. "
            "They do not establish full-result sufficiency or on-manifold writes."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
