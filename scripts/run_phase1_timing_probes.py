#!/usr/bin/env python3
"""Probe each next result digit at its teacher-forced generation boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.phase1_data import (
    build_phase1_additions,
    phase1_addition_sha256,
)
from open_latent_interfaces.probes import (
    CategoricalRidgeProbe,
    categorical_metrics,
)


def result_digits(value: int) -> list[int]:
    return [int(digit) for digit in str(value)]


def shuffled(labels: torch.Tensor, *, seed: int) -> torch.Tensor:
    values = labels.tolist()
    random.Random(seed).shuffle(values)
    return torch.tensor(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    examples = build_phase1_additions(**config["dataset"]["parameters"])
    observed_hash = phase1_addition_sha256(examples)
    if observed_hash != config["dataset"]["sha256"]:
        raise SystemExit("Phase 1 dataset hash mismatch")
    train = [example for example in examples if example.split == "train"]
    development = [
        example for example in examples if example.split == "development"
    ]

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"], revision=config["model"]["revision"]
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["revision"],
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    def render(example: object, step: int) -> str:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": example.prompt}],  # type: ignore[attr-defined]
            tokenize=False,
            add_generation_prompt=True,
        )
        prefix = str(example.result)[:step]  # type: ignore[attr-defined]
        return prompt + prefix

    all_examples = train + development
    train_count = len(train)
    capture = ActivationCapture(model, tokenizer, device=device)
    captures = {}
    started = time.perf_counter()
    for step in range(3):
        captures[step] = capture.capture_last_token(
            [render(example, step) for example in all_examples],
            hidden_state_indices=config["hidden_state_indices"],
            batch_size=config["batch_size"],
        )

    train_targets = {
        step: torch.tensor([result_digits(example.result)[step] for example in train])
        for step in range(3)
    }
    development_targets = {
        step: torch.tensor(
            [result_digits(example.result)[step] for example in development]
        )
        for step in range(3)
    }
    layer_results = {}
    for hidden_index in config["hidden_state_indices"]:
        step_results = {}
        step_predictions = {}
        for step in range(3):
            values = captures[step][hidden_index].values
            train_values = values[:train_count]
            dev_values = values[train_count:]
            probe = CategoricalRidgeProbe.fit(
                train_values,
                train_targets[step],
                number_of_classes=10,
                l2=config["probe_l2"],
            )
            scores = probe.score(dev_values)
            control_probe = CategoricalRidgeProbe.fit(
                train_values,
                shuffled(
                    train_targets[step],
                    seed=config["shuffle_seed"] + step,
                ),
                number_of_classes=10,
                l2=config["probe_l2"],
            )
            step_results[str(step)] = {
                "metrics": categorical_metrics(
                    scores,
                    development_targets[step],
                    number_of_classes=10,
                ),
                "shuffled_label_control": categorical_metrics(
                    control_probe.score(dev_values),
                    development_targets[step],
                    number_of_classes=10,
                ),
            }
            step_predictions[step] = probe.predict(dev_values)
        exact = torch.ones(len(development), dtype=torch.bool)
        for step in range(3):
            exact &= step_predictions[step] == development_targets[step]
        layer_results[str(hidden_index)] = {
            "steps": step_results,
            "teacher_forced_full_result_accuracy": float(exact.float().mean()),
        }

    report = {
        "schema_version": "oli.phase1-timing-probes/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "development_only",
        "model": config["model"],
        "dataset": {
            "sha256": observed_hash,
            "train_examples": len(train),
            "development_examples": len(development),
            "audit_examples_unopened": sum(
                example.split == "audit" for example in examples
            ),
        },
        "probe": {"kind": "categorical_ridge", "l2": config["probe_l2"]},
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "layers": layer_results,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Each suffix probe sees the correct preceding answer digits. Results "
            "map autoregressive timing and do not show simultaneous pre-output "
            "decodability of the full result."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
