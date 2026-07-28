#!/usr/bin/env python3
"""Map exact arithmetic variables with train-only ridge probes at every layer."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.phase1_data import (
    build_phase1_additions,
    phase1_addition_sha256,
)
from open_latent_interfaces.probes import (
    BinaryRidgeProbe,
    CategoricalRidgeProbe,
    binary_metrics,
    categorical_metrics,
)


def digits(value: int) -> tuple[int, int, int]:
    text = f"{value:03d}"
    return int(text[0]), int(text[1]), int(text[2])


def labels_for(examples: list[Any]) -> dict[str, tuple[torch.Tensor, int]]:
    categorical: dict[str, tuple[list[int], int]] = {
        "operand_a_hundreds": ([], 10),
        "operand_a_tens": ([], 10),
        "operand_a_ones": ([], 10),
        "operand_b_hundreds": ([], 10),
        "operand_b_tens": ([], 10),
        "operand_b_ones": ([], 10),
        "result_hundreds": ([], 10),
        "result_tens": ([], 10),
        "result_ones": ([], 10),
    }
    binary: dict[str, tuple[list[int], int]] = {
        "ones_carry": ([], 2),
        "tens_carry": ([], 2),
    }
    for example in examples:
        for prefix, value in (
            ("operand_a", example.operand_a),
            ("operand_b", example.operand_b),
            ("result", example.result),
        ):
            for suffix, digit in zip(
                ("hundreds", "tens", "ones"), digits(value), strict=True
            ):
                categorical[f"{prefix}_{suffix}"][0].append(digit)
        binary["ones_carry"][0].append(example.ones_carry)
        binary["tens_carry"][0].append(example.tens_carry)
    return {
        name: (torch.tensor(values), classes)
        for name, (values, classes) in {**categorical, **binary}.items()
    }


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

    def render(example: Any) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": example.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    all_examples = train + development
    rendered = [render(example) for example in all_examples]
    started = time.perf_counter()
    capture = ActivationCapture(model, tokenizer, device=device)
    captured = capture.capture_last_token(
        rendered,
        hidden_state_indices=config["hidden_state_indices"],
        batch_size=config["batch_size"],
    )
    train_count = len(train)
    train_labels = labels_for(train)
    development_labels = labels_for(development)
    layer_results = {}
    l2 = config["probe_l2"]
    shuffle_seed = config["shuffle_seed"]

    for hidden_index, layer in sorted(captured.items()):
        train_values = layer.values[:train_count]
        dev_values = layer.values[train_count:]
        variables = {}
        fitted: dict[str, Any] = {}
        for variable_index, (name, (labels, classes)) in enumerate(
            train_labels.items()
        ):
            dev_target = development_labels[name][0]
            if classes == 2 and "carry" in name:
                probe = BinaryRidgeProbe.fit(train_values, labels, l2=l2)
                score = probe.score(dev_values)
                metric = binary_metrics(score, dev_target)
                shuffled_probe = BinaryRidgeProbe.fit(
                    train_values,
                    shuffled(labels, seed=shuffle_seed + variable_index),
                    l2=l2,
                )
                control = binary_metrics(
                    shuffled_probe.score(dev_values), dev_target
                )
                predictions = probe.predict(dev_values)
            else:
                probe = CategoricalRidgeProbe.fit(
                    train_values,
                    labels,
                    number_of_classes=classes,
                    l2=l2,
                )
                score = probe.score(dev_values)
                metric = categorical_metrics(
                    score, dev_target, number_of_classes=classes
                )
                shuffled_probe = CategoricalRidgeProbe.fit(
                    train_values,
                    shuffled(labels, seed=shuffle_seed + variable_index),
                    number_of_classes=classes,
                    l2=l2,
                )
                control = categorical_metrics(
                    shuffled_probe.score(dev_values),
                    dev_target,
                    number_of_classes=classes,
                )
                predictions = probe.predict(dev_values)
            fitted[name] = predictions
            variables[name] = {
                "metrics": metric,
                "shuffled_label_control": control,
            }
        result_exact = torch.ones(len(development), dtype=torch.bool)
        operand_a_exact = torch.ones(len(development), dtype=torch.bool)
        operand_b_exact = torch.ones(len(development), dtype=torch.bool)
        for prefix, accumulator in (
            ("result", result_exact),
            ("operand_a", operand_a_exact),
            ("operand_b", operand_b_exact),
        ):
            for suffix in ("hundreds", "tens", "ones"):
                name = f"{prefix}_{suffix}"
                accumulator &= fitted[name] == development_labels[name][0]
        layer_results[str(hidden_index)] = {
            "variables": variables,
            "exact_result_accuracy": float(result_exact.float().mean()),
            "exact_operand_a_accuracy": float(operand_a_exact.float().mean()),
            "exact_operand_b_accuracy": float(operand_b_exact.float().mean()),
            "mean_activation_norm": float(dev_values.norm(dim=1).mean()),
        }

    report = {
        "schema_version": "oli.phase1-probe-map/v1",
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
            "train_template": train[0].template_family,
            "development_template": development[0].template_family,
        },
        "probe": {"kind": "ridge", "l2": l2},
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
            "Development probe decodability is correlational. Audit and causal "
            "necessity/sufficiency remain unopened."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
