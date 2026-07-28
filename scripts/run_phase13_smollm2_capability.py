#!/usr/bin/env python3
"""Measure frozen SmolLM2 arithmetic behavior on the exposed Phase 13 fit split."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from run_phase3_native_boundary import verify_sha256
from run_phase4_carry_sequence_boundary import value_sha256
from run_phase8_operand_reader_selection import render_and_locate
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.phase13_data import (
    build_phase13_examples,
    phase13_sha256,
)
from open_latent_interfaces.prefill import verify_decimal_digit_contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite capability result")
    config = json.loads(args.config.read_text())
    if str(args.output) != config["output"]:
        raise SystemExit("capability output differs from frozen path")
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if runner_hash != config["runner_sha256"]:
        raise SystemExit("capability runner hash mismatch")
    for dependency, expected_hash in config["code_dependencies"].items():
        verify_sha256(Path(dependency), expected_hash)
    dataset_path = Path(config["dataset_config"])
    verify_sha256(dataset_path, config["dataset_config_sha256"])
    onboarding_path = Path(config["onboarding_result"])
    verify_sha256(onboarding_path, config["onboarding_result_sha256"])
    onboarding = json.loads(onboarding_path.read_text())
    if onboarding.get("passes") is not True:
        raise SystemExit("model onboarding did not pass")

    dataset_config = json.loads(dataset_path.read_text())
    if dataset_config.get("audit_authorized", False):
        raise SystemExit("capability measurement may not use sealed audit data")
    examples = build_phase13_examples(
        **dataset_config["dataset"]["parameters"]
    )
    if phase13_sha256(examples) != dataset_config["dataset"]["sha256"]:
        raise SystemExit("Phase 13 dataset hash mismatch")
    fit = [row for row in examples if row.split == "fit"]
    if value_sha256([row.example_id for row in fit]) != config[
        "fit_examples_sha256"
    ]:
        raise SystemExit("fit example hash mismatch")

    model_config = dataset_config["model"]
    if model_config != config["model"]:
        raise SystemExit("capability model differs from frozen model")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rendered, _, token_contract = render_and_locate(
        tokenizer,
        fit,
        dataset_config["assistant_prefix"],
    )
    if value_sha256(rendered) != config["rendered_prompts_sha256"]:
        raise SystemExit("rendered prompt hash mismatch")
    if value_sha256(token_contract) != config["token_contract_sha256"]:
        raise SystemExit("token contract hash mismatch")
    digit_token_ids = verify_decimal_digit_contract(tokenizer, rendered[0])
    if value_sha256(digit_token_ids) != config["digit_token_ids_sha256"]:
        raise SystemExit("digit-token map hash mismatch")
    inverse_digits = {
        token_id: digit for digit, token_id in digit_token_ids.items()
    }

    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        dtype=getattr(torch, args.dtype),
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("base model parameters must remain frozen")
    started = time.perf_counter()
    output_rows = []
    position_correct = [0, 0, 0]
    exact_correct = 0
    digit_token_count = 0
    for start in range(0, len(rendered), config["batch_size"]):
        prompts = rendered[start : start + config["batch_size"]]
        expected_rows = fit[start : start + config["batch_size"]]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in encoded.items()
        }
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=3,
                min_new_tokens=3,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_ids = generated[:, encoded["input_ids"].shape[1] :].cpu()
        for ids, expected in zip(new_ids.tolist(), expected_rows, strict=True):
            digits = [
                inverse_digits[token_id]
                for token_id in ids
                if token_id in inverse_digits
            ]
            digit_token_count += len(digits)
            target_digits = [
                expected.leading_digit,
                expected.tens_digit,
                expected.ones_digit,
            ]
            for index in range(min(3, len(digits))):
                position_correct[index] += digits[index] == target_digits[index]
            exact = len(digits) == 3 and digits == target_digits
            exact_correct += exact
            output_rows.append(
                {
                    "example_id": expected.example_id,
                    "generated_token_ids": ids,
                    "generated_text": tokenizer.decode(ids),
                    "parsed_digits": digits if len(digits) == 3 else None,
                    "target": expected.result,
                    "exact": exact,
                }
            )
    report = {
        "schema_version": "oli.phase13-smollm2-capability/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "exposed_fit_measurement",
        "model": model_config,
        "dataset_sha256": dataset_config["dataset"]["sha256"],
        "split": "fit",
        "n": len(fit),
        "metrics": {
            "exact_correct": exact_correct,
            "exact_accuracy": exact_correct / len(fit),
            "position_correct": position_correct,
            "position_accuracy": [
                value / len(fit) for value in position_correct
            ],
            "digit_token_count": digit_token_count,
            "digit_token_rate": digit_token_count / (3 * len(fit)),
        },
        "rows": output_rows,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "runner_sha256": runner_hash,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Exposed fit-split baseline measurement only. This is not reader, "
            "writer, development, or audit evidence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
