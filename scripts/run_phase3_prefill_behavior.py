#!/usr/bin/env python3
"""Verify Phi addition behavior under the frozen assistant-prefix contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.capability import parse_first_integer
from open_latent_interfaces.phase3_data import (
    build_phase3_additions,
    phase3_addition_sha256,
)
from open_latent_interfaces.prefill import (
    contextual_continuation_ids,
    render_prefilled_chat,
    verify_decimal_digit_contract,
)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "exact_accuracy": sum(row["exact"] for row in rows) / len(rows),
        "parse_rate": sum(row["parsed"] is not None for row in rows) / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing result: {args.output}")
    config = json.loads(args.config.read_text())
    dataset_config_path = Path(config["dataset_config"])
    dataset_config = json.loads(dataset_config_path.read_text())
    if dataset_config.get("audit_authorized", False):
        raise SystemExit("development behavior runner requires a sealed audit")
    examples = build_phase3_additions(**dataset_config["dataset"]["parameters"])
    observed_hash = phase3_addition_sha256(examples)
    if observed_hash != dataset_config["dataset"]["sha256"]:
        raise SystemExit("Phase 3 dataset hash mismatch")
    splits = config["splits"]
    if "audit" in splits:
        raise SystemExit("behavior development may not evaluate the audit split")
    selected = [example for example in examples if example.split in splits]

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    model_config = dataset_config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rendered = [
        render_prefilled_chat(
            tokenizer,
            example.prompt,
            assistant_prefix=dataset_config["assistant_prefix"],
        )
        for example in selected
    ]
    digit_token_ids = verify_decimal_digit_contract(tokenizer, rendered[0])
    for prompt, example in zip(rendered, selected, strict=True):
        expected_ids = [digit_token_ids[int(digit)] for digit in str(example.result)]
        if contextual_continuation_ids(
            tokenizer,
            prompt,
            str(example.result),
        ) != expected_ids:
            raise SystemExit(
                f"token contract failed for {example.example_id}"
            )

    model = AutoModelForCausalLM.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    rows = []
    started = time.perf_counter()
    batch_size = config["generation"]["batch_size"]
    with torch.inference_mode():
        for start in range(0, len(selected), batch_size):
            batch_examples = selected[start : start + batch_size]
            batch_prompts = rendered[start : start + batch_size]
            encoded = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config["generation"]["max_input_tokens"],
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=3,
                pad_token_id=tokenizer.pad_token_id,
            )
            continuation = generated[:, encoded["input_ids"].shape[1] :]
            responses = tokenizer.batch_decode(
                continuation,
                skip_special_tokens=True,
            )
            for example, prompt, response, token_ids in zip(
                batch_examples,
                batch_prompts,
                responses,
                continuation.tolist(),
                strict=True,
            ):
                parsed = parse_first_integer(response)
                rows.append(
                    {
                        "example_id": example.example_id,
                        "split": example.split,
                        "operand_a": example.operand_a,
                        "operand_b": example.operand_b,
                        "result": example.result,
                        "rendered_prompt_sha256": hashlib.sha256(
                            prompt.encode()
                        ).hexdigest(),
                        "response": response,
                        "generated_token_ids": token_ids,
                        "parsed": parsed,
                        "exact": parsed == example.result,
                    }
                )

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
    gate_threshold = config["minimum_exact_accuracy"]
    split_summaries = {
        split: {
            **summarize(split_rows),
            "passes": summarize(split_rows)["exact_accuracy"] >= gate_threshold,
        }
        for split, split_rows in sorted(by_split.items())
    }
    report = {
        "schema_version": "oli.phase3-prefill-behavior/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "dataset_config_sha256": hashlib.sha256(
            dataset_config_path.read_bytes()
        ).hexdigest(),
        "dataset_sha256": observed_hash,
        "model": model_config,
        "assistant_prefix": dataset_config["assistant_prefix"],
        "digit_token_ids": digit_token_ids,
        "generation": config["generation"],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "overall": summarize(rows),
        "splits": split_summaries,
        "passes": all(summary["passes"] for summary in split_summaries.values()),
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
        "claim_boundary": (
            "This verifies behavior and token alignment under a frozen prefill. "
            "It does not establish a latent mechanism."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
