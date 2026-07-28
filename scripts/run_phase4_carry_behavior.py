#!/usr/bin/env python3
"""Verify Phi behavior on the frozen matched carry quartets."""

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
from open_latent_interfaces.phase4_data import (
    build_phase4_carry_quartets,
    phase4_carry_sha256,
)
from open_latent_interfaces.prefill import (
    contextual_continuation_ids,
    render_prefilled_chat,
    verify_decimal_digit_contract,
)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "exact_correct": sum(row["exact"] for row in rows),
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
        raise SystemExit(f"refusing to overwrite behavior result: {args.output}")

    config = json.loads(args.config.read_text())
    dataset_path = Path(config["dataset_config"])
    dataset_config = json.loads(dataset_path.read_text())
    if dataset_config.get("audit_authorized", False):
        raise SystemExit("behavior development requires a sealed audit")
    examples = build_phase4_carry_quartets(
        **dataset_config["dataset"]["parameters"]
    )
    observed_hash = phase4_carry_sha256(examples)
    if observed_hash != dataset_config["dataset"]["sha256"]:
        raise SystemExit("Phase 4 dataset hash mismatch")
    splits = config["splits"]
    if "audit" in splits:
        raise SystemExit("behavior development may not evaluate audit")
    selected = [example for example in examples if example.split in splits]

    device = torch.device(args.device)
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
            raise SystemExit(f"token contract failed for {example.example_id}")

    model = AutoModelForCausalLM.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        torch_dtype=getattr(torch, args.dtype),
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    started = time.perf_counter()

    rows = []
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
                        "quartet_id": example.quartet_id,
                        "split": example.split,
                        "variant": example.variant,
                        "operand_a": example.operand_a,
                        "operand_b": example.operand_b,
                        "result": example.result,
                        "ones_carry": example.ones_carry,
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
    by_quartet: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
        by_quartet[row["quartet_id"]].append(row)
    split_summaries = {}
    for split, split_rows in sorted(by_split.items()):
        quartet_rows = [
            quartet
            for quartet in by_quartet.values()
            if quartet[0]["split"] == split
        ]
        complete_correct = sum(
            len(quartet) == 4 and all(row["exact"] for row in quartet)
            for quartet in quartet_rows
        )
        row_summary = summarize(split_rows)
        complete_rate = complete_correct / len(quartet_rows)
        split_summaries[split] = {
            **row_summary,
            "quartets": len(quartet_rows),
            "complete_correct_quartets": complete_correct,
            "complete_quartet_accuracy": complete_rate,
            "passes": (
                row_summary["exact_accuracy"]
                >= config["gate"]["minimum_row_accuracy"]
                and complete_rate
                >= config["gate"]["minimum_complete_quartet_accuracy"]
                and row_summary["parse_rate"] == 1.0
            ),
        }
    report = {
        "schema_version": "oli.phase4-carry-behavior/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "development_only",
        "model": model_config,
        "dataset_sha256": observed_hash,
        "dataset_config_sha256": hashlib.sha256(
            dataset_path.read_bytes()
        ).hexdigest(),
        "assistant_prefix": dataset_config["assistant_prefix"],
        "digit_token_ids": digit_token_ids,
        "generation": config["generation"],
        "gate": config["gate"],
        "splits": split_summaries,
        "passes": all(summary["passes"] for summary in split_summaries.values()),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
        "claim_boundary": (
            "Behavior and token-alignment gate for matched carry quartets. "
            "No latent or causal carry claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
