#!/usr/bin/env python3
"""Run a frozen arithmetic capability split on an untouched causal LM."""

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

from open_latent_interfaces.capability import (
    build_capability_sweep,
    capability_dataset_sha256,
    parse_first_integer,
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
    parser.add_argument("--split", choices=("development", "audit"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    if args.split == "audit" and not config.get("audit_authorized", False):
        raise SystemExit(
            "audit is sealed; commit a selected development regime and set "
            "audit_authorized=true in a new config revision before running"
        )
    examples = build_capability_sweep(
        seed=config["dataset"]["seed"],
        development_pairs=config["dataset"]["development_pairs"],
        audit_pairs=config["dataset"]["audit_pairs"],
        protocol_version=config["dataset"].get("protocol_version", "v1"),
    )
    observed_hash = capability_dataset_sha256(examples)
    if observed_hash != config["dataset"]["sha256"]:
        raise SystemExit(
            f"dataset hash mismatch: {observed_hash} != {config['dataset']['sha256']}"
        )
    selected = [example for example in examples if example.split == args.split]
    if args.split == "audit":
        selected_regime = config.get("selected_regime")
        if not selected_regime:
            raise SystemExit("authorized audit config must name one selected_regime")
        selected = [
            example for example in selected if example.regime == selected_regime
        ]
    evaluated_presentations = config["generation"].get(
        "presentations", ["raw", "chat"]
    )
    unknown_presentations = set(evaluated_presentations) - {"raw", "chat"}
    if unknown_presentations:
        raise SystemExit(
            f"unknown generation presentations: {sorted(unknown_presentations)}"
        )
    if not evaluated_presentations:
        raise SystemExit("generation presentations cannot be empty")
    selected = [
        example
        for example in selected
        if example.presentation in evaluated_presentations
    ]

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"], revision=config["model"]["revision"]
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["revision"],
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    rendered_prompts = []
    for example in selected:
        if example.presentation == "chat":
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": example.prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered = example.prompt
        rendered_prompts.append(rendered)

    rows = []
    started = time.perf_counter()
    batch_size = config["generation"]["batch_size"]
    with torch.inference_mode():
        for start in range(0, len(selected), batch_size):
            batch_examples = selected[start : start + batch_size]
            batch_prompts = rendered_prompts[start : start + batch_size]
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
                max_new_tokens=config["generation"]["max_new_tokens"],
                pad_token_id=tokenizer.pad_token_id,
            )
            continuations = generated[:, encoded["input_ids"].shape[1] :]
            responses = tokenizer.batch_decode(
                continuations, skip_special_tokens=True
            )
            for example, rendered, response in zip(
                batch_examples, batch_prompts, responses, strict=True
            ):
                parsed = parse_first_integer(response)
                rows.append(
                    {
                        "example_id": example.example_id,
                        "regime": example.regime,
                        "template_family": example.template_family,
                        "presentation": example.presentation,
                        "operand_a": example.operand_a,
                        "operand_b": example.operand_b,
                        "result": example.result,
                        "rendered_prompt_sha256": hashlib.sha256(
                            rendered.encode()
                        ).hexdigest(),
                        "response": response,
                        "parsed": parsed,
                        "exact": parsed == example.result,
                    }
                )

    cells: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    regimes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cells[(row["regime"], row["template_family"], row["presentation"])].append(
            row
        )
        regimes[row["regime"]].append(row)
    primary_presentations = config["selection_rule"].get(
        "primary_presentations", ["raw", "chat"]
    )
    primary_rows = [
        row for row in rows if row["presentation"] in primary_presentations
    ]
    primary_regimes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    primary_cells: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in primary_rows:
        primary_regimes[row["regime"]].append(row)
        primary_cells[
            (row["regime"], row["template_family"], row["presentation"])
        ].append(row)
    aggregate_threshold = config["selection_rule"]["aggregate_exact_accuracy"]
    cell_threshold = config["selection_rule"]["worst_cell_exact_accuracy"]
    gate_by_regime = {}
    for regime, regime_rows in sorted(primary_regimes.items()):
        aggregate = summarize(regime_rows)["exact_accuracy"]
        relevant_cells = [
            summarize(cell_rows)["exact_accuracy"]
            for key, cell_rows in primary_cells.items()
            if key[0] == regime
        ]
        worst_cell = min(relevant_cells)
        gate_by_regime[regime] = {
            "aggregate_exact_accuracy": aggregate,
            "worst_cell_exact_accuracy": worst_cell,
            "passes": aggregate >= aggregate_threshold
            and worst_cell >= cell_threshold,
        }
    report = {
        "schema_version": "oli.capability-sweep/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "stage": args.split,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "dataset_sha256": observed_hash,
        "model": config["model"],
        "generation": config["generation"],
        "evaluated_presentations": evaluated_presentations,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "overall": summarize(rows),
        "primary_selection": {
            "presentations": primary_presentations,
            "overall": summarize(primary_rows),
            "regimes": gate_by_regime,
        },
        "regimes": {
            regime: summarize(regime_rows)
            for regime, regime_rows in sorted(regimes.items())
        },
        "cells": {
            "|".join(key): summarize(cell_rows)
            for key, cell_rows in sorted(cells.items())
        },
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Development results select a task regime only. Audit remains sealed, "
            "and behavioral competence does not establish a latent mechanism."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
