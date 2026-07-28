#!/usr/bin/env python3
"""Prepare and fit a reproducible Jacobian lens in an isolated upstream env."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

JACOBIAN_LENS_REPOSITORY = "https://github.com/anthropics/jacobian-lens"
JACOBIAN_LENS_REVISION = "581d398613e5602a5af361e1c34d3a92ea82ba8e"


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_external_packages() -> tuple[Any, Any, Any]:
    try:
        import datasets
        import jlens
        import transformers
    except ImportError as exc:
        raise SystemExit(
            "Run this script inside an isolated environment containing the pinned "
            "jacobian-lens release and its dev dependencies."
        ) from exc
    return datasets, jlens, transformers


def load_corpus(config: dict[str, Any]) -> Any:
    datasets, _, _ = load_external_packages()
    return datasets.load_dataset(
        config["dataset"]["id"],
        config["dataset"]["config"],
        split=config["dataset"]["split"],
        revision=config["dataset"]["revision"],
    )


def prepare(args: argparse.Namespace) -> None:
    _, _, transformers = load_external_packages()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model, revision=args.model_revision
    )
    seed_config = {
        "schema_version": "oli.jlens-fit-selection/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": args.purpose,
        "model": {"id": args.model, "revision": args.model_revision},
        "dataset": {
            "id": args.dataset,
            "config": args.dataset_config,
            "split": args.dataset_split,
            "revision": args.dataset_revision,
            "text_field": args.text_field,
        },
        "selection": {
            "seed": args.seed,
            "n_prompts": args.n_prompts,
            "min_tokens": args.min_tokens,
            "max_tokens": args.max_tokens,
        },
        "fit": {
            "source_layers": args.source_layers,
            "target_layer": args.target_layer,
            "dim_batch": args.dim_batch,
            "max_seq_len": args.max_seq_len,
            "skip_first": args.skip_first,
        },
        "upstream": {
            "repository": JACOBIAN_LENS_REPOSITORY,
            "revision": JACOBIAN_LENS_REVISION,
            "license": "Apache-2.0",
        },
    }
    corpus = load_corpus(seed_config)
    eligible: list[dict[str, Any]] = []
    for row_index, row in enumerate(corpus):
        text = row[args.text_field].strip()
        if not text:
            continue
        token_count = len(
            tokenizer(
                text,
                truncation=True,
                max_length=args.max_tokens + 1,
                add_special_tokens=True,
            )["input_ids"]
        )
        if args.min_tokens <= token_count <= args.max_tokens:
            eligible.append(
                {
                    "row_index": row_index,
                    "text_sha256": text_sha256(text),
                    "token_count": token_count,
                }
            )
    if len(eligible) < args.n_prompts:
        raise SystemExit(
            f"only {len(eligible)} eligible rows for {args.n_prompts} prompts"
        )
    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    selected = eligible[: args.n_prompts]
    seed_config["selection"]["eligible_rows"] = len(eligible)
    seed_config["prompts"] = selected
    selection_payload = json.dumps(
        selected, sort_keys=True, separators=(",", ":")
    ).encode()
    seed_config["selection"]["selected_rows_sha256"] = hashlib.sha256(
        selection_payload
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(seed_config, indent=2, sort_keys=True) + "\n")
    print(f"wrote frozen selection to {args.output}")


def fit(args: argparse.Namespace) -> None:
    _, jlens, transformers = load_external_packages()
    config = json.loads(args.config.read_text())
    if config["upstream"]["revision"] != JACOBIAN_LENS_REVISION:
        raise SystemExit("config does not pin the adapter's audited J-lens revision")

    corpus = load_corpus(config)
    prompts: list[str] = []
    for record in config["prompts"]:
        text = corpus[record["row_index"]][config["dataset"]["text_field"]].strip()
        if text_sha256(text) != record["text_sha256"]:
            raise SystemExit(f"dataset drift at row {record['row_index']}")
        prompts.append(text)

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config["model"]["id"], revision=config["model"]["revision"]
    )
    started = time.perf_counter()
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["revision"],
        dtype=dtype,
    ).to(device)
    lens_model = jlens.from_hf(hf_model, tokenizer)
    fit_config = config["fit"]
    lens = jlens.fit(
        lens_model,
        prompts,
        source_layers=fit_config["source_layers"],
        target_layer=fit_config["target_layer"],
        dim_batch=fit_config["dim_batch"],
        max_seq_len=fit_config["max_seq_len"],
        skip_first=fit_config["skip_first"],
        checkpoint_path=str(args.checkpoint),
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lens.save(str(args.output))
    elapsed = time.perf_counter() - started
    report = {
        "schema_version": "oli.jlens-fit-result/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "selection_config": str(args.config),
        "selection_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "lens_path": str(args.output),
        "lens_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "checkpoint_path": str(args.checkpoint),
        "model": config["model"],
        "dataset": config["dataset"],
        "upstream": config["upstream"],
        "n_prompts": lens.n_prompts,
        "d_model": lens.d_model,
        "source_layers": lens.source_layers,
        "target_layer": fit_config["target_layer"],
        "dtype": args.dtype,
        "device": args.device,
        "elapsed_seconds": elapsed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"saved lens to {args.output}")
    print(f"saved fit report to {args.report}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare", help="select and freeze corpus rows")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--purpose", default="phase1-development")
    prep.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    prep.add_argument("--model-revision", required=True)
    prep.add_argument("--dataset", default="Salesforce/wikitext")
    prep.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    prep.add_argument("--dataset-split", default="train")
    prep.add_argument("--dataset-revision", required=True)
    prep.add_argument("--text-field", default="text")
    prep.add_argument("--seed", type=int, default=20260727)
    prep.add_argument("--n-prompts", type=int, default=24)
    prep.add_argument("--min-tokens", type=int, default=40)
    prep.add_argument("--max-tokens", type=int, default=64)
    prep.add_argument(
        "--source-layers", type=int, nargs="+", default=list(range(23))
    )
    prep.add_argument("--target-layer", type=int, default=23)
    prep.add_argument("--dim-batch", type=int, default=16)
    prep.add_argument("--max-seq-len", type=int, default=64)
    prep.add_argument("--skip-first", type=int, default=16)
    prep.set_defaults(func=prepare)

    run = subparsers.add_parser("fit", help="fit from a committed selection")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--report", type=Path, required=True)
    run.add_argument("--device", default="mps")
    run.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    run.add_argument("--checkpoint-every", type=int, default=4)
    run.add_argument("--no-resume", action="store_true")
    run.set_defaults(func=fit)

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
