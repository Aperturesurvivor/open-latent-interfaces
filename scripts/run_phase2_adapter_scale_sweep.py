#!/usr/bin/env python3
"""Select fixed-adapter scales, then evaluate the frozen development protocol."""

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
from run_phase1_conditional_transport_bridge import render_prompts
from run_phase2_causal_adapter import result_list_sha256
from run_phase2_scaled_adapter import evaluate_condition, select_adapters
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.adapter import load_online_adapter
from open_latent_interfaces.evaluation import select_bounded_candidate
from open_latent_interfaces.phase2_data import (
    balanced_counterfactual_results,
    build_phase2_additions,
    phase2_addition_sha256,
)


def verify_sha256(path: Path, expected: str) -> None:
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise SystemExit(f"hash mismatch for {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())

    source_path = Path(config["source_result"])
    weights_path = Path(config["source_weights"])
    dataset_config_path = Path(config["dataset_config"])
    verify_sha256(source_path, config["source_result_sha256"])
    verify_sha256(weights_path, config["source_weights_sha256"])
    verify_sha256(dataset_config_path, config["dataset_config_sha256"])
    source = json.loads(source_path.read_text())
    dataset_config = json.loads(dataset_config_path.read_text())
    examples = build_phase2_additions(**dataset_config["dataset"]["parameters"])
    if phase2_addition_sha256(examples) != source["dataset"]["sha256"]:
        raise SystemExit("dataset hash mismatch")

    selection = [example for example in examples if example.split == "selection"]
    development = [example for example in examples if example.split == "development"]
    selection_targets = balanced_counterfactual_results(selection)
    development_targets = balanced_counterfactual_results(development)
    target_hashes = {
        "selection": result_list_sha256(selection_targets),
        "development": result_list_sha256(development_targets),
    }
    if target_hashes != config["target_sha256"]:
        raise SystemExit("counterfactual target hash mismatch")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        source["model"]["id"],
        revision=source["model"]["revision"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        source["model"]["id"],
        revision=source["model"]["revision"],
        torch_dtype=getattr(torch, args.dtype),
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    capture = ActivationCapture(model, tokenizer, device=device)
    started = time.perf_counter()

    adapters = [load_online_adapter(str(weights_path), step=step) for step in range(3)]
    candidates = []
    for adapter in adapters:
        hidden_width = adapter.members[0].network[0].out_features
        candidates.append({(hidden_width, adapter.transport_rank): adapter})
    _, selections = select_adapters(
        model,
        tokenizer,
        capture,
        candidates,
        examples=selection,
        prompts=render_prompts(tokenizer, selection),
        targets=selection_targets,
        config={
            **config,
            "hidden_state_index": config["hidden_state_index"],
            "base_model_batch_size": config["base_model_batch_size"],
        },
        device=device,
    )
    for details in selections:
        details["selected"] = select_bounded_candidate(
            details["candidates"],
            max_relative_norm=config["selection_max_relative_norm"],
        )

    conditions = (
        "base",
        "adapter",
        "same_digit",
        "shuffled_target_norm_matched",
        "shuffled_state_norm_matched",
        "random_norm_matched",
    )
    development_prompts = render_prompts(tokenizer, development)
    condition_results = {
        condition: evaluate_condition(
            condition,
            model,
            tokenizer,
            capture,
            adapters,
            selections,
            examples=development,
            prompts=development_prompts,
            targets=development_targets,
            config={
                **config,
                "hidden_state_index": config["hidden_state_index"],
                "base_model_batch_size": config["base_model_batch_size"],
            },
            device=device,
            condition_index=index,
        )
        for index, condition in enumerate(conditions)
    }
    report: dict[str, Any] = {
        "schema_version": "oli.phase2-adapter-scale-sweep/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "development_only",
        "model": source["model"],
        "dataset": source["dataset"],
        "source": {
            "result": str(source_path),
            "result_sha256": config["source_result_sha256"],
            "weights": str(weights_path),
            "weights_sha256": config["source_weights_sha256"],
        },
        "target_assignment": {
            "scheme": "balanced_all_digits_changed",
            "sha256": target_hashes,
        },
        "scale_selection": {
            "grid": config["scales"],
            "max_relative_norm": config["selection_max_relative_norm"],
            "steps": selections,
        },
        "conditions": condition_results,
        "advancement_gate": source["advancement_gate"],
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Development-only fixed-weight scale diagnosis; audit remains sealed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
