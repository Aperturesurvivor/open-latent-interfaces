#!/usr/bin/env python3
"""Causally fine-tune Phase 2 adapters through the frozen downstream model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from run_phase1_conditional_transport_bridge import render_prompts
from run_phase2_scaled_adapter import evaluate_condition, target_results
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture, last_nonpadding_positions
from open_latent_interfaces.adapter import load_online_adapter
from open_latent_interfaces.interventions import online_adapter_intervention
from open_latent_interfaces.phase2_data import (
    build_phase2_additions,
    phase2_addition_sha256,
)


def encode(tokenizer: Any, prompts: list[str], device: torch.device) -> dict[str, Any]:
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in encoded.items()
    }


def next_logits(model: Any, encoded: dict[str, Any]) -> torch.Tensor:
    outputs = model(**encoded, use_cache=False, return_dict=True)
    positions = last_nonpadding_positions(encoded["attention_mask"])
    rows = torch.arange(positions.shape[0], device=positions.device)
    return outputs.logits[rows, positions]


def expected_ids(
    tokenizer: Any,
    values: list[int],
    step: int,
    device: torch.device,
) -> torch.Tensor:
    rows = [tokenizer(str(value), add_special_tokens=False)["input_ids"] for value in values]
    if any(len(row) != 3 for row in rows):
        raise ValueError("results must tokenize to three digits")
    return torch.tensor([row[step] for row in rows], device=device)


def train_step_adapter(
    step: int,
    adapter: Any,
    model: Any,
    tokenizer: Any,
    *,
    prompts: list[str],
    original_results: list[int],
    targets: list[int],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[list[Any], list[dict[str, float]]]:
    adapter = adapter.to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=config["adapter_learning_rate"],
        weight_decay=1e-4,
    )
    generator = torch.Generator().manual_seed(config["training_seed"] + step)
    checkpoints = []
    history = []
    for epoch in range(config["causal_epochs"]):
        permutation = torch.randperm(len(prompts), generator=generator).tolist()
        totals = {"loss": 0.0, "target_ce": 0.0, "identity_ce": 0.0, "kl": 0.0}
        seen = 0
        adapter.train()
        for start in range(0, len(prompts), config["base_model_batch_size"]):
            indices = permutation[start : start + config["base_model_batch_size"]]
            batch_prompts = [prompts[index] for index in indices]
            batch_targets = [targets[index] for index in indices]
            batch_originals = [original_results[index] for index in indices]
            target_prompts = [
                prompt + str(value)[:step]
                for prompt, value in zip(batch_prompts, batch_targets, strict=True)
            ]
            identity_prompts = [
                prompt + str(value)[:step]
                for prompt, value in zip(batch_prompts, batch_originals, strict=True)
            ]
            target_encoded = encode(tokenizer, target_prompts, device)
            identity_encoded = encode(tokenizer, identity_prompts, device)
            target_digits = torch.tensor(
                [int(str(value)[step]) for value in batch_targets],
                device=device,
            )
            identity_digits = torch.tensor(
                [int(str(value)[step]) for value in batch_originals],
                device=device,
            )
            with online_adapter_intervention(
                model,
                hidden_state_index=23,
                adapter=adapter,
                target_digits=target_digits,
                attention_mask=target_encoded["attention_mask"],
                scale=1.0,
                norm_cap=config["norm_cap"],
            ) as target_hook:
                target_logits = next_logits(model, target_encoded)
            with torch.no_grad():
                base_identity_logits = next_logits(model, identity_encoded)
            with online_adapter_intervention(
                model,
                hidden_state_index=23,
                adapter=adapter,
                target_digits=identity_digits,
                attention_mask=identity_encoded["attention_mask"],
                scale=1.0,
                norm_cap=config["norm_cap"],
            ):
                identity_logits = next_logits(model, identity_encoded)
            target_ce = functional.cross_entropy(
                target_logits,
                expected_ids(tokenizer, batch_targets, step, device),
            )
            identity_ce = functional.cross_entropy(
                identity_logits,
                expected_ids(tokenizer, batch_originals, step, device),
            )
            kl = functional.kl_div(
                functional.log_softmax(identity_logits, dim=-1),
                functional.softmax(base_identity_logits, dim=-1),
                reduction="batchmean",
            )
            assert target_hook.applied_delta is not None
            assert target_hook.recipient_states is not None
            relative_norm = (
                target_hook.applied_delta.norm(dim=1)
                / target_hook.recipient_states.float().norm(dim=1)
            ).mean()
            loss = target_ce
            loss = loss + config["identity_ce_weight"] * identity_ce
            loss = loss + config["identity_kl_weight"] * kl
            loss = loss + config["norm_loss_weight"] * relative_norm
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            count = len(indices)
            for name, value in (
                ("loss", loss),
                ("target_ce", target_ce),
                ("identity_ce", identity_ce),
                ("kl", kl),
            ):
                totals[name] += float(value.detach()) * count
            seen += count
        history.append(
            {"epoch": epoch + 1, **{name: value / seen for name, value in totals.items()}}
        )
        checkpoints.append(copy.deepcopy(adapter).cpu())
        adapter = adapter.to(device)
    return checkpoints, history


def select_checkpoint(
    step: int,
    checkpoints: list[Any],
    model: Any,
    tokenizer: Any,
    capture: ActivationCapture,
    *,
    examples: list[Any],
    prompts: list[str],
    targets: list[int],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[Any, dict[str, Any]]:
    from run_phase2_scaled_adapter import select_adapters

    candidate_map = [
        {(0, epoch + 1): checkpoint}
        for epoch, checkpoint in enumerate(checkpoints)
    ]
    rows = []
    adapters = []
    for epoch, candidate in enumerate(candidate_map, start=1):
        selected, details = select_adapters(
            model,
            tokenizer,
            capture,
            [candidate],
            examples=examples,
            prompts=prompts,
            targets=targets,
            config={
                **config,
                "hidden_state_index": 23,
                "scales": config["scales"],
                "base_model_batch_size": config["base_model_batch_size"],
            },
            device=device,
        )
        row = details[0]["selected"]
        row["epoch"] = epoch
        rows.append(row)
        adapters.append(selected[0])
    best = max(
        range(len(rows)),
        key=lambda index: (
            rows[index]["minimum_accuracy"],
            rows[index]["target_token_accuracy"],
            rows[index]["identity_token_accuracy"],
            -rows[index]["mean_target_relative_norm"],
        ),
    )
    return adapters[best], {"step": step, "selected": rows[best], "candidates": rows}


def save_online_adapters(path: Path, adapters: list[Any]) -> str:
    tensors = {}
    for step, adapter in enumerate(adapters):
        for name in (
            "state_mean",
            "state_basis",
            "state_scale",
            "delta_basis",
            "coefficient_scale",
        ):
            tensors[f"step{step}.projection.{name}"] = getattr(
                adapter,
                name,
            ).contiguous().cpu()
        for member_index, member in enumerate(adapter.members):
            for name, tensor in member.state_dict().items():
                tensors[
                    f"step{step}.member{member_index}.model.{name}"
                ] = tensor.contiguous().cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights-output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for path_key, hash_key in (
        ("initial_result", "initial_result_sha256"),
        ("initial_weights", "initial_weights_sha256"),
    ):
        if hashlib.sha256(Path(config[path_key]).read_bytes()).hexdigest() != config[hash_key]:
            raise SystemExit(f"{path_key} hash mismatch")
    previous = json.loads(Path(config["initial_result"]).read_text())
    dataset_config = json.loads(Path("configs/phase2_dataset_frozen.json").read_text())
    examples = build_phase2_additions(**dataset_config["dataset"]["parameters"])
    if phase2_addition_sha256(examples) != previous["dataset"]["sha256"]:
        raise SystemExit("dataset hash mismatch")
    fit = [example for example in examples if example.split == "fit"]
    selection = [example for example in examples if example.split == "selection"]
    development = [example for example in examples if example.split == "development"]
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        previous["model"]["id"], revision=previous["model"]["revision"]
    )
    model = AutoModelForCausalLM.from_pretrained(
        previous["model"]["id"],
        revision=previous["model"]["revision"],
        torch_dtype=getattr(torch, args.dtype),
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    capture = ActivationCapture(model, tokenizer, device=device)
    started = time.perf_counter()
    fit_prompts = render_prompts(tokenizer, fit)
    fit_originals = [example.result for example in fit]
    fit_targets = target_results(fit)
    selected_adapters = []
    histories = []
    selections = []
    selection_prompts = render_prompts(tokenizer, selection)
    selection_targets = target_results(selection)
    for step in range(3):
        initial = load_online_adapter(config["initial_weights"], step=step)
        checkpoints, history = train_step_adapter(
            step,
            initial,
            model,
            tokenizer,
            prompts=fit_prompts,
            original_results=fit_originals,
            targets=fit_targets,
            config=config,
            device=device,
        )
        selected, details = select_checkpoint(
            step,
            checkpoints,
            model,
            tokenizer,
            capture,
            examples=selection,
            prompts=selection_prompts,
            targets=selection_targets,
            config=config,
            device=device,
        )
        selected_adapters.append(selected)
        histories.append({"step": step, "epochs": history})
        selections.append(details)
    weights_hash = save_online_adapters(args.weights_output, selected_adapters)
    development_prompts = render_prompts(tokenizer, development)
    development_targets = target_results(development)
    conditions = (
        "base",
        "adapter",
        "same_digit",
        "shuffled_target_norm_matched",
        "shuffled_state_norm_matched",
        "random_norm_matched",
    )
    condition_results = {
        condition: evaluate_condition(
            condition,
            model,
            tokenizer,
            capture,
            selected_adapters,
            selections,
            examples=development,
            prompts=development_prompts,
            targets=development_targets,
            config={
                **config,
                "hidden_state_index": 23,
                "base_model_batch_size": config["base_model_batch_size"],
            },
            device=device,
            condition_index=index,
        )
        for index, condition in enumerate(conditions)
    }
    report = {
        "schema_version": "oli.phase2-causal-adapter/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "development_only",
        "model": previous["model"],
        "dataset": previous["dataset"],
        "training": histories,
        "selection": selections,
        "weights_path": str(args.weights_output),
        "weights_sha256": weights_hash,
        "conditions": condition_results,
        "advancement_gate": previous["advancement_gate"],
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": "Development must pass every frozen gate before audit.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
