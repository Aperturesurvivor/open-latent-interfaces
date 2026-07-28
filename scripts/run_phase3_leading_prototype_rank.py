#!/usr/bin/env python3
"""Select a donor-free leading prototype rank after the rank-8 non-pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from run_phase3_native_boundary import (
    prefix_prompts,
    render_examples,
    value_list_sha256,
    verify_sha256,
)
from run_phase3_prototype_selection import (
    capture_context,
    evaluate,
    prototype_delta,
    scale_and_gate,
    select_scale,
)
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.phase2_data import balanced_counterfactual_results
from open_latent_interfaces.phase3_data import (
    build_phase3_additions,
    phase3_addition_sha256,
)
from open_latent_interfaces.prefill import verify_decimal_digit_contract


def leading_prototypes(
    states: torch.Tensor,
    results: list[int],
    basis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coordinates = states.float() @ basis.T
    prototypes = torch.zeros((10, basis.shape[0]))
    counts = torch.zeros(10, dtype=torch.long)
    for digit in range(1, 10):
        mask = torch.tensor([int(str(result)[0]) == digit for result in results])
        prototypes[digit] = coordinates[mask].mean(dim=0)
        counts[digit] = int(mask.sum())
    return prototypes, counts, coordinates


def candidate_passes(row: dict[str, float], config: dict[str, object]) -> bool:
    rule = config["selection_rule"]
    assert isinstance(rule, dict)
    return (
        row["target_accuracy"] + 1e-7 >= rule["minimum_target_accuracy"]
        and row["identity_accuracy"] + 1e-7
        >= rule["minimum_identity_accuracy"]
        and row["target_relative_norm"] <= config["norm_cap"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prototype-output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()

    if args.output.exists() or args.prototype_output.exists():
        raise SystemExit("refusing to overwrite leading-rank result or artifact")
    config = json.loads(args.config.read_text())
    source_paths = {
        "dataset": Path(config["dataset_config"]),
        "behavior": Path(config["behavior_result"]),
        "rank": Path(config["rank_result"]),
        "basis": Path(config["basis"]),
        "prototype": Path(config["prototype_result"]),
    }
    for name, path in source_paths.items():
        verify_sha256(path, config[f"{name}_sha256"])
    dataset_config = json.loads(source_paths["dataset"].read_text())
    behavior = json.loads(source_paths["behavior"].read_text())
    rank_result = json.loads(source_paths["rank"].read_text())
    prototype_result = json.loads(source_paths["prototype"].read_text())
    if dataset_config.get("audit_authorized", False):
        raise SystemExit("leading-rank selection requires a sealed audit")
    if prototype_result["passes"]:
        raise SystemExit("follow-up requires the frozen prototype non-pass")
    if not all(
        prototype_result["positions"][str(position)]["passes"]
        for position in (1, 2)
    ):
        raise SystemExit("suffix interface was not locked by the source result")

    examples = build_phase3_additions(**dataset_config["dataset"]["parameters"])
    observed_hash = phase3_addition_sha256(examples)
    if observed_hash != dataset_config["dataset"]["sha256"]:
        raise SystemExit("Phase 3 dataset hash mismatch")
    exact_fit_ids = {
        row["example_id"]
        for row in behavior["rows"]
        if row["split"] == "fit" and row["exact"]
    }
    fit = [
        example
        for example in examples
        if example.split == "fit" and example.example_id in exact_fit_ids
    ]
    selection = [example for example in examples if example.split == "selection"]
    targets = balanced_counterfactual_results(selection)
    if value_list_sha256(targets) != config["selection_targets_sha256"]:
        raise SystemExit("selection target hash mismatch")
    originals = [example.result for example in selection]
    fit_results = [example.result for example in fit]

    basis_artifact = load_file(str(source_paths["basis"]))
    full_basis = basis_artifact["leading_basis"].float()
    if max(config["ranks"]) > full_basis.shape[0]:
        raise SystemExit("requested leading rank exceeds fitted basis")
    hidden_index = config["hidden_state_index"]
    if (
        rank_result["basis"]["hidden_state_indices"]["0"] != hidden_index
        or prototype_result["positions"]["0"]["hidden_state_index"] != hidden_index
    ):
        raise SystemExit("leading hidden-state index differs from sources")

    device = torch.device(args.device)
    model_config = dataset_config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rendered_fit = render_examples(
        tokenizer,
        fit,
        assistant_prefix=dataset_config["assistant_prefix"],
    )
    rendered_selection = render_examples(
        tokenizer,
        selection,
        assistant_prefix=dataset_config["assistant_prefix"],
    )
    digit_token_ids = verify_decimal_digit_contract(tokenizer, rendered_fit[0])
    model = AutoModelForCausalLM.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        torch_dtype=getattr(torch, args.dtype),
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    capture = ActivationCapture(model, tokenizer, device=device)
    started = time.perf_counter()

    fit_prompts = prefix_prompts(rendered_fit, fit_results, position=0)
    fit_states = capture.capture_last_token(
        fit_prompts,
        hidden_state_indices=[hidden_index],
        batch_size=config["base_model_batch_size"],
    )[hidden_index].values.float()
    target_prompts, target_states, target_base = capture_context(
        capture,
        tokenizer,
        examples=selection,
        rendered=rendered_selection,
        results=targets,
        position=0,
        hidden_state_index=hidden_index,
        batch_size=config["base_model_batch_size"],
    )
    identity_prompts, identity_states, identity_base = capture_context(
        capture,
        tokenizer,
        examples=selection,
        rendered=rendered_selection,
        results=originals,
        position=0,
        hidden_state_index=hidden_index,
        batch_size=config["base_model_batch_size"],
    )
    target_ids = torch.tensor(
        [digit_token_ids[int(str(value)[0])] for value in targets]
    )
    identity_ids = torch.tensor(
        [digit_token_ids[int(str(value)[0])] for value in originals]
    )

    rank_results = {}
    artifacts = {}
    passing_ranks = []
    for rank in config["ranks"]:
        basis = full_basis[:rank]
        prototypes, counts, coordinates = leading_prototypes(
            fit_states,
            fit_results,
            basis,
        )
        artifacts[rank] = {
            "prototypes": prototypes,
            "counts": counts,
            "coordinates_sha256": hashlib.sha256(
                coordinates.contiguous().numpy().tobytes()
            ).hexdigest(),
        }
        raw_target = prototype_delta(
            target_states,
            targets,
            prototypes,
            basis,
            position=0,
        )
        raw_identity = prototype_delta(
            identity_states,
            originals,
            prototypes,
            basis,
            position=0,
        )
        rows = []
        metrics_by_scale = {}
        for scale in config["scales"]:
            target_delta, target_gate = scale_and_gate(
                raw_target,
                target_states,
                target_base,
                target_ids,
                scale=scale,
                norm_cap=config["norm_cap"],
            )
            identity_delta, identity_gate = scale_and_gate(
                raw_identity,
                identity_states,
                identity_base,
                identity_ids,
                scale=scale,
                norm_cap=config["norm_cap"],
            )
            target_metrics = evaluate(
                model,
                tokenizer,
                prompts=target_prompts,
                states=target_states,
                results=targets,
                delta=target_delta,
                position=0,
                hidden_state_index=hidden_index,
                digit_token_ids=digit_token_ids,
                batch_size=config["base_model_batch_size"],
                device=device,
            )
            identity_metrics = evaluate(
                model,
                tokenizer,
                prompts=identity_prompts,
                states=identity_states,
                results=originals,
                delta=identity_delta,
                position=0,
                hidden_state_index=hidden_index,
                digit_token_ids=digit_token_ids,
                batch_size=config["base_model_batch_size"],
                device=device,
            )
            target_metrics["hard_gate_rate"] = float(target_gate.float().mean())
            identity_metrics["hard_gate_rate"] = float(identity_gate.float().mean())
            metrics_by_scale[str(scale)] = {
                "target": target_metrics,
                "identity": identity_metrics,
            }
            rows.append(
                {
                    "scale": scale,
                    "target_accuracy": target_metrics["top1_exact"],
                    "identity_accuracy": identity_metrics["top1_exact"],
                    "target_margin": target_metrics["mean_target_margin"],
                    "target_relative_norm": target_metrics["mean_relative_norm"],
                }
            )
        selected_scale = select_scale(rows)
        passes = candidate_passes(selected_scale, config)
        if passes:
            passing_ranks.append(rank)
        rank_results[str(rank)] = {
            "selected_scale": selected_scale["scale"],
            "passes": passes,
            "metrics_by_scale": metrics_by_scale,
        }

    selected_rank = min(passing_ranks) if passing_ranks else max(
        config["ranks"],
        key=lambda rank: (
            rank_results[str(rank)]["metrics_by_scale"][
                str(rank_results[str(rank)]["selected_scale"])
            ]["target"]["top1_exact"],
            -rank,
        ),
    )
    selected_scale = rank_results[str(selected_rank)]["selected_scale"]
    passes = bool(passing_ranks)
    selected_artifact = artifacts[selected_rank]
    args.prototype_output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "leading_digit": selected_artifact["prototypes"],
            "leading_counts": selected_artifact["counts"],
        },
        str(args.prototype_output),
    )
    prototype_hash = hashlib.sha256(args.prototype_output.read_bytes()).hexdigest()
    report = {
        "schema_version": "oli.phase3-leading-prototype-rank/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "selection_only_followup",
        "model": model_config,
        "dataset_sha256": observed_hash,
        "sources": {
            f"{name}_sha256": config[f"{name}_sha256"]
            for name in ("behavior", "rank", "basis", "prototype")
        },
        "fit": {
            "examples": len(fit),
            "states_sha256": hashlib.sha256(
                fit_states.contiguous().numpy().tobytes()
            ).hexdigest(),
            "coordinate_sha256_by_rank": {
                str(rank): artifact["coordinates_sha256"]
                for rank, artifact in artifacts.items()
            },
        },
        "hidden_state_index": hidden_index,
        "ranks": config["ranks"],
        "scales": config["scales"],
        "norm_cap": config["norm_cap"],
        "selection_rule": config["selection_rule"],
        "selection_targets_sha256": config["selection_targets_sha256"],
        "rank_results": rank_results,
        "selected_rank": selected_rank,
        "selected_scale": selected_scale,
        "passes": passes,
        "prototype": {
            "path": str(args.prototype_output),
            "sha256": prototype_hash,
            "shape": list(selected_artifact["prototypes"].shape),
        },
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Bounded selection-only leading-rank follow-up after a frozen "
            "rank-8 prototype non-pass."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
