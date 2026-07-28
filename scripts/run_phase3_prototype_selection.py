#!/usr/bin/env python3
"""Fit and select donor-free digit prototypes for frozen Phi-3.5."""

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
from run_phase3_native_boundary import (
    predict_with_delta,
    prefix_prompts,
    render_examples,
    value_list_sha256,
    verify_sha256,
)
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.evaluation import token_metrics
from open_latent_interfaces.phase2_data import balanced_counterfactual_results
from open_latent_interfaces.phase3_data import (
    build_phase3_additions,
    phase3_addition_sha256,
)
from open_latent_interfaces.prefill import verify_decimal_digit_contract


def fit_position_prototypes(
    capture: ActivationCapture,
    *,
    examples: list[Any],
    rendered: list[str],
    basis: torch.Tensor,
    position: int,
    hidden_state_index: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, str]]:
    natural_results = [example.result for example in examples]
    prompts = prefix_prompts(rendered, natural_results, position=position)
    states = capture.capture_last_token(
        prompts,
        hidden_state_indices=[hidden_state_index],
        batch_size=batch_size,
    )[hidden_state_index].values.float()
    coordinates = states @ basis.T
    prototypes = torch.empty((10, basis.shape[0]))
    counts = torch.zeros(10, dtype=torch.long)
    for digit in range(10):
        mask = torch.tensor(
            [int(str(result)[position]) == digit for result in natural_results]
        )
        if not bool(mask.any()):
            raise ValueError(f"missing fit examples for digit {digit} at {position}")
        prototypes[digit] = coordinates[mask].mean(dim=0)
        counts[digit] = int(mask.sum())
    hashes = {
        "states": hashlib.sha256(
            states.contiguous().numpy().tobytes()
        ).hexdigest(),
        "coordinates": hashlib.sha256(
            coordinates.contiguous().numpy().tobytes()
        ).hexdigest(),
    }
    return prototypes, counts, hashes


def prototype_delta(
    states: torch.Tensor,
    results: list[int],
    prototypes: torch.Tensor,
    basis: torch.Tensor,
    *,
    position: int,
) -> torch.Tensor:
    indices = torch.tensor([int(str(result)[position]) for result in results])
    desired = prototypes[indices]
    recipient = states.float() @ basis.T
    return (desired - recipient) @ basis


def scale_and_gate(
    delta: torch.Tensor,
    states: torch.Tensor,
    base_logits: torch.Tensor,
    requested_ids: torch.Tensor,
    *,
    scale: float,
    norm_cap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    scaled = delta * scale
    maximum = states.norm(dim=1) * norm_cap
    factor = (maximum / scaled.norm(dim=1).clamp_min(1e-12)).clamp(max=1.0)
    scaled = scaled * factor[:, None]
    gate = base_logits.argmax(dim=1) == requested_ids
    scaled[gate] = 0
    return scaled, gate


def capture_context(
    capture: ActivationCapture,
    tokenizer: Any,
    *,
    examples: list[Any],
    rendered: list[str],
    results: list[int],
    position: int,
    hidden_state_index: int,
    batch_size: int,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    prompts = prefix_prompts(rendered, results, position=position)
    states = capture.capture_last_token(
        prompts,
        hidden_state_indices=[hidden_state_index],
        batch_size=batch_size,
    )[hidden_state_index].values.float()
    base_logits = capture.next_token_logits(prompts, batch_size=batch_size)
    return prompts, states, base_logits


def evaluate(
    model: Any,
    tokenizer: Any,
    *,
    prompts: list[str],
    states: torch.Tensor,
    results: list[int],
    delta: torch.Tensor,
    position: int,
    hidden_state_index: int,
    digit_token_ids: dict[int, int],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    expected = torch.tensor(
        [digit_token_ids[int(str(result)[position])] for result in results]
    )
    logits = predict_with_delta(
        model,
        tokenizer,
        prompts,
        delta,
        hidden_state_index=hidden_state_index,
        batch_size=batch_size,
        device=device,
    )
    predicted = logits.argmax(dim=1).tolist()
    metrics = token_metrics(logits, expected)
    metrics["mean_relative_norm"] = float(
        (delta.norm(dim=1) / states.norm(dim=1)).mean()
    )
    metrics["digit_token_rate"] = sum(
        token_id in set(digit_token_ids.values()) for token_id in predicted
    ) / len(predicted)
    metrics["predicted_token_ids"] = predicted
    return metrics


def select_scale(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            min(row["target_accuracy"], row["identity_accuracy"]),
            row["target_accuracy"],
            row["identity_accuracy"],
            row["target_margin"],
            -row["target_relative_norm"],
            -row["scale"],
        ),
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
        raise SystemExit("refusing to overwrite prototype result or artifact")
    config = json.loads(args.config.read_text())
    paths = {
        "dataset": Path(config["dataset_config"]),
        "behavior": Path(config["behavior_result"]),
        "rank": Path(config["rank_result"]),
        "basis": Path(config["basis"]),
    }
    for name, path in paths.items():
        verify_sha256(path, config[f"{name}_sha256"])
    dataset_config = json.loads(paths["dataset"].read_text())
    behavior = json.loads(paths["behavior"].read_text())
    rank_result = json.loads(paths["rank"].read_text())
    if dataset_config.get("audit_authorized", False):
        raise SystemExit("prototype selection requires a sealed audit")
    if not behavior["passes"] or not rank_result["passes"]:
        raise SystemExit("behavior or rank gate did not pass")

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

    basis_artifact = load_file(str(paths["basis"]))
    leading_rank = rank_result["selection"]["selected_leading_rank"]
    suffix_rank = rank_result["selection"]["selected_suffix_rank"]
    bases = {
        0: basis_artifact["leading_basis"][:leading_rank].float(),
        1: basis_artifact["suffix_basis"][:suffix_rank].float(),
        2: basis_artifact["suffix_basis"][:suffix_rank].float(),
    }
    hidden_indices = {
        int(key): value for key, value in config["hidden_state_indices"].items()
    }
    if hidden_indices != {
        int(key): value
        for key, value in rank_result["basis"]["hidden_state_indices"].items()
    }:
        raise SystemExit("hidden-state indices differ from rank result")

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

    prototype_artifact = {}
    fit_hashes = {}
    prototypes = {}
    for position in range(3):
        prototype, counts, hashes = fit_position_prototypes(
            capture,
            examples=fit,
            rendered=rendered_fit,
            basis=bases[position],
            position=position,
            hidden_state_index=hidden_indices[position],
            batch_size=config["base_model_batch_size"],
        )
        prototypes[position] = prototype
        prototype_artifact[f"position_{position}_digit"] = prototype
        prototype_artifact[f"position_{position}_counts"] = counts
        fit_hashes[str(position)] = hashes
    args.prototype_output.parent.mkdir(parents=True, exist_ok=True)
    save_file(prototype_artifact, str(args.prototype_output))
    prototype_hash = hashlib.sha256(args.prototype_output.read_bytes()).hexdigest()

    positions = {}
    all_pass = True
    for position in range(3):
        target_context = capture_context(
            capture,
            tokenizer,
            examples=selection,
            rendered=rendered_selection,
            results=targets,
            position=position,
            hidden_state_index=hidden_indices[position],
            batch_size=config["base_model_batch_size"],
        )
        identity_context = capture_context(
            capture,
            tokenizer,
            examples=selection,
            rendered=rendered_selection,
            results=originals,
            position=position,
            hidden_state_index=hidden_indices[position],
            batch_size=config["base_model_batch_size"],
        )
        target_ids = torch.tensor(
            [digit_token_ids[int(str(value)[position])] for value in targets]
        )
        identity_ids = torch.tensor(
            [digit_token_ids[int(str(value)[position])] for value in originals]
        )
        target_prompts, target_states, target_base = target_context
        identity_prompts, identity_states, identity_base = identity_context
        raw_target = prototype_delta(
            target_states,
            targets,
            prototypes[position],
            bases[position],
            position=position,
        )
        raw_identity = prototype_delta(
            identity_states,
            originals,
            prototypes[position],
            bases[position],
            position=position,
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
                position=position,
                hidden_state_index=hidden_indices[position],
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
                position=position,
                hidden_state_index=hidden_indices[position],
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
        selected = select_scale(rows)
        selected_metrics = metrics_by_scale[str(selected["scale"])]
        position_passes = (
            selected_metrics["target"]["top1_exact"]
            >= config["selection_rule"]["minimum_target_accuracy"]
            and selected_metrics["identity"]["top1_exact"]
            >= config["selection_rule"]["minimum_identity_accuracy"]
            and selected_metrics["target"]["mean_relative_norm"]
            <= config["norm_cap"]
            and selected_metrics["target"]["digit_token_rate"] == 1.0
        )
        all_pass = all_pass and position_passes
        positions[str(position)] = {
            "hidden_state_index": hidden_indices[position],
            "rank": bases[position].shape[0],
            "selected_scale": selected["scale"],
            "passes": position_passes,
            "metrics_by_scale": metrics_by_scale,
        }

    report = {
        "schema_version": "oli.phase3-prototype-selection/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "selection_only",
        "model": model_config,
        "dataset": {
            "sha256": observed_hash,
            "eligible_fit_examples": len(fit),
            "selection_examples": len(selection),
        },
        "sources": {
            "behavior_sha256": config["behavior_sha256"],
            "rank_sha256": config["rank_sha256"],
            "basis_sha256": config["basis_sha256"],
        },
        "fit": {
            "method": "digit_class_mean_native_coordinates",
            "state_coordinate_sha256": fit_hashes,
        },
        "prototypes": {
            "path": str(args.prototype_output),
            "sha256": prototype_hash,
            "tensors": {
                key: list(value.shape)
                for key, value in prototype_artifact.items()
            },
        },
        "selection_targets_sha256": config["selection_targets_sha256"],
        "scales": config["scales"],
        "norm_cap": config["norm_cap"],
        "selection_rule": config["selection_rule"],
        "positions": positions,
        "passes": all_pass,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Selection-only donor-free next-digit prototype evaluation. "
            "Closed-loop development and matched controls remain untested."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
