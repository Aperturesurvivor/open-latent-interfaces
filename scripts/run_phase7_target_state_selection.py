#!/usr/bin/env python3
"""Fit and select a Qwen target-tens state-overwrite interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from run_phase4_carry_sequence_boundary import (
    generate_chunks,
    norm_match_sequences,
    random_norm_matched_sequences,
    sequence_norms,
    summarize_outputs,
    value_sha256,
    verify_sha256,
)
from run_phase4_donor_free_prototypes import one_token_sequences
from run_phase6_conditional_carry_selection import (
    exact_fit_quartets,
    extract_at,
    render_rows,
    token_positions,
)
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.phase7_data import (
    build_phase7_carry_quartets,
    phase7_carry_sha256,
)
from open_latent_interfaces.prefill import verify_decimal_digit_contract
from open_latent_interfaces.typed_writer import fit_digit_subspace


def result_tens(rows: list[Any]) -> torch.Tensor:
    return torch.tensor([(row.result // 10) % 10 for row in rows])


def rotate_digits(digits: torch.Tensor, classes: tuple[int, ...]) -> torch.Tensor:
    class_index = {digit: index for index, digit in enumerate(classes)}
    return torch.tensor(
        [
            classes[(class_index[int(digit)] + 1) % len(classes)]
            for digit in digits.tolist()
        ]
    )


def select_fit_rank(
    singular_values: torch.Tensor,
    *,
    minimum_explained_variance: float,
    maximum_rank: int,
) -> tuple[int, list[float]]:
    energy = singular_values.square()
    ratios = (energy.cumsum(dim=0) / energy.sum()).tolist()
    selected = next(
        (
            index + 1
            for index, ratio in enumerate(ratios[:maximum_rank])
            if ratio >= minimum_explained_variance
        ),
        maximum_rank,
    )
    return selected, ratios


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()
    if args.output.exists() or args.artifact_output.exists():
        raise SystemExit("refusing to overwrite selection result or artifact")

    config = json.loads(args.config.read_text())
    dataset_path = Path(config["dataset_config"])
    behavior_path = Path(config["behavior_result"])
    verify_sha256(dataset_path, config["dataset_config_sha256"])
    verify_sha256(behavior_path, config["behavior_result_sha256"])
    dataset_config = json.loads(dataset_path.read_text())
    behavior = json.loads(behavior_path.read_text())
    if dataset_config.get("audit_authorized", False):
        raise SystemExit("selection requires a sealed audit")
    for split in ("selection", "development"):
        if not behavior["splits"][split]["passes"]:
            raise SystemExit(f"untouched {split} behavior gate did not pass")
    eligible_ids = exact_fit_quartets(behavior)
    if len(eligible_ids) < config["minimum_eligible_fit_quartets"]:
        raise SystemExit("insufficient behavior-exact fit quartets")
    if value_sha256(eligible_ids) != config["eligible_fit_quartets_sha256"]:
        raise SystemExit("eligible fit quartet hash mismatch")

    examples = build_phase7_carry_quartets(
        **dataset_config["dataset"]["parameters"]
    )
    observed_dataset_hash = phase7_carry_sha256(examples)
    if observed_dataset_hash != dataset_config["dataset"]["sha256"]:
        raise SystemExit("Phase 7 dataset hash mismatch")
    split_ids = {
        "fit": eligible_ids,
        "selection": sorted(
            {
                row.quartet_id
                for row in examples
                if row.split == "selection"
            }
        ),
    }
    if value_sha256(split_ids["selection"]) != config[
        "selection_quartets_sha256"
    ]:
        raise SystemExit("selection quartet hash mismatch")
    by_quartet = {
        quartet_id: {
            row.variant: row
            for row in examples
            if row.quartet_id == quartet_id
        }
        for quartet_id in split_ids["fit"] + split_ids["selection"]
    }
    variants = (
        "carry_base",
        "carry_increment",
        "control_base",
        "control_increment",
    )
    rows = {
        split: {
            variant: [by_quartet[quartet_id][variant] for quartet_id in ids]
            for variant in variants
        }
        for split, ids in split_ids.items()
    }

    model_config = dataset_config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rendered = {
        split: render_rows(
            tokenizer,
            split_rows,
            dataset_config["assistant_prefix"],
        )
        for split, split_rows in rows.items()
    }
    positions = {}
    for split in ("fit", "selection"):
        _, context, contract = token_positions(
            tokenizer,
            rendered[split],
            label=f"Phase 7 {split}",
        )
        positions[split] = context
        if split == "selection" and value_sha256(contract) != config[
            "token_region_contract_sha256"
        ]:
            raise SystemExit("selection token-region contract mismatch")
    digit_token_ids = verify_decimal_digit_contract(
        tokenizer,
        rendered["selection"]["carry_base"][0],
    )

    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        torch_dtype=getattr(torch, args.dtype),
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    capture = ActivationCapture(model, tokenizer, device=device)
    hidden_index = config["carry_context_hidden_state_index"]
    started = time.perf_counter()
    states = {
        split: {
            variant: capture.capture_sequences(
                prompts,
                hidden_state_indices=[hidden_index],
                batch_size=config["base_model_batch_size"],
            )[hidden_index].values
            for variant, prompts in split_rendered.items()
        }
        for split, split_rendered in rendered.items()
    }

    fit_targets = extract_at(
        states["fit"]["carry_increment"],
        positions["fit"],
    )
    fit_digits = result_tens(rows["fit"]["carry_increment"])
    counts = Counter(fit_digits.tolist())
    if set(counts) != set(range(10)) or min(counts.values()) < config[
        "minimum_fit_examples_per_target_digit"
    ]:
        raise SystemExit(f"insufficient target-digit support: {counts}")
    subspace = fit_digit_subspace(fit_targets, fit_digits)
    centered = subspace.centroids - subspace.centroids.mean(dim=0)
    singular_values = torch.linalg.svdvals(centered)
    coordinate_rank, explained_variance = select_fit_rank(
        singular_values,
        minimum_explained_variance=config[
            "minimum_between_centroid_explained_variance"
        ],
        maximum_rank=config["maximum_coordinate_rank"],
    )

    selection_base = states["selection"]["carry_base"]
    selection_states = extract_at(selection_base, positions["selection"])
    target_digits = result_tens(rows["selection"]["carry_increment"])
    identity_digits = result_tens(rows["selection"]["carry_base"])
    wrong_digits = rotate_digits(target_digits, subspace.classes)
    metrics = {}
    passing_scales = []
    rule = config["selection_rule"]
    for scale in config["scales"]:
        target_vectors = subspace.write_delta(
            selection_states,
            target_digits,
            rank=coordinate_rank,
            scale=scale,
        )
        identity_vectors = subspace.write_delta(
            selection_states,
            identity_digits,
            rank=coordinate_rank,
            scale=scale,
        )
        wrong_vectors = subspace.write_delta(
            selection_states,
            wrong_digits,
            rank=coordinate_rank,
            scale=scale,
        )
        target = one_token_sequences(
            selection_base,
            positions["selection"],
            target_vectors,
        )
        target_norms = sequence_norms(target)
        conditions = {
            "target": target,
            "identity_digit_norm_matched": norm_match_sequences(
                one_token_sequences(
                    selection_base,
                    positions["selection"],
                    identity_vectors,
                ),
                target_norms,
            ),
            "wrong_digit_norm_matched": norm_match_sequences(
                one_token_sequences(
                    selection_base,
                    positions["selection"],
                    wrong_vectors,
                ),
                target_norms,
            ),
            "random_norm_matched": random_norm_matched_sequences(
                target,
                target_norms,
                seed=config["random_control_seed"],
            ),
        }
        scale_metrics = {}
        for name, delta in conditions.items():
            responses = generate_chunks(
                model,
                tokenizer,
                rendered["selection"]["carry_base"],
                delta,
                hidden_state_index=hidden_index,
                batch_size=config["base_model_batch_size"],
                device=device,
            )
            scale_metrics[name] = summarize_outputs(
                responses,
                rows["selection"]["carry_base"],
                rows["selection"]["carry_increment"],
                delta,
                selection_base,
            )
        controls = tuple(name for name in conditions if name != "target")
        strongest = max(
            controls,
            key=lambda name: scale_metrics[name]["target_tens_accuracy"],
        )
        target_accuracy = scale_metrics["target"]["target_tens_accuracy"]
        control_accuracy = scale_metrics[strongest]["target_tens_accuracy"]
        passes = (
            target_accuracy >= rule["minimum_tens_accuracy"]
            and target_accuracy - control_accuracy
            >= rule["minimum_control_advantage"]
            and (
                not rule["require_parse_rate"]
                or scale_metrics["target"]["parse_rate"] == 1.0
            )
        )
        scale_metrics["gate"] = {
            "target_tens_accuracy": target_accuracy,
            "strongest_control": strongest,
            "strongest_control_tens_accuracy": control_accuracy,
            "control_advantage": target_accuracy - control_accuracy,
            "passes": passes,
        }
        if passes:
            passing_scales.append(float(scale))
        metrics[str(scale)] = scale_metrics
    selected_scale = min(passing_scales) if passing_scales else max(
        config["scales"],
        key=lambda scale: (
            metrics[str(scale)]["target"]["target_tens_accuracy"],
            metrics[str(scale)]["gate"]["control_advantage"],
            -float(scale),
        ),
    )
    passes = bool(passing_scales)

    args.artifact_output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "target_digits": torch.tensor(
                subspace.classes,
                dtype=torch.int64,
            ),
            "target_centroids": subspace.centroids.contiguous(),
            "coordinate_basis": subspace.basis[:coordinate_rank].contiguous(),
            "fit_class_counts": torch.tensor(
                [counts[digit] for digit in subspace.classes],
                dtype=torch.int64,
            ),
        },
        str(args.artifact_output),
        metadata={
            "schema_version": "oli.target-state-overwrite-tensors/v1",
            "model_id": model_config["id"],
            "model_revision": model_config["revision"],
        },
    )
    artifact_hash = hashlib.sha256(args.artifact_output.read_bytes()).hexdigest()
    report = {
        "schema_version": "oli.phase7-qwen-target-state-selection/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "selection_only",
        "model": model_config,
        "dataset_sha256": observed_dataset_hash,
        "behavior_result_sha256": config["behavior_result_sha256"],
        "behavior_gate": {
            "overall_passed": behavior["passes"],
            "fit_passed": behavior["splits"]["fit"]["passes"],
            "selection_passed": behavior["splits"]["selection"]["passes"],
            "development_passed": behavior["splits"]["development"]["passes"],
        },
        "eligible_fit_quartets": len(eligible_ids),
        "eligible_fit_quartets_sha256": config[
            "eligible_fit_quartets_sha256"
        ],
        "fit_target_digit_counts": {
            str(digit): counts[digit] for digit in subspace.classes
        },
        "selection_quartets_sha256": config["selection_quartets_sha256"],
        "token_region_contract_sha256": config[
            "token_region_contract_sha256"
        ],
        "digit_token_ids": digit_token_ids,
        "hidden_state_index": hidden_index,
        "fit_only_rank_selection": {
            "minimum_explained_variance": config[
                "minimum_between_centroid_explained_variance"
            ],
            "maximum_rank": config["maximum_coordinate_rank"],
            "selected_rank": coordinate_rank,
            "singular_values": singular_values.tolist(),
            "cumulative_explained_variance": explained_variance,
        },
        "metrics": metrics,
        "selection": {
            "coordinate_rank": coordinate_rank,
            "scale": selected_scale,
            "passes": passes,
        },
        "passes": passes,
        "artifact": {
            "path": str(args.artifact_output),
            "sha256": artifact_hash,
            "width": int(fit_targets.shape[1]),
            "classes": list(subspace.classes),
        },
        "selection_rule": rule,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Fit-only target-state rank and selection-only causal scale for "
            "a deterministic-target-conditioned overwrite. No development or "
            "audit claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    print(f"wrote {args.artifact_output}")


if __name__ == "__main__":
    main()
