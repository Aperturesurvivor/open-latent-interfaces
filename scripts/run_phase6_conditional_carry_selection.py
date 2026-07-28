#!/usr/bin/env python3
"""Fit and select a recipient-conditioned Qwen carry writer."""

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
from run_phase4_carry_sequence_boundary import (
    generate_chunks,
    norm_match_sequences,
    random_norm_matched_sequences,
    sequence_norms,
    summarize_outputs,
    value_sha256,
    verify_sha256,
)
from run_phase4_donor_free_prototypes import (
    differing_position,
    one_token_sequences,
    rotate_classes,
)
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.phase6_data import (
    Phase6CarryExample,
    build_phase6_carry_quartets,
    phase6_carry_sha256,
)
from open_latent_interfaces.prefill import (
    render_prefilled_chat,
    verify_decimal_digit_contract,
)
from open_latent_interfaces.typed_writer import (
    ConditionalTransportDesign,
    ConditionalTransportModel,
    build_conditional_transport_design,
)


def exact_fit_quartets(behavior: dict[str, Any]) -> list[str]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in behavior["rows"]:
        if row["split"] == "fit":
            grouped.setdefault(row["quartet_id"], []).append(row)
    return sorted(
        quartet_id
        for quartet_id, rows in grouped.items()
        if len(rows) == 4 and all(row["exact"] for row in rows)
    )


def render_rows(
    tokenizer: Any,
    rows: dict[str, list[Phase6CarryExample]],
    assistant_prefix: str,
) -> dict[str, list[str]]:
    return {
        variant: [
            render_prefilled_chat(
                tokenizer,
                row.prompt,
                assistant_prefix=assistant_prefix,
            )
            for row in variant_rows
        ]
        for variant, variant_rows in rows.items()
    }


def token_positions(
    tokenizer: Any,
    rendered: dict[str, list[str]],
    *,
    label: str,
) -> tuple[list[int], list[int], list[list[int]]]:
    token_ids = {
        variant: [tokenizer(prompt)["input_ids"] for prompt in prompts]
        for variant, prompts in rendered.items()
    }
    changed_positions = []
    context_positions = []
    contract = []
    for index in range(len(rendered["carry_base"])):
        changed = differing_position(
            token_ids["carry_base"][index],
            token_ids["carry_increment"][index],
            label=f"{label} operand",
        )
        if changed != differing_position(
            token_ids["control_base"][index],
            token_ids["control_increment"][index],
            label=f"{label} control operand",
        ):
            raise SystemExit(f"{label} operand positions differ")
        context = differing_position(
            token_ids["carry_base"][index],
            token_ids["control_base"][index],
            label=f"{label} context",
        )
        if context != differing_position(
            token_ids["carry_increment"][index],
            token_ids["control_increment"][index],
            label=f"{label} increment context",
        ):
            raise SystemExit(f"{label} context positions differ")
        changed_positions.append(changed)
        context_positions.append(context)
        contract.append(
            [len(token_ids["carry_base"][index]), changed, context]
        )
    return changed_positions, context_positions, contract


def extract_at(
    sequences: tuple[torch.Tensor, ...],
    positions: list[int],
) -> torch.Tensor:
    return torch.stack(
        [
            sequence[position].float()
            for sequence, position in zip(sequences, positions, strict=True)
        ]
    )


def build_design(
    states: torch.Tensor,
    deltas: torch.Tensor,
    digits: torch.Tensor,
    *,
    state_rank: int,
    max_transport_rank: int,
) -> ConditionalTransportDesign:
    return build_conditional_transport_design(
        states,
        deltas,
        digits,
        state_rank=state_rank,
        max_transport_rank=max_transport_rank,
    )


def cross_validate(
    states: torch.Tensor,
    deltas: torch.Tensor,
    digits: torch.Tensor,
    *,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    folds = config["fit_cross_validation_folds"]
    fold_ids = torch.arange(states.shape[0]) % folds
    candidates = []
    for state_rank in config["state_ranks"]:
        fold_designs = []
        for fold in range(folds):
            training = fold_ids != fold
            fold_designs.append(
                build_design(
                    states[training],
                    deltas[training],
                    digits[training],
                    state_rank=state_rank,
                    max_transport_rank=max(config["transport_ranks"]),
                )
            )
        for transport_rank in config["transport_ranks"]:
            for ridge in config["ridge_values"]:
                squared_error = 0.0
                squared_target = 0.0
                cosine_sum = 0.0
                rows = 0
                for fold, design in enumerate(fold_designs):
                    validation = fold_ids == fold
                    fitted = design.fit(
                        transport_rank=transport_rank,
                        ridge=ridge,
                    )
                    predicted = fitted.predict(
                        states[validation],
                        digits[validation],
                    )
                    target = deltas[validation]
                    squared_error += float((predicted - target).square().sum())
                    squared_target += float(target.square().sum())
                    cosine_sum += float(
                        torch.nn.functional.cosine_similarity(
                            predicted,
                            target,
                            dim=1,
                        ).sum()
                    )
                    rows += int(validation.sum())
                candidates.append(
                    {
                        "state_rank": state_rank,
                        "transport_rank": transport_rank,
                        "ridge": ridge,
                        "normalized_mse": squared_error / squared_target,
                        "mean_cosine_similarity": cosine_sum / rows,
                    }
                )
    selected = min(
        candidates,
        key=lambda row: (
            row["normalized_mse"],
            row["state_rank"],
            row["transport_rank"],
            -row["ridge"],
        ),
    )
    return selected, candidates


def model_tensors(
    prefix: str,
    model: ConditionalTransportModel,
) -> dict[str, torch.Tensor]:
    return {
        f"{prefix}_classes": torch.tensor(model.classes, dtype=torch.int64),
        f"{prefix}_state_mean": model.state_mean,
        f"{prefix}_state_basis": model.state_basis,
        f"{prefix}_score_scale": model.score_scale,
        f"{prefix}_delta_basis": model.delta_basis,
        f"{prefix}_weights": model.weights,
    }


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
    if not behavior["passes"]:
        raise SystemExit("fresh Qwen behavior gate did not pass")
    eligible_ids = exact_fit_quartets(behavior)
    if value_sha256(eligible_ids) != config["eligible_fit_quartets_sha256"]:
        raise SystemExit("eligible fit quartet hash mismatch")

    examples = build_phase6_carry_quartets(
        **dataset_config["dataset"]["parameters"]
    )
    observed_dataset_hash = phase6_carry_sha256(examples)
    if observed_dataset_hash != dataset_config["dataset"]["sha256"]:
        raise SystemExit("Phase 6 dataset hash mismatch")
    split_ids = {
        "fit": eligible_ids,
        "selection": sorted(
            {
                example.quartet_id
                for example in examples
                if example.split == "selection"
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
            label=f"Phase 6 {split}",
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
    fit_position = positions["fit"]
    carry_fit_states = extract_at(states["fit"]["carry_base"], fit_position)
    carry_fit_targets = (
        extract_at(states["fit"]["carry_increment"], fit_position)
        - carry_fit_states
    )
    control_fit_states = extract_at(states["fit"]["control_base"], fit_position)
    control_fit_targets = (
        extract_at(states["fit"]["control_increment"], fit_position)
        - control_fit_states
    )
    fit_digits = torch.tensor(
        [row.operand_a % 10 for row in rows["fit"]["carry_base"]]
    )
    selected_architecture, cv_candidates = cross_validate(
        carry_fit_states,
        carry_fit_targets,
        fit_digits,
        config=config,
    )
    state_rank = selected_architecture["state_rank"]
    transport_rank = selected_architecture["transport_rank"]
    ridge = selected_architecture["ridge"]
    carry_design = build_design(
        carry_fit_states,
        carry_fit_targets,
        fit_digits,
        state_rank=state_rank,
        max_transport_rank=transport_rank,
    )
    control_design = build_design(
        control_fit_states,
        control_fit_targets,
        fit_digits,
        state_rank=state_rank,
        max_transport_rank=transport_rank,
    )
    carry_model = carry_design.fit(transport_rank=transport_rank, ridge=ridge)
    control_model = control_design.fit(transport_rank=transport_rank, ridge=ridge)

    selection_position = positions["selection"]
    selection_base = states["selection"]["carry_base"]
    selection_states = extract_at(selection_base, selection_position)
    selection_digits = torch.tensor(
        [row.operand_a % 10 for row in rows["selection"]["carry_base"]]
    )
    classes = list(carry_model.classes)
    rotated_digits = torch.tensor(
        rotate_classes(selection_digits.tolist(), classes)
    )
    generator = torch.Generator().manual_seed(config["random_control_seed"])
    permutation = torch.randperm(selection_states.shape[0], generator=generator)
    carry_vectors = carry_model.predict(selection_states, selection_digits)
    no_carry_vectors = control_model.predict(selection_states, selection_digits)
    wrong_vectors = carry_model.predict(selection_states, rotated_digits)
    shuffled_vectors = carry_model.predict(
        selection_states[permutation],
        selection_digits,
    )
    templates = selection_base
    target = one_token_sequences(
        templates,
        selection_position,
        carry_vectors,
    )
    target_norms = sequence_norms(target)
    controls = {
        "matched_no_carry_norm_matched": norm_match_sequences(
            one_token_sequences(templates, selection_position, no_carry_vectors),
            target_norms,
        ),
        "wrong_class_norm_matched": norm_match_sequences(
            one_token_sequences(templates, selection_position, wrong_vectors),
            target_norms,
        ),
        "shuffled_recipient_norm_matched": norm_match_sequences(
            one_token_sequences(templates, selection_position, shuffled_vectors),
            target_norms,
        ),
        "random_norm_matched": random_norm_matched_sequences(
            target,
            target_norms,
            seed=config["random_control_seed"] + 1,
        ),
    }
    conditions = {"target": target, **controls}
    metrics = {}
    passing_scales = []
    rule = config["selection_rule"]
    for scale in config["scales"]:
        scale_metrics = {}
        for name, raw_delta in conditions.items():
            delta = tuple(value * scale for value in raw_delta)
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
        strongest_name = max(
            controls,
            key=lambda name: scale_metrics[name]["target_tens_accuracy"],
        )
        target_accuracy = scale_metrics["target"]["target_tens_accuracy"]
        strongest_accuracy = scale_metrics[strongest_name][
            "target_tens_accuracy"
        ]
        passes = (
            target_accuracy >= rule["minimum_tens_accuracy"]
            and target_accuracy - strongest_accuracy
            >= rule["minimum_control_advantage"]
            and (
                not rule["require_parse_rate"]
                or scale_metrics["target"]["parse_rate"] == 1.0
            )
        )
        scale_metrics["gate"] = {
            "target_tens_accuracy": target_accuracy,
            "strongest_control": strongest_name,
            "strongest_control_tens_accuracy": strongest_accuracy,
            "control_advantage": target_accuracy - strongest_accuracy,
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

    artifact_tensors = {
        **model_tensors("carry", carry_model),
        **model_tensors("control", control_model),
    }
    args.artifact_output.parent.mkdir(parents=True, exist_ok=True)
    save_file(artifact_tensors, str(args.artifact_output))
    artifact_hash = hashlib.sha256(args.artifact_output.read_bytes()).hexdigest()
    report = {
        "schema_version": "oli.phase6-qwen-conditional-carry-selection/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "selection_only",
        "model": model_config,
        "dataset_sha256": observed_dataset_hash,
        "behavior_result_sha256": config["behavior_result_sha256"],
        "eligible_fit_quartets": len(eligible_ids),
        "eligible_fit_quartets_sha256": config[
            "eligible_fit_quartets_sha256"
        ],
        "selection_quartets_sha256": config["selection_quartets_sha256"],
        "token_region_contract_sha256": config[
            "token_region_contract_sha256"
        ],
        "digit_token_ids": digit_token_ids,
        "hidden_state_index": hidden_index,
        "fit_cross_validation": {
            "folds": config["fit_cross_validation_folds"],
            "selection_metric": "normalized_mse",
            "selected": selected_architecture,
            "candidates": cv_candidates,
        },
        "metrics": metrics,
        "selection": {
            "scale": selected_scale,
            "passes": passes,
        },
        "passes": passes,
        "artifact": {
            "path": str(args.artifact_output),
            "sha256": artifact_hash,
            "width": int(carry_fit_states.shape[1]),
            "classes": classes,
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
            "Fit-only architecture choice and selection-only causal scale "
            "choice for a donor-free conditional carry writer. No development "
            "or audit claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    print(f"wrote {args.artifact_output}")


if __name__ == "__main__":
    main()
