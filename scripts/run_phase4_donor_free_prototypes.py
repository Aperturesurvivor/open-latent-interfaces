#!/usr/bin/env python3
"""Fit and select donor-free operand and carry-context class prototypes."""

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
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.phase4_data import (
    build_phase4_carry_quartets,
    phase4_carry_sha256,
)
from open_latent_interfaces.prefill import (
    render_prefilled_chat,
    verify_decimal_digit_contract,
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


def differing_position(left: list[int], right: list[int], *, label: str) -> int:
    if len(left) != len(right):
        raise ValueError(f"{label} token lengths differ")
    positions = [
        index
        for index, (left_id, right_id) in enumerate(zip(left, right, strict=True))
        if left_id != right_id
    ]
    if len(positions) != 1:
        raise ValueError(f"{label} requires exactly one differing token")
    return positions[0]


def fit_class_prototypes(
    deltas: list[torch.Tensor],
    source_digits: list[int],
    classes: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    prototypes = []
    counts = []
    for digit in classes:
        selected = [
            delta.float()
            for delta, observed in zip(deltas, source_digits, strict=True)
            if observed == digit
        ]
        if not selected:
            raise ValueError(f"source digit {digit} has no fit examples")
        prototypes.append(torch.stack(selected).mean(dim=0))
        counts.append(len(selected))
    return torch.stack(prototypes), torch.tensor(counts, dtype=torch.int64)


def class_lookup(
    prototypes: torch.Tensor,
    source_digits: list[int],
    classes: list[int],
) -> torch.Tensor:
    index = {digit: position for position, digit in enumerate(classes)}
    return torch.stack([prototypes[index[digit]] for digit in source_digits])


def one_token_sequences(
    templates: tuple[torch.Tensor, ...],
    positions: list[int],
    vectors: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    values = []
    for template, position, vector in zip(
        templates,
        positions,
        vectors,
        strict=True,
    ):
        value = torch.zeros_like(template)
        value[position] = vector
        values.append(value)
    return tuple(values)


def rotate_classes(source_digits: list[int], classes: list[int]) -> list[int]:
    index = {digit: position for position, digit in enumerate(classes)}
    return [classes[(index[digit] + 1) % len(classes)] for digit in source_digits]


def select_scale(
    metrics: dict[str, dict[str, dict[str, Any]]],
    *,
    target: str,
    controls: tuple[str, ...],
    minimum_accuracy: float,
    minimum_advantage: float,
) -> tuple[float, bool]:
    def score(scale: str) -> tuple[float, float, float, float]:
        row = metrics[scale]
        strongest = max(row[name]["target_tens_accuracy"] for name in controls)
        target_metrics = row[target]
        return (
            target_metrics["target_tens_accuracy"],
            target_metrics["target_tens_accuracy"] - strongest,
            target_metrics["target_full_accuracy"],
            -float(scale),
        )

    selected = max(metrics, key=score)
    row = metrics[selected]
    target_metrics = row[target]
    strongest = max(row[name]["target_tens_accuracy"] for name in controls)
    passes = (
        target_metrics["target_tens_accuracy"] >= minimum_accuracy
        and target_metrics["target_tens_accuracy"] - strongest
        >= minimum_advantage
        and target_metrics["parse_rate"] == 1.0
    )
    return float(selected), passes


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
    dataset_path = Path(config["dataset_config"])
    behavior_path = Path(config["behavior_result"])
    parent_path = Path(config["parent_result"])
    verify_sha256(dataset_path, config["dataset_config_sha256"])
    verify_sha256(behavior_path, config["behavior_result_sha256"])
    verify_sha256(parent_path, config["parent_result_sha256"])
    dataset_config = json.loads(dataset_path.read_text())
    behavior = json.loads(behavior_path.read_text())
    if dataset_config.get("audit_authorized", False):
        raise SystemExit("prototype selection requires a sealed audit")

    eligible_ids = exact_fit_quartets(behavior)
    if value_sha256(eligible_ids) != config["eligible_fit_quartets_sha256"]:
        raise SystemExit("eligible fit quartet hash mismatch")
    examples = build_phase4_carry_quartets(
        **dataset_config["dataset"]["parameters"]
    )
    observed_dataset_hash = phase4_carry_sha256(examples)
    if observed_dataset_hash != dataset_config["dataset"]["sha256"]:
        raise SystemExit("Phase 4 dataset hash mismatch")

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
    if value_sha256(split_ids["selection"]) != config["selection_quartets_sha256"]:
        raise SystemExit("selection quartet hash mismatch")
    by_quartet = {
        quartet_id: {
            row.variant: row
            for row in examples
            if row.quartet_id == quartet_id
        }
        for quartet_id in split_ids["fit"] + split_ids["selection"]
    }
    variant_names = (
        "carry_base",
        "carry_increment",
        "control_base",
        "control_increment",
    )
    rows = {
        split: {
            variant: [by_quartet[quartet_id][variant] for quartet_id in ids]
            for variant in variant_names
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
        split: {
            variant: [
                render_prefilled_chat(
                    tokenizer,
                    row.prompt,
                    assistant_prefix=dataset_config["assistant_prefix"],
                )
                for row in variant_rows
            ]
            for variant, variant_rows in split_rows.items()
        }
        for split, split_rows in rows.items()
    }
    token_ids = {
        split: {
            variant: [tokenizer(prompt)["input_ids"] for prompt in prompts]
            for variant, prompts in split_rendered.items()
        }
        for split, split_rendered in rendered.items()
    }
    positions = {}
    token_contract = []
    for split in ("fit", "selection"):
        changed = []
        context = []
        for index in range(len(split_ids[split])):
            carry_base = token_ids[split]["carry_base"][index]
            carry_increment = token_ids[split]["carry_increment"][index]
            control_base = token_ids[split]["control_base"][index]
            operand_position = differing_position(
                carry_base,
                carry_increment,
                label=f"{split} operand",
            )
            if operand_position != differing_position(
                control_base,
                token_ids[split]["control_increment"][index],
                label=f"{split} control operand",
            ):
                raise SystemExit("carry/control operand positions differ")
            context_position = differing_position(
                carry_base,
                control_base,
                label=f"{split} context",
            )
            changed.append(operand_position)
            context.append(context_position)
            if split == "selection":
                token_contract.append(
                    [len(carry_base), operand_position, context_position]
                )
        positions[split] = {"changed": changed, "context": context}
    if value_sha256(token_contract) != config["token_region_contract_sha256"]:
        raise SystemExit("selection token-region contract mismatch")

    classes = config["source_digits"]
    fit_source_digits = [
        row.operand_a % 10 for row in rows["fit"]["carry_base"]
    ]
    selection_source_digits = [
        row.operand_a % 10 for row in rows["selection"]["carry_base"]
    ]
    counts = Counter(fit_source_digits)
    if set(counts) != set(classes) or min(counts.values()) < config[
        "minimum_examples_per_source_digit"
    ]:
        raise SystemExit(f"insufficient source-digit support: {counts}")
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
    hidden_indices = (
        config["operand_hidden_state_index"],
        config["carry_context_hidden_state_index"],
    )
    started = time.perf_counter()
    states = {
        split: {
            variant: capture.capture_sequences(
                prompts,
                hidden_state_indices=hidden_indices,
                batch_size=config["base_model_batch_size"],
            )
            for variant, prompts in split_rendered.items()
        }
        for split, split_rendered in rendered.items()
    }

    operand_index = config["operand_hidden_state_index"]
    carry_index = config["carry_context_hidden_state_index"]
    operand_fit_deltas = []
    carry_fit_deltas = []
    control_fit_deltas = []
    for index in range(len(split_ids["fit"])):
        changed = positions["fit"]["changed"][index]
        context = positions["fit"]["context"][index]
        operand_fit_deltas.append(
            states["fit"]["carry_increment"][operand_index].values[index][changed]
            - states["fit"]["carry_base"][operand_index].values[index][changed]
        )
        carry_fit_deltas.append(
            states["fit"]["carry_increment"][carry_index].values[index][context]
            - states["fit"]["carry_base"][carry_index].values[index][context]
        )
        control_fit_deltas.append(
            states["fit"]["control_increment"][carry_index].values[index][context]
            - states["fit"]["control_base"][carry_index].values[index][context]
        )
    operand_prototypes, class_counts = fit_class_prototypes(
        operand_fit_deltas,
        fit_source_digits,
        classes,
    )
    carry_prototypes, _ = fit_class_prototypes(
        carry_fit_deltas,
        fit_source_digits,
        classes,
    )
    control_prototypes, _ = fit_class_prototypes(
        control_fit_deltas,
        fit_source_digits,
        classes,
    )
    args.prototype_output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "source_digits": torch.tensor(classes, dtype=torch.int64),
            "class_counts": class_counts,
            "operand_delta": operand_prototypes,
            "carry_context_delta": carry_prototypes,
            "control_context_delta": control_prototypes,
        },
        str(args.prototype_output),
    )
    prototype_hash = hashlib.sha256(args.prototype_output.read_bytes()).hexdigest()

    selection_base = states["selection"]["carry_base"]
    operand_vectors = class_lookup(
        operand_prototypes,
        selection_source_digits,
        classes,
    )
    operand_wrong_vectors = class_lookup(
        operand_prototypes,
        rotate_classes(selection_source_digits, classes),
        classes,
    )
    carry_vectors = class_lookup(
        carry_prototypes,
        selection_source_digits,
        classes,
    )
    carry_control_vectors = class_lookup(
        control_prototypes,
        selection_source_digits,
        classes,
    )
    carry_wrong_vectors = class_lookup(
        carry_prototypes,
        rotate_classes(selection_source_digits, classes),
        classes,
    )
    templates = selection_base[operand_index].values
    operand_target = one_token_sequences(
        templates,
        positions["selection"]["changed"],
        operand_vectors,
    )
    operand_wrong = norm_match_sequences(
        one_token_sequences(
            templates,
            positions["selection"]["changed"],
            operand_wrong_vectors,
        ),
        sequence_norms(operand_target),
    )
    operand_random = random_norm_matched_sequences(
        operand_target,
        sequence_norms(operand_target),
        seed=config["random_control_seed"] + 1,
    )
    carry_templates = selection_base[carry_index].values
    carry_target = one_token_sequences(
        carry_templates,
        positions["selection"]["context"],
        carry_vectors,
    )
    carry_control = norm_match_sequences(
        one_token_sequences(
            carry_templates,
            positions["selection"]["context"],
            carry_control_vectors,
        ),
        sequence_norms(carry_target),
    )
    carry_wrong = norm_match_sequences(
        one_token_sequences(
            carry_templates,
            positions["selection"]["context"],
            carry_wrong_vectors,
        ),
        sequence_norms(carry_target),
    )
    carry_random = random_norm_matched_sequences(
        carry_target,
        sequence_norms(carry_target),
        seed=config["random_control_seed"] + 2,
    )
    writer_conditions = {
        "operand": {
            "target": operand_target,
            "wrong_class_norm_matched": operand_wrong,
            "random_norm_matched": operand_random,
        },
        "carry_context": {
            "target": carry_target,
            "matched_no_carry_norm_matched": carry_control,
            "wrong_class_norm_matched": carry_wrong,
            "random_norm_matched": carry_random,
        },
    }
    writer_indices = {"operand": operand_index, "carry_context": carry_index}
    metrics = {}
    for writer, conditions in writer_conditions.items():
        writer_metrics = {}
        for scale in config["scales"]:
            scale_metrics = {}
            for condition, raw_delta in conditions.items():
                delta = tuple(value * scale for value in raw_delta)
                responses = generate_chunks(
                    model,
                    tokenizer,
                    rendered["selection"]["carry_base"],
                    delta,
                    hidden_state_index=writer_indices[writer],
                    batch_size=config["base_model_batch_size"],
                    device=device,
                )
                scale_metrics[condition] = summarize_outputs(
                    responses,
                    rows["selection"]["carry_base"],
                    rows["selection"]["carry_increment"],
                    delta,
                    selection_base[writer_indices[writer]].values,
                )
            writer_metrics[str(scale)] = scale_metrics
        metrics[writer] = writer_metrics

    rule = config["selection_rule"]
    operand_scale, operand_passes = select_scale(
        metrics["operand"],
        target="target",
        controls=("wrong_class_norm_matched", "random_norm_matched"),
        minimum_accuracy=rule["operand_minimum_tens_accuracy"],
        minimum_advantage=rule["minimum_control_advantage"],
    )
    carry_scale, carry_passes = select_scale(
        metrics["carry_context"],
        target="target",
        controls=(
            "matched_no_carry_norm_matched",
            "wrong_class_norm_matched",
            "random_norm_matched",
        ),
        minimum_accuracy=rule["carry_minimum_tens_accuracy"],
        minimum_advantage=rule["minimum_control_advantage"],
    )
    report = {
        "schema_version": "oli.phase4-donor-free-prototype-selection/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "selection_only",
        "model": model_config,
        "dataset_sha256": observed_dataset_hash,
        "parent_result_sha256": config["parent_result_sha256"],
        "behavior_source": {
            "sha256": config["behavior_result_sha256"],
            "original_gate_passed": behavior["passes"],
            "eligible_fit_quartets": len(eligible_ids),
            "eligible_fit_quartets_sha256": config[
                "eligible_fit_quartets_sha256"
            ],
        },
        "fit_class_counts": {
            str(digit): counts[digit] for digit in classes
        },
        "selection_quartets_sha256": config["selection_quartets_sha256"],
        "token_region_contract_sha256": config["token_region_contract_sha256"],
        "digit_token_ids": digit_token_ids,
        "hidden_state_indices": {
            "operand": operand_index,
            "carry_context": carry_index,
        },
        "scales": config["scales"],
        "metrics": metrics,
        "selection": {
            "operand": {"scale": operand_scale, "passes": operand_passes},
            "carry_context": {"scale": carry_scale, "passes": carry_passes},
        },
        "passes": {
            "operand": operand_passes,
            "carry_context": carry_passes,
            "all": operand_passes and carry_passes,
        },
        "prototype": {
            "path": str(args.prototype_output),
            "sha256": prototype_hash,
            "source_digits": classes,
            "width": int(operand_prototypes.shape[1]),
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
            "Selection-only donor-free class-mean prototype viability. "
            "Passing does not establish compact rank, development "
            "generalization, an individual neuron, or a thought transcript."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    print(f"wrote {args.prototype_output}")


if __name__ == "__main__":
    main()
