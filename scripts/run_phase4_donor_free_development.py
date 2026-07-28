#!/usr/bin/env python3
"""Validate fixed donor-free operand and carry coordinates on development."""

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
    class_lookup,
    differing_position,
    one_token_sequences,
    rotate_classes,
)
from safetensors.torch import load_file
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


def gate(
    metrics: dict[str, dict[str, Any]],
    *,
    target: str,
    controls: tuple[str, ...],
    minimum_accuracy: float,
    minimum_advantage: float,
    require_parse_rate: bool,
) -> dict[str, Any]:
    target_metrics = metrics[target]
    strongest_name = max(
        controls,
        key=lambda name: metrics[name]["target_tens_accuracy"],
    )
    strongest_accuracy = metrics[strongest_name]["target_tens_accuracy"]
    advantage = target_metrics["target_tens_accuracy"] - strongest_accuracy
    passes = (
        target_metrics["target_tens_accuracy"] >= minimum_accuracy
        and advantage >= minimum_advantage
        and (
            not require_parse_rate
            or target_metrics["parse_rate"] == 1.0
        )
    )
    return {
        "target_tens_accuracy": target_metrics["target_tens_accuracy"],
        "target_full_accuracy": target_metrics["target_full_accuracy"],
        "strongest_control": strongest_name,
        "strongest_control_tens_accuracy": strongest_accuracy,
        "control_advantage": advantage,
        "parse_rate": target_metrics["parse_rate"],
        "passes": passes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite development result: {args.output}")

    config = json.loads(args.config.read_text())
    paths = {
        "dataset": Path(config["dataset_config"]),
        "behavior": Path(config["behavior_result"]),
        "correction": Path(config["parent_correction"]),
        "class_prototype": Path(config["class_prototype_artifact"]),
        "universal_carry": Path(config["universal_carry_artifact"]),
    }
    for name, suffix in (
        ("dataset", "dataset_config_sha256"),
        ("behavior", "behavior_result_sha256"),
        ("correction", "parent_correction_sha256"),
        ("class_prototype", "class_prototype_artifact_sha256"),
        ("universal_carry", "universal_carry_artifact_sha256"),
    ):
        verify_sha256(paths[name], config[suffix])
    dataset_config = json.loads(paths["dataset"].read_text())
    behavior = json.loads(paths["behavior"].read_text())
    correction = json.loads(paths["correction"].read_text())
    if dataset_config.get("audit_authorized", False):
        raise SystemExit("development validation requires a sealed audit")
    if not behavior["splits"]["development"]["passes"]:
        raise SystemExit("untouched development behavior gate did not pass")
    if not correction["passes"] or correction["selected_scale"] != config[
        "carry_scale"
    ]:
        raise SystemExit("carry scale differs from bounded correction")

    examples = build_phase4_carry_quartets(
        **dataset_config["dataset"]["parameters"]
    )
    observed_dataset_hash = phase4_carry_sha256(examples)
    if observed_dataset_hash != dataset_config["dataset"]["sha256"]:
        raise SystemExit("Phase 4 dataset hash mismatch")
    development_ids = sorted(
        {
            example.quartet_id
            for example in examples
            if example.split == "development"
        }
    )
    if value_sha256(development_ids) != config["development_quartets_sha256"]:
        raise SystemExit("development quartet hash mismatch")
    by_quartet = {
        quartet_id: {
            row.variant: row
            for row in examples
            if row.quartet_id == quartet_id
        }
        for quartet_id in development_ids
    }
    variant_names = (
        "carry_base",
        "carry_increment",
        "control_base",
        "control_increment",
    )
    rows = {
        variant: [
            by_quartet[quartet_id][variant] for quartet_id in development_ids
        ]
        for variant in variant_names
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
        variant: [
            render_prefilled_chat(
                tokenizer,
                row.prompt,
                assistant_prefix=dataset_config["assistant_prefix"],
            )
            for row in variant_rows
        ]
        for variant, variant_rows in rows.items()
    }
    token_ids = {
        variant: [tokenizer(prompt)["input_ids"] for prompt in prompts]
        for variant, prompts in rendered.items()
    }
    changed_positions = []
    context_positions = []
    token_contract = []
    for index in range(len(development_ids)):
        carry_base = token_ids["carry_base"][index]
        carry_increment = token_ids["carry_increment"][index]
        control_base = token_ids["control_base"][index]
        control_increment = token_ids["control_increment"][index]
        changed = differing_position(
            carry_base,
            carry_increment,
            label="development operand",
        )
        if changed != differing_position(
            control_base,
            control_increment,
            label="development control operand",
        ):
            raise SystemExit("development operand positions differ")
        context = differing_position(
            carry_base,
            control_base,
            label="development carry context",
        )
        if context != differing_position(
            carry_increment,
            control_increment,
            label="development increment context",
        ):
            raise SystemExit("development context positions differ")
        changed_positions.append(changed)
        context_positions.append(context)
        token_contract.append([len(carry_base), changed, context])
    if value_sha256(token_contract) != config[
        "development_token_region_contract_sha256"
    ]:
        raise SystemExit("development token-region contract mismatch")
    digit_token_ids = verify_decimal_digit_contract(
        tokenizer,
        rendered["carry_base"][0],
    )

    class_artifact = load_file(str(paths["class_prototype"]))
    universal_artifact = load_file(str(paths["universal_carry"]))
    classes = class_artifact["source_digits"].tolist()
    source_digits = [row.operand_a % 10 for row in rows["carry_base"]]
    if not set(source_digits).issubset(set(classes)):
        raise SystemExit("development contains an unsupported source digit")
    operand_vectors = class_lookup(
        class_artifact["operand_delta"].float(),
        source_digits,
        classes,
    )
    operand_wrong_vectors = class_lookup(
        class_artifact["operand_delta"].float(),
        rotate_classes(source_digits, classes),
        classes,
    )
    carry_vector = universal_artifact["carry_context_delta"].float()
    no_carry_vector = universal_artifact["control_context_delta"].float()

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
    base_states = capture.capture_sequences(
        rendered["carry_base"],
        hidden_state_indices=hidden_indices,
        batch_size=config["base_model_batch_size"],
    )

    operand_index = config["operand_hidden_state_index"]
    operand_target = one_token_sequences(
        base_states[operand_index].values,
        changed_positions,
        operand_vectors,
    )
    operand_wrong = norm_match_sequences(
        one_token_sequences(
            base_states[operand_index].values,
            changed_positions,
            operand_wrong_vectors,
        ),
        sequence_norms(operand_target),
    )
    operand_random = random_norm_matched_sequences(
        operand_target,
        sequence_norms(operand_target),
        seed=config["random_control_seed"] + 1,
    )
    carry_index = config["carry_context_hidden_state_index"]
    carry_target = one_token_sequences(
        base_states[carry_index].values,
        context_positions,
        carry_vector.repeat(len(development_ids), 1),
    )
    carry_no_carry = norm_match_sequences(
        one_token_sequences(
            base_states[carry_index].values,
            context_positions,
            no_carry_vector.repeat(len(development_ids), 1),
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
            "matched_no_carry_norm_matched": carry_no_carry,
            "random_norm_matched": carry_random,
        },
    }
    writer_indices = {"operand": operand_index, "carry_context": carry_index}
    writer_scales = {
        "operand": config["operand_scale"],
        "carry_context": config["carry_scale"],
    }
    metrics = {}
    for writer, conditions in writer_conditions.items():
        writer_metrics = {}
        scale = writer_scales[writer]
        for condition, raw_delta in conditions.items():
            delta = tuple(value * scale for value in raw_delta)
            responses = generate_chunks(
                model,
                tokenizer,
                rendered["carry_base"],
                delta,
                hidden_state_index=writer_indices[writer],
                batch_size=config["base_model_batch_size"],
                device=device,
            )
            writer_metrics[condition] = summarize_outputs(
                responses,
                rows["carry_base"],
                rows["carry_increment"],
                delta,
                base_states[writer_indices[writer]].values,
            )
        metrics[writer] = writer_metrics

    rule = config["development_rule"]
    operand_gate = gate(
        metrics["operand"],
        target="target",
        controls=("wrong_class_norm_matched", "random_norm_matched"),
        minimum_accuracy=rule["operand_minimum_tens_accuracy"],
        minimum_advantage=rule["minimum_control_advantage"],
        require_parse_rate=rule["require_parse_rate"],
    )
    carry_gate = gate(
        metrics["carry_context"],
        target="target",
        controls=("matched_no_carry_norm_matched", "random_norm_matched"),
        minimum_accuracy=rule["carry_minimum_tens_accuracy"],
        minimum_advantage=rule["minimum_control_advantage"],
        require_parse_rate=rule["require_parse_rate"],
    )
    report = {
        "schema_version": "oli.phase4-donor-free-development/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "one_shot_development",
        "model": model_config,
        "dataset_sha256": observed_dataset_hash,
        "behavior_source": {
            "sha256": config["behavior_result_sha256"],
            "original_gate_passed": behavior["passes"],
            "development_behavior_passed": behavior["splits"]["development"][
                "passes"
            ],
        },
        "development_quartets_sha256": config["development_quartets_sha256"],
        "development_token_region_contract_sha256": config[
            "development_token_region_contract_sha256"
        ],
        "digit_token_ids": digit_token_ids,
        "sources": {
            "correction_sha256": config["parent_correction_sha256"],
            "class_prototype_sha256": config[
                "class_prototype_artifact_sha256"
            ],
            "universal_carry_sha256": config[
                "universal_carry_artifact_sha256"
            ],
        },
        "fixed_interface": {
            "operand": {
                "hidden_state_index": operand_index,
                "scale": config["operand_scale"],
                "writer_vectors": len(classes),
                "class_count": len(classes),
            },
            "carry_context": {
                "hidden_state_index": carry_index,
                "scale": config["carry_scale"],
                "rank": 1,
                "class_invariant": True,
            },
        },
        "metrics": metrics,
        "gates": {
            "operand": operand_gate,
            "carry_context": carry_gate,
        },
        "passes": {
            "operand": operand_gate["passes"],
            "carry_context": carry_gate["passes"],
            "all": operand_gate["passes"] and carry_gate["passes"],
        },
        "development_rule": rule,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "One-shot untouched-development validation of fixed donor-free "
            "operand and rank-one universal carry coordinates. Audit remains "
            "sealed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
