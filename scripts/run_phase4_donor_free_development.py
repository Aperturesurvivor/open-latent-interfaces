#!/usr/bin/env python3
"""Validate fixed donor-free operand and carry coordinates on a frozen split."""

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
    accuracy_field: str = "target_tens_accuracy",
) -> dict[str, Any]:
    target_metrics = metrics[target]
    strongest_name = max(
        controls,
        key=lambda name: metrics[name][accuracy_field],
    )
    strongest_accuracy = metrics[strongest_name][accuracy_field]
    advantage = target_metrics[accuracy_field] - strongest_accuracy
    passes = (
        target_metrics[accuracy_field] >= minimum_accuracy
        and advantage >= minimum_advantage
        and (
            not require_parse_rate
            or target_metrics["parse_rate"] == 1.0
        )
    )
    return {
        "accuracy_field": accuracy_field,
        "target_accuracy": target_metrics[accuracy_field],
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
        raise SystemExit(f"refusing to overwrite validation result: {args.output}")

    config = json.loads(args.config.read_text())
    evaluation_split = config.get("evaluation_split", "development")
    if evaluation_split not in ("development", "audit"):
        raise SystemExit(f"unsupported evaluation split: {evaluation_split}")
    if evaluation_split == "audit":
        if not config.get("audit_authorized", False):
            raise SystemExit("audit is sealed")
        if config.get("maximum_audit_runs") != 1:
            raise SystemExit("audit config must authorize exactly one run")
        if str(args.output) != config.get("audit_output"):
            raise SystemExit("output path differs from frozen audit path")
        verify_sha256(Path(__file__), config["runner_sha256"])
    elif config.get("audit_authorized", False):
        raise SystemExit("development config cannot authorize audit")
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
    if evaluation_split == "development" and dataset_config.get(
        "audit_authorized",
        False,
    ):
        raise SystemExit("development validation requires a sealed audit")
    if not behavior["splits"]["development"]["passes"]:
        raise SystemExit("untouched development behavior gate did not pass")
    corrected_scale = correction.get(
        "selected_scale",
        correction.get("selection", {}).get("scale"),
    )
    if not correction["passes"] or corrected_scale != config["carry_scale"]:
        raise SystemExit("carry scale differs from bounded correction")
    if evaluation_split == "audit":
        audit_paths = {
            "development_config": Path(config["development_config"]),
            "development_result": Path(config["development_result"]),
            "development_metric_correction": Path(
                config["development_metric_correction"]
            ),
        }
        for name, path in audit_paths.items():
            verify_sha256(path, config[f"{name}_sha256"])
        development_config = json.loads(
            audit_paths["development_config"].read_text()
        )
        development_result = json.loads(
            audit_paths["development_result"].read_text()
        )
        metric_correction = json.loads(
            audit_paths["development_metric_correction"].read_text()
        )
        if not metric_correction["passes"]:
            raise SystemExit("corrected development package did not pass")
        if metric_correction["original_result"]["sha256"] != config[
            "development_result_sha256"
        ]:
            raise SystemExit("development correction/result mismatch")
        if development_result["config_sha256"] != config[
            "development_config_sha256"
        ]:
            raise SystemExit("development result/config mismatch")
        locked_keys = (
            "base_model_batch_size",
            "behavior_result_sha256",
            "carry_context_hidden_state_index",
            "carry_scale",
            "class_prototype_artifact_sha256",
            "dataset_config_sha256",
            "development_rule",
            "operand_hidden_state_index",
            "operand_scale",
            "parent_correction_sha256",
            "random_control_seed",
            "universal_carry_artifact_sha256",
        )
        for key in locked_keys:
            if config[key] != development_config[key]:
                raise SystemExit(f"audit changed development field: {key}")

    examples = build_phase4_carry_quartets(
        **dataset_config["dataset"]["parameters"]
    )
    observed_dataset_hash = phase4_carry_sha256(examples)
    if observed_dataset_hash != dataset_config["dataset"]["sha256"]:
        raise SystemExit("Phase 4 dataset hash mismatch")
    evaluation_ids = sorted(
        {
            example.quartet_id
            for example in examples
            if example.split == evaluation_split
        }
    )
    quartet_hash_key = f"{evaluation_split}_quartets_sha256"
    if value_sha256(evaluation_ids) != config[quartet_hash_key]:
        raise SystemExit(f"{evaluation_split} quartet hash mismatch")
    by_quartet = {
        quartet_id: {
            row.variant: row
            for row in examples
            if row.quartet_id == quartet_id
        }
        for quartet_id in evaluation_ids
    }
    variant_names = (
        "carry_base",
        "carry_increment",
        "control_base",
        "control_increment",
    )
    rows = {
        variant: [
            by_quartet[quartet_id][variant] for quartet_id in evaluation_ids
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
    for index in range(len(evaluation_ids)):
        carry_base = token_ids["carry_base"][index]
        carry_increment = token_ids["carry_increment"][index]
        control_base = token_ids["control_base"][index]
        control_increment = token_ids["control_increment"][index]
        changed = differing_position(
            carry_base,
            carry_increment,
            label=f"{evaluation_split} operand",
        )
        if changed != differing_position(
            control_base,
            control_increment,
            label=f"{evaluation_split} control operand",
        ):
            raise SystemExit(f"{evaluation_split} operand positions differ")
        context = differing_position(
            carry_base,
            control_base,
            label=f"{evaluation_split} carry context",
        )
        if context != differing_position(
            carry_increment,
            control_increment,
            label=f"{evaluation_split} increment context",
        ):
            raise SystemExit(f"{evaluation_split} context positions differ")
        changed_positions.append(changed)
        context_positions.append(context)
        token_contract.append([len(carry_base), changed, context])
    token_contract_key = f"{evaluation_split}_token_region_contract_sha256"
    if value_sha256(token_contract) != config[token_contract_key]:
        raise SystemExit(f"{evaluation_split} token-region contract mismatch")
    digit_token_ids = verify_decimal_digit_contract(
        tokenizer,
        rendered["carry_base"][0],
    )

    class_artifact = load_file(str(paths["class_prototype"]))
    universal_artifact = load_file(str(paths["universal_carry"]))
    classes = class_artifact["source_digits"].tolist()
    source_digits = [row.operand_a % 10 for row in rows["carry_base"]]
    if not set(source_digits).issubset(set(classes)):
        raise SystemExit(f"{evaluation_split} contains an unsupported source digit")
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
        carry_vector.repeat(len(evaluation_ids), 1),
    )
    carry_no_carry = norm_match_sequences(
        one_token_sequences(
            base_states[carry_index].values,
            context_positions,
            no_carry_vector.repeat(len(evaluation_ids), 1),
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
        accuracy_field=config.get(
            "operand_accuracy_field",
            "target_tens_accuracy",
        ),
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
        "schema_version": "oli.phase4-donor-free-validation/v2",
        "created_at": datetime.now(UTC).isoformat(),
        "status": f"one_shot_{evaluation_split}",
        "evaluation_split": evaluation_split,
        "model": model_config,
        "dataset_sha256": observed_dataset_hash,
        "behavior_source": {
            "sha256": config["behavior_result_sha256"],
            "original_gate_passed": behavior["passes"],
            "development_behavior_passed": behavior["splits"]["development"][
                "passes"
            ],
        },
        "evaluation_quartets_sha256": config[quartet_hash_key],
        "evaluation_token_region_contract_sha256": config[token_contract_key],
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
            f"One-shot {evaluation_split} validation of fixed donor-free "
            "operand and rank-one universal carry coordinates."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
