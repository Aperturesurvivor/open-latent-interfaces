#!/usr/bin/env python3
"""Select a scale for the fixed Qwen carry-context donor delta."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from run_phase4_carry_sequence_boundary import (
    generate_chunks,
    norm_match_sequences,
    random_norm_matched_sequences,
    sequence_norms,
    subtract_sequences,
    summarize_outputs,
    value_sha256,
    verify_sha256,
)
from run_phase4_carry_token_regions import mask_positions
from run_phase4_donor_free_prototypes import differing_position
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite Qwen scale result: {args.output}")

    config = json.loads(args.config.read_text())
    dataset_path = Path(config["dataset_config"])
    behavior_path = Path(config["behavior_result"])
    parent_path = Path(config["parent_result"])
    verify_sha256(dataset_path, config["dataset_config_sha256"])
    verify_sha256(behavior_path, config["behavior_result_sha256"])
    verify_sha256(parent_path, config["parent_result_sha256"])
    dataset_config = json.loads(dataset_path.read_text())
    behavior = json.loads(behavior_path.read_text())
    parent = json.loads(parent_path.read_text())
    if dataset_config.get("audit_authorized", False):
        raise SystemExit("Qwen carry scale selection requires a sealed audit")
    if not behavior["passes"]:
        raise SystemExit("Qwen behavior gate did not pass")
    if parent["passes"]["carry_context_token"]:
        raise SystemExit("scale follow-up requires the unit-scale carry non-pass")
    if parent["selection"]["carry_context_token"]["hidden_state_index"] != config[
        "hidden_state_index"
    ]:
        raise SystemExit("scale follow-up changed the selected boundary")

    examples = build_phase4_carry_quartets(
        **dataset_config["dataset"]["parameters"]
    )
    observed_dataset_hash = phase4_carry_sha256(examples)
    if observed_dataset_hash != dataset_config["dataset"]["sha256"]:
        raise SystemExit("Phase 5 dataset hash mismatch")
    selection_ids = sorted(
        {
            example.quartet_id
            for example in examples
            if example.split == "selection"
        }
    )
    if value_sha256(selection_ids) != config["selection_quartets_sha256"]:
        raise SystemExit("selection quartet hash mismatch")
    by_quartet = {
        quartet_id: {
            row.variant: row
            for row in examples
            if row.quartet_id == quartet_id
        }
        for quartet_id in selection_ids
    }
    variant_names = (
        "carry_base",
        "carry_increment",
        "control_base",
        "control_increment",
    )
    rows = {
        variant: [
            by_quartet[quartet_id][variant] for quartet_id in selection_ids
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
    context_positions = []
    token_contract = []
    for index in range(len(selection_ids)):
        carry_base = token_ids["carry_base"][index]
        carry_increment = token_ids["carry_increment"][index]
        control_base = token_ids["control_base"][index]
        control_increment = token_ids["control_increment"][index]
        changed = differing_position(
            carry_base,
            carry_increment,
            label="Qwen selection operand",
        )
        if changed != differing_position(
            control_base,
            control_increment,
            label="Qwen selection control operand",
        ):
            raise SystemExit("Qwen operand positions differ")
        context = differing_position(
            carry_base,
            control_base,
            label="Qwen selection context",
        )
        if context != differing_position(
            carry_increment,
            control_increment,
            label="Qwen selection increment context",
        ):
            raise SystemExit("Qwen context positions differ")
        context_positions.append(context)
        token_contract.append([len(carry_base), changed, context])
    if value_sha256(token_contract) != config["token_region_contract_sha256"]:
        raise SystemExit("Qwen selection token contract mismatch")
    digit_token_ids = verify_decimal_digit_contract(
        tokenizer,
        rendered["carry_base"][0],
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
    hidden_index = config["hidden_state_index"]
    started = time.perf_counter()
    states = {
        variant: capture.capture_sequences(
            prompts,
            hidden_state_indices=[hidden_index],
            batch_size=config["base_model_batch_size"],
        )[hidden_index].values
        for variant, prompts in rendered.items()
    }
    carry_delta = subtract_sequences(
        states["carry_increment"],
        states["carry_base"],
        label="Qwen carry",
    )
    control_delta = subtract_sequences(
        states["control_increment"],
        states["control_base"],
        label="Qwen control",
    )
    masks = [[position] for position in context_positions]
    carry_context = mask_positions(carry_delta, masks)
    control_context = norm_match_sequences(
        mask_positions(control_delta, masks),
        sequence_norms(carry_context),
    )
    random_context = random_norm_matched_sequences(
        carry_context,
        sequence_norms(carry_context),
        seed=config["random_control_seed"],
    )
    conditions = {
        "target": carry_context,
        "matched_no_carry_norm_matched": control_context,
        "random_norm_matched": random_context,
    }
    metrics = {}
    rule = config["selection_rule"]
    passing_scales = []
    for scale in config["scales"]:
        scale_metrics = {}
        for name, raw_delta in conditions.items():
            delta = tuple(value * scale for value in raw_delta)
            responses = generate_chunks(
                model,
                tokenizer,
                rendered["carry_base"],
                delta,
                hidden_state_index=hidden_index,
                batch_size=config["base_model_batch_size"],
                device=device,
            )
            scale_metrics[name] = summarize_outputs(
                responses,
                rows["carry_base"],
                rows["carry_increment"],
                delta,
                states["carry_base"],
            )
        target_accuracy = scale_metrics["target"]["target_tens_accuracy"]
        strongest_control = max(
            scale_metrics[name]["target_tens_accuracy"]
            for name in (
                "matched_no_carry_norm_matched",
                "random_norm_matched",
            )
        )
        passes = (
            target_accuracy >= rule["minimum_tens_accuracy"]
            and target_accuracy - strongest_control
            >= rule["minimum_control_advantage"]
            and (
                not rule["require_parse_rate"]
                or scale_metrics["target"]["parse_rate"] == 1.0
            )
        )
        scale_metrics["gate"] = {
            "target_tens_accuracy": target_accuracy,
            "strongest_control_tens_accuracy": strongest_control,
            "control_advantage": target_accuracy - strongest_control,
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
    report = {
        "schema_version": "oli.phase5-qwen-carry-context-scale/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "selection_only",
        "model": model_config,
        "dataset_sha256": observed_dataset_hash,
        "behavior_result_sha256": config["behavior_result_sha256"],
        "parent_result_sha256": config["parent_result_sha256"],
        "selection_quartets_sha256": config["selection_quartets_sha256"],
        "token_region_contract_sha256": config["token_region_contract_sha256"],
        "digit_token_ids": digit_token_ids,
        "hidden_state_index": hidden_index,
        "scales": config["scales"],
        "metrics": metrics,
        "selection": {
            "passing_scales": passing_scales,
            "selected_scale": selected_scale,
            "passes": passes,
        },
        "passes": passes,
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
            "Selection-only scale follow-up at the fixed Qwen carry-context "
            "token and hidden-state index. Donor-dependent upper bound only."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
