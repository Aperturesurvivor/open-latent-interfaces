#!/usr/bin/env python3
"""Run one-shot development for the matched Qwen carry interaction."""

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
    one_token_sequences,
    rotate_classes,
)
from run_phase6_conditional_carry_selection import (
    build_design,
    cross_validate,
    exact_fit_quartets,
    extract_at,
    model_tensors,
    render_rows,
    token_positions,
)
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.phase6_data import (
    build_phase6_carry_quartets,
    phase6_carry_sha256,
)
from open_latent_interfaces.prefill import verify_decimal_digit_contract


def gate(
    metrics: dict[str, dict[str, Any]],
    rule: dict[str, Any],
) -> dict[str, Any]:
    controls = tuple(name for name in metrics if name != "target")
    strongest = max(
        controls,
        key=lambda name: metrics[name]["target_tens_accuracy"],
    )
    target_accuracy = metrics["target"]["target_tens_accuracy"]
    control_accuracy = metrics[strongest]["target_tens_accuracy"]
    advantage = target_accuracy - control_accuracy
    passes = (
        target_accuracy >= rule["minimum_tens_accuracy"]
        and advantage >= rule["minimum_control_advantage"]
        and (
            not rule["require_parse_rate"]
            or metrics["target"]["parse_rate"] == 1.0
        )
    )
    return {
        "target_tens_accuracy": target_accuracy,
        "target_full_accuracy": metrics["target"]["target_full_accuracy"],
        "strongest_control": strongest,
        "strongest_control_tens_accuracy": control_accuracy,
        "control_advantage": advantage,
        "parse_rate": metrics["target"]["parse_rate"],
        "passes": passes,
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
        raise SystemExit("refusing to overwrite development result or artifact")

    config = json.loads(args.config.read_text())
    paths = {
        "dataset": Path(config["dataset_config"]),
        "behavior": Path(config["behavior_result"]),
        "parent": Path(config["parent_result"]),
    }
    hash_keys = {
        "dataset": "dataset_config_sha256",
        "behavior": "behavior_result_sha256",
        "parent": "parent_result_sha256",
    }
    for name, path in paths.items():
        verify_sha256(path, config[hash_keys[name]])
    dataset_config = json.loads(paths["dataset"].read_text())
    behavior = json.loads(paths["behavior"].read_text())
    parent = json.loads(paths["parent"].read_text())
    if dataset_config.get("audit_authorized", False):
        raise SystemExit("development requires a sealed audit")
    if not behavior["passes"]:
        raise SystemExit("fresh behavior gate did not pass")
    if parent["passes"]:
        raise SystemExit("interaction recovery requires the parent non-pass")
    if config["intervention_scale"] != 1.0:
        raise SystemExit("interaction recovery scale must remain exactly 1.0")
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
        "development": sorted(
            {
                row.quartet_id
                for row in examples
                if row.split == "development"
            }
        ),
    }
    if value_sha256(split_ids["development"]) != config[
        "development_quartets_sha256"
    ]:
        raise SystemExit("development quartet hash mismatch")
    by_quartet = {
        quartet_id: {
            row.variant: row
            for row in examples
            if row.quartet_id == quartet_id
        }
        for quartet_id in split_ids["fit"] + split_ids["development"]
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
    for split in ("fit", "development"):
        _, context, contract = token_positions(
            tokenizer,
            rendered[split],
            label=f"Phase 6B {split}",
        )
        positions[split] = context
        if split == "development" and value_sha256(contract) != config[
            "development_token_region_contract_sha256"
        ]:
            raise SystemExit("development token-region contract mismatch")
    digit_token_ids = verify_decimal_digit_contract(
        tokenizer,
        rendered["development"]["carry_base"][0],
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

    fit_positions = positions["fit"]
    carry_fit_states = extract_at(
        states["fit"]["carry_base"],
        fit_positions,
    )
    carry_fit_delta = (
        extract_at(states["fit"]["carry_increment"], fit_positions)
        - carry_fit_states
    )
    control_fit_states = extract_at(
        states["fit"]["control_base"],
        fit_positions,
    )
    control_fit_delta = (
        extract_at(states["fit"]["control_increment"], fit_positions)
        - control_fit_states
    )
    interaction_fit_delta = carry_fit_delta - control_fit_delta
    fit_digits = torch.tensor(
        [row.operand_a % 10 for row in rows["fit"]["carry_base"]]
    )
    selected_architecture, cv_candidates = cross_validate(
        carry_fit_states,
        interaction_fit_delta,
        fit_digits,
        config=config,
    )
    state_rank = selected_architecture["state_rank"]
    transport_rank = selected_architecture["transport_rank"]
    ridge = selected_architecture["ridge"]
    interaction_design = build_design(
        carry_fit_states,
        interaction_fit_delta,
        fit_digits,
        state_rank=state_rank,
        max_transport_rank=transport_rank,
    )
    control_design = build_design(
        control_fit_states,
        control_fit_delta,
        fit_digits,
        state_rank=state_rank,
        max_transport_rank=transport_rank,
    )
    interaction_model = interaction_design.fit(
        transport_rank=transport_rank,
        ridge=ridge,
    )
    control_model = control_design.fit(
        transport_rank=transport_rank,
        ridge=ridge,
    )

    dev_positions = positions["development"]
    dev_base = states["development"]["carry_base"]
    dev_states = extract_at(dev_base, dev_positions)
    dev_digits = torch.tensor(
        [row.operand_a % 10 for row in rows["development"]["carry_base"]]
    )
    classes = list(interaction_model.classes)
    rotated_digits = torch.tensor(
        rotate_classes(dev_digits.tolist(), classes)
    )
    generator = torch.Generator().manual_seed(config["random_control_seed"])
    permutation = torch.randperm(dev_states.shape[0], generator=generator)
    target_vectors = interaction_model.predict(dev_states, dev_digits)
    no_carry_vectors = control_model.predict(dev_states, dev_digits)
    wrong_vectors = interaction_model.predict(dev_states, rotated_digits)
    shuffled_vectors = interaction_model.predict(
        dev_states[permutation],
        dev_digits,
    )
    target = one_token_sequences(dev_base, dev_positions, target_vectors)
    target_norms = sequence_norms(target)
    conditions = {
        "target": target,
        "matched_no_carry_norm_matched": norm_match_sequences(
            one_token_sequences(dev_base, dev_positions, no_carry_vectors),
            target_norms,
        ),
        "wrong_class_norm_matched": norm_match_sequences(
            one_token_sequences(dev_base, dev_positions, wrong_vectors),
            target_norms,
        ),
        "shuffled_recipient_norm_matched": norm_match_sequences(
            one_token_sequences(dev_base, dev_positions, shuffled_vectors),
            target_norms,
        ),
        "random_norm_matched": random_norm_matched_sequences(
            target,
            target_norms,
            seed=config["random_control_seed"] + 1,
        ),
    }
    metrics = {}
    for name, raw_delta in conditions.items():
        delta = tuple(
            value * config["intervention_scale"] for value in raw_delta
        )
        responses = generate_chunks(
            model,
            tokenizer,
            rendered["development"]["carry_base"],
            delta,
            hidden_state_index=hidden_index,
            batch_size=config["base_model_batch_size"],
            device=device,
        )
        metrics[name] = summarize_outputs(
            responses,
            rows["development"]["carry_base"],
            rows["development"]["carry_increment"],
            delta,
            dev_base,
        )
    development_gate = gate(metrics, config["development_rule"])

    args.artifact_output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            **model_tensors("interaction", interaction_model),
            **model_tensors("control", control_model),
        },
        str(args.artifact_output),
    )
    artifact_hash = hashlib.sha256(args.artifact_output.read_bytes()).hexdigest()
    report = {
        "schema_version": "oli.phase6b-qwen-carry-interaction-development/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "development_only",
        "model": model_config,
        "dataset_sha256": observed_dataset_hash,
        "behavior_result_sha256": config["behavior_result_sha256"],
        "parent_result_sha256": config["parent_result_sha256"],
        "eligible_fit_quartets": len(eligible_ids),
        "eligible_fit_quartets_sha256": config[
            "eligible_fit_quartets_sha256"
        ],
        "development_quartets_sha256": config[
            "development_quartets_sha256"
        ],
        "development_token_region_contract_sha256": config[
            "development_token_region_contract_sha256"
        ],
        "digit_token_ids": digit_token_ids,
        "hidden_state_index": hidden_index,
        "intervention_scale": config["intervention_scale"],
        "fit_cross_validation": {
            "folds": config["fit_cross_validation_folds"],
            "selection_metric": "normalized_mse",
            "selected": selected_architecture,
            "candidates": cv_candidates,
        },
        "metrics": metrics,
        "development_gate": development_gate,
        "passes": development_gate["passes"],
        "artifact": {
            "path": str(args.artifact_output),
            "sha256": artifact_hash,
            "width": int(carry_fit_states.shape[1]),
            "classes": classes,
        },
        "development_rule": config["development_rule"],
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "One-shot development test of a donor-free matched carry "
            "interaction. Passing would not establish an audit claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    print(f"wrote {args.artifact_output}")


if __name__ == "__main__":
    main()
