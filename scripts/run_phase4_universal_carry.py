#!/usr/bin/env python3
"""Select a class-invariant donor-free carry-context prototype."""

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
    summarize_outputs,
    value_sha256,
    verify_sha256,
)
from run_phase4_donor_free_prototypes import (
    differing_position,
    one_token_sequences,
    select_scale,
)
from safetensors.torch import load_file, save_file
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
    parser.add_argument("--prototype-output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()
    if args.output.exists() or args.prototype_output.exists():
        raise SystemExit("refusing to overwrite universal carry result or artifact")

    config = json.loads(args.config.read_text())
    dataset_path = Path(config["dataset_config"])
    parent_path = Path(config["parent_result"])
    prototype_path = Path(config["prototype_artifact"])
    verify_sha256(dataset_path, config["dataset_config_sha256"])
    verify_sha256(parent_path, config["parent_result_sha256"])
    verify_sha256(prototype_path, config["prototype_artifact_sha256"])
    dataset_config = json.loads(dataset_path.read_text())
    parent = json.loads(parent_path.read_text())
    if dataset_config.get("audit_authorized", False):
        raise SystemExit("universal carry selection requires a sealed audit")
    if parent["passes"]["carry_context"]:
        raise SystemExit("universal follow-up requires the class-specific non-pass")
    if not parent["passes"]["operand"]:
        raise SystemExit("upstream donor-free operand writer did not pass")

    examples = build_phase4_carry_quartets(
        **dataset_config["dataset"]["parameters"]
    )
    observed_dataset_hash = phase4_carry_sha256(examples)
    if observed_dataset_hash != dataset_config["dataset"]["sha256"]:
        raise SystemExit("Phase 4 dataset hash mismatch")
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
    base_rows = [by_quartet[quartet_id]["carry_base"] for quartet_id in selection_ids]
    target_rows = [
        by_quartet[quartet_id]["carry_increment"] for quartet_id in selection_ids
    ]

    model_config = dataset_config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_prompts = [
        render_prefilled_chat(
            tokenizer,
            row.prompt,
            assistant_prefix=dataset_config["assistant_prefix"],
        )
        for row in base_rows
    ]
    target_prompts = [
        render_prefilled_chat(
            tokenizer,
            row.prompt,
            assistant_prefix=dataset_config["assistant_prefix"],
        )
        for row in target_rows
    ]
    control_rows = [
        by_quartet[quartet_id]["control_base"] for quartet_id in selection_ids
    ]
    control_prompts = [
        render_prefilled_chat(
            tokenizer,
            row.prompt,
            assistant_prefix=dataset_config["assistant_prefix"],
        )
        for row in control_rows
    ]
    context_positions = []
    token_contract = []
    for base_prompt, target_prompt, control_prompt in zip(
        base_prompts,
        target_prompts,
        control_prompts,
        strict=True,
    ):
        base_ids = tokenizer(base_prompt)["input_ids"]
        target_ids = tokenizer(target_prompt)["input_ids"]
        control_ids = tokenizer(control_prompt)["input_ids"]
        changed_position = differing_position(
            base_ids,
            target_ids,
            label="selection operand",
        )
        context_position = differing_position(
            base_ids,
            control_ids,
            label="selection carry context",
        )
        context_positions.append(context_position)
        token_contract.append(
            [len(base_ids), changed_position, context_position]
        )
    if value_sha256(token_contract) != config["token_region_contract_sha256"]:
        raise SystemExit("selection token-region contract mismatch")
    digit_token_ids = verify_decimal_digit_contract(tokenizer, base_prompts[0])

    source = load_file(str(prototype_path))
    counts = source["class_counts"].float()
    weights = counts / counts.sum()
    carry_vector = (
        source["carry_context_delta"].float() * weights[:, None]
    ).sum(dim=0)
    no_carry_vector = (
        source["control_context_delta"].float() * weights[:, None]
    ).sum(dim=0)
    args.prototype_output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "carry_context_delta": carry_vector,
            "control_context_delta": no_carry_vector,
            "fit_class_counts": source["class_counts"],
        },
        str(args.prototype_output),
    )
    prototype_hash = hashlib.sha256(args.prototype_output.read_bytes()).hexdigest()

    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        torch_dtype=getattr(torch, args.dtype),
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    hidden_index = config["carry_context_hidden_state_index"]
    capture = ActivationCapture(model, tokenizer, device=device)
    started = time.perf_counter()
    base_states = capture.capture_sequences(
        base_prompts,
        hidden_state_indices=[hidden_index],
        batch_size=config["base_model_batch_size"],
    )[hidden_index].values
    carry_target = one_token_sequences(
        base_states,
        context_positions,
        carry_vector.repeat(len(base_rows), 1),
    )
    no_carry = norm_match_sequences(
        one_token_sequences(
            base_states,
            context_positions,
            no_carry_vector.repeat(len(base_rows), 1),
        ),
        sequence_norms(carry_target),
    )
    random = random_norm_matched_sequences(
        carry_target,
        sequence_norms(carry_target),
        seed=config["random_control_seed"],
    )
    conditions = {
        "target": carry_target,
        "matched_no_carry_norm_matched": no_carry,
        "random_norm_matched": random,
    }
    metrics = {}
    for scale in config["scales"]:
        scale_metrics = {}
        for name, raw_delta in conditions.items():
            delta = tuple(value * scale for value in raw_delta)
            responses = generate_chunks(
                model,
                tokenizer,
                base_prompts,
                delta,
                hidden_state_index=hidden_index,
                batch_size=config["base_model_batch_size"],
                device=device,
            )
            scale_metrics[name] = summarize_outputs(
                responses,
                base_rows,
                target_rows,
                delta,
                base_states,
            )
        metrics[str(scale)] = scale_metrics

    rule = config["selection_rule"]
    selected_scale, passes = select_scale(
        metrics,
        target="target",
        controls=("matched_no_carry_norm_matched", "random_norm_matched"),
        minimum_accuracy=rule["minimum_tens_accuracy"],
        minimum_advantage=rule["minimum_control_advantage"],
    )
    report = {
        "schema_version": "oli.phase4-universal-carry-selection/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "selection_only",
        "model": model_config,
        "dataset_sha256": observed_dataset_hash,
        "parent_result_sha256": config["parent_result_sha256"],
        "source_prototype_sha256": config["prototype_artifact_sha256"],
        "selection_quartets_sha256": config["selection_quartets_sha256"],
        "token_region_contract_sha256": config["token_region_contract_sha256"],
        "digit_token_ids": digit_token_ids,
        "hidden_state_index": hidden_index,
        "fit_class_counts": source["class_counts"].tolist(),
        "scales": config["scales"],
        "metrics": metrics,
        "selection": {"scale": selected_scale, "passes": passes},
        "passes": passes,
        "prototype": {
            "path": str(args.prototype_output),
            "sha256": prototype_hash,
            "width": int(carry_vector.shape[0]),
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
            "Selection-only universal donor-free carry prototype. Passing "
            "does not establish compact rank or development generalization."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    print(f"wrote {args.prototype_output}")


if __name__ == "__main__":
    main()
