#!/usr/bin/env python3
"""Execute the authorized one-shot Phi closed-loop audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from run_phase3_closed_loop_development import (
    advancement_gate,
    evaluate_condition,
)
from run_phase3_native_boundary import (
    render_examples,
    value_list_sha256,
    verify_sha256,
)
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.phase2_data import balanced_counterfactual_results
from open_latent_interfaces.phase3_data import (
    build_phase3_additions,
    phase3_addition_sha256,
)
from open_latent_interfaces.prefill import verify_decimal_digit_contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    if not config.get("audit_authorized", False):
        raise SystemExit("audit is sealed")
    if config.get("maximum_audit_runs") != 1:
        raise SystemExit("audit config must authorize exactly one run")
    if config.get("evaluation_split") != "audit":
        raise SystemExit("authorized config must evaluate only the audit split")
    if str(args.output) != config.get("audit_output"):
        raise SystemExit("output path differs from frozen audit path")
    if args.output.exists():
        raise SystemExit("refusing to overwrite an existing audit result")

    runner_path = Path(__file__)
    engine_path = Path(config["engine"])
    verify_sha256(runner_path, config["runner_sha256"])
    verify_sha256(engine_path, config["engine_sha256"])
    source_paths = {
        "dataset": Path(config["dataset_config"]),
        "basis": Path(config["basis"]),
        "suffix_result": Path(config["suffix_result"]),
        "suffix_prototype": Path(config["suffix_prototype"]),
        "leading_result": Path(config["leading_result"]),
        "leading_prototype": Path(config["leading_prototype"]),
        "development_config": Path(config["development_config"]),
        "development_result": Path(config["development_result"]),
    }
    for name, path in source_paths.items():
        verify_sha256(path, config[f"{name}_sha256"])

    development_config = json.loads(
        source_paths["development_config"].read_text()
    )
    development_result = json.loads(
        source_paths["development_result"].read_text()
    )
    if (
        development_result["config_sha256"]
        != config["development_config_sha256"]
    ):
        raise SystemExit("development result/config provenance mismatch")
    if not development_result["passes"] or not all(
        development_result["gate"]["checks"].values()
    ):
        raise SystemExit("development result did not pass every gate")
    locked_keys = (
        "basis_sha256",
        "dataset_sha256",
        "hidden_state_indices",
        "leading_prototype_sha256",
        "leading_result_sha256",
        "norm_cap",
        "random_control_seed",
        "scales",
        "suffix_prototype_sha256",
        "suffix_result_sha256",
        "gate",
    )
    for key in locked_keys:
        if config[key] != development_config[key]:
            raise SystemExit(f"audit changed frozen development field: {key}")

    dataset_config = json.loads(source_paths["dataset"].read_text())
    suffix_result = json.loads(source_paths["suffix_result"].read_text())
    leading_result = json.loads(source_paths["leading_result"].read_text())
    if not leading_result["passes"] or not all(
        suffix_result["positions"][str(position)]["passes"]
        for position in (1, 2)
    ):
        raise SystemExit("controller source no longer passes selection")
    examples = build_phase3_additions(**dataset_config["dataset"]["parameters"])
    observed_hash = phase3_addition_sha256(examples)
    if observed_hash != dataset_config["dataset"]["sha256"]:
        raise SystemExit("Phase 3 dataset hash mismatch")
    audit_examples = [example for example in examples if example.split == "audit"]
    targets = balanced_counterfactual_results(audit_examples)
    if value_list_sha256(targets) != config["audit_targets_sha256"]:
        raise SystemExit("audit target hash mismatch")

    basis_artifact = load_file(str(source_paths["basis"]))
    suffix_artifact = load_file(str(source_paths["suffix_prototype"]))
    leading_artifact = load_file(str(source_paths["leading_prototype"]))
    bases = {
        0: basis_artifact["leading_basis"][:32].float(),
        1: basis_artifact["suffix_basis"][:32].float(),
        2: basis_artifact["suffix_basis"][:32].float(),
    }
    prototypes = {
        0: leading_artifact["leading_digit"].float(),
        1: suffix_artifact["position_1_digit"].float(),
        2: suffix_artifact["position_2_digit"].float(),
    }

    device = torch.device(args.device)
    model_config = dataset_config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rendered = render_examples(
        tokenizer,
        audit_examples,
        assistant_prefix=dataset_config["assistant_prefix"],
    )
    digit_token_ids = verify_decimal_digit_contract(tokenizer, rendered[0])
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

    condition_names = (
        "base",
        "donor_free_targeted",
        "identity_hard_gated",
        "wrong_digit_norm_matched",
        "shuffled_target_norm_matched",
        "random_subspace_norm_matched",
    )
    conditions = {
        condition: evaluate_condition(
            condition,
            model,
            tokenizer,
            capture,
            examples=audit_examples,
            targets=targets,
            rendered_prompts=rendered,
            bases=bases,
            prototypes=prototypes,
            digit_token_ids=digit_token_ids,
            config=config,
            device=device,
            condition_index=index,
        )
        for index, condition in enumerate(condition_names)
    }
    passes, gate_details = advancement_gate(conditions, config)
    report = {
        "schema_version": "oli.phase3-closed-loop-audit/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "one_shot_audit",
        "model": model_config,
        "dataset": {
            "sha256": observed_hash,
            "split": "audit",
            "examples": len(audit_examples),
        },
        "authorization": {
            "maximum_audit_runs": 1,
            "audit_output": config["audit_output"],
            "runner_sha256": config["runner_sha256"],
            "engine_sha256": config["engine_sha256"],
        },
        "development_evidence": {
            "config_sha256": config["development_config_sha256"],
            "result_sha256": config["development_result_sha256"],
        },
        "sources": {
            f"{name}_sha256": config[f"{name}_sha256"]
            for name in (
                "basis",
                "suffix_result",
                "suffix_prototype",
                "leading_result",
                "leading_prototype",
            )
        },
        "controller": {
            "hidden_state_indices": config["hidden_state_indices"],
            "ranks": {"0": 32, "1": 32, "2": 32},
            "scales": config["scales"],
            "norm_cap": config["norm_cap"],
            "hard_gate": "exact zero delta when base argmax is requested digit",
        },
        "audit_targets_sha256": config["audit_targets_sha256"],
        "conditions": conditions,
        "gate": {
            "thresholds": config["gate"],
            **gate_details,
        },
        "passes": passes,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "One-shot held-out audit of a donor-free answer-channel interface "
            "in one additional model family."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
