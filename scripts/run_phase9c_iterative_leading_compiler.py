#!/usr/bin/env python3
"""Select bounded relinearization depth for Phi's leading-token compiler."""

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
from run_phase4_carry_sequence_boundary import value_sha256
from run_phase9_leading_causal_compiler import (
    evaluate_logits,
    wrong_leading_results,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.causal_compiler import (
    compile_iterative_margin_deltas,
)
from open_latent_interfaces.evaluation import norm_match, random_norm_matched
from open_latent_interfaces.phase2_data import balanced_counterfactual_results
from open_latent_interfaces.phase7_data import (
    build_phase7_carry_quartets,
    phase7_carry_sha256,
)
from open_latent_interfaces.prefill import verify_decimal_digit_contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite iterative-compiler result")

    config = json.loads(args.config.read_text())
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if runner_hash != config["runner_sha256"]:
        raise SystemExit("iterative-compiler runner hash mismatch")
    verify_sha256(
        Path(config["compiler_module"]),
        config["compiler_module_sha256"],
    )
    dataset_path = Path(config["dataset_config"])
    behavior_path = Path(config["behavior_result"])
    verify_sha256(dataset_path, config["dataset_config_sha256"])
    verify_sha256(behavior_path, config["behavior_result_sha256"])
    dataset_config = json.loads(dataset_path.read_text())
    behavior = json.loads(behavior_path.read_text())
    if dataset_config.get("audit_authorized", False):
        raise SystemExit("iterative selection requires a sealed audit")
    if not behavior["passes"]:
        raise SystemExit("wide-distribution behavior gate did not pass")

    examples = build_phase7_carry_quartets(
        **dataset_config["dataset"]["parameters"]
    )
    dataset_hash = phase7_carry_sha256(examples)
    if dataset_hash != dataset_config["dataset"]["sha256"]:
        raise SystemExit("Phase 9 dataset hash mismatch")
    selection = [row for row in examples if row.split == "selection"]
    if value_sha256([row.example_id for row in selection]) != config[
        "selection_examples_sha256"
    ]:
        raise SystemExit("selection example hash mismatch")
    targets = balanced_counterfactual_results(selection)
    if value_list_sha256(targets) != config["selection_targets_sha256"]:
        raise SystemExit("selection target hash mismatch")
    originals = [row.result for row in selection]
    wrong_targets = wrong_leading_results(targets)

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
        selection,
        assistant_prefix=dataset_config["assistant_prefix"],
    )
    digit_token_ids = verify_decimal_digit_contract(tokenizer, rendered[0])
    candidate_ids = torch.tensor(
        [digit_token_ids[digit] for digit in range(10)],
        dtype=torch.long,
    )
    target_prompts = prefix_prompts(rendered, targets, position=0)
    identity_prompts = prefix_prompts(rendered, originals, position=0)
    target_expected = torch.tensor(
        [digit_token_ids[int(str(value)[0])] for value in targets]
    )
    identity_expected = torch.tensor(
        [digit_token_ids[int(str(value)[0])] for value in originals]
    )
    wrong_expected = torch.tensor(
        [digit_token_ids[int(str(value)[0])] for value in wrong_targets]
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
    started = time.perf_counter()
    common = {
        "model": model,
        "tokenizer": tokenizer,
        "hidden_state_index": config["hidden_state_index"],
        "candidate_token_ids": candidate_ids,
        "desired_margin": config["desired_margin"],
        "max_relative_norm": config["norm_cap"],
        "device": device,
        "batch_size": config["compiler_batch_size"],
    }
    target_trace = compile_iterative_margin_deltas(
        prompts=target_prompts,
        target_token_ids=target_expected,
        iterations=config["maximum_iterations"],
        **common,
    )
    wrong_trace = compile_iterative_margin_deltas(
        prompts=target_prompts,
        target_token_ids=wrong_expected,
        iterations=config["maximum_iterations"],
        **common,
    )
    identity_trace = compile_iterative_margin_deltas(
        prompts=identity_prompts,
        target_token_ids=identity_expected,
        iterations=1,
        **common,
    )
    identity_delta = identity_trace.cumulative_deltas[0]

    def intervened(prompts: list[str], deltas: torch.Tensor) -> torch.Tensor:
        return predict_with_delta(
            model,
            tokenizer,
            prompts,
            deltas,
            hidden_state_index=config["hidden_state_index"],
            batch_size=config["base_model_batch_size"],
            device=device,
        )

    rule = config["selection_rule"]
    rows: dict[str, Any] = {}
    passing_iterations = []
    for index, target_delta in enumerate(target_trace.cumulative_deltas):
        iteration = index + 1
        norms = target_delta.norm(dim=1)
        wrong_delta = norm_match(
            wrong_trace.cumulative_deltas[index],
            norms,
        )
        random_delta = random_norm_matched(
            tuple(target_delta.shape),
            norms,
            seed=config["random_control_seed"] + iteration,
        )
        metrics = {
            "target": evaluate_logits(
                intervened(target_prompts, target_delta),
                target_expected,
                digit_token_ids=digit_token_ids,
            ),
            "identity": evaluate_logits(
                intervened(identity_prompts, identity_delta),
                identity_expected,
                digit_token_ids=digit_token_ids,
            ),
            "wrong_digit_norm_matched": evaluate_logits(
                intervened(target_prompts, wrong_delta),
                target_expected,
                digit_token_ids=digit_token_ids,
            ),
            "random_norm_matched": evaluate_logits(
                intervened(target_prompts, random_delta),
                target_expected,
                digit_token_ids=digit_token_ids,
            ),
        }
        strongest_control = max(
            metrics["wrong_digit_norm_matched"]["accuracy"],
            metrics["random_norm_matched"]["accuracy"],
        )
        advantage = metrics["target"]["accuracy"] - strongest_control
        mean_relative_norm = float(
            (
                norms
                / target_trace.base_recipient_states.norm(dim=1).clamp_min(1e-12)
            ).mean()
        )
        passes = (
            metrics["target"]["accuracy"] >= rule["minimum_target_accuracy"]
            and metrics["identity"]["accuracy"] >= rule["minimum_identity_accuracy"]
            and advantage >= rule["minimum_control_advantage"]
            and mean_relative_norm <= rule["maximum_mean_relative_norm"]
            and (
                not rule["require_digit_token_rate"]
                or metrics["target"]["digit_token_rate"] == 1.0
            )
        )
        metrics["gate"] = {
            "strongest_control_accuracy": strongest_control,
            "control_advantage": advantage,
            "mean_target_relative_norm": mean_relative_norm,
            "passes": passes,
        }
        rows[str(iteration)] = metrics
        if passes:
            passing_iterations.append(iteration)

    selected_iteration = (
        min(passing_iterations)
        if passing_iterations
        else max(
            range(1, config["maximum_iterations"] + 1),
            key=lambda iteration: (
                rows[str(iteration)]["target"]["accuracy"],
                rows[str(iteration)]["gate"]["control_advantage"],
                -rows[str(iteration)]["gate"]["mean_target_relative_norm"],
            ),
        )
    )
    zero_target = torch.zeros_like(target_trace.base_recipient_states)
    zero_identity = torch.zeros_like(identity_trace.base_recipient_states)
    report = {
        "schema_version": "oli.phase9c-iterative-leading-compiler-selection/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "selection_only",
        "model": model_config,
        "dataset_sha256": dataset_hash,
        "behavior_result_sha256": config["behavior_result_sha256"],
        "selection_examples_sha256": config["selection_examples_sha256"],
        "selection_targets_sha256": config["selection_targets_sha256"],
        "hidden_state_index": config["hidden_state_index"],
        "desired_margin": config["desired_margin"],
        "maximum_iterations": config["maximum_iterations"],
        "metrics": rows,
        "selection": {
            "iterations": selected_iteration,
            "passes": bool(passing_iterations),
        },
        "base": {
            "target": evaluate_logits(
                intervened(target_prompts, zero_target),
                target_expected,
                digit_token_ids=digit_token_ids,
            ),
            "identity": evaluate_logits(
                intervened(identity_prompts, zero_identity),
                identity_expected,
                digit_token_ids=digit_token_ids,
            ),
        },
        "selection_rule": rule,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "runner_sha256": runner_hash,
        "compiler_module_sha256": config["compiler_module_sha256"],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Selection-only bounded iterative hidden-state compiler on an "
            "already exposed split. No audit or model-general claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
