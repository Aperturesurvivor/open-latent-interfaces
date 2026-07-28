#!/usr/bin/env python3
"""Select a prompt-local causal compiler for Phi's leading answer digit."""

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
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.causal_compiler import compile_local_margin_plan
from open_latent_interfaces.evaluation import norm_match, random_norm_matched
from open_latent_interfaces.phase2_data import balanced_counterfactual_results
from open_latent_interfaces.phase7_data import (
    build_phase7_carry_quartets,
    phase7_carry_sha256,
)
from open_latent_interfaces.prefill import verify_decimal_digit_contract


def evaluate_logits(
    logits: torch.Tensor,
    expected_ids: torch.Tensor,
    *,
    digit_token_ids: dict[int, int],
) -> dict[str, Any]:
    predicted = logits.argmax(dim=1)
    correct = int((predicted == expected_ids).sum())
    digit_ids = set(digit_token_ids.values())
    return {
        "n": expected_ids.numel(),
        "correct": correct,
        "accuracy": correct / expected_ids.numel(),
        "digit_token_rate": sum(int(token_id) in digit_ids for token_id in predicted)
        / expected_ids.numel(),
        "predicted_token_ids": predicted.tolist(),
    }


def wrong_leading_results(values: list[int]) -> list[int]:
    rows = []
    for value in values:
        digits = str(value)
        wrong = int(digits[0]) % 9 + 1
        rows.append(int(f"{wrong}{digits[1:]}"))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite causal-compiler result")

    config = json.loads(args.config.read_text())
    observed_runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if observed_runner_hash != config["runner_sha256"]:
        raise SystemExit("causal-compiler runner hash mismatch")
    compiler_module_path = Path(config["compiler_module"])
    verify_sha256(
        compiler_module_path,
        config["compiler_module_sha256"],
    )
    dataset_config_path = Path(config["dataset_config"])
    behavior_result_path = Path(config["behavior_result"])
    verify_sha256(dataset_config_path, config["dataset_config_sha256"])
    verify_sha256(behavior_result_path, config["behavior_result_sha256"])
    dataset_config = json.loads(dataset_config_path.read_text())
    behavior = json.loads(behavior_result_path.read_text())
    if dataset_config.get("audit_authorized", False):
        raise SystemExit("causal-compiler selection requires a sealed audit")
    if not behavior["passes"]:
        raise SystemExit("wide-distribution behavior gate did not pass")

    examples = build_phase7_carry_quartets(
        **dataset_config["dataset"]["parameters"]
    )
    dataset_hash = phase7_carry_sha256(examples)
    if dataset_hash != dataset_config["dataset"]["sha256"]:
        raise SystemExit("Phase 9 dataset hash mismatch")
    selection = [row for row in examples if row.split == "selection"]
    selection_ids = [row.example_id for row in selection]
    if value_sha256(selection_ids) != config["selection_examples_sha256"]:
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
    compile_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "hidden_state_index": config["hidden_state_index"],
        "candidate_token_ids": candidate_ids,
        "device": device,
        "batch_size": config["compiler_batch_size"],
    }
    target_plan = compile_local_margin_plan(
        prompts=target_prompts,
        target_token_ids=target_expected,
        **compile_kwargs,
    )
    identity_plan = compile_local_margin_plan(
        prompts=identity_prompts,
        target_token_ids=identity_expected,
        **compile_kwargs,
    )
    wrong_plan = compile_local_margin_plan(
        prompts=target_prompts,
        target_token_ids=wrong_expected,
        **compile_kwargs,
    )

    def intervened(
        prompts: list[str],
        deltas: torch.Tensor,
    ) -> torch.Tensor:
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
    rows = {}
    passing_margins = []
    for desired_margin in config["desired_margins"]:
        target_delta = target_plan.deltas(
            desired_margin=desired_margin,
            max_relative_norm=config["norm_cap"],
        )
        identity_delta = identity_plan.deltas(
            desired_margin=desired_margin,
            max_relative_norm=config["norm_cap"],
        )
        norms = target_delta.norm(dim=1)
        wrong_delta = norm_match(
            wrong_plan.deltas(
                desired_margin=desired_margin,
                max_relative_norm=config["norm_cap"],
            ),
            norms,
        )
        random_delta = random_norm_matched(
            tuple(target_delta.shape),
            norms,
            seed=config["random_control_seed"],
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
            (norms / target_plan.recipient_states.norm(dim=1)).mean()
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
        rows[str(desired_margin)] = metrics
        if passes:
            passing_margins.append(float(desired_margin))

    if passing_margins:
        selected_margin = min(
            passing_margins,
            key=lambda margin: (
                rows[str(margin)]["gate"]["mean_target_relative_norm"],
                margin,
            ),
        )
    else:
        selected_margin = max(
            config["desired_margins"],
            key=lambda margin: (
                rows[str(margin)]["target"]["accuracy"],
                rows[str(margin)]["gate"]["control_advantage"],
                -rows[str(margin)]["gate"]["mean_target_relative_norm"],
            ),
        )
    report = {
        "schema_version": "oli.phase9-leading-causal-compiler-selection/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "selection_only",
        "model": model_config,
        "dataset_sha256": dataset_hash,
        "behavior_result_sha256": config["behavior_result_sha256"],
        "selection_examples_sha256": config["selection_examples_sha256"],
        "selection_targets_sha256": config["selection_targets_sha256"],
        "hidden_state_index": config["hidden_state_index"],
        "candidate_token_ids": candidate_ids.tolist(),
        "desired_margins": config["desired_margins"],
        "metrics": rows,
        "selection": {
            "desired_margin": selected_margin,
            "passes": bool(passing_margins),
        },
        "base": {
            "target": evaluate_logits(
                target_plan.base_logits,
                target_expected,
                digit_token_ids=digit_token_ids,
            ),
            "identity": evaluate_logits(
                identity_plan.base_logits,
                identity_expected,
                digit_token_ids=digit_token_ids,
            ),
        },
        "plan_diagnostics": {
            "target_hard_gate_count": int(target_plan.hard_gate.sum()),
            "identity_hard_gate_count": int(identity_plan.hard_gate.sum()),
            "mean_target_base_digit_margin": float(
                target_plan.current_margins.mean()
            ),
            "mean_target_gradient_norm": float(
                target_plan.margin_gradients.norm(dim=1).mean()
            ),
        },
        "selection_rule": rule,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "runner_sha256": config["runner_sha256"],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Selection-only prompt-local hidden-state compiler. It differentiates "
            "the frozen model suffix and does not establish a cognitive feature, "
            "model-general interface, or audit claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
