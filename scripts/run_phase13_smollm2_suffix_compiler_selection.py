#!/usr/bin/env python3
"""Select prompt-local tens and ones compilers for frozen SmolLM2."""

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
from run_phase9_leading_causal_compiler import evaluate_logits
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.causal_compiler import compile_local_margin_plan
from open_latent_interfaces.evaluation import norm_match, random_norm_matched
from open_latent_interfaces.phase2_data import balanced_counterfactual_results
from open_latent_interfaces.phase13_data import (
    build_phase13_examples,
    phase13_sha256,
)
from open_latent_interfaces.prefill import verify_decimal_digit_contract


def wrong_position_results(
    values: list[int],
    *,
    position: int,
) -> list[int]:
    wrong = []
    for value in values:
        digits = list(str(value))
        if len(digits) != 3 or position not in (1, 2):
            raise ValueError("suffix control requires a three-digit value")
        digits[position] = str((int(digits[position]) + 1) % 10)
        wrong.append(int("".join(digits)))
    return wrong


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite suffix-compiler result")

    config = json.loads(args.config.read_text())
    if str(args.output) != config["output"]:
        raise SystemExit("suffix-compiler output differs from frozen path")
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if runner_hash != config["runner_sha256"]:
        raise SystemExit("suffix-compiler runner hash mismatch")
    for dependency, expected_hash in config["code_dependencies"].items():
        verify_sha256(Path(dependency), expected_hash)

    dataset_path = Path(config["dataset_config"])
    verify_sha256(dataset_path, config["dataset_config_sha256"])
    dataset_config = json.loads(dataset_path.read_text())
    if dataset_config.get("audit_authorized", False):
        raise SystemExit("suffix selection may not use sealed audit data")
    for result_name in ("onboarding_result", "reader_result", "leading_result"):
        result_path = Path(config[result_name])
        verify_sha256(result_path, config[f"{result_name}_sha256"])
        result = json.loads(result_path.read_text())
        if result.get("passes", result.get("selection", {}).get("passes")) is not True:
            raise SystemExit(f"{result_name} did not pass")
    prototype_path = Path(config["prototype_result"])
    verify_sha256(prototype_path, config["prototype_result_sha256"])
    prototype_result = json.loads(prototype_path.read_text())
    if prototype_result.get("passes") is not False:
        raise SystemExit("suffix fallback requires a preserved prototype nonpass")

    examples = build_phase13_examples(
        **dataset_config["dataset"]["parameters"]
    )
    dataset_hash = phase13_sha256(examples)
    if dataset_hash != dataset_config["dataset"]["sha256"]:
        raise SystemExit("Phase 13 dataset hash mismatch")
    selection = [row for row in examples if row.split == "selection"]
    if value_sha256([row.example_id for row in selection]) != config[
        "selection_examples_sha256"
    ]:
        raise SystemExit("selection example hash mismatch")
    targets = balanced_counterfactual_results(selection)
    if value_list_sha256(targets) != config["selection_targets_sha256"]:
        raise SystemExit("selection target hash mismatch")
    originals = [row.result for row in selection]

    model_config = dataset_config["model"]
    if model_config != config["model"]:
        raise SystemExit("suffix-compiler model differs from frozen model")
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
    if value_sha256(rendered) != config["rendered_prompts_sha256"]:
        raise SystemExit("rendered prompt hash mismatch")
    digit_token_ids = verify_decimal_digit_contract(tokenizer, rendered[0])
    if value_sha256(digit_token_ids) != config["digit_token_ids_sha256"]:
        raise SystemExit("digit-token map hash mismatch")
    candidate_ids = torch.tensor(
        [digit_token_ids[digit] for digit in range(10)],
        dtype=torch.long,
    )

    prompt_contract: dict[str, Any] = {}
    for position in (1, 2):
        wrong = wrong_position_results(targets, position=position)
        prompt_contract[str(position)] = {
            "target": prefix_prompts(rendered, targets, position=position),
            "identity": prefix_prompts(rendered, originals, position=position),
            "target_expected": [
                digit_token_ids[int(str(value)[position])] for value in targets
            ],
            "identity_expected": [
                digit_token_ids[int(str(value)[position])] for value in originals
            ],
            "wrong_expected": [
                digit_token_ids[int(str(value)[position])] for value in wrong
            ],
        }
    if value_sha256(prompt_contract) != config["prompt_contract_sha256"]:
        raise SystemExit("suffix-compiler prompt contract mismatch")

    hidden_state_index = config["hidden_state_index"]
    if not 1 <= hidden_state_index < config["expected_hidden_state_count"]:
        raise SystemExit("suffix boundary exceeds frozen model")
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        dtype=getattr(torch, args.dtype),
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("base model parameters must remain frozen")

    started = time.perf_counter()
    rule = config["selection_rule"]
    position_results: dict[str, Any] = {}
    all_pass = True
    for position in (1, 2):
        contract = prompt_contract[str(position)]
        target_prompts = contract["target"]
        identity_prompts = contract["identity"]
        target_expected = torch.tensor(contract["target_expected"])
        identity_expected = torch.tensor(contract["identity_expected"])
        wrong_expected = torch.tensor(contract["wrong_expected"])
        compile_kwargs = {
            "model": model,
            "tokenizer": tokenizer,
            "hidden_state_index": hidden_state_index,
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
                hidden_state_index=hidden_state_index,
                batch_size=config["base_model_batch_size"],
                device=device,
            )

        candidates = []
        passing = []
        for norm_cap_index, norm_cap in enumerate(config["norm_caps"]):
            for margin_index, desired_margin in enumerate(
                config["desired_margins"]
            ):
                target_delta = target_plan.deltas(
                    desired_margin=desired_margin,
                    max_relative_norm=norm_cap,
                )
                identity_delta = identity_plan.deltas(
                    desired_margin=desired_margin,
                    max_relative_norm=norm_cap,
                )
                norms = target_delta.norm(dim=1)
                wrong_delta = norm_match(
                    wrong_plan.deltas(
                        desired_margin=desired_margin,
                        max_relative_norm=norm_cap,
                    ),
                    norms,
                )
                random_delta = random_norm_matched(
                    tuple(target_delta.shape),
                    norms,
                    seed=(
                        config["random_control_seed"]
                        + 10_000 * position
                        + 100 * norm_cap_index
                        + margin_index
                    ),
                )
                target_logits = intervened(target_prompts, target_delta)
                identity_logits = intervened(
                    identity_prompts,
                    identity_delta,
                )
                wrong_logits = intervened(target_prompts, wrong_delta)
                random_logits = intervened(target_prompts, random_delta)
                metrics = {
                    "target": evaluate_logits(
                        target_logits,
                        target_expected,
                        digit_token_ids=digit_token_ids,
                    ),
                    "identity": evaluate_logits(
                        identity_logits,
                        identity_expected,
                        digit_token_ids=digit_token_ids,
                    ),
                    "wrong_digit_norm_matched": evaluate_logits(
                        wrong_logits,
                        target_expected,
                        digit_token_ids=digit_token_ids,
                    ),
                    "wrong_digit_alignment": evaluate_logits(
                        wrong_logits,
                        wrong_expected,
                        digit_token_ids=digit_token_ids,
                    ),
                    "random_norm_matched": evaluate_logits(
                        random_logits,
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
                        / target_plan.recipient_states.norm(dim=1).clamp_min(
                            1e-12
                        )
                    ).mean()
                )
                passes = (
                    metrics["target"]["accuracy"]
                    >= rule["minimum_target_accuracy"]
                    and metrics["identity"]["accuracy"]
                    >= rule["minimum_identity_accuracy"]
                    and advantage >= rule["minimum_control_advantage"]
                    and mean_relative_norm
                    <= rule["maximum_mean_relative_norm"]
                    and (
                        not rule["require_digit_token_rate"]
                        or metrics["target"]["digit_token_rate"] == 1.0
                    )
                )
                row = {
                    "desired_margin": float(desired_margin),
                    "norm_cap": float(norm_cap),
                    "metrics": metrics,
                    "gate": {
                        "strongest_control_accuracy": strongest_control,
                        "control_advantage": advantage,
                        "mean_target_relative_norm": mean_relative_norm,
                        "passes": passes,
                    },
                }
                candidates.append(row)
                if passes:
                    passing.append(row)
        selected = (
            min(
                passing,
                key=lambda row: (
                    row["gate"]["mean_target_relative_norm"],
                    row["norm_cap"],
                    row["desired_margin"],
                ),
            )
            if passing
            else max(
                candidates,
                key=lambda row: (
                    row["metrics"]["target"]["accuracy"],
                    row["metrics"]["identity"]["accuracy"],
                    row["gate"]["control_advantage"],
                    -row["gate"]["mean_target_relative_norm"],
                ),
            )
        )
        position_passes = bool(passing)
        all_pass = all_pass and position_passes
        position_results[str(position)] = {
            "passes": position_passes,
            "selection": {
                "desired_margin": selected["desired_margin"],
                "norm_cap": selected["norm_cap"],
                "iterations": 1,
                **selected["gate"],
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
            "candidates": candidates,
        }

    report = {
        "schema_version": "oli.phase13-smollm2-suffix-compiler-selection/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "selection_only",
        "model": model_config,
        "dataset_sha256": dataset_hash,
        "hidden_state_index": hidden_state_index,
        "prototype_result_sha256": config["prototype_result_sha256"],
        "selection_examples_sha256": config["selection_examples_sha256"],
        "selection_targets_sha256": config["selection_targets_sha256"],
        "candidate_token_ids": candidate_ids.tolist(),
        "desired_margins": config["desired_margins"],
        "norm_caps": config["norm_caps"],
        "positions": position_results,
        "passes": all_pass,
        "selection_rule": rule,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "runner_sha256": runner_hash,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Selection-only prompt-local SmolLM2 tens/ones compilers after "
            "a preserved native-prototype nonpass. No integrated, "
            "development, audit, cognitive-feature, or model-general claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
