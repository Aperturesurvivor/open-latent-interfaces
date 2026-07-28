#!/usr/bin/env python3
"""Select a prompt-local leading-digit compiler for frozen SmolLM2."""

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

from open_latent_interfaces.causal_compiler import compile_local_margin_plan
from open_latent_interfaces.evaluation import norm_match, random_norm_matched
from open_latent_interfaces.phase2_data import balanced_counterfactual_results
from open_latent_interfaces.phase13_data import (
    build_phase13_examples,
    phase13_sha256,
)
from open_latent_interfaces.prefill import verify_decimal_digit_contract


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
        raise SystemExit("refusing to overwrite leading-compiler result")

    config = json.loads(args.config.read_text())
    if str(args.output) != config["output"]:
        raise SystemExit("leading-compiler output differs from frozen path")
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if runner_hash != config["runner_sha256"]:
        raise SystemExit("leading-compiler runner hash mismatch")
    for dependency, expected_hash in config["code_dependencies"].items():
        verify_sha256(Path(dependency), expected_hash)

    dataset_path = Path(config["dataset_config"])
    verify_sha256(dataset_path, config["dataset_config_sha256"])
    dataset_config = json.loads(dataset_path.read_text())
    if dataset_config.get("audit_authorized", False):
        raise SystemExit("compiler selection may not use sealed audit data")
    onboarding_path = Path(config["onboarding_result"])
    verify_sha256(onboarding_path, config["onboarding_result_sha256"])
    if json.loads(onboarding_path.read_text()).get("passes") is not True:
        raise SystemExit("model onboarding did not pass")
    reader_path = Path(config["reader_result"])
    verify_sha256(reader_path, config["reader_result_sha256"])
    reader = json.loads(reader_path.read_text())
    if reader.get("passes") is not True:
        raise SystemExit("operand-reader selection did not pass")

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
    wrong_targets = wrong_leading_results(targets)

    model_config = dataset_config["model"]
    if model_config != config["model"]:
        raise SystemExit("compiler model differs from frozen model")
    if reader.get("model") != model_config:
        raise SystemExit("reader result used a different model")
    if reader.get("dataset_sha256") != dataset_hash:
        raise SystemExit("reader result used a different dataset")

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
    prompt_contract = {
        "target": target_prompts,
        "identity": identity_prompts,
        "target_expected": target_expected.tolist(),
        "identity_expected": identity_expected.tolist(),
        "wrong_expected": wrong_expected.tolist(),
    }
    if value_sha256(prompt_contract) != config["prompt_contract_sha256"]:
        raise SystemExit("compiler prompt contract hash mismatch")

    candidate_indices = config["candidate_hidden_state_indices"]
    if candidate_indices != sorted(set(candidate_indices)):
        raise SystemExit("candidate hidden-state indices must be unique and sorted")
    if min(candidate_indices) < 1:
        raise SystemExit("candidate hidden-state indices must be block outputs")
    if max(candidate_indices) >= config["expected_hidden_state_count"]:
        raise SystemExit("candidate boundary exceeds frozen model")

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
    results: dict[str, Any] = {}
    passing: list[dict[str, float | int]] = []
    for hidden_state_index in candidate_indices:
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
            boundary: int = hidden_state_index,
        ) -> torch.Tensor:
            return predict_with_delta(
                model,
                tokenizer,
                prompts,
                deltas,
                hidden_state_index=boundary,
                batch_size=config["base_model_batch_size"],
                device=device,
            )

        boundary_rows: dict[str, Any] = {}
        for norm_cap_index, norm_cap in enumerate(config["norm_caps"]):
            for margin_index, desired_margin in enumerate(
                config["desired_margins"]
            ):
                key = f"margin={desired_margin},cap={norm_cap}"
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
                        + 10_000 * hidden_state_index
                        + 100 * norm_cap_index
                        + margin_index
                    ),
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
                metrics["gate"] = {
                    "strongest_control_accuracy": strongest_control,
                    "control_advantage": advantage,
                    "mean_target_relative_norm": mean_relative_norm,
                    "passes": passes,
                }
                boundary_rows[key] = metrics
                if passes:
                    passing.append(
                        {
                            "hidden_state_index": hidden_state_index,
                            "desired_margin": float(desired_margin),
                            "norm_cap": float(norm_cap),
                            "mean_target_relative_norm": mean_relative_norm,
                        }
                    )
        results[str(hidden_state_index)] = {
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
            "candidates": boundary_rows,
        }

    if passing:
        selected = min(
            passing,
            key=lambda row: (
                row["mean_target_relative_norm"],
                row["norm_cap"],
                row["desired_margin"],
                row["hidden_state_index"],
            ),
        )
    else:
        scored = []
        for hidden_state_index, boundary in results.items():
            for key, metrics in boundary["candidates"].items():
                margin_text, cap_text = key.split(",")
                scored.append(
                    {
                        "hidden_state_index": int(hidden_state_index),
                        "desired_margin": float(margin_text.split("=")[1]),
                        "norm_cap": float(cap_text.split("=")[1]),
                        "mean_target_relative_norm": metrics["gate"][
                            "mean_target_relative_norm"
                        ],
                        "target_accuracy": metrics["target"]["accuracy"],
                        "control_advantage": metrics["gate"][
                            "control_advantage"
                        ],
                    }
                )
        selected = max(
            scored,
            key=lambda row: (
                row["target_accuracy"],
                row["control_advantage"],
                -row["mean_target_relative_norm"],
            ),
        )

    report = {
        "schema_version": "oli.phase13-smollm2-leading-compiler-selection/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "selection_only",
        "model": model_config,
        "dataset_sha256": dataset_hash,
        "reader_result_sha256": config["reader_result_sha256"],
        "selection_examples_sha256": config["selection_examples_sha256"],
        "selection_targets_sha256": config["selection_targets_sha256"],
        "candidate_hidden_state_indices": candidate_indices,
        "desired_margins": config["desired_margins"],
        "norm_caps": config["norm_caps"],
        "iterations": 1,
        "candidate_token_ids": candidate_ids.tolist(),
        "metrics": results,
        "selection": {
            **selected,
            "iterations": 1,
            "passes": bool(passing),
        },
        "selection_rule": rule,
        "selection_order": [
            "minimum mean target relative norm",
            "minimum norm cap",
            "minimum desired margin",
            "earliest hidden-state index",
        ],
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
            "Selection-only prompt-local SmolLM2 leading-digit compiler on "
            "fresh exposed data. No suffix, integrated, development, audit, "
            "cognitive-feature, or model-general claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
