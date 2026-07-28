#!/usr/bin/env python3
"""Evaluate the Phi reader-compute-hybrid-writer graft on exposed development."""

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
from run_phase3_closed_loop_development import wrong_all_digits
from run_phase3_native_boundary import predict_with_delta, verify_sha256
from run_phase3_prototype_selection import prototype_delta, scale_and_gate
from run_phase4_carry_sequence_boundary import value_sha256
from run_phase8_latent_graft import group_predictions, true_result_metrics
from run_phase8_operand_reader_selection import (
    flatten_states_and_labels,
    reader_metrics,
    render_and_locate,
)
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.capability import parse_first_integer
from open_latent_interfaces.causal_compiler import (
    IterativeMarginTrace,
    compile_iterative_margin_deltas,
)
from open_latent_interfaces.evaluation import norm_match, random_norm_matched
from open_latent_interfaces.operand_reader import NearestCentroidDigitReader
from open_latent_interfaces.phase7_data import (
    build_phase7_carry_quartets,
    phase7_carry_sha256,
)
from open_latent_interfaces.prefill import verify_decimal_digit_contract


def evaluate_hybrid_condition(
    condition: str,
    model: Any,
    tokenizer: Any,
    capture: ActivationCapture,
    *,
    examples: list[Any],
    targets: list[int],
    originals: list[int],
    rendered_prompts: list[str],
    leading_delta: torch.Tensor,
    leading_reference_states: torch.Tensor,
    suffix_basis: torch.Tensor,
    suffix_prototypes: dict[int, torch.Tensor],
    digit_token_ids: dict[int, int],
    config: dict[str, Any],
    device: torch.device,
    condition_index: int,
) -> dict[str, Any]:
    prefixes = ["" for _ in examples]
    predicted_ids: list[list[int]] = [[] for _ in examples]
    target_correct = []
    original_correct = []
    relative_norms = []
    gate_rates = []

    leading_prompts = list(rendered_prompts)
    leading_logits = predict_with_delta(
        model,
        tokenizer,
        leading_prompts,
        leading_delta,
        hidden_state_index=config["leading_compiler"]["hidden_state_index"],
        batch_size=config["base_model_batch_size"],
        device=device,
    )
    leading_ids = leading_logits.argmax(dim=1).tolist()
    leading_target_ids = [
        digit_token_ids[int(str(value)[0])] for value in targets
    ]
    leading_original_ids = [
        digit_token_ids[int(str(value)[0])] for value in originals
    ]
    target_correct.append(
        sum(
            actual == expected
            for actual, expected in zip(
                leading_ids,
                leading_target_ids,
                strict=True,
            )
        )
    )
    original_correct.append(
        sum(
            actual == expected
            for actual, expected in zip(
                leading_ids,
                leading_original_ids,
                strict=True,
            )
        )
    )
    relative_norms.append(
        float(
            (
                leading_delta.norm(dim=1)
                / leading_reference_states.norm(dim=1).clamp_min(1e-12)
            ).mean()
        )
    )
    gate_rates.append(float((leading_delta.norm(dim=1) == 0).float().mean()))
    for index, token_id in enumerate(leading_ids):
        predicted_ids[index].append(int(token_id))
        prefixes[index] += tokenizer.decode([int(token_id)])

    wrong_targets = wrong_all_digits(targets)
    for position in (1, 2):
        prompts = [
            prompt + prefix
            for prompt, prefix in zip(rendered_prompts, prefixes, strict=True)
        ]
        hidden_index = config["suffix_writer"]["hidden_state_index"]
        states = capture.capture_last_token(
            prompts,
            hidden_state_indices=[hidden_index],
            batch_size=config["base_model_batch_size"],
        )[hidden_index].values.float()
        base_logits = capture.next_token_logits(
            prompts,
            batch_size=config["base_model_batch_size"],
        )
        target_ids = torch.tensor(
            [digit_token_ids[int(str(value)[position])] for value in targets]
        )
        raw_target = prototype_delta(
            states,
            targets,
            suffix_prototypes[position],
            suffix_basis,
            position=position,
        )
        targeted, gate = scale_and_gate(
            raw_target,
            states,
            base_logits,
            target_ids,
            scale=config["suffix_writer"]["scale"],
            norm_cap=config["suffix_writer"]["norm_cap"],
        )
        if condition == "base":
            delta = torch.zeros_like(targeted)
        elif condition in (
            "oracle_compute_hybrid_write",
            "latent_read_compute_hybrid_write",
            "shuffled_read_compute_hybrid_write",
        ):
            delta = targeted
        elif condition == "wrong_target_norm_matched":
            raw_wrong = prototype_delta(
                states,
                wrong_targets,
                suffix_prototypes[position],
                suffix_basis,
                position=position,
            )
            delta = norm_match(raw_wrong, targeted.norm(dim=1))
            gate = torch.zeros(len(examples), dtype=torch.bool)
        elif condition == "random_norm_matched":
            generator = torch.Generator().manual_seed(
                config["random_control_seed"]
                + condition_index * 100
                + position
            )
            random_coefficients = torch.randn(
                (len(examples), suffix_basis.shape[0]),
                generator=generator,
            )
            delta = norm_match(
                random_coefficients @ suffix_basis,
                targeted.norm(dim=1),
            )
            gate = torch.zeros(len(examples), dtype=torch.bool)
        else:
            raise ValueError(f"unknown hybrid condition: {condition}")
        logits = predict_with_delta(
            model,
            tokenizer,
            prompts,
            delta,
            hidden_state_index=hidden_index,
            batch_size=config["base_model_batch_size"],
            device=device,
        )
        next_ids = logits.argmax(dim=1).tolist()
        expected_target = [
            digit_token_ids[int(str(value)[position])] for value in targets
        ]
        expected_original = [
            digit_token_ids[int(str(value)[position])] for value in originals
        ]
        target_correct.append(
            sum(
                actual == expected
                for actual, expected in zip(
                    next_ids,
                    expected_target,
                    strict=True,
                )
            )
        )
        original_correct.append(
            sum(
                actual == expected
                for actual, expected in zip(
                    next_ids,
                    expected_original,
                    strict=True,
                )
            )
        )
        relative_norms.append(
            float((delta.norm(dim=1) / states.norm(dim=1)).mean())
        )
        gate_rates.append(float(gate.float().mean()))
        for index, token_id in enumerate(next_ids):
            predicted_ids[index].append(int(token_id))
            prefixes[index] += tokenizer.decode([int(token_id)])

    text = [tokenizer.decode(token_ids) for token_ids in predicted_ids]
    parsed = [parse_first_integer(value) for value in text]
    digit_ids = set(digit_token_ids.values())
    return {
        "n": len(examples),
        "step_target_correct": target_correct,
        "step_target_accuracy": [
            count / len(examples) for count in target_correct
        ],
        "step_original_correct": original_correct,
        "step_original_accuracy": [
            count / len(examples) for count in original_correct
        ],
        "target_full_result_correct": sum(
            value == target
            for value, target in zip(parsed, targets, strict=True)
        ),
        "target_full_result_accuracy": sum(
            value == target
            for value, target in zip(parsed, targets, strict=True)
        )
        / len(examples),
        "original_full_result_correct": sum(
            value == target
            for value, target in zip(parsed, originals, strict=True)
        ),
        "original_full_result_accuracy": sum(
            value == target
            for value, target in zip(parsed, originals, strict=True)
        )
        / len(examples),
        "parse_count": sum(value is not None for value in parsed),
        "parse_rate": sum(value is not None for value in parsed) / len(examples),
        "digit_token_count": sum(
            token_id in digit_ids for row in predicted_ids for token_id in row
        ),
        "digit_token_rate": sum(
            token_id in digit_ids for row in predicted_ids for token_id in row
        )
        / (len(examples) * 3),
        "mean_relative_norm_by_step": relative_norms,
        "hard_gate_rate_by_step": gate_rates,
        "outputs": [
            {
                "example_id": example.example_id,
                "original_result": originals[index],
                "target_result": targets[index],
                "generated_text": text[index],
                "parsed": parsed[index],
                "predicted_token_ids": predicted_ids[index],
            }
            for index, example in enumerate(examples)
        ],
    }


def paired_gate(
    conditions: dict[str, dict[str, Any]],
    *,
    reader: dict[str, Any],
    computed_accuracy: float,
    rule: dict[str, Any],
) -> dict[str, Any]:
    base = {
        row["example_id"]: row for row in conditions["base"]["outputs"]
    }
    true_results = {
        example_id: row["original_result"] for example_id, row in base.items()
    }
    base_errors = [
        example_id
        for example_id, target in true_results.items()
        if base[example_id]["parsed"] != target
    ]
    base_correct = [
        example_id
        for example_id, target in true_results.items()
        if base[example_id]["parsed"] == target
    ]

    def paired(name: str) -> dict[str, Any]:
        outputs = {
            row["example_id"]: row for row in conditions[name]["outputs"]
        }
        recovered = sum(
            outputs[example_id]["parsed"] == true_results[example_id]
            for example_id in base_errors
        )
        preserved = sum(
            outputs[example_id]["parsed"] == true_results[example_id]
            for example_id in base_correct
        )
        harmed = len(base_correct) - preserved
        return {
            "base_error_count": len(base_errors),
            "base_correct_count": len(base_correct),
            "recovered_base_errors": recovered,
            "base_error_recovery": recovered / max(1, len(base_errors)),
            "preserved_base_correct": preserved,
            "base_correct_preservation": preserved / max(1, len(base_correct)),
            "harmed_base_correct": harmed,
            "net_exact_improvement": recovered - harmed,
            "net_exact_improvement_rate": (
                (recovered - harmed) / len(true_results)
            ),
        }

    paired_metrics = {
        name: paired(name)
        for name in (
            "oracle_compute_hybrid_write",
            "latent_read_compute_hybrid_write",
            "shuffled_read_compute_hybrid_write",
            "random_norm_matched",
            "wrong_target_norm_matched",
        )
    }
    latent = conditions["latent_read_compute_hybrid_write"]
    oracle = conditions["oracle_compute_hybrid_write"]
    shuffled = conditions["shuffled_read_compute_hybrid_write"]
    wrong = conditions["wrong_target_norm_matched"]
    latent_paired = paired_metrics["latent_read_compute_hybrid_write"]
    random_paired = paired_metrics["random_norm_matched"]
    n = latent["n"]
    oracle_gap = (
        oracle["true_result_correct"] - latent["true_result_correct"]
    ) / n
    excess_recovery = (
        latent_paired["recovered_base_errors"]
        - random_paired["recovered_base_errors"]
    ) / max(1, len(base_errors))
    checks = {
        "reader": reader["pair_accuracy"] >= rule["minimum_reader_pair_accuracy"],
        "compute": computed_accuracy
        >= rule["minimum_computed_target_accuracy"],
        "final_exact": latent["true_result_accuracy"]
        >= rule["minimum_final_exact_accuracy"],
        "oracle_gap": oracle_gap <= rule["maximum_oracle_exact_gap"],
        "base_error_recovery": latent_paired["base_error_recovery"]
        >= rule["minimum_base_error_recovery"],
        "base_correct_preservation": latent_paired["base_correct_preservation"]
        >= rule["minimum_base_correct_preservation"],
        "net_improvement": latent_paired["net_exact_improvement_rate"]
        >= rule["minimum_net_improvement_over_base"],
        "excess_recovery": excess_recovery
        >= rule["minimum_excess_base_error_recovery_over_random"],
        "shuffled_control": shuffled["true_result_accuracy"]
        <= rule["maximum_shuffled_true_accuracy"],
        "wrong_target_control": wrong["true_result_accuracy"]
        <= rule["maximum_wrong_target_true_accuracy"],
        "parse": (
            not rule["require_parse_rate"] or latent["parse_rate"] == 1.0
        ),
        "digit_tokens": (
            not rule["require_digit_token_rate"]
            or latent["digit_token_rate"] == 1.0
        ),
    }
    return {
        "checks": checks,
        "passes": all(checks.values()),
        "paired_metrics": paired_metrics,
        "derived": {
            "oracle_exact_gap": oracle_gap,
            "excess_base_error_recovery_over_random": excess_recovery,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite hybrid-graft result")
    config = json.loads(args.config.read_text())
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if runner_hash != config["runner_sha256"]:
        raise SystemExit("hybrid-graft runner hash mismatch")
    paths = {
        name: Path(config[name])
        for name in (
            "dataset_config",
            "reader_selection_result",
            "reader_artifact",
            "compiler_selection_result",
            "compiler_module",
            "suffix_selection_result",
            "suffix_artifact",
            "suffix_basis_artifact",
        )
    }
    for name, path in paths.items():
        verify_sha256(path, config[f"{name}_sha256"])
    dataset_config = json.loads(paths["dataset_config"].read_text())
    reader_selection = json.loads(paths["reader_selection_result"].read_text())
    compiler_selection = json.loads(
        paths["compiler_selection_result"].read_text()
    )
    suffix_selection = json.loads(
        paths["suffix_selection_result"].read_text()
    )
    if dataset_config.get("audit_authorized", False):
        raise SystemExit("hybrid development requires a sealed audit")
    if not reader_selection["passes"]:
        raise SystemExit("operand reader source did not pass")
    if not compiler_selection["selection"]["passes"]:
        raise SystemExit("leading compiler source did not pass")
    selected_iterations = compiler_selection["selection"]["iterations"]
    if selected_iterations != config["leading_compiler"]["iterations"]:
        raise SystemExit("leading compiler iteration count changed")
    for position in ("1", "2"):
        selection = suffix_selection["positions"][position]["selection"]
        if not selection["passes"]:
            raise SystemExit(f"suffix position {position} did not pass")
        if selection["scale"] != config["suffix_writer"]["scale"]:
            raise SystemExit(f"suffix position {position} scale changed")

    examples = build_phase7_carry_quartets(
        **dataset_config["dataset"]["parameters"]
    )
    dataset_hash = phase7_carry_sha256(examples)
    if dataset_hash != dataset_config["dataset"]["sha256"]:
        raise SystemExit("Phase 9D dataset hash mismatch")
    selected = [
        row
        for row in examples
        if row.split == "development" and row.variant == "carry_base"
    ]
    if value_sha256([row.example_id for row in selected]) != config[
        "development_examples_sha256"
    ]:
        raise SystemExit("development example hash mismatch")

    model_config = dataset_config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rendered, positions, token_contract = render_and_locate(
        tokenizer,
        selected,
        dataset_config["assistant_prefix"],
    )
    if value_sha256(token_contract) != config[
        "development_token_contract_sha256"
    ]:
        raise SystemExit("development token contract mismatch")
    digit_token_ids = verify_decimal_digit_contract(tokenizer, rendered[0])
    candidate_ids = torch.tensor(
        [digit_token_ids[digit] for digit in range(10)],
        dtype=torch.long,
    )

    reader_tensors = load_file(str(paths["reader_artifact"]))
    reader = NearestCentroidDigitReader(
        classes=reader_tensors["digit_classes"],
        centroids=reader_tensors["digit_centroids"],
    )
    suffix_tensors = load_file(str(paths["suffix_artifact"]))
    basis_tensors = load_file(str(paths["suffix_basis_artifact"]))
    suffix_basis = basis_tensors["suffix_basis"][
        : config["suffix_writer"]["rank"]
    ].float()
    suffix_prototypes = {
        1: suffix_tensors["tens_digit"].float(),
        2: suffix_tensors["ones_digit"].float(),
    }

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
    started = time.perf_counter()

    captured = capture.capture_token_positions(
        rendered,
        positions,
        hidden_state_indices=[config["reader_hidden_state_index"]],
        batch_size=config["base_model_batch_size"],
    )[config["reader_hidden_state_index"]]
    reader_states, _ = flatten_states_and_labels(captured.values, selected)
    grouped = group_predictions(reader.predict(reader_states).tolist(), positions)
    read_metrics = reader_metrics(grouped, selected)
    computed_targets = [
        row["predicted_operand_a"] + row["predicted_operand_b"]
        for row in read_metrics["rows"]
    ]
    true_targets = [row.result for row in selected]
    computed_correct = sum(
        actual == expected
        for actual, expected in zip(
            computed_targets,
            true_targets,
            strict=True,
        )
    )
    computed_accuracy = computed_correct / len(selected)
    if any(target < 100 or target > 999 for target in computed_targets):
        raise SystemExit("decoded target outside three-digit writer contract")
    shuffled_targets = computed_targets[1:] + computed_targets[:1]

    trace_cache: dict[tuple[int, ...], IterativeMarginTrace] = {}
    compiler = config["leading_compiler"]
    for targets in (true_targets, computed_targets, shuffled_targets):
        key = tuple(targets)
        if key in trace_cache:
            continue
        target_ids = torch.tensor(
            [digit_token_ids[int(str(value)[0])] for value in targets]
        )
        trace_cache[key] = compile_iterative_margin_deltas(
            model,
            tokenizer,
            rendered,
            hidden_state_index=compiler["hidden_state_index"],
            target_token_ids=target_ids,
            candidate_token_ids=candidate_ids,
            desired_margin=compiler["desired_margin"],
            iterations=compiler["iterations"],
            max_relative_norm=compiler["norm_cap"],
            device=device,
            batch_size=config["compiler_batch_size"],
        )
    true_trace = trace_cache[tuple(true_targets)]
    computed_trace = trace_cache[tuple(computed_targets)]
    shuffled_trace = trace_cache[tuple(shuffled_targets)]
    true_delta = true_trace.cumulative_deltas[-1]
    computed_delta = computed_trace.cumulative_deltas[-1]
    shuffled_delta = shuffled_trace.cumulative_deltas[-1]
    leading_random = random_norm_matched(
        tuple(true_delta.shape),
        true_delta.norm(dim=1),
        seed=config["random_control_seed"],
    )
    wrong_targets = wrong_all_digits(true_targets)
    wrong_ids = torch.tensor(
        [digit_token_ids[int(str(value)[0])] for value in wrong_targets]
    )
    wrong_trace = compile_iterative_margin_deltas(
        model,
        tokenizer,
        rendered,
        hidden_state_index=compiler["hidden_state_index"],
        target_token_ids=wrong_ids,
        candidate_token_ids=candidate_ids,
        desired_margin=compiler["desired_margin"],
        iterations=compiler["iterations"],
        max_relative_norm=compiler["norm_cap"],
        device=device,
        batch_size=config["compiler_batch_size"],
    )
    leading_wrong = norm_match(
        wrong_trace.cumulative_deltas[-1],
        true_delta.norm(dim=1),
    )

    condition_specs = {
        "base": (
            "base",
            true_targets,
            torch.zeros_like(true_delta),
            true_trace.base_recipient_states,
        ),
        "oracle_compute_hybrid_write": (
            "oracle_compute_hybrid_write",
            true_targets,
            true_delta,
            true_trace.base_recipient_states,
        ),
        "latent_read_compute_hybrid_write": (
            "latent_read_compute_hybrid_write",
            computed_targets,
            computed_delta,
            computed_trace.base_recipient_states,
        ),
        "shuffled_read_compute_hybrid_write": (
            "shuffled_read_compute_hybrid_write",
            shuffled_targets,
            shuffled_delta,
            shuffled_trace.base_recipient_states,
        ),
        "random_norm_matched": (
            "random_norm_matched",
            true_targets,
            leading_random,
            true_trace.base_recipient_states,
        ),
        "wrong_target_norm_matched": (
            "wrong_target_norm_matched",
            true_targets,
            leading_wrong,
            true_trace.base_recipient_states,
        ),
    }
    conditions = {}
    for condition_index, (
        name,
        (engine_name, targets, leading_delta, reference_states),
    ) in enumerate(condition_specs.items()):
        result = evaluate_hybrid_condition(
            engine_name,
            model,
            tokenizer,
            capture,
            examples=selected,
            targets=targets,
            originals=true_targets,
            rendered_prompts=rendered,
            leading_delta=leading_delta,
            leading_reference_states=reference_states,
            suffix_basis=suffix_basis,
            suffix_prototypes=suffix_prototypes,
            digit_token_ids=digit_token_ids,
            config=config,
            device=device,
            condition_index=condition_index,
        )
        conditions[name] = {
            **result,
            **true_result_metrics(result, true_targets),
        }
    gate = paired_gate(
        conditions,
        reader=read_metrics,
        computed_accuracy=computed_accuracy,
        rule=config["development_rule"],
    )
    report = {
        "schema_version": "oli.phase9d-phi-hybrid-graft-development/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "development_only",
        "model": model_config,
        "dataset_sha256": dataset_hash,
        "evaluation_split": "development",
        "reader": {
            "hidden_state_index": config["reader_hidden_state_index"],
            "metrics": read_metrics,
        },
        "deterministic_compute": {
            "operation": "integer_addition",
            "correct": computed_correct,
            "accuracy": computed_accuracy,
            "targets": computed_targets,
        },
        "writer": {
            "leading_compiler": config["leading_compiler"],
            "suffix_writer": config["suffix_writer"],
        },
        "conditions": conditions,
        "gate": {
            "thresholds": config["development_rule"],
            **gate,
        },
        "passes": gate["passes"],
        "source_hashes": {
            f"{name}_sha256": config[f"{name}_sha256"]
            for name in paths
        },
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
            "Exposed development-only latent operand read, deterministic "
            "addition, iterative leading write, and native suffix write."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
