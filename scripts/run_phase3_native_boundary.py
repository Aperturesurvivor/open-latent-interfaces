#!/usr/bin/env python3
"""Rediscover full-native next-digit write boundaries in frozen Phi-3.5."""

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
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.donors import choose_position_donors
from open_latent_interfaces.evaluation import (
    norm_match,
    random_norm_matched,
    token_metrics,
)
from open_latent_interfaces.interventions import intervened_next_token_logits
from open_latent_interfaces.phase2_data import balanced_counterfactual_results
from open_latent_interfaces.phase3_data import (
    build_phase3_additions,
    phase3_addition_sha256,
)
from open_latent_interfaces.prefill import (
    render_prefilled_chat,
    verify_decimal_digit_contract,
)


def verify_sha256(path: Path, expected: str) -> None:
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise SystemExit(f"hash mismatch for {path}: {observed} != {expected}")


def value_list_sha256(values: list[Any]) -> str:
    encoded = json.dumps(values, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def render_examples(
    tokenizer: Any,
    examples: list[Any],
    *,
    assistant_prefix: str,
) -> list[str]:
    return [
        render_prefilled_chat(
            tokenizer,
            example.prompt,
            assistant_prefix=assistant_prefix,
        )
        for example in examples
    ]


def prefix_prompts(
    rendered: list[str],
    results: list[int],
    *,
    position: int,
) -> list[str]:
    return [
        prompt + str(result)[:position]
        for prompt, result in zip(rendered, results, strict=True)
    ]


def predict_with_delta(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    delta: torch.Tensor,
    *,
    hidden_state_index: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    chunks = []
    for start in range(0, len(prompts), batch_size):
        chunks.append(
            intervened_next_token_logits(
                model,
                tokenizer,
                prompts[start : start + batch_size],
                hidden_state_index=hidden_state_index,
                deltas=delta[start : start + batch_size],
                device=device,
            )
        )
    return torch.cat(chunks)


def select_boundary(
    results: dict[str, dict[str, Any]],
    *,
    minimum_target_accuracy: float,
    minimum_control_advantage: float,
    maximum_relative_norm: float,
) -> tuple[int, bool]:
    def values(hidden_index: str) -> tuple[float, float, float, float, int]:
        layer = results[hidden_index]
        targeted = layer["targeted_donor"]
        controls = [
            row["top1_exact"]
            for name, row in layer.items()
            if name not in ("targeted_donor", "base")
        ]
        advantage = targeted["top1_exact"] - max(controls)
        return (
            targeted["top1_exact"],
            advantage,
            targeted["mean_target_margin"],
            -targeted["mean_relative_norm"],
            -int(hidden_index),
        )

    selected = max(results, key=values)
    targeted = results[selected]["targeted_donor"]
    strongest_control = max(
        row["top1_exact"]
        for name, row in results[selected].items()
        if name not in ("targeted_donor", "base")
    )
    passes = (
        targeted["top1_exact"] >= minimum_target_accuracy
        and targeted["top1_exact"] - strongest_control
        >= minimum_control_advantage
        and targeted["mean_relative_norm"] <= maximum_relative_norm
    )
    return int(selected), passes


def evaluate_position(
    model: Any,
    tokenizer: Any,
    capture: ActivationCapture,
    *,
    fit: list[Any],
    recipients: list[Any],
    targets: list[int],
    targeted_indices: list[int],
    wrong_indices: list[int],
    rendered_fit: list[str],
    rendered_recipients: list[str],
    digit_token_ids: dict[int, int],
    position: int,
    hidden_state_indices: list[int],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    recipient_prompts = prefix_prompts(
        rendered_recipients,
        targets,
        position=position,
    )
    targeted_donors = [fit[index] for index in targeted_indices]
    wrong_donors = [fit[index] for index in wrong_indices]
    targeted_prompts = prefix_prompts(
        [rendered_fit[index] for index in targeted_indices],
        [donor.result for donor in targeted_donors],
        position=position,
    )
    wrong_prompts = prefix_prompts(
        [rendered_fit[index] for index in wrong_indices],
        [donor.result for donor in wrong_donors],
        position=position,
    )
    recipient_states = capture.capture_last_token(
        recipient_prompts,
        hidden_state_indices=hidden_state_indices,
        batch_size=config["base_model_batch_size"],
    )
    targeted_states = capture.capture_last_token(
        targeted_prompts,
        hidden_state_indices=hidden_state_indices,
        batch_size=config["base_model_batch_size"],
    )
    wrong_states = capture.capture_last_token(
        wrong_prompts,
        hidden_state_indices=hidden_state_indices,
        batch_size=config["base_model_batch_size"],
    )
    expected = torch.tensor(
        [digit_token_ids[int(str(target)[position])] for target in targets]
    )
    conditions = (
        "base",
        "targeted_donor",
        "wrong_digit_norm_matched",
        "shuffled_donor_norm_matched",
        "random_norm_matched",
    )
    results = {}
    for hidden_index in hidden_state_indices:
        recipient = recipient_states[hidden_index].values
        targeted = targeted_states[hidden_index].values
        wrong = wrong_states[hidden_index].values
        targeted_delta = targeted - recipient
        targeted_norms = targeted_delta.norm(dim=1)
        layer_results = {}
        for condition_index, condition in enumerate(conditions):
            if condition == "base":
                delta = torch.zeros_like(targeted_delta)
            elif condition == "targeted_donor":
                delta = targeted_delta
            elif condition == "wrong_digit_norm_matched":
                delta = norm_match(wrong - recipient, targeted_norms)
            elif condition == "shuffled_donor_norm_matched":
                shuffled = torch.cat((targeted[1:], targeted[:1]))
                delta = norm_match(shuffled - recipient, targeted_norms)
            else:
                delta = random_norm_matched(
                    tuple(targeted_delta.shape),
                    targeted_norms,
                    seed=(
                        config["random_control_seed"]
                        + position * 1000
                        + hidden_index * 10
                        + condition_index
                    ),
                )
            logits = predict_with_delta(
                model,
                tokenizer,
                recipient_prompts,
                delta,
                hidden_state_index=hidden_index,
                batch_size=config["base_model_batch_size"],
                device=device,
            )
            metrics = token_metrics(logits, expected)
            metrics["mean_relative_norm"] = float(
                (delta.norm(dim=1) / recipient.norm(dim=1)).mean()
            )
            metrics["predicted_token_ids"] = logits.argmax(dim=1).tolist()
            layer_results[condition] = metrics
        results[str(hidden_index)] = layer_results
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing result: {args.output}")
    config = json.loads(args.config.read_text())
    dataset_path = Path(config["dataset_config"])
    behavior_path = Path(config["behavior_result"])
    verify_sha256(dataset_path, config["dataset_config_sha256"])
    verify_sha256(behavior_path, config["behavior_result_sha256"])
    dataset_config = json.loads(dataset_path.read_text())
    if dataset_config.get("audit_authorized", False):
        raise SystemExit("boundary mapping requires a sealed audit")
    behavior = json.loads(behavior_path.read_text())
    if not behavior["passes"]:
        raise SystemExit("prefill behavior gate did not pass")

    examples = build_phase3_additions(**dataset_config["dataset"]["parameters"])
    observed_hash = phase3_addition_sha256(examples)
    if observed_hash != dataset_config["dataset"]["sha256"]:
        raise SystemExit("Phase 3 dataset hash mismatch")
    exact_fit_ids = {
        row["example_id"]
        for row in behavior["rows"]
        if row["split"] == "fit" and row["exact"]
    }
    fit = [
        example
        for example in examples
        if example.split == "fit" and example.example_id in exact_fit_ids
    ]
    selection = [example for example in examples if example.split == "selection"]
    targets = balanced_counterfactual_results(selection)
    if value_list_sha256(targets) != config["target_sha256"]:
        raise SystemExit("counterfactual target hash mismatch")

    targeted_by_position = {}
    wrong_by_position = {}
    donor_hashes = {}
    for position in range(3):
        targeted = choose_position_donors(
            fit,
            selection,
            targets,
            position=position,
        )
        wrong = choose_position_donors(
            fit,
            selection,
            targets,
            position=position,
            wrong_digit=True,
        )
        targeted_by_position[position] = targeted
        wrong_by_position[position] = wrong
        donor_hashes[str(position)] = {
            "targeted": value_list_sha256(
                [fit[index].example_id for index in targeted]
            ),
            "wrong": value_list_sha256(
                [fit[index].example_id for index in wrong]
            ),
        }
    if donor_hashes != config["donor_assignment_sha256"]:
        raise SystemExit("donor assignment hash mismatch")

    device = torch.device(args.device)
    model_config = dataset_config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rendered_fit = render_examples(
        tokenizer,
        fit,
        assistant_prefix=dataset_config["assistant_prefix"],
    )
    rendered_selection = render_examples(
        tokenizer,
        selection,
        assistant_prefix=dataset_config["assistant_prefix"],
    )
    digit_token_ids = verify_decimal_digit_contract(tokenizer, rendered_fit[0])
    if {str(key): value for key, value in digit_token_ids.items()} != behavior[
        "digit_token_ids"
    ]:
        raise SystemExit("digit token map differs from behavior result")

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

    positions = {}
    for position in range(3):
        layers = evaluate_position(
            model,
            tokenizer,
            capture,
            fit=fit,
            recipients=selection,
            targets=targets,
            targeted_indices=targeted_by_position[position],
            wrong_indices=wrong_by_position[position],
            rendered_fit=rendered_fit,
            rendered_recipients=rendered_selection,
            digit_token_ids=digit_token_ids,
            position=position,
            hidden_state_indices=config["hidden_state_indices"],
            config=config,
            device=device,
        )
        selected_hidden_index, passes = select_boundary(
            layers,
            minimum_target_accuracy=config["selection_rule"][
                "minimum_target_accuracy"
            ],
            minimum_control_advantage=config["selection_rule"][
                "minimum_control_advantage"
            ],
            maximum_relative_norm=config["selection_rule"][
                "maximum_relative_norm"
            ],
        )
        positions[str(position)] = {
            "selected_hidden_state_index": selected_hidden_index,
            "passes": passes,
            "layers": layers,
        }

    report = {
        "schema_version": "oli.phase3-native-boundary/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "selection_only",
        "model": model_config,
        "dataset": {
            "config": str(dataset_path),
            "sha256": observed_hash,
            "eligible_fit_examples": len(fit),
            "selection_examples": len(selection),
        },
        "behavior_source": {
            "path": str(behavior_path),
            "sha256": config["behavior_result_sha256"],
        },
        "target_assignment": {
            "scheme": "balanced_all_digits_changed",
            "sha256": config["target_sha256"],
        },
        "donor_assignment_sha256": donor_hashes,
        "hidden_state_indices": config["hidden_state_indices"],
        "selection_rule": config["selection_rule"],
        "positions": positions,
        "passes": all(row["passes"] for row in positions.values()),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Selection-only full-native donor boundary map. This is a causal "
            "upper bound, not a compressed or donor-free interface."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
