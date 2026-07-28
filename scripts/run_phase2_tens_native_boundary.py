#!/usr/bin/env python3
"""Map native donor control of counterfactual tens digits across late layers."""

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
from run_phase1_conditional_transport_bridge import render_prompts, result_token_ids
from run_phase2_causal_adapter import result_list_sha256
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.donors import choose_prefix_donors
from open_latent_interfaces.evaluation import (
    norm_match,
    random_norm_matched,
    token_metrics,
)
from open_latent_interfaces.interventions import intervened_next_token_logits
from open_latent_interfaces.phase2_data import (
    PHASE2_TEMPLATES,
    balanced_counterfactual_results,
    build_phase2_additions,
    phase2_addition_sha256,
)


def verify_sha256(path: Path, expected: str) -> None:
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise SystemExit(f"hash mismatch for {path}")


def id_list_sha256(values: list[str]) -> str:
    payload = json.dumps(values, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def wrong_tens_results(targets: list[int]) -> list[int]:
    results = []
    for target in targets:
        digits = list(str(target))
        digits[1] = str((int(digits[1]) + 1) % 10)
        results.append(int("".join(digits)))
    return results


def rendered_donor_prompts(
    tokenizer: Any,
    donors: list[Any],
    *,
    template_split: str,
) -> list[str]:
    template = PHASE2_TEMPLATES[template_split]
    return [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": template.format(
                        a=donor.operand_a,
                        b=donor.operand_b,
                    ),
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for donor in donors
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


def evaluate_boundaries(
    model: Any,
    tokenizer: Any,
    capture: ActivationCapture,
    *,
    examples: list[Any],
    targets: list[int],
    targeted_donors: list[Any],
    wrong_donors: list[Any],
    hidden_state_indices: list[int],
    config: dict[str, Any],
    device: torch.device,
    include_outputs: bool,
) -> dict[str, dict[str, Any]]:
    template_split = examples[0].split
    recipient_prompts = [
        prompt + str(target)[0]
        for prompt, target in zip(
            render_prompts(tokenizer, examples),
            targets,
            strict=True,
        )
    ]
    targeted_prompts = [
        prompt + str(donor.result)[0]
        for prompt, donor in zip(
            rendered_donor_prompts(
                tokenizer,
                targeted_donors,
                template_split=template_split,
            ),
            targeted_donors,
            strict=True,
        )
    ]
    wrong_prompts = [
        prompt + str(donor.result)[0]
        for prompt, donor in zip(
            rendered_donor_prompts(
                tokenizer,
                wrong_donors,
                template_split=template_split,
            ),
            wrong_donors,
            strict=True,
        )
    ]
    captured_recipient = capture.capture_last_token(
        recipient_prompts,
        hidden_state_indices=hidden_state_indices,
        batch_size=config["base_model_batch_size"],
    )
    captured_targeted = capture.capture_last_token(
        targeted_prompts,
        hidden_state_indices=hidden_state_indices,
        batch_size=config["base_model_batch_size"],
    )
    captured_wrong = capture.capture_last_token(
        wrong_prompts,
        hidden_state_indices=hidden_state_indices,
        batch_size=config["base_model_batch_size"],
    )
    expected = torch.tensor(
        [row[1] for row in result_token_ids(tokenizer, targets)]
    )
    conditions = (
        "base",
        "targeted_donor",
        "wrong_tens_norm_matched",
        "shuffled_donor_norm_matched",
        "random_norm_matched",
    )
    results = {}
    for hidden_index in hidden_state_indices:
        recipient = captured_recipient[hidden_index].values
        targeted = captured_targeted[hidden_index].values
        wrong = captured_wrong[hidden_index].values
        targeted_delta = targeted - recipient
        targeted_norms = targeted_delta.norm(dim=1)
        layer_results = {}
        for condition_index, condition in enumerate(conditions):
            if condition == "base":
                delta = torch.zeros_like(targeted_delta)
            elif condition == "targeted_donor":
                delta = targeted_delta
            elif condition == "wrong_tens_norm_matched":
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
            if include_outputs:
                predicted = logits.argmax(dim=1).tolist()
                metrics["outputs"] = [
                    {
                        "example_id": example.example_id,
                        "original_result": example.result,
                        "target_result": target,
                        "target_tens": int(str(target)[1]),
                        "predicted_token_id": int(token_id),
                        "predicted_text": tokenizer.decode([int(token_id)]),
                    }
                    for example, target, token_id in zip(
                        examples,
                        targets,
                        predicted,
                        strict=True,
                    )
                ]
            layer_results[condition] = metrics
        results[str(hidden_index)] = layer_results
    return results


def select_boundary(results: dict[str, dict[str, Any]]) -> int:
    def score(hidden_index: str) -> tuple[float, float, float, float]:
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
        )

    return int(max(results, key=score))


def donor_assignments(
    fit: list[Any],
    examples: list[Any],
    targets: list[int],
) -> tuple[list[int], list[int]]:
    targeted = choose_prefix_donors(
        fit,
        examples,
        targets,
        prefix_length=2,
    )
    wrong = choose_prefix_donors(
        fit,
        examples,
        wrong_tens_results(targets),
        prefix_length=2,
    )
    return targeted, wrong


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source_path = Path(config["source_result"])
    dataset_config_path = Path(config["dataset_config"])
    verify_sha256(source_path, config["source_result_sha256"])
    verify_sha256(dataset_config_path, config["dataset_config_sha256"])
    source = json.loads(source_path.read_text())
    dataset_config = json.loads(dataset_config_path.read_text())
    examples = build_phase2_additions(**dataset_config["dataset"]["parameters"])
    if phase2_addition_sha256(examples) != source["dataset"]["sha256"]:
        raise SystemExit("dataset hash mismatch")
    fit = [example for example in examples if example.split == "fit"]
    selection = [example for example in examples if example.split == "selection"]
    development = [example for example in examples if example.split == "development"]
    selection_targets = balanced_counterfactual_results(selection)
    development_targets = balanced_counterfactual_results(development)
    target_hashes = {
        "selection": result_list_sha256(selection_targets),
        "development": result_list_sha256(development_targets),
    }
    if target_hashes != config["target_sha256"]:
        raise SystemExit("counterfactual target hash mismatch")

    selection_targeted, selection_wrong = donor_assignments(
        fit,
        selection,
        selection_targets,
    )
    development_targeted, development_wrong = donor_assignments(
        fit,
        development,
        development_targets,
    )
    donor_hashes = {
        "selection_targeted": id_list_sha256(
            [fit[index].example_id for index in selection_targeted]
        ),
        "selection_wrong_tens": id_list_sha256(
            [fit[index].example_id for index in selection_wrong]
        ),
        "development_targeted": id_list_sha256(
            [fit[index].example_id for index in development_targeted]
        ),
        "development_wrong_tens": id_list_sha256(
            [fit[index].example_id for index in development_wrong]
        ),
    }
    if donor_hashes != config["donor_assignment_sha256"]:
        raise SystemExit("donor assignment hash mismatch")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        source["model"]["id"],
        revision=source["model"]["revision"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        source["model"]["id"],
        revision=source["model"]["revision"],
        torch_dtype=getattr(torch, args.dtype),
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    capture = ActivationCapture(model, tokenizer, device=device)
    started = time.perf_counter()

    selection_results = evaluate_boundaries(
        model,
        tokenizer,
        capture,
        examples=selection,
        targets=selection_targets,
        targeted_donors=[fit[index] for index in selection_targeted],
        wrong_donors=[fit[index] for index in selection_wrong],
        hidden_state_indices=config["hidden_state_indices"],
        config=config,
        device=device,
        include_outputs=False,
    )
    selected_hidden_index = select_boundary(selection_results)
    development_results = evaluate_boundaries(
        model,
        tokenizer,
        capture,
        examples=development,
        targets=development_targets,
        targeted_donors=[fit[index] for index in development_targeted],
        wrong_donors=[fit[index] for index in development_wrong],
        hidden_state_indices=[selected_hidden_index],
        config=config,
        device=device,
        include_outputs=True,
    )
    report = {
        "schema_version": "oli.phase2-tens-native-boundary/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "development_only",
        "model": source["model"],
        "dataset": source["dataset"],
        "source": {
            "result": str(source_path),
            "result_sha256": config["source_result_sha256"],
        },
        "target_assignment": {
            "scheme": "balanced_all_digits_changed",
            "sha256": target_hashes,
        },
        "donors": {
            "pool": "fit",
            "prefix_length": 2,
            "assignment_sha256": donor_hashes,
        },
        "selection": {
            "hidden_state_indices": config["hidden_state_indices"],
            "selected_hidden_state_index": selected_hidden_index,
            "layers": selection_results,
        },
        "development": {
            "hidden_state_index": selected_hidden_index,
            "decoder_block": selected_hidden_index - 1,
            "conditions": development_results[str(selected_hidden_index)],
        },
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Native fit-donor, teacher-forced tens-position boundary diagnosis; "
            "not a compact adapter and not an audit."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
