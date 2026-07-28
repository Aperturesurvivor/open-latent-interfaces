#!/usr/bin/env python3
"""Evaluate a digit-restricted nearest-state native transport dictionary."""

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
from run_phase1_conditional_transport_bridge import (
    capture_deduplicated,
    predict_ids,
    render_prompts,
    result_token_ids,
    target_results,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.capability import parse_first_integer
from open_latent_interfaces.donors import choose_multi_donors
from open_latent_interfaces.evaluation import norm_match, random_norm_matched
from open_latent_interfaces.phase1_data import (
    build_phase1_additions,
    phase1_addition_sha256,
)
from open_latent_interfaces.typed_writer import (
    LocalTransportDictionary,
    build_local_transport_dictionary,
)


def build_dictionaries(
    capture: ActivationCapture,
    examples: list[Any],
    prompts: list[str],
    *,
    config: dict[str, Any],
) -> tuple[list[LocalTransportDictionary], list[dict[str, Any]]]:
    donor_rows = choose_multi_donors(examples)
    pairs = [
        (recipient_index, donor_index)
        for recipient_index, row in enumerate(donor_rows)
        for donor_index in row
    ]
    if len(pairs) != len(examples) * config["donors_per_fit_recipient"]:
        raise ValueError("unexpected training pair count")
    dictionaries = []
    metadata = []
    hidden_index = config["hidden_state_index"]
    for step in range(3):
        recipient_inputs = [
            prompts[recipient_index] + str(examples[donor_index].result)[:step]
            for recipient_index, donor_index in pairs
        ]
        donor_inputs = [
            prompts[donor_index] + str(examples[donor_index].result)[:step]
            for _, donor_index in pairs
        ]
        recipient_states = capture_deduplicated(
            capture,
            recipient_inputs,
            hidden_index=hidden_index,
            batch_size=config["batch_size"],
        )
        donor_states = capture_deduplicated(
            capture,
            donor_inputs,
            hidden_index=hidden_index,
            batch_size=config["batch_size"],
        )
        deltas = donor_states - recipient_states
        digits = torch.tensor(
            [int(str(examples[donor_index].result)[step]) for _, donor_index in pairs]
        )
        dictionary = build_local_transport_dictionary(
            recipient_states,
            deltas,
            digits,
            state_rank=config["state_rank"],
            max_transport_rank=max(config["transport_ranks"]),
        )
        dictionaries.append(dictionary)
        metadata.append(
            {
                "step": step,
                "training_pairs": len(pairs),
                "unique_recipient_contexts": len(set(recipient_inputs)),
                "unique_donor_contexts": len(set(donor_inputs)),
                "minimum_digit_dictionary_size": min(
                    int((digits == value).sum()) for value in digits.unique()
                ),
                "mean_full_transport_relative_norm": float(
                    (deltas.norm(dim=1) / recipient_states.norm(dim=1)).mean()
                ),
            }
        )
    return dictionaries, metadata


def select_parameters(
    model: Any,
    tokenizer: Any,
    capture: ActivationCapture,
    dictionaries: list[LocalTransportDictionary],
    *,
    prompts: list[str],
    targets: list[int],
    config: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    expected = result_token_ids(tokenizer, targets)
    hidden_index = config["hidden_state_index"]
    selections = []
    for step, dictionary in enumerate(dictionaries):
        step_prompts = [
            prompt + str(result)[:step]
            for prompt, result in zip(prompts, targets, strict=True)
        ]
        states = capture.capture_last_token(
            step_prompts,
            hidden_state_indices=[hidden_index],
            batch_size=config["batch_size"],
        )[hidden_index].values
        digits = torch.tensor([int(str(result)[step]) for result in targets])
        candidates = []
        for neighbors in config["neighbor_counts"]:
            for rank in config["transport_ranks"]:
                raw = dictionary.predict(
                    states,
                    digits,
                    neighbors=neighbors,
                    transport_rank=rank,
                )
                for scale in config["scales"]:
                    delta = raw * scale
                    predicted = predict_ids(
                        model,
                        tokenizer,
                        step_prompts,
                        delta,
                        hidden_index=hidden_index,
                        batch_size=config["batch_size"],
                        device=device,
                    )
                    candidates.append(
                        {
                            "neighbors": neighbors,
                            "transport_rank": rank,
                            "scale": scale,
                            "target_token_accuracy": sum(
                                actual == wanted[step]
                                for actual, wanted in zip(
                                    predicted, expected, strict=True
                                )
                            )
                            / len(predicted),
                            "mean_relative_norm": float(
                                (delta.norm(dim=1) / states.norm(dim=1)).mean()
                            ),
                        }
                    )
        selected = max(
            candidates,
            key=lambda row: (
                row["target_token_accuracy"],
                -row["transport_rank"],
                -row["neighbors"],
                -row["mean_relative_norm"],
            ),
        )
        selections.append({"step": step, "selected": selected, "candidates": candidates})
    return selections


def evaluate(
    condition: str,
    model: Any,
    tokenizer: Any,
    capture: ActivationCapture,
    dictionaries: list[LocalTransportDictionary],
    selections: list[dict[str, Any]],
    *,
    examples: list[Any],
    prompts: list[str],
    targets: list[int],
    config: dict[str, Any],
    device: torch.device,
    condition_index: int,
) -> dict[str, Any]:
    originals = [example.result for example in examples]
    expected = result_token_ids(tokenizer, targets)
    shuffled_targets = targets[1:] + targets[:1]
    prefixes = ["" for _ in examples]
    predicted_ids: list[list[int]] = [[] for _ in examples]
    accuracies = []
    norms = []
    hidden_index = config["hidden_state_index"]
    for step, (dictionary, selection) in enumerate(
        zip(dictionaries, selections, strict=True)
    ):
        step_prompts = [
            prompt + prefix for prompt, prefix in zip(prompts, prefixes, strict=True)
        ]
        states = capture.capture_last_token(
            step_prompts,
            hidden_state_indices=[hidden_index],
            batch_size=config["batch_size"],
        )[hidden_index].values
        selected = selection["selected"]
        target_digits = torch.tensor([int(str(value)[step]) for value in targets])
        typed = dictionary.predict(
            states,
            target_digits,
            neighbors=selected["neighbors"],
            transport_rank=selected["transport_rank"],
        ) * selected["scale"]
        if condition == "base":
            delta = torch.zeros_like(typed)
        elif condition == "local_transport":
            delta = typed
        elif condition == "same_digit":
            digits = torch.tensor([int(str(value)[step]) for value in originals])
            delta = dictionary.predict(
                states,
                digits,
                neighbors=selected["neighbors"],
                transport_rank=selected["transport_rank"],
            ) * selected["scale"]
        elif condition == "shuffled_target_norm_matched":
            digits = torch.tensor(
                [int(str(value)[step]) for value in shuffled_targets]
            )
            raw = dictionary.predict(
                states,
                digits,
                neighbors=selected["neighbors"],
                transport_rank=selected["transport_rank"],
            ) * selected["scale"]
            delta = norm_match(raw, typed.norm(dim=1))
        elif condition == "shuffled_state_norm_matched":
            shuffled_states = torch.cat((states[1:], states[:1]))
            raw = dictionary.predict(
                shuffled_states,
                target_digits,
                neighbors=selected["neighbors"],
                transport_rank=selected["transport_rank"],
            ) * selected["scale"]
            delta = norm_match(raw, typed.norm(dim=1))
        else:
            delta = random_norm_matched(
                tuple(typed.shape),
                typed.norm(dim=1),
                seed=config["random_seed"] + condition_index * 10 + step,
            )
        next_ids = predict_ids(
            model,
            tokenizer,
            step_prompts,
            delta,
            hidden_index=hidden_index,
            batch_size=config["batch_size"],
            device=device,
        )
        accuracies.append(
            sum(
                actual == wanted[step]
                for actual, wanted in zip(next_ids, expected, strict=True)
            )
            / len(next_ids)
        )
        norms.append(float((delta.norm(dim=1) / states.norm(dim=1)).mean()))
        for index, token_id in enumerate(next_ids):
            predicted_ids[index].append(int(token_id))
            prefixes[index] += tokenizer.decode([int(token_id)])
    text = [tokenizer.decode(row) for row in predicted_ids]
    parsed = [parse_first_integer(value) for value in text]
    return {
        "step_target_token_accuracy": accuracies,
        "target_full_result_accuracy": sum(
            value == target for value, target in zip(parsed, targets, strict=True)
        )
        / len(parsed),
        "original_full_result_accuracy": sum(
            value == original
            for value, original in zip(parsed, originals, strict=True)
        )
        / len(parsed),
        "parse_rate": sum(value is not None for value in parsed) / len(parsed),
        "mean_relative_norm_by_step": norms,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    prerequisite = Path(config["full_result_transport"])
    if hashlib.sha256(prerequisite.read_bytes()).hexdigest() != config[
        "full_result_transport_sha256"
    ]:
        raise SystemExit("full-result transport hash mismatch")
    previous = json.loads(prerequisite.read_text())
    examples = build_phase1_additions()
    observed_hash = phase1_addition_sha256(examples)
    if observed_hash != previous["dataset"]["sha256"]:
        raise SystemExit("Phase 1 dataset hash mismatch")
    training = [example for example in examples if example.split == "train"]
    development = [
        example for example in examples if example.split == "development"
    ]
    fit_examples = training[: config["fit_examples"]]
    selection_examples = training[config["fit_examples"] :]
    if len(selection_examples) != config["selection_examples"]:
        raise ValueError("training fit/selection split mismatch")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        previous["model"]["id"], revision=previous["model"]["revision"]
    )
    model = AutoModelForCausalLM.from_pretrained(
        previous["model"]["id"],
        revision=previous["model"]["revision"],
        torch_dtype=getattr(torch, args.dtype),
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    capture = ActivationCapture(model, tokenizer, device=device)
    started = time.perf_counter()

    fit_prompts = render_prompts(tokenizer, fit_examples)
    dictionaries, fit_metadata = build_dictionaries(
        capture,
        fit_examples,
        fit_prompts,
        config=config,
    )
    selection_prompts = render_prompts(tokenizer, selection_examples)
    selection_targets = target_results(selection_examples)
    selections = select_parameters(
        model,
        tokenizer,
        capture,
        dictionaries,
        prompts=selection_prompts,
        targets=selection_targets,
        config=config,
        device=device,
    )
    development_prompts = render_prompts(tokenizer, development)
    development_targets = target_results(development)
    conditions = (
        "base",
        "local_transport",
        "same_digit",
        "shuffled_target_norm_matched",
        "shuffled_state_norm_matched",
        "random_norm_matched",
    )
    condition_results = {
        condition: evaluate(
            condition,
            model,
            tokenizer,
            capture,
            dictionaries,
            selections,
            examples=development,
            prompts=development_prompts,
            targets=development_targets,
            config=config,
            device=device,
            condition_index=index,
        )
        for index, condition in enumerate(conditions)
    }
    report = {
        "schema_version": "oli.phase1-local-transport-dictionary/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "development_only",
        "model": previous["model"],
        "dataset": {
            "sha256": observed_hash,
            "fit_examples": len(fit_examples),
            "fit_pairs": len(fit_examples)
            * config["donors_per_fit_recipient"],
            "selection_examples": len(selection_examples),
            "development_examples": len(development),
            "audit_examples_unopened": sum(
                example.split == "audit" for example in examples
            ),
        },
        "dictionary": {
            "type": "digit-restricted PCA nearest-state transport",
            "hidden_state_index": config["hidden_state_index"],
            "decoder_block": config["hidden_state_index"] - 1,
            "state_rank": config["state_rank"],
            "fit": fit_metadata,
            "selection": selections,
            "inference_inputs": ["recipient native state", "desired next digit"],
            "live_donor_required": False,
        },
        "conditions": condition_results,
        "prior_results": {
            "full_result_artifact": str(prerequisite),
            "full_result_sha256": config["full_result_transport_sha256"],
            "full_result_exact_accuracy": previous["closed_loop_conditions"][
                "full_result_transport"
            ]["target_full_result_accuracy"],
            "linear_conditional_exact_accuracy": previous["prior_results"][
                "next_digit_exact_accuracy"
            ],
            "full_donor_exact_accuracy": previous["prior_results"][
                "full_donor_exact_accuracy"
            ],
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
            "A development-only retrieval dictionary is a diagnostic for local "
            "nonlinearity, not a compact or audited deterministic graft."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
