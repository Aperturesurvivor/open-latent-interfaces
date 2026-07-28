#!/usr/bin/env python3
"""Train, select, and evaluate the Phase 2 scaled native transport adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from run_phase1_conditional_transport_bridge import (
    capture_deduplicated,
    predict_ids,
    render_prompts,
    result_token_ids,
)
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.adapter import (
    AdapterProjection,
    FittedTransportAdapter,
    fit_transport_adapter,
    prepare_adapter_projection,
)
from open_latent_interfaces.capability import parse_first_integer
from open_latent_interfaces.donors import choose_cyclic_donors, choose_donors
from open_latent_interfaces.evaluation import norm_match, random_norm_matched
from open_latent_interfaces.phase2_data import (
    build_phase2_additions,
    phase2_addition_sha256,
)


@dataclass
class AdapterEnsemble:
    members: list[FittedTransportAdapter]

    def predict(self, states: torch.Tensor, digits: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [member.predict(states, digits) for member in self.members]
        ).mean(dim=0)


def target_results(examples: list[Any]) -> list[int]:
    indices, _ = choose_donors(examples)
    return [examples[index].result for index in indices]


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def build_training_designs(
    capture: ActivationCapture,
    examples: list[Any],
    prompts: list[str],
    *,
    config: dict[str, Any],
) -> tuple[
    list[AdapterProjection],
    list[dict[str, torch.Tensor]],
    list[dict[str, Any]],
]:
    donor_rows = choose_cyclic_donors(
        examples,
        offsets=tuple(config["donor_offsets"]),
    )
    pairs = [
        (recipient_index, donor_index)
        for recipient_index, row in enumerate(donor_rows)
        for donor_index in row
    ]
    projections = []
    training = []
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
        identity_inputs = [
            prompt + str(example.result)[:step]
            for prompt, example in zip(prompts, examples, strict=True)
        ]
        recipient_states = capture_deduplicated(
            capture,
            recipient_inputs,
            hidden_index=hidden_index,
            batch_size=config["base_model_batch_size"],
        )
        donor_states = capture_deduplicated(
            capture,
            donor_inputs,
            hidden_index=hidden_index,
            batch_size=config["base_model_batch_size"],
        )
        identity_states = capture_deduplicated(
            capture,
            identity_inputs,
            hidden_index=hidden_index,
            batch_size=config["base_model_batch_size"],
        )
        targeted_deltas = donor_states - recipient_states
        states = torch.cat((recipient_states, identity_states))
        deltas = torch.cat((targeted_deltas, torch.zeros_like(identity_states)))
        digits = torch.tensor(
            [int(str(examples[donor_index].result)[step]) for _, donor_index in pairs]
            + [int(str(example.result)[step]) for example in examples]
        )
        identity_mask = torch.cat(
            (
                torch.zeros(len(pairs), dtype=torch.bool),
                torch.ones(len(examples), dtype=torch.bool),
            )
        )
        projection = prepare_adapter_projection(
            states,
            deltas,
            state_rank=config["state_rank"],
            max_transport_rank=max(config["transport_ranks"]),
        )
        projections.append(projection)
        training.append(
            {
                "states": states,
                "deltas": deltas,
                "digits": digits,
                "identity_mask": identity_mask,
            }
        )
        metadata.append(
            {
                "step": step,
                "targeted_pairs": len(pairs),
                "identity_pairs": len(examples),
                "unique_recipient_contexts": len(set(recipient_inputs)),
                "unique_donor_contexts": len(set(donor_inputs)),
                "states_sha256": tensor_sha256(states),
                "deltas_sha256": tensor_sha256(deltas),
                "mean_target_transport_relative_norm": float(
                    (
                        targeted_deltas.norm(dim=1)
                        / recipient_states.norm(dim=1)
                    ).mean()
                ),
            }
        )
    return projections, training, metadata


def train_candidates(
    projections: list[AdapterProjection],
    training: list[dict[str, torch.Tensor]],
    *,
    config: dict[str, Any],
) -> tuple[list[dict[tuple[int, int], AdapterEnsemble]], list[dict[str, Any]]]:
    all_candidates = []
    histories = []
    for step, (projection, data) in enumerate(zip(projections, training, strict=True)):
        candidates = {}
        step_histories = []
        for hidden_width in config["hidden_widths"]:
            for transport_rank in config["transport_ranks"]:
                members = []
                for seed in config["adapter_seeds"]:
                    adapter, history = fit_transport_adapter(
                        projection,
                        data["states"],
                        data["deltas"],
                        data["digits"],
                        data["identity_mask"],
                        hidden_width=hidden_width,
                        transport_rank=transport_rank,
                        epochs=config["epochs"],
                        learning_rate=config["learning_rate"],
                        batch_size=config["adapter_batch_size"],
                        identity_weight=config["identity_weight"],
                        norm_cap=config["norm_cap"],
                        norm_penalty=config["norm_penalty"],
                        seed=seed + step * 100,
                        device=config["adapter_device"],
                    )
                    members.append(adapter)
                    step_histories.append(
                        {
                            "hidden_width": hidden_width,
                            "transport_rank": transport_rank,
                            "seed": seed + step * 100,
                            "initial_loss": history[0],
                            "final_loss": history[-1],
                        }
                    )
                candidates[(hidden_width, transport_rank)] = AdapterEnsemble(members)
        all_candidates.append(candidates)
        histories.append({"step": step, "runs": step_histories})
    return all_candidates, histories


def select_adapters(
    model: Any,
    tokenizer: Any,
    capture: ActivationCapture,
    candidates: list[dict[tuple[int, int], AdapterEnsemble]],
    *,
    examples: list[Any],
    prompts: list[str],
    targets: list[int],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[list[AdapterEnsemble], list[dict[str, Any]]]:
    target_ids = result_token_ids(tokenizer, targets)
    original_results = [example.result for example in examples]
    original_ids = result_token_ids(tokenizer, original_results)
    hidden_index = config["hidden_state_index"]
    selected_adapters = []
    selections = []
    for step, step_candidates in enumerate(candidates):
        target_prompts = [
            prompt + str(result)[:step]
            for prompt, result in zip(prompts, targets, strict=True)
        ]
        identity_prompts = [
            prompt + str(result)[:step]
            for prompt, result in zip(prompts, original_results, strict=True)
        ]
        target_states = capture.capture_last_token(
            target_prompts,
            hidden_state_indices=[hidden_index],
            batch_size=config["base_model_batch_size"],
        )[hidden_index].values
        identity_states = capture.capture_last_token(
            identity_prompts,
            hidden_state_indices=[hidden_index],
            batch_size=config["base_model_batch_size"],
        )[hidden_index].values
        target_digits = torch.tensor([int(str(result)[step]) for result in targets])
        identity_digits = torch.tensor(
            [int(str(result)[step]) for result in original_results]
        )
        rows = []
        row_adapters = []
        for (hidden_width, transport_rank), ensemble in step_candidates.items():
            raw_target = ensemble.predict(target_states, target_digits)
            raw_identity = ensemble.predict(identity_states, identity_digits)
            for scale in config["scales"]:
                target_delta = raw_target * scale
                identity_delta = raw_identity * scale
                predicted_target = predict_ids(
                    model,
                    tokenizer,
                    target_prompts,
                    target_delta,
                    hidden_index=hidden_index,
                    batch_size=config["base_model_batch_size"],
                    device=device,
                )
                predicted_identity = predict_ids(
                    model,
                    tokenizer,
                    identity_prompts,
                    identity_delta,
                    hidden_index=hidden_index,
                    batch_size=config["base_model_batch_size"],
                    device=device,
                )
                target_accuracy = sum(
                    actual == expected[step]
                    for actual, expected in zip(
                        predicted_target, target_ids, strict=True
                    )
                ) / len(examples)
                identity_accuracy = sum(
                    actual == expected[step]
                    for actual, expected in zip(
                        predicted_identity, original_ids, strict=True
                    )
                ) / len(examples)
                rows.append(
                    {
                        "hidden_width": hidden_width,
                        "transport_rank": transport_rank,
                        "scale": scale,
                        "target_token_accuracy": target_accuracy,
                        "identity_token_accuracy": identity_accuracy,
                        "minimum_accuracy": min(target_accuracy, identity_accuracy),
                        "mean_target_relative_norm": float(
                            (
                                target_delta.norm(dim=1)
                                / target_states.norm(dim=1)
                            ).mean()
                        ),
                        "mean_identity_relative_norm": float(
                            (
                                identity_delta.norm(dim=1)
                                / identity_states.norm(dim=1)
                            ).mean()
                        ),
                    }
                )
                row_adapters.append(ensemble)
        selected_index = max(
            range(len(rows)),
            key=lambda index: (
                rows[index]["minimum_accuracy"],
                rows[index]["target_token_accuracy"],
                rows[index]["identity_token_accuracy"],
                -rows[index]["mean_target_relative_norm"],
                -rows[index]["transport_rank"],
                -rows[index]["hidden_width"],
            ),
        )
        selected_adapters.append(row_adapters[selected_index])
        selections.append(
            {"step": step, "selected": rows[selected_index], "candidates": rows}
        )
    return selected_adapters, selections


def evaluate_condition(
    condition: str,
    model: Any,
    tokenizer: Any,
    capture: ActivationCapture,
    adapters: list[AdapterEnsemble],
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
    for step, (adapter, selection) in enumerate(
        zip(adapters, selections, strict=True)
    ):
        step_prompts = [
            prompt + prefix for prompt, prefix in zip(prompts, prefixes, strict=True)
        ]
        states = capture.capture_last_token(
            step_prompts,
            hidden_state_indices=[hidden_index],
            batch_size=config["base_model_batch_size"],
        )[hidden_index].values
        target_digits = torch.tensor([int(str(value)[step]) for value in targets])
        scale = selection["selected"]["scale"]
        typed = adapter.predict(states, target_digits) * scale
        if condition == "base":
            delta = torch.zeros_like(typed)
        elif condition == "adapter":
            delta = typed
        elif condition == "same_digit":
            digits = torch.tensor([int(str(value)[step]) for value in originals])
            delta = adapter.predict(states, digits) * scale
        elif condition == "shuffled_target_norm_matched":
            digits = torch.tensor(
                [int(str(value)[step]) for value in shuffled_targets]
            )
            delta = norm_match(
                adapter.predict(states, digits) * scale,
                typed.norm(dim=1),
            )
        elif condition == "shuffled_state_norm_matched":
            shuffled_states = torch.cat((states[1:], states[:1]))
            delta = norm_match(
                adapter.predict(shuffled_states, target_digits) * scale,
                typed.norm(dim=1),
            )
        else:
            delta = random_norm_matched(
                tuple(typed.shape),
                typed.norm(dim=1),
                seed=config["random_control_seed"] + condition_index * 10 + step,
            )
        next_ids = predict_ids(
            model,
            tokenizer,
            step_prompts,
            delta,
            hidden_index=hidden_index,
            batch_size=config["base_model_batch_size"],
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


def save_adapters(
    path: Path,
    adapters: list[AdapterEnsemble],
) -> str:
    tensors = {}
    for step, ensemble in enumerate(adapters):
        projection = ensemble.members[0].projection
        for name in (
            "state_mean",
            "state_basis",
            "state_scale",
            "delta_basis",
            "coefficient_scale",
        ):
            tensors[f"step{step}.projection.{name}"] = getattr(projection, name)
        for member_index, member in enumerate(ensemble.members):
            for name, tensor in member.model.state_dict().items():
                tensors[f"step{step}.member{member_index}.model.{name}"] = tensor
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights-output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    dataset_config_path = Path(config["dataset_config"])
    if hashlib.sha256(dataset_config_path.read_bytes()).hexdigest() != config[
        "dataset_config_sha256"
    ]:
        raise SystemExit("dataset config hash mismatch")
    dataset_config = json.loads(dataset_config_path.read_text())
    prerequisite = Path(dataset_config["phase1_local_result"])
    if hashlib.sha256(prerequisite.read_bytes()).hexdigest() != dataset_config[
        "phase1_local_result_sha256"
    ]:
        raise SystemExit("Phase 1 result hash mismatch")
    previous = json.loads(prerequisite.read_text())
    examples = build_phase2_additions(**dataset_config["dataset"]["parameters"])
    observed_hash = phase2_addition_sha256(examples)
    if observed_hash != dataset_config["dataset"]["sha256"]:
        raise SystemExit("Phase 2 dataset hash mismatch")
    fit_examples = [example for example in examples if example.split == "fit"]
    selection_examples = [
        example for example in examples if example.split == "selection"
    ]
    development = [
        example for example in examples if example.split == "development"
    ]

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
    projections, training, fit_metadata = build_training_designs(
        capture,
        fit_examples,
        fit_prompts,
        config=config,
    )
    candidates, histories = train_candidates(
        projections,
        training,
        config=config,
    )
    selection_prompts = render_prompts(tokenizer, selection_examples)
    selection_targets = target_results(selection_examples)
    adapters, selections = select_adapters(
        model,
        tokenizer,
        capture,
        candidates,
        examples=selection_examples,
        prompts=selection_prompts,
        targets=selection_targets,
        config=config,
        device=device,
    )
    weights_sha256 = save_adapters(args.weights_output, adapters)

    development_prompts = render_prompts(tokenizer, development)
    development_targets = target_results(development)
    conditions = (
        "base",
        "adapter",
        "same_digit",
        "shuffled_target_norm_matched",
        "shuffled_state_norm_matched",
        "random_norm_matched",
    )
    condition_results = {
        condition: evaluate_condition(
            condition,
            model,
            tokenizer,
            capture,
            adapters,
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
        "schema_version": "oli.phase2-scaled-adapter/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "development_only",
        "model": previous["model"],
        "dataset": {
            "sha256": observed_hash,
            "fit_examples": len(fit_examples),
            "selection_examples": len(selection_examples),
            "development_examples": len(development),
            "audit_examples_unopened": sum(
                example.split == "audit" for example in examples
            ),
        },
        "adapter": {
            "type": "three-seed GELU bottleneck transport ensemble",
            "hidden_state_index": config["hidden_state_index"],
            "decoder_block": config["hidden_state_index"] - 1,
            "state_rank": config["state_rank"],
            "fit": fit_metadata,
            "training": histories,
            "selection": selections,
            "weights_path": str(args.weights_output),
            "weights_sha256": weights_sha256,
            "inference_inputs": ["recipient native state", "desired next digit"],
            "live_donor_required": False,
        },
        "conditions": condition_results,
        "advancement_gate": {
            "exact_target_minimum": 0.5,
            "per_position_target_minimum": 0.7,
            "control_advantage_minimum": 0.25,
            "identity_preservation_minimum": 0.9,
            "relative_norm_maximum": 1.0,
            "parse_rate_required": 1.0,
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
            "Development results cannot authorize audit unless every frozen "
            "advancement gate passes."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    print(f"wrote {args.weights_output}")


if __name__ == "__main__":
    main()
