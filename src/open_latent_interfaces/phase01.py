from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.dataset import Example, build_phase01_dataset
from open_latent_interfaces.evaluation import (
    first_digit_token_id,
    margin_delta,
    norm_match,
    random_norm_matched,
    token_metrics,
    wrong_digit_labels,
)
from open_latent_interfaces.interventions import intervened_next_token_logits
from open_latent_interfaces.manifest import environment_manifest, stable_json_sha256
from open_latent_interfaces.phase0 import choose_device, select_hidden_state_indices
from open_latent_interfaces.probes import (
    BinaryRidgeProbe,
    CategoricalRidgeProbe,
    ScalarRidgeProbe,
    binary_metrics,
    categorical_metrics,
    regression_metrics,
)


def _load_model(
    model_name: str,
    revision: str,
    *,
    device_name: str,
) -> tuple[Any, Any, torch.device]:
    device = choose_device(device_name)
    dtype = torch.float16 if device.type in {"mps", "cuda"} else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer, device


def _examples_for(examples: list[Example], split: str) -> list[Example]:
    return [example for example in examples if example.split == split]


def _labels(examples: list[Example]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    routes = torch.tensor([example.route for example in examples], dtype=torch.long)
    results = torch.tensor(
        [example.result if example.result is not None else -1 for example in examples],
        dtype=torch.float,
    )
    leading_digits = torch.tensor(
        [int(str(example.result)[0]) if example.result is not None else -1 for example in examples],
        dtype=torch.long,
    )
    return routes, results, leading_digits


def _positive_mask(examples: list[Example]) -> torch.Tensor:
    return torch.tensor([example.route == 1 for example in examples], dtype=torch.bool)


def _dataset_record(examples: list[Example]) -> dict[str, object]:
    rows = [example.to_dict() for example in examples]
    split_hashes = {
        split: stable_json_sha256(
            [example.to_dict() for example in examples if example.split == split]
        )
        for split in ("train", "development", "audit")
    }
    return {
        "sha256": stable_json_sha256(rows),
        "split_sha256": split_hashes,
        "examples": len(examples),
        "generator": "open_latent_interfaces.dataset.build_phase01_dataset",
    }


def _digit_token_ids(tokenizer: Any) -> dict[int, int]:
    values: dict[int, int] = {}
    for digit in range(10):
        token_id = first_digit_token_id(tokenizer, digit)
        if token_id is None:
            raise RuntimeError(f"digit {digit} is not a single tokenizer token")
        values[digit] = token_id
    return values


def _donor_directions(
    fit_values: torch.Tensor,
    fit_labels: torch.Tensor,
    target_values: torch.Tensor,
    target_labels: torch.Tensor,
) -> torch.Tensor:
    directions: list[torch.Tensor] = []
    for row, label in enumerate(target_labels.tolist()):
        candidates = torch.where(fit_labels == label)[0]
        if len(candidates) == 0:
            raise ValueError(f"no donor for leading digit {label}")
        donor = fit_values[candidates[row % len(candidates)]]
        directions.append(donor - target_values[row])
    return torch.stack(directions)


def _logit_directions(
    model: Any,
    digit_token_ids: dict[int, int],
    labels: torch.Tensor,
) -> torch.Tensor:
    output_weights = model.get_output_embeddings().weight.detach().float().cpu()
    digit_weights = torch.stack(
        [output_weights[digit_token_ids[digit]] for digit in range(10)]
    )
    directions: list[torch.Tensor] = []
    for label in labels.tolist():
        alternatives = torch.cat((digit_weights[:label], digit_weights[label + 1 :]))
        directions.append(digit_weights[label] - alternatives.mean(dim=0))
    return torch.stack(directions)


def _condition_deltas(
    *,
    digit_probe: CategoricalRidgeProbe,
    scalar_probe: ScalarRidgeProbe,
    fit_values: torch.Tensor,
    fit_digits: torch.Tensor,
    target_values: torch.Tensor,
    target_digits: torch.Tensor,
    target_results: torch.Tensor,
    model: Any,
    digit_token_ids: dict[int, int],
    strength: float,
    digit_margin: float,
    max_relative_norm: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    targeted = digit_probe.minimal_margin_shift(
        target_values,
        target_digits,
        margin=digit_margin,
        strength=strength,
        max_relative_norm=max_relative_norm,
    )
    target_norms = targeted.norm(dim=1)
    wrong = digit_probe.minimal_margin_shift(
        target_values,
        wrong_digit_labels(target_digits),
        margin=digit_margin,
        strength=strength,
        max_relative_norm=None,
    )
    scalar = scalar_probe.minimal_shift(
        target_values,
        target_results,
        strength=strength,
        max_relative_norm=None,
    )
    donor = _donor_directions(
        fit_values,
        fit_digits,
        target_values,
        target_digits,
    )
    logit = _logit_directions(model, digit_token_ids, target_digits)
    return {
        "targeted_digit_probe": targeted,
        "wrong_digit_probe_control": norm_match(wrong, target_norms),
        "random_direction_control": random_norm_matched(
            tuple(targeted.shape),
            target_norms,
            seed=seed,
        ),
        "scalar_sum_probe": norm_match(scalar, target_norms),
        "same_digit_donor": norm_match(donor, target_norms),
        "digit_logit_direction": norm_match(logit, target_norms),
    }


def _causal_conditions(
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompts: list[str],
    target_ids: torch.Tensor,
    hidden_state_index: int,
    deltas: dict[str, torch.Tensor],
    batch_size: int,
) -> dict[str, object]:
    capture = ActivationCapture(model, tokenizer, device=device)
    base_logits = capture.next_token_logits(prompts, batch_size=batch_size)
    records: dict[str, object] = {
        "base": token_metrics(base_logits, target_ids),
        "conditions": {},
    }
    condition_records: dict[str, object] = {}
    for name, condition_delta in deltas.items():
        if bool((condition_delta.norm(dim=1) == 0).all()):
            logits = base_logits
        else:
            logits = intervened_next_token_logits(
                model,
                tokenizer,
                prompts,
                hidden_state_index=hidden_state_index,
                deltas=condition_delta,
                device=device,
            )
        condition_records[name] = {
            "metrics": token_metrics(logits, target_ids),
            "mean_delta_norm": float(condition_delta.norm(dim=1).mean()),
            "active_rows": int((condition_delta.norm(dim=1) > 0).sum()),
            "mean_margin_delta": margin_delta(base_logits, logits, target_ids),
        }
    records["conditions"] = condition_records
    targeted = condition_records["targeted_digit_probe"]["mean_margin_delta"]
    wrong = condition_records["wrong_digit_probe_control"]["mean_margin_delta"]
    random_control = condition_records["random_direction_control"]["mean_margin_delta"]
    records["targeted_control_advantage"] = float(targeted - max(wrong, random_control))
    return records


def _probe_metrics(
    *,
    fit_values: torch.Tensor,
    fit_examples: list[Example],
    target_values: torch.Tensor,
    target_examples: list[Example],
    l2: float,
) -> tuple[
    BinaryRidgeProbe,
    ScalarRidgeProbe,
    CategoricalRidgeProbe,
    dict[str, object],
]:
    fit_route, fit_result, fit_digit = _labels(fit_examples)
    target_route, target_result, target_digit = _labels(target_examples)
    fit_positive = _positive_mask(fit_examples)
    target_positive = _positive_mask(target_examples)
    route_probe = BinaryRidgeProbe.fit(fit_values, fit_route, l2=l2)
    scalar_probe = ScalarRidgeProbe.fit(
        fit_values[fit_positive],
        fit_result[fit_positive],
        l2=l2,
    )
    digit_probe = CategoricalRidgeProbe.fit(
        fit_values[fit_positive],
        fit_digit[fit_positive],
        number_of_classes=10,
        l2=l2,
    )
    metrics: dict[str, object] = {
        "route": binary_metrics(route_probe.score(target_values), target_route),
        "scalar_sum": regression_metrics(
            scalar_probe.predict(target_values[target_positive]),
            target_result[target_positive],
        ),
        "leading_digit": categorical_metrics(
            digit_probe.score(target_values[target_positive]),
            target_digit[target_positive],
            number_of_classes=10,
        ),
    }
    return route_probe, scalar_probe, digit_probe, metrics


def _target_payload(
    examples: list[Example],
    values: torch.Tensor,
    tokenizer: Any,
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    positive = _positive_mask(examples)
    positives = [example for example in examples if example.route]
    prompts = [example.prompt for example in positives]
    results = torch.tensor([example.result for example in positives], dtype=torch.float)
    target_ids = torch.tensor(
        [first_digit_token_id(tokenizer, int(example.result)) for example in positives],
        dtype=torch.long,
    )
    if bool((target_ids < 0).any()):
        raise RuntimeError("a leading digit did not map to a tokenizer token")
    return prompts, values[positive], results, target_ids


def develop(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object]]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    repo = Path(__file__).resolve().parents[2]
    examples = build_phase01_dataset(
        seed=args.seed,
        train_pairs_per_digit=args.train_pairs_per_digit,
        development_pairs_per_digit=args.development_pairs_per_digit,
        audit_pairs_per_digit=args.audit_pairs_per_digit,
    )
    dataset_record = _dataset_record(examples)
    train_examples = _examples_for(examples, "train")
    development_examples = _examples_for(examples, "development")
    fit_and_development = train_examples + development_examples
    model, tokenizer, device = _load_model(
        args.model,
        args.revision,
        device_name=args.device,
    )
    number_of_blocks = int(model.config.num_hidden_layers)
    indices = select_hidden_state_indices(number_of_blocks, args.hidden_state_index)
    capture = ActivationCapture(model, tokenizer, device=device)
    captured = capture.capture_last_token(
        [example.prompt for example in fit_and_development],
        hidden_state_indices=indices,
        batch_size=args.batch_size,
    )
    train_count = len(train_examples)
    digit_token_ids = _digit_token_ids(tokenizer)
    development_records: dict[str, object] = {}
    selection_candidates: list[tuple[tuple[float, float, float], int, float]] = []
    selection_limit = int(number_of_blocks * args.selection_max_layer_fraction)

    for index in indices:
        fit_values = captured[index].values[:train_count]
        development_values = captured[index].values[train_count:]
        _, scalar_probe, digit_probe, metrics = _probe_metrics(
            fit_values=fit_values,
            fit_examples=train_examples,
            target_values=development_values,
            target_examples=development_examples,
            l2=args.probe_l2,
        )
        fit_positive = _positive_mask(train_examples)
        fit_digits = _labels(train_examples)[2][fit_positive]
        prompts, target_values, target_results, target_ids = _target_payload(
            development_examples,
            development_values,
            tokenizer,
        )
        target_digits = torch.tensor(
            [int(str(example.result)[0]) for example in development_examples if example.route]
        )
        strength_records: dict[str, object] = {}
        for strength in args.graft_strength:
            deltas = _condition_deltas(
                digit_probe=digit_probe,
                scalar_probe=scalar_probe,
                fit_values=fit_values[fit_positive],
                fit_digits=fit_digits,
                target_values=target_values,
                target_digits=target_digits,
                target_results=target_results,
                model=model,
                digit_token_ids=digit_token_ids,
                strength=strength,
                digit_margin=args.digit_margin,
                max_relative_norm=args.max_relative_norm,
                seed=args.seed + index * 101 + round(strength * 10),
            )
            causal = _causal_conditions(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompts=prompts,
                target_ids=target_ids,
                hidden_state_index=index,
                deltas=deltas,
                batch_size=args.batch_size,
            )
            strength_records[str(strength)] = causal
            if index <= selection_limit:
                targeted_metrics = causal["conditions"]["targeted_digit_probe"]["metrics"]
                base_metrics = causal["base"]
                score = (
                    float(causal["targeted_control_advantage"]),
                    float(targeted_metrics["top1_exact"] - base_metrics["top1_exact"]),
                    -float(
                        causal["conditions"]["targeted_digit_probe"]["mean_delta_norm"]
                    ),
                )
                selection_candidates.append((score, index, strength))
        development_records[str(index)] = {
            "probe_metrics": metrics,
            "strengths": strength_records,
            "eligible_for_internal_selection": index <= selection_limit,
        }

    best_score, selected_index, selected_strength = max(selection_candidates)
    resolved_revision = getattr(model.config, "_commit_hash", None)
    frozen_config: dict[str, object] = {
        "study": "phase01-native-math-cartography",
        "stage": "frozen_configuration",
        "model": args.model,
        "resolved_revision": resolved_revision,
        "seed": args.seed,
        "train_pairs_per_digit": args.train_pairs_per_digit,
        "development_pairs_per_digit": args.development_pairs_per_digit,
        "audit_pairs_per_digit": args.audit_pairs_per_digit,
        "dataset_sha256": dataset_record["sha256"],
        "split_sha256": dataset_record["split_sha256"],
        "hidden_state_indices_developed": indices,
        "selected_hidden_state_index": selected_index,
        "selected_strength": selected_strength,
        "selection_score": list(best_score),
        "selection_max_layer_fraction": args.selection_max_layer_fraction,
        "probe_l2": args.probe_l2,
        "digit_margin": args.digit_margin,
        "max_relative_norm": args.max_relative_norm,
        "batch_size": args.batch_size,
        "fit_policy_for_audit": "refit probes on train plus development",
        "audit_policy": "one run; no audit-driven layer, strength, or bridge changes",
        "source_environment": environment_manifest(repo),
    }
    output: dict[str, object] = {
        "study": "phase01-native-math-cartography",
        "stage": "development",
        "status": "development_complete",
        "claim_boundary": (
            "Development selects a frozen internal layer and strength. It does not "
            "report audit performance or establish a latent graft."
        ),
        "dataset": dataset_record,
        "model": {
            "name": args.model,
            "resolved_revision": resolved_revision,
            "number_of_blocks": number_of_blocks,
            "frozen_parameters": all(
                not parameter.requires_grad for parameter in model.parameters()
            ),
        },
        "selection_policy": {
            "maximum_hidden_state_index": selection_limit,
            "primary": "targeted margin advantage over max(wrong-digit, random)",
            "tie_breakers": ["top1 gain over base", "smaller mean intervention norm"],
        },
        "development_layers": development_records,
        "frozen_config": frozen_config,
        "environment": environment_manifest(repo),
    }
    return output, frozen_config


def audit(args: argparse.Namespace) -> dict[str, object]:
    repo = Path(__file__).resolve().parents[2]
    config = json.loads(args.config.read_text())
    if config.get("stage") != "frozen_configuration":
        raise ValueError("config is not a Phase 0.1 frozen configuration")
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    examples = build_phase01_dataset(
        seed=seed,
        train_pairs_per_digit=int(config["train_pairs_per_digit"]),
        development_pairs_per_digit=int(config["development_pairs_per_digit"]),
        audit_pairs_per_digit=int(config["audit_pairs_per_digit"]),
    )
    dataset_record = _dataset_record(examples)
    if dataset_record["sha256"] != config["dataset_sha256"]:
        raise RuntimeError("dataset hash differs from frozen configuration")
    fit_examples = _examples_for(examples, "train") + _examples_for(
        examples,
        "development",
    )
    audit_examples = _examples_for(examples, "audit")
    model, tokenizer, device = _load_model(
        str(config["model"]),
        str(config["resolved_revision"]),
        device_name=args.device,
    )
    selected_index = int(config["selected_hidden_state_index"])
    combined = fit_examples + audit_examples
    capture = ActivationCapture(model, tokenizer, device=device)
    values = capture.capture_last_token(
        [example.prompt for example in combined],
        hidden_state_indices=[selected_index],
        batch_size=int(config["batch_size"]),
    )[selected_index].values
    fit_count = len(fit_examples)
    fit_values = values[:fit_count]
    audit_values = values[fit_count:]
    _, scalar_probe, digit_probe, probe_metrics = _probe_metrics(
        fit_values=fit_values,
        fit_examples=fit_examples,
        target_values=audit_values,
        target_examples=audit_examples,
        l2=float(config["probe_l2"]),
    )
    fit_positive = _positive_mask(fit_examples)
    fit_digits = _labels(fit_examples)[2][fit_positive]
    prompts, target_values, target_results, target_ids = _target_payload(
        audit_examples,
        audit_values,
        tokenizer,
    )
    target_digits = torch.tensor(
        [int(str(example.result)[0]) for example in audit_examples if example.route]
    )
    deltas = _condition_deltas(
        digit_probe=digit_probe,
        scalar_probe=scalar_probe,
        fit_values=fit_values[fit_positive],
        fit_digits=fit_digits,
        target_values=target_values,
        target_digits=target_digits,
        target_results=target_results,
        model=model,
        digit_token_ids=_digit_token_ids(tokenizer),
        strength=float(config["selected_strength"]),
        digit_margin=float(config["digit_margin"]),
        max_relative_norm=float(config["max_relative_norm"]),
        seed=seed + selected_index * 101 + round(float(config["selected_strength"]) * 10),
    )
    causal = _causal_conditions(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompts=prompts,
        target_ids=target_ids,
        hidden_state_index=selected_index,
        deltas=deltas,
        batch_size=int(config["batch_size"]),
    )
    return {
        "study": "phase01-native-math-cartography",
        "stage": "audit",
        "status": "audit_complete",
        "claim_boundary": (
            "This is a frozen, template-held-out audit of first-digit read/write "
            "diagnostics. It is not multi-token exact arithmetic or an end-to-end graft."
        ),
        "frozen_config_sha256": stable_json_sha256(config),
        "frozen_config": config,
        "dataset": dataset_record,
        "audit_probe_metrics": probe_metrics,
        "audit_causal_conditions": causal,
        "model": {
            "name": config["model"],
            "resolved_revision": getattr(model.config, "_commit_hash", None),
            "frozen_parameters": all(
                not parameter.requires_grad for parameter in model.parameters()
            ),
        },
        "environment": environment_manifest(repo),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 0.1 development and frozen audit.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    develop_parser = subparsers.add_parser("develop")
    develop_parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    develop_parser.add_argument("--revision", default="main")
    develop_parser.add_argument("--device", default="auto")
    develop_parser.add_argument("--seed", type=int, default=240801)
    develop_parser.add_argument("--train-pairs-per-digit", type=int, default=12)
    develop_parser.add_argument("--development-pairs-per-digit", type=int, default=4)
    develop_parser.add_argument("--audit-pairs-per-digit", type=int, default=4)
    develop_parser.add_argument("--batch-size", type=int, default=16)
    develop_parser.add_argument("--probe-l2", type=float, default=10.0)
    develop_parser.add_argument("--digit-margin", type=float, default=1.0)
    develop_parser.add_argument("--max-relative-norm", type=float, default=0.15)
    develop_parser.add_argument("--selection-max-layer-fraction", type=float, default=0.8)
    develop_parser.add_argument(
        "--hidden-state-index",
        type=int,
        action="append",
    )
    develop_parser.add_argument(
        "--graft-strength",
        type=float,
        action="append",
        default=None,
    )
    develop_parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase01_development.json"),
    )
    develop_parser.add_argument(
        "--config-output",
        type=Path,
        default=Path("configs/phase01_frozen.json"),
    )

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase01_frozen.json"),
    )
    audit_parser.add_argument("--device", default="auto")
    audit_parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase01_audit.json"),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "develop":
        if args.graft_strength is None:
            args.graft_strength = [0.5, 1.0, 2.0, 4.0]
        result, config = develop(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.config_output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        args.config_output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
