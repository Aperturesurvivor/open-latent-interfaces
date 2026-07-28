from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.dataset import (
    Example,
    assert_dataset_invariants,
    build_phase0_dataset,
)
from open_latent_interfaces.evaluation import (
    first_digit_token_id,
    margin_delta,
    token_metrics,
)
from open_latent_interfaces.interventions import intervened_next_token_logits
from open_latent_interfaces.manifest import (
    environment_manifest,
    stable_json_sha256,
)
from open_latent_interfaces.probes import (
    BinaryRidgeProbe,
    CategoricalRidgeProbe,
    ScalarRidgeProbe,
    binary_metrics,
    categorical_metrics,
    regression_metrics,
)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_hidden_state_indices(number_of_blocks: int, requested: list[int] | None) -> list[int]:
    if requested:
        indices = sorted(set(requested))
    else:
        fractions = (0.2, 0.4, 0.6, 0.8, 1.0)
        indices = sorted(
            {max(1, min(number_of_blocks, round(number_of_blocks * value))) for value in fractions}
        )
    if indices[0] < 1 or indices[-1] > number_of_blocks:
        raise ValueError(
            f"hidden-state indices must be in [1, {number_of_blocks}], got {indices}"
        )
    return indices


def _mask(examples: list[Example], *, split: str, positive_only: bool = False) -> torch.Tensor:
    return torch.tensor(
        [
            example.split == split and (not positive_only or example.route == 1)
            for example in examples
        ],
        dtype=torch.bool,
    )


def _model_revision(model: Any) -> str | None:
    return getattr(model.config, "_commit_hash", None)


def _shuffle(values: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return values[torch.randperm(len(values), generator=generator)]


def run(args: argparse.Namespace) -> dict[str, object]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    repo = Path(__file__).resolve().parents[2]
    device = choose_device(args.device)
    dtype = torch.float16 if device.type in {"mps", "cuda"} else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    examples = build_phase0_dataset(
        seed=args.seed,
        train_pairs=args.train_pairs,
        development_pairs=args.development_pairs,
        test_pairs=args.test_pairs,
    )
    assert_dataset_invariants(examples)
    dataset_rows = [example.to_dict() for example in examples]
    prompts = [example.prompt for example in examples]
    number_of_blocks = int(model.config.num_hidden_layers)
    indices = select_hidden_state_indices(number_of_blocks, args.hidden_state_index)
    capture = ActivationCapture(model, tokenizer, device=device)
    layers = capture.capture_last_token(
        prompts,
        hidden_state_indices=indices,
        batch_size=args.batch_size,
    )

    train = _mask(examples, split="train")
    development = _mask(examples, split="development")
    test = _mask(examples, split="test")
    train_positive = _mask(examples, split="train", positive_only=True)
    development_positive = _mask(examples, split="development", positive_only=True)
    test_positive = _mask(examples, split="test", positive_only=True)
    route_labels = torch.tensor([example.route for example in examples])
    results = torch.tensor(
        [example.result if example.result is not None else -1 for example in examples],
        dtype=torch.float,
    )
    leading_digits = torch.tensor(
        [int(str(example.result)[0]) if example.result is not None else -1 for example in examples],
        dtype=torch.long,
    )

    layer_metrics: dict[str, object] = {}
    development_selection: dict[int, tuple[float, float]] = {}
    for index, captured in layers.items():
        values = captured.values
        route_probe = BinaryRidgeProbe.fit(
            values[train],
            route_labels[train],
            l2=args.probe_l2,
        )
        sum_probe = ScalarRidgeProbe.fit(
            values[train_positive],
            results[train_positive],
            l2=args.probe_l2,
        )
        digit_probe = CategoricalRidgeProbe.fit(
            values[train_positive],
            leading_digits[train_positive],
            number_of_classes=10,
            l2=args.probe_l2,
        )
        route_development = binary_metrics(
            route_probe.score(values[development]),
            route_labels[development],
        )
        route_test = binary_metrics(
            route_probe.score(values[test]),
            route_labels[test],
        )
        sum_development = regression_metrics(
            sum_probe.predict(values[development_positive]),
            results[development_positive],
        )
        sum_test = regression_metrics(
            sum_probe.predict(values[test_positive]),
            results[test_positive],
        )
        digit_development = categorical_metrics(
            digit_probe.score(values[development_positive]),
            leading_digits[development_positive],
            number_of_classes=10,
        )
        digit_test = categorical_metrics(
            digit_probe.score(values[test_positive]),
            leading_digits[test_positive],
            number_of_classes=10,
        )
        development_selection[index] = (
            float(digit_development["accuracy"]),
            sum_development["r2"],
        )
        layer_metrics[str(index)] = {
            "route_development": route_development,
            "route_test": route_test,
            "sum_development": sum_development,
            "sum_test": sum_test,
            "leading_digit_development": digit_development,
            "leading_digit_test": digit_test,
        }

    best_index = max(development_selection, key=development_selection.get)
    best_values = layers[best_index].values
    fit_positive = train_positive | development_positive
    final_sum_probe = ScalarRidgeProbe.fit(
        best_values[fit_positive],
        results[fit_positive],
        l2=args.probe_l2,
    )
    final_digit_probe = CategoricalRidgeProbe.fit(
        best_values[fit_positive],
        leading_digits[fit_positive],
        number_of_classes=10,
        l2=args.probe_l2,
    )

    test_examples = [example for example in examples if example.split == "test" and example.route]
    test_values = best_values[test_positive]
    desired_results = results[test_positive]
    target_ids_and_mask = [
        first_digit_token_id(tokenizer, int(example.result)) for example in test_examples
    ]
    fit_digit_support = set(leading_digits[fit_positive].tolist())
    test_digit_labels = leading_digits[test_positive]
    eligible = torch.tensor(
        [
            token_id is not None and int(label) in fit_digit_support
            for token_id, label in zip(
                target_ids_and_mask,
                test_digit_labels.tolist(),
                strict=True,
            )
        ]
    )
    causal: dict[str, object]
    if not bool(eligible.any()):
        causal = {
            "status": "not_run",
            "reason": "no test result had a verified single-token first digit",
        }
    else:
        eligible_prompts = [
            example.prompt
            for example, keep in zip(test_examples, eligible.tolist(), strict=True)
            if keep
        ]
        eligible_values = test_values[eligible]
        eligible_digit_labels = test_digit_labels[eligible]
        target_ids = torch.tensor(
            [value for value in target_ids_and_mask if value is not None],
            dtype=torch.long,
        )
        base_logits = capture.next_token_logits(
            eligible_prompts,
            batch_size=args.batch_size,
        )

        strengths = tuple(args.graft_strength)
        strength_records: dict[str, object] = {}
        chosen_strength = strengths[0]
        best_margin = -float("inf")
        for strength in strengths:
            deltas = final_digit_probe.minimal_margin_shift(
                eligible_values,
                eligible_digit_labels,
                margin=args.digit_margin,
                strength=strength,
                max_relative_norm=args.max_relative_norm,
            )
            graft_logits = intervened_next_token_logits(
                model,
                tokenizer,
                eligible_prompts,
                hidden_state_index=best_index,
                deltas=deltas,
                device=device,
            )
            shuffled_deltas = final_digit_probe.minimal_margin_shift(
                eligible_values,
                _shuffle(eligible_digit_labels, args.seed + 91),
                margin=args.digit_margin,
                strength=strength,
                max_relative_norm=args.max_relative_norm,
            )
            shuffled_logits = intervened_next_token_logits(
                model,
                tokenizer,
                eligible_prompts,
                hidden_state_index=best_index,
                deltas=shuffled_deltas,
                device=device,
            )
            targeted_margin_delta = margin_delta(base_logits, graft_logits, target_ids)
            shuffled_margin_delta = margin_delta(base_logits, shuffled_logits, target_ids)
            strength_records[str(strength)] = {
                "delta_norm_mean": float(deltas.norm(dim=1).mean()),
                "activation_norm_mean": float(eligible_values.norm(dim=1).mean()),
                "targeted": token_metrics(graft_logits, target_ids),
                "shuffled_target_control": token_metrics(shuffled_logits, target_ids),
                "targeted_margin_delta": targeted_margin_delta,
                "shuffled_margin_delta": shuffled_margin_delta,
                "targeted_minus_shuffled_margin_delta": (
                    targeted_margin_delta - shuffled_margin_delta
                ),
            }
            if targeted_margin_delta - shuffled_margin_delta > best_margin:
                best_margin = targeted_margin_delta - shuffled_margin_delta
                chosen_strength = strength

        causal = {
            "status": "pilot_complete",
            "eligible_examples": int(eligible.sum()),
            "ineligible_first_digit_examples": int((~eligible).sum()),
            "endpoint": "first generated result digit only",
            "write_bridge": "leading-digit categorical ridge margin direction",
            "fit_digit_support": sorted(fit_digit_support),
            "unused_scalar_probe_test_metrics": regression_metrics(
                final_sum_probe.predict(test_values),
                desired_results,
            ),
            "base": token_metrics(base_logits, target_ids),
            "strength_sweep": strength_records,
            "pilot_selected_strength": chosen_strength,
            "selection_warning": (
                "Strength was selected on the pilot test set; it must be frozen "
                "on a new audit before any confirmatory claim."
            ),
        }

    output: dict[str, object] = {
        "study": "phase0-native-mathematical-channel-pilot",
        "status": "pilot",
        "claim_boundary": (
            "This run tests frozen-model observability and a probe-defined oracle-result "
            "intervention. It is not an NLA, an end-to-end deterministic graft, or proof "
            "of a model-native mathematical API."
        ),
        "model": {
            "name": args.model,
            "requested_revision": args.revision,
            "resolved_revision": _model_revision(model),
            "frozen_parameters": all(
                not parameter.requires_grad for parameter in model.parameters()
            ),
            "number_of_blocks": number_of_blocks,
        },
        "configuration": {
            "seed": args.seed,
            "hidden_state_indices": indices,
            "batch_size": args.batch_size,
            "probe_l2": args.probe_l2,
            "graft_strengths": list(args.graft_strength),
            "max_relative_norm": args.max_relative_norm,
            "digit_margin": args.digit_margin,
        },
        "dataset": {
            "sha256": stable_json_sha256(dataset_rows),
            "examples": len(examples),
            "train_examples": int(train.sum()),
            "development_examples": int(development.sum()),
            "test_examples": int(test.sum()),
            "generator": "open_latent_interfaces.dataset.build_phase0_dataset",
        },
        "layer_metrics": layer_metrics,
        "pilot_selected_hidden_state_index": best_index,
        "causal_result_intervention": causal,
        "environment": environment_manifest(repo),
    }
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Phase 0 frozen-model latent math channel pilot."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=240727)
    parser.add_argument("--train-pairs", type=int, default=96)
    parser.add_argument("--development-pairs", type=int, default=32)
    parser.add_argument("--test-pairs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--probe-l2", type=float, default=10.0)
    parser.add_argument(
        "--hidden-state-index",
        type=int,
        action="append",
        help="HF hidden-state index to inspect; repeat for multiple indices.",
    )
    parser.add_argument(
        "--graft-strength",
        type=float,
        action="append",
        default=None,
        help="Probe-defined intervention strength; repeat to sweep.",
    )
    parser.add_argument("--max-relative-norm", type=float, default=0.15)
    parser.add_argument("--digit-margin", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("results/phase0_local_pilot.json"))
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a small dataset and two layers to verify the complete pipeline.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.graft_strength is None:
        args.graft_strength = [0.5, 1.0, 2.0]
    if args.smoke:
        args.train_pairs = 24
        args.development_pairs = 8
        args.test_pairs = 8
        if args.hidden_state_index is None:
            args.hidden_state_index = [12, 24]
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
