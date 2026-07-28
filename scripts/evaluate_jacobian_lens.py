#!/usr/bin/env python3
"""Compare a fitted Jacobian lens with vanilla logit lens on Phase 0.1 math."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

import torch

from open_latent_interfaces.dataset import build_phase01_dataset
from open_latent_interfaces.interpretability import write_jsonl
from open_latent_interfaces.interpretability_backends import JacobianLensAdapter


def rank_of(logits: torch.Tensor, token_id: int) -> int:
    target = logits[token_id]
    return int((logits > target).sum().item()) + 1


def summarize(
    ranks: list[int], digit_ranks: list[int], digit_margins: list[float]
) -> dict[str, float | int]:
    reciprocal_rank = sum(1.0 / rank for rank in ranks) / len(ranks)
    return {
        "n": len(ranks),
        "median_rank": float(median(ranks)),
        "mean_reciprocal_rank": reciprocal_rank,
        "hit_at_1": sum(rank <= 1 for rank in ranks) / len(ranks),
        "hit_at_10": sum(rank <= 10 for rank in ranks) / len(ranks),
        "hit_at_100": sum(rank <= 100 for rank in ranks) / len(ranks),
        "leading_digit_accuracy": sum(rank == 1 for rank in digit_ranks)
        / len(digit_ranks),
        "leading_digit_median_rank": float(median(digit_ranks)),
        "leading_digit_mean_margin": sum(digit_margins) / len(digit_margins),
        "leading_digit_median_margin": float(median(digit_margins)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-config", type=Path, required=True)
    parser.add_argument("--lens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--readouts", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--artifact-examples", type=int, default=4)
    args = parser.parse_args()

    import jlens
    import transformers

    fit_config = json.loads(args.fit_config.read_text())
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        fit_config["model"]["id"], revision=fit_config["model"]["revision"]
    )
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        fit_config["model"]["id"],
        revision=fit_config["model"]["revision"],
        dtype=dtype,
    ).to(device)
    model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.load(str(args.lens))
    lens_diagnostics = {}
    identity = torch.eye(lens.d_model)
    identity_norm = identity.norm().item()
    for layer, jacobian in sorted(lens.jacobians.items()):
        matrix = jacobian.float()
        matrix_norm = matrix.norm().item()
        lens_diagnostics[str(layer)] = {
            "finite": bool(torch.isfinite(matrix).all()),
            "frobenius_norm_over_sqrt_d": matrix_norm / (lens.d_model**0.5),
            "mean_absolute_entry": float(matrix.abs().mean()),
            "cosine_with_identity": float(
                (matrix * identity).sum().item()
                / max(matrix_norm * identity_norm, 1e-12)
            ),
        }
    lens.jacobians = {
        layer: jacobian.to(device) for layer, jacobian in lens.jacobians.items()
    }

    examples = [
        example
        for example in build_phase01_dataset()
        if example.split == "development" and example.kind == "addition"
    ]
    if args.limit is not None:
        examples = examples[: args.limit]

    ranks: dict[str, dict[int, list[int]]] = {
        "jacobian_lens": defaultdict(list),
        "logit_lens": defaultdict(list),
    }
    digit_ranks: dict[str, dict[int, list[int]]] = {
        "jacobian_lens": defaultdict(list),
        "logit_lens": defaultdict(list),
    }
    digit_margins: dict[str, dict[int, list[float]]] = {
        "jacobian_lens": defaultdict(list),
        "logit_lens": defaultdict(list),
    }
    final_ranks: list[int] = []
    final_digit_ranks: list[int] = []
    final_digit_margins: list[float] = []
    sample_rows: list[dict[str, Any]] = []
    readout_artifacts = []
    lens_sha256 = hashlib.sha256(args.lens.read_bytes()).hexdigest()
    adapter = JacobianLensAdapter(
        lens,
        model,
        tokenizer,
        lens_checkpoint=str(args.lens),
        lens_checkpoint_revision=lens_sha256,
    )
    artifact_layers = {3, 7, 11, 15, 19}
    digit_token_ids = [
        int(tokenizer(str(digit), add_special_tokens=False)["input_ids"][0])
        for digit in range(1, 10)
    ]
    started = time.perf_counter()

    for example_index, example in enumerate(examples):
        assert example.result is not None
        correct_digit_index = int(str(example.result)[0]) - 1
        target_tokens = tokenizer(
            str(example.result), add_special_tokens=False
        )["input_ids"]
        target_id = int(target_tokens[0])
        input_ids = model.encode(example.prompt, max_length=128)
        record_at = sorted(set(lens.source_layers) | {model.n_layers - 1})
        with jlens.ActivationRecorder(model.layers, at=record_at) as recorder:
            model.forward(input_ids)
            activations = {
                layer: recorder.activations[layer].detach() for layer in record_at
            }

        top_tokens: dict[str, Any] = {}
        for layer in lens.source_layers:
            residual = activations[layer][0, -1].float()
            jacobian_logits = model.unembed(lens.transport(residual, layer)).reshape(-1)
            logit_logits = model.unembed(residual).reshape(-1)
            for method, logits in (
                ("jacobian_lens", jacobian_logits),
                ("logit_lens", logit_logits),
            ):
                ranks[method][layer].append(rank_of(logits, target_id))
                scores = logits[digit_token_ids]
                correct_score = scores[correct_digit_index]
                wrong_scores = torch.cat(
                    (scores[:correct_digit_index], scores[correct_digit_index + 1 :])
                )
                digit_ranks[method][layer].append(
                    int((scores > correct_score).sum().item()) + 1
                )
                digit_margins[method][layer].append(
                    float((correct_score - wrong_scores.max()).item())
                )
            if example_index < args.artifact_examples and layer in artifact_layers:
                artifact = adapter.readout(
                    residual,
                    example_id=example.example_id,
                    target_model=fit_config["model"]["id"],
                    target_model_revision=fit_config["model"]["revision"],
                    hidden_state_index=layer + 1,
                    token_position=-1,
                    top_k=10,
                )
                readout_artifacts.append(artifact)
                top_tokens[str(layer)] = artifact.observation["tokens"][:5]

        final_logits = model.unembed(
            activations[model.n_layers - 1][0, -1].float()
        ).reshape(-1)
        final_ranks.append(rank_of(final_logits, target_id))
        final_digit_scores = final_logits[digit_token_ids]
        final_correct_score = final_digit_scores[correct_digit_index]
        final_wrong_scores = torch.cat(
            (
                final_digit_scores[:correct_digit_index],
                final_digit_scores[correct_digit_index + 1 :],
            )
        )
        final_digit_ranks.append(
            int((final_digit_scores > final_correct_score).sum().item()) + 1
        )
        final_digit_margins.append(
            float((final_correct_score - final_wrong_scores.max()).item())
        )
        if example_index < args.artifact_examples:
            sample_rows.append(
                {
                    "example_id": example.example_id,
                    "operand_a": example.operand_a,
                    "operand_b": example.operand_b,
                    "result": example.result,
                    "target_first_token_id": target_id,
                    "target_first_token": tokenizer.decode([target_id]),
                    "result_token_count": len(target_tokens),
                    "jacobian_top_tokens": top_tokens,
                }
            )

    summaries = {
        method: {
            str(layer): summarize(
                layer_ranks,
                digit_ranks[method][layer],
                digit_margins[method][layer],
            )
            for layer, layer_ranks in sorted(method_ranks.items())
        }
        for method, method_ranks in ranks.items()
    }
    best_layers = {
        method: min(
            layer_metrics,
            key=lambda layer: (
                -layer_metrics[layer]["leading_digit_accuracy"],
                -layer_metrics[layer]["leading_digit_mean_margin"],
                -layer_metrics[layer]["mean_reciprocal_rank"],
                int(layer),
            ),
        )
        for method, layer_metrics in summaries.items()
    }
    report = {
        "schema_version": "oli.jlens-leading-digit-eval/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "development_only",
        "metric_target": "first token of exact three-digit addition result",
        "model": fit_config["model"],
        "lens": {
            "path": str(args.lens),
            "sha256": lens_sha256,
            "n_prompts": lens.n_prompts,
            "source_layers": lens.source_layers,
            "matrix_diagnostics": lens_diagnostics,
            "fit_config_sha256": hashlib.sha256(
                args.fit_config.read_bytes()
            ).hexdigest(),
        },
        "dataset": {
            "builder": "build_phase01_dataset",
            "split": "development",
            "template_family": "word_problem",
            "kind": "addition",
            "n_examples": len(examples),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "balanced_leading_digit_chance": 1 / 9,
        "final_model": summarize(
            final_ranks, final_digit_ranks, final_digit_margins
        ),
        "best_development_layers": best_layers,
        "layer_metrics": summaries,
        "samples": sample_rows,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Ranks are correlational development readouts at the last prompt token. "
            "They do not establish exact-value decoding or causal use."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_jsonl(args.readouts, readout_artifacts)
    print(f"wrote {args.output}")
    print(f"wrote {len(readout_artifacts)} readouts to {args.readouts}")


if __name__ == "__main__":
    main()
