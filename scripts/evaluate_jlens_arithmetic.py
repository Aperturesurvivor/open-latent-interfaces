#!/usr/bin/env python3
"""Evaluate J-lens and logit lens across three teacher-forced answer steps."""

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

from open_latent_interfaces.capability import build_capability_sweep
from open_latent_interfaces.interpretability import write_jsonl
from open_latent_interfaces.interpretability_backends import JacobianLensAdapter


def rank_of(logits: torch.Tensor, token_id: int) -> int:
    return int((logits > logits[token_id]).sum().item()) + 1


def summarize(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    ranks = [row["vocabulary_rank"] for row in rows]
    digit_ranks = [row["digit_rank"] for row in rows]
    margins = [row["digit_margin"] for row in rows]
    return {
        "n": len(rows),
        "median_vocabulary_rank": float(median(ranks)),
        "mean_reciprocal_rank": sum(1 / rank for rank in ranks) / len(ranks),
        "vocabulary_hit_at_10": sum(rank <= 10 for rank in ranks) / len(ranks),
        "digit_accuracy": sum(rank == 1 for rank in digit_ranks) / len(digit_ranks),
        "median_digit_rank": float(median(digit_ranks)),
        "mean_digit_margin": sum(margins) / len(margins),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-config", type=Path, required=True)
    parser.add_argument("--capability-config", type=Path, required=True)
    parser.add_argument("--lens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--readouts", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--artifact-examples", type=int, default=3)
    args = parser.parse_args()

    import jlens
    import transformers

    fit_config = json.loads(args.fit_config.read_text())
    capability_config = json.loads(args.capability_config.read_text())
    if fit_config["model"] != capability_config["model"]:
        raise SystemExit("lens and capability configs target different model revisions")

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
    matrix_diagnostics = {}
    identity = torch.eye(lens.d_model)
    for layer, jacobian in sorted(lens.jacobians.items()):
        matrix = jacobian.float()
        matrix_norm = matrix.norm().item()
        matrix_diagnostics[str(layer)] = {
            "finite": bool(torch.isfinite(matrix).all()),
            "frobenius_norm_over_sqrt_d": matrix_norm / (lens.d_model**0.5),
            "cosine_with_identity": float(
                (matrix * identity).sum().item()
                / max(matrix_norm * identity.norm().item(), 1e-12)
            ),
        }
    lens.jacobians = {
        layer: jacobian.to(device) for layer, jacobian in lens.jacobians.items()
    }

    dataset_config = capability_config["dataset"]
    examples = [
        example
        for example in build_capability_sweep(
            seed=dataset_config["seed"],
            development_pairs=dataset_config["development_pairs"],
            audit_pairs=dataset_config["audit_pairs"],
            protocol_version=dataset_config["protocol_version"],
        )
        if example.split == "development"
        and example.regime == "three_digit_mixed"
        and example.presentation == "chat"
    ]
    digit_token_ids = [
        int(tokenizer(str(digit), add_special_tokens=False)["input_ids"][0])
        for digit in range(10)
    ]
    metrics: dict[str, dict[int, dict[int, list[dict[str, Any]]]]] = {
        "jacobian_lens": defaultdict(lambda: defaultdict(list)),
        "logit_lens": defaultdict(lambda: defaultdict(list)),
    }
    predictions: dict[str, dict[int, dict[str, list[int]]]] = {
        "jacobian_lens": defaultdict(lambda: defaultdict(list)),
        "logit_lens": defaultdict(lambda: defaultdict(list)),
    }
    final_metrics: dict[int, list[dict[str, Any]]] = defaultdict(list)
    final_predictions: dict[str, list[int]] = defaultdict(list)
    readout_artifacts = []
    artifact_layers = {3, 7, 11, 15, 19, 23}
    lens_sha256 = hashlib.sha256(args.lens.read_bytes()).hexdigest()
    adapter = JacobianLensAdapter(
        lens,
        model,
        tokenizer,
        lens_checkpoint=str(args.lens),
        lens_checkpoint_revision=lens_sha256,
    )
    started = time.perf_counter()

    for example_index, example in enumerate(examples):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": example.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        target_ids = tokenizer(
            str(example.result), add_special_tokens=False
        )["input_ids"]
        if len(target_ids) != 3:
            raise ValueError(f"{example.result} does not tokenize to three digits")
        prefix = ""
        for step, target_id in enumerate(target_ids):
            input_text = rendered + prefix
            input_ids = model.encode(input_text, max_length=128)
            record_at = sorted(set(lens.source_layers) | {model.n_layers - 1})
            with jlens.ActivationRecorder(model.layers, at=record_at) as recorder:
                model.forward(input_ids)
                activations = {
                    layer: recorder.activations[layer].detach()
                    for layer in record_at
                }
            correct_digit = int(tokenizer.decode([target_id]))

            for layer in lens.source_layers:
                residual = activations[layer][0, -1].float()
                readouts = {
                    "jacobian_lens": model.unembed(
                        lens.transport(residual, layer)
                    ).reshape(-1),
                    "logit_lens": model.unembed(residual).reshape(-1),
                }
                for method, logits in readouts.items():
                    digit_scores = logits[digit_token_ids]
                    predicted_digit = int(digit_scores.argmax().item())
                    correct_score = digit_scores[correct_digit]
                    wrong_scores = torch.cat(
                        (
                            digit_scores[:correct_digit],
                            digit_scores[correct_digit + 1 :],
                        )
                    )
                    row = {
                        "vocabulary_rank": rank_of(logits, int(target_id)),
                        "digit_rank": int(
                            (digit_scores > correct_score).sum().item()
                        )
                        + 1,
                        "digit_margin": float(
                            (correct_score - wrong_scores.max()).item()
                        ),
                    }
                    metrics[method][step][layer].append(row)
                    predictions[method][layer][example.example_id].append(
                        predicted_digit
                    )
                if (
                    step == 0
                    and example_index < args.artifact_examples
                    and layer in artifact_layers
                ):
                    readout_artifacts.append(
                        adapter.readout(
                            residual,
                            example_id=example.example_id,
                            target_model=fit_config["model"]["id"],
                            target_model_revision=fit_config["model"]["revision"],
                            hidden_state_index=layer + 1,
                            token_position=int(input_ids.shape[1] - 1),
                            top_k=10,
                        )
                    )

            final_logits = model.unembed(
                activations[model.n_layers - 1][0, -1].float()
            ).reshape(-1)
            final_digit_scores = final_logits[digit_token_ids]
            final_predicted_digit = int(final_digit_scores.argmax().item())
            final_correct_score = final_digit_scores[correct_digit]
            final_wrong_scores = torch.cat(
                (
                    final_digit_scores[:correct_digit],
                    final_digit_scores[correct_digit + 1 :],
                )
            )
            final_metrics[step].append(
                {
                    "vocabulary_rank": rank_of(final_logits, int(target_id)),
                    "digit_rank": int(
                        (final_digit_scores > final_correct_score).sum().item()
                    )
                    + 1,
                    "digit_margin": float(
                        (final_correct_score - final_wrong_scores.max()).item()
                    ),
                }
            )
            final_predictions[example.example_id].append(final_predicted_digit)
            prefix += tokenizer.decode([target_id])

    layer_summaries = {
        method: {
            str(layer): {
                "steps": {
                    str(step): summarize(metrics[method][step][layer])
                    for step in range(3)
                },
                "teacher_forced_full_result_accuracy": sum(
                    predictions[method][layer][example.example_id]
                    == [int(digit) for digit in str(example.result)]
                    for example in examples
                )
                / len(examples),
            }
            for layer in lens.source_layers
        }
        for method in metrics
    }
    report = {
        "schema_version": "oli.jlens-arithmetic-timing/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "development_only",
        "model": fit_config["model"],
        "dataset": {
            "source": "capability_sweep_v2",
            "split": "development",
            "regime": "three_digit_mixed",
            "presentation": "chat",
            "n_examples": len(examples),
            "template_families": sorted(
                {example.template_family for example in examples}
            ),
        },
        "lens": {
            "path": str(args.lens),
            "sha256": lens_sha256,
            "n_prompts": lens.n_prompts,
            "matrix_diagnostics": matrix_diagnostics,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "final_model": {
            "steps": {
                str(step): summarize(final_metrics[step]) for step in range(3)
            },
            "teacher_forced_full_result_accuracy": sum(
                final_predictions[example.example_id]
                == [int(digit) for digit in str(example.result)]
                for example in examples
            )
            / len(examples),
        },
        "layer_metrics": layer_summaries,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Teacher-forced digit readouts are correlational. Later steps include "
            "already-emitted answer prefixes and cannot establish pre-output access "
            "to the full result."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_jsonl(args.readouts, readout_artifacts)
    print(f"wrote {args.output}")
    print(f"wrote {len(readout_artifacts)} readouts to {args.readouts}")


if __name__ == "__main__":
    main()
