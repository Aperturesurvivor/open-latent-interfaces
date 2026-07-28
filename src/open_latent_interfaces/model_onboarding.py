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
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import resolve_decoder_block_path
from open_latent_interfaces.causal_compiler import compile_local_margin_plan
from open_latent_interfaces.operand_reader import locate_operand_digit_tokens
from open_latent_interfaces.prefill import (
    render_prefilled_chat,
    result_digit_token_ids,
    verify_decimal_digit_contract,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ModelOnboardingSpec:
    schema_version: str
    name: str
    model_id: str
    model_revision: str
    expected_model_type: str
    expected_architectures: tuple[str, ...]
    expected_hidden_size: int
    expected_num_hidden_layers: int
    expected_vocab_size: int
    task_contract: dict[str, Any]
    compatibility: dict[str, Any]
    evidence: dict[str, Any]
    claim_boundary: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ModelOnboardingSpec:
        if value.get("schema_version") != "oli.model-onboarding/v1":
            raise ValueError("unsupported model-onboarding schema")
        model = value["model"]
        expected = model["expected"]
        spec = cls(
            schema_version=str(value["schema_version"]),
            name=str(value["name"]),
            model_id=str(model["id"]),
            model_revision=str(model["revision"]),
            expected_model_type=str(expected["model_type"]),
            expected_architectures=tuple(
                str(row) for row in expected["architectures"]
            ),
            expected_hidden_size=int(expected["hidden_size"]),
            expected_num_hidden_layers=int(expected["num_hidden_layers"]),
            expected_vocab_size=int(expected["vocab_size"]),
            task_contract=dict(value["task_contract"]),
            compatibility=dict(value["compatibility"]),
            evidence=dict(value.get("evidence", {})),
            claim_boundary=tuple(
                str(row) for row in value["claim_boundary"]
            ),
        )
        spec.validate()
        return spec

    @classmethod
    def load(cls, path: Path) -> ModelOnboardingSpec:
        return cls.from_dict(json.loads(path.read_text()))

    def validate(self) -> None:
        if not self.name or not self.model_id or not self.model_revision:
            raise ValueError("onboarding identity fields cannot be empty")
        if (
            self.expected_hidden_size < 1
            or self.expected_num_hidden_layers < 1
            or self.expected_vocab_size < 2
        ):
            raise ValueError("onboarding model dimensions must be positive")
        if not self.expected_architectures:
            raise ValueError("at least one model architecture is required")
        contract = self.task_contract
        if contract.get("assistant_prefix") != "Answer=":
            raise ValueError("onboarding currently requires the Answer= contract")
        probe = contract.get("probe")
        if not isinstance(probe, dict):
            raise ValueError("onboarding task contract requires a probe")
        for key in ("user_prompt", "operand_a", "operand_b", "target"):
            if key not in probe:
                raise ValueError(f"onboarding probe is missing {key}")
        if int(probe["operand_a"]) + int(probe["operand_b"]) != int(
            probe["target"]
        ):
            raise ValueError("onboarding probe target is not the operand sum")
        width = int(contract["result_width"])
        probe_results = [int(value) for value in contract["probe_results"]]
        if width < 1 or any(len(str(value)) != width for value in probe_results):
            raise ValueError("probe results violate the fixed-width contract")
        required = {
            "fast_tokenizer_offsets",
            "chat_prefill",
            "single_token_decimal_digits",
            "compositional_fixed_width_results",
            "decoder_block_stack",
            "hidden_state_convention",
            "block_to_logits_gradient",
        }
        if not required <= set(self.compatibility):
            missing = sorted(required - set(self.compatibility))
            raise ValueError(f"missing compatibility gate: {missing[0]}")
        if not all(self.compatibility[name] is True for name in required):
            raise ValueError("all v1 compatibility gates must be required")
        if self.evidence.get("status") not in {
            "audited_reference",
            "prospective_candidate",
        }:
            raise ValueError("invalid onboarding evidence status")

    def verify_evidence(self, root: Path) -> None:
        sources = self.evidence.get("sources", {})
        if not isinstance(sources, dict):
            raise ValueError("onboarding evidence sources must be an object")
        for name, reference in sources.items():
            if not isinstance(reference, dict):
                raise ValueError(f"invalid onboarding evidence: {name}")
            path = root / str(reference["path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            if _sha256(path) != str(reference["sha256"]):
                raise ValueError(f"onboarding evidence hash mismatch: {name}")
        manifest_reference = sources.get("hybrid_graft_manifest")
        if manifest_reference is not None:
            from open_latent_interfaces.hybrid_graft import HybridGraftManifest

            manifest = HybridGraftManifest.load(
                root / str(manifest_reference["path"])
            )
            manifest.verify(root)
            if {
                "id": manifest.model_id,
                "revision": manifest.model_revision,
            } != {
                "id": self.model_id,
                "revision": self.model_revision,
            }:
                raise ValueError("onboarding evidence/model mismatch")


def candidate_hidden_state_indices(num_hidden_layers: int) -> list[int]:
    """Return deterministic early-to-late reader candidates for a new model."""
    if num_hidden_layers < 4:
        raise ValueError("onboarding requires at least four decoder blocks")
    fractions = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)
    indices = {
        max(1, min(num_hidden_layers, 1 + round(fraction * num_hidden_layers)))
        for fraction in fractions
    }
    return sorted(indices)


def _metadata_observations(
    spec: ModelOnboardingSpec,
    config: Any,
    tokenizer: Any,
) -> tuple[dict[str, bool], dict[str, Any], str, dict[int, int]]:
    prompt = spec.task_contract["probe"]
    rendered = render_prefilled_chat(
        tokenizer,
        str(prompt["user_prompt"]),
        assistant_prefix=str(spec.task_contract["assistant_prefix"]),
    )
    digit_token_ids = verify_decimal_digit_contract(tokenizer, rendered)
    composed = result_digit_token_ids(
        tokenizer,
        rendered,
        [int(value) for value in spec.task_contract["probe_results"]],
    )
    located = locate_operand_digit_tokens(
        tokenizer,
        rendered,
        str(prompt["user_prompt"]),
        int(prompt["operand_a"]),
        int(prompt["operand_b"]),
    )
    observed_model_type = str(config.model_type)
    observed_architectures = tuple(str(row) for row in config.architectures)
    observed_hidden_size = int(config.hidden_size)
    observed_layers = int(config.num_hidden_layers)
    observed_vocab = int(config.vocab_size)
    checks = {
        "model_type": observed_model_type == spec.expected_model_type,
        "architecture": observed_architectures
        == spec.expected_architectures,
        "hidden_size": observed_hidden_size == spec.expected_hidden_size,
        "num_hidden_layers": (
            observed_layers == spec.expected_num_hidden_layers
        ),
        "vocab_size": observed_vocab == spec.expected_vocab_size,
        "fast_tokenizer_offsets": bool(tokenizer.is_fast),
        "chat_prefill": (
            bool(tokenizer.chat_template)
            and rendered.endswith(spec.task_contract["assistant_prefix"])
        ),
        "single_token_decimal_digits": (
            len(digit_token_ids) == 10
            and len(set(digit_token_ids.values())) == 10
        ),
        "compositional_fixed_width_results": all(
            len(row) == int(spec.task_contract["result_width"])
            for row in composed
        ),
        "operand_locator": (
            len(located.operand_a) == len(str(prompt["operand_a"]))
            and len(located.operand_b) == len(str(prompt["operand_b"]))
        ),
    }
    observations = {
        "model_type": observed_model_type,
        "architectures": list(observed_architectures),
        "hidden_size": observed_hidden_size,
        "num_hidden_layers": observed_layers,
        "vocab_size": observed_vocab,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_is_fast": bool(tokenizer.is_fast),
        "rendered_probe": rendered,
        "digit_token_ids": {
            str(digit): token_id
            for digit, token_id in digit_token_ids.items()
        },
        "operand_positions": {
            "operand_a": list(located.operand_a),
            "operand_b": list(located.operand_b),
        },
        "candidate_reader_hidden_state_indices": (
            candidate_hidden_state_indices(observed_layers)
        ),
    }
    return checks, observations, rendered, digit_token_ids


def run_live_preflight(
    spec: ModelOnboardingSpec,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    metadata_only: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = AutoConfig.from_pretrained(
        spec.model_id,
        revision=spec.model_revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_id,
        revision=spec.model_revision,
        use_fast=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    checks, observed, rendered, digit_token_ids = _metadata_observations(
        spec,
        config,
        tokenizer,
    )
    if metadata_only:
        return {
            "mode": "metadata_only",
            "passes": all(checks.values()),
            "checks": checks,
            "observed": observed,
            "elapsed_seconds": time.perf_counter() - started,
        }

    target_device = torch.device(device)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id,
        revision=spec.model_revision,
        dtype=dtype,
    ).to(target_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    block_path, blocks = resolve_decoder_block_path(model)
    encoded = tokenizer(
        [rendered],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    encoded = {
        key: value.to(target_device) if isinstance(value, torch.Tensor) else value
        for key, value in encoded.items()
    }
    with torch.inference_mode():
        outputs = model(
            **encoded,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    hidden_states = outputs.hidden_states
    observed_widths = sorted(
        {int(state.shape[-1]) for state in hidden_states}
    )
    checks.update(
        {
            "decoder_block_stack": len(blocks)
            == spec.expected_num_hidden_layers,
            "hidden_state_convention": (
                len(hidden_states) == len(blocks) + 1
                and observed_widths == [spec.expected_hidden_size]
            ),
            "next_token_logits": (
                outputs.logits.ndim == 3
                and outputs.logits.shape[-1] == spec.expected_vocab_size
            ),
            "all_parameters_frozen": not any(
                parameter.requires_grad for parameter in model.parameters()
            ),
        }
    )
    gradient_hidden_index = max(1, len(blocks) // 2)
    target_digit = int(str(spec.task_contract["probe"]["target"])[0])
    plan = compile_local_margin_plan(
        model,
        tokenizer,
        [rendered],
        hidden_state_index=gradient_hidden_index,
        target_token_ids=torch.tensor([digit_token_ids[target_digit]]),
        candidate_token_ids=torch.tensor(
            [digit_token_ids[digit] for digit in range(10)]
        ),
        device=target_device,
        batch_size=1,
    )
    gradient = plan.margin_gradients
    checks["block_to_logits_gradient"] = (
        gradient.shape == (1, spec.expected_hidden_size)
        and bool(torch.isfinite(gradient).all())
        and float(gradient.norm()) > 0
    )
    observed.update(
        {
            "decoder_block_path": list(block_path),
            "decoder_block_count": len(blocks),
            "hidden_state_count": len(hidden_states),
            "hidden_state_widths": observed_widths,
            "gradient_probe_hidden_state_index": gradient_hidden_index,
            "gradient_probe_norm": float(gradient.norm()),
            "model_class": type(model).__name__,
            "dtype": str(dtype).removeprefix("torch."),
            "device": str(target_device),
        }
    )
    return {
        "mode": "live",
        "passes": all(checks.values()),
        "checks": checks,
        "observed": observed,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preflight a frozen causal LM for latent-interface onboarding."
    )
    parser.add_argument("spec", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite model-onboarding result")
    spec = ModelOnboardingSpec.load(args.spec)
    root = args.root or args.spec.parent.parent
    spec.verify_evidence(root)
    try:
        preflight = run_live_preflight(
            spec,
            device=args.device,
            dtype=getattr(torch, args.dtype),
            metadata_only=args.metadata_only,
        )
        error = None
    except Exception as exc:
        preflight = {
            "mode": "metadata_only" if args.metadata_only else "live",
            "passes": False,
            "checks": {},
            "observed": {},
            "elapsed_seconds": 0.0,
        }
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    report = {
        "schema_version": "oli.model-onboarding-result/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "compatibility_preflight",
        "model": {
            "id": spec.model_id,
            "revision": spec.model_revision,
        },
        "spec_sha256": _sha256(args.spec),
        "runner_sha256": _sha256(Path(__file__)),
        **preflight,
        "error": error,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "claim_boundary": (
            "Compatibility only: a pass authorizes model-specific discovery "
            "but is not evidence that a latent reader or writer exists."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    if not report["passes"]:
        raise SystemExit("model-onboarding preflight did not pass")


if __name__ == "__main__":
    main()
