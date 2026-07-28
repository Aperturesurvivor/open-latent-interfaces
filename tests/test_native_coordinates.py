from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from open_latent_interfaces.native_coordinates import (
    NativeCoordinateManifest,
    NativeCoordinateWriter,
    fit_digit_prototypes,
)


def test_fit_digit_prototypes_averages_native_coordinates() -> None:
    basis = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    states = torch.tensor([[1.0, 2.0], [3.0, 4.0], [8.0, 10.0]])
    digits = torch.tensor([0, 0, 1])
    prototypes, counts = fit_digit_prototypes(
        states,
        digits,
        basis,
        class_count=2,
    )
    assert torch.equal(counts, torch.tensor([2, 1]))
    assert torch.allclose(prototypes, torch.tensor([[2.0, 3.0], [8.0, 10.0]]))


def test_coordinate_writer_replaces_only_basis_coordinates() -> None:
    basis = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    prototypes = torch.tensor([[2.0, 3.0], [7.0, 11.0]])
    writer = NativeCoordinateWriter(
        basis,
        prototypes,
        scale=1.0,
        norm_cap=100.0,
    )
    states = torch.tensor([[5.0, 6.0, 13.0], [17.0, 19.0, 23.0]])
    digits = torch.tensor([0, 1])
    updated = states + writer.raw_delta(states, digits)
    assert torch.allclose(updated @ basis.T, prototypes[digits])
    assert torch.equal(updated[:, 2], states[:, 2])


def test_coordinate_writer_hard_gate_and_norm_cap() -> None:
    basis = torch.eye(2)
    prototypes = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    writer = NativeCoordinateWriter(
        basis,
        prototypes,
        scale=2.0,
        norm_cap=0.5,
    )
    states = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    digits = torch.tensor([0, 1])
    logits = torch.tensor([[0.0, 5.0, 1.0], [4.0, 0.0, 1.0]])
    token_ids = torch.tensor([1, 2])
    write = writer.write(
        states,
        digits,
        base_logits=logits,
        requested_token_ids=token_ids,
    )
    assert write.hard_gate.tolist() == [True, False]
    assert torch.equal(write.delta[0], torch.zeros(2))
    assert write.delta[1].norm() <= states[1].norm() * 0.5 + 1e-6


def test_fit_digit_prototypes_rejects_missing_class() -> None:
    with pytest.raises(ValueError, match="no fit state"):
        fit_digit_prototypes(
            torch.eye(2),
            torch.tensor([0, 0]),
            torch.eye(2),
            class_count=2,
        )


def test_manifest_loads_audited_interface() -> None:
    root = Path(__file__).parents[1]
    manifest = NativeCoordinateManifest.load(
        root / "manifests/qwen25-15b-next-digit-interface-v1.json"
    )
    manifest.verify(root)
    assert manifest.model_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert set(manifest.positions) == {1, 2}
    tens = manifest.load_writer(1, root=root)
    ones = manifest.load_writer(2, root=root)
    assert (tens.rank, tens.residual_width, tens.scale) == (16, 1536, 1.25)
    assert (ones.rank, ones.residual_width, ones.scale) == (16, 1536, 2.0)


def test_v2_manifest_supports_position_specific_bases() -> None:
    root = Path(__file__).parents[1]
    manifest = NativeCoordinateManifest.load(
        root / "manifests/phi35-mini-next-digit-interface-v1.json"
    )
    manifest.verify(root)
    assert manifest.schema_version == "oli.native-coordinate-interface/v2"
    assert manifest.model_id == "microsoft/Phi-3.5-mini-instruct"
    assert manifest.assistant_prefix == "Answer="
    assert set(manifest.positions) == {0, 1, 2}
    leading = manifest.load_writer(0, root=root)
    tens = manifest.load_writer(1, root=root)
    ones = manifest.load_writer(2, root=root)
    assert (leading.rank, leading.residual_width, leading.scale) == (
        32,
        3072,
        1.0,
    )
    assert (tens.rank, tens.residual_width, tens.scale) == (32, 3072, 1.25)
    assert (ones.rank, ones.residual_width, ones.scale) == (32, 3072, 1.25)
    assert not torch.equal(leading.basis, tens.basis)
    assert torch.equal(tens.basis, ones.basis)


def test_manifest_verification_rejects_bad_artifact_hash(tmp_path: Path) -> None:
    basis_path = tmp_path / "basis.safetensors"
    prototype_path = tmp_path / "prototypes.safetensors"
    evidence_path = tmp_path / "evidence.json"
    save_file({"delta_basis": torch.eye(2)}, str(basis_path))
    save_file({"digit": torch.zeros(10, 2)}, str(prototype_path))
    evidence_path.write_text("{}")
    digest = "0" * 64
    manifest = NativeCoordinateManifest.from_dict(
        {
            "schema_version": "oli.native-coordinate-interface/v1",
            "name": "bad-hash",
            "model": {"id": "test", "revision": "test"},
            "representation": {
                "residual_width": 2,
                "basis": {
                    "path": basis_path.name,
                    "sha256": digest,
                    "key": "delta_basis",
                },
            },
            "positions": {
                "1": {
                    "answer_position": 1,
                    "hidden_state_index": 1,
                    "scale": 1.0,
                    "norm_cap": 1.0,
                    "rank": 2,
                    "prototypes": {
                        "path": prototype_path.name,
                        "sha256": digest,
                        "key": "digit",
                    },
                }
            },
            "evidence": {
                "development_result": {
                    "path": evidence_path.name,
                    "sha256": digest,
                },
                "audit_config": {
                    "path": evidence_path.name,
                    "sha256": digest,
                },
                "audit_result": {
                    "path": evidence_path.name,
                    "sha256": digest,
                },
                "audit_gate_passed": True,
                "audit_runs": 1,
            },
        }
    )
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        manifest.verify(tmp_path)
