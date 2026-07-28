from pathlib import Path

import pytest

from open_latent_interfaces.compiler_graft import CompilerGraftManifest


def test_audited_smollm2_compiler_graft_manifest() -> None:
    root = Path(__file__).parents[1]
    manifest = CompilerGraftManifest.load(
        root / "manifests/smollm2-17b-compiler-arithmetic-graft-v1.json"
    )
    manifest.verify(root)
    reader = manifest.load_reader(root)
    assert manifest.model_id == "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    assert manifest.residual_width == 2048
    assert reader.residual_width == 2048
    assert set(manifest.writer["positions"]) == {"0", "1", "2"}
    assert manifest.writer["positions"]["1"]["desired_margin"] == 4.0


def test_compiler_graft_rejects_unsupported_schema() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        CompilerGraftManifest.from_dict({"schema_version": "wrong"})
