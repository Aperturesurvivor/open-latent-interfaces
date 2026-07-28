from pathlib import Path

import pytest

from open_latent_interfaces.hybrid_graft import HybridGraftManifest


def test_audited_phi_hybrid_graft_manifest() -> None:
    root = Path(__file__).parents[1]
    manifest = HybridGraftManifest.load(
        root / "manifests/phi35-mini-hybrid-arithmetic-graft-v1.json"
    )
    manifest.verify(root)
    reader = manifest.load_reader(root)
    basis, prototypes = manifest.load_suffix_components(root)
    assert manifest.model_id == "microsoft/Phi-3.5-mini-instruct"
    assert reader.residual_width == 3072
    assert basis.shape == (32, 3072)
    assert set(prototypes) == {1, 2}
    assert all(value.shape == (10, 32) for value in prototypes.values())


def test_hybrid_graft_rejects_unsupported_schema() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        HybridGraftManifest.from_dict({"schema_version": "wrong"})
