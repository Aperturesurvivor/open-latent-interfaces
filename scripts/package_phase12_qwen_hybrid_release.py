#!/usr/bin/env python3
"""Build the deterministic Phase 12 Qwen hybrid-graft release package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from open_latent_interfaces.hybrid_graft import HybridGraftManifest
from open_latent_interfaces.operand_reader import OperandReaderManifest

TAG = "phase12-qwen-hybrid-graft-audit-v1"
BUNDLE_NAME = "qwen25-15b-hybrid-graft-audit-v1.zip"

BUNDLE_PATHS = (
    "PHASE12_QWEN_HYBRID_GRAFT_AUDIT_SUMMARY.md",
    "artifacts/phase11_qwen_operand_reader.safetensors",
    "artifacts/phase2_ones_prototypes.safetensors",
    "artifacts/phase2_tens_delta_basis.safetensors",
    "artifacts/phase2_tens_prototypes.safetensors",
    "configs/phase11_qwen_operand_reader_selection.json",
    "configs/phase12_qwen_hybrid_graft_audit.json",
    "configs/phase12_qwen_hybrid_graft_audit_dataset_frozen.json",
    "configs/phase12_qwen_hybrid_graft_development.json",
    "docs/HYBRID_GRAFT_INTERFACE.md",
    "docs/OPERAND_READER_INTERFACE.md",
    "manifests/qwen25-15b-hybrid-arithmetic-graft-v1.json",
    "manifests/qwen25-15b-operand-reader-v1.json",
    "protocols/PHASE12_QWEN_COMPILER_HARDENING.md",
    "results/phase11_qwen_operand_reader_selection.json",
    "results/phase12_qwen_compiler_robustness_selection.json",
    "results/phase12_qwen_hybrid_graft_audit.json",
    "results/phase12_qwen_hybrid_graft_development.json",
    "schemas/hybrid-graft-interface-v2.schema.json",
    "schemas/operand-reader-interface-v2.schema.json",
    "scripts/package_phase12_qwen_hybrid_release.py",
    "scripts/run_phase11_qwen_hybrid_graft_audit.py",
    "scripts/run_phase11_qwen_hybrid_graft_development.py",
    "src/open_latent_interfaces/causal_compiler.py",
    "src/open_latent_interfaces/hybrid_graft.py",
    "src/open_latent_interfaces/operand_reader.py",
)

FLAT_ASSETS = {
    "PHASE12_QWEN_HYBRID_GRAFT_AUDIT_SUMMARY.md": (
        "PHASE12_QWEN_HYBRID_GRAFT_AUDIT_SUMMARY.md"
    ),
    "artifacts/phase11_qwen_operand_reader.safetensors": (
        "phase11_qwen_operand_reader.safetensors"
    ),
    "artifacts/phase2_ones_prototypes.safetensors": (
        "phase2_ones_prototypes.safetensors"
    ),
    "artifacts/phase2_tens_delta_basis.safetensors": (
        "phase2_tens_delta_basis.safetensors"
    ),
    "artifacts/phase2_tens_prototypes.safetensors": (
        "phase2_tens_prototypes.safetensors"
    ),
    "configs/phase12_qwen_hybrid_graft_audit.json": (
        "phase12_qwen_hybrid_graft_audit_config.json"
    ),
    "configs/phase12_qwen_hybrid_graft_audit_dataset_frozen.json": (
        "phase12_qwen_hybrid_graft_audit_dataset_frozen.json"
    ),
    "manifests/qwen25-15b-hybrid-arithmetic-graft-v1.json": (
        "qwen25-15b-hybrid-arithmetic-graft-v1.json"
    ),
    "manifests/qwen25-15b-operand-reader-v1.json": (
        "qwen25-15b-operand-reader-v1.json"
    ),
    "results/phase12_qwen_hybrid_graft_audit.json": (
        "phase12_qwen_hybrid_graft_audit_result.json"
    ),
    "schemas/hybrid-graft-interface-v2.schema.json": (
        "hybrid-graft-interface-v2.schema.json"
    ),
    "schemas/operand-reader-interface-v2.schema.json": (
        "operand-reader-interface-v2.schema.json"
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_deterministic_zip(root: Path, output: Path) -> None:
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for relative in sorted(BUNDLE_PATHS):
            source = root / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            info = zipfile.ZipInfo(relative, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes(), compresslevel=9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit("refusing to write into a nonempty output directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    reader_manifest = OperandReaderManifest.load(
        root / "manifests/qwen25-15b-operand-reader-v1.json"
    )
    reader_manifest.verify(root)
    hybrid_manifest = HybridGraftManifest.load(
        root / "manifests/qwen25-15b-hybrid-arithmetic-graft-v1.json"
    )
    hybrid_manifest.verify(root)

    bundle_path = output_dir / BUNDLE_NAME
    build_deterministic_zip(root, bundle_path)
    for source_name, asset_name in FLAT_ASSETS.items():
        shutil.copyfile(root / source_name, output_dir / asset_name)

    source_records = [
        {
            "path": relative,
            "sha256": sha256(root / relative),
        }
        for relative in sorted(BUNDLE_PATHS)
    ]
    asset_records = [
        {
            "name": path.name,
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(output_dir.iterdir())
    ]
    package_manifest = {
        "schema_version": "oli.release-package/v1",
        "tag": TAG,
        "bundle": BUNDLE_NAME,
        "sources": source_records,
        "assets": asset_records,
    }
    package_manifest_path = output_dir / "release-package.json"
    package_manifest_path.write_text(
        json.dumps(package_manifest, indent=2, sort_keys=True) + "\n"
    )
    checksummed = sorted(
        path
        for path in output_dir.iterdir()
        if path.name != "SHA256SUMS"
    )
    checksum_path = output_dir / "SHA256SUMS"
    checksum_path.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in checksummed)
    )
    print(f"built {len(checksummed)} assets in {output_dir}")


if __name__ == "__main__":
    main()
