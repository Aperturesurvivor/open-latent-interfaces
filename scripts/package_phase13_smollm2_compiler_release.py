#!/usr/bin/env python3
"""Build the deterministic Phase 13 SmolLM2 compiler-graft release package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from open_latent_interfaces.compiler_graft import CompilerGraftManifest
from open_latent_interfaces.operand_reader import OperandReaderManifest

TAG = "phase13-smollm2-compiler-graft-audit-v1"
BUNDLE_NAME = "smollm2-17b-compiler-graft-audit-v1.zip"

BUNDLE_PATHS = (
    "PHASE13_SMOLLM2_COMPILER_GRAFT_AUDIT_SUMMARY.md",
    "artifacts/phase13_smollm2_operand_reader.safetensors",
    "configs/phase13_smollm2_compiler_graft_audit.json",
    "configs/phase13_smollm2_compiler_graft_audit_dataset_frozen.json",
    "configs/phase13_smollm2_compiler_graft_development.json",
    "configs/phase13_smollm2_operand_reader_selection.json",
    "docs/HYBRID_GRAFT_INTERFACE.md",
    "docs/OPERAND_READER_INTERFACE.md",
    "manifests/smollm2-17b-compiler-arithmetic-graft-v1.json",
    "manifests/smollm2-17b-operand-reader-v1.json",
    "protocols/PHASE13_MODEL_ONBOARDING_AND_THIRD_FAMILY.md",
    "results/model_onboarding_smollm2_17b_candidate_live.json",
    "results/phase13_smollm2_capability.json",
    "results/phase13_smollm2_compiler_graft_audit.json",
    "results/phase13_smollm2_compiler_graft_development.json",
    "results/phase13_smollm2_leading_compiler_selection.json",
    "results/phase13_smollm2_operand_reader_selection.json",
    "results/phase13_smollm2_suffix_compiler_selection.json",
    "results/phase13_smollm2_suffix_prototype_selection.json",
    "schemas/compiler-graft-interface-v1.schema.json",
    "schemas/operand-reader-interface-v2.schema.json",
    "scripts/package_phase13_smollm2_compiler_release.py",
    "scripts/run_phase13_smollm2_compiler_graft_audit.py",
    "scripts/run_phase13_smollm2_compiler_graft_development.py",
    "scripts/run_phase3_closed_loop_development.py",
    "scripts/run_phase3_native_boundary.py",
    "scripts/run_phase4_carry_sequence_boundary.py",
    "scripts/run_phase8_latent_graft.py",
    "scripts/run_phase8_operand_reader_selection.py",
    "src/open_latent_interfaces/activations.py",
    "src/open_latent_interfaces/capability.py",
    "src/open_latent_interfaces/causal_compiler.py",
    "src/open_latent_interfaces/compiler_graft.py",
    "src/open_latent_interfaces/compiler_writer.py",
    "src/open_latent_interfaces/evaluation.py",
    "src/open_latent_interfaces/interventions.py",
    "src/open_latent_interfaces/operand_reader.py",
    "src/open_latent_interfaces/phase13_audit_data.py",
    "src/open_latent_interfaces/phase13_data.py",
    "src/open_latent_interfaces/prefill.py",
)

FLAT_ASSETS = {
    "PHASE13_SMOLLM2_COMPILER_GRAFT_AUDIT_SUMMARY.md": (
        "PHASE13_SMOLLM2_COMPILER_GRAFT_AUDIT_SUMMARY.md"
    ),
    "artifacts/phase13_smollm2_operand_reader.safetensors": (
        "phase13_smollm2_operand_reader.safetensors"
    ),
    "configs/phase13_smollm2_compiler_graft_audit.json": (
        "phase13_smollm2_compiler_graft_audit_config.json"
    ),
    "configs/phase13_smollm2_compiler_graft_audit_dataset_frozen.json": (
        "phase13_smollm2_compiler_graft_audit_dataset_frozen.json"
    ),
    "manifests/smollm2-17b-compiler-arithmetic-graft-v1.json": (
        "smollm2-17b-compiler-arithmetic-graft-v1.json"
    ),
    "manifests/smollm2-17b-operand-reader-v1.json": ("smollm2-17b-operand-reader-v1.json"),
    "results/phase13_smollm2_compiler_graft_audit.json": (
        "phase13_smollm2_compiler_graft_audit_result.json"
    ),
    "schemas/compiler-graft-interface-v1.schema.json": ("compiler-graft-interface-v1.schema.json"),
    "schemas/operand-reader-interface-v2.schema.json": ("operand-reader-interface-v2.schema.json"),
    "src/open_latent_interfaces/compiler_writer.py": "compiler_writer.py",
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
        root / "manifests/smollm2-17b-operand-reader-v1.json"
    )
    reader_manifest.verify(root)
    compiler_manifest = CompilerGraftManifest.load(
        root / "manifests/smollm2-17b-compiler-arithmetic-graft-v1.json"
    )
    compiler_manifest.verify(root)

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
    package_manifest_path.write_text(json.dumps(package_manifest, indent=2, sort_keys=True) + "\n")
    checksummed = sorted(path for path in output_dir.iterdir() if path.name != "SHA256SUMS")
    checksum_path = output_dir / "SHA256SUMS"
    checksum_path.write_text("".join(f"{sha256(path)}  {path.name}\n" for path in checksummed))
    print(f"built {len(checksummed)} assets in {output_dir}")


if __name__ == "__main__":
    main()
