from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch


def stable_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_manifest(repo: Path) -> dict[str, object]:
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(repo),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "cuda_available": torch.cuda.is_available(),
    }

