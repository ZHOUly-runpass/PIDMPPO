from __future__ import annotations

import json
import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import gymnasium
import numpy
import torch

from ..config import ExperimentConfig


def build_run_manifest(config: ExperimentConfig, variant: str) -> dict:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "variant": variant,
        "config": config.to_dict(),
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "numpy": numpy.__version__,
            "gymnasium": gymnasium.__version__,
        },
        "git_commit": _git_commit(),
        "source_fingerprint": source_fingerprint(),
        "implementation_version": "correctness-v2",
    }


def write_run_manifest(path: str | Path, config: ExperimentConfig, variant: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_run_manifest(config, variant), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def source_fingerprint() -> str:
    """Hash implementation/tests and optimization protocol, not output artifacts."""
    root = Path(__file__).resolve().parents[3]
    paths = [path for folder in ("src", "scripts", "tests") for path in (root / folder).rglob("*.py")]
    paths += [root / "configs/paper.yaml", root / "configs/optimization_v2.yaml"]
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()
