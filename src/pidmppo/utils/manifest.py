from __future__ import annotations

import json
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
