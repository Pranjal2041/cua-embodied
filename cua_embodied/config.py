"""Task configuration and API keys."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Read the per-app task configuration (dataset paths, checkpoints, task IDs)."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    return {name: value for name, value in config.items() if not name.startswith("_")}


def load_dotenv(path: str | Path = ".env") -> None:
    """Set variables such as GEMINI_API_KEY from a .env file without overriding the environment."""
    path = Path(path)
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
