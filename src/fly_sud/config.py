from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate the experiment configuration."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    required = {"experiment", "protocol", "learning", "odor", "arena", "movement"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")
    if config["experiment"]["n_flies"] < 1:
        raise ValueError("n_flies must be positive")
    return config
