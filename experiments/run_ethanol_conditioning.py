from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fly_sud.config import load_config
from fly_sud.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ethanol-cue conditioning experiment.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "default.json")
    parser.add_argument("--n-flies", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.n_flies is not None:
        config["experiment"]["n_flies"] = args.n_flies
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    output = args.output_dir or ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    _, summary = run_experiment(config, output)
    print(summary.to_string(index=False))
    print(f"\nResults written to: {output.resolve()}")


if __name__ == "__main__":
    main()

