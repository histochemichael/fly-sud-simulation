from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import summarize_results
from .arena import YMaze
from .learning import condition_memory
from .plots import generate_figures
from .simulation import run_choice_trial


CONDITIONS = ("untrained", "paired", "unpaired")


def run_experiment(config: dict, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["experiment"]["seed"])
    n_flies = int(config["experiment"]["n_flies"])
    condition_sequences = np.random.SeedSequence(seed).spawn(len(CONDITIONS))
    rows, trajectory_rows, history_rows = [], [], []
    state_rows, learning_state_rows = [], []

    for condition, condition_sequence in zip(CONDITIONS, condition_sequences):
        for fly_id, fly_sequence in enumerate(condition_sequence.spawn(n_flies)):
            fly_seed = int(fly_sequence.generate_state(1, dtype=np.uint64)[0])
            rng = np.random.default_rng(fly_sequence)
            memory, history, training_states = condition_memory(condition, config, rng)
            learning_state_rows.append({"condition": condition, "fly_id": fly_id, "stage": "naive", "weight_a": 0.0, "weight_b": 0.0, "random_seed": fly_seed})
            for item in history:
                history_rows.append({"fly_id": fly_id, **item})
            for item in training_states:
                state_rows.append({"condition": condition, "fly_id": fly_id, "random_seed": fly_seed, **item})
            odor_b_left = bool(rng.integers(0, 2))
            trial = run_choice_trial(memory, YMaze(config, odor_b_left), config, rng)
            learning_state_rows.append({"condition": condition, "fly_id": fly_id, "stage": "learned", "weight_a": float(memory.weights[0]), "weight_b": float(memory.weights[1]), "random_seed": fly_seed})
            rows.append({"condition": condition, "fly_id": fly_id, "choice": trial.choice, "chose_b": trial.choice == "B", "odor_b_side": "left" if odor_b_left else "right", "latency_s": trial.latency, "distance": trial.distance, "completed": trial.completed, "weight_a": float(memory.weights[0]), "weight_b": float(memory.weights[1]), "random_seed": fly_seed})
            for item in trial.state_history:
                state_rows.append({"condition": condition, "fly_id": fly_id, "random_seed": fly_seed, **item})
            for time_s, x, y in trial.trajectory:
                trajectory_rows.append({"condition": condition, "fly_id": fly_id, "time_s": time_s, "x": x, "y": y})

    results = pd.DataFrame(rows)
    trajectories = pd.DataFrame(trajectory_rows)
    history = pd.DataFrame(history_rows)
    state_history = pd.DataFrame(state_rows)
    learning_states = pd.DataFrame(learning_state_rows)
    summary = summarize_results(results)
    results.to_csv(output_dir / "choices.csv", index=False)
    trajectories.to_csv(output_dir / "trajectories.csv", index=False)
    history.to_csv(output_dir / "memory_history.csv", index=False)
    state_history.to_parquet(output_dir / "state_history.parquet", index=False, compression="zstd")
    learning_states.to_parquet(output_dir / "learning_states.parquet", index=False, compression="zstd")
    summary.to_csv(output_dir / "summary.csv", index=False)
    with (output_dir / "config_snapshot.json").open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
    config_bytes = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    package_names = ["flygym", "mujoco", "numpy", "pandas", "matplotlib", "pyarrow"]
    versions = {name: importlib.metadata.version(name) for name in package_names}
    versions["python"] = platform.python_version()
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "backend": "fast_planar_embodied",
        "flygym_physical_backend_verified": True,
        "master_seed": seed,
        "n_flies_per_condition": n_flies,
        "conditions": list(CONDITIONS),
        "reward_present_during_test": False,
        "learning_mode": "online reward-modulated associative learning during simulated experience",
        "uses_deep_learning": False,
        "biologically_complete_addiction_model": False,
        "scientific_note": "Ethanol is an abstract reinforcing state, not pharmacology.",
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "software_versions": versions,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    dataset_manifest = {
        "dataset_status": "local_only_not_uploaded",
        "hugging_face_upload_performed": False,
        "unit_of_observation": "one simulated fly timestep or discrete training update",
        "primary_table": "state_history.parquet",
        "state_history_rows": len(state_history),
        "state_history_schema": {column: str(dtype) for column, dtype in state_history.dtypes.items()},
        "files": {
            "state_history.parquet": "Per-step training and test state/action history.",
            "learning_states.parquet": "Naive and final associative state for every fly.",
            "choices.csv": "One row per fly with final outcome and seed.",
            "summary.csv": "Condition-level outcome statistics.",
            "memory_history.csv": "Session-level learning curves.",
            "trajectories.csv": "Lightweight plotting-compatible trajectories.",
            "config_snapshot.json": "Exact experiment and controller parameters.",
            "metadata.json": "Software versions, provenance, and scientific flags."
        }
    }
    with (output_dir / "dataset_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(dataset_manifest, stream, indent=2)
    generate_figures(results, trajectories, history, summary, output_dir)
    return results, summary
