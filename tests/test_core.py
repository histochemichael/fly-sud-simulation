from pathlib import Path

import numpy as np
import pandas as pd

from fly_sud.analysis import preference_index
from fly_sud.config import load_config
from fly_sud.experiment import run_experiment
from fly_sud.learning import condition_memory
from fly_sud.odor import OdorField, bilateral_readout


def config():
    return load_config(Path(__file__).parents[1] / "config" / "default.json")


def test_odor_field_decays_with_distance():
    field = OdorField(np.array([[0.0, 0.0], [10.0, 0.0]]), 1.0, 1.0)
    concentrations = field.concentration(np.array([[0.1, 0.0], [2.0, 0.0]]))
    assert concentrations[0, 0] > concentrations[1, 0]


def test_bilateral_sensor_is_stronger_on_source_side():
    field = OdorField(np.array([[-1.0, 1.0], [1.0, 1.0]]), 1.0, 1.0)
    sensed = bilateral_readout(field, np.zeros(2), np.pi / 2, 0.2, 0.0, 0.0, np.random.default_rng(1))
    assert sensed[0, 0] > sensed[0, 1]
    assert sensed[1, 1] > sensed[1, 0]


def test_paired_reward_associates_with_b():
    memory, _, states = condition_memory("paired", config(), np.random.default_rng(1))
    assert memory.weights[1] > memory.weights[0]
    assert states[-1]["teaching_signal"] > 0


def test_unpaired_reward_does_not_specifically_associate_with_b():
    memory, _, _ = condition_memory("unpaired", config(), np.random.default_rng(1))
    assert abs(memory.weights[1] - memory.weights[0]) < 0.12


def test_odor_side_randomization_is_balanced():
    rng = np.random.default_rng(42)
    sides = [bool(rng.integers(0, 2)) for _ in range(1000)]
    assert 0.45 < np.mean(sides) < 0.55


def test_preference_index():
    assert preference_index(pd.Series(["B", "B", "B", "A"])) == 0.5


def test_reproducibility_with_fixed_seed(tmp_path):
    cfg = config()
    cfg["experiment"]["n_flies"] = 8
    first, _ = run_experiment(cfg, tmp_path / "first")
    second, _ = run_experiment(cfg, tmp_path / "second")
    pd.testing.assert_frame_equal(first, second)
    first_states = pd.read_parquet(tmp_path / "first" / "state_history.parquet")
    second_states = pd.read_parquet(tmp_path / "second" / "state_history.parquet")
    pd.testing.assert_frame_equal(first_states, second_states)


def test_scientific_history_and_learning_states(tmp_path):
    cfg = config()
    cfg["experiment"]["n_flies"] = 3
    run_experiment(cfg, tmp_path / "run")
    states = pd.read_parquet(tmp_path / "run" / "state_history.parquet")
    required = {
        "fly_id", "condition", "phase", "conditioning_session", "conditioning_trial",
        "simulation_time_s", "x", "y", "heading_rad", "odor_a_left",
        "odor_a_right", "odor_b_left", "odor_b_right", "ethanol_state",
        "teaching_signal", "weight_a", "weight_b", "action_forward_speed",
        "action_turn_rate", "final_choice", "choice_latency_s", "random_seed",
    }
    assert required <= set(states.columns)
    assert set(states["phase"]) == {"training", "test"}
    assert (states.loc[states["phase"] == "test", "teaching_signal"] == 0).all()
    learning_states = pd.read_parquet(tmp_path / "run" / "learning_states.parquet")
    assert set(learning_states["stage"]) == {"naive", "learned"}
    assert len(learning_states) == 3 * 3 * 2


def test_paired_condition_exceeds_controls(tmp_path):
    cfg = config()
    cfg["experiment"]["n_flies"] = 40
    _, summary = run_experiment(cfg, tmp_path / "batch")
    pi = summary.set_index("condition")["preference_index"]
    assert pi["paired"] > pi["untrained"] + 0.15
    assert pi["paired"] > pi["unpaired"] + 0.15
