from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .arena import YMaze
from .learning import AssociativeMemory
from .odor import bilateral_readout


@dataclass
class TrialResult:
    choice: str
    latency: float
    distance: float
    completed: bool
    trajectory: list[tuple[float, float, float]]
    state_history: list[dict]


def run_choice_trial(memory: AssociativeMemory, maze: YMaze, config: dict, rng: np.random.Generator) -> TrialResult:
    """Navigate a reward-free Y-maze from bilateral olfactory observations."""
    a, o, m = config["arena"], config["odor"], config["movement"]
    position = np.array([0.0, a["start_y"]], dtype=float)
    heading = np.pi / 2 + rng.normal(0.0, m["initial_heading_sd"])
    angular_velocity = 0.0
    speed = max(0.5 * m["speed"], m["speed"] * (1.0 + rng.normal(0.0, m["speed_variability"])))
    distance = 0.0
    trajectory = [(0.0, *position)]
    state_history = []
    salience = m["innate_odor_salience"] + memory.weights

    def finish(choice: str, latency: float, completed: bool) -> TrialResult:
        for row in state_history:
            row["final_choice"] = choice
            row["choice_latency_s"] = latency
            row["completed"] = completed
        return TrialResult(choice, latency, distance, completed, trajectory, state_history)

    for step in range(1, m["max_steps"] + 1):
        sensors = bilateral_readout(maze.odor_field, position, heading, o["sensor_separation"], o["sensor_forward_offset"], o["sensory_noise"], rng)
        left_value, right_value = salience @ sensors
        error = (left_value - right_value) / (left_value + right_value + 1e-9)
        angular_velocity = m["turn_damping"] * angular_velocity + m["turn_gain"] * error + rng.normal(0.0, m["motor_noise"])
        angular_velocity = float(np.clip(angular_velocity, -m["max_turn_rate"], m["max_turn_rate"]))
        state_history.append({
            "phase": "test",
            "conditioning_session": np.nan,
            "conditioning_trial": 1,
            "step": step,
            "simulation_time_s": (step - 1) * m["dt"],
            "x": float(position[0]),
            "y": float(position[1]),
            "heading_rad": float(heading),
            "odor_a_left": float(sensors[0, 0]),
            "odor_a_right": float(sensors[0, 1]),
            "odor_b_left": float(sensors[1, 0]),
            "odor_b_right": float(sensors[1, 1]),
            "ethanol_state": 0.0,
            "internal_reward_state": 0.0,
            "teaching_signal": 0.0,
            "weight_a": float(memory.weights[0]),
            "weight_b": float(memory.weights[1]),
            "action_forward_speed": float(speed),
            "action_turn_rate": angular_velocity,
            "final_choice": None,
            "choice_latency_s": np.nan,
            "completed": False,
        })
        heading = float(np.clip(heading + angular_velocity * m["dt"], 0.20, np.pi - 0.20))
        previous = position.copy()
        position += speed * m["dt"] * np.array([np.cos(heading), np.sin(heading)])
        position = maze.constrain_to_stem(position)
        distance += float(np.linalg.norm(position - previous))
        trajectory.append((step * m["dt"], *position))
        choice = maze.choice(position)
        if choice is not None:
            return finish(choice, step * m["dt"], True)

    if position[0] < 0:
        fallback = "B" if maze.odor_b_left else "A"
    else:
        fallback = "A" if maze.odor_b_left else "B"
    return finish(fallback, m["max_steps"] * m["dt"], False)
