from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AssociativeMemory:
    """Reward-modulated odor association; not ethanol pharmacology."""

    learning_rate: float
    weight_decay: float
    max_weight: float
    weights: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))

    def update(self, odor_activation: np.ndarray, reward: float) -> np.ndarray:
        activation = np.asarray(odor_activation, dtype=float)
        if activation.shape != (2,):
            raise ValueError("odor_activation must have shape (2,)")
        total = float(activation.sum())
        eligibility = activation / total if total > 0 else np.zeros(2)
        prediction = float(np.dot(self.weights, eligibility))
        prediction_error = reward - prediction
        self.weights *= 1.0 - self.weight_decay
        self.weights += self.learning_rate * prediction_error * eligibility
        self.weights = np.clip(self.weights, 0.0, self.max_weight)
        return self.weights.copy()


def condition_memory(condition: str, config: dict, rng: np.random.Generator):
    learning = config["learning"]
    protocol = config["protocol"]
    memory = AssociativeMemory(learning["learning_rate"], learning["weight_decay"], learning["max_weight"])
    history = [{"session": 0, "condition": condition, "weight_a": 0.0, "weight_b": 0.0}]
    state_history = []
    training_time = 0.0
    reward = learning["reward_strength"]

    def apply_exposure(session: int, trial: int, activation: np.ndarray, ethanol: float, teaching_signal: float) -> None:
        nonlocal training_time
        training_time += protocol["exposure_duration_s"]
        memory.update(activation, teaching_signal)
        state_history.append({
            "phase": "training",
            "conditioning_session": session,
            "conditioning_trial": trial,
            "step": trial,
            "simulation_time_s": training_time,
            "x": np.nan,
            "y": np.nan,
            "heading_rad": np.nan,
            "odor_a_left": float(activation[0]),
            "odor_a_right": float(activation[0]),
            "odor_b_left": float(activation[1]),
            "odor_b_right": float(activation[1]),
            "ethanol_state": ethanol,
            "internal_reward_state": teaching_signal,
            "teaching_signal": teaching_signal,
            "weight_a": float(memory.weights[0]),
            "weight_b": float(memory.weights[1]),
            "action_forward_speed": np.nan,
            "action_turn_rate": np.nan,
            "final_choice": None,
            "choice_latency_s": np.nan,
            "completed": None,
        })
        training_time += protocol["intertrial_interval_s"]

    for session in range(1, config["experiment"]["conditioning_sessions"] + 1):
        if condition == "paired":
            apply_exposure(session, 1, np.array([1.0, 0.0]), 0.0, 0.0)
            apply_exposure(session, 2, np.array([0.0, 1.0]), 1.0, reward)
        elif condition == "unpaired":
            apply_exposure(session, 1, np.array([1.0, 0.0]), 0.0, 0.0)
            apply_exposure(session, 2, np.array([0.0, 1.0]), 0.0, 0.0)
            context = learning["unpaired_context_activation"]
            jitter = rng.uniform(-0.05, 0.05)
            apply_exposure(session, 3, np.array([context + jitter, context - jitter]), 1.0, reward)
        elif condition != "untrained":
            raise ValueError(f"Unknown condition: {condition}")
        history.append({"session": session, "condition": condition, "weight_a": float(memory.weights[0]), "weight_b": float(memory.weights[1])})
    if condition == "untrained":
        state_history.append({
            "phase": "training",
            "conditioning_session": 0,
            "conditioning_trial": 0,
            "step": 0,
            "simulation_time_s": 0.0,
            "x": np.nan,
            "y": np.nan,
            "heading_rad": np.nan,
            "odor_a_left": 0.0,
            "odor_a_right": 0.0,
            "odor_b_left": 0.0,
            "odor_b_right": 0.0,
            "ethanol_state": 0.0,
            "internal_reward_state": 0.0,
            "teaching_signal": 0.0,
            "weight_a": 0.0,
            "weight_b": 0.0,
            "action_forward_speed": np.nan,
            "action_turn_rate": np.nan,
            "final_choice": None,
            "choice_latency_s": np.nan,
            "completed": None,
        })
    return memory, history, state_history
