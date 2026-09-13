from __future__ import annotations

import numpy as np
import pandas as pd


def preference_index(choices) -> float:
    values = np.asarray(choices)
    valid = values[np.isin(values, ["A", "B"])]
    if len(valid) == 0:
        return float("nan")
    return float((np.sum(valid == "B") - np.sum(valid == "A")) / len(valid))


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, frame in results.groupby("condition", sort=False):
        n = len(frame)
        pi = preference_index(frame["choice"])
        se = np.sqrt(max(0.0, 1.0 - pi**2) / n)
        rows.append({
            "condition": condition,
            "n": n,
            "n_b": int(frame["chose_b"].sum()),
            "percent_b": 100.0 * float(frame["chose_b"].mean()),
            "preference_index": pi,
            "pi_ci95_low": max(-1.0, pi - 1.96 * se),
            "pi_ci95_high": min(1.0, pi + 1.96 * se),
            "mean_latency_s": float(frame["latency_s"].mean()),
            "mean_distance": float(frame["distance"].mean()),
            "completion_rate": float(frame["completed"].mean()),
            "mean_weight_a": float(frame["weight_a"].mean()),
            "mean_weight_b": float(frame["weight_b"].mean()),
        })
    return pd.DataFrame(rows)

