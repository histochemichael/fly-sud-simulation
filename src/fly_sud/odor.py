from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OdorField:
    """Two-channel, distance-decaying odor field (channels A then B)."""

    source_positions: np.ndarray
    strength: float
    length_scale: float

    def concentration(self, points: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        distances = np.linalg.norm(points[:, None, :] - self.source_positions[None, :, :], axis=2)
        return self.strength * np.exp(-distances / self.length_scale)


def antenna_positions(position, heading, separation, forward_offset) -> np.ndarray:
    """Return left and right antenna positions for a planar fly body."""
    forward = np.array([np.cos(heading), np.sin(heading)])
    leftward = np.array([-np.sin(heading), np.cos(heading)])
    center = np.asarray(position, dtype=float) + forward_offset * forward
    return np.stack([center + 0.5 * separation * leftward, center - 0.5 * separation * leftward])


def bilateral_readout(field, position, heading, separation, forward_offset, sensory_noise, rng) -> np.ndarray:
    """Return odor-channel x antenna concentrations with optional noise."""
    antennae = antenna_positions(position, heading, separation, forward_offset)
    readout = field.concentration(antennae).T
    if sensory_noise:
        readout += rng.normal(0.0, sensory_noise, size=readout.shape)
    return np.clip(readout, 0.0, None)

