from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .odor import OdorField


@dataclass(frozen=True)
class YMaze:
    config: dict
    odor_b_left: bool

    @property
    def source_positions(self) -> np.ndarray:
        a = self.config["arena"]
        left = np.array([-a["source_x"], a["source_y"]])
        right = np.array([a["source_x"], a["source_y"]])
        return np.stack([right, left]) if self.odor_b_left else np.stack([left, right])

    @property
    def odor_field(self) -> OdorField:
        o = self.config["odor"]
        return OdorField(self.source_positions, o["source_strength"], o["length_scale"])

    def constrain_to_stem(self, position: np.ndarray) -> np.ndarray:
        result = position.copy()
        a = self.config["arena"]
        if result[1] < a["junction_y"]:
            result[0] = np.clip(result[0], -a["stem_half_width"], a["stem_half_width"])
        return result

    def choice(self, position: np.ndarray) -> str | None:
        a = self.config["arena"]
        if position[1] >= a["choice_y"] and abs(position[0]) >= a["choice_x"]:
            chose_left = position[0] < 0
            return "B" if chose_left == self.odor_b_left else "A"
        return None

