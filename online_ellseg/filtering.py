from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .types import EllipseTuple


def unwrap_angle_near(theta: float, reference: float) -> float:
    """Return an equivalent theta closest to reference."""

    period = math.pi
    while theta - reference > period / 2:
        theta -= period
    while theta - reference < -period / 2:
        theta += period
    return theta


@dataclass
class ExponentialEllipseFilter:
    alpha: float = 0.45
    value: Optional[EllipseTuple] = None

    def reset(self) -> None:
        self.value = None

    def update(self, ellipse: EllipseTuple) -> EllipseTuple:
        if self.value is None:
            self.value = ellipse
            return ellipse

        alpha = max(0.0, min(1.0, self.alpha))
        prev = self.value
        theta = unwrap_angle_near(ellipse[4], prev[4])
        filtered = (
            alpha * ellipse[0] + (1.0 - alpha) * prev[0],
            alpha * ellipse[1] + (1.0 - alpha) * prev[1],
            alpha * ellipse[2] + (1.0 - alpha) * prev[2],
            alpha * ellipse[3] + (1.0 - alpha) * prev[3],
            alpha * theta + (1.0 - alpha) * prev[4],
        )
        self.value = filtered
        return filtered
