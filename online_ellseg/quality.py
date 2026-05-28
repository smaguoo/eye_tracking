from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from .types import EllipseTuple, PupilQuality


@dataclass(frozen=True)
class QualityConfig:
    min_area: float = 50.0
    max_axis_ratio: float = 4.0
    min_axis: float = 1.0
    max_center_jump: float = 35.0
    margin: float = 0.0


def _is_finite_ellipse(ellipse: EllipseTuple) -> bool:
    return all(math.isfinite(value) for value in ellipse)


def evaluate_pupil_quality(
    pupil: EllipseTuple,
    image_shape: Tuple[int, int],
    previous_valid: Optional[EllipseTuple],
    config: QualityConfig,
) -> PupilQuality:
    height, width = image_shape[:2]
    reasons = []
    cx, cy, axis_a, axis_b, _ = pupil

    if not _is_finite_ellipse(pupil):
        reasons.append("non_finite")
        return PupilQuality(False, 0.0, math.inf, math.inf, reasons)

    if axis_a <= 0 or axis_b <= 0:
        reasons.append("missing")
        return PupilQuality(False, 0.0, math.inf, math.inf, reasons)

    area = math.pi * axis_a * axis_b
    minor = max(1e-6, min(axis_a, axis_b))
    major = max(axis_a, axis_b)
    axis_ratio = major / minor

    if axis_a < config.min_axis or axis_b < config.min_axis:
        reasons.append("axis_too_small")
    if area < config.min_area:
        reasons.append("area_too_small")
    if axis_ratio > config.max_axis_ratio:
        reasons.append("axis_ratio")
    if not (config.margin <= cx < width - config.margin and config.margin <= cy < height - config.margin):
        reasons.append("center_outside")

    center_jump = 0.0
    if previous_valid is not None:
        center_jump = math.hypot(cx - previous_valid[0], cy - previous_valid[1])
        if center_jump > config.max_center_jump:
            reasons.append("center_jump")

    return PupilQuality(not reasons, area, axis_ratio, center_jump, reasons)
