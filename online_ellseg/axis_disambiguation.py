from __future__ import annotations

import itertools
import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Optional, Sequence, Tuple

import numpy as np

from .types import Vector3, VirtualAxisEstimate
from .virtual_axis import clamp, normalize, to_vector3


@dataclass(frozen=True)
class AxisCandidates:
    center_ray: Vector3
    candidate0: Vector3
    candidate1: Vector3


class EyeModelAxisDisambiguator:
    """Select the two-solution pupil axis branch with a multi-frame eye model.

    For each candidate branch, the virtual pupil center lies on the camera ray
    and the eye center is assumed to be a fixed offset along the pupil axis.
    A sliding window chooses the branch combination with the smallest common
    eye-center residual.
    """

    def __init__(
        self,
        window_size: int = 8,
        first_candidate: Optional[int] = None,
        eye_axis_distance: float = 1.0,
        smoothness_weight: float = 1.0,
    ):
        self.window_size = max(1, int(window_size))
        self.first_candidate = None if first_candidate is None else 1 if first_candidate == 1 else 0
        self.eye_axis_distance = float(eye_axis_distance)
        self.smoothness_weight = max(0.0, float(smoothness_weight))
        self._window: Deque[AxisCandidates] = deque(maxlen=self.window_size)

    def update(self, estimate: VirtualAxisEstimate) -> VirtualAxisEstimate:
        if not estimate.valid:
            return estimate

        item = AxisCandidates(
            center_ray=estimate.center_ray,
            candidate0=estimate.candidate0,
            candidate1=estimate.candidate1,
        )
        self._window.append(item)

        selected_index = self._select_current_index()
        if selected_index is None:
            return VirtualAxisEstimate.invalid("axis_disambiguation_warmup")
        normal = np.asarray(item.candidate1 if selected_index == 1 else item.candidate0, dtype=float)
        ray = np.asarray(item.center_ray, dtype=float)
        alpha = math.acos(clamp(-float(np.dot(normalize(normal), normalize(ray))), -1.0, 1.0))

        return VirtualAxisEstimate(
            valid=True,
            reason="ok",
            center_ray=estimate.center_ray,
            normal=to_vector3(normal),
            alpha=alpha,
            candidate0=estimate.candidate0,
            candidate1=estimate.candidate1,
            selected_index=selected_index,
        )

    def _select_current_index(self) -> Optional[int]:
        if len(self._window) < 3:
            return self.first_candidate

        paths, scores = score_candidate_paths(
            list(self._window),
            self.eye_axis_distance,
            self.smoothness_weight,
        )
        if len(paths) == 0 or len(scores) == 0:
            return self.first_candidate
        best_index = int(np.argmin(scores))
        return int(paths[best_index, -1])


def parse_axis_candidate(value: str) -> Optional[int]:
    text = str(value).strip().lower()
    if text == "auto":
        return None
    if text in ("0", "1"):
        return int(text)
    raise ValueError("--axis-candidate must be auto, 0, or 1")


def score_candidate_paths(
    records: Sequence[AxisCandidates],
    eye_axis_distance: float = 1.0,
    smoothness_weight: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Score all branch paths for a window with one vectorized least-squares solve."""

    if not records:
        return np.empty((0, 0), dtype=np.int8), np.empty((0,), dtype=float)

    window_size = len(records)
    paths = np.asarray(list(itertools.product((0, 1), repeat=window_size)), dtype=np.int8)
    rays = np.asarray([normalize(record.center_ray) for record in records], dtype=float)
    candidates = np.asarray(
        [
            [normalize(record.candidate0), normalize(record.candidate1)]
            for record in records
        ],
        dtype=float,
    )
    normals = candidates[np.arange(window_size)[None, :], paths]

    projections = np.eye(3, dtype=float)[None, :, :] - rays[:, :, None] * rays[:, None, :]
    projection_sum = projections.sum(axis=0)
    distance = float(eye_axis_distance)

    try:
        solver = np.linalg.pinv(projection_sum)
    except Exception:
        return paths, np.full((len(paths),), math.inf, dtype=float)

    rhs = np.einsum("ijk,pik->pj", projections, distance * normals)
    eye_centers = rhs @ solver.T
    residual_vectors = np.einsum("ijk,pik->pij", projections, eye_centers[:, None, :] - distance * normals)
    eye_scores = np.mean(np.sum(residual_vectors * residual_vectors, axis=2), axis=1)

    if window_size < 2 or smoothness_weight <= 0.0:
        return paths, eye_scores

    adjacent_dots = np.sum(normals[:, :-1, :] * normals[:, 1:, :], axis=2)
    smoothness_scores = np.mean(1.0 - adjacent_dots, axis=1)
    return paths, eye_scores + float(smoothness_weight) * smoothness_scores


def eye_model_residual(
    center_rays: Sequence[Iterable[float]],
    normals: Sequence[Iterable[float]],
    eye_axis_distance: float = 1.0,
) -> float:
    """Fit one common eye center and return mean squared model residual."""

    if len(center_rays) != len(normals) or not center_rays:
        return math.inf

    projection_sum = np.zeros((3, 3), dtype=float)
    rhs = np.zeros(3, dtype=float)
    projections = []
    unit_normals = []
    distance = float(eye_axis_distance)

    for ray_value, normal_value in zip(center_rays, normals):
        ray = normalize(ray_value)
        normal = normalize(normal_value)
        projection = np.eye(3, dtype=float) - np.outer(ray, ray)
        projection_sum += projection
        rhs += projection @ (distance * normal)
        projections.append(projection)
        unit_normals.append(normal)

    try:
        eye_center = np.linalg.lstsq(projection_sum, rhs, rcond=None)[0]
    except Exception:
        return math.inf

    residuals = [
        projection @ (eye_center - distance * normal)
        for projection, normal in zip(projections, unit_normals)
    ]
    return float(np.mean([np.dot(residual, residual) for residual in residuals]))


def path_smoothness_residual(normals: Sequence[Iterable[float]]) -> float:
    if len(normals) < 2:
        return 0.0

    unit_normals = [normalize(normal) for normal in normals]
    residuals = [
        1.0 - float(np.dot(unit_normals[index - 1], unit_normals[index]))
        for index in range(1, len(unit_normals))
    ]
    return float(np.mean(residuals))
