from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .axis_disambiguation import EyeModelAxisDisambiguator
from .ellseg_adapter import EllSegAdapter
from .filtering import ExponentialEllipseFilter
from .quality import QualityConfig, evaluate_pupil_quality
from .real_pupil_axis import CornealRefractionModel, estimate_real_axis
from .roi import FixedRoiProvider
from .types import EllipseObservation, EllipseTuple, Roi, Vector3, offset_ellipse
from .virtual_axis import estimate_virtual_axis


@dataclass
class PipelineConfig:
    quality: QualityConfig = QualityConfig()
    filter_alpha: float = 0.45
    flip_horizontal: bool = False
    camera_matrix: Optional[np.ndarray] = None
    dist_coeffs: Optional[np.ndarray] = None
    first_axis_candidate: Optional[int] = None
    corneal_model: CornealRefractionModel = CornealRefractionModel()
    axis_disambiguation_window: int = 8
    axis_disambiguation_smoothness: float = 1.0


class OnlineEllSegPipeline:
    def __init__(self, adapter: EllSegAdapter, roi_provider: FixedRoiProvider, config: PipelineConfig):
        self.adapter = adapter
        self.roi_provider = roi_provider
        self.config = config
        self.filter = ExponentialEllipseFilter(alpha=config.filter_alpha)
        self.axis_disambiguator = EyeModelAxisDisambiguator(
            window_size=config.axis_disambiguation_window,
            first_candidate=config.first_axis_candidate,
            smoothness_weight=config.axis_disambiguation_smoothness,
        )
        self.previous_valid: Optional[EllipseTuple] = None
        self.previous_axis: Optional[Vector3] = None
        self.frame_index = 0

    def process(self, frame, timestamp: Optional[float] = None) -> EllipseObservation:
        import cv2

        timestamp = time.perf_counter() if timestamp is None else timestamp
        if self.config.flip_horizontal:
            frame = cv2.flip(frame, 1)

        roi = self.roi_provider.get(frame)
        eye = roi.crop(frame)
        gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY) if eye.ndim == 3 else eye

        result = self.adapter.predict_gray(gray)
        pupil_roi = result["pupil"]
        iris_roi = result["iris"]
        pupil_frame = offset_ellipse(pupil_roi, roi)
        iris_frame = offset_ellipse(iris_roi, roi)

        quality = evaluate_pupil_quality(
            pupil_roi,
            gray.shape[:2],
            self._previous_in_roi(roi),
            self.config.quality,
        )

        filtered = None
        if result["status"] == "ok" and quality.valid:
            filtered = self.filter.update(pupil_frame)
            self.previous_valid = pupil_frame

        virtual_axis = None
        real_axis = None
        if result["status"] == "ok" and quality.valid and self.config.camera_matrix is not None:
            virtual_axis = estimate_virtual_axis(
                filtered or pupil_frame,
                self.config.camera_matrix,
                dist_coeffs=self.config.dist_coeffs,
                previous_normal=None,
                first_candidate=0 if self.config.first_axis_candidate is None else self.config.first_axis_candidate,
            )
            virtual_axis = self.axis_disambiguator.update(virtual_axis)
            if virtual_axis.valid:
                self.previous_axis = virtual_axis.normal
                real_axis = estimate_real_axis(virtual_axis, self.config.corneal_model)

        observation = EllipseObservation(
            frame_index=self.frame_index,
            timestamp=timestamp,
            roi=roi,
            status=result["status"],
            pupil_roi=pupil_roi,
            iris_roi=iris_roi,
            pupil_frame=pupil_frame,
            iris_frame=iris_frame,
            quality=quality,
            filtered_pupil_frame=filtered,
            virtual_axis=virtual_axis,
            real_axis=real_axis,
            seg_map=result["seg_map"],
        )
        updater = getattr(self.roi_provider, "update", None)
        if updater is not None:
            updater(observation, frame.shape[:2])
        self.frame_index += 1
        return observation

    def render_overlay(self, frame, observation: EllipseObservation):
        import cv2

        roi = observation.roi
        out = frame.copy()
        eye = roi.crop(frame)
        gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY) if eye.ndim == 3 else eye
        roi_overlay = self.adapter.overlay(gray, observation.seg_map, observation.pupil_roi, observation.iris_roi)
        out[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w] = roi_overlay
        color = (0, 220, 0) if observation.valid else (0, 0, 255)
        cv2.rectangle(out, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), color, 1)
        label = f"{observation.frame_index} {observation.quality.reason_text()}"
        cv2.putText(out, label, (roi.x, max(16, roi.y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        return out

    def _previous_in_roi(self, roi: Roi) -> Optional[EllipseTuple]:
        if self.previous_valid is None:
            return None
        cx, cy, axis_a, axis_b, theta = self.previous_valid
        return cx - roi.x, cy - roi.y, axis_a, axis_b, theta
