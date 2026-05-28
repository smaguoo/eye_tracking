from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .types import Roi


@dataclass
class FixedRoiProvider:
    roi: Optional[Roi] = None

    def get(self, frame) -> Roi:
        height, width = frame.shape[:2]
        if self.roi is None:
            return Roi(0, 0, width, height)
        return self.roi.clamp((height, width))


def select_roi_interactively(
    frame,
    window_name: str = "Select eye ROI",
    max_display_width: int = 1280,
    max_display_height: int = 800,
) -> Roi:
    import cv2

    height, width = frame.shape[:2]
    scale = min(
        1.0,
        float(max_display_width) / max(1, width),
        float(max_display_height) / max(1, height),
    )
    display_width = max(1, int(round(width * scale)))
    display_height = max(1, int(round(height * scale)))

    state = {
        "dragging": False,
        "start": None,
        "current": None,
        "roi": None,
    }
    base = frame.copy()

    def normalized_roi(p0, p1) -> Optional[Roi]:
        x0, y0 = p0
        x1, y1 = p1
        x = min(x0, x1)
        y = min(y0, y1)
        w = abs(x1 - x0)
        h = abs(y1 - y0)
        if w < 4 or h < 4:
            return None
        return Roi(int(x), int(y), int(w), int(h)).clamp(frame.shape[:2])

    def to_image_point(x, y):
        return (
            int(round(max(0, min(display_width - 1, x)) / scale)),
            int(round(max(0, min(display_height - 1, y)) / scale)),
        )

    def scale_point(point):
        return int(round(point[0] * scale)), int(round(point[1] * scale))

    def scale_roi(roi: Roi) -> Roi:
        x0, y0 = scale_point((roi.x, roi.y))
        x1, y1 = scale_point((roi.x + roi.w, roi.y + roi.h))
        return Roi(x0, y0, max(1, x1 - x0), max(1, y1 - y0))

    def on_mouse(event, x, y, flags, param):
        point = to_image_point(int(x), int(y))
        if event == cv2.EVENT_LBUTTONDOWN:
            state["dragging"] = True
            state["start"] = point
            state["current"] = point
            state["roi"] = None
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            state["current"] = point
        elif event == cv2.EVENT_LBUTTONUP and state["dragging"]:
            state["dragging"] = False
            state["current"] = point
            roi = normalized_roi(state["start"], state["current"])
            if roi is not None:
                state["roi"] = roi

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_width, display_height)
    cv2.setMouseCallback(window_name, on_mouse)
    try:
        while True:
            preview = base.copy()
            label = "Drag ROI box. Space/Enter accepts. R resets. Esc/Q cancels."
            cv2.putText(preview, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(preview, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 1, cv2.LINE_AA)

            roi = state["roi"]
            if state["dragging"] and state["start"] is not None and state["current"] is not None:
                roi = normalized_roi(state["start"], state["current"])
            if roi is not None:
                cv2.rectangle(preview, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), (0, 220, 255), 2)
                text = f"{roi.x},{roi.y},{roi.w},{roi.h}"
                cv2.putText(preview, text, (roi.x, max(48, roi.y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)

            if scale < 1.0:
                display = cv2.resize(preview, (display_width, display_height), interpolation=cv2.INTER_AREA)
                if roi is not None:
                    shown = scale_roi(roi)
                    cv2.rectangle(display, (shown.x, shown.y), (shown.x + shown.w, shown.y + shown.h), (0, 220, 255), 2)
                cv2.imshow(window_name, display)
            else:
                cv2.imshow(window_name, preview)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 32) and state["roi"] is not None:
                return state["roi"]
            if key in (27, ord("q")):
                raise RuntimeError("ROI selection cancelled.")
            if key == ord("r"):
                state["dragging"] = False
                state["start"] = None
                state["current"] = None
                state["roi"] = None
    finally:
        cv2.destroyWindow(window_name)


def click_center_roi_interactively(
    frame,
    window_name: str = "Click eye center",
    roi_width: int = 96,
    roi_height: int = 56,
) -> Roi:
    points = click_points_interactively(frame, window_name, 1)
    return roi_from_center(points[0], roi_width, roi_height, frame.shape[:2])


def click_binocular_center_rois_interactively(
    frame,
    roi_width: int = 96,
    roi_height: int = 56,
) -> Tuple[Roi, Roi]:
    points = click_points_interactively(frame, "Click left eye, then right eye", 2)
    rois = [roi_from_center(point, roi_width, roi_height, frame.shape[:2]) for point in points]
    left, right = sorted(rois, key=lambda roi: roi.x + roi.w / 2.0)
    return left, right


def click_points_interactively(frame, window_name: str, count: int) -> List[Tuple[int, int]]:
    import cv2

    points: List[Tuple[int, int]] = []
    base = frame.copy()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < count:
            points.append((int(x), int(y)))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    try:
        while len(points) < count:
            preview = base.copy()
            label = f"Click eye center {len(points) + 1}/{count}. R resets. Esc/Q cancels."
            cv2.putText(preview, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(preview, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 1, cv2.LINE_AA)
            for index, point in enumerate(points, start=1):
                cv2.drawMarker(preview, point, (0, 220, 255), cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
                cv2.putText(
                    preview,
                    str(index),
                    (point[0] + 8, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 220, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q")):
                raise RuntimeError("Click ROI selection cancelled.")
            if key == ord("r"):
                points.clear()
    finally:
        cv2.destroyWindow(window_name)
    return points


def roi_from_center(point: Tuple[int, int], width: int, height: int, frame_shape) -> Roi:
    cx, cy = point
    roi = Roi(
        int(round(cx - width / 2.0)),
        int(round(cy - height / 2.0)),
        int(width),
        int(height),
    )
    return roi.clamp(frame_shape)
