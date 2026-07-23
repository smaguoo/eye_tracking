from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .axis_disambiguation import parse_axis_candidate
from .camera import CameraConfig, LatestFrameCamera
from .ellseg_adapter import EllSegAdapter, EllSegPaths
from .geometry_gaze import (
    ScreenGeometry,
    origin_at_depth,
    origin_from_screen_center_offset,
    predict_screen_point,
)
from .pipeline import OnlineEllSegPipeline, PipelineConfig
from .quality import QualityConfig
from .roi import FixedRoiProvider, click_center_roi_interactively, select_roi_interactively
from .types import Roi
from .virtual_axis import approximate_camera_matrix, load_calibration, parse_camera_matrix


@dataclass(frozen=True)
class ReferenceTarget:
    index: int
    row: int
    col: int
    x: int
    y: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Display a monocular pure-geometry gaze point on screen.")
    parser.add_argument(
        "--geometry",
        action="store_true",
        help="Accepted for compatibility; this command always uses pure geometry.",
    )
    parser.add_argument("--ellseg-root", type=Path, default=EllSegPaths.root)
    parser.add_argument("--weights", type=Path, default=EllSegPaths.weights)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--backend", default="dshow", choices=["dshow", "msmf", "vfw", "auto"])
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-warmup-seconds", type=float, default=0.8)
    parser.add_argument("--min-first-frame-mean", type=float, default=2.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--ellipse-source", default="network", choices=["network", "mask", "none"])
    parser.add_argument("--dark-threshold", type=int, default=20)
    parser.add_argument("--roi", default="", help="Fixed eye ROI as x,y,w,h.")
    parser.add_argument("--click-roi", action="store_true", help="Click eye center to create a fixed-size ROI.")
    parser.add_argument("--click-roi-width", type=int, default=96)
    parser.add_argument("--click-roi-height", type=int, default=56)
    parser.add_argument("--select-roi", action="store_true", help="Open a one-time ROI selector.")
    parser.add_argument("--flip-horizontal", action="store_true")
    parser.add_argument("--intrinsics", default="", help="Eye-camera intrinsics as fx,fy,cx,cy or full 3x3 matrix.")
    parser.add_argument("--calibration", type=Path, default=None, help="JSON calibration file with camera_matrix and dist_coeffs.")
    parser.add_argument("--approx-focal-px", type=float, default=500.0)
    parser.add_argument(
        "--axis-candidate",
        type=parse_axis_candidate,
        default=None,
        metavar="{auto,0,1}",
        help="Initial pupil-axis branch. auto waits for the sliding-window eye model instead of fixing 0 or 1.",
    )
    parser.add_argument("--axis-disambiguation-window", type=int, default=8)
    parser.add_argument("--axis-disambiguation-smoothness", type=float, default=1.0)
    parser.add_argument("--filter-alpha", type=float, default=0.45, help="EMA alpha for pupil ellipse smoothing.")
    parser.add_argument("--smooth-alpha", type=float, default=0.25, help="EMA alpha for gaze-dot smoothing.")
    parser.add_argument(
        "--max-output-scale",
        type=float,
        default=1.5,
        help="Reject raw predictions farther than this screen multiple from the viewport.",
    )
    parser.add_argument("--screen-width-mm", type=float, default=344.0)
    parser.add_argument("--screen-height-mm", type=float, default=194.0)
    parser.add_argument("--screen-center-x-mm", type=float, default=0.0)
    parser.add_argument("--screen-center-y-mm", type=float, default=0.0)
    parser.add_argument("--screen-z-mm", type=float, default=0.0)
    parser.add_argument("--eye-depth-mm", type=float, default=600.0)
    parser.add_argument("--eye-screen-center-x-mm", type=float, default=0.0)
    parser.add_argument("--eye-screen-center-y-mm", type=float, default=0.0)
    parser.add_argument("--eye-screen-center-z-mm", type=float, default=None)
    parser.add_argument("--eye-midpoint-x-mm", type=float, default=None)
    parser.add_argument("--eye-midpoint-y-mm", type=float, default=None)
    parser.add_argument("--eye-midpoint-z-mm", type=float, default=None)
    parser.add_argument("--kappa-yaw-deg", type=float, default=0.0)
    parser.add_argument("--kappa-pitch-deg", type=float, default=0.0)
    parser.add_argument("--invert-gaze-x", action="store_true")
    parser.add_argument("--invert-gaze-y", action="store_true")
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--display-width", type=int, default=0)
    parser.add_argument("--display-height", type=int, default=0)
    parser.add_argument("--hide-eye-debug", action="store_true")
    parser.add_argument("--eye-debug-width", type=int, default=380)
    parser.add_argument("--hide-reference-grid", action="store_true")
    parser.add_argument("--reference-grid-margin-ratio", type=float, default=0.18)
    parser.add_argument("--csv", type=Path, default=Path("outputs/live_gaze_geometry_mono.csv"))
    parser.add_argument("--max-frames", type=int, default=0, help=argparse.SUPPRESS)
    return parser


def wait_for_first_frame(
    camera: LatestFrameCamera,
    timeout: float = 5.0,
    warmup_seconds: float = 0.8,
    min_mean: float = 2.0,
):
    deadline = time.perf_counter() + timeout
    warmup_deadline = time.perf_counter() + max(0.0, float(warmup_seconds))
    latest = None
    latest_timestamp = 0.00
    while time.perf_counter() < deadline:
        ok, frame, timestamp = camera.read_latest()
        if ok:
            latest = frame
            latest_timestamp = timestamp
            if time.perf_counter() >= warmup_deadline and float(frame.mean()) >= float(min_mean):
                return frame, timestamp
        time.sleep(0.01)
    if latest is not None:
        return latest, latest_timestamp
    raise RuntimeError("Timed out waiting for first camera frame.")


def choose_roi(args, first_frame) -> Optional[Roi]:
    frame = selection_frame(args, first_frame)
    if args.click_roi:
        roi = click_center_roi_interactively(
            frame,
            "Click eye center",
            args.click_roi_width,
            args.click_roi_height,
        )
        print(f"click_roi: {roi.x},{roi.y},{roi.w},{roi.h}")
        return roi
    if args.select_roi:
        roi = select_roi_interactively(frame)
        print(f"selected_roi: {roi.x},{roi.y},{roi.w},{roi.h}")
        return roi
    if args.roi:
        return Roi.from_string(args.roi)
    return None


def selection_frame(args, first_frame):
    if not args.flip_horizontal:
        return first_frame
    return cv2.flip(first_frame, 1)


def get_screen_size(default: Tuple[int, int] = (1280, 720)) -> Tuple[int, int]:
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        width = int(root.winfo_screenwidth())
        height = int(root.winfo_screenheight())
        root.destroy()
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass
    return default


def make_reference_targets(width: int, height: int, rows: int = 3, cols: int = 3, margin_ratio: float = 0.18):
    margin_x = int(round(width * float(margin_ratio)))
    margin_y = int(round(height * float(margin_ratio)))
    usable_w = max(1, width - 2 * margin_x)
    usable_h = max(1, height - 2 * margin_y)
    targets = []
    index = 1
    for row in range(rows):
        y = margin_y + int(round(row * usable_h / max(1, rows - 1)))
        for col in range(cols):
            x = margin_x + int(round(col * usable_w / max(1, cols - 1)))
            targets.append(ReferenceTarget(index=index, row=row, col=col, x=x, y=y))
            index += 1
    return targets


def draw_prediction(
    width: int,
    height: int,
    point,
    valid: bool,
    text: str,
    eye_debug=None,
    debug_lines=None,
    reference_targets=None,
) -> np.ndarray:
    image = np.full((height, width, 3), (20, 20, 20), dtype=np.uint8)
    if reference_targets:
        draw_reference_grid(image, reference_targets)
    cv2.putText(image, text, (36, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(image, "Predicted gaze point", (36, height - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (190, 190, 190), 2, cv2.LINE_AA)
    if valid and point is not None:
        x, y = point
        cv2.circle(image, (x, y), 26, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), 44, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(image, (x - 34, y), (x + 34, y), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(image, (x, y - 34), (x, y + 34), (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(image, "No valid pupil axis", (36, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (80, 160, 255), 2, cv2.LINE_AA)
    if eye_debug is not None:
        draw_eye_debug_panel(image, eye_debug, debug_lines or [])
    return image


def draw_reference_grid(image: np.ndarray, targets) -> None:
    rows = sorted(set(target.row for target in targets))
    cols = sorted(set(target.col for target in targets))
    grid_color = (58, 58, 58)
    circle_color = (72, 88, 110)
    ring_color = (128, 145, 170)
    text_color = (215, 226, 238)

    for row in rows:
        row_targets = [target for target in targets if target.row == row]
        if len(row_targets) > 1:
            cv2.line(image, (row_targets[0].x, row_targets[0].y), (row_targets[-1].x, row_targets[-1].y), grid_color, 1, cv2.LINE_AA)
    for col in cols:
        col_targets = [target for target in targets if target.col == col]
        if len(col_targets) > 1:
            cv2.line(image, (col_targets[0].x, col_targets[0].y), (col_targets[-1].x, col_targets[-1].y), grid_color, 1, cv2.LINE_AA)

    for target in targets:
        cv2.circle(image, (target.x, target.y), 36, circle_color, -1, cv2.LINE_AA)
        cv2.circle(image, (target.x, target.y), 52, ring_color, 2, cv2.LINE_AA)
        label = str(target.index)
        size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.35, 3)
        cv2.putText(
            image,
            label,
            (target.x - size[0] // 2, target.y + size[1] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.35,
            text_color,
            3,
            cv2.LINE_AA,
        )


def draw_eye_debug_panel(image: np.ndarray, eye_debug: np.ndarray, lines) -> None:
    height, width = image.shape[:2]
    inset_height, inset_width = eye_debug.shape[:2]
    margin = 28
    if width <= 2 * margin or height <= 2 * margin:
        return
    text_height = 24 * max(1, len(lines))
    panel_width = min(width - 2 * margin, max(inset_width + 24, 330))
    panel_height = inset_height + text_height + 54
    x0 = max(margin, width - panel_width - margin)
    y0 = 82
    if y0 + panel_height > height - margin:
        y0 = max(margin, height - panel_height - margin)
    x1 = min(width - margin, x0 + panel_width)
    y1 = min(height - margin, y0 + panel_height)

    cv2.rectangle(image, (x0, y0), (x1, y1), (8, 8, 8), -1)
    cv2.rectangle(image, (x0, y0), (x1, y1), (130, 130, 130), 1)
    cv2.putText(image, "Eye ROI / EllSeg", (x0 + 12, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1, cv2.LINE_AA)

    ix = x0 + 12
    iy = y0 + 38
    paste_width = min(inset_width, x1 - ix - 12)
    paste_height = min(inset_height, y1 - iy - text_height - 18)
    if paste_width > 0 and paste_height > 0:
        image[iy : iy + paste_height, ix : ix + paste_width] = eye_debug[:paste_height, :paste_width]
        cv2.rectangle(image, (ix, iy), (ix + paste_width, iy + paste_height), (210, 210, 210), 1)

    ty = iy + paste_height + 24
    for line in lines:
        if ty > y1 - 10:
            break
        cv2.putText(image, line, (x0 + 12, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
        ty += 22


def make_eye_debug_inset(frame, observation, pipeline: OnlineEllSegPipeline, max_width: int, max_height: int):
    if max_width <= 0 or max_height <= 0:
        return None
    overlay = pipeline.render_overlay(frame, observation)
    roi = observation.roi.clamp(overlay.shape[:2])
    crop = roi.crop(overlay)
    if crop.size == 0:
        crop = overlay
    height, width = crop.shape[:2]
    scale = min(max_width / max(1, width), max_height / max(1, height))
    if not math.isfinite(scale) or scale <= 0:
        return None
    return cv2.resize(
        crop,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )


def debug_lines_for_observation(observation, yaw: float, pitch: float, reason: str):
    lines = [f"quality: {observation.quality.reason_text()}"]
    axis = preferred_axis(observation)
    if axis is not None and axis.valid:
        name = "real" if axis is getattr(observation, "real_axis", None) else "virtual"
        gamma = getattr(axis, "gamma", float("nan"))
        if math.isfinite(gamma):
            lines.append(f"{name}: c{axis.selected_index} gamma {math.degrees(gamma):.1f} deg")
        else:
            lines.append(f"{name}: c{axis.selected_index} alpha {math.degrees(axis.alpha):.1f} deg")
        if math.isfinite(yaw) and math.isfinite(pitch):
            lines.append(f"yaw/pitch: {yaw:.1f}, {pitch:.1f} deg")
    else:
        axis_reason = "none" if axis is None else axis.reason
        lines.append(f"axis: invalid ({axis_reason})")
    if reason != "ok":
        lines.append(f"status: {reason[:36]}")
    return lines


def preferred_axis(observation):
    if getattr(observation, "real_axis", None) is not None and observation.real_axis.valid:
        return observation.real_axis
    if observation.virtual_axis is not None and observation.virtual_axis.valid:
        return observation.virtual_axis
    return observation.real_axis or observation.virtual_axis


def geometry_center_ray(observation):
    virtual_axis = getattr(observation, "virtual_axis", None)
    if virtual_axis is not None and virtual_axis.valid:
        return virtual_axis.center_ray
    axis = preferred_axis(observation)
    return getattr(axis, "center_ray", None)


def axis_angles_for_debug(axis):
    if axis is None or not getattr(axis, "valid", False):
        return float("nan"), float("nan")
    try:
        return direction_angles_from_normal(axis.normal)
    except Exception:
        return float("nan"), float("nan")


def direction_angles_from_normal(normal) -> Tuple[float, float]:
    nx, ny, nz = [float(value) for value in normal]
    dx, dy, dz = -nx, -ny, -nz
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm < 1e-12 or not math.isfinite(norm):
        raise ValueError("Invalid normal vector")
    dx, dy, dz = dx / norm, dy / norm, dz / norm
    yaw = math.degrees(math.atan2(dx, dz))
    pitch = math.degrees(math.atan2(dy, math.sqrt(dx * dx + dz * dz)))
    return yaw, pitch


def fixed_eye_midpoint_enabled(args) -> bool:
    return args.eye_midpoint_x_mm is not None or args.eye_midpoint_y_mm is not None or args.eye_midpoint_z_mm is not None


def fixed_eye_offset_enabled(args) -> bool:
    return args.eye_screen_center_z_mm is not None


def fixed_single_origin(args, geometry: ScreenGeometry):
    if fixed_eye_midpoint_enabled(args):
        if args.eye_midpoint_x_mm is None or args.eye_midpoint_y_mm is None or args.eye_midpoint_z_mm is None:
            raise ValueError("--eye-midpoint-x-mm, --eye-midpoint-y-mm and --eye-midpoint-z-mm must be provided together.")
        return np.array([args.eye_midpoint_x_mm, args.eye_midpoint_y_mm, args.eye_midpoint_z_mm], dtype=float)
    if fixed_eye_offset_enabled(args):
        return origin_from_screen_center_offset(
            geometry,
            args.eye_screen_center_x_mm,
            args.eye_screen_center_y_mm,
            args.eye_screen_center_z_mm,
        )
    return None


def clip_point(x: float, y: float, width: int, height: int) -> Tuple[int, int]:
    px = int(round(max(0.0, min(float(width - 1), x))))
    py = int(round(max(0.0, min(float(height - 1), y))))
    return px, py


def smooth_point(previous: Optional[Tuple[float, float]], current: Tuple[float, float], alpha: float) -> Tuple[float, float]:
    if previous is None:
        return current
    alpha = max(0.0, min(1.0, float(alpha)))
    return (
        alpha * current[0] + (1.0 - alpha) * previous[0],
        alpha * current[1] + (1.0 - alpha) * previous[1],
    )


def point_within_output_bounds(x: float, y: float, width: int, height: int, scale: float) -> bool:
    if not math.isfinite(x) or not math.isfinite(y):
        return False
    margin_x = max(0.0, (float(scale) - 1.0) * width)
    margin_y = max(0.0, (float(scale) - 1.0) * height)
    return -margin_x <= x <= width + margin_x and -margin_y <= y <= height + margin_y


def main() -> None:
    args = build_parser().parse_args()

    geometry = ScreenGeometry(
        width_mm=args.screen_width_mm,
        height_mm=args.screen_height_mm,
        center_x_mm=args.screen_center_x_mm,
        center_y_mm=args.screen_center_y_mm,
        z_mm=args.screen_z_mm,
    )
    geometry.validate()

    screen_width, screen_height = get_screen_size()
    display_width = args.display_width if args.display_width > 0 else screen_width
    display_height = args.display_height if args.display_height > 0 else screen_height
    reference_targets = None
    if not args.hide_reference_grid:
        reference_targets = make_reference_targets(display_width, display_height, 3, 3, args.reference_grid_margin_ratio)

    adapter = EllSegAdapter(
        EllSegPaths(root=args.ellseg_root, weights=args.weights),
        device=args.device,
        ellipse_source=args.ellipse_source,
        dark_threshold=args.dark_threshold,
    )
    camera = LatestFrameCamera(
        CameraConfig(index=args.camera, width=args.width, height=args.height, fps=args.fps, backend=args.backend)
    ).start()

    try:
        first_frame, _ = wait_for_first_frame(
            camera,
            warmup_seconds=args.camera_warmup_seconds,
            min_mean=args.min_first_frame_mean,
        )
        dist_coeffs = None
        if args.calibration:
            camera_matrix, dist_coeffs = load_calibration(args.calibration)
        elif args.intrinsics:
            camera_matrix = parse_camera_matrix(args.intrinsics)
        else:
            frame_height, frame_width = first_frame.shape[:2]
            camera_matrix = approximate_camera_matrix(frame_width, frame_height, args.approx_focal_px)

        quality_config = QualityConfig()
        pipeline_config = PipelineConfig(
            quality=quality_config,
            filter_alpha=args.filter_alpha,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            flip_horizontal=args.flip_horizontal,
            first_axis_candidate=args.axis_candidate,
            axis_disambiguation_window=args.axis_disambiguation_window,
            axis_disambiguation_smoothness=args.axis_disambiguation_smoothness,
        )
        roi = choose_roi(args, first_frame)
        pipeline = OnlineEllSegPipeline(adapter, FixedRoiProvider(roi), pipeline_config)

        window = "predicted gaze"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        if not args.windowed:
            cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(window, display_width, display_height)

        args.csv.parent.mkdir(parents=True, exist_ok=True)
        handle = args.csv.open("w", newline="", encoding="utf-8")
        fieldnames = [
            "timestamp",
            "valid",
            "raw_x",
            "raw_y",
            "smooth_x",
            "smooth_y",
            "yaw_deg",
            "pitch_deg",
            "axis_reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        smoothed = None
        last_timestamp = 0.0
        processed_frames = 0
        try:
            while True:
                ok, frame, timestamp = camera.read_latest()
                if not ok or timestamp == last_timestamp:
                    time.sleep(0.002)
                    continue
                last_timestamp = timestamp
                observation = pipeline.process(frame, timestamp)

                point = None
                raw_x = raw_y = float("nan")
                reason = "no_axis"
                valid = False
                eye_debug = None
                debug_lines = []

                axis = preferred_axis(observation)
                center_ray = geometry_center_ray(observation)
                yaw, pitch = axis_angles_for_debug(axis)
                fixed_origin = fixed_single_origin(args, geometry)

                if observation.valid and axis is not None and axis.valid and (fixed_origin is not None or center_ray is not None):
                    try:
                        origin = fixed_origin if fixed_origin is not None else origin_at_depth(center_ray, args.eye_depth_mm)
                        prediction = predict_screen_point(
                            axis,
                            origin,
                            geometry,
                            display_width,
                            display_height,
                            kappa_yaw_deg=args.kappa_yaw_deg,
                            kappa_pitch_deg=args.kappa_pitch_deg,
                            invert_x=args.invert_gaze_x,
                            invert_y=args.invert_gaze_y,
                        )
                    except Exception:
                        prediction = None
                        reason = "geometry_failed"

                    if prediction is not None and prediction.valid:
                        raw_x, raw_y = prediction.x_px, prediction.y_px
                        if point_within_output_bounds(raw_x, raw_y, display_width, display_height, args.max_output_scale):
                            smoothed = smooth_point(smoothed, (raw_x, raw_y), args.smooth_alpha)
                            point = clip_point(smoothed[0], smoothed[1], display_width, display_height)
                            reason = "ok"
                            valid = True
                        else:
                            reason = "offscreen"
                    elif prediction is not None:
                        reason = prediction.reason

                text = f"geometry ray-plane | screen {geometry.width_mm:.0f}x{geometry.height_mm:.0f}mm z={geometry.z_mm:.0f} | q/Esc quits"
                if not args.hide_eye_debug:
                    debug_frame = cv2.flip(frame, 1) if args.flip_horizontal else frame
                    max_eye_height = max(120, int(display_height * 0.36))
                    eye_debug = make_eye_debug_inset(
                        debug_frame,
                        observation,
                        pipeline,
                        args.eye_debug_width,
                        max_eye_height,
                    )
                    debug_lines = debug_lines_for_observation(observation, yaw, pitch, reason)

                csv_row = {
                    "timestamp": f"{timestamp:.6f}",
                    "valid": int(valid),
                    "raw_x": raw_x,
                    "raw_y": raw_y,
                    "smooth_x": "" if smoothed is None else smoothed[0],
                    "smooth_y": "" if smoothed is None else smoothed[1],
                    "yaw_deg": yaw,
                    "pitch_deg": pitch,
                    "axis_reason": reason,
                }
                image = draw_prediction(
                    display_width,
                    display_height,
                    point,
                    valid,
                    text,
                    eye_debug,
                    debug_lines,
                    reference_targets,
                )
                cv2.imshow(window, image)
                key = cv2.waitKey(1) & 0xFF
                writer.writerow(csv_row)
                handle.flush()

                processed_frames += 1
                if args.max_frames > 0 and processed_frames >= args.max_frames:
                    break
                if key in (27, ord("q")):
                    break
        finally:
            handle.close()
            cv2.destroyWindow(window)
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
