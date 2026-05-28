from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class TargetPoint:
    index: int
    row: int
    col: int
    x: int
    y: int


def parse_grid(value: str) -> Tuple[int, int]:
    clean = value.lower().replace("*", "x")
    parts = [part.strip() for part in clean.split("x") if part.strip()]
    if len(parts) != 2:
        raise ValueError("Grid must be formatted as rowsxcols, for example 3x3")
    rows, cols = int(parts[0]), int(parts[1])
    if rows < 2 or cols < 2:
        raise ValueError("Grid must have at least 2 rows and 2 columns")
    return rows, cols


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


def make_grid_targets(width: int, height: int, rows: int, cols: int, margin_ratio: float) -> List[TargetPoint]:
    margin_ratio = max(0.05, min(0.35, margin_ratio))
    x0 = int(round(width * margin_ratio))
    x1 = int(round(width * (1.0 - margin_ratio)))
    y0 = int(round(height * margin_ratio))
    y1 = int(round(height * (1.0 - margin_ratio)))

    targets: List[TargetPoint] = []
    index = 1
    for row in range(rows):
        y = y0 if rows == 1 else int(round(y0 + (y1 - y0) * row / (rows - 1)))
        for col in range(cols):
            x = x0 if cols == 1 else int(round(x0 + (x1 - x0) * col / (cols - 1)))
            targets.append(TargetPoint(index=index, row=row, col=col, x=x, y=y))
            index += 1
    return targets


def target_for_elapsed(
    elapsed: float,
    targets: List[TargetPoint],
    prep_seconds: float,
    dwell_seconds: float,
    cycles: int,
) -> Tuple[str, Optional[TargetPoint], int, float]:
    if elapsed < prep_seconds:
        return "prep", None, 0, max(0.0, prep_seconds - elapsed)

    sequence_elapsed = elapsed - prep_seconds
    total_target_time = len(targets) * dwell_seconds * cycles
    if sequence_elapsed >= total_target_time:
        return "done", None, cycles, 0.0

    step = int(sequence_elapsed // dwell_seconds)
    target = targets[step % len(targets)]
    cycle = step // len(targets) + 1
    remaining = dwell_seconds - (sequence_elapsed - step * dwell_seconds)
    return "target", target, cycle, max(0.0, remaining)


def draw_target_frame(
    width: int,
    height: int,
    targets: List[TargetPoint],
    active: Optional[TargetPoint],
    phase: str,
    remaining: float,
    cycle: int,
    cycles: int,
) -> np.ndarray:
    image = np.full((height, width, 3), (18, 18, 18), dtype=np.uint8)
    _draw_grid_reference(image, targets)

    if active is not None:
        cv2.circle(image, (active.x, active.y), 38, (0, 220, 255), -1, cv2.LINE_AA)
        cv2.circle(image, (active.x, active.y), 54, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(image, (active.x, active.y), 8, (0, 0, 0), -1, cv2.LINE_AA)
        label = str(active.index)
        scale = 1.5
        thickness = 3
        size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv2.putText(
            image,
            label,
            (active.x - size[0] // 2, active.y + size[1] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    if phase == "prep":
        title = "Get ready. Look at each target in number order."
    elif phase == "target" and active is not None:
        title = f"Look at target {active.index}   cycle {cycle}/{cycles}   {remaining:0.1f}s"
    else:
        title = "Done. Press q or Esc to close."

    cv2.putText(image, title, (40, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(image, "Keep your head still. Follow the highlighted dot.", (40, height - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (210, 210, 210), 2, cv2.LINE_AA)
    return image


def draw_target_chart(width: int, height: int, targets: List[TargetPoint]) -> np.ndarray:
    image = np.full((height, width, 3), (245, 247, 250), dtype=np.uint8)
    _draw_grid_reference(image, targets, dark=False)
    for target in targets:
        cv2.circle(image, (target.x, target.y), 32, (0, 160, 255), -1, cv2.LINE_AA)
        cv2.circle(image, (target.x, target.y), 45, (30, 30, 30), 2, cv2.LINE_AA)
        label = str(target.index)
        size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.putText(
            image,
            label,
            (target.x - size[0] // 2, target.y + size[1] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
    cv2.putText(image, "Fixation sequence: look at points 1 -> 9", (40, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2, cv2.LINE_AA)
    return image


def save_target_chart(path: Path, width: int, height: int, rows: int, cols: int, margin_ratio: float) -> None:
    path = Path(path)
    targets = make_grid_targets(width, height, rows, cols, margin_ratio)
    image = draw_target_chart(width, height, targets)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def _draw_grid_reference(image: np.ndarray, targets: List[TargetPoint], dark: bool = True) -> None:
    color = (70, 70, 70) if dark else (210, 215, 220)
    point_color = (92, 92, 92) if dark else (160, 165, 170)
    rows = sorted(set(target.row for target in targets))
    cols = sorted(set(target.col for target in targets))
    for row in rows:
        row_targets = [target for target in targets if target.row == row]
        if len(row_targets) > 1:
            cv2.line(image, (row_targets[0].x, row_targets[0].y), (row_targets[-1].x, row_targets[-1].y), color, 1, cv2.LINE_AA)
    for col in cols:
        col_targets = [target for target in targets if target.col == col]
        if len(col_targets) > 1:
            cv2.line(image, (col_targets[0].x, col_targets[0].y), (col_targets[-1].x, col_targets[-1].y), color, 1, cv2.LINE_AA)
    for target in targets:
        cv2.circle(image, (target.x, target.y), 7, point_color, -1, cv2.LINE_AA)
