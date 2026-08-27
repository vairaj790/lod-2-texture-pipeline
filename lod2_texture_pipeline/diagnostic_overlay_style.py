# -*- coding: utf-8 -*-
"""Single visual language for SAM, projection, fit, and refit diagnostics.

Colors are defined once in RGB. OpenCV/BGR renderers must use ``style.bgr`` or
the helpers in this module, which prevents the same tuple from changing meaning
between PIL/RGB and OpenCV/BGR images.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np


RGBColor = Tuple[int, int, int]


def rgb_to_bgr(color: RGBColor) -> RGBColor:
    return int(color[2]), int(color[1]), int(color[0])


@dataclass(frozen=True)
class OverlayLineStyle:
    rgb: RGBColor
    width_px: int = 2
    dashed: bool = False
    dash_length_px: float = 7.0
    dash_gap_px: float = 5.0

    @property
    def bgr(self) -> RGBColor:
        return rgb_to_bgr(self.rgb)


@dataclass(frozen=True)
class OverlayLegendStyle:
    origin_xy: Tuple[int, int] = (8, 22)
    row_spacing_px: int = 21
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.43
    text_rgb: RGBColor = (20, 20, 20)
    outline_rgb: RGBColor = (255, 255, 255)
    text_width_px: int = 1
    outline_width_px: int = 3
    horizontal_margin_px: int = 8


# Semantic fills. These are evidence returned by SAM3, not model projections.
SAM_TARGET_BUILDING_FILL_RGB: RGBColor = (0, 190, 240)  # cyan
SAM_TARGET_ROOF_FILL_RGB: RGBColor = (255, 210, 0)  # yellow
SAM_PROMPTED_OCCLUDER_FILL_RGB: RGBColor = (255, 0, 170)  # pink
SAM_BACKGROUND_CONTEXT_FILL_RGB: RGBColor = (100, 155, 255)  # light blue
SAM_GENERIC_NON_TARGET_FILL_RGB: RGBColor = (255, 125, 0)  # orange
SEMANTIC_FILL_ALPHA: float = 0.28
OUTSIDE_SEARCH_DIM_FACTOR: float = 0.45


# Semantic image/model guide lines.
SAM_TARGET_SILHOUETTE_LINE = OverlayLineStyle((30, 150, 255), 2)  # blue
SAM_ROOF_GUIDE_LINE = OverlayLineStyle((255, 235, 0), 2)  # yellow
SAM_WALL_GUIDE_LINE = OverlayLineStyle((0, 230, 90), 2)  # green
SAM_BASE_GUIDE_LINE = OverlayLineStyle((185, 185, 185), 2)  # gray
SAM_SUPPRESSED_ROOF_GUIDE_RGB: RGBColor = (125, 125, 125)  # dotted gray
SAM_INFERRED_ROOF_BRIDGE_RGB: RGBColor = (255, 135, 0)  # dotted orange
SEMANTIC_GUIDE_LINES = {
    "silhouette": SAM_TARGET_SILHOUETTE_LINE,
    "roof": SAM_ROOF_GUIDE_LINE,
    "wall": SAM_WALL_GUIDE_LINE,
    "base": SAM_BASE_GUIDE_LINE,
}


# Model geometry. These meanings and patterns are invariant across candidate,
# selected-source, boundary-fit, and refit images.
RAW_MODEL_LINE = OverlayLineStyle((100, 120, 255), 1)  # solid violet
ACCEPTED_MODEL_LINE = OverlayLineStyle(
    (255, 0, 220), 1, dashed=True, dash_length_px=7.0, dash_gap_px=5.0
)  # dashed magenta
REJECTED_MODEL_LINE = OverlayLineStyle(
    (255, 165, 0), 2, dashed=True, dash_length_px=7.0, dash_gap_px=5.0
)  # dashed orange
OSM_OBSTRUCTION_LINE = OverlayLineStyle((255, 128, 0), 2)  # solid orange
WALL_ONLY_MODEL_LINE = OverlayLineStyle((40, 120, 255), 2)  # solid blue


LEGEND_STYLE = OverlayLegendStyle()


SEMANTIC_LEGEND_ROWS: Tuple[str, ...] = (
    "SAM3 fills: cyan=selected target-building evidence | yellow=selected target-roof evidence",
    "exclusions: pink=prompted/vegetation + safety border",
    "SAM3 guides: blue=target silhouette | yellow=roof | green=wall | gray=base",
)
BACKGROUND_AWARE_SEMANTIC_LEGEND_ROWS: Tuple[str, ...] = (
    "SAM3 fills: cyan=selected target-building evidence | yellow=selected target-roof evidence",
    "fit masks: pink=foreground/do-not-disturb | light blue=background/trusted context",
    "SAM3 guides: blue=target silhouette | yellow=roof | green=wall | gray=base",
)
STRICT_ROOF_AUDIT_LEGEND_ROW = (
    "roof audit: solid yellow=consumed | dotted gray=suppressed | "
    "dotted orange=inferred/not consumed"
)
SEARCH_LEGEND_ROW = "darkened=outside projection-local model search"
OSM_LEGEND_ROW = "solid orange=external OSM obstruction"


def model_projection_legend(*, fitted: bool, rejected: bool = False) -> str:
    if fitted:
        return (
            "model: solid violet=original/raw whole-model projection | "
            "dashed magenta=accepted fitted/refitted projection"
        )
    if rejected:
        return (
            "model: solid violet=original/raw whole-model projection | "
            "dashed orange=rejected fit candidate"
        )
    return (
        "model: solid violet=original/raw whole-model projection | "
        "no accepted fitted transform"
    )


def _color_for_space(color: RGBColor, color_space: str) -> RGBColor:
    normalized = str(color_space).strip().lower()
    if normalized == "rgb":
        return color
    if normalized == "bgr":
        return rgb_to_bgr(color)
    raise ValueError("color_space must be 'rgb' or 'bgr'.")


def draw_styled_line(
    image: np.ndarray,
    point0,
    point1,
    style: OverlayLineStyle,
    *,
    color_space: str,
) -> None:
    """Draw one solid or dashed anti-aliased segment using a shared style."""
    p0 = np.asarray(point0, dtype=np.float64).reshape(2)
    p1 = np.asarray(point1, dtype=np.float64).reshape(2)
    if not np.isfinite(p0).all() or not np.isfinite(p1).all():
        return
    color = _color_for_space(style.rgb, color_space)
    width = max(1, int(style.width_px))
    if not style.dashed:
        cv2.line(
            image,
            tuple(np.round(p0).astype(int)),
            tuple(np.round(p1).astype(int)),
            color,
            width,
            cv2.LINE_AA,
        )
        return

    vector = p1 - p0
    length = float(np.linalg.norm(vector))
    if length < 1.0e-6:
        return
    direction = vector / length
    period = max(
        float(style.dash_length_px) + float(style.dash_gap_px),
        1.0,
    )
    cursor = 0.0
    while cursor < length:
        end = min(cursor + max(float(style.dash_length_px), 1.0), length)
        dash0 = p0 + direction * cursor
        dash1 = p0 + direction * end
        cv2.line(
            image,
            tuple(np.round(dash0).astype(int)),
            tuple(np.round(dash1).astype(int)),
            color,
            width,
            cv2.LINE_AA,
        )
        cursor += period


def wrap_legend_rows(
    rows: Iterable[str],
    image_width_px: int,
    style: OverlayLegendStyle = LEGEND_STYLE,
) -> Tuple[str, ...]:
    maximum_width = max(
        80,
        int(image_width_px) - 2 * int(style.horizontal_margin_px),
    )
    wrapped = []
    for raw_row in rows:
        words = str(raw_row).split()
        current = ""
        for word in words:
            proposal = word if not current else f"{current} {word}"
            width = cv2.getTextSize(
                proposal,
                style.font_face,
                style.font_scale,
                style.text_width_px,
            )[0][0]
            if current and width > maximum_width:
                wrapped.append(current)
                current = word
            else:
                current = proposal
        if current:
            wrapped.append(current)
    return tuple(wrapped)


def draw_legend(
    image: np.ndarray,
    rows: Sequence[str],
    *,
    color_space: str,
    style: OverlayLegendStyle = LEGEND_STYLE,
) -> Tuple[str, ...]:
    """Draw every diagnostic legend with one font, outline, and spacing."""
    wrapped = wrap_legend_rows(rows, int(image.shape[1]), style)
    text_color = _color_for_space(style.text_rgb, color_space)
    outline_color = _color_for_space(style.outline_rgb, color_space)
    origin_x, origin_y = style.origin_xy
    for index, row in enumerate(wrapped):
        origin = (int(origin_x), int(origin_y + index * style.row_spacing_px))
        cv2.putText(
            image,
            row,
            origin,
            style.font_face,
            style.font_scale,
            outline_color,
            style.outline_width_px,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            row,
            origin,
            style.font_face,
            style.font_scale,
            text_color,
            style.text_width_px,
            cv2.LINE_AA,
        )
    return wrapped
