#!/usr/bin/env python3
"""Create Figure 5: quality-risk and multi-seed ensemble uncertainty.

The upper panels map the 2023 quality-risk class and ensemble coefficient of
variation (CV). The lower panels summarize annual risk-class shares and annual
ensemble CV statistics. These layers are relative interpretation aids, not
pixel-level validation errors or statistical confidence intervals.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / "stage6_pydeps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds

from create_figure2_annual_income import (
    BOUNDARY_CITATION,
    BOUNDARY_DOI,
    BOUNDARY_LICENSE,
    COUNTRY_SHP,
    MAIN_LONLAT_BOUNDS,
    MUTED_RGB,
    NODATA_LAND_RGB,
    OUTSIDE_RGB,
    PANEL_BORDER_RGB,
    PROVINCE_LINE_RGB,
    PROVINCE_SHP,
    TARGET_CRS,
    TEXT_RGB,
    font,
    project_parts,
    read_shapefile_parts,
)


RELEASE = (
    ROOT
    / "outputs"
    / "public_release"
    / "china_rural_income_1km_2019_2023_v1.0-draft"
)
QUALITY_DIR = RELEASE / "data" / "uncertainty_proxy_layers"
ENSEMBLE_DIR = RELEASE / "data" / "ensemble_uncertainty_layers"
TABLE_DIR = RELEASE / "tables"
FIGURE_DIR = ROOT / "docs" / "figures"

QUALITY_CLASS_2023 = QUALITY_DIR / "uncertainty_proxy_class_2023_strict.tif"
CV_2023 = ENSEMBLE_DIR / "china_rural_income_1km_2023_ensemble_cv_percent_strict_unbounded.tif"
QUALITY_SUMMARY = TABLE_DIR / "uncertainty_proxy_summary_2019_2023.csv"
ENSEMBLE_SUMMARY = TABLE_DIR / "ensemble_uncertainty_summary_2019_2023.csv"

PNG_PATH = FIGURE_DIR / "figure5_quality_uncertainty.png"
TIFF_PATH = FIGURE_DIR / "figure5_quality_uncertainty.tif"
QA_PATH = FIGURE_DIR / "figure5_quality_uncertainty_qa.json"

CANVAS_W = 4000
CANVAS_H = 3100
MARGIN_X = 80
TOP = 100
COL_GAP = 70
ROW_GAP = 80
PANEL_W = (CANVAS_W - 2 * MARGIN_X - COL_GAP) // 2
MAP_PANEL_H = 1370
CHART_PANEL_H = 1370
MAP_W = 1780
MAP_H = 1000

QUALITY_LABELS = ["Low (<25)", "Moderate (25–<50)", "High (≥50)"]
QUALITY_RGB = [(102, 189, 99), (253, 174, 97), (215, 48, 39)]

CV_BOUNDS = [float("-inf"), 5, 10, 15, 20, 30, float("inf")]
CV_LABELS = ["≤5%", "5–10%", "10–15%", "15–20%", "20–30%", ">30%"]
CV_RGB = [
    (239, 243, 255),
    (198, 219, 239),
    (158, 202, 225),
    (107, 174, 214),
    (49, 130, 189),
    (8, 81, 156),
]

FONT_PANEL = font(47, bold=True)
FONT_TITLE = font(43, bold=True)
FONT_LEGEND = font(28)
FONT_NOTE = font(25)
FONT_AXIS = font(30)
FONT_AXIS_BOLD = font(30, bold=True)
FONT_VALUE = font(27, bold=True)
FONT_FOOTNOTE = font(27)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def target_grid():
    bounds = transform_bounds("EPSG:4326", TARGET_CRS, *MAIN_LONLAT_BOUNDS, densify_pts=41)
    return bounds, from_bounds(*bounds, width=MAP_W, height=MAP_H)


def reproject_layer(path: Path, grid_transform, resampling: Resampling) -> np.ndarray:
    destination = np.full((MAP_H, MAP_W), np.nan, dtype="float32")
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=grid_transform,
            dst_crs=TARGET_CRS,
            dst_nodata=np.nan,
            resampling=resampling,
            num_threads=2,
        )
    return destination


def xy_to_map(point, bounds):
    left, bottom, right, top = bounds
    x, y = point
    return (
        int(round((x - left) / (right - left) * (MAP_W - 1))),
        int(round((top - y) / (top - bottom) * (MAP_H - 1))),
    )


def draw_land(image: Image.Image, projected_parts, bounds) -> None:
    draw = ImageDraw.Draw(image)
    for part in projected_parts:
        points = [xy_to_map(point, bounds) for point in part]
        if len(points) >= 3:
            draw.polygon(points, fill=NODATA_LAND_RGB)


def draw_boundaries(image: Image.Image, projected_parts, bounds) -> None:
    draw = ImageDraw.Draw(image)
    for part in projected_parts:
        points = [xy_to_map(point, bounds) for point in part]
        if len(points) >= 2:
            draw.line(points, fill=PROVINCE_LINE_RGB, width=2, joint="curve")


def colourize_quality(data: np.ndarray) -> Image.Image:
    rgba = np.zeros((MAP_H, MAP_W, 4), dtype=np.uint8)
    for value, colour in zip([1, 2, 3], QUALITY_RGB):
        selected = np.isfinite(data) & (np.rint(data) == value)
        rgba[selected, :3] = colour
        rgba[selected, 3] = 255
    return Image.fromarray(rgba, mode="RGBA")


def colourize_cv(data: np.ndarray) -> Image.Image:
    rgba = np.zeros((MAP_H, MAP_W, 4), dtype=np.uint8)
    valid = np.isfinite(data)
    indices = np.digitize(data[valid], CV_BOUNDS[1:-1], right=False)
    rgba[valid, :3] = np.asarray(CV_RGB, dtype=np.uint8)[indices]
    rgba[valid, 3] = 255
    return Image.fromarray(rgba, mode="RGBA")


def lonlat_to_local(point, size, bounds):
    width, height = size
    left, bottom, right, top = bounds
    lon, lat = point
    return (
        int(round((lon - left) / (right - left) * (width - 1))),
        int(round((top - lat) / (top - bottom) * (height - 1))),
    )


def draw_map_legend(draw, labels, colours, y: int, title: str) -> None:
    draw.text((70, y), title, font=FONT_NOTE, fill=MUTED_RGB)
    y0 = y + 46
    available = PANEL_W - 140
    item_w = available / (len(labels) + 1)
    all_labels = list(labels) + ["No valid rural estimate"]
    all_colours = list(colours) + [NODATA_LAND_RGB]
    for idx, (label, colour) in enumerate(zip(all_labels, all_colours)):
        x0 = 70 + idx * item_w
        x1 = x0 + item_w - 12
        draw.rectangle((x0, y0, x1, y0 + 35), fill=colour, outline=(145, 150, 155), width=1)
        box = draw.textbbox((0, 0), label, font=FONT_LEGEND)
        label_w = box[2] - box[0]
        draw.text((x0 + (x1 - x0 - label_w) / 2, y0 + 44), label, font=FONT_LEGEND, fill=TEXT_RGB)


def make_map_panel(
    label: str,
    title: str,
    path: Path,
    kind: str,
    bounds,
    grid_transform,
    projected_provinces,
) -> tuple[Image.Image, dict[str, object]]:
    panel = Image.new("RGB", (PANEL_W, MAP_PANEL_H), OUTSIDE_RGB)
    map_image = Image.new("RGB", (MAP_W, MAP_H), OUTSIDE_RGB)
    draw_land(map_image, projected_provinces, bounds)
    resampling = Resampling.nearest if kind == "quality" else Resampling.bilinear
    data = reproject_layer(path, grid_transform, resampling)
    overlay = colourize_quality(data) if kind == "quality" else colourize_cv(data)
    map_image = Image.alpha_composite(map_image.convert("RGBA"), overlay).convert("RGB")
    draw_boundaries(map_image, projected_provinces, bounds)
    panel.paste(map_image, ((PANEL_W - MAP_W) // 2, 0))

    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, PANEL_W - 1, MAP_PANEL_H - 1), outline=PANEL_BORDER_RGB, width=3)
    draw.text((32, 22), label, font=FONT_PANEL, fill=TEXT_RGB)
    title_box = draw.textbbox((0, 0), title, font=FONT_TITLE)
    draw.text(((PANEL_W - (title_box[2] - title_box[0])) / 2, 26), title, font=FONT_TITLE, fill=TEXT_RGB)

    draw.text((MAP_W - 405, 952), "Study area: mainland China", font=FONT_NOTE, fill=MUTED_RGB)

    if kind == "quality":
        draw_map_legend(draw, QUALITY_LABELS, QUALITY_RGB, 1010, "Quality-risk class (higher = greater caution)")
        note = "Composite of training-province support, calibration magnitude, and bounded/unbounded sensitivity."
    else:
        draw_map_legend(draw, CV_LABELS, CV_RGB, 1010, "Ensemble coefficient of variation (%)")
        note = "CV summarizes spread across five calibrated model members with different random seeds."
    draw.text((70, 1318), note, font=FONT_NOTE, fill=MUTED_RGB)

    finite = data[np.isfinite(data)]
    return panel, {
        "input": str(path.relative_to(ROOT)).replace("\\", "/"),
        "reprojected_valid_display_pixels": int(finite.size),
        "display_min": float(finite.min()) if finite.size else math.nan,
        "display_max": float(finite.max()) if finite.size else math.nan,
    }


def panel_frame(draw: ImageDraw.ImageDraw, left: int, top: int, label: str, title: str) -> None:
    draw.rounded_rectangle(
        (left, top, left + PANEL_W, top + CHART_PANEL_H),
        radius=18,
        fill=OUTSIDE_RGB,
        outline=PANEL_BORDER_RGB,
        width=3,
    )
    draw.text((left + 32, top + 22), label, font=FONT_PANEL, fill=TEXT_RGB)
    draw.text((left + 112, top + 29), title, font=FONT_TITLE, fill=TEXT_RGB)


def centered(draw, xy, text: str, face, fill=TEXT_RGB) -> None:
    box = draw.textbbox((0, 0), text, font=face)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2), text, font=face, fill=fill)


def draw_risk_share_chart(draw: ImageDraw.ImageDraw, left: int, top: int, rows: list[dict[str, str]]) -> None:
    plot_left = left + 185
    plot_right = left + PANEL_W - 75
    plot_top = top + 205
    plot_bottom = top + CHART_PANEL_H - 155
    plot_w = plot_right - plot_left
    for tick in [0, 20, 40, 60, 80, 100]:
        x = plot_left + plot_w * tick / 100
        draw.line((x, plot_top, x, plot_bottom), fill=(220, 225, 230), width=2)
        centered(draw, (x, plot_bottom + 42), str(tick), FONT_AXIS, MUTED_RGB)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=TEXT_RGB, width=3)

    row_h = (plot_bottom - plot_top) / len(rows)
    bar_h = 100
    for idx, row in enumerate(rows):
        cy = plot_top + row_h * (idx + 0.5)
        year = row["year"]
        year_box = draw.textbbox((0, 0), year, font=FONT_AXIS_BOLD)
        draw.text((plot_left - (year_box[2] - year_box[0]) - 28, cy - 18), year, font=FONT_AXIS_BOLD, fill=TEXT_RGB)
        cumulative = 0.0
        for key, colour in zip(["low_risk_pct", "moderate_risk_pct", "high_risk_pct"], QUALITY_RGB):
            value = float(row[key])
            x0 = plot_left + plot_w * cumulative / 100
            cumulative += value
            x1 = plot_left + plot_w * cumulative / 100
            draw.rectangle((x0, cy - bar_h / 2, x1, cy + bar_h / 2), fill=colour)
            if value >= 8:
                label = f"{value:.1f}%"
                centered(draw, ((x0 + x1) / 2, cy), label, FONT_VALUE, TEXT_RGB)
    centered(draw, ((plot_left + plot_right) / 2, plot_bottom + 98), "Share of valid rural pixels (%)", FONT_AXIS_BOLD)

    legend_x = left + 620
    for idx, (label, colour) in enumerate(zip(["Low", "Moderate", "High"], QUALITY_RGB)):
        x = legend_x + idx * 260
        draw.rectangle((x, top + 112, x + 36, top + 142), fill=colour)
        draw.text((x + 48, top + 108), label, font=FONT_NOTE, fill=TEXT_RGB)


def draw_cv_chart(draw: ImageDraw.ImageDraw, left: int, top: int, rows: list[dict[str, str]]) -> None:
    plot_left = left + 180
    plot_right = left + PANEL_W - 85
    plot_top = top + 205
    plot_bottom = top + CHART_PANEL_H - 155
    y_min, y_max = 5.0, 17.0
    blue = (49, 130, 189)
    orange = (230, 85, 13)

    def y_pos(value: float) -> float:
        return plot_bottom - (value - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

    for tick in [6, 8, 10, 12, 14, 16]:
        y = y_pos(tick)
        draw.line((plot_left, y, plot_right, y), fill=(220, 225, 230), width=2)
        centered(draw, (plot_left - 45, y), str(tick), FONT_AXIS, MUTED_RGB)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=TEXT_RGB, width=3)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=TEXT_RGB, width=3)

    years = [int(row["year"]) for row in rows]
    series = [
        ("cv_median_pct", "Median CV", blue, -38),
        ("cv_p95_pct", "95th percentile CV", orange, 20),
    ]
    for key, _, colour, label_offset in series:
        points = []
        for idx, row in enumerate(rows):
            x = plot_left + (plot_right - plot_left) * idx / (len(rows) - 1)
            points.append((x, y_pos(float(row[key]))))
        draw.line(points, fill=colour, width=9, joint="curve")
        for (x, y), row in zip(points, rows):
            draw.ellipse((x - 13, y - 13, x + 13, y + 13), fill=OUTSIDE_RGB, outline=colour, width=7)
            value = float(row[key])
            centered(draw, (x, y + label_offset), f"{value:.2f}", FONT_VALUE, TEXT_RGB)

    for idx, year in enumerate(years):
        x = plot_left + (plot_right - plot_left) * idx / (len(years) - 1)
        centered(draw, (x, plot_bottom + 43), str(year), FONT_AXIS, TEXT_RGB)
    draw.text((left + 45, plot_top - 58), "Ensemble CV (%)", font=FONT_AXIS_BOLD, fill=TEXT_RGB)
    centered(draw, ((plot_left + plot_right) / 2, plot_bottom + 98), "Year", FONT_AXIS_BOLD)

    legend_x = left + 560
    for idx, (_, label, colour, _) in enumerate(series):
        x = legend_x + idx * 420
        draw.line((x, top + 127, x + 55, top + 127), fill=colour, width=9)
        draw.text((x + 70, top + 109), label, font=FONT_NOTE, fill=TEXT_RGB)


def main() -> None:
    inputs = [QUALITY_CLASS_2023, CV_2023, QUALITY_SUMMARY, ENSEMBLE_SUMMARY, COUNTRY_SHP, PROVINCE_SHP]
    missing = [str(path) for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Figure 5 input(s):\n" + "\n".join(missing))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    quality_rows = read_csv_rows(QUALITY_SUMMARY)
    ensemble_rows = read_csv_rows(ENSEMBLE_SUMMARY)
    province_parts = read_shapefile_parts(PROVINCE_SHP)
    projected_provinces = project_parts(province_parts)
    bounds, grid_transform = target_grid()

    panel_a, map_a_qa = make_map_panel(
        "a", "Quality-risk class, 2023", QUALITY_CLASS_2023, "quality", bounds, grid_transform,
        projected_provinces,
    )
    panel_b, map_b_qa = make_map_panel(
        "b", "Ensemble coefficient of variation, 2023", CV_2023, "cv", bounds, grid_transform,
        projected_provinces,
    )

    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), OUTSIDE_RGB)
    left_a = MARGIN_X
    left_b = MARGIN_X + PANEL_W + COL_GAP
    top_maps = TOP
    top_charts = TOP + MAP_PANEL_H + ROW_GAP
    canvas.paste(panel_a, (left_a, top_maps))
    canvas.paste(panel_b, (left_b, top_maps))
    draw = ImageDraw.Draw(canvas)
    panel_frame(draw, left_a, top_charts, "c", "Annual quality-risk composition")
    panel_frame(draw, left_b, top_charts, "d", "Annual ensemble uncertainty")
    draw_risk_share_chart(draw, left_a, top_charts, quality_rows)
    draw_cv_chart(draw, left_b, top_charts, ensemble_rows)

    foot_y = top_charts + CHART_PANEL_H + 38
    centered(draw, (CANVAS_W / 2, foot_y), "Quality-risk is an interpretation proxy; ensemble CV is model/sampling spread after province calibration.", FONT_FOOTNOTE)
    centered(draw, (CANVAS_W / 2, foot_y + 45), "Neither quantity is a pixel-level validation error or a full statistical confidence interval.", FONT_FOOTNOTE, (184, 55, 55))

    canvas.save(PNG_PATH, format="PNG", dpi=(600, 600), optimize=True)
    canvas.save(TIFF_PATH, format="TIFF", compression="tiff_lzw", dpi=(600, 600))

    qa = {
        "figure": "Figure 5. Quality-risk and ensemble uncertainty",
        "projection": TARGET_CRS,
        "main_lonlat_bounds": MAIN_LONLAT_BOUNDS,
        "canvas_pixels": [CANVAS_W, CANVAS_H],
        "dpi": 600,
        "quality_map": {
            **map_a_qa,
            "classes": [
                {"value": value, "label": label, "rgb": colour}
                for value, label, colour in zip([1, 2, 3], QUALITY_LABELS, QUALITY_RGB)
            ],
        },
        "ensemble_cv_map": {
            **map_b_qa,
            "class_bounds": ["-inf", 5, 10, 15, 20, 30, "inf"],
            "class_labels": CV_LABELS,
            "class_rgb": CV_RGB,
        },
        "summary_inputs": [
            str(QUALITY_SUMMARY.relative_to(ROOT)).replace("\\", "/"),
            str(ENSEMBLE_SUMMARY.relative_to(ROOT)).replace("\\", "/"),
        ],
        "summary_rows": {"quality": len(quality_rows), "ensemble": len(ensemble_rows)},
        "interpretation": {
            "quality_risk": "Composite interpretation proxy, not model standard deviation.",
            "ensemble_cv": "Five-member model/sampling spread after province calibration, not pixel-level validation error.",
            "confidence_interval": False,
        },
        "display_boundary_sources": [
            str(COUNTRY_SHP.relative_to(ROOT)).replace("\\", "/"),
            str(PROVINCE_SHP.relative_to(ROOT)).replace("\\", "/"),
        ],
        "display_boundary_year": 2019,
        "display_boundary_citation": BOUNDARY_CITATION,
        "display_boundary_doi": BOUNDARY_DOI,
        "display_boundary_license": BOUNDARY_LICENSE,
        "study_area_note": "Released data product covers mainland China; Hong Kong, Macao and Taiwan are outside product coverage.",
        "outputs": [str(PNG_PATH.relative_to(ROOT)).replace("\\", "/"), str(TIFF_PATH.relative_to(ROOT)).replace("\\", "/")],
    }
    QA_PATH.write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Created {PNG_PATH}")
    print(f"Created {TIFF_PATH}")
    print(f"Created {QA_PATH}")


if __name__ == "__main__":
    main()
