#!/usr/bin/env python3
"""Create Figure 3: absolute and percentage rural-income change, 2019–2023."""

from __future__ import annotations

import json
import math
from pathlib import Path

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
    RELEASE,
    ROOT,
    TARGET_CRS,
    TEXT_RGB,
    font,
    project_parts,
    read_shapefile_parts,
)


FIGURE_DIR = ROOT / "docs" / "figures"
CHANGE_DIR = RELEASE / "data" / "change_maps"

ABS_PATH = CHANGE_DIR / "china_rural_income_change_1km_2019_2023_absolute_2019yuan_strict_unbounded.tif"
PCT_PATH = CHANGE_DIR / "china_rural_income_change_1km_2019_2023_percent_strict_unbounded.tif"

ABS_BOUNDS = [float("-inf"), -3000, -1000, 0, 2000, 5000, 8000, float("inf")]
ABS_LABELS = [
    "≤ −3,000",
    "−3,000–−1,000",
    "−1,000–0",
    "0–2,000",
    "2,000–5,000",
    "5,000–8,000",
    "> 8,000",
]
ABS_RGB = [
    (33, 102, 172),
    (67, 147, 195),
    (146, 197, 222),
    (254, 224, 182),
    (253, 174, 107),
    (239, 101, 72),
    (178, 24, 43),
]

PCT_BOUNDS = [float("-inf"), -20, 0, 10, 20, 30, 50, float("inf")]
PCT_LABELS = [
    "≤ −20%",
    "−20–0%",
    "0–10%",
    "10–20%",
    "20–30%",
    "30–50%",
    "> 50%",
]
PCT_RGB = [
    (33, 102, 172),
    (103, 169, 207),
    (255, 255, 178),
    (254, 217, 118),
    (254, 178, 76),
    (252, 78, 42),
    (177, 0, 38),
]

MAP_W = 1760
MAP_H = 1000
PANEL_W = 1900
PANEL_H = 1280
LEFT_MARGIN = 55
RIGHT_MARGIN = 55
TOP_MARGIN = 45
BOTTOM_MARGIN = 55
COL_GAP = 55
CANVAS_W = LEFT_MARGIN + 2 * PANEL_W + COL_GAP + RIGHT_MARGIN
CANVAS_H = TOP_MARGIN + PANEL_H + BOTTOM_MARGIN

FONT_PANEL = font(45, bold=True)
FONT_TITLE = font(46, bold=True)
FONT_LEGEND = font(29)
FONT_NOTE = font(25)


def target_grid():
    bounds = transform_bounds(
        "EPSG:4326",
        TARGET_CRS,
        *MAIN_LONLAT_BOUNDS,
        densify_pts=41,
    )
    grid_transform = from_bounds(*bounds, width=MAP_W, height=MAP_H)
    return bounds, grid_transform


def reproject_change(path: Path, grid_transform) -> np.ndarray:
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
            resampling=Resampling.bilinear,
            num_threads=2,
        )
    return destination


def xy_to_map(point, bounds):
    left, bottom, right, top = bounds
    x, y = point
    px = int(round((x - left) / (right - left) * (MAP_W - 1)))
    py = int(round((top - y) / (top - bottom) * (MAP_H - 1)))
    return px, py


def draw_land(map_image, projected_parts, bounds):
    draw = ImageDraw.Draw(map_image)
    for part in projected_parts:
        pts = [xy_to_map(p, bounds) for p in part]
        if len(pts) >= 3:
            draw.polygon(pts, fill=NODATA_LAND_RGB)


def colourize(data, class_bounds, class_rgb):
    rgba = np.zeros((MAP_H, MAP_W, 4), dtype=np.uint8)
    valid = np.isfinite(data)
    indices = np.digitize(data[valid], class_bounds[1:-1], right=False)
    palette = np.asarray(class_rgb, dtype=np.uint8)
    rgba[valid, :3] = palette[indices]
    rgba[valid, 3] = 255
    return Image.fromarray(rgba, mode="RGBA")


def draw_boundaries(map_image, projected_parts, bounds):
    draw = ImageDraw.Draw(map_image)
    for part in projected_parts:
        pts = [xy_to_map(p, bounds) for p in part]
        if len(pts) >= 2:
            draw.line(pts, fill=PROVINCE_LINE_RGB, width=2, joint="curve")


def lonlat_to_local(point, size, bounds):
    width, height = size
    left, bottom, right, top = bounds
    lon, lat = point
    x = int(round((lon - left) / (right - left) * (width - 1)))
    y = int(round((top - lat) / (top - bottom) * (height - 1)))
    return x, y


def draw_horizontal_legend(draw, y, labels, colours, unit_title, panel_x=45):
    draw.text((panel_x, y), unit_title, font=FONT_NOTE, fill=MUTED_RGB)
    start_y = y + 48
    swatch_w = 172
    swatch_h = 38
    gap = 14
    legend_labels = list(labels) + ["No valid change"]
    legend_colours = list(colours) + [NODATA_LAND_RGB]
    for index, (label, colour) in enumerate(zip(legend_labels, legend_colours)):
        x = panel_x + index * (swatch_w + gap)
        draw.rectangle((x, start_y, x + swatch_w, start_y + swatch_h), fill=colour, outline=(150, 150, 150), width=1)
        bbox = draw.textbbox((0, 0), label, font=FONT_LEGEND)
        label_w = bbox[2] - bbox[0]
        draw.text((x + (swatch_w - label_w) / 2, start_y + 49), label, font=FONT_LEGEND, fill=TEXT_RGB)


def full_resolution_summary(path: Path):
    with rasterio.open(path) as src:
        values = src.read(1, masked=True).compressed().astype("float64")
    quantile_levels = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    quantiles = np.percentile(values, quantile_levels)
    return {
        "valid_pixels": int(values.size),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "quantiles": {str(k): float(v) for k, v in zip(quantile_levels, quantiles)},
        "negative_share_percent": float((values < 0).mean() * 100.0),
        "positive_share_percent": float((values > 0).mean() * 100.0),
    }


def make_panel(
    label,
    title,
    path,
    class_bounds,
    class_labels,
    class_rgb,
    unit_title,
    explanatory_note,
    bounds,
    grid_transform,
    projected_provinces,
):
    panel = Image.new("RGB", (PANEL_W, PANEL_H), OUTSIDE_RGB)
    map_image = Image.new("RGB", (MAP_W, MAP_H), OUTSIDE_RGB)
    draw_land(map_image, projected_provinces, bounds)
    data = reproject_change(path, grid_transform)
    map_image = Image.alpha_composite(
        map_image.convert("RGBA"), colourize(data, class_bounds, class_rgb)
    ).convert("RGB")
    draw_boundaries(map_image, projected_provinces, bounds)
    panel.paste(map_image, ((PANEL_W - MAP_W) // 2, 0))

    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, PANEL_W - 1, PANEL_H - 1), outline=PANEL_BORDER_RGB, width=2)
    draw.text((30, 24), label, font=FONT_PANEL, fill=TEXT_RGB)
    title_box = draw.textbbox((0, 0), title, font=FONT_TITLE)
    title_w = title_box[2] - title_box[0]
    draw.text(((PANEL_W - title_w) / 2, 23), title, font=FONT_TITLE, fill=TEXT_RGB)

    legend_y = 1015
    draw_horizontal_legend(draw, legend_y, class_labels, class_rgb, unit_title, panel_x=55)
    draw.text((55, 1198), explanatory_note, font=FONT_NOTE, fill=MUTED_RGB)

    draw.text((MAP_W - 420, 952), "Study area: mainland China", font=FONT_NOTE, fill=MUTED_RGB)
    return panel


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    missing = [str(p) for p in [ABS_PATH, PCT_PATH, COUNTRY_SHP, PROVINCE_SHP] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing Figure 3 input(s):\n" + "\n".join(missing))

    province_parts_lonlat = read_shapefile_parts(PROVINCE_SHP)
    projected_provinces = project_parts(province_parts_lonlat)
    bounds, grid_transform = target_grid()

    panel_abs = make_panel(
        "(a)",
        "Absolute change, 2019–2023",
        ABS_PATH,
        ABS_BOUNDS,
        ABS_LABELS,
        ABS_RGB,
        "Change in 2019 constant yuan per person per year",
        "Absolute change = 2023 estimate − 2019 estimate.",
        bounds,
        grid_transform,
        projected_provinces,
    )
    panel_pct = make_panel(
        "(b)",
        "Percentage change, 2019–2023",
        PCT_PATH,
        PCT_BOUNDS,
        PCT_LABELS,
        PCT_RGB,
        "Percentage change relative to the 2019 estimate",
        "Relative change may be large where the 2019 baseline is low.",
        bounds,
        grid_transform,
        projected_provinces,
    )

    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), OUTSIDE_RGB)
    canvas.paste(panel_abs, (LEFT_MARGIN, TOP_MARGIN))
    canvas.paste(panel_pct, (LEFT_MARGIN + PANEL_W + COL_GAP, TOP_MARGIN))

    png_path = FIGURE_DIR / "figure3_income_change_2019_2023.png"
    tif_path = FIGURE_DIR / "figure3_income_change_2019_2023.tif"
    qa_path = FIGURE_DIR / "figure3_income_change_2019_2023_qa.json"
    canvas.save(png_path, format="PNG", dpi=(600, 600), optimize=True)
    canvas.save(tif_path, format="TIFF", compression="tiff_lzw", dpi=(600, 600))

    qa = {
        "figure": "Figure 3 rural-income change, 2019–2023",
        "canonical_product": "strict mask, unbounded, province calibrated",
        "projection": TARGET_CRS,
        "main_lonlat_bounds": MAIN_LONLAT_BOUNDS,
        "canvas_pixels": [CANVAS_W, CANVAS_H],
        "absolute": {
            "input": str(ABS_PATH.relative_to(ROOT)).replace("\\", "/"),
            "class_bounds": ["-inf", -3000, -1000, 0, 2000, 5000, 8000, "inf"],
            "class_labels": ABS_LABELS,
            "class_rgb": ABS_RGB,
            "full_resolution_summary": full_resolution_summary(ABS_PATH),
        },
        "percentage": {
            "input": str(PCT_PATH.relative_to(ROOT)).replace("\\", "/"),
            "class_bounds": ["-inf", -20, 0, 10, 20, 30, 50, "inf"],
            "class_labels": PCT_LABELS,
            "class_rgb": PCT_RGB,
            "full_resolution_summary": full_resolution_summary(PCT_PATH),
        },
        "nodata_land_rgb": NODATA_LAND_RGB,
        "display_boundary_sources": [
            str(COUNTRY_SHP.relative_to(ROOT)).replace("\\", "/"),
            str(PROVINCE_SHP.relative_to(ROOT)).replace("\\", "/"),
        ],
        "display_boundary_year": 2019,
        "display_boundary_citation": BOUNDARY_CITATION,
        "display_boundary_doi": BOUNDARY_DOI,
        "display_boundary_license": BOUNDARY_LICENSE,
        "study_area_note": "Released data product covers mainland China; Hong Kong, Macao and Taiwan are outside product coverage.",
        "outputs": [
            str(png_path.relative_to(ROOT)).replace("\\", "/"),
            str(tif_path.relative_to(ROOT)).replace("\\", "/"),
        ],
    }
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Created {}".format(png_path))
    print("Created {}".format(tif_path))
    print("Created {}".format(qa_path))


if __name__ == "__main__":
    main()
