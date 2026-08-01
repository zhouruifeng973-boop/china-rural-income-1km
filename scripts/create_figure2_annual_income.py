#!/usr/bin/env python3
"""Create the Scientific Data Figure 2 annual rural-income small multiples.

The script reads only the five canonical strict-mask, unbounded GeoTIFFs from
the public-release package. It reprojects them to a shared China Albers grid,
applies fixed classes and colours, and exports a publication-sized PNG/TIFF.

Boundary files are used only for cartographic context in the derived figure.
They are not copied into the output directory or public-release package.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform, transform_bounds


ROOT = Path(__file__).resolve().parents[1]
RELEASE = (
    ROOT
    / "outputs"
    / "public_release"
    / "china_rural_income_1km_2019_2023_v1.0-draft"
)
INCOME_DIR = RELEASE / "data" / "income_maps"
FIGURE_DIR = ROOT / "docs" / "figures"

YEARS = [2019, 2020, 2021, 2022, 2023]
INPUT_TEMPLATE = "china_rural_income_1km_{year}_2019yuan_strict_unbounded.tif"

# Fixed classes used for every panel. The upper class is open ended.
CLASS_BOUNDS = [float("-inf"), 8000, 12000, 16000, 20000, 25000, 35000, float("inf")]
CLASS_LABELS = [
    "≤ 8,000",
    "8,000–12,000",
    "12,000–16,000",
    "16,000–20,000",
    "20,000–25,000",
    "25,000–35,000",
    "> 35,000",
]

# ColorBrewer YlOrRd seven-class palette, matching the established ArcMap logic.
CLASS_RGB = [
    (255, 255, 178),
    (254, 217, 118),
    (254, 178, 76),
    (253, 141, 60),
    (252, 78, 42),
    (227, 26, 28),
    (177, 0, 38),
]

NODATA_LAND_RGB = (235, 237, 239)
OUTSIDE_RGB = (255, 255, 255)
PROVINCE_LINE_RGB = (100, 108, 114)
COUNTRY_LINE_RGB = (55, 65, 72)
TEXT_RGB = (31, 46, 58)
MUTED_RGB = (88, 102, 113)
PANEL_BORDER_RGB = (205, 212, 218)

# Equal-area projection suitable for nationwide comparison.
TARGET_CRS = (
    "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 "
    "+datum=WGS84 +units=m +no_defs"
)

# Main-map geographic frame. The South China Sea is shown separately in the
# legend panel so the mainland panels retain enough spatial detail.
MAIN_LONLAT_BOUNDS = (73.0, 17.0, 136.0, 54.5)

# Draft internal boundary source. Its redistribution/use permission must be
# checked before the final submitted figure is frozen.
PROVINCE_SHP = ROOT / "data" / "2020xianBJ" / "省.shp"
NINE_DASH_SHP = ROOT / "data" / "2020xianBJ" / "九段线.shp"

PANEL_W = 1400
PANEL_H = 850
LEFT_MARGIN = 70
RIGHT_MARGIN = 70
TOP_MARGIN = 45
BOTTOM_MARGIN = 75
COL_GAP = 45
ROW_GAP = 70
CANVAS_W = LEFT_MARGIN + 3 * PANEL_W + 2 * COL_GAP + RIGHT_MARGIN
CANVAS_H = TOP_MARGIN + 2 * PANEL_H + ROW_GAP + BOTTOM_MARGIN


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a stable Windows font, with a bundled fallback."""

    candidates = []
    if bold:
        candidates.extend(
            [
                Path("C:/Windows/Fonts/arialbd.ttf"),
                Path("C:/Windows/Fonts/calibrib.ttf"),
            ]
        )
    else:
        candidates.extend(
            [
                Path("C:/Windows/Fonts/arial.ttf"),
                Path("C:/Windows/Fonts/calibri.ttf"),
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


FONT_PANEL = font(44, bold=True)
FONT_YEAR = font(46, bold=True)
FONT_LEGEND_TITLE = font(39, bold=True)
FONT_LEGEND = font(33)
FONT_NOTE = font(27)
FONT_SMALL = font(24)


def read_shapefile_parts(path: Path) -> list[list[tuple[float, float]]]:
    """Read polygon/polyline parts from a .shp using only the standard library."""

    parts_out: list[list[tuple[float, float]]] = []
    with path.open("rb") as handle:
        header = handle.read(100)
        if len(header) != 100:
            raise ValueError("Invalid shapefile header: {}".format(path))
        while True:
            record_header = handle.read(8)
            if not record_header:
                break
            if len(record_header) != 8:
                raise ValueError("Truncated shapefile record header: {}".format(path))
            _, content_words = struct.unpack(">2i", record_header)
            content = handle.read(content_words * 2)
            if len(content) < 4:
                continue
            shape_type = struct.unpack_from("<i", content, 0)[0]
            if shape_type == 0:
                continue
            if shape_type not in {3, 5, 13, 15, 23, 25}:
                continue
            if len(content) < 44:
                continue
            num_parts, num_points = struct.unpack_from("<2i", content, 36)
            if num_parts <= 0 or num_points <= 0:
                continue
            part_offset = 44
            point_offset = part_offset + 4 * num_parts
            if len(content) < point_offset + 16 * num_points:
                continue
            starts = list(struct.unpack_from("<{}i".format(num_parts), content, part_offset))
            starts.append(num_points)
            coords_flat = struct.unpack_from("<{}d".format(2 * num_points), content, point_offset)
            points = list(zip(coords_flat[0::2], coords_flat[1::2]))
            for idx in range(num_parts):
                part = points[starts[idx] : starts[idx + 1]]
                if len(part) >= 2:
                    parts_out.append(part)
    return parts_out


def project_parts(
    parts: list[list[tuple[float, float]]],
) -> list[list[tuple[float, float]]]:
    projected: list[list[tuple[float, float]]] = []
    for part in parts:
        xs = [p[0] for p in part]
        ys = [p[1] for p in part]
        px, py = transform("EPSG:4326", TARGET_CRS, xs, ys)
        projected.append(list(zip(px, py)))
    return projected


def main_target_grid() -> tuple[tuple[float, float, float, float], rasterio.Affine]:
    bounds = transform_bounds(
        "EPSG:4326",
        TARGET_CRS,
        *MAIN_LONLAT_BOUNDS,
        densify_pts=41,
    )
    grid_transform = from_bounds(*bounds, width=PANEL_W, height=PANEL_H)
    return bounds, grid_transform


def reproject_income(path: Path, grid_transform: rasterio.Affine) -> np.ndarray:
    destination = np.full((PANEL_H, PANEL_W), np.nan, dtype="float32")
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


def xy_to_panel(
    point: tuple[float, float], bounds: tuple[float, float, float, float]
) -> tuple[int, int]:
    left, bottom, right, top = bounds
    x, y = point
    px = int(round((x - left) / (right - left) * (PANEL_W - 1)))
    py = int(round((top - y) / (top - bottom) * (PANEL_H - 1)))
    return px, py


def draw_projected_land(
    base: Image.Image,
    projected_parts: list[list[tuple[float, float]]],
    bounds: tuple[float, float, float, float],
) -> None:
    draw = ImageDraw.Draw(base)
    for part in projected_parts:
        pts = [xy_to_panel(point, bounds) for point in part]
        if len(pts) >= 3:
            draw.polygon(pts, fill=NODATA_LAND_RGB)


def colourize_income(data: np.ndarray) -> Image.Image:
    rgba = np.zeros((PANEL_H, PANEL_W, 4), dtype=np.uint8)
    valid = np.isfinite(data)
    # Internal thresholds separate the seven classes.
    indices = np.digitize(data[valid], CLASS_BOUNDS[1:-1], right=False)
    palette = np.asarray(CLASS_RGB, dtype=np.uint8)
    rgba[valid, :3] = palette[indices]
    rgba[valid, 3] = 255
    return Image.fromarray(rgba, mode="RGBA")


def draw_boundaries(
    panel: Image.Image,
    projected_parts: list[list[tuple[float, float]]],
    bounds: tuple[float, float, float, float],
) -> None:
    draw = ImageDraw.Draw(panel)
    for part in projected_parts:
        pts = [xy_to_panel(point, bounds) for point in part]
        if len(pts) >= 2:
            draw.line(pts, fill=PROVINCE_LINE_RGB, width=2, joint="curve")


def make_map_panel(
    year: int,
    label: str,
    input_path: Path,
    bounds: tuple[float, float, float, float],
    grid_transform: rasterio.Affine,
    projected_provinces: list[list[tuple[float, float]]],
) -> tuple[Image.Image, dict[str, float | int | str]]:
    data = reproject_income(input_path, grid_transform)
    panel = Image.new("RGB", (PANEL_W, PANEL_H), OUTSIDE_RGB)
    draw_projected_land(panel, projected_provinces, bounds)
    panel = Image.alpha_composite(panel.convert("RGBA"), colourize_income(data)).convert("RGB")
    draw_boundaries(panel, projected_provinces, bounds)

    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, PANEL_W - 1, PANEL_H - 1), outline=PANEL_BORDER_RGB, width=2)
    draw.text((26, 21), label, font=FONT_PANEL, fill=TEXT_RGB)
    year_text = str(year)
    year_box = draw.textbbox((0, 0), year_text, font=FONT_YEAR)
    year_width = year_box[2] - year_box[0]
    draw.text(((PANEL_W - year_width) / 2, 20), year_text, font=FONT_YEAR, fill=TEXT_RGB)

    finite = data[np.isfinite(data)]
    summary = {
        "year": year,
        "input": str(input_path.relative_to(ROOT)).replace("\\", "/"),
        "reprojected_valid_display_pixels": int(finite.size),
        "display_min": float(np.min(finite)) if finite.size else math.nan,
        "display_max": float(np.max(finite)) if finite.size else math.nan,
    }
    return panel, summary


def lonlat_to_inset(
    point: tuple[float, float],
    rect: tuple[int, int, int, int],
    bounds: tuple[float, float, float, float],
) -> tuple[int, int]:
    x0, y0, x1, y1 = rect
    left, bottom, right, top = bounds
    lon, lat = point
    x = int(round(x0 + (lon - left) / (right - left) * (x1 - x0)))
    y = int(round(y0 + (top - lat) / (top - bottom) * (y1 - y0)))
    return x, y


def make_legend_panel(
    province_parts_lonlat: list[list[tuple[float, float]]],
    nine_dash_parts_lonlat: list[list[tuple[float, float]]],
) -> Image.Image:
    panel = Image.new("RGB", (PANEL_W, PANEL_H), OUTSIDE_RGB)
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, PANEL_W - 1, PANEL_H - 1), outline=PANEL_BORDER_RGB, width=2)

    draw.text((65, 55), "Rural disposable income", font=FONT_LEGEND_TITLE, fill=TEXT_RGB)
    draw.text((65, 108), "2019 constant yuan per person per year", font=FONT_NOTE, fill=MUTED_RGB)

    swatch_x0 = 70
    swatch_x1 = 180
    label_x = 210
    start_y = 175
    step_y = 70
    for idx, (rgb, class_label) in enumerate(zip(CLASS_RGB, CLASS_LABELS)):
        y = start_y + idx * step_y
        draw.rectangle((swatch_x0, y, swatch_x1, y + 42), fill=rgb, outline=(150, 150, 150), width=1)
        draw.text((label_x, y + 1), class_label, font=FONT_LEGEND, fill=TEXT_RGB)

    nodata_y = start_y + len(CLASS_LABELS) * step_y + 8
    draw.rectangle(
        (swatch_x0, nodata_y, swatch_x1, nodata_y + 42),
        fill=NODATA_LAND_RGB,
        outline=(150, 150, 150),
        width=1,
    )
    draw.text((label_x, nodata_y + 1), "No valid rural estimate", font=FONT_LEGEND, fill=TEXT_RGB)

    inset_rect = (690, 175, 1320, 650)
    inset_bounds = (105.0, 3.0, 125.0, 25.0)
    inset_w = inset_rect[2] - inset_rect[0]
    inset_h = inset_rect[3] - inset_rect[1]
    inset = Image.new("RGB", (inset_w, inset_h), (250, 251, 252))
    inset_draw = ImageDraw.Draw(inset)
    local_rect = (0, 0, inset_w - 1, inset_h - 1)
    for part in province_parts_lonlat:
        pts = [lonlat_to_inset(p, local_rect, inset_bounds) for p in part]
        if len(pts) >= 2:
            inset_draw.line(pts, fill=COUNTRY_LINE_RGB, width=2, joint="curve")
    for part in nine_dash_parts_lonlat:
        pts = [lonlat_to_inset(p, local_rect, inset_bounds) for p in part]
        if len(pts) >= 2:
            inset_draw.line(pts, fill=COUNTRY_LINE_RGB, width=3, joint="curve")
    inset_draw.rectangle(local_rect, outline=PANEL_BORDER_RGB, width=2)
    panel.paste(inset, (inset_rect[0], inset_rect[1]))
    draw.text((700, 135), "South China Sea inset", font=FONT_NOTE, fill=TEXT_RGB)

    note_y = 695
    draw.text((65, note_y), "All panels use the strict rural mask and unbounded", font=FONT_NOTE, fill=MUTED_RGB)
    draw.text((65, note_y + 38), "province-calibrated canonical products.", font=FONT_NOTE, fill=MUTED_RGB)
    draw.text((65, note_y + 92), "Projection: China Albers equal-area conic", font=FONT_SMALL, fill=MUTED_RGB)
    return panel


def paste_panel(canvas: Image.Image, panel: Image.Image, col: int, row: int) -> None:
    x = LEFT_MARGIN + col * (PANEL_W + COL_GAP)
    y = TOP_MARGIN + row * (PANEL_H + ROW_GAP)
    canvas.paste(panel, (x, y))


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    input_paths = {year: INCOME_DIR / INPUT_TEMPLATE.format(year=year) for year in YEARS}
    missing = [str(path) for path in list(input_paths.values()) + [PROVINCE_SHP, NINE_DASH_SHP] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Figure 2 input(s):\n" + "\n".join(missing))

    province_parts_lonlat = read_shapefile_parts(PROVINCE_SHP)
    nine_dash_parts_lonlat = read_shapefile_parts(NINE_DASH_SHP)
    projected_provinces = project_parts(province_parts_lonlat)
    bounds, grid_transform = main_target_grid()

    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), OUTSIDE_RGB)
    summaries = []
    panel_positions = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)]
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    for year, label, (col, row) in zip(YEARS, panel_labels, panel_positions):
        panel, summary = make_map_panel(
            year,
            label,
            input_paths[year],
            bounds,
            grid_transform,
            projected_provinces,
        )
        paste_panel(canvas, panel, col, row)
        summaries.append(summary)

    legend_panel = make_legend_panel(province_parts_lonlat, nine_dash_parts_lonlat)
    paste_panel(canvas, legend_panel, 2, 1)

    png_path = FIGURE_DIR / "figure2_annual_income_2019_2023.png"
    tif_path = FIGURE_DIR / "figure2_annual_income_2019_2023.tif"
    qa_path = FIGURE_DIR / "figure2_annual_income_2019_2023_qa.json"
    canvas.save(png_path, format="PNG", dpi=(600, 600), optimize=True)
    canvas.save(tif_path, format="TIFF", compression="tiff_lzw", dpi=(600, 600))

    qa = {
        "figure": "Figure 2 annual rural-income maps",
        "canonical_product": "strict mask, unbounded, province calibrated",
        "years": YEARS,
        "projection": TARGET_CRS,
        "main_lonlat_bounds": MAIN_LONLAT_BOUNDS,
        "panel_pixels": [PANEL_W, PANEL_H],
        "canvas_pixels": [CANVAS_W, CANVAS_H],
        "class_bounds": ["-inf", 8000, 12000, 16000, 20000, 25000, 35000, "inf"],
        "class_labels": CLASS_LABELS,
        "class_rgb": CLASS_RGB,
        "nodata_land_rgb": NODATA_LAND_RGB,
        "boundary_source_internal_draft": str(PROVINCE_SHP.relative_to(ROOT)).replace("\\", "/"),
        "boundary_license_action": "Verify publication permission or replace before submission.",
        "panels": summaries,
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
