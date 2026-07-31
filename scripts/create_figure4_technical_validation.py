#!/usr/bin/env python3
"""Create Figure 4: public, aggregate technical-validation metrics.

The figure intentionally excludes county-level observations and residual records.
All displayed metrics were computed after rural-population-weighted aggregation of
pixel predictions back to counties.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RELEASE = (
    ROOT
    / "outputs"
    / "public_release"
    / "china_rural_income_1km_2019_2023_v1.0-draft"
)
TABLE_DIR = RELEASE / "tables"
FIGURE_DIR = ROOT / "docs" / "figures"

SUMMARY_CSV = TABLE_DIR / "validation_summary.csv"
PROVINCE_FOLD_CSV = TABLE_DIR / "validation_leave_province_2020_by_fold_before_after.csv"
YEAR_CSV = TABLE_DIR / "validation_leave_year_by_year_before_after.csv"

PNG_PATH = FIGURE_DIR / "figure4_technical_validation.png"
TIFF_PATH = FIGURE_DIR / "figure4_technical_validation.tif"
QA_PATH = FIGURE_DIR / "figure4_technical_validation_qa.json"

CANVAS_W = 4000
CANVAS_H = 3000
MARGIN_X = 115
TOP = 160
PANEL_GAP_X = 90
PANEL_GAP_Y = 95
PANEL_W = (CANVAS_W - 2 * MARGIN_X - PANEL_GAP_X) // 2
PANEL_H = 1190

WHITE = (255, 255, 255)
TEXT = (42, 48, 56)
MUTED = (105, 113, 122)
GRID = (218, 223, 229)
PANEL_BORDER = (196, 203, 211)
BLUE = (76, 120, 168)
ORANGE = (242, 142, 43)
TEAL = (114, 183, 178)
LIGHT_BLUE = (173, 202, 230)
NEGATIVE = (191, 70, 70)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    candidates = [Path("C:/Windows/Fonts") / name, Path("/usr/share/fonts/truetype/dejavu") / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


FONT_MAIN_TITLE = font(61, bold=True)
FONT_PANEL_LETTER = font(48, bold=True)
FONT_PANEL_TITLE = font(43, bold=True)
FONT_AXIS = font(31)
FONT_AXIS_BOLD = font(31, bold=True)
FONT_VALUE = font(29, bold=True)
FONT_NOTE = font(29)
FONT_SMALL = font(26)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def text_size(draw: ImageDraw.ImageDraw, value: str, face: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), value, font=face)
    return box[2] - box[0], box[3] - box[1]


def centered(draw: ImageDraw.ImageDraw, xy: tuple[float, float], value: str, face, fill=TEXT) -> None:
    width, height = text_size(draw, value, face)
    draw.text((xy[0] - width / 2, xy[1] - height / 2), value, font=face, fill=fill)


def panel_frame(draw: ImageDraw.ImageDraw, left: int, top: int, letter: str, title: str) -> None:
    draw.rounded_rectangle(
        (left, top, left + PANEL_W, top + PANEL_H),
        radius=18,
        fill=WHITE,
        outline=PANEL_BORDER,
        width=3,
    )
    draw.text((left + 34, top + 24), letter, font=FONT_PANEL_LETTER, fill=TEXT)
    draw.text((left + 112, top + 29), title, font=FONT_PANEL_TITLE, fill=TEXT)


def color_for(validation: str, constraint: str, mask: str) -> tuple[int, int, int]:
    if mask == "broad":
        return TEAL
    if "after province calibration" in constraint:
        return ORANGE
    if "before province calibration" in constraint or "no province calibration" in constraint:
        return BLUE
    return LIGHT_BLUE


def draw_horizontal_bars(
    draw: ImageDraw.ImageDraw,
    left: int,
    top: int,
    rows: list[dict[str, object]],
    key: str,
    x_max: float,
    ticks: list[float],
    tick_format,
    axis_label: str,
    lower_better: bool = False,
) -> None:
    plot_left = left + 610
    plot_right = left + PANEL_W - 95
    plot_top = top + 172
    plot_bottom = top + PANEL_H - 145
    plot_w = plot_right - plot_left

    for tick in ticks:
        x = plot_left + plot_w * tick / x_max
        draw.line((x, plot_top, x, plot_bottom), fill=GRID, width=2)
        centered(draw, (x, plot_bottom + 39), tick_format(tick), FONT_AXIS, MUTED)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=TEXT, width=3)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=TEXT, width=3)

    row_h = (plot_bottom - plot_top) / len(rows)
    bar_h = 72
    for index, row in enumerate(rows):
        cy = plot_top + row_h * (index + 0.5)
        value = float(row[key])
        bar_right = plot_left + plot_w * max(0.0, value) / x_max
        draw.rounded_rectangle(
            (plot_left, cy - bar_h / 2, bar_right, cy + bar_h / 2),
            radius=10,
            fill=row["color"],
        )
        label = str(row["short_label"])
        n_text = f"n = {int(row['n']):,}"
        label_w, label_h = text_size(draw, label, FONT_AXIS_BOLD)
        draw.text((plot_left - 30 - label_w, cy - label_h - 2), label, font=FONT_AXIS_BOLD, fill=TEXT)
        n_w, _ = text_size(draw, n_text, FONT_SMALL)
        draw.text((plot_left - 30 - n_w, cy + 6), n_text, font=FONT_SMALL, fill=MUTED)
        value_text = f"{value:.3f}" if key == "r2" else f"{value:.2f}%"
        value_x = min(bar_right + 18, plot_right - 105)
        draw.text((value_x, cy - 18), value_text, font=FONT_VALUE, fill=TEXT)

    centered(draw, ((plot_left + plot_right) / 2, plot_bottom + 96), axis_label, FONT_AXIS_BOLD, TEXT)
    if lower_better:
        draw.text((plot_right - 260, plot_top - 44), "Lower is better", font=FONT_SMALL, fill=MUTED)


def draw_grouped_fold_bars(draw: ImageDraw.ImageDraw, left: int, top: int, rows: list[dict[str, str]]) -> None:
    plot_left = left + 180
    plot_right = left + PANEL_W - 90
    plot_top = top + 190
    plot_bottom = top + PANEL_H - 155
    y_min, y_max = -0.30, 0.90
    plot_h = plot_bottom - plot_top

    def y_pos(value: float) -> float:
        return plot_bottom - (value - y_min) / (y_max - y_min) * plot_h

    for tick in [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8]:
        y = y_pos(tick)
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=2)
        tick_text = f"{tick:.1f}"
        width, height = text_size(draw, tick_text, FONT_AXIS)
        draw.text((plot_left - width - 22, y - height / 2), tick_text, font=FONT_AXIS, fill=MUTED)
    zero_y = y_pos(0.0)
    draw.line((plot_left, zero_y, plot_right, zero_y), fill=TEXT, width=3)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=TEXT, width=3)

    by_fold: dict[int, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_fold.setdefault(int(row["fold"]), {})[row["prediction"]] = row
    group_w = (plot_right - plot_left) / len(by_fold)
    bar_w = 72
    for idx, fold_number in enumerate(sorted(by_fold)):
        cx = plot_left + group_w * (idx + 0.5)
        for offset, prediction, color in [(-43, "before_calibration", BLUE), (43, "after_calibration", ORANGE)]:
            value = float(by_fold[fold_number][prediction]["r2"])
            x0 = cx + offset - bar_w / 2
            x1 = cx + offset + bar_w / 2
            y_value = y_pos(value)
            y0, y1 = sorted((zero_y, y_value))
            draw.rectangle((x0, y0, x1, y1), fill=color)
            value_text = f"{value:.2f}"
            value_w, value_h = text_size(draw, value_text, FONT_SMALL)
            text_y = y_value - value_h - 10 if value >= 0 else y_value + 8
            draw.text((cx + offset - value_w / 2, text_y), value_text, font=FONT_SMALL, fill=TEXT)
        centered(draw, (cx, plot_bottom + 43), str(fold_number), FONT_AXIS, TEXT)

    centered(draw, ((plot_left + plot_right) / 2, plot_bottom + 96), "Held-out province fold", FONT_AXIS_BOLD)
    draw.text((left + 42, plot_top - 58), "County-level R²", font=FONT_AXIS_BOLD, fill=TEXT)
    draw_legend(draw, left + PANEL_W - 625, top + 95)


def draw_legend(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    for index, (label, color) in enumerate([("Before calibration", BLUE), ("After calibration", ORANGE)]):
        yy = y + index * 50
        draw.rounded_rectangle((x, yy, x + 34, yy + 26), radius=4, fill=color)
        draw.text((x + 48, yy - 4), label, font=FONT_SMALL, fill=TEXT)


def draw_year_lines(draw: ImageDraw.ImageDraw, left: int, top: int, rows: list[dict[str, str]]) -> None:
    plot_left = left + 185
    plot_right = left + PANEL_W - 90
    plot_top = top + 190
    plot_bottom = top + PANEL_H - 155
    y_min, y_max = 0.58, 0.80
    plot_h = plot_bottom - plot_top

    def y_pos(value: float) -> float:
        return plot_bottom - (value - y_min) / (y_max - y_min) * plot_h

    for tick in [0.60, 0.65, 0.70, 0.75, 0.80]:
        y = y_pos(tick)
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=2)
        tick_text = f"{tick:.2f}"
        width, height = text_size(draw, tick_text, FONT_AXIS)
        draw.text((plot_left - width - 22, y - height / 2), tick_text, font=FONT_AXIS, fill=MUTED)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=TEXT, width=3)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=TEXT, width=3)

    years = sorted({int(row["year"]) for row in rows})
    by_prediction = {
        prediction: {int(row["year"]): float(row["r2"]) for row in rows if row["prediction"] == prediction}
        for prediction in ["before_calibration", "after_calibration"]
    }
    points_by_series: list[tuple[str, tuple[int, int, int]]] = [
        ("before_calibration", BLUE),
        ("after_calibration", ORANGE),
    ]
    for prediction, color in points_by_series:
        points: list[tuple[float, float]] = []
        for idx, year in enumerate(years):
            x = plot_left + (plot_right - plot_left) * idx / (len(years) - 1)
            value = by_prediction[prediction][year]
            y = y_pos(value)
            points.append((x, y))
        draw.line(points, fill=color, width=9, joint="curve")
        for (x, y), year in zip(points, years):
            value = by_prediction[prediction][year]
            draw.ellipse((x - 13, y - 13, x + 13, y + 13), fill=WHITE, outline=color, width=7)
            value_text = f"{value:.3f}"
            value_w, value_h = text_size(draw, value_text, FONT_SMALL)
            text_y = y - value_h - 17 if prediction == "after_calibration" else y + 15
            draw.text((x - value_w / 2, text_y), value_text, font=FONT_SMALL, fill=TEXT)

    for idx, year in enumerate(years):
        x = plot_left + (plot_right - plot_left) * idx / (len(years) - 1)
        centered(draw, (x, plot_bottom + 43), str(year), FONT_AXIS, TEXT)
    centered(draw, ((plot_left + plot_right) / 2, plot_bottom + 96), "Held-out year", FONT_AXIS_BOLD)
    draw.text((left + 42, plot_top - 58), "County-level R²", font=FONT_AXIS_BOLD, fill=TEXT)
    draw_legend(draw, left + PANEL_W - 625, top + 95)


def build_summary_rows(summary: list[dict[str, str]]) -> list[dict[str, object]]:
    label_map = {
        "2020_leave_county_5fold_strict": "Leave-county, strict",
        "2020_leave_county_5fold_broad": "Leave-county, broad",
        "2020_leave_province_strict_before_calibration": "Leave-province, before",
        "2020_leave_province_strict_after_calibration": "Leave-province, after",
        "2019_2023_leave_one_year_out_strict_before_calibration": "Leave-year, before",
        "2019_2023_leave_one_year_out_strict_after_calibration": "Leave-year, after",
    }
    ordered = []
    by_name = {row["validation"]: row for row in summary}
    for key in label_map:
        row = by_name[key]
        ordered.append(
            {
                **row,
                "short_label": label_map[key],
                "n": int(row["n"]),
                "r2": float(row["r2"]),
                "mape": float(row["mape"]),
                "color": color_for(row["validation"], row["constraint"], row["mask"]),
            }
        )
    return ordered


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    summary_source = read_rows(SUMMARY_CSV)
    summary = build_summary_rows(summary_source)
    province_rows = read_rows(PROVINCE_FOLD_CSV)
    year_rows = read_rows(YEAR_CSV)

    image = Image.new("RGB", (CANVAS_W, CANVAS_H), WHITE)
    draw = ImageDraw.Draw(image)
    centered(draw, (CANVAS_W / 2, 70), "Technical validation after county-level population-weighted aggregation", FONT_MAIN_TITLE)

    left_a = MARGIN_X
    left_b = MARGIN_X + PANEL_W + PANEL_GAP_X
    top_ab = TOP
    top_cd = TOP + PANEL_H + PANEL_GAP_Y

    panel_frame(draw, left_a, top_ab, "a", "Overall coefficient of determination")
    draw_horizontal_bars(
        draw,
        left_a,
        top_ab,
        summary,
        key="r2",
        x_max=0.80,
        ticks=[0.0, 0.2, 0.4, 0.6, 0.8],
        tick_format=lambda value: f"{value:.1f}",
        axis_label="County-level R²",
    )

    panel_frame(draw, left_b, top_ab, "b", "Overall percentage error")
    draw_horizontal_bars(
        draw,
        left_b,
        top_ab,
        summary,
        key="mape",
        x_max=20.0,
        ticks=[0, 5, 10, 15, 20],
        tick_format=lambda value: f"{int(value)}",
        axis_label="County-level MAPE (%)",
        lower_better=True,
    )

    panel_frame(draw, left_a, top_cd, "c", "Leave-province performance by fold (2020)")
    draw_grouped_fold_bars(draw, left_a, top_cd, province_rows)

    panel_frame(draw, left_b, top_cd, "d", "Leave-one-year-out performance")
    draw_year_lines(draw, left_b, top_cd, year_rows)

    foot_y = top_cd + PANEL_H + 42
    note1 = "All metrics are county-level after rural-population-weighted aggregation of 1-km predictions."
    note2 = "Province calibration is a product-generation constraint; after-calibration results are not independent pixel-level validation."
    centered(draw, (CANVAS_W / 2, foot_y), note1, FONT_NOTE, TEXT)
    centered(draw, (CANVAS_W / 2, foot_y + 48), note2, FONT_NOTE, NEGATIVE)

    image.save(PNG_PATH, dpi=(600, 600), optimize=True)
    image.save(TIFF_PATH, dpi=(600, 600), compression="tiff_lzw")

    qa = {
        "figure": "Figure 4. Technical validation",
        "outputs": {"png": str(PNG_PATH.relative_to(ROOT)), "tif": str(TIFF_PATH.relative_to(ROOT))},
        "dimensions_pixels": [CANVAS_W, CANVAS_H],
        "dpi": 600,
        "inputs": [str(path.relative_to(ROOT)) for path in [SUMMARY_CSV, PROVINCE_FOLD_CSV, YEAR_CSV]],
        "records": {"summary_rows": len(summary), "province_fold_rows": len(province_rows), "year_rows": len(year_rows)},
        "privacy": "Only aggregate/grouped public metrics are plotted; county observations and residuals are excluded.",
        "interpretation_caveat": "After-calibration values describe consistency of the province-constrained product and are not independent pixel-level validation.",
        "overall_metrics": [
            {"validation": row["validation"], "n": row["n"], "r2": row["r2"], "mape": row["mape"]}
            for row in summary
        ],
    }
    QA_PATH.write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {TIFF_PATH}")
    print(f"Wrote {QA_PATH}")


if __name__ == "__main__":
    main()
