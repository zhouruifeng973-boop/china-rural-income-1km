# -*- coding: utf-8 -*-
"""Write exact band statistics for ArcGIS/GDAL display without changing pixels."""

from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

from stage3_weak_supervised_downscaling import ROOT

OUT = ROOT / "outputs" / "stage5_final_maps"


def main():
    paths = sorted(OUT.glob("rural_income_20??_final_calibrated_2019yuan_strict.tif"))
    paths += sorted(OUT.glob("rural_income_change_2019_2023_*.tif"))
    rows = []
    for path in paths:
        with rasterio.open(path, "r+") as src:
            data = src.read(1, masked=True)
            values = data.compressed().astype(np.float64)
            stats = {
                "STATISTICS_MINIMUM": f"{values.min():.10g}",
                "STATISTICS_MAXIMUM": f"{values.max():.10g}",
                "STATISTICS_MEAN": f"{values.mean():.10g}",
                "STATISTICS_STDDEV": f"{values.std():.10g}",
                "STATISTICS_VALID_PERCENT": f"{100 * len(values) / data.size:.10g}",
            }
            src.update_tags(1, **stats)
            rows.append({"file": path.name, **stats})
            print(path.name, stats)
    pd.DataFrame(rows).to_csv(
        OUT / "raster_band_statistics.csv", index=False, encoding="utf-8-sig"
    )


if __name__ == "__main__":
    main()
