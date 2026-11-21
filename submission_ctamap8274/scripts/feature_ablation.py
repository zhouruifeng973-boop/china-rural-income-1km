"""Helpers for versioned feature-ablation sensitivity experiments."""

from __future__ import annotations

from collections.abc import Iterable


def patch_weak_pixel_loader(weak_module, excluded_features: Iterable[str]) -> None:
    """Remove selected columns from every loaded training pixel panel."""
    excluded = set(excluded_features)
    if not excluded:
        return
    original = weak_module.load_pixel_panel

    def load_pixel_panel_without_excluded(year: int, mask_name: str) -> dict:
        panel = original(year, mask_name)
        names = list(panel["band_names"])
        missing = excluded.difference(names)
        if missing:
            raise RuntimeError(f"Excluded features are absent from training data: {sorted(missing)}")
        keep = [index for index, name in enumerate(names) if name not in excluded]
        panel["X"] = panel["X"][:, keep]
        panel["band_names"] = [names[index] for index in keep]
        return panel

    weak_module.load_pixel_panel = load_pixel_panel_without_excluded


def patch_final_map_features(final_maps_module, excluded_features: Iterable[str]) -> None:
    """Remove selected standalone raster features from nationwide prediction."""
    excluded = set(excluded_features)
    if not excluded:
        return
    original_names = list(final_maps_module.BAND_NAMES)
    missing = excluded.difference(original_names)
    if missing:
        raise RuntimeError(f"Excluded features are absent from prediction bands: {sorted(missing)}")
    original_specs = final_maps_module.feature_specs

    def feature_specs_without_excluded(year: int):
        result = []
        for path, names in original_specs(year):
            overlap = excluded.intersection(names)
            if overlap and overlap != set(names):
                raise RuntimeError(
                    "Feature ablation currently requires excluded bands to occupy a "
                    f"standalone raster; mixed raster encountered: {path}, {names}"
                )
            if not overlap:
                result.append((path, names))
        return result

    final_maps_module.BAND_NAMES = [name for name in original_names if name not in excluded]
    final_maps_module.feature_specs = feature_specs_without_excluded
