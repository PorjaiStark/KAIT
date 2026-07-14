"""
Compute median NDVI, mean NDVI and valid pixel fraction from B4 band for each AOI across all dates.

Input: data from drive directory name /kait_observe/preprocessing_ex
Output: ndvi_timeseries.csv in data/preprocessed/data_before_split
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DRIVE_ROOT = (
    Path.home()
    / "Library"
    / "CloudStorage"
    / "GoogleDrive-porjaichavez@gmail.com"
    / "My Drive"
)
SENTINEL_DIR = DRIVE_ROOT / "kait_observe" / "preprocessing_ex"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "ndvi"
OUTPUT_CSV = OUTPUT_DIR / "ndvi_timeseries.csv"
BAND_ORDER = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")
RED_BAND = "B4"
NIR_BAND = "B8"

def parse_band_desc(desc):
    if not desc:
        return None
    m = re.match(r"^(\d{4}-\d{2}-\d{2})_(B\d+A?)$", desc)
    if not m:
        return None
    return m.groups()


def compute_median_ndvi_for_aoi(tif_path, red_band=RED_BAND, nir_band=NIR_BAND):
    with rasterio.open(tif_path) as src:
        descs = src.descriptions

        # group band index by date
        date_bands = {}
        for i, desc in enumerate(descs):
            parsed = parse_band_desc(desc)
            if parsed is None:
                continue
            date_str, band = parsed
            date_bands.setdefault(date_str, {})[band] = i + 1  # rasterio 1-based

        rows = []
        for date_str in sorted(date_bands.keys()):
            band_map = date_bands[date_str]
            if red_band not in band_map or nir_band not in band_map:
                print(f"    [SKIP] {date_str}: missing {red_band} or {nir_band}")
                continue

            red = src.read(band_map[red_band])
            nir = src.read(band_map[nir_band])

            with np.errstate(invalid="ignore", divide="ignore"):
                ndvi = (nir - red) / (nir + red)

            valid = ~np.isnan(ndvi)
            median_ndvi = np.nanmedian(ndvi) if valid.any() else np.nan
            mean_ndvi = np.nanmean(ndvi) if valid.any() else np.nan

            rows.append({
                "date": date_str,
                "median_ndvi": median_ndvi,
                "mean_ndvi": mean_ndvi,
                "valid_frac": valid.mean(),
            })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df


def batch_compute_ndvi(sentinel_dir, output_csv):
    aoi_folders = sorted(
        f for f in Path(sentinel_dir).iterdir()
        if f.is_dir() and f.name.startswith("s2_")
    )

    if not aoi_folders:
        raise ValueError(f"No AOI folders found in {sentinel_dir}")

    print(f"Found {len(aoi_folders)} AOI folders")

    all_rows = []
    n_ok, n_skip, n_error = 0, 0, 0

    for i, aoi_folder in enumerate(aoi_folders, start=1):
        aoi_id = aoi_folder.name
        tif_path = aoi_folder / "Allyear_deduped.tif"

        if not tif_path.exists():
            print(f"[{i}/{len(aoi_folders)}] [SKIP] {aoi_id}: Allyear_deduped.tif not found")
            n_skip += 1
            continue

        try:
            df = compute_median_ndvi_for_aoi(tif_path)
            if len(df) == 0:
                print(f"[{i}/{len(aoi_folders)}] [SKIP] {aoi_id}: no valid dates")
                n_skip += 1
                continue

            df["aoi_id"] = aoi_id
            all_rows.append(df)
            print(f"[{i}/{len(aoi_folders)}] [OK] {aoi_id}: {len(df)} dates")
            n_ok += 1

        except Exception as e:
            print(f"[{i}/{len(aoi_folders)}] [ERROR] {aoi_id}: {e}")
            n_error += 1

    if not all_rows:
        raise ValueError("No AOI produced any NDVI data — nothing to save")

    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined[["aoi_id", "date", "median_ndvi", "mean_ndvi", "valid_frac"]]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)

    print(f"\n=== SUMMARY ===")
    print(f"OK: {n_ok}, Skipped: {n_skip}, Errors: {n_error}")
    print(f"Total rows: {len(combined)} ({combined['aoi_id'].nunique()} AOIs)")
    print(f"Saved to: {output_csv}")

    return combined


if __name__ == "__main__":
    batch_compute_ndvi(SENTINEL_DIR, OUTPUT_CSV)