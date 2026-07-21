"""
Regenerates train/valid/test1/test2 split CSVs using a shared 30-day
calendar anchor grid (identical anchor dates across every AOI) instead
of anchoring on each AOI's own irregular real observation dates.

Rationale: anchoring on real dates makes anchors AOI-specific (AOIs
don't share observation calendars) and, at ~5-20 day real spacing,
produces heavily overlapping 90-day observe windows -- redundant
near-duplicate samples. A shared grid fixes both: anchors are
comparable across AOIs, and 30-day spacing meaningfully reduces
window overlap. An anchor is kept only if the AOI has enough real
observations in its preceding 90-day window (MIN_OBS_IN_WINDOW) --
observe content itself is always real, unmodified data.

Split policy is unchanged from the existing aoi_membership.json:
  - "seen" AOIs, anchor year <= 2023 -> train
  - "seen" AOIs, anchor year == 2024 -> valid
  - "seen" AOIs, anchor year == 2025 -> test1
  - "unseen" AOIs, anchor year == 2025 -> test2

Input: ndvi_timeseries.csv (with bracket_span from whittaker_smooth_ndvi.py,
       used only to report expected target coverage -- not required),
       aoi_membership.json
Output: data/processed/split/{train,valid,test1,test2}.csv
"""
import argparse
import json

import numpy as np
import pandas as pd

AOI_ID_COL = "aoi_id"
DATE_COL = "date"

OBSERVE_WINDOW_DAYS = 90
PREDICT_WINDOW_DAYS = 60
ANCHOR_SPACING_DAYS = 30
MIN_OBS_IN_WINDOW = 3

GRID_START = "2019-02-01"
GRID_END = "2025-12-01"

OUT_COLUMNS = ["aoi_id", "tile", "anchor_idx", "anchor_date", "observe_len", "target_len"]


def year_to_split(year: int) -> str | None:
    if year <= 2023:
        return "train"
    if year == 2024:
        return "valid"
    if year == 2025:
        return "test"
    return None


def generate_anchors_for_aoi(aoi_df: pd.DataFrame, grid: pd.DatetimeIndex) -> list[dict]:
    dates = np.sort(pd.to_datetime(aoi_df[DATE_COL]).to_numpy())

    rows = []
    anchor_idx = 0

    for g in grid:
        window_start = g - pd.Timedelta(days=OBSERVE_WINDOW_DAYS)
        observe_len = int(((dates > window_start) & (dates <= g)).sum())

        if observe_len < MIN_OBS_IN_WINDOW:
            continue

        future_end = g + pd.Timedelta(days=PREDICT_WINDOW_DAYS)
        target_len = int(((dates > g) & (dates <= future_end)).sum())

        if target_len == 0:
            continue

        rows.append({
            "anchor_idx": anchor_idx,
            "anchor_date": g.date().isoformat(),
            "observe_len": observe_len,
            "target_len": target_len,
        })
        anchor_idx += 1

    return rows


def generate_splits(ndvi_df: pd.DataFrame, membership: dict) -> dict[str, pd.DataFrame]:
    tile_map = membership["tile_map"]
    seen_aois = set(membership["seen_aois"])
    unseen_aois = set(membership["unseen_aois"])

    grid = pd.date_range(GRID_START, GRID_END, freq=f"{ANCHOR_SPACING_DAYS}D")

    splits: dict[str, list[dict]] = {"train": [], "valid": [], "test1": [], "test2": []}

    for aoi_id, aoi_df in ndvi_df.groupby(AOI_ID_COL):
        is_seen = aoi_id in seen_aois
        is_unseen = aoi_id in unseen_aois
        if not is_seen and not is_unseen:
            continue

        rows = generate_anchors_for_aoi(aoi_df, grid)

        for row in rows:
            year = pd.to_datetime(row["anchor_date"]).year
            split = year_to_split(year)
            if split is None:
                continue

            row["aoi_id"] = aoi_id
            row["tile"] = tile_map.get(aoi_id, "")

            if is_seen:
                if split == "test":
                    splits["test1"].append(row)
                else:
                    splits[split].append(row)
            elif split == "test":
                splits["test2"].append(row)

    return {
        name: pd.DataFrame(rows, columns=OUT_COLUMNS)
        for name, rows in splits.items()
    }


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate train/valid/test1/test2 splits on a shared calendar anchor grid."
    )
    parser.add_argument("--ndvi-csv", default="data/processed/data_before_split/ndvi_timeseries.csv")
    parser.add_argument("--membership", default="data/processed/split/aoi_membership.json")
    parser.add_argument("--out-dir", default="data/processed/split")
    args = parser.parse_args()

    ndvi_df = pd.read_csv(args.ndvi_csv)
    ndvi_df[DATE_COL] = pd.to_datetime(ndvi_df[DATE_COL])

    with open(args.membership) as f:
        membership = json.load(f)

    splits = generate_splits(ndvi_df, membership)

    for name, df in splits.items():
        out_path = f"{args.out_dir}/{name}.csv"
        df.to_csv(out_path, index=False)
        print(f"{name}: {len(df)} anchors -> {out_path}")


if __name__ == "__main__":
    main()
