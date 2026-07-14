""""
For temporal_encode,We add cyclical month encoding, gap_days, t_index, and location encoding.

Input: ndvi_timeseries.csv After preprocessing/compute_ndvi.py
Output: adding some columns about time in data/preprocessed/data_before_split/ndvi_timeseries.csv
"""
import argparse
import numpy as np
import pandas as pd

AOI_ID_COL = "aoi_id"
DATE_COL = "date"
NDVI_COL = "median_ndvi"
LAT_COL = "lat"
LON_COL = "lon"
LOCATION_COLS = [AOI_ID_COL]

def parse_aoi_id(df: pd.DataFrame, aoi_id_col: str = AOI_ID_COL) -> pd.DataFrame:
    """
    Adds "lat" and "lon" float columns to the dataframe.
    """
    df = df.copy()
    parts = df[aoi_id_col].astype(str).str.split("_")
    df["lat"] = parts.str[-2].astype(float)
    df["lon"] = parts.str[-1].astype(float)
    return df


def add_cyclical_month(df: pd.DataFrame, date_col: str = DATE_COL) -> pd.DataFrame:
    """Adds year, month, month_sin, month_cos columns."""
    dates = pd.to_datetime(df[date_col])
    df["year"] = dates.dt.year
    df["month"] = dates.dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_gap_days(
    df: pd.DataFrame,
    date_col: str = DATE_COL,
    location_cols=LOCATION_COLS,
) -> pd.DataFrame:
    """
    Adds t_index and gap_days:
    the number of days since the PREVIOUS observation in that AOI's series
    (first observation gets gap_days = 0).
    """
    df = df.copy()
    df["_date_parsed"] = pd.to_datetime(df[date_col])
    df = df.sort_values(location_cols + ["_date_parsed"]).reset_index(drop=True)
    df["gap_days"] = (
        df.groupby(location_cols)["_date_parsed"].diff().dt.days.fillna(0).astype(float)
    )
    df["t_index"] = df.groupby(location_cols).cumcount()
    return df.drop(columns="_date_parsed")


_LOCATION_CACHE: dict = {}


def add_location_encoding(
    df: pd.DataFrame,
    aoi_id_col: str = AOI_ID_COL,
    lat_col: str = LAT_COL,
    lon_col: str = LON_COL,
    cache: dict = None,
) -> pd.DataFrame:
    """
    Projects lat/lon onto a unit sphere in 3D Cartesian coordinates.Safe to call multiple times.
    """
    if cache is None:
        cache = _LOCATION_CACHE

    df = df.copy()

    # Remove old location columns if they already exist
    cols_to_drop = [
        "loc_x", "loc_y", "loc_z",
        "loc_x_x", "loc_y_x", "loc_z_x",
        "loc_x_y", "loc_y_y", "loc_z_y",
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    unique_aois = df[[aoi_id_col, lat_col, lon_col]].drop_duplicates(subset=aoi_id_col)
    missing = unique_aois[~unique_aois[aoi_id_col].isin(cache.keys())]

    if not missing.empty:
        lat_rad = np.radians(missing[lat_col].to_numpy())
        lon_rad = np.radians(missing[lon_col].to_numpy())
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)

        for aoi_id, xi, yi, zi in zip(missing[aoi_id_col], x, y, z):
            cache[aoi_id] = (float(xi), float(yi), float(zi))

    loc_lookup = pd.DataFrame(
        [(aoi_id, *cache[aoi_id]) for aoi_id in unique_aois[aoi_id_col]],
        columns=[aoi_id_col, "loc_x", "loc_y", "loc_z"],
    )

    return df.merge(loc_lookup, on=aoi_id_col, how="left")


def encode_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs all temporal + location encodings and returns the augmented dataframe.
    """
    df = parse_aoi_id(df)
    df = add_cyclical_month(df)
    df = add_gap_days(df)
    df = add_location_encoding(df)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Add temporal + location encoding columns to an NDVI CSV, in place."
    )
    parser.add_argument("--input", required=True, help="Path to ndvi_timeseries.csv")
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write to (defaults to overwriting --input, same file).",
    )
    args = parser.parse_args()
    output_path = args.output or args.input

    df = pd.read_csv(args.input)
    df = encode_temporal(df)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows (with temporal + location columns) to {output_path}")


if __name__ == "__main__":
    main()