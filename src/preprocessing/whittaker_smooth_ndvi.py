"""
Reconstructs a denoised NDVI curve per AOI with a Whittaker smoother
(Eilers 2003), weighted by valid_frac so low-quality/cloud-contaminated
observations are trusted less than clean ones.

Also computes bracket_span per real observation: the gap (in days) from
the nearest real observation before it to the nearest real observation
after it. Downstream, samples are only allowed to use ndvi_smoothed as a
training target where bracket_span is small enough to trust -- large
gaps (e.g. the Jun-Aug monsoon season) mean the smoothed value is more
"invented" than "denoised".

Input: ndvi_timeseries.csv (after preprocessing/encode_temporal.py)
Output: same file + ndvi_smoothed, bracket_span columns

Only the target/answer path is meant to use ndvi_smoothed. The observe
(input) window must keep using the raw median_ndvi column untouched.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.sparse import eye, diags, csc_matrix
from scipy.sparse.linalg import spsolve

AOI_ID_COL = "aoi_id"
DATE_COL = "date"
NDVI_COL = "median_ndvi"
VALID_FRAC_COL = "valid_frac"

LAMBDA = 50.0        # Whittaker roughness penalty; higher = smoother
DIFF_ORDER = 2        # penalize curvature (standard choice)
MIN_WEIGHT = 0.05      # floor so a real point is never fully ignored


def whittaker_smooth(y: np.ndarray, weights: np.ndarray, lmbd: float, d: int = DIFF_ORDER) -> np.ndarray:
    """
    Eilers (2003) "A perfect smoother". Solves for z minimizing
    sum(w_i (y_i - z_i)^2) + lmbd * sum((D^d z)_i^2)
    where D^d is the d-th order finite-difference operator.

    Points with weight 0 (missing days) are pulled toward their
    smooth neighbors rather than toward any observed value.
    """
    m = len(y)
    E = eye(m, format="csc")
    for _ in range(d):
        E = E[1:] - E[:-1]
    W = diags(weights, 0, shape=(m, m))
    A = csc_matrix(W + lmbd * E.T.dot(E))
    b = weights * y
    return spsolve(A, b)


def smooth_aoi(aoi_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    aoi_df: real observations for one AOI, any row order, with date/median_ndvi/valid_frac.
    Returns (smoothed_at_real_dates, bracket_span_at_real_dates), aligned to aoi_df's row order.
    """
    original_index = aoi_df.index
    sorted_df = aoi_df.sort_values(DATE_COL)
    dates = pd.to_datetime(sorted_df[DATE_COL]).to_numpy()

    grid = pd.date_range(dates.min(), dates.max(), freq="D")
    idx = grid.get_indexer(pd.DatetimeIndex(dates))

    y = np.zeros(len(grid))
    w = np.zeros(len(grid))
    y[idx] = sorted_df[NDVI_COL].to_numpy()
    w[idx] = sorted_df[VALID_FRAC_COL].to_numpy().clip(min=MIN_WEIGHT)

    z = whittaker_smooth(y, w, LAMBDA)
    smoothed_at_real = z[idx]

    d = pd.DatetimeIndex(dates)
    prev_gap = np.r_[np.nan, (d[1:] - d[:-1]).days.astype(float)]
    next_gap = np.r_[(d[1:] - d[:-1]).days.astype(float), np.nan]
    bracket_span = prev_gap + next_gap

    # scatter back to aoi_df's original (possibly unsorted) row order
    smoothed_series = pd.Series(smoothed_at_real, index=sorted_df.index)
    bracket_series = pd.Series(bracket_span, index=sorted_df.index)
    return (
        smoothed_series.reindex(original_index).to_numpy(),
        bracket_series.reindex(original_index).to_numpy(),
    )


def smooth_all(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    smoothed_col = np.full(len(df), np.nan)
    bracket_col = np.full(len(df), np.nan)

    for aoi_id, group in df.groupby(AOI_ID_COL, sort=False):
        smoothed, bracket = smooth_aoi(group)
        smoothed_col[group.index.to_numpy()] = smoothed
        bracket_col[group.index.to_numpy()] = bracket

    df["ndvi_smoothed"] = smoothed_col
    df["bracket_span"] = bracket_col
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Add Whittaker-smoothed NDVI + bracket_span columns to an NDVI CSV."
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
    n_aoi = df[AOI_ID_COL].nunique()
    print(f"Smoothing {len(df)} rows across {n_aoi} AOIs...")

    df = smooth_all(df)

    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows (with ndvi_smoothed, bracket_span) to {output_path}")


if __name__ == "__main__":
    main()
