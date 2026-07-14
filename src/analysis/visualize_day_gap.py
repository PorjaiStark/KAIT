import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AOI_ID_COL = "aoi_id"
DATE_COL = "date"
LOCATION_COLS = [AOI_ID_COL]
CANDIDATE_WINDOWS_DAYS = [15, 30, 45, 60, 75, 90]


def compute_gap_days(df: pd.DataFrame) -> pd.DataFrame:
    """Reuses an existing gap_days column if present; otherwise computes it."""
    if "gap_days" in df.columns:
        return df
    df = df.copy()
    df["_date_parsed"] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(LOCATION_COLS + ["_date_parsed"]).reset_index(drop=True)
    df["gap_days"] = (
        df.groupby(LOCATION_COLS)["_date_parsed"].diff().dt.days.fillna(0).astype(float)
    )
    return df.drop(columns="_date_parsed")


def _windowed_counts(df: pd.DataFrame, window_days: int, direction: str) -> pd.Series:
    """
    For every observation (as a potential target), counts how many OTHER
    observations of the SAME AOI fall within `window_days`:
      - direction="trailing": counts prior observations (input/observe side)
      - direction="leading":  counts subsequent observations (predict/forecast side)
    """
    assert direction in ("trailing", "leading")
    dates_all = pd.to_datetime(df[DATE_COL])
    result = pd.Series(index=df.index, dtype="int64")

    for aoi, aoi_dates in dates_all.groupby(df[AOI_ID_COL]):
        order = aoi_dates.sort_values()
        d = order.to_numpy()
        idx = order.index.to_numpy()
        n_points = len(d)
        counts = np.empty(n_points, dtype="int64")

        for i, di in enumerate(d):
            if direction == "trailing":
                window_start = di - np.timedelta64(window_days, "D")
                counts[i] = int(((d[:i] >= window_start) & (d[:i] < di)).sum())
            else:  # leading
                window_end = di + np.timedelta64(window_days, "D")
                counts[i] = int(((d[i + 1:] > di) & (d[i + 1:] <= window_end)).sum())

        result.loc[idx] = counts

    return result


def obs_count_in_trailing_window(df: pd.DataFrame, window_days: int) -> pd.Series:
    """How many prior observations are available as INPUT (observe side)."""
    return _windowed_counts(df, window_days, direction="trailing").reset_index(drop=True)


def obs_count_in_leading_window(df: pd.DataFrame, window_days: int) -> pd.Series:
    """How many future observations fall inside the PREDICTION horizon (predict side)."""
    return _windowed_counts(df, window_days, direction="leading").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Analyze NDVI observation gap_days distribution.")
    parser.add_argument("--input", required=True, help="Path to ndvi_timeseries.csv (raw or encoded)")
    parser.add_argument("--output-dir", default=".", help="Where to save the plot(s)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = compute_gap_days(df)

    gaps = df.loc[df["gap_days"] > 0, "gap_days"]  # exclude the artificial 0 at each series start

    print("=== gap_days summary (days between consecutive observations) ===")
    print(gaps.describe(percentiles=[0.5, 0.9, 0.99]))
    print()

    print("=== observations per AOI (total count) ===")
    obs_per_aoi = df.groupby(AOI_ID_COL).size()
    print(obs_per_aoi.describe(percentiles=[0.1, 0.5, 0.9]))
    print()

    # 1 row for gap_days histogram + 1 row per window (observe | predict side by side)
    n_rows = 1 + len(CANDIDATE_WINDOWS_DAYS)
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4 * n_rows))

    # Top row: gap_days histogram spans both columns
    for col in (0, 1):
        axes[0, col].hist(gaps, bins=50)
        axes[0, col].set_title("Distribution of gap_days (all AOIs, all observations)")
        axes[0, col].set_xlabel("days since previous observation")
        axes[0, col].set_ylabel("count")

    print("=== observe (trailing) vs predict (leading) counts per window ===")
    for row_idx, window_days in enumerate(CANDIDATE_WINDOWS_DAYS, start=1):
        observe_counts = obs_count_in_trailing_window(df, window_days)
        predict_counts = obs_count_in_leading_window(df, window_days)

        pct_too_few_observe = (observe_counts < 2).mean() * 100
        pct_zero_predict = (predict_counts < 1).mean() * 100

        print(
            f"[{window_days}d] observe: mean={observe_counts.mean():.2f} "
            f"median={observe_counts.median():.1f} <2pts={pct_too_few_observe:.1f}% "
            f"| predict: mean={predict_counts.mean():.2f} "
            f"median={predict_counts.median():.1f} zero={pct_zero_predict:.1f}%"
        )

        ax_obs, ax_pred = axes[row_idx]

        ax_obs.hist(observe_counts, bins=range(0, int(observe_counts.max()) + 2))
        ax_obs.set_title(
            f"OBSERVE: trailing {window_days}d window per target "
            f"({pct_too_few_observe:.1f}% would have <2 input points)"
        )
        ax_obs.set_xlabel("observations available as input sequence")
        ax_obs.set_ylabel("count of target rows")

        ax_pred.hist(predict_counts, bins=range(0, int(predict_counts.max()) + 2), color="tab:orange")
        ax_pred.set_title(
            f"PREDICT: leading {window_days}d horizon per target "
            f"({pct_zero_predict:.1f}% would have 0 future points)"
        )
        ax_pred.set_xlabel("observations available as prediction targets")
        ax_pred.set_ylabel("count of target rows")

    print()
    fig.tight_layout()
    out_path = f"{args.output_dir}/data/visualization/gap_days_analysis.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()