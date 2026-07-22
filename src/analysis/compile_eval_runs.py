import os
import pandas as pd
import matplotlib.pyplot as plt

RUNS_LOG = "outputs/evaluation/runs_log.csv"

OUTPUT_DIR = "outputs/evaluation"


# =========================
# Load
# =========================

print("Loading run log...")


runs = pd.read_csv(RUNS_LOG)

runs["timestamp"] = pd.to_datetime(
    runs["timestamp"]
)

runs = runs.sort_values("timestamp")



# =========================
# Table
# =========================

print("\n====================")
print(f"{len(runs)} run(s) found")
print("====================")


cols = [
    c for c in [
        "run_id", "timestamp", "mae", "rmse",
        "pearson_corr", "r2",
        "within_sample_corr_mean", "within_sample_corr_std",
        "std_ratio",
    ]
    if c in runs.columns
]

print(
    runs[cols].to_string(index=False)
)



# =========================
# Best run so far, per metric
# =========================

print("\n====================")
print("Best run per metric")
print("====================")

if "mae" in runs.columns:
    best = runs.loc[runs.mae.idxmin()]
    print(f"Lowest MAE  : {best.mae:.6f}  ({best.run_id})")

if "rmse" in runs.columns:
    best = runs.loc[runs.rmse.idxmin()]
    print(f"Lowest RMSE : {best.rmse:.6f}  ({best.run_id})")

if "pearson_corr" in runs.columns:
    best = runs.loc[runs.pearson_corr.idxmax()]
    print(f"Highest corr: {best.pearson_corr:.4f}  ({best.run_id})")

if "within_sample_corr_mean" in runs.columns:
    best = runs.loc[runs.within_sample_corr_mean.idxmax()]
    print(f"Highest within-sample corr: {best.within_sample_corr_mean:.4f}  ({best.run_id})")



# =========================
# Trend plot across runs
# =========================

plt.figure(figsize=(9, 5))

if "mae" in runs.columns:
    plt.plot(runs.timestamp, runs.mae, marker="o", label="MAE")

if "rmse" in runs.columns:
    plt.plot(runs.timestamp, runs.rmse, marker="o", label="RMSE")

if "pearson_corr" in runs.columns:
    plt.plot(runs.timestamp, runs.pearson_corr, marker="o", label="Pearson corr")

if "within_sample_corr_mean" in runs.columns:
    plt.plot(runs.timestamp, runs.within_sample_corr_mean, marker="o", label="Within-sample corr")

plt.xlabel("Run timestamp")
plt.ylabel("Metric value")
plt.title("Evaluation metrics across runs")
plt.legend()
plt.grid()
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "runs_comparison.png"),
    dpi=150
)

print(f"\nSaved trend plot to {os.path.join(OUTPUT_DIR, 'runs_comparison.png')}")

plt.close()



print("\nDone")
