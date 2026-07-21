import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# =========================
# Path
# =========================

TRAIN_PATH = (
    "data/processed/split/train.csv"
)

NDVI_PATH = (
    "data/processed/data_before_split/"
    "ndvi_timeseries.csv"
)


# =========================
# Load
# =========================

print("Loading data...")


train = pd.read_csv(TRAIN_PATH)

ndvi = pd.read_csv(NDVI_PATH)


train["anchor_date"] = pd.to_datetime(
    train["anchor_date"]
)

ndvi["date"] = pd.to_datetime(
    ndvi["date"]
)



# =========================
# Add AOI location
# =========================

aoi_location = (
    ndvi[
        [
            "aoi_id",
            "lat",
            "lon"
        ]
    ]
    .drop_duplicates(
        "aoi_id"
    )
)


train = train.merge(
    aoi_location,
    on="aoi_id",
    how="left"
)



# =========================
# Time feature
# =========================

train["year"] = (
    train.anchor_date.dt.year
)

train["month"] = (
    train.anchor_date.dt.month
)



# =========================
# Statistics
# =========================


print("====================")
print("Basic statistics")
print("====================")


print(
    "Number of AOI:",
    train.aoi_id.nunique()
)


print(
    "Number of samples:",
    len(train)
)


print("\nMonth distribution")

print(
    train.month
    .value_counts()
    .sort_index()
)


print("\nYear distribution")

print(
    train.year
    .value_counts()
    .sort_index()
)



# =====================================================
# 1. Month distribution
# =====================================================

plt.figure(figsize=(10,5))


sns.countplot(
    data=train,
    x="month",
    order=range(1,13)
)


plt.title(
    "Training Sample Distribution by Month"
)

plt.xlabel(
    "Month"
)

plt.ylabel(
    "Samples"
)


plt.tight_layout()

plt.savefig(
    "month_distribution.png",
    dpi=300
)

plt.show()



# =====================================================
# 2. Year distribution
# =====================================================

plt.figure(figsize=(8,5))


sns.countplot(
    data=train,
    x="year"
)


plt.title(
    "Training Sample Distribution by Year"
)


plt.xlabel(
    "Year"
)

plt.ylabel(
    "Samples"
)


plt.tight_layout()

plt.savefig(
    "year_distribution.png",
    dpi=300
)

plt.show()



# =====================================================
# 3. Year-Month heatmap
# =====================================================


year_month = (
    train
    .groupby(
        [
            "year",
            "month"
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)



plt.figure(figsize=(12,5))


sns.heatmap(
    year_month,
    annot=True,
    fmt="d",
    cmap="viridis"
)


plt.title(
    "Training Samples Distribution (Year × Month)"
)


plt.xlabel(
    "Month"
)

plt.ylabel(
    "Year"
)


plt.tight_layout()


plt.savefig(
    "year_month_heatmap.png",
    dpi=300
)

plt.show()



# =====================================================
# 4. AOI distribution
# =====================================================


aoi_count = (
    train
    .groupby(
        [
            "aoi_id",
            "lat",
            "lon"
        ]
    )
    .size()
    .reset_index(
        name="sample_count"
    )
)



print("\nAOI sample statistics")

print(
    aoi_count.sample_count.describe()
)



# histogram

plt.figure(figsize=(8,5))


plt.hist(
    aoi_count.sample_count,
    bins=50
)


plt.xlabel(
    "Samples per AOI"
)


plt.ylabel(
    "Number of AOIs"
)


plt.title(
    "AOI Sample Count Distribution"
)


plt.tight_layout()


plt.savefig(
    "aoi_sample_distribution.png",
    dpi=300
)


plt.show()



# log histogram

plt.figure(figsize=(8,5))


plt.hist(
    aoi_count.sample_count,
    bins=50,
    log=True
)


plt.xlabel(
    "Samples per AOI"
)


plt.ylabel(
    "Number of AOIs (log)"
)


plt.title(
    "AOI Sample Count Distribution (Log Scale)"
)


plt.tight_layout()


plt.savefig(
    "aoi_sample_distribution_log.png",
    dpi=300
)


plt.show()



# =====================================================
# 5. Spatial density
# =====================================================


plt.figure(figsize=(8,7))


scatter = plt.scatter(
    aoi_count.lon,
    aoi_count.lat,
    c=aoi_count.sample_count,
    s=40,
    cmap="viridis"
)


plt.colorbar(
    scatter,
    label="Training samples"
)


plt.xlabel(
    "Longitude"
)


plt.ylabel(
    "Latitude"
)


plt.title(
    "Spatial Distribution of Training Samples"
)


plt.grid()


plt.tight_layout()


plt.savefig(
    "spatial_sample_density.png",
    dpi=300
)


plt.show()



# =====================================================
# 6. Lorenz curve
# =====================================================


values = (
    np.sort(
        aoi_count.sample_count.values
    )
)


cum_samples = (
    np.cumsum(values)
    /
    np.sum(values)
)


cum_aoi = (
    np.arange(
        1,
        len(values)+1
    )
    /
    len(values)
)



plt.figure(figsize=(6,6))


plt.plot(
    cum_aoi,
    cum_samples
)


plt.plot(
    [0,1],
    [0,1],
    "--"
)


plt.xlabel(
    "Fraction of AOIs"
)


plt.ylabel(
    "Fraction of samples"
)


plt.title(
    "AOI Sample Concentration (Lorenz Curve)"
)


plt.grid()


plt.tight_layout()


plt.savefig(
    "aoi_lorenz_curve.png",
    dpi=300
)


plt.show()



print("\nDone")