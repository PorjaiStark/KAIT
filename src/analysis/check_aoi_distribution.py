import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

print("Loading...")


train = pd.read_csv(TRAIN_PATH)

ndvi = pd.read_csv(NDVI_PATH)


train["anchor_date"] = pd.to_datetime(
    train["anchor_date"]
)

ndvi["date"] = pd.to_datetime(
    ndvi["date"]
)



# =========================
# AOI location
# =========================

aoi_location = (
    ndvi[
        [
            "aoi_id",
            "lat",
            "lon"
        ]
    ]
    .drop_duplicates("aoi_id")
)


train = train.merge(
    aoi_location,
    on="aoi_id",
    how="left"
)



# =========================
# AOI sample count
# =========================

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


print("\n====================")
print("AOI statistics")
print("====================")


print(
    aoi_count.sample_count.describe()
)



# =========================
# Histogram AOI count
# =========================

plt.figure(figsize=(10,5))


sns.histplot(
    aoi_count.sample_count,
    bins=50
)


plt.xlabel(
    "Samples per AOI"
)

plt.ylabel(
    "Number of AOI"
)

plt.title(
    "AOI Sample Count Distribution"
)


plt.tight_layout()

plt.savefig(
    "aoi_sample_hist.png",
    dpi=300
)

plt.show()



# =========================
# Split two groups
# =========================

threshold = (
    aoi_count.sample_count.median()
)


aoi_count["group"] = (
    aoi_count.sample_count
    .apply(
        lambda x:
        "Low_AOI"
        if x < threshold
        else
        "High_AOI"
    )
)


print("\nThreshold")

print(threshold)



print(
    "\nGroup size"
)

print(
    aoi_count.group.value_counts()
)



print(
    "\nGroup statistics"
)

print(
    aoi_count
    .groupby("group")
    .sample_count
    .describe()
)



# =========================
# Spatial distribution
# =========================


plt.figure(figsize=(8,7))


sns.scatterplot(
    data=aoi_count,
    x="lon",
    y="lat",
    hue="group",
    size="sample_count",
    sizes=(20,200)
)


plt.title(
    "Spatial Distribution of AOI Groups"
)


plt.xlabel(
    "Longitude"
)

plt.ylabel(
    "Latitude"
)


plt.tight_layout()


plt.savefig(
    "aoi_group_spatial.png",
    dpi=300
)


plt.show()



# =========================
# Add group label to train
# =========================

train = train.merge(
    aoi_count[
        [
            "aoi_id",
            "group"
        ]
    ],
    on="aoi_id",
    how="left"
)



# =========================
# Temporal distribution
# =========================

train["year"] = (
    train.anchor_date.dt.year
)

train["month"] = (
    train.anchor_date.dt.month
)



plt.figure(figsize=(10,5))


sns.countplot(
    data=train,
    x="month",
    hue="group"
)


plt.title(
    "Month Distribution by AOI Group"
)


plt.tight_layout()


plt.savefig(
    "month_by_aoi_group.png",
    dpi=300
)


plt.show()



plt.figure(figsize=(8,5))


sns.countplot(
    data=train,
    x="year",
    hue="group"
)


plt.title(
    "Year Distribution by AOI Group"
)


plt.tight_layout()


plt.savefig(
    "year_by_aoi_group.png",
    dpi=300
)


plt.show()



# =========================
# Target NDVI distribution
# =========================


plt.figure(figsize=(10,5))


sns.histplot(
    data=train,
    x="target_len",
    hue="group",
    bins=30
)


plt.title(
    "Target Length Distribution by AOI Group"
)


plt.tight_layout()


plt.savefig(
    "target_len_by_group.png",
    dpi=300
)


plt.show()



# =========================
# Save table
# =========================


aoi_count.sort_values(
    "sample_count"
).to_csv(
    "aoi_sample_distribution.csv",
    index=False
)



print("\nDone")