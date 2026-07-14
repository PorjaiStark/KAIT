import ee
import pandas as pd
import shutil
import time
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Download Sentinel-2 Time Series from Google Earth Engine")
parser.add_argument("--start", type=int, default=0, help="Start AOI index")
parser.add_argument("--end", type=int, default=None, help="End AOI index")
parser.add_argument("--buffer", type=int, default=250, help="Buffer size in meters")
parser.add_argument("--start-year", type=int, default=2019, help="Start year")
parser.add_argument("--end-year", type=int, default=2025, help="End year")

args = parser.parse_args()

PROJECT = "kait-499816"
BUFFER_M = args.buffer
AOI_CSV = Path("data/aoi_list.csv")
START_YEAR = args.start_year
END_YEAR = args.end_year
START_IDX = args.start
END_IDX = args.end

DRIVE_ROOT = (
    Path.home()
    / "Library"
    / "CloudStorage"
    / "GoogleDrive-porjaichavez@gmail.com"
    / "My Drive"
)

TARGET_DIR = DRIVE_ROOT / "kait_observe" / "sentinel"
PREFIX = "s2_"
POLL_SECONDS = 30
ee.Initialize(project=PROJECT)
print("GEE initialized")


def mask_clouds(img):

    scl = img.select("SCL")

    mask = (
        scl.neq(3)      # Cloud Shadow
        .And(scl.neq(8))    # Cloud Medium
        .And(scl.neq(9))    # Cloud High
        .And(scl.neq(10))   # Cirrus
    )

    return img.updateMask(mask)


def add_features(img, aoi):

    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A","B11","B12",]

    return (
        img
        .resample("bilinear")
        .select(bands)
        .toFloat()
        .clip(aoi)
        .copyProperties(img, ["system:time_start"])
    )


def add_valid_fraction(img, aoi):

    valid_frac = (
        img.select("B4")
        .mask()
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=10,
            maxPixels=1e9,
        )
        .get("B4")
    )

    return img.set("valid_fraction", valid_frac)


def export_year(
    aoi,
    folder_name,
    start_str,
    end_str,
    label,
):

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_str, end_str)
        .filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                70,
            )
        )
    )

    processed = (
        s2
        .map(mask_clouds)
        .map(lambda img: add_features(img, aoi))
        .map(lambda img: add_valid_fraction(img, aoi))
        .filter(ee.Filter.gt("valid_fraction",0.8,)
        )
        .sort("system:time_start")
    )

    n = processed.size().getInfo()

    print(f" Year {label} : {n} images")

    if n == 0:
        print(" Skip")
        return None

    image_list = processed.toList(n)

    first = ee.Image(image_list.get(0))
    first_date = (ee.Date(first.get("system:time_start")).format("YYYY-MM-dd"))
    stacked = first.rename(
        first.bandNames().map(
            lambda b:
            ee.String(first_date)
            .cat("_")
            .cat(b)
        )
    )

    for i in range(1, n):
        img = ee.Image(image_list.get(i))
        date = (ee.Date(img.get("system:time_start")).format("YYYY-MM-dd"))
        stacked = stacked.addBands(
            img.rename(
                img.bandNames().map(
                    lambda b:
                    ee.String(date)
                    .cat("_")
                    .cat(b)
                )

            )

        )

    drive_folder = f"{PREFIX}{folder_name}"
    task = ee.batch.Export.image.toDrive(

        image=stacked,
        description=f"S2_{folder_name}_{label}",
        folder=drive_folder,
        fileNamePrefix=f"S2_{label}_REALDATE_STACKED",
        region=aoi,
        scale=10,
        maxPixels=1e13,

    )

    task.start()
    print(f"Task Started")

    return task


def wait_for_tasks(tasks):

    print(f"\nWaiting {len(tasks)} Tasks...\n")

    done_states = {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
    }

    while True:

        states = {
            name: task.status()["state"]
            for name, task in tasks.items()
        }

        pending = [
            name
            for name, state in states.items()
            if state not in done_states
        ]

        completed = sum(
            1
            for state in states.values()
            if state == "COMPLETED"
        )

        failed = sum(
            1
            for state in states.values()
            if state == "FAILED"
        )

        print(
            f"Completed : {completed} | "
            f"Failed : {failed} | "
            f"Waiting : {len(pending)}"
        )

        if len(pending) == 0:
            break

        time.sleep(POLL_SECONDS)

    print("\n All Tasks Finished\n")


def move_s2_folders():

    if not DRIVE_ROOT.exists():
        print("Google Drive not found.")

        return

    TARGET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    folders = sorted(p for p in DRIVE_ROOT.iterdir()
        if p.is_dir()
        and p.name.startswith(PREFIX)

    )

    if len(folders) == 0:
        print("No exported folders found.")

        return

    moved = 0
    skipped = 0

    for folder in folders:
        dest = TARGET_DIR / folder.name

        if dest.exists():
            print(f"Skip : {folder.name}")

            skipped += 1

            continue

        shutil.move(str(folder),str(dest),)

        print(f"Move : {folder.name}")
        moved += 1

    print("\n==========================")
    print(f"Moved   : {moved}")
    print(f"Skipped : {skipped}")
    print("==========================")
    

aoi_df = pd.read_csv(AOI_CSV)
if END_IDX is None:

    END_IDX = len(aoi_df)

aoi_df = (
    aoi_df
    .iloc[START_IDX:END_IDX]
    .reset_index(drop=True)
)
print(f"\nExport AOI : {START_IDX} - {END_IDX}")
print(f"Total AOI : {len(aoi_df)}\n")

all_tasks = {}
for idx, row in aoi_df.iterrows():

    lat = row["lat"]
    lon = row["lon"]

    folder_name = f"{lat}_{lon}"

    print(f"[{idx+1}/{len(aoi_df)}] {folder_name}")

    aoi = (
        ee.Geometry
        .Point([lon, lat])
        .buffer(BUFFER_M)
        .bounds()
    )

    for year in range(
        START_YEAR,
        END_YEAR + 1,
    ):

        task = export_year(

            aoi=aoi,
            folder_name=folder_name,
            start_str=f"{year}-01-01",
            end_str=f"{year+1}-01-01",
            label=str(year),

        )

        if task:
            all_tasks[
                f"{folder_name}_{year}"
            ] = task

print(f"\nSubmitted {len(all_tasks)} Tasks")
print("Check Tasks : " "https://code.earthengine.google.com/tasks")

wait_for_tasks(all_tasks)
move_s2_folders()