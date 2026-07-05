import argparse
import glob
import os
import re
import shutil
import numpy as np
import pandas as pd
import rasterio

from collections import defaultdict
from pathlib import Path
from rasterio.windows import Window

AOI_CSV = Path("/Users/porjai/code/kait/data_capture/data/AOI_list.csv")
DRIVE_ROOT = (
    Path.home()
    / "Library"
    / "CloudStorage"
    / "GoogleDrive-porjaichavez@gmail.com"
    / "My Drive"
)
BASE_PATH = DRIVE_ROOT / "kait_observe" / "sentinel"
PREPROCESSING_DIR = DRIVE_ROOT / "kait_observe" / "preprocessing_ex"
LOG_PATH = DRIVE_ROOT / "kait_observe" / "preprocess_run_log.csv"
TARGET_SIZE = (50, 50)
BAND_ORDER = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")
REF_BAND = "B4"

def find_duplicate_folders(base_path, aoi_id):
    if not os.path.exists(base_path):
        print(f"don't see this base_path: {base_path}")
        return []

    all_entries = os.listdir(base_path)
    pattern = re.compile(rf"^{re.escape(aoi_id)}(?: \(\d+\))?$")

    matched = [
        e for e in all_entries
        if pattern.match(e) and os.path.isdir(os.path.join(base_path, e))
    ]
    return sorted(os.path.join(base_path, e) for e in matched)


def merge_duplicate_folders(base_path, aoi_id, dry_run=True):
    """
    Merge duplicate folders (s2_lat_lon, s2_lat_lon (1), s2_lat_lon (2), ...)
    into the main folder: s2_lat_lon (no parentheses).
    Then remove empty duplicate folders.
    """
    if not os.path.exists(base_path):
        print(f"don't see this base_path: {base_path}")
        return

    all_entries = os.listdir(base_path)
    pattern = re.compile(rf"^{re.escape(aoi_id)}(?: \((\d+)\))?$")

    matched = [
        e for e in all_entries
        if pattern.match(e) and os.path.isdir(os.path.join(base_path, e))
    ]

    if len(matched) <= 1:
        print(f"no duplicate folder for {aoi_id}")
        return

    main_folder = os.path.join(base_path, aoi_id)
    dup_folders = sorted(
        os.path.join(base_path, e) for e in matched if e != aoi_id
    )

    if not os.path.exists(main_folder):
        main_folder = dup_folders[0]
        dup_folders = dup_folders[1:]
        print(f"[NOTE] no folder without parentheses, using {main_folder} as main")

    print(f"main folder: {main_folder}")
    print(f"folders to merge + remove: {dup_folders}\n")

    for dup in dup_folders:
        files = os.listdir(dup)
        if not files:
            print(f"  {dup}: already empty")

        for fname in files:
            src_path = os.path.join(dup, fname)
            dst_path = os.path.join(main_folder, fname)

            if os.path.exists(dst_path):
                print(f"  [CONFLICT] {fname} already exists in main folder — skipped")
                continue

            if dry_run:
                print(f"  [DRY-RUN] would move: {src_path} -> {dst_path}")
            else:
                shutil.move(src_path, dst_path)
                print(f"  [MOVED] {fname}")

        if not dry_run:
            remaining = os.listdir(dup)
            if len(remaining) == 0:
                os.rmdir(dup)
                print(f"  [REMOVED] empty folder: {dup}")
            else:
                print(f"  [WARNING] {dup} still has files {remaining}, not removed")
        else:
            print(f"  [DRY-RUN] would remove: {dup} (if empty after move)")

    print(f"\ndone merging for {aoi_id}")


def build_allyear_stack(aoi_folder, target_size=TARGET_SIZE):
    aoi_folder = Path(aoi_folder)
    th, tw = target_size
    pattern = str(aoi_folder / "S2_*_REALDATE_STACKED.tif")
    files = glob.glob(pattern)

    def extract_year(fname):
        m = re.search(r"S2_(\d{4})_REALDATE_STACKED", fname)
        return int(m.group(1)) if m else None

    files_with_year = [(extract_year(f), f) for f in files]
    files_with_year = [(y, f) for y, f in files_with_year if y is not None]
    files_with_year.sort(key=lambda x: x[0])

    if not files_with_year:
        raise ValueError(f"no .tif file found in {aoi_folder}")

    all_arrays = []
    all_descriptions = []
    ref_profile = None
    ref_transform = None

    for year, fpath in files_with_year:
        with rasterio.open(fpath) as src:
            h, w = src.height, src.width

            if h < th or w < tw:
                print(f"  [SKIP YEAR] {year}: size ({h},{w}) smaller than target {target_size}")
                continue

            row_start = (h - th) // 2
            col_start = (w - tw) // 2
            window = Window(col_start, row_start, tw, th)

            if ref_profile is None:
                ref_profile = src.profile.copy()
                ref_transform = src.window_transform(window)

            for i in range(1, src.count + 1):
                arr = src.read(i, window=window)
                all_arrays.append(arr)
                all_descriptions.append(src.descriptions[i - 1])

        print(f"  [OK] {year}: {src.count} bands, original size ({h},{w}) -> cropped {target_size}")

    if not all_arrays:
        raise ValueError(f"no year passed the size check for {aoi_folder}")

    n_bands_total = len(all_arrays)

    parts = aoi_folder.parts
    kait_idx = parts.index("kait_observe")
    root_before = Path(*parts[:kait_idx])
    aoi_name = parts[-1]

    output_dir = root_before / "kait_observe" / "preprocessing_ex" / aoi_name
    output_path = output_dir / "Allyear.tif"
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_profile.update(
        count=n_bands_total,
        dtype=all_arrays[0].dtype,
        height=th,
        width=tw,
        transform=ref_transform,
    )

    with rasterio.open(output_path, "w", **ref_profile) as dst:
        for i, (arr, desc) in enumerate(zip(all_arrays, all_descriptions), start=1):
            dst.write(arr, i)
            dst.set_band_description(i, desc)

    print(f"  Saved to: {output_path}")
    return output_path


def parse_band_desc(desc):
    if not desc:
        return None
    m = re.match(r"^(\d{4}-\d{2}-\d{2})_(B\d+A?)(?:_(\d+))?$", desc)
    if not m:
        return None
    date_str, band, acq_str = m.groups()
    acq = int(acq_str) if acq_str is not None else 0
    return date_str, band, acq


def dedupe_multi_acquisition(input_path, output_path, band_order=BAND_ORDER, ref_band=REF_BAND):
    band_order = list(band_order)

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        date_acq_band = defaultdict(lambda: defaultdict(dict))

        for i, desc in enumerate(src.descriptions):
            parsed = parse_band_desc(desc)
            if parsed is None:
                print(f"  [WARNING] cannot parse: {desc} (band {i + 1}) - skipped")
                continue
            date_str, band, acq = parsed
            date_acq_band[date_str][acq][band] = i + 1  # rasterio 1-based

        all_arrays = []
        all_descriptions = []

        for date_str in sorted(date_acq_band.keys()):
            acq_dict = date_acq_band[date_str]  # {acq_num: {band: idx}}
            acq_nums = sorted(acq_dict.keys())

            if len(acq_nums) == 1:
                best_acq = acq_nums[0]
            else:
                best_acq = None
                best_valid_frac = -1
                for acq in acq_nums:
                    if ref_band not in acq_dict[acq]:
                        continue
                    idx = acq_dict[acq][ref_band]
                    arr = src.read(idx)
                    valid_frac = np.mean(~np.isnan(arr))
                    if valid_frac > best_valid_frac:
                        best_valid_frac = valid_frac
                        best_acq = acq
                if best_acq is None:
                    print(f"  [SKIP] {date_str}: no acquisition has {ref_band}")
                    continue

            band_map = acq_dict[best_acq]
            missing = [b for b in band_order if b not in band_map]
            if missing:
                print(f"  [SKIP] {date_str}: missing bands {missing}")
                continue

            for b in band_order:
                idx = band_map[b]
                arr = src.read(idx)
                all_arrays.append(arr)
                all_descriptions.append(f"{date_str}_{b}")

        n_bands_total = len(all_arrays)
        if n_bands_total == 0:
            raise ValueError(f"no band survived dedupe for {input_path}")

        print(f"  full band: {n_bands_total} bands ({n_bands_total // len(band_order)} dates)")

        profile.update(
            count=n_bands_total,
            dtype=all_arrays[0].dtype,
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            for i, (arr, desc) in enumerate(zip(all_arrays, all_descriptions), start=1):
                dst.write(arr, i)
                dst.set_band_description(i, desc)

    print(f"  Saved deduped stack to: {output_path}")
    return output_path


def build_deduped_stack_direct(aoi_folder, target_size=TARGET_SIZE, band_order=BAND_ORDER, ref_band=REF_BAND):
    aoi_folder = Path(aoi_folder)
    th, tw = target_size
    band_order = list(band_order)

    pattern = str(aoi_folder / "S2_*_REALDATE_STACKED.tif")
    files = glob.glob(pattern)

    def extract_year(fname):
        m = re.search(r"S2_(\d{4})_REALDATE_STACKED", fname)
        return int(m.group(1)) if m else None

    files_with_year = [(extract_year(f), f) for f in files]
    files_with_year = [(y, f) for y, f in files_with_year if y is not None]
    files_with_year.sort(key=lambda x: x[0])

    if not files_with_year:
        raise ValueError(f"no .tif file found in {aoi_folder}")

    # 1. read + crop every year, group bands by date/acquisition 
    date_acq_band = defaultdict(lambda: defaultdict(dict))  
    ref_profile = None
    ref_transform = None

    for year, fpath in files_with_year:
        with rasterio.open(fpath) as src:
            h, w = src.height, src.width

            if h < th or w < tw:
                print(f"  [SKIP YEAR] {year}: size ({h},{w}) smaller than target {target_size}")
                continue

            row_start = (h - th) // 2
            col_start = (w - tw) // 2
            window = Window(col_start, row_start, tw, th)

            if ref_profile is None:
                ref_profile = src.profile.copy()
                ref_transform = src.window_transform(window)

            for i, desc in enumerate(src.descriptions):
                parsed = parse_band_desc(desc)
                if parsed is None:
                    print(f"  [WARNING] cannot parse: {desc} (band {i + 1}) - skipped")
                    continue
                date_str, band, acq = parsed
                arr = src.read(i + 1, window=window)
                date_acq_band[date_str][acq][band] = arr

        print(f"  [OK] {year}: {src.count} bands, original size ({h},{w}) -> cropped {target_size}")

    if ref_profile is None:
        raise ValueError(f"no year passed the size check for {aoi_folder}")

    # 2. pick the acquisition with least NaN per date 
    all_arrays = []
    all_descriptions = []

    for date_str in sorted(date_acq_band.keys()):
        acq_dict = date_acq_band[date_str]  
        acq_nums = sorted(acq_dict.keys())

        if len(acq_nums) == 1:
            best_acq = acq_nums[0]
        else:
            best_acq = None
            best_valid_frac = -1
            for acq in acq_nums:
                if ref_band not in acq_dict[acq]:
                    continue
                arr = acq_dict[acq][ref_band]
                valid_frac = np.mean(~np.isnan(arr))
                if valid_frac > best_valid_frac:
                    best_valid_frac = valid_frac
                    best_acq = acq
            if best_acq is None:
                print(f"  [SKIP] {date_str}: no acquisition has {ref_band}")
                continue

        band_map = acq_dict[best_acq]
        missing = [b for b in band_order if b not in band_map]
        if missing:
            print(f"  [SKIP] {date_str}: missing bands {missing}")
            continue

        for b in band_order:
            all_arrays.append(band_map[b])
            all_descriptions.append(f"{date_str}_{b}")

    n_bands_total = len(all_arrays)
    if n_bands_total == 0:
        raise ValueError(f"no band survived dedupe for {aoi_folder}")

    print(f"  full band: {n_bands_total} bands ({n_bands_total // len(band_order)} dates)")

    # 3. write only the final deduped file 
    parts = aoi_folder.parts
    kait_idx = parts.index("kait_observe")
    root_before = Path(*parts[:kait_idx])
    aoi_name = parts[-1]

    output_dir = root_before / "kait_observe" / "preprocessing_ex" / aoi_name
    output_path = output_dir / "Allyear_deduped.tif"
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_profile.update(
        count=n_bands_total,
        dtype=all_arrays[0].dtype,
        height=th,
        width=tw,
        transform=ref_transform,
    )

    with rasterio.open(output_path, "w", **ref_profile) as dst:
        for i, (arr, desc) in enumerate(zip(all_arrays, all_descriptions), start=1):
            dst.write(arr, i)
            dst.set_band_description(i, desc)

    print(f"  Saved deduped stack to: {output_path}")
    return output_path


def process_aoi(base_path, preprocessing_dir, aoi_id):
    aoi_folder = os.path.join(base_path, aoi_id)
    status = {"aoi_id": aoi_id, "step": None, "error": None}

    dup_folders = find_duplicate_folders(base_path, aoi_id)

    if len(dup_folders) == 0:
        status["step"] = "no_folder"
        status["error"] = "folder not found"
        print("  [SKIP] no folder found")
        return status

    if len(dup_folders) > 1:
        merge_duplicate_folders(base_path, aoi_id, dry_run=False)

    deduped_path = os.path.join(preprocessing_dir, aoi_id, "Allyear_deduped.tif")

    if os.path.exists(deduped_path):
        print("  [SKIP] Allyear_deduped.tif already exists")
    else:
        build_deduped_stack_direct(aoi_folder, target_size=TARGET_SIZE)

    status["step"] = "done"
    return status


def main():
    parser = argparse.ArgumentParser(description="Sentinel-2 preprocessing batch runner")
    parser.add_argument("--start", type=int, default=0, help="Start row index in AOI_list.csv")
    parser.add_argument("--end", type=int, default=None, help="End row index (exclusive). Default: run to the end of the file")
    parser.add_argument("--aoi-csv", type=str, default=str(AOI_CSV), help="Path to AOI_list.csv")
    args = parser.parse_args()

    aoi_csv_path = Path(args.aoi_csv)
    if not aoi_csv_path.exists():
        raise FileNotFoundError(f"AOI CSV not found: {aoi_csv_path}")

    aoi_df = pd.read_csv(aoi_csv_path)

    end = args.end if args.end is not None else len(aoi_df)
    aoi_df = aoi_df.iloc[args.start: end].reset_index(drop=True)
    print(f"Processing {len(aoi_df)} AOIs (rows {args.start}-{args.start + len(aoi_df) - 1})")

    results = []

    for i, row in aoi_df.iterrows():
        lat, lon = row["lat"], row["lon"]
        aoi_id = f"s2_{lat}_{lon}"

        print(f"\n[{i + 1}/{len(aoi_df)}] {aoi_id}")

        try:
            status = process_aoi(str(BASE_PATH), str(PREPROCESSING_DIR), aoi_id)
            status["id"] = row.get("id", i)
        except Exception as e:
            status = {"id": row.get("id", i), "aoi_id": aoi_id, "step": "error", "error": str(e)}
            print(f"  [ERROR] {e}")

        results.append(status)

    results_df = pd.DataFrame(results)

    print("\n=== SUMMARY ===")
    print(results_df["step"].value_counts())

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(LOG_PATH, index=False)
    print(f"\nLog saved to: {LOG_PATH}")

    problem_rows = results_df[results_df["step"] != "done"]
    if len(problem_rows) > 0:
        print(f"\n{len(problem_rows)} AOI(s) not fully done:")
        print(problem_rows)


if __name__ == "__main__":
    main()