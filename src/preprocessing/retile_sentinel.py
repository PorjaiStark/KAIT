"""
Re-encodes sentinel AOI tifs (Allyear_deduped.tif) from pixel- to
band-interleave.

Each AOI's tif was written pixel-interleaved with GDAL's default 256x256
block size, but the image itself is only 50x50. Since the whole image
fits inside one block, reading even a single band forces GDAL to
decompress the *entire* block across all ~2240 bands at once: measured
directly at ~600MB per open+read (see MAX_OPEN_SENTINEL_HANDLES in
datasets/dataloader.py). Re-writing with interleave=band lets a read of
N bands touch only those N bands' data, cutting per-read cost to a few
hundred KB and making it safe to raise NUM_WORKERS / the handle cap back
up without risking an OOM kill.

Input:  <input_root>/<aoi_id>/Allyear_deduped.tif   (pixel-interleaved)
Output: <output_root>/<aoi_id>/Allyear_deduped.tif  (band-interleaved)

Originals are left untouched. Point sentinel_root at output_root in
train.py once you've spot-checked the --verify output.
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import rasterio

TIF_NAME = "Allyear_deduped.tif"


def retile_one(paths):
    src_path, dst_path = paths

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    with rasterio.open(src_path) as src:
        data = src.read()  # one big decode -- unavoidable on the old pixel-interleaved layout
        descriptions = src.descriptions
        profile = src.profile.copy()

    profile.update(interleave="band", tiled=False, compress="lzw")
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)

    tmp_path = dst_path + ".tmp"
    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(data)
        for i, desc in enumerate(descriptions, start=1):
            dst.set_band_description(i, desc)

    os.replace(tmp_path, dst_path)
    return src_path


def verify_one(paths):
    src_path, dst_path = paths
    with rasterio.open(src_path) as s, rasterio.open(dst_path) as d:
        assert s.descriptions == d.descriptions, f"description mismatch: {src_path}"
        assert (s.count, s.width, s.height) == (d.count, d.width, d.height), f"shape mismatch: {src_path}"
        a = s.read(indexes=list(range(1, 11)))
        b = d.read(indexes=list(range(1, 11)))
        assert np.array_equal(a, b, equal_nan=True), f"pixel mismatch: {src_path}"
    return src_path


def collect_jobs(input_root, output_root, limit=None):
    aoi_dirs = sorted(
        d for d in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, d)) and not d.startswith(".")
    )
    if limit:
        aoi_dirs = aoi_dirs[:limit]

    jobs = [
        (
            os.path.join(input_root, aoi, TIF_NAME),
            os.path.join(output_root, aoi, TIF_NAME),
        )
        for aoi in aoi_dirs
    ]
    return [(s, d) for s, d in jobs if os.path.isfile(s)]


def main():
    parser = argparse.ArgumentParser(
        description="Re-tile sentinel AOI tifs from pixel- to band-interleave."
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N AOIs (for a dry run).")
    parser.add_argument("--verify", action="store_true", help="Re-open every converted file and assert it's pixel-identical to its source.")
    args = parser.parse_args()

    jobs = collect_jobs(args.input_root, args.output_root, args.limit)
    print(f"Re-tiling {len(jobs)} AOIs -> {args.output_root} with {args.workers} workers...")

    t0 = time.time()
    done = 0
    mapper = (lambda fn, xs: map(fn, xs)) if args.workers <= 1 else (
        lambda fn, xs: Pool(args.workers).imap_unordered(fn, xs)
    )
    for _ in mapper(retile_one, jobs):
        done += 1
        if done % 50 == 0 or done == len(jobs):
            print(f"  converted {done}/{len(jobs)}  ({time.time()-t0:.0f}s elapsed)", flush=True)

    if args.verify:
        print("Verifying pixel equality on all converted files...")
        done = 0
        for _ in mapper(verify_one, jobs):
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f"  verified {done}/{len(jobs)}", flush=True)
        print("All converted files match their source data exactly.")

    print(f"Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
