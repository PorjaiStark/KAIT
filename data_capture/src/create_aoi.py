import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd
import rasterio

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from rasterio.windows import Window
from rasterio.transform import rowcol, xy
from rasterio.merge import merge
from pyproj import Geod

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

GEOD = Geod(ellps="WGS84")

# config
TIF_PATHS       = ["data/jaxa"]
MIN_DISTANCE_M  = 1000   
NUM_POINTS      = 1000
PATCH_SIZE_M    = 500
MIN_COVERAGE    = 0.9    
PADDY_CLASS     = 3
OUTPUT_CSV      = "data/AOI_list.csv"
GRID_N          = 25
MAX_CANDIDATES  = 500_000
SEED            = 42

def distance_m(lat1, lon1, lat2, lon2):
    _, _, d = GEOD.inv(lon1, lat1, lon2, lat2)
    return d


def meters_to_degrees_lat(meters):
    return meters / 111320.0


def meters_to_degrees_lon(meters, lat):
    return meters / (111320.0 * np.cos(np.radians(lat)))


@dataclass
class TileInfo:
    path: str
    bounds: Tuple[float, float, float, float]
    crs: object


def discover_tiles(tif_inputs):
    resolved = []
    for item in tif_inputs:
        if os.path.isdir(item):
            found = sorted(glob.glob(os.path.join(item, "*.tif")))
            if not found:
                print(f"  [WARNING] No .tif in: {item}")
            resolved.extend(found)
        elif os.path.isfile(item):
            resolved.append(item)
        else:
            raise FileNotFoundError(f"Path not found: {item}")
    if not resolved:
        raise FileNotFoundError("No .tif files found.")
    tiles = []
    for path in resolved:
        with rasterio.open(path) as src:
            tiles.append(TileInfo(path=path, bounds=src.bounds, crs=src.crs))
    return tiles


def get_patch_coverage(lat, lon, patch_size_m, paddy_class, all_tiles):
    half_lat = meters_to_degrees_lat(patch_size_m / 2)
    half_lon = meters_to_degrees_lon(patch_size_m / 2, lat)

    patch_left   = lon - half_lon
    patch_right  = lon + half_lon
    patch_bottom = lat - half_lat
    patch_top    = lat + half_lat

    candidate_paths = [
        t.path for t in all_tiles
        if not (t.bounds[2] < patch_left or t.bounds[0] > patch_right
                or t.bounds[3] < patch_bottom or t.bounds[1] > patch_top)
    ]
    if not candidate_paths:
        return None

    srcs = [rasterio.open(p) for p in candidate_paths]
    try:
        if len(srcs) == 1:
            src = srcs[0]
            row_min, col_min = rowcol(src.transform, patch_left, patch_top)
            row_max, col_max = rowcol(src.transform, patch_right, patch_bottom)
            row_min, row_max = sorted([row_min, row_max])
            col_min, col_max = sorted([col_min, col_max])
            row_min = max(row_min, 0); col_min = max(col_min, 0)
            row_max = min(row_max, src.height); col_max = min(col_max, src.width)
            if row_max <= row_min or col_max <= col_min:
                return None
            data = src.read(1, window=Window(col_min, row_min,
                                             col_max - col_min, row_max - row_min))
        else:
            merged, _ = merge(srcs, bounds=(patch_left, patch_bottom,
                                            patch_right, patch_top))
            data = merged[0]
    finally:
        for s in srcs:
            s.close()

    if data.size == 0:
        return None
    return np.sum(data == paddy_class) / data.size


def get_all_candidate_points(tile: TileInfo, paddy_class: int):
    with rasterio.open(tile.path) as src:
        band = src.read(1)
        rows, cols = np.where(band == paddy_class)
        if len(rows) == 0:
            return []
        lons, lats = xy(src.transform, rows, cols, offset="center")
    return list(zip(lats, lons))


def grid_sample_candidates(candidates, bbox, grid_n, seed):
    lat_min, lon_min, lat_max, lon_max = bbox
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min

    rng = np.random.default_rng(seed)

    if len(candidates) > MAX_CANDIDATES:
        idx = rng.choice(len(candidates), size=MAX_CANDIDATES, replace=False)
        candidates = [candidates[i] for i in idx]
        print(f"  Subsampled to {MAX_CANDIDATES:,} candidates")

    cells = {}
    for lat, lon in candidates:
        row = min(int((lat - lat_min) / lat_range * grid_n), grid_n - 1)
        col = min(int((lon - lon_min) / lon_range * grid_n), grid_n - 1)
        cells.setdefault((row, col), []).append((lat, lon))

    for pts in cells.values():
        rng.shuffle(pts)

    cell_keys = list(cells.keys())
    rng.shuffle(cell_keys)
    result = []
    max_per_cell = max(len(v) for v in cells.values())
    for i in range(max_per_cell):
        for key in cell_keys:
            if i < len(cells[key]):
                result.append(cells[key][i])

    print(f"  Grid {grid_n}x{grid_n}: {len(cells)} cells มี candidates, "
          f"ordered {len(result):,} candidates")
    return result


def load_existing_aois(output_csv):
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        print(f"Found existing {output_csv} with {len(df)} AOI(s) — will append.")
        return df
    return pd.DataFrame(columns=["id", "lat", "lon", "paddy_coverage"])


def select_aois(tif_inputs, min_distance_m, num_points, patch_size_m,
                min_coverage, paddy_class, existing_df, grid_n, seed):

    print(f"Resolving tif files from: {tif_inputs}")
    all_tiles = discover_tiles(tif_inputs)
    print(f"  Resolved {len(all_tiles)} tile(s)")

    bbox_left   = min(t.bounds[0] for t in all_tiles)
    bbox_bottom = min(t.bounds[1] for t in all_tiles)
    bbox_right  = max(t.bounds[2] for t in all_tiles)
    bbox_top    = max(t.bounds[3] for t in all_tiles)
    print(f"  TIF extent: lon [{bbox_left:.4f}, {bbox_right:.4f}] "
          f"lat [{bbox_bottom:.4f}, {bbox_top:.4f}]")

    print("Collecting candidate pixels (all tiles)...")
    candidates = []
    for t in tqdm(all_tiles, desc="Tiles"):
        pts = get_all_candidate_points(t, paddy_class)
        candidates.extend(pts)
        print(f"  {Path(t.path).name}: {len(pts):,} paddy pixels")
    print(f"  Total candidates: {len(candidates):,}")

    if not candidates:
        raise RuntimeError(f"No pixels of class={paddy_class} found.")

    print(f"Applying grid sampling (grid={grid_n}x{grid_n})...")
    ordered = grid_sample_candidates(
        candidates,
        (bbox_bottom, bbox_left, bbox_top, bbox_right),
        grid_n, seed
    )


    selected = []
    next_id = 1
    if not existing_df.empty:
        next_id = int(existing_df["id"].max()) + 1
        for _, row in existing_df.iterrows():
            selected.append({"lat": row["lat"], "lon": row["lon"]})

    new_points = []
    skipped_overlap = skipped_coverage = 0

    print(f"Selecting up to {num_points} AOIs "
          f"(patch={patch_size_m}m, min_coverage={min_coverage:.0%}, "
          f"min_spacing={min_distance_m}m)...")

    pbar = tqdm(total=num_points, desc="Selected AOIs")
    for lat, lon in ordered:
        if any(distance_m(lat, lon, s["lat"], s["lon"]) < min_distance_m
               for s in selected):
            skipped_overlap += 1
            continue

        coverage = get_patch_coverage(lat, lon, patch_size_m, paddy_class, all_tiles)
        if coverage is None or coverage < min_coverage:
            skipped_coverage += 1
            continue

        selected.append({"lat": lat, "lon": lon})
        new_points.append({
            "id": next_id,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "paddy_coverage": round(coverage, 4),
        })
        next_id += 1
        pbar.update(1)

        if len(new_points) >= num_points:
            break

    pbar.close()
    print(f"\nSkipped overlap: {skipped_overlap:,} | "
          f"Skipped coverage: {skipped_coverage:,}")
    print(f"New AOIs selected: {len(new_points)} / {num_points} requested")

    if len(new_points) < num_points:
        print("[WARNING]  not enough data, please try new min-distance-m cond")

    new_df = pd.DataFrame(new_points)
    combined_df = (pd.concat([existing_df, new_df], ignore_index=True)
                   if not new_df.empty else existing_df)
    return combined_df, new_df

# CIL
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--tif", nargs="+", default=TIF_PATHS)
    p.add_argument("--min-distance-m", type=float, default=MIN_DISTANCE_M)
    p.add_argument("--num-points", type=int, default=NUM_POINTS)
    p.add_argument("--patch-size-m", type=float, default=PATCH_SIZE_M)
    p.add_argument("--min-coverage", type=float, default=MIN_COVERAGE)
    p.add_argument("--paddy-class", type=int, default=PADDY_CLASS)
    p.add_argument("--output", type=str, default=OUTPUT_CSV)
    p.add_argument("--grid-n", type=int, default=GRID_N)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main():
    args = parse_args()
    existing_df = load_existing_aois(args.output)
    try:
        combined_df, new_df = select_aois(
            tif_inputs=args.tif,
            min_distance_m=args.min_distance_m,
            num_points=args.num_points,
            patch_size_m=args.patch_size_m,
            min_coverage=args.min_coverage,
            paddy_class=args.paddy_class,
            existing_df=existing_df,
            grid_n=args.grid_n,
            seed=args.seed,
        )
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(out_path, index=False)
    print(f"\nTotal AOIs in {out_path}: {len(combined_df)}")


if __name__ == "__main__":
    main()