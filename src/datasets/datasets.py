"""
Split-building (one-time) :
    a 900 seen / 100 unseen AOI split (unseen sampled), then further splits by anchor date into:
        train  -> seen AOIs,  anchor_date <  train_end          (temporal train)
        valid  -> seen AOIs,  train_end <= anchor_date < valid_end   (temporal valid)
        test1  -> seen AOIs,  valid_end <= anchor_date < test_end    (temporal-only test)
        test2  -> unseen AOIs, valid_end <= anchor_date < test_end   (spatial-generalization test)
"""

import os
import re
import json
import pandas as pd
import numpy as np


def _infer_tile(aoi_id: str, row: pd.Series) -> str:

    if 'tile' in row.index and pd.notna(row['tile']):
        return str(row['tile'])

    m = re.match(r'^([NS]\d{2}[EW]\d{3})', str(aoi_id))
    if m:
        return m.group(1)

    if 'lat' in row.index and 'lon' in row.index:
        lat_tile = int(np.floor(row['lat']))
        lon_tile = int(np.floor(row['lon']))
        ns = 'N' if lat_tile >= 0 else 'S'
        ew = 'E' if lon_tile >= 0 else 'W'
        return f"{ns}{abs(lat_tile):02d}{ew}{abs(lon_tile):03d}"

    raise ValueError(
        f"Cannot infer tile for aoi_id={aoi_id!r}: no 'tile' column, "
        "no tile-prefix in aoi_id, and no lat/lon columns found. "
        "Add one of these to the CSV, or adjust _infer_tile()."
    )


def split_seen_unseen_aois(df: pd.DataFrame, n_unseen_total: int = 100, seed: int = 42):
    """
    Stratified seen / unseen AOI split.
    """
    rng = np.random.RandomState(seed)

    # one representative row per AOI (tile is assumed constant per AOI)
    first_rows = df.groupby('aoi_id').first()
    tile_map = {aoi_id: _infer_tile(aoi_id, row) for aoi_id, row in first_rows.iterrows()}

    tile_to_aois = {}
    for aoi_id, tile in tile_map.items():
        tile_to_aois.setdefault(tile, []).append(aoi_id)

    total_aois = len(tile_map)
    if n_unseen_total > total_aois:
        raise ValueError(f"n_unseen_total ({n_unseen_total}) exceeds total AOI count ({total_aois})")

    raw = {t: len(a) / total_aois * n_unseen_total for t, a in tile_to_aois.items()}
    base = {t: int(np.floor(v)) for t, v in raw.items()}
    remainder = n_unseen_total - sum(base.values())
    frac_order = sorted(raw.keys(), key=lambda t: (raw[t] - base[t]), reverse=True)
    for t in frac_order[:remainder]:
        base[t] += 1

    seen_aois, unseen_aois = [], []
    for tile, aois in tile_to_aois.items():
        shuffled = list(aois)
        rng.shuffle(shuffled)
        n_unseen_tile = base[tile]
        unseen_aois.extend(shuffled[:n_unseen_tile])
        seen_aois.extend(shuffled[n_unseen_tile:])

    print("Tile-wise seen/unseen split:")
    for tile in sorted(tile_to_aois):
        n_tile = len(tile_to_aois[tile])
        print(f"  {tile}: total={n_tile:4d}, unseen={base[tile]:3d}, seen={n_tile - base[tile]:4d}")
    print(f"TOTAL: {total_aois} AOIs -> seen={len(seen_aois)}, unseen={len(unseen_aois)}")

    return sorted(seen_aois), sorted(unseen_aois), tile_map


def _collect_samples(aoi_groups, aoi_ids, tile_map, start_ts, end_ts,
                      observe_window_days, predict_window_days, min_observe, min_targets):
    """Scan given AOIs once, keep every valid anchor whose date falls in [start_ts, end_ts)."""
    rows = []
    for aoi_id in aoi_ids:
        group = aoi_groups[aoi_id]
        dates = group['date']
        for anchor_idx in range(1, len(group)):
            anchor_date = dates.iloc[anchor_idx]
            if start_ts is not None and anchor_date < start_ts:
                continue
            if end_ts is not None and anchor_date >= end_ts:
                continue

            observe_cutoff = anchor_date - pd.Timedelta(days=observe_window_days)
            observe_mask = (group['date'] > observe_cutoff) & (group['date'] <= anchor_date)
            n_observe = int(observe_mask.sum())

            predict_cutoff = anchor_date + pd.Timedelta(days=predict_window_days)
            predict_mask = (group['date'] > anchor_date) & (group['date'] <= predict_cutoff)
            n_target = int(predict_mask.sum())

            if n_observe < min_observe or n_target < min_targets:
                continue

            rows.append({
                'aoi_id': aoi_id,
                'tile': tile_map[aoi_id],
                'anchor_idx': anchor_idx,
                'anchor_date': anchor_date.date().isoformat(),
                'observe_len': n_observe,
                'target_len': n_target,
            })
    return pd.DataFrame(rows)


def build_and_save_splits(
    csv_path: str,
    output_dir: str,
    train_end: str = '2022-01-01',
    valid_end: str = '2023-01-01',
    test_end: str = '2024-01-01',
    observe_window_days: int = 90,
    predict_window_days: int = 60,
    min_observe: int = 2,
    min_targets: int = 1,
    n_unseen_total: int = 100,
    seed: int = 42,
):

    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])

    seen_aois, unseen_aois, tile_map = split_seen_unseen_aois(
        df, n_unseen_total=n_unseen_total, seed=seed
    )

    aoi_groups = {
        aoi_id: group.sort_values('date').reset_index(drop=True)
        for aoi_id, group in df.groupby('aoi_id')
    }

    train_end_ts = pd.to_datetime(train_end)
    valid_end_ts = pd.to_datetime(valid_end)
    test_end_ts = pd.to_datetime(test_end)

    common = dict(
        observe_window_days=observe_window_days,
        predict_window_days=predict_window_days,
        min_observe=min_observe,
        min_targets=min_targets,
    )
    splits = {
        'train': _collect_samples(aoi_groups, seen_aois, tile_map, None, train_end_ts, **common),
        'valid': _collect_samples(aoi_groups, seen_aois, tile_map, train_end_ts, valid_end_ts, **common),
        'test1': _collect_samples(aoi_groups, seen_aois, tile_map, valid_end_ts, test_end_ts, **common),
        'test2': _collect_samples(aoi_groups, unseen_aois, tile_map, valid_end_ts, test_end_ts, **common),
    }

    print()
    for name, sdf in splits.items():
        out_path = os.path.join(output_dir, f'{name}.csv')
        sdf.to_csv(out_path, index=False)
        n_aois = sdf['aoi_id'].nunique() if len(sdf) else 0
        print(f"{name:6s}: {len(sdf):6d} samples from {n_aois:4d} AOIs -> {out_path}")

    with open(os.path.join(output_dir, 'aoi_membership.json'), 'w') as f:
        json.dump({
            'seen_aois': seen_aois,
            'unseen_aois': unseen_aois,
            'tile_map': tile_map,
            'n_unseen_total': n_unseen_total,
            'seed': seed,
            'train_end': train_end,
            'valid_end': valid_end,
            'test_end': test_end,
        }, f, indent=2)
    print(f"aoi_membership.json -> {os.path.join(output_dir, 'aoi_membership.json')}")

    return splits


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NDVI dataset utilities.")
    subparsers = parser.add_subparsers(dest="command")

    parser.add_argument("--csv", default=None, help="Path to train/valid/test.csv (debug mode)")
    parser.add_argument("--split-start", default=None)
    parser.add_argument("--split-end", default=None)
    parser.add_argument("--num-samples", type=int, default=3)

    p_build = subparsers.add_parser("build-splits", help="One-time 900-seen/100-unseen split builder")
    p_build.add_argument("--csv", required=True, help="Path to the raw, un-split AOI time series CSV")
    p_build.add_argument("--out", required=True, help="Output directory for train/valid/test1/test2 CSVs")
    p_build.add_argument("--train-end", default='2022-01-01')
    p_build.add_argument("--valid-end", default='2023-01-01')
    p_build.add_argument("--test-end", default='2024-01-01')
    p_build.add_argument("--observe-window-days", type=int, default=90)
    p_build.add_argument("--predict-window-days", type=int, default=60)
    p_build.add_argument("--min-observe", type=int, default=2)
    p_build.add_argument("--min-targets", type=int, default=1)
    p_build.add_argument("--n-unseen-total", type=int, default=100)
    p_build.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command == "build-splits":
        build_and_save_splits(
            csv_path=args.csv,
            output_dir=args.out,
            train_end=args.train_end,
            valid_end=args.valid_end,
            test_end=args.test_end,
            observe_window_days=args.observe_window_days,
            predict_window_days=args.predict_window_days,
            min_observe=args.min_observe,
            min_targets=args.min_targets,
            n_unseen_total=args.n_unseen_total,
            seed=args.seed,
        )