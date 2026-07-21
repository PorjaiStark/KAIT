import os
import sys

from collections import OrderedDict

import torch
import rasterio
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from .collate import multimodal_collate_fn

# Sentinel tifs were originally pixel-interleaved with a 256x256 block size
# on a 50x50 image, so reading even one band forced GDAL to decompress all
# ~2000 bands of the single block covering the whole image at once --
# measured at ~600MB per open+read. preprocessing/retile_sentinel.py
# re-encodes them as band-interleaved (see sentinel_root in train.py,
# which now points at preprocessing_ex_retiled), cutting that to ~1MB per
# read since only the requested bands are touched. Verified directly:
# ~600MB warmed handle before -> ~1MB after, on the retiled files. This cap
# is now cheap headroom rather than a tight memory bound; raise further if
# profiling shows cache misses are still costing meaningful time.
MAX_OPEN_SENTINEL_HANDLES = 64

# weather_cache mirrors sentinel_cache's LRU cap: without one, each
# persistent worker accumulates every AOI-year it has ever touched for the
# life of the run. Individually small (~50KB/AOI-year) but unbounded growth
# across NUM_WORKERS persistent workers over a multi-epoch run adds up, and
# it's the same class of leak the sentinel cache was already capped for.
MAX_CACHED_WEATHER_AOIS = 64


class MultiModalNDVIDataset(Dataset):

    def __init__(
        self,
        split_csv,
        ndvi_timeseries_csv,
        weather_root,
        sentinel_root,
        observe_window_days=90,
        predict_window_days=60,
        time_norm_days=90,
        max_bracket_span_days=40,
    ):
        """
        ndvi_timeseries_csv must have been run through
        preprocessing/whittaker_smooth_ndvi.py, which adds two columns:
          - ndvi_smoothed: denoised NDVI (used for targets only)
          - bracket_span: gap (days) between the nearest real
            observations before/after each point (used to mask out
            targets where the smoothed value isn't trustworthy)

        Input (observe) sequence always uses the raw median_ndvi column
        at real dates -- only the target/answer is smoothed.
        """

        self.samples = pd.read_csv(split_csv)
        self.samples["anchor_date"] = pd.to_datetime(
            self.samples["anchor_date"]
        )

        df = pd.read_csv(ndvi_timeseries_csv)
        df["date"] = pd.to_datetime(df["date"])

        # depend on AOI groups and sort date
        self.aoi_groups = {
            aoi:
            g.sort_values("date").reset_index(drop=True)
            for aoi,g in df.groupby("aoi_id")
        }

        self.weather_root = weather_root
        self.sentinel_root = sentinel_root
        self.observe_window_days = observe_window_days
        self.predict_window_days = predict_window_days
        self.time_norm_days = time_norm_days
        self.max_bracket_span_days = max_bracket_span_days

        self.weather_cache = OrderedDict()
        self.sentinel_cache = OrderedDict()

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples.iloc[idx]
        
        aoi_id = sample["aoi_id"]
        anchor_date = sample["anchor_date"]
        aoi_df = self.aoi_groups[aoi_id]
        
        observe_start = anchor_date - pd.Timedelta(days=self.observe_window_days)
        observe_mask = ((aoi_df["date"] > observe_start)&(aoi_df["date"] <= anchor_date))
        observe = (aoi_df[observe_mask].sort_values("date"))
        
        future_end = (anchor_date + pd.Timedelta(days=self.predict_window_days))
        future_mask = (
            (aoi_df["date"] > anchor_date)
            & (aoi_df["date"] <= future_end)
            & (aoi_df["bracket_span"] <= self.max_bracket_span_days)
        )
        future = (aoi_df[future_mask].sort_values("date"))

        observe_ndvi = (observe["median_ndvi"].values.astype(np.float32))
        target_ndvi = (future["ndvi_smoothed"].values.astype(np.float32))

        observe_time = self.encode_time(observe["date"], anchor_date)
        future_time = self.encode_time(future["date"],anchor_date)
        location = self.encode_location(aoi_id)

        observe_location = np.repeat(location[None, :],len(observe), axis=0)
        future_location = np.repeat(location[None, :],len(future), axis=0)
        
        weather = self.load_weather(aoi_id, observe["date"])
        observe_weather = self.match_weather_dates( weather, observe["date"])
        
        sentinel = self.load_sentinel(aoi_id)
        observe_sentinel = self.match_sentinel_dates( sentinel, observe["date"])
        
        
        return {
            
            "aoi_id": aoi_id,
            "anchor_date": str(anchor_date.date()),

            "observe_ndvi": observe_ndvi[:, None],

            "observe_time": observe_time,
            "future_time": future_time,

            "observe_location": observe_location,
            "future_location": future_location,

            "target_ndvi": target_ndvi,

            "observe_len": len(observe),
            "target_len": len(future),

            "observe_weather": observe_weather,
            "observe_sentinel": observe_sentinel
        }
    
    def encode_time(self, dates, anchor_date):
        """
        Create temporal features
        output:
        [ gap_norm, month_sin, month_cos, year_norm]
        """
        dates = pd.to_datetime(dates)
        
        gap_days = np.array(
            [(d - anchor_date).days for d in dates],dtype=np.float32
        )
        gap_norm = (gap_days / self.time_norm_days)
        months = dates.dt.month.values
        month_sin = np.sin(2 * np.pi * months / 12).astype(np.float32)
        month_cos = np.cos(2 * np.pi * months / 12).astype(np.float32)
        years = dates.dt.year.values.astype(np.float32)
        year_norm = (years - 2019) / (2025 - 2019)
        time_features = np.stack([ gap_norm, month_sin, month_cos, year_norm],axis=-1).astype(np.float32)
          
        return time_features
    
    def encode_location(self, aoi_id):

        coords = aoi_id.replace("s2_", "").split("_")

        lat = np.deg2rad(float(coords[0]))
        lon = np.deg2rad(float(coords[1]))

        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)

        return np.array([x,y,z],dtype=np.float32)
    
    def load_weather(self, aoi_id, dates):

        weather_aoi = aoi_id.replace("s2_", "")
        if weather_aoi in self.weather_cache:
            self.weather_cache.move_to_end(weather_aoi)
        else:
            self.weather_cache[weather_aoi] = {}

        weather_data = self.weather_cache[weather_aoi]
        years = pd.to_datetime(dates).dt.year.unique()

        for year in years:

            if year not in weather_data:

                weather_path = os.path.join(self.weather_root, weather_aoi, f"{year}.pt")
                weather_data[year] = torch.load( weather_path, map_location="cpu")

        while len(self.weather_cache) > MAX_CACHED_WEATHER_AOIS:
            self.weather_cache.popitem(last=False)

        return weather_data

    def match_weather_dates(self, weather_data, dates):
        weather_features = []

        for d in pd.to_datetime(dates):
            year = d.year
            if year not in weather_data:
                weather_features.append(np.zeros(30, dtype=np.float32))
                continue

            year_data = weather_data[year]
            weather_tensor = year_data["weather"]     #(N, 30)
            weather_dates = year_data["dates"]        #string
            date_to_idx = {
                date: idx
                for idx, date in enumerate(weather_dates)
            }
            date_str = d.strftime("%Y-%m-%d")
            
            if date_str in date_to_idx:
                idx = date_to_idx[date_str]
                weather_features.append(
                    weather_tensor[idx].numpy()
                )
            else:

                weather_features.append(
                    np.zeros(weather_tensor.shape[1], dtype=np.float32)
                )

        return np.stack(weather_features).astype(np.float32)
    
    def load_sentinel(self, aoi_id):
        """
        Cache an OPEN rasterio handle per AOI (not the decompressed pixel
        array), LRU-bounded at MAX_OPEN_SENTINEL_HANDLES. An open GDAL
        dataset keeps internal decompression/block-cache buffers alive
        even though we only read metadata here, so leaving this unbounded
        let every worker accumulate handles for every AOI it ever touched
        (up to ~900) and never release them -- fine within one epoch, but
        additive with a second DataLoader's fresh workers (e.g. validation
        starting while train's persistent_workers are still resident),
        which is what caused an OOM kill right at that transition.
        Data is local now, so reopening on a cache miss is cheap.
        """

        if aoi_id in self.sentinel_cache:
            self.sentinel_cache.move_to_end(aoi_id)
            return self.sentinel_cache[aoi_id]

        sentinel_dir = os.path.join(self.sentinel_root, aoi_id)
        sentinel_path = os.path.join(sentinel_dir, "Allyear_deduped.tif")

        src = rasterio.open(sentinel_path)
        descriptions = list(src.descriptions)
        date_to_slice = {}

        for i in range(0, len(descriptions), 10):
            date = descriptions[i].split("_")[0]
            date_to_slice[date] = slice(i, i + 10)

        sentinel = {"handle": src, "date_to_slice": date_to_slice}
        self.sentinel_cache[aoi_id] = sentinel

        while len(self.sentinel_cache) > MAX_OPEN_SENTINEL_HANDLES:
            _, evicted = self.sentinel_cache.popitem(last=False)
            evicted["handle"].close()

        return sentinel

    def match_sentinel_dates(self, sentinel, dates):

        src = sentinel["handle"]
        date_to_slice = sentinel["date_to_slice"]

        sentinel_images = []

        for date in pd.to_datetime(dates):
            date_str = str(date.date())
            band_slice = date_to_slice[date_str]
            # rasterio band indexes are 1-based
            indexes = list(range(band_slice.start + 1, band_slice.stop + 1))
            sentinel_images.append(src.read(indexes=indexes))

        sentinel_images = np.stack(sentinel_images).astype(np.float32)
        sentinel_images = np.nan_to_num(sentinel_images, nan=0.0)
        return sentinel_images
    

if __name__ == "__main__":

    dataset = MultiModalNDVIDataset(
        split_csv="data/processed/split/train.csv",
        ndvi_timeseries_csv="data/processed/data_before_split/ndvi_timeseries.csv",
        weather_root="/Users/porjai/Library/CloudStorage/GoogleDrive-porjaichavez@gmail.com/My Drive/kait_observe/weather",
        sentinel_root="/Users/porjai/Library/CloudStorage/GoogleDrive-porjaichavez@gmail.com/My Drive/kait_observe/preprocessing_ex"
    )
    


    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=multimodal_collate_fn
    )


    batch = next(iter(loader))

   
    
    