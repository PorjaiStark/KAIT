import os
import sys

sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from unittest.mock import sentinel
from matplotlib import dates
from numpy.random import sample
import torch
import rasterio
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from collate import multimodal_collate_fn

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
    ):

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

        self.weather_cache = {}
        self.sentinel_cache = {}

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
        future_mask = ((aoi_df["date"] > anchor_date)&(aoi_df["date"] <= future_end))
        future = (aoi_df[future_mask].sort_values("date"))

        observe_ndvi = (observe["median_ndvi"].values.astype(np.float32))
        target_ndvi = (future["median_ndvi"].values.astype(np.float32))

        observe_time = self.encode_time(observe["date"], anchor_date)
        future_time = self.encode_time(future["date"],anchor_date)
        location = self.encode_location(aoi_id)

        return_location = np.repeat(location[None, :], len(observe), axis=0)
        
        weather = self.load_weather(aoi_id, observe["date"])
        observe_weather = self.match_weather_dates( weather, observe["date"])
        
        sentinel = self.load_sentinel(aoi_id)
        observe_sentinel = self.match_sentinel_dates( sentinel, observe["date"])
        
        
        return {
            
            "aoi_id": aoi_id,
            "anchor_date": str(anchor_date.date()),
            "observe_ndvi": observe_ndvi[:, None],
            "observe_time": observe_time,
            "observe_location": return_location,
            "future_time": future_time,
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
        if weather_aoi not in self.weather_cache:
            self.weather_cache[weather_aoi] = {}

        weather_data = self.weather_cache[weather_aoi]
        years = pd.to_datetime(dates).dt.year.unique()

        for year in years:

            if year not in weather_data:

                weather_path = os.path.join(self.weather_root, weather_aoi, f"{year}.pt")
                weather_data[year] = torch.load( weather_path, map_location="cpu")

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

        if aoi_id in self.sentinel_cache:
            return self.sentinel_cache[aoi_id]

        sentinel_dir = os.path.join(self.sentinel_root, aoi_id)
        sentinel_path = os.path.join(sentinel_dir, "Allyear_deduped.tif")

        with rasterio.open(sentinel_path) as src:
            image = src.read().astype(np.float32)
            descriptions = list(src.descriptions)
            date_to_slice = {}

            for i in range(0, len(descriptions), 10):
                date = descriptions[i].split("_")[0]
                date_to_slice[date] = slice(i, i + 10)

        sentinel = {"image": image,"date_to_slice": date_to_slice}
        self.sentinel_cache[aoi_id] = sentinel

        return sentinel

    def match_sentinel_dates(self, sentinel, dates):

        image = sentinel["image"]
        date_to_slice = sentinel["date_to_slice"]

        sentinel_images = []

        for date in pd.to_datetime(dates):
            date_str = str(date.date())
            band_slice = date_to_slice[date_str]
            sentinel_images.append(image[band_slice])
        sentinel_images = np.stack(sentinel_images).astype(np.float32)
        sentinel_images = np.nan_to_num(sentinel_images,nan=0.0
                                        )
        return sentinel_images
    

if __name__ == "__main__":

    dataset = MultiModalNDVIDataset(
        split_csv="data/processed/split/train.csv",
        ndvi_timeseries_csv="data/processed/data_before_split/ndvi_timeseries.csv",
        weather_root="/Users/porjai/Library/CloudStorage/GoogleDrive-porjaichavez@gmail.com/My Drive/kait_observe/weather",
        sentinel_root="/Users/porjai/Library/CloudStorage/GoogleDrive-porjaichavez@gmail.com/My Drive/kait_observe/preprocessing_ex")

    for i in range(50):
        sample = dataset[i]

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=multimodal_collate_fn
    )

    batch = next(iter(loader))
    print("NDVI:",batch["observe_ndvi"].shape)
    print("TIME:",batch["observe_time"].shape)
    print("LOCATION:",batch["observe_location"].shape)
    print("WEATHER:",batch["observe_weather"].shape)
    print("SENTINEL:",batch["observe_sentinel"].shape)
    print("MASK:",batch["observe_mask"].shape)
   
    
    