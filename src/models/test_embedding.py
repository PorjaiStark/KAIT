import torch
import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
from datasets.dataloader import MultiModalNDVIDataset
from torch.utils.data import DataLoader
from datasets.collate import multimodal_collate_fn

from embedding.ndvi_encoder import NDVIEncoder
from embedding.time_encoder import TimeEncoder
from embedding.location_encoder import LocationEncoder
from embedding.weather_encoder import WeatherEncoder
from embedding.sentinel_encoder import SentinelEncoder

from fusion.concat_fusion import ConcatFusion


dataset = MultiModalNDVIDataset(
        split_csv="data/processed/split/train.csv",
        ndvi_timeseries_csv="data/processed/data_before_split/ndvi_timeseries.csv",
        weather_root="/Users/porjai/Library/CloudStorage/GoogleDrive-porjaichavez@gmail.com/My Drive/kait_observe/weather",
        sentinel_root="/Users/porjai/Library/CloudStorage/GoogleDrive-porjaichavez@gmail.com/My Drive/kait_observe/preprocessing_ex")


loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=multimodal_collate_fn
)


batch = next(iter(loader))


# encoder
ndvi_encoder = NDVIEncoder(16)
time_encoder = TimeEncoder()
location_encoder = LocationEncoder(32)
weather_encoder = WeatherEncoder()
sentinel_encoder = SentinelEncoder()


# embedding
ndvi_emb = ndvi_encoder(
    batch["observe_ndvi"]
)

time_emb = time_encoder(
    batch["observe_time"]
)

location_emb = location_encoder(
    batch["observe_location"]
)

weather_emb = weather_encoder(
    batch["observe_weather"]
)

sentinel_emb = sentinel_encoder(
    batch["observe_sentinel"]
)


print("NDVI", ndvi_emb.shape)
print("TIME", time_emb.shape)
print("LOC", location_emb.shape)
print("WEATHER", weather_emb.shape)
print("SENTINEL", sentinel_emb.shape)


# fusion
fusion = ConcatFusion()

out = fusion(
    ndvi_emb,
    time_emb,
    location_emb,
    weather_emb,
    sentinel_emb
)

print("FUSION", out.shape)