import os
import sys

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb


ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, ROOT)

from src.datasets.dataloader import MultiModalNDVIDataset
from src.datasets.collate import multimodal_collate_fn

from src.models.model import NDVITransformerModel
from src.models.embedding.ndvi_encoder import NDVIEncoder
from src.models.embedding.time_encoder import TimeEncoder
from src.models.embedding.location_encoder import LocationEncoder
from src.models.embedding.weather_encoder import WeatherEncoder
from src.models.embedding.sentinel_encoder import SentinelEncoder
from src.losses.loss import BaselineDynamicsMAELoss

# Config
EPOCHS = 20  # full-data run: no LR scheduler, so best.pt (checkpointed on best
             # valid_loss every epoch) is what protects against a bad late epoch --
             # extra epochs just cost time, not model quality, if training plateaus early
LR = 1e-4
BATCH_SIZE = 32  # sentinel chips are 50x50 (tiny for ResNet18); watch nvidia-smi, likely room to push to 128+
NUM_WORKERS = 8  # sentinel tifs are now re-tiled band-interleaved (see sentinel_root
                 # above and MAX_OPEN_SENTINEL_HANDLES in datasets/dataloader.py), so a
                 # cached handle costs ~1MB instead of the ~600MB that OOM-killed earlier
                 # runs at this worker count. Watch `free -h` if you push this higher.
USE_AMP = True
USE_COMPILE = True


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

AUTOCAST_DTYPE = torch.bfloat16

# Encode batch
def encode_batch(
    batch,
    ndvi_encoder,
    time_encoder,
    location_encoder,
    weather_encoder,
    sentinel_encoder
):

    ndvi_emb = ndvi_encoder(
        batch["ndvi_sequence"]
    )


    time_emb = time_encoder(
        batch["time_sequence"]
    )


    location_emb = location_encoder(
        batch["location_sequence"]
    )


    weather_emb = weather_encoder(
        batch["observe_weather"]
    )


    sentinel_emb = sentinel_encoder(
        batch["observe_sentinel"]
    )


    return (
        ndvi_emb,
        time_emb,
        location_emb,
        weather_emb,
        sentinel_emb
    )

# Train
def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    ndvi_encoder,
    time_encoder,
    location_encoder,
    weather_encoder,
    sentinel_encoder
):

    model.train()

    ndvi_encoder.train()
    time_encoder.train()
    location_encoder.train()
    weather_encoder.train()
    sentinel_encoder.train()


    total_loss = 0
    total_baseline_loss = 0
    total_dynamics_loss = 0


    for step, batch in enumerate(loader):


        print(
            f"Batch {step+1}/{len(loader)}",
            flush=True
        )


        batch = {
            k:v.to(DEVICE)
            for k,v in batch.items()
            if torch.is_tensor(v)
        }



        with torch.autocast(
            device_type=DEVICE,
            dtype=AUTOCAST_DTYPE,
            enabled=USE_AMP and DEVICE == "cuda"
        ):

            (
                ndvi_emb,
                time_emb,
                location_emb,
                weather_emb,
                sentinel_emb
            ) = encode_batch(
                batch,
                ndvi_encoder,
                time_encoder,
                location_encoder,
                weather_encoder,
                sentinel_encoder
            )



            outputs = model(
                ndvi_emb,
                time_emb,
                location_emb,
                weather_emb,
                sentinel_emb,
                batch["time_sequence"][:, :, 0],
                batch["future_query_mask"],
                batch["sequence_mask"]
            )

            prediction = outputs["prediction"]

            loss, components = criterion.forward_with_components(
                prediction,
                batch["target_ndvi"],
                batch["target_mask"]
            )


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        total_loss += loss.item()
        total_baseline_loss += components["baseline_loss"]
        total_dynamics_loss += components["dynamics_loss"]


        print(
            f"loss={loss.item():.5f} "
            f"(baseline={components['baseline_loss']:.5f}, "
            f"dynamics={components['dynamics_loss']:.5f})",
            flush=True
        )


    return (
        total_loss / len(loader),
        total_baseline_loss / len(loader),
        total_dynamics_loss / len(loader)
    )


# Validation
@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    ndvi_encoder,
    time_encoder,
    location_encoder,
    weather_encoder,
    sentinel_encoder
):

    model.eval()

    ndvi_encoder.eval()
    time_encoder.eval()
    location_encoder.eval()
    weather_encoder.eval()
    sentinel_encoder.eval()


    total_loss = 0
    total_baseline_loss = 0
    total_dynamics_loss = 0


    for batch in loader:


        batch = {
            k:v.to(DEVICE)
            for k,v in batch.items()
            if torch.is_tensor(v)
        }



        with torch.autocast(
            device_type=DEVICE,
            dtype=AUTOCAST_DTYPE,
            enabled=USE_AMP and DEVICE == "cuda"
        ):

            (
                ndvi_emb,
                time_emb,
                location_emb,
                weather_emb,
                sentinel_emb
            ) = encode_batch(
                batch,
                ndvi_encoder,
                time_encoder,
                location_encoder,
                weather_encoder,
                sentinel_encoder
            )

            outputs = model(
                ndvi_emb,
                time_emb,
                location_emb,
                weather_emb,
                sentinel_emb,
                batch["time_sequence"][:, :, 0],
                batch["future_query_mask"],
                batch["sequence_mask"]
            )

            prediction = outputs["prediction"]

            loss, components = criterion.forward_with_components(
                prediction,
                batch["target_ndvi"],
                batch["target_mask"]
            )


        total_loss += loss.item()
        total_baseline_loss += components["baseline_loss"]
        total_dynamics_loss += components["dynamics_loss"]



    return (
        total_loss / len(loader),
        total_baseline_loss / len(loader),
        total_dynamics_loss / len(loader)
    )

# Main
def main():


    wandb.init(
        project="ndvi-transformer",
        config={
            "epochs": EPOCHS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "loss": "BaselineDynamicsMAELoss",
            "lambda_dynamics": 3.0,
            "lambda_baseline": 1.0,
            "device": DEVICE,
            "amp": USE_AMP,
            "compile": USE_COMPILE
        }
    )

    # Dataset
    train_dataset = MultiModalNDVIDataset(
        split_csv="/home/takanolab/デスクトップ/KAIT/data/processed/split/train.csv",
        ndvi_timeseries_csv="/home/takanolab/デスクトップ/KAIT/data/processed/data_before_split/ndvi_timeseries.csv",
        weather_root="/home/takanolab/デスクトップ/kait_observe/weather",
        sentinel_root="/home/takanolab/デスクトップ/kait_observe/preprocessing_ex_retiled"
    )

    valid_dataset = MultiModalNDVIDataset(
        split_csv="/home/takanolab/デスクトップ/KAIT/data/processed/split/valid.csv",
        ndvi_timeseries_csv="/home/takanolab/デスクトップ/KAIT/data/processed/data_before_split/ndvi_timeseries.csv",
        weather_root="/home/takanolab/デスクトップ/kait_observe/weather",
        sentinel_root="/home/takanolab/デスクトップ/kait_observe/preprocessing_ex_retiled"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=multimodal_collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE == "cuda",
        persistent_workers=NUM_WORKERS > 0
    )



    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=multimodal_collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE == "cuda",
        persistent_workers=NUM_WORKERS > 0
    )



    # Model + Encoder
    model = NDVITransformerModel(
        d_model=256
    )


    ndvi_encoder = NDVIEncoder(
        out_dim=64
    )

    time_encoder = TimeEncoder(
        out_dim=64
    )

    location_encoder = LocationEncoder(
        out_dim=32
    )

    weather_encoder = WeatherEncoder(
        out_dim=128
    )

    sentinel_encoder = SentinelEncoder(
        out_dim=128
    )



    model.to(DEVICE)

    ndvi_encoder.to(DEVICE)
    time_encoder.to(DEVICE)
    location_encoder.to(DEVICE)
    weather_encoder.to(DEVICE)
    sentinel_encoder.to(DEVICE)

    if USE_COMPILE and DEVICE == "cuda":
        model = torch.compile(model, dynamic=True)
        sentinel_encoder = torch.compile(sentinel_encoder, dynamic=True)



    criterion = BaselineDynamicsMAELoss()



    optimizer = optim.Adam(
        list(model.parameters())
        +
        list(ndvi_encoder.parameters())
        +
        list(time_encoder.parameters())
        +
        list(location_encoder.parameters())
        +
        list(weather_encoder.parameters())
        +
        list(sentinel_encoder.parameters()),

        lr=LR
    )



    best_loss = float("inf")



    for epoch in range(EPOCHS):


        print(
            f"\nEpoch {epoch+1}/{EPOCHS}"
        )


        train_loss, train_baseline_loss, train_dynamics_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            ndvi_encoder,
            time_encoder,
            location_encoder,
            weather_encoder,
            sentinel_encoder
        )



        valid_loss, valid_baseline_loss, valid_dynamics_loss = evaluate(
            model,
            valid_loader,
            criterion,
            ndvi_encoder,
            time_encoder,
            location_encoder,
            weather_encoder,
            sentinel_encoder
        )



        print(
            f"""
Train total: {train_loss:.5f} (baseline={train_baseline_loss:.5f}, dynamics={train_dynamics_loss:.5f})
Valid total: {valid_loss:.5f} (baseline={valid_baseline_loss:.5f}, dynamics={valid_dynamics_loss:.5f})
"""
        )



        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_baseline_loss": train_baseline_loss,
                "train_dynamics_loss": train_dynamics_loss,
                "valid_loss": valid_loss,
                "valid_baseline_loss": valid_baseline_loss,
                "valid_dynamics_loss": valid_dynamics_loss
            }
        )



        if valid_loss < best_loss:


            best_loss = valid_loss


            os.makedirs(
                "outputs/checkpoints",
                exist_ok=True
            )


            torch.save(
                {
                    "model":
                        model.state_dict(),

                    "ndvi_encoder":
                        ndvi_encoder.state_dict(),

                    "time_encoder":
                        time_encoder.state_dict(),

                    "location_encoder":
                        location_encoder.state_dict(),

                    "weather_encoder":
                        weather_encoder.state_dict(),

                    "sentinel_encoder":
                        sentinel_encoder.state_dict(),

                    "epoch":
                        epoch,

                    "valid_loss":
                        valid_loss
                },

                "outputs/checkpoints/best.pt"
            )


            print(
                "Saved best checkpoint"
            )



    wandb.finish()



if __name__ == "__main__":
    main()
