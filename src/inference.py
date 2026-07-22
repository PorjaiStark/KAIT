import os
import sys
import shutil

import torch
import pandas as pd
from torch.utils.data import DataLoader


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
USE_AMP = True
USE_COMPILE = True

CHECKPOINT = "outputs/checkpoints/best.pt"
BATCH_SIZE = 64        # no gradients held during inference; raise further if VRAM allows
NUM_WORKERS = 8

NDVI_TIMESERIES_CSV = "/home/takanolab/デスクトップ/KAIT/data/processed/data_before_split/ndvi_timeseries.csv"
WEATHER_ROOT = "/home/takanolab/デスクトップ/kait_observe/weather"
SENTINEL_ROOT = "/home/takanolab/デスクトップ/kait_observe/preprocessing_ex_retiled"

# (split name, split csv) pairs to run inference on -- each produces its own
# outputs/predictions/<name>_prediction.pt, so evaluation.py can be pointed
# at either independently.
TEST_SPLITS = [
    ("test1", "/home/takanolab/デスクトップ/KAIT/data/processed/split/test1.csv"),
    ("test2", "/home/takanolab/デスクトップ/KAIT/data/processed/split/test2.csv"),
]


# Encode
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

# Inference
@torch.no_grad()
def inference(
    model,
    loader,
    encoders,
    time_norm_days
):

    model.eval()

    for encoder in encoders:
        encoder.eval()


    predictions = []
    targets = []
    masks = []

    # geometry-analysis captures: embeddings before/after the transformer
    # encoder and before/after the regression head (see model.py forward())
    pre_transformer_list = []
    transformer_output_list = []
    future_embedding_list = []
    sequence_masks = []
    future_query_masks = []

    # real calendar dates for each future/target timestep, so plots can use
    # a true date axis instead of a plain 0..T index -- see
    # datasets/dataloader.py encode_time(): time_sequence[..., 0] is
    # gap_norm = (date - anchor_date).days / time_norm_days, so the real
    # date is recoverable as anchor_date + gap_norm * time_norm_days days.
    future_dates_list = []

    # observe (input history) window, so sample plots can show it leading
    # up to the future/prediction window instead of starting cold at the
    # anchor date -- same date-recovery trick, gap_days is <= 0 here.
    observe_ndvi_list = []
    observe_dates_list = []


    for step, batch in enumerate(loader):

        print(
            f"Inference batch {step+1}/{len(loader)}",
            flush=True
        )


        # non-tensor fields (plain python lists) -- must be grabbed before
        # the tensor-only filter below, which would otherwise silently drop them
        anchor_dates_batch = batch["anchor_date"]

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
                *encoders
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
        pre_transformer = outputs["pre_transformer"]
        transformer_output = outputs["transformer_output"]
        future_embedding = outputs["future_embedding"]



        # IMPORTANT:
        # keep each sample separately
        # because target length is variable

        for i in range(
            prediction.shape[0]
        ):

            predictions.append(
                prediction[i].float().cpu()
            )


            targets.append(
                batch["target_ndvi"][i].cpu()
            )


            masks.append(
                batch["target_mask"][i].cpu()
            )

            pre_transformer_list.append(
                pre_transformer[i].float().cpu()
            )

            transformer_output_list.append(
                transformer_output[i].float().cpu()
            )

            future_embedding_list.append(
                future_embedding[i].float().cpu()
            )

            sequence_masks.append(
                batch["sequence_mask"][i].cpu()
            )

            future_query_masks.append(
                batch["future_query_mask"][i].cpu()
            )

            gap_norm = batch["time_sequence"][i, :, 0][
                batch["future_query_mask"][i]
            ].float().cpu().numpy()

            anchor_date = pd.Timestamp(anchor_dates_batch[i])

            future_dates_list.append([
                (anchor_date + pd.Timedelta(days=float(g) * time_norm_days)).strftime("%Y-%m-%d")
                for g in gap_norm
            ])

            observe_positions = (
                batch["sequence_mask"][i] & (~batch["future_query_mask"][i])
            )

            observe_ndvi_list.append(
                batch["ndvi_sequence"][i, :, 0][observe_positions].float().cpu()
            )

            observe_gap_norm = batch["time_sequence"][i, :, 0][
                observe_positions
            ].float().cpu().numpy()

            observe_dates_list.append([
                (anchor_date + pd.Timedelta(days=float(g) * time_norm_days)).strftime("%Y-%m-%d")
                for g in observe_gap_norm
            ])


    return (
        predictions,
        targets,
        masks,
        pre_transformer_list,
        transformer_output_list,
        future_embedding_list,
        sequence_masks,
        future_query_masks,
        future_dates_list,
        observe_ndvi_list,
        observe_dates_list,
    )

# Main
def main():

    # Build model
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



    # ======================
    # Load checkpoint
    # ======================


    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    # which training run produced this checkpoint (embedded by train.py
    # since the run-tracking system was added -- see outputs/checkpoints/
    # runs/<run_id>/). Used below to tag saved predictions so they stay
    # tied to this exact checkpoint even if you never retrain.
    checkpoint_run_id = checkpoint.get("run_id")

    def strip_compile_prefix(state_dict):
        # train.py saves state_dict() from a torch.compile-wrapped module for
        # "model" and "sentinel_encoder", which prefixes every key with
        # "_orig_mod." -- strip it so it loads into the plain module here
        # (this file applies torch.compile itself, after loading, below).
        return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}

    model.load_state_dict(
        strip_compile_prefix(checkpoint["model"])
    )


    ndvi_encoder.load_state_dict(
        checkpoint["ndvi_encoder"]
    )


    time_encoder.load_state_dict(
        checkpoint["time_encoder"]
    )


    location_encoder.load_state_dict(
        checkpoint["location_encoder"]
    )


    weather_encoder.load_state_dict(
        checkpoint["weather_encoder"]
    )


    sentinel_encoder.load_state_dict(
        strip_compile_prefix(checkpoint["sentinel_encoder"])
    )



    model.to(DEVICE)


    encoders = [

        ndvi_encoder.to(DEVICE),

        time_encoder.to(DEVICE),

        location_encoder.to(DEVICE),

        weather_encoder.to(DEVICE),

        sentinel_encoder.to(DEVICE)

    ]

    if USE_COMPILE and DEVICE == "cuda":
        model = torch.compile(model, dynamic=True)
        encoders[-1] = torch.compile(encoders[-1], dynamic=True)  # sentinel_encoder



    os.makedirs(
        "outputs/predictions",
        exist_ok=True
    )


    # ======================
    # Run + save, once per split
    # ======================

    for split_name, split_csv in TEST_SPLITS:

        print(f"\n==== Running inference on split: {split_name} ====")

        test_dataset = MultiModalNDVIDataset(
            split_csv=split_csv,
            ndvi_timeseries_csv=NDVI_TIMESERIES_CSV,
            weather_root=WEATHER_ROOT,
            sentinel_root=SENTINEL_ROOT
        )

        loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=multimodal_collate_fn,
            num_workers=NUM_WORKERS,
            pin_memory=DEVICE == "cuda"
        )

        (
            predictions,
            targets,
            masks,
            pre_transformer_list,
            transformer_output_list,
            future_embedding_list,
            sequence_masks,
            future_query_masks,
            future_dates_list,
            observe_ndvi_list,
            observe_dates_list,
        ) = inference(
            model,
            loader,
            encoders,
            test_dataset.time_norm_days
        )

        result = {
            "prediction": predictions,

            "target": targets,

            "mask": masks,

            # geometry-analysis captures (see model.py NDVITransformerModel.forward):
            # pre_transformer  -- fusion+positional-encoding output, before the transformer encoder
            # transformer_output -- full observe+future sequence, after the transformer encoder
            # future_embedding -- future-only tokens, i.e. the regression head's input
            # prediction (above) -- the regression head's output
            "pre_transformer": pre_transformer_list,

            "transformer_output": transformer_output_list,

            "future_embedding": future_embedding_list,

            "sequence_mask": sequence_masks,

            "future_query_mask": future_query_masks,

            # real calendar dates (list[str] per sample, "YYYY-MM-DD"), same
            # order/length as target/mask/prediction for that sample -- lets
            # evaluation.py plot on a true date axis instead of a plain index
            "future_dates": future_dates_list,

            # observe (input history) window -- same idea, lets sample
            # plots show the lead-up to the future/prediction window
            "observe_ndvi": observe_ndvi_list,

            "observe_dates": observe_dates_list,

            # which training run's checkpoint produced this prediction file
            # (None if CHECKPOINT predates run-tracking) -- lets
            # evaluation.py trace results back to that run's config/summary
            "checkpoint_run_id": checkpoint_run_id
        }

        latest_path = f"outputs/predictions/{split_name}_prediction.pt"

        torch.save(result, latest_path)

        if checkpoint_run_id is not None:

            run_pred_dir = os.path.join("outputs/predictions/runs", checkpoint_run_id)
            os.makedirs(run_pred_dir, exist_ok=True)

            run_pred_path = os.path.join(run_pred_dir, f"{split_name}_prediction.pt")
            shutil.copyfile(latest_path, run_pred_path)

            print(f"Also archived to: {run_pred_path}")

        print(f"\nSaved inference result for {split_name}")

        print(
            "Number of samples:",
            len(predictions)
        )

        print(
            "Example prediction shape:",
            predictions[0].shape
        )



if __name__ == "__main__":

    main()