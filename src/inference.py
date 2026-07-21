import os
import sys

import torch
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
    encoders
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


    for step, batch in enumerate(loader):

        print(
            f"Inference batch {step+1}/{len(loader)}",
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


    return (
        predictions,
        targets,
        masks,
        pre_transformer_list,
        transformer_output_list,
        future_embedding_list,
        sequence_masks,
        future_query_masks,
    )

# Main
def main():


    # Dataset
    test_dataset = MultiModalNDVIDataset(

        split_csv=
        "/home/takanolab/デスクトップ/KAIT/data/processed/split/test1.csv",

        ndvi_timeseries_csv=
        "/home/takanolab/デスクトップ/KAIT/data/processed/data_before_split/ndvi_timeseries.csv",

        weather_root=
        "/home/takanolab/デスクトップ/kait_observe/weather",

        sentinel_root=
        "/home/takanolab/デスクトップ/kait_observe/preprocessing_ex_retiled"

    )
    
    loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        collate_fn=multimodal_collate_fn,

        num_workers=NUM_WORKERS,

        pin_memory=DEVICE == "cuda"

    )


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



    # ======================
    # Run
    # ======================


    (
        predictions,
        targets,
        masks,
        pre_transformer_list,
        transformer_output_list,
        future_embedding_list,
        sequence_masks,
        future_query_masks,
    ) = inference(

        model,

        loader,

        encoders

    )



    # ======================
    # Save
    # ======================


    os.makedirs(
        "outputs/predictions",
        exist_ok=True
    )


    torch.save(

        {
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

            "future_query_mask": future_query_masks

        },

        "outputs/predictions/test1_prediction.pt"

    )


    print("\nSaved inference result")

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