import torch
import numpy as np


def pad_sequence(arrays, max_len):

    batch_size = len(arrays)

    feature_shape = arrays[0].shape[1:]

    output = np.zeros(
        (batch_size, max_len, *feature_shape),
        dtype=np.float32
    )

    mask = np.zeros(
        (batch_size, max_len),
        dtype=bool
    )


    for i, arr in enumerate(arrays):

        length = arr.shape[0]

        output[i, :length] = arr
        mask[i, :length] = True


    return (
        torch.tensor(output, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.bool)
    )



def multimodal_collate_fn(batch):


    # =========================
    # observation
    # =========================

    max_obs_len = max(
        x["observe_len"]
        for x in batch
    )


    observe_ndvi, observe_mask = pad_sequence(
        [
            x["observe_ndvi"]
            for x in batch
        ],
        max_obs_len
    )


    observe_time, _ = pad_sequence(
        [
            x["observe_time"]
            for x in batch
        ],
        max_obs_len
    )


    observe_location, _ = pad_sequence(
        [
            x["observe_location"]
            for x in batch
        ],
        max_obs_len
    )


    observe_weather, _ = pad_sequence(
        [
            x["observe_weather"]
            for x in batch
        ],
        max_obs_len
    )


    observe_sentinel, _ = pad_sequence(
        [
            x["observe_sentinel"]
            for x in batch
        ],
        max_obs_len
    )



    # =========================
    # future query
    # =========================

    max_future_len = max(
        x["target_len"]
        for x in batch
    )


    future_time, future_mask = pad_sequence(
        [
            x["future_time"]
            for x in batch
        ],
        max_future_len
    )


    future_location, _ = pad_sequence(
        [
            x["future_location"]
            for x in batch
        ],
        max_future_len
    )


    target_ndvi, target_mask = pad_sequence(
        [
            x["target_ndvi"][:,None]
            for x in batch
        ],
        max_future_len
    )



    # =========================
    # full transformer sequence
    # =========================


    time_sequence = torch.cat(
        [
            observe_time,
            future_time
        ],
        dim=1
    )


    location_sequence = torch.cat(
        [
            observe_location,
            future_location
        ],
        dim=1
    )


    future_ndvi = torch.zeros_like(
        target_ndvi
    )


    ndvi_sequence = torch.cat(
        [
            observe_ndvi,
            future_ndvi
        ],
        dim=1
    )


    sequence_mask = torch.cat(
        [
            observe_mask,
            future_mask
        ],
        dim=1
    )


    future_query_mask = torch.cat(
        [
            torch.zeros_like(observe_mask),
            future_mask
        ],
        dim=1
    )



    return {

        "ndvi_sequence":
            ndvi_sequence,

        "time_sequence":
            time_sequence,

        "location_sequence":
            location_sequence,

        "sequence_mask":
            sequence_mask,


        "future_query_mask":
            future_query_mask,


        "observe_weather":
            observe_weather,

        "observe_sentinel":
            observe_sentinel,


        "target_ndvi":
            target_ndvi,

        "target_mask":
            target_mask,


        "observe_len":
            max_obs_len,

        "future_len":
            max_future_len,


        "aoi_id":
            [
                x["aoi_id"]
                for x in batch
            ],


        "anchor_date":
            [
                x["anchor_date"]
                for x in batch
            ]
    }