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
        torch.tensor(output),
        torch.tensor(mask)
    )



def multimodal_collate_fn(batch):

    observe_lengths = [
        x["observe_len"]
        for x in batch
    ]

    max_len = max(observe_lengths)


    observe_ndvi, ndvi_mask = pad_sequence(
        [
            x["observe_ndvi"]
            for x in batch
        ],
        max_len
    )


    observe_time, _ = pad_sequence(
        [
            x["observe_time"]
            for x in batch
        ],
        max_len
    )


    observe_location, _ = pad_sequence(
        [
            x["observe_location"]
            for x in batch
        ],
        max_len
    )


    observe_weather, _ = pad_sequence(
        [
            x["observe_weather"]
            for x in batch
        ],
        max_len
    )


    observe_sentinel, _ = pad_sequence(
        [
            x["observe_sentinel"]
            for x in batch
        ],
        max_len
    )


    return {

        "observe_ndvi":
            observe_ndvi,


        "observe_time":
            observe_time,


        "observe_location":
            observe_location,


        "observe_weather":
            observe_weather,


        "observe_sentinel":
            observe_sentinel,


        "observe_mask":
            ndvi_mask,


        "future_time":
            [
                torch.tensor(x["future_time"])
                for x in batch
            ],


        "target_ndvi":
            [
                torch.tensor(x["target_ndvi"])
                for x in batch
            ],


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