
## Preprocessed Data

Download the preprocessed (cropped + deduped) Sentinel-2 stacks and weather from Google Drive: https://drive.google.com/drive/folders/1KV7jadN0qmi9seUakgGEccYR8BgRmyxg?usp=sharing

## Dataset Description

This dataset was created to capture paddy (rice) field areas, with
locations referenced against JAXA land cover data. It primarily covers
the **Kansai region of Japan**, containing **1,000 Areas of Interest
(AOIs)**, each covering a **500m x 500m** patch, spanning **7 years
(2019-2025)**.

The data is organized into three main parts:

1. **`weather/`** - Daily weather data from **JMA** (Japan Meteorological Agency).
2. **`sentinel/`** - Raw **Sentinel-2 multispectral imagery** (10 bands),
   exported directly from Google Earth Engine.
3. **`preprocessing_ex/`** - Cleaned version of the Sentinel-2 data (fixing
   issues introduced during the GEE export process), stacked into a
   single file per AOI.

Download dataset from Google Drive:
https://drive.google.com/drive/folders/1KV7jadN0qmi9seUakgGEccYR8BgRmyxg?usp=sharing
