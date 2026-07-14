import io
import json
import time
import requests
import pandas as pd
import torch
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path

DRIVE_ROOT   = Path.home() / "Library/CloudStorage/GoogleDrive-porjaichavez@gmail.com/My Drive"
OUTPUT_ROOT  = DRIVE_ROOT / "kait_observe" / "weather"
AOI_CSV      = Path("data/AOI_list.csv")
STATIONS_CSV = Path("data/jma_located_original.csv")
PROGRESS_FILE = Path("data/weather_progress.json")
START_YEAR   = 2019
END_YEAR     = 2025
REQUEST_DELAY = 1.5  # seconds between request

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

COLUMN_MAPPING = {
    "日_日_日_日": "day",
    "気圧(hPa)_現地_平均_平均": "avg_station_pressure_hpa",
    "気圧(hPa)_海面_平均_平均": "avg_sea_level_pressure_hpa",
    "降水量(mm)_降水量(mm)_合計_合計": "total_precipitation_mm",
    "降水量(mm)_降水量(mm)_最大_1時間": "max_hourly_precipitation_mm",
    "降水量(mm)_降水量(mm)_最大_10分間": "max_10min_precipitation_mm",
    "気温(℃)_気温(℃)_平均_平均": "avg_temperature_c",
    "気温(℃)_気温(℃)_最高_最高": "max_temperature_c",
    "気温(℃)_気温(℃)_最低_最低": "min_temperature_c",
    "湿度(％)_湿度(％)_平均_平均": "avg_humidity_percent",
    "湿度(％)_湿度(％)_最小_最小": "min_humidity_percent",
    "風向・風速(m/s)_風向・風速(m/s)_平均 風速_平均 風速": "avg_wind_speed_ms",
    "風向・風速(m/s)_風向・風速(m/s)_最大風速_風速": "max_wind_speed_ms",
    "風向・風速(m/s)_風向・風速(m/s)_最大風速_風向": "wind_direction",
    "風向・風速(m/s)_風向・風速(m/s)_最大瞬間風速_風速": "max_gust_speed_ms",
    "風向・風速(m/s)_風向・風速(m/s)_最大瞬間風速_風向": "gust_direction",
    "日照 時間 (h)_日照 時間 (h)_日照 時間 (h)_日照 時間 (h)": "sunshine_hours",
    "雪(cm)_雪(cm)_降雪_合計": "total_snowfall_cm",
    "雪(cm)_雪(cm)_最深積雪_値": "max_snow_depth_cm",
    "天気概況_天気概況_昼 (06:00-18:00)_昼 (06:00-18:00)": "daytime_weather_code",
    "天気概況_天気概況_夜 (18:00-翌日06:00)_夜 (18:00-翌日06:00)": "nighttime_weather_code",
}

KEEP_COLS = [
    "date", "avg_station_pressure_hpa", "total_precipitation_mm",
    "avg_temperature_c", "avg_humidity_percent", "avg_wind_speed_ms", "sunshine_hours",
]
FEATURES = [c for c in KEEP_COLS if c != "date"]
# ──────────────────────────────────────────


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


def find_nearest_stations(lat, lon, stations, top_n=5):
    s = stations.copy()
    s["distance_km"] = s.apply(lambda r: haversine(lat, lon, r["lat"], r["lon"]), axis=1)
    return s.sort_values("distance_km").head(top_n)


def probe_station(prec_no, block_no):
    url = (
        "https://ds.data.jma.go.jp/obd/stats/etrn/view/daily_s1.php"
        f"?prec_no={int(prec_no)}&block_no={int(block_no)}&year=2023&month=6&day=1&view="
    )
    try:
        time.sleep(REQUEST_DELAY)
        res = requests.get(url, headers=HEADERS, timeout=15)
        df = pd.read_html(io.StringIO(res.text))[0]
        return len(df) > 5
    except Exception:
        return False


def download_and_process(prec_no, block_no):
    all_data = []
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            url = (
                "https://ds.data.jma.go.jp/obd/stats/etrn/view/daily_s1.php"
                f"?prec_no={int(prec_no)}&block_no={int(block_no)}"
                f"&year={year}&month={month}&day=1&view="
            )
            print(f"  Downloading {year}-{month:02d}", end="\r")
            try:
                time.sleep(REQUEST_DELAY)
                res = requests.get(url, headers=HEADERS, timeout=15)
                df = pd.read_html(io.StringIO(res.text))[0]
                df.columns = [
                    "_".join(str(x).replace("\xa0","").strip() for x in col if str(x)!="nan")
                    for col in df.columns
                ]
                first_col = df.columns[0]
                df = df[pd.to_numeric(df[first_col], errors="coerce").notna()].copy()
                df["day"]   = pd.to_numeric(df[first_col], errors="coerce").astype(int)
                df["year"]  = year
                df["month"] = month
                all_data.append(df)
            except Exception as e:
                print(f"  Error {year}-{month:02d}: {e}")

    if not all_data:
        return None

    df = pd.concat(all_data, ignore_index=True)
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]))
    df = df.sort_values("date")
    for col in df.columns:
        if col not in {"date","year","month","day"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.rename(columns=COLUMN_MAPPING)
    df = df[[c for c in KEEP_COLS if c in df.columns]]
    df = df.interpolate(method="linear").bfill().ffill()
    for col in FEATURES:
        for lag in [7, 14]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        df[f"{col}_roll7_mean"] = df[col].rolling(7).mean()
        df[f"{col}_roll7_std"]  = df[col].rolling(7).std()
    return df.dropna().reset_index(drop=True)


def save_by_year(df, out_dir, meta):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    weather_cols = [c for c in df.columns if c != "date"]
    for year in range(START_YEAR, END_YEAR + 1):
        df_y = df[df["date"].dt.year == year].reset_index(drop=True)
        if df_y.empty:
            continue
        torch.save({
            "weather":      torch.tensor(df_y[weather_cols].values, dtype=torch.float32),
            "dates":        df_y["date"].dt.strftime("%Y-%m-%d").tolist(),
            "weather_cols": weather_cols,
        }, out_dir / f"{year}.pt")


def load_progress():
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed": [], "station_cache": {}}


def save_progress(p):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(p, indent=2))


aoi_df   = pd.read_csv(AOI_CSV)
stations = pd.read_csv(STATIONS_CSV)
stations = stations[stations["station_type"] == "官"].copy()
stations = stations[
    (stations["lat"] >= 35.0) & (stations["lat"] <= 37.0) &
    (stations["lon"] >= 139.0) & (stations["lon"] <= 141.0)
].copy()
print(f"Stations in tile area: {len(stations)}")
print(f"AOI list: {len(aoi_df)} locations\n")

progress = load_progress()
completed = set(progress["completed"])


# station_cache: key="prec_no_block_no" → df (in-memory only, reloaded each run)
station_df_cache = {}

for i, row in aoi_df.iterrows():
    lat = row["lat"]
    lon = row["lon"]
    aoi_id = int(row["id"])
    folder_name = f"{lat}_{lon}"

    if aoi_id in completed:
        print(f"[{i+1}/{len(aoi_df)}] id={aoi_id} — already done, skip")
        continue

    print(f"\n[{i+1}/{len(aoi_df)}] id={aoi_id} ({lat}, {lon})")

    candidates = find_nearest_stations(lat, lon, stations, top_n=5)

    df = None
    nearest = None

    for _, cand in candidates.iterrows():
        cache_key = f"{int(cand['prec_no'])}_{int(cand['block_no'])}"

        # if station already downloaded, use cache
        if cache_key in station_df_cache:
            print(f"  Using cached: {cand['station_name']} ({cand['distance_km']:.2f} km) ⚡")
            df = station_df_cache[cache_key]
            nearest = cand
            break

        print(f"  Trying: {cand['station_name']} ({cand['distance_km']:.2f} km)")
        if not probe_station(cand["prec_no"], cand["block_no"]):
            print(f"    Probe failed")
            continue

        df_try = download_and_process(cand["prec_no"], cand["block_no"])
        if df_try is None or len(df_try) < 100:
            print(f"    Download failed or too few rows")
            continue

        station_df_cache[cache_key] = df_try
        df = df_try
        nearest = cand
        break

    if df is None:
        print(f"  [SKIP] No valid station found")
        continue

    meta = {
        "lat": lat, "lon": lon,
        "station_name": nearest["station_name"],
        "prec_no": int(nearest["prec_no"]),
        "block_no": int(nearest["block_no"]),
        "distance_km": round(float(nearest["distance_km"]), 2),
    }

    out_dir = OUTPUT_ROOT / folder_name
    save_by_year(df, out_dir, meta)
    print(f"  Saved → {folder_name}")

    completed.add(aoi_id)
    if (i + 1) % 10 == 0:
        progress["completed"] = sorted(completed)
        save_progress(progress)
        print(f"  (progress saved: {len(completed)} done)")

progress["completed"] = sorted(completed)
save_progress(progress)
print(f"\nDone! {len(completed)}/{len(aoi_df)} AOI completed.")
