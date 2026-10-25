"""Pull Salt Lake Valley groundwater levels (USGS NWIS) and weather forcing (Open-Meteo)."""
import io
import time
import requests
import pandas as pd

START, END = "2021-01-01", "2025-12-31"
PARAM_DTW = "72019"  # depth to water level, ft below land surface


def list_sites(state="ut"):
    r = requests.get(
        "https://waterservices.usgs.gov/nwis/site/",
        params=dict(format="rdb", stateCd=state, siteType="GW", hasDataTypeCd="dv",
                    parameterCd=PARAM_DTW, siteStatus="active"), timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), sep="\t", comment="#", dtype=str)
    df = df[df.agency_cd != "5s"].reset_index(drop=True)
    df["dec_lat_va"] = df.dec_lat_va.astype(float)
    df["dec_long_va"] = df.dec_long_va.astype(float)
    return df


def fetch_levels(sites):
    rows = []
    for s in sites:
        r = requests.get("https://waterservices.usgs.gov/nwis/dv/", params=dict(
            format="json", sites=s, parameterCd=PARAM_DTW, startDT=START, endDT=END), timeout=120)
        if r.status_code != 200:
            print("skip", s, r.status_code)
            continue
        for ts in r.json()["value"]["timeSeries"]:
            for v in ts["values"][0]["value"]:
                val = float(v["value"])
                if val > -999:
                    rows.append((s, v["dateTime"][:10], val))
    out = pd.DataFrame(rows, columns=["site_no", "date", "dtw_ft"])
    out["date"] = pd.to_datetime(out.date)
    return out


def fetch_weather(lat, lon):
    for wait in (0, 15, 45, 90):
        time.sleep(wait)
        r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=dict(
            latitude=lat, longitude=lon, start_date=START, end_date=END,
            daily="precipitation_sum,et0_fao_evapotranspiration", timezone="America/Denver"), timeout=60)
        if r.status_code != 429:
            break
    r.raise_for_status()
    d = r.json()["daily"]
    return pd.DataFrame({"date": pd.to_datetime(d["time"]), "precip_mm": d["precipitation_sum"],
                         "et0_mm": d["et0_fao_evapotranspiration"]})


if __name__ == "__main__":
    sites = list_sites("ut")
    box = sites[(sites.dec_lat_va.between(40.3, 40.8)) & (sites.dec_long_va.between(-112.2, -111.7))]
    levels = fetch_levels(box.site_no)
    n = levels.groupby("site_no").size()
    keep = n[n > 1200].index
    box = box[box.site_no.isin(keep)]
    levels = levels[levels.site_no.isin(keep)]
    box.to_csv("data/wells.csv", index=False)
    levels.to_csv("data/levels.csv", index=False)
    fetch_weather(40.6, -111.9).to_csv("data/weather.csv", index=False)
    print(box[["site_no", "station_nm", "dec_lat_va", "dec_long_va"]].to_string(index=False))
    print(levels.groupby("site_no").agg(n=("dtw_ft", "size"), mn=("dtw_ft", "min"), mx=("dtw_ft", "max")))
