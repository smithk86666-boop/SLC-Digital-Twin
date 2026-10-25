"""Pull Platte River alluvial aquifer data near Ashland, Nebraska.

Sources (all free, no keys):
  * USGS NWIS daily groundwater levels (72019, depth to water, ft below land surface)
  * USGS NWIS daily Platte River stage / discharge at 06801000 (Ashland)
  * Open-Meteo archive: daily precipitation and reference ET
"""
import datetime as dt
import io
import time

import pandas as pd
import requests

START = "2021-01-01"
END = dt.date.today().isoformat()
BOX = dict(lat=(40.90, 41.20), lon=(-96.50, -96.10))   # cluster search window
GAUGE = "06801000"
MIN_DAYS = 1200
NWIS = "https://waterservices.usgs.gov/nwis"


def _get(url, params, tries=4):
    for i in range(tries):
        r = requests.get(url, params=params, timeout=120)
        if r.status_code not in (429, 500, 502, 503):
            return r
        time.sleep(15 * 3 ** i)
    return r


def list_sites(state="ne"):
    r = _get(f"{NWIS}/site/", dict(format="rdb", stateCd=state, siteType="GW", hasDataTypeCd="dv",
                                   parameterCd="72019", siteStatus="active"))
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), sep="\t", comment="#", dtype=str)
    df = df[df.agency_cd != "5s"].reset_index(drop=True)
    for c in ("dec_lat_va", "dec_long_va", "alt_va"):
        df[c] = df[c].astype(float)
    return df


def _dv(site, param, start=START, end=END):
    r = _get(f"{NWIS}/dv/", dict(format="json", sites=site, parameterCd=param, startDT=start, endDT=end))
    if r.status_code != 200:
        return pd.DataFrame(columns=["date", "value"])
    rows = []
    for ts in r.json()["value"]["timeSeries"]:
        for v in ts["values"][0]["value"]:
            x = float(v["value"])
            if x > -999:
                rows.append((v["dateTime"][:10], x))
    out = pd.DataFrame(rows, columns=["date", "value"])
    out["date"] = pd.to_datetime(out.date)
    return out.drop_duplicates("date")


def fetch_levels(site_nos):
    frames = []
    for s in site_nos:
        d = _dv(s, "72019")
        d["site_no"] = s
        frames.append(d.rename(columns={"value": "dtw_ft"}))
    return pd.concat(frames, ignore_index=True)[["site_no", "date", "dtw_ft"]]


def fetch_river():
    stage = _dv(GAUGE, "00065").rename(columns={"value": "stage_ft"})
    q = _dv(GAUGE, "00060").rename(columns={"value": "q_cfs"})
    return stage.merge(q, on="date", how="outer").sort_values("date")


def fetch_weather(lat, lon):
    r = _get("https://archive-api.open-meteo.com/v1/archive", dict(
        latitude=lat, longitude=lon, start_date=START, end_date=END,
        daily="precipitation_sum,et0_fao_evapotranspiration", timezone="America/Chicago"))
    r.raise_for_status()
    d = r.json()["daily"]
    return pd.DataFrame({"date": pd.to_datetime(d["time"]), "precip_mm": d["precipitation_sum"],
                         "et0_mm": d["et0_fao_evapotranspiration"]})


if __name__ == "__main__":
    sites = list_sites("ne")
    sites = sites[sites.dec_lat_va.between(*BOX["lat"]) & sites.dec_long_va.between(*BOX["lon"])]
    lv = fetch_levels(sites.site_no)
    g = lv.groupby("site_no").date.agg(["max", "count"])
    live = g[(g["count"] > MIN_DAYS) & (g["max"] >= pd.Timestamp.today() - pd.Timedelta(days=21))].index
    sites = sites[sites.site_no.isin(live)]
    lv = lv[lv.site_no.isin(live)]
    sites.to_csv("data/wells.csv", index=False)
    lv.to_csv("data/levels.csv", index=False)
    fetch_river().to_csv("data/river.csv", index=False)
    fetch_weather(sites.dec_lat_va.mean(), sites.dec_long_va.mean()).to_csv("data/weather.csv", index=False)
    print(len(sites), "wells;", lv.date.min().date(), "->", lv.date.max().date())
