"""Local grid geometry, DEM sampling, and stream geometry for the Ashland, NE model.

Model units are feet and days.  Local coordinates: x east / y north in feet from the
south-west corner (LON0, LAT0).
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

LON0, LAT0 = -96.39, 41.00
LON1, LAT1 = -96.26, 41.13
CELL_FT = 800.0
M_PER_FT = 0.3048
LAT_FT = 110574.0 / M_PER_FT                                  # ft per degree latitude
LON_FT = 111320.0 * np.cos(np.radians((LAT0 + LAT1) / 2)) / M_PER_FT

NCOL = int(round((LON1 - LON0) * LON_FT / CELL_FT))
NROW = int(round((LAT1 - LAT0) * LAT_FT / CELL_FT))
DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def to_xy(lon, lat):
    return (np.asarray(lon) - LON0) * LON_FT, (np.asarray(lat) - LAT0) * LAT_FT


def cell_centers():
    xc = (np.arange(NCOL) + 0.5) * CELL_FT
    yc = (np.arange(NROW) + 0.5) * CELL_FT          # row 0 is the SOUTH edge here; flipped in model.py
    return xc, yc


def _epqs(lonlat):
    lon, lat = lonlat
    for _ in range(4):
        try:
            r = requests.get("https://epqs.nationalmap.gov/v1/json", timeout=30,
                             params=dict(x=lon, y=lat, wkid=4326, units="Feet", includeDate="false"))
            if r.status_code == 200:
                return float(r.json()["value"])
        except requests.RequestException:
            pass
    return np.nan


def sample_dem(step_ft=4000.0):
    path = os.path.join(DATA, "dem_samples.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    xs = np.arange(0, NCOL * CELL_FT + 1, step_ft)
    ys = np.arange(0, NROW * CELL_FT + 1, step_ft)
    pts = [(LON0 + x / LON_FT, LAT0 + y / LAT_FT) for y in ys for x in xs]
    with ThreadPoolExecutor(6) as ex:
        z = list(ex.map(_epqs, pts))
    df = pd.DataFrame(pts, columns=["lon", "lat"]).assign(elev_ft=z)
    df.to_csv(path, index=False)
    return df


def fetch_streams():
    path = os.path.join(DATA, "streams.json")
    if os.path.exists(path):
        return json.load(open(path))
    r = requests.get("https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/6/query", timeout=90, params=dict(
        where="1=1", geometry=f"{LON0 - .02},{LAT0 - .02},{LON1 + .02},{LAT1 + .02}",
        geometryType="esriGeometryEnvelope", inSR=4326, outSR=4326, outFields="gnis_name,ftype",
        returnGeometry="true", f="geojson", resultRecordCount=3000))
    r.raise_for_status()
    keep = [f for f in r.json()["features"]
            if f["properties"].get("gnis_name") in ("Platte River", "Salt Creek", "Wahoo Creek")]
    out = {"type": "FeatureCollection", "features": keep}
    json.dump(out, open(path, "w"))
    return out


if __name__ == "__main__":
    print("grid", NROW, "x", NCOL, "cells of", CELL_FT, "ft")
    d = sample_dem()
    print(d.elev_ft.describe())
    s = fetch_streams()
    print(len(s["features"]), "stream segments")
