"""Observation handling: depth-to-water -> head, plus a pumping-free 'ambient' head."""
import os

import numpy as np
import pandas as pd

import geo

DATA = geo.DATA


def heads_daily():
    """Daily water-table elevation (ft NAVD88) per well, columns = site_no."""
    w = pd.read_csv(os.path.join(DATA, "wells.csv"), dtype={"site_no": str}).set_index("site_no")
    lv = pd.read_csv(os.path.join(DATA, "levels.csv"), parse_dates=["date"], dtype={"site_no": str})
    dtw = lv.pivot(index="date", columns="site_no", values="dtw_ft")
    dtw = dtw.reindex(pd.date_range(dtw.index.min(), dtw.index.max(), freq="D"))
    return w.alt_va - dtw


def ambient(h, window=30, q=0.9):
    """Rolling upper percentile: the water table with pumping drawdowns filtered out."""
    return h.rolling(window, min_periods=10, center=True).quantile(q)
