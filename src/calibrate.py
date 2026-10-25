"""Calibrate aquifer/boundary parameters to ambient (pumping-filtered) well heads, 2021-07 .. 2024-12."""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import model
import obs

CAL = ("2021-01-01", "2024-12-31")
SPINUP_END = "2021-07-01"
NAMES = ["k", "sy", "alpha", "gamma", "cond_riv", "d_bnd", "cond_ghb"]
LOG = {"k", "cond_riv", "cond_ghb"}
LO = dict(k=30, sy=0.05, alpha=0.01, gamma=0.0, cond_riv=50, d_bnd=0.0, cond_ghb=100)
HI = dict(k=3000, sy=0.35, alpha=0.5, gamma=1.0, cond_riv=1e5, d_bnd=25.0, cond_ghb=1e5)
LEVEL_W = 0.3
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
CAL_WELLS = json.load(open(os.path.join(ROOT, "calibration_wells.json")))


def to_params(x):
    return {n: (10 ** v if n in LOG else v) for n, v in zip(NAMES, x)}


def to_x(p):
    return np.array([np.log10(p[n]) if n in LOG else p[n] for n in NAMES])


def simulate(p, start, end, **kw):
    g = model.grid()
    f = model.forcing_weekly(start, end)
    dates, hs = model.build_and_run(p, f, **kw)
    w = g.wells.set_index("site_no")
    return pd.DataFrame({s: hs[:, w.row[s], w.col[s]] for s in w.index}, index=dates)


def residuals(x):
    sim = simulate(to_params(x), *CAL)[CAL_WELLS]
    o = obs.ambient(obs.heads_daily())[CAL_WELLS].reindex(sim.index)
    m = sim.index > pd.Timestamp(SPINUP_END)
    sim, o = sim[m], o[m]
    ok = o.notna() & sim.notna()
    sim, o = sim.where(ok), o.where(ok)
    anom = (sim - sim.mean()) - (o - o.mean())
    level = sim.mean() - o.mean()
    r = np.r_[anom.values[ok.values], LEVEL_W * np.repeat(level.values, 1)]
    return np.nan_to_num(r, nan=0.0)


def jac(x, pool, eps=0.02):
    cols = []
    xs = [x + eps * (np.array(HI_X) - np.array(LO_X)) * np.eye(len(x))[i] for i in range(len(x))]
    outs = list(pool.map(residuals, xs))
    r0 = residuals(x)
    return np.column_stack([(o - r0) / (eps * (HI_X[i] - LO_X[i])) for i, o in enumerate(outs)])


LO_X = to_x(LO)
HI_X = to_x(HI)

if __name__ == "__main__":
    x0 = to_x(model.DEFAULT)
    with ProcessPoolExecutor(4) as pool:
        sol = least_squares(residuals, x0, jac=lambda x: jac(x, pool), bounds=(LO_X, HI_X),
                            loss="soft_l1", f_scale=1.0, max_nfev=30, verbose=2, x_scale=(HI_X - LO_X) / 4)
    p = to_params(sol.x)
    json.dump(p, open(os.path.join(ROOT, "params.json"), "w"), indent=2)
    print(json.dumps(p, indent=2))
    print("start cost", 0.5 * np.sum(residuals(x0) ** 2), "final cost", 0.5 * np.sum(residuals(sol.x) ** 2))
