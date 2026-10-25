"""Skill of the calibrated free-running model against ambient well heads, split by period."""
import json
import os

import numpy as np
import pandas as pd

import calibrate as C
import obs

ROOT = C.ROOT


def skill(sim, o):
    """Per-well anomaly RMSE, NSE of anomalies, and correlation (means removed over the window)."""
    ok = sim.notna() & o.notna()
    sim, o = sim.where(ok), o.where(ok)
    a, b = sim - sim.mean(), o - o.mean()
    rmse = np.sqrt(((a - b) ** 2).mean())
    nse = 1 - ((a - b) ** 2).sum() / (b ** 2).sum()
    return pd.DataFrame({"rmse_ft": rmse, "obs_std_ft": o.std(), "nse": nse, "corr": a.corrwith(b)})


def run(params=None, start="2021-01-01", end=None):
    params = params or json.load(open(os.path.join(ROOT, "params.json")))
    end = end or str(pd.Timestamp.today().date())
    sim = C.simulate(params, start, end)
    o = obs.ambient(obs.heads_daily()).reindex(sim.index)
    return sim, o


if __name__ == "__main__":
    sim, o = run()
    sel = json.load(open(os.path.join(ROOT, "calibration_wells.json")))
    for label, a, b in [("calibration 2021-07..2024-12", "2021-07-01", "2024-12-31"),
                        ("HELD-OUT 2025-01..2026-10", "2025-01-01", "2026-10-31")]:
        s = skill(sim.loc[a:b], o.loc[a:b])
        print(f"\n{label}\n  selected wells : median NSE {s.loc[sel, 'nse'].median():.2f}  median corr {s.loc[sel, 'corr'].median():.2f}  "
              f"median RMSE {s.loc[sel, 'rmse_ft'].median():.2f} ft (obs std {s.loc[sel, 'obs_std_ft'].median():.2f})")
        rest = s.drop(sel)
        print(f"  other wells    : median NSE {rest['nse'].median():.2f}  median corr {rest['corr'].median():.2f}  "
              f"median RMSE {rest['rmse_ft'].median():.2f} ft (obs std {rest['obs_std_ft'].median():.2f})")
