"""The twin: weekly forecast -> assimilate well observations (optimal interpolation) -> next week.

Each cycle:
  1. forecast  : MODFLOW 6 advances the last analysis one week with the real river stage, rain and ET
  2. observe   : causal 'ambient' head at each well (trailing 14-day upper percentile; no future data)
  3. analyse   : optimal interpolation of the innovations onto the grid (Gaussian covariance)
Runs a free (no-assimilation) model alongside so the value added by the data link is measurable.
"""
import json
import os

import numpy as np
import pandas as pd

import geo
import model

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
CYCLE0 = "2025-10-25"
SIGMA_B, LCORR = 0.8, 4000.0          # background error (ft) and correlation length (ft)
SIGMA_O_GOOD, SIGMA_O_OTHER = 0.3, 1.5


def causal_ambient(h, window=14, q=0.9):
    return h.rolling(window, min_periods=5).quantile(q)


def oi_update(hf, innov, wells, g, sig_o):
    """Analysis increment on the grid from innovations at wells (row/col of each well)."""
    ok = ~np.isnan(innov)
    if not ok.any():
        return hf, np.zeros_like(hf)
    wx, wy = wells.x.values[ok], wells.y.values[ok]
    xx, yy = np.meshgrid(g.xc, np.arange(g.nrow)[::-1] * geo.CELL_FT + geo.CELL_FT / 2)
    d2_ww = (wx[:, None] - wx[None]) ** 2 + (wy[:, None] - wy[None]) ** 2
    S = SIGMA_B ** 2 * np.exp(-d2_ww / (2 * LCORR ** 2)) + np.diag(sig_o[ok] ** 2)
    w = np.linalg.solve(S, innov[ok])
    d2_gw = (xx[..., None] - wx) ** 2 + (yy[..., None] - wy) ** 2
    inc = (SIGMA_B ** 2 * np.exp(-d2_gw / (2 * LCORR ** 2))) @ w
    inc = np.where(g.active, inc, 0.0)
    return hf + inc, inc


def well_bias(params, wells_idx):
    """Static per-well offset (obs - model) from the 2021-07..2024-12 free run: fixes level bias, keeps dynamics."""
    import calibrate as C
    from obs import heads_daily
    sim = C.simulate(params, "2021-01-01", "2024-12-31")
    o = causal_ambient(heads_daily()).reindex(sim.index)
    m = sim.index > pd.Timestamp("2021-07-01")
    return (o[m] - sim[m]).mean().reindex(wells_idx).values


HORIZONS = (1, 4, 8)


def run(params, end=None, scenario_forcing=None):
    g = model.grid()
    end = end or str(pd.Timestamp.today().date())
    wells = g.wells.set_index("site_no")
    good = set(json.load(open(os.path.join(ROOT, "calibration_wells.json"))))
    sig_o = np.array([SIGMA_O_GOOD if s in good else SIGMA_O_OTHER for s in wells.index])
    bias = well_bias(params, wells.index)
    from obs import heads_daily
    h = heads_daily()
    amb = causal_ambient(h)
    # spin-up to the first cycle with the free model
    spin = model.forcing_weekly("2021-01-01", CYCLE0)
    _, hs = model.build_and_run(params, spin)
    h_free = hs[-1].copy()
    h_an = h_free.copy()
    forc = model.forcing_weekly(CYCLE0, end)
    nweek = (len(forc) - 1) // 7
    rec = dict(dates=[], an=[], fc=[], free=[], inc=[], obs=[], fc_w=[], free_w=[], an_w=[], hz=[])
    r, c = wells.row.values, wells.col.values
    for k in range(nweek):
        f = forc.iloc[1 + 7 * k: 1 + 7 * (k + 1)]
        _, a = model.build_and_run(params, f, start_heads=h_an)
        _, b = model.build_and_run(params, f, start_heads=h_free)
        hf, h_free = a[-1], b[-1]
        date = f.index[-1]
        o = amb.reindex([date]).iloc[0].reindex(wells.index).values
        innov = o - (hf[r, c] + bias)
        h_an, inc = oi_update(hf, innov, wells.reset_index(), g, sig_o)
        # multi-week hindcast from this analysis with the observed forcing (nan where forcing runs out)
        ahead = forc.iloc[1 + 7 * (k + 1): 1 + 7 * (k + 1 + max(HORIZONS))]
        hz = np.full((len(HORIZONS), len(r)), np.nan)
        if len(ahead) == 7 * max(HORIZONS):
            _, hh = model.build_and_run(params, ahead, start_heads=h_an)
            hz = np.array([hh[n - 1][r, c] + bias for n in HORIZONS])
        for key, val in (("dates", date), ("an", h_an), ("fc", hf), ("free", h_free), ("inc", inc), ("obs", o),
                         ("fc_w", hf[r, c] + bias), ("free_w", h_free[r, c] + bias), ("an_w", h_an[r, c] + bias),
                         ("hz", hz)):
            rec[key].append(val)
    out = {k: (np.array(v) if k != "dates" else np.array([str(d.date()) for d in v])) for k, v in rec.items()}
    out["wells"] = np.array(list(wells.index))
    out["bias"] = bias
    return out


def metrics(out):
    from obs import heads_daily
    amb = causal_ambient(heads_daily())
    clim_w = amb["2021-07-01":"2024-12-31"].mean().reindex(list(out["wells"])).values
    """One-week-ahead skill vs persistence and vs the free run; plus leave-one-out analysis skill."""
    good = set(json.load(open(os.path.join(ROOT, "calibration_wells.json"))))
    ids = list(out["wells"])
    sel = np.array([s in good for s in ids])
    o = out["obs"]                       # nweek x nwell
    res = {}
    for name, mask in (("river-connected wells", sel), ("other wells", ~sel)):
        fc = (out["fc_w"] - o)[1:, mask]
        fr = (out["free_w"] - o)[1:, mask]
        pers = (o[:-1] - o[1:])[:, mask]
        rm = lambda x: float(np.sqrt(np.nanmean(x ** 2)))
        res[name] = dict(twin_1wk_forecast_rmse_ft=rm(fc), free_run_rmse_ft=rm(fr), persistence_rmse_ft=rm(pers),
                         n_wells=int(mask.sum()))
        for i, n in enumerate(HORIZONS):          # hindcast vs persistence at each lead time
            pred, tgt, base = [], [], []
            for k in range(len(o) - n):
                if np.isnan(out["hz"][k, i]).all():
                    continue
                pred.append(out["hz"][k, i, mask]); tgt.append(o[k + n, mask]); base.append(o[k, mask])
            if pred:
                pred, tgt, base = map(np.array, (pred, tgt, base))
                res[name][f"lead_{n}wk"] = dict(twin_rmse_ft=rm(pred - tgt), persistence_rmse_ft=rm(base - tgt),
                                                climatology_rmse_ft=rm(clim_w[mask] - tgt), n=int(len(pred)))
    return res


if __name__ == "__main__":
    import sys
    params = json.load(open(os.path.join(ROOT, "params.json")))
    if "--metrics-only" in sys.argv:
        out = dict(np.load(os.path.join(ROOT, "twin_out.npz"), allow_pickle=True))
    else:
        out = run(params)
        np.savez_compressed(os.path.join(ROOT, "twin_out.npz"), **out)
    m = metrics(out)
    json.dump(m, open(os.path.join(ROOT, "twin_metrics.json"), "w"), indent=2)
    print(len(out["dates"]), "weekly cycles", out["dates"][0], "->", out["dates"][-1])
    print(json.dumps(m, indent=2))
