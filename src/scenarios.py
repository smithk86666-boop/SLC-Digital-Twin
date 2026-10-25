"""What-if runs: 13 weeks forward from the twin's latest analysis under three futures.

  typical   : day-of-year climatology of rain, ET and river stage (2021-2025)
  drought   : 30% of normal rain, +15% ET, river 1.0 ft below normal
  new well  : typical weather + a HYPOTHETICAL 1 MGD production well beside the river-connected wells
"""
import json
import os

import numpy as np
import pandas as pd

import geo
import model

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
MGD_FT3_PER_DAY = 133_680.6            # 1 million gallons/day
WEEKS = 13


def climatology_forcing(start, weeks):
    f = model.forcing_weekly("2021-01-01", str(pd.Timestamp.today().date()))
    doy = f.index.dayofyear
    clim = f.groupby(doy).mean().rolling(15, center=True, min_periods=1).mean()
    idx = pd.date_range(start, periods=weeks * 7, freq="D")
    out = clim.reindex(idx.dayofyear).set_axis(idx)
    return out


def run_all():
    params = json.load(open(os.path.join(ROOT, "params.json")))
    tw = np.load(os.path.join(ROOT, "twin_out.npz"), allow_pickle=True)
    h0 = tw["an"][-1]
    start = pd.Timestamp(str(tw["dates"][-1])) + pd.Timedelta(days=1)
    base = climatology_forcing(start, WEEKS)
    dry = base.copy()
    dry["p"] *= 0.3
    dry["et0"] *= 1.15
    dry["stage"] -= 1.0
    g = model.grid()
    wells = g.wells
    # hypothetical well: the active cell nearest the centroid of the river-connected wells, 1500 ft off the river
    good = wells[wells.site_no.isin(json.load(open(os.path.join(ROOT, "calibration_wells.json"))))]
    cx, cy = good.x.mean(), good.y.mean() - 1500.0
    r, c = g.cell_of(cx, cy)
    q = pd.Series(MGD_FT3_PER_DAY, index=base.index)
    runs = {
        "typical": model.build_and_run(params, base, start_heads=h0, riv_flux=True),
        "drought": model.build_and_run(params, dry, start_heads=h0, riv_flux=True),
        "new_well": model.build_and_run(params, base, start_heads=h0, pumping={(int(r), int(c)): q}, riv_flux=True),
    }
    out = {"start": str(start.date()), "well_cell": np.array([r, c]), "h0": h0}
    for k, (d, h, fl) in runs.items():
        out[k] = h
        out[k + "_riv_flux"] = fl
    out["dates"] = np.array([str(x.date()) for x in runs["typical"][0]])
    out["forcing_typical"] = base.reset_index(drop=True).to_numpy()
    out["forcing_drought"] = dry.reset_index(drop=True).to_numpy()
    return out


if __name__ == "__main__":
    out = run_all()
    np.savez_compressed(os.path.join(ROOT, "scenarios.npz"), **out)
    g = model.grid()
    w = g.wells
    r, c = w.row.values, w.col.values
    for k in ("drought", "new_well"):
        dd = out[k][-1][r, c] - out["typical"][-1][r, c]
        print(k, "change vs typical at wells after", WEEKS, "wks: median %.2f ft, min %.2f ft" % (np.median(dd), dd.min()))
    extra = out["new_well_riv_flux"] - out["typical_riv_flux"]
    print("river flux change from the new well (ft3/d, weeks 1,4,8,13):", np.round(extra[[0, 3, 7, 12]]).tolist(),
          "-> share of pumping from river: %s" % np.round(extra[[0, 3, 7, 12]] / MGD_FT3_PER_DAY, 2).tolist())
    dmap = out["new_well"][-1] - out["typical"][-1]
    print("new-well drawdown max %.1f ft; cells >0.5 ft: %d" % (-np.nanmin(dmap), (dmap < -0.5).sum()))
    print("typical vs start: median change at wells %.2f ft" % np.median(out["typical"][-1][r, c] - out["h0"][r, c]))
