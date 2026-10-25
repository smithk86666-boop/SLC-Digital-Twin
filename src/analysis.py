"""Daily 'expected vs actual' for the live period, and drawdown-event detection.

expected(day) interpolates the twin's weekly forecasts (priors that have not seen that week's observations),
bias-corrected per well.  An event is a run of days with the observed level more
than EVENT_FT below expected.
"""
import json
import os

import numpy as np
import pandas as pd

import obs

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
EVENT_FT = 2.5


def build():
    tw = np.load(os.path.join(ROOT, "twin_out.npz"), allow_pickle=True)
    wells = [str(w) for w in tw["wells"]]
    wk = pd.to_datetime(tw["dates"])
    start = wk[0] - pd.Timedelta(days=7)
    days = pd.date_range(start + pd.Timedelta(days=1), wk[-1], freq="D")
    # expected level: start state, then each week-end forecast (a prior that has not seen that week's
    # observations), linearly interpolated between them
    x = np.r_[-1, 7 * np.arange(len(wk)) + 6]
    y = np.vstack([tw["free_w"][0], tw["fc_w"]])
    exp = np.column_stack([np.interp(np.arange(len(days)), x, y[:, j]) for j in range(len(wells))])
    h = obs.heads_daily().reindex(days)[wells]
    dep = h.values - exp
    events = {}
    for j, w in enumerate(wells):
        below = (dep[:, j] < -EVENT_FT) & ~np.isnan(dep[:, j])
        starts = np.where(below & ~np.r_[False, below[:-1]])[0]
        events[w] = [int(s) for s in starts]
    np.savez_compressed(os.path.join(ROOT, "live_daily.npz"), days=np.array([str(d.date()) for d in days]),
                        obs=h.values, expected=exp, wells=np.array(wells))
    summary = dict(event_threshold_ft=EVENT_FT, total_events=int(sum(len(v) for v in events.values())),
                   per_well={w: len(v) for w, v in events.items()},
                   deepest_departure_ft=float(np.nanmin(dep)))
    json.dump(summary, open(os.path.join(ROOT, "events.json"), "w"), indent=2)
    return summary, events


if __name__ == "__main__":
    s, ev = build()
    print("events:", s["total_events"], "deepest departure %.1f ft" % s["deepest_departure_ft"])
    good = set(json.load(open(os.path.join(ROOT, "calibration_wells.json"))))
    print("river-connected:", {w: n for w, n in s["per_well"].items() if w in good})
    print("others:", sorted(s["per_well"].values(), reverse=True))
