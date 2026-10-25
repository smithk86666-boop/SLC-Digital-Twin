"""MODFLOW 6 model of the Platte River alluvial aquifer near Ashland, NE (FloPy).

One unconfined layer on an 800 ft grid, valley floor only (land surface < VALLEY_MAX_FT).
Stresses: Platte stage (RIV, from the Ashland gauge), Salt/Wahoo Creek (RIV, static),
open valley ends (GHB), rain recharge (RCH), ET (EVT) and optional pumping (WEL).
"""
import json
import os
import shutil
import tempfile

import flopy
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

import geo

VALLEY_MAX_FT = 1085.0
THICKNESS_FT = 90.0
GAUGE_DATUM_FT = 1040.27          # Ashland gauge zero, NAVD88 (USGS site metadata)
RIVER_SLOPE = 0.0008              # ft/ft, from Venice vs Ashland gauge water-surface elevations
MF6 = os.environ.get("MF6") or shutil.which("mf6") or "mf6"
DATA = geo.DATA

DEFAULT = dict(k=300.0, sy=0.2, alpha=0.15, gamma=0.5, cond_riv=2000.0, cond_creek=500.0,
               d_bnd=8.0, cond_ghb=4000.0)


class Grid:
    """Static geometry, north-up arrays (row 0 = north) as MODFLOW expects."""

    def __init__(self):
        d = geo.sample_dem()
        x, y = geo.to_xy(d.lon, d.lat)
        xc, yc = geo.cell_centers()
        X, Y = np.meshgrid(xc, yc)
        z = griddata((x, y), d.elev_ft, (X, Y), method="cubic")
        z = np.where(np.isnan(z), griddata((x, y), d.elev_ft, (X, Y), method="nearest"), z)
        z = gaussian_filter(z, 0.7)
        self.xc, self.yc = xc, yc[::-1]
        self.top = np.flipud(z)
        self.active = self.top < VALLEY_MAX_FT
        self.nrow, self.ncol = self.top.shape
        self.botm = self.top - THICKNESS_FT
        self._streams()
        self._wells()

    def cell_of(self, x, y):
        col = np.clip((np.asarray(x) // geo.CELL_FT).astype(int), 0, self.ncol - 1)
        row = np.clip(self.nrow - 1 - (np.asarray(y) // geo.CELL_FT).astype(int), 0, self.nrow - 1)
        return row, col

    def _streams(self):
        feats = json.load(open(os.path.join(DATA, "streams.json")))["features"]
        lines = {"Platte River": [], "Salt Creek": [], "Wahoo Creek": []}
        for f in feats:
            g = f["geometry"]
            parts = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]
            for p in parts:
                a = np.array(p)
                lines[f["properties"]["gnis_name"]].append(np.c_[geo.to_xy(a[:, 0], a[:, 1])])
        self.platte, self.creek = {}, {}
        for name, segs in lines.items():
            pts = np.vstack([np.vstack([s[0], s[-1]]) for s in segs]) if segs else np.empty((0, 2))
            dense = []
            for s in segs:
                for a, b in zip(s[:-1], s[1:]):
                    n = max(2, int(np.hypot(*(b - a)) // 200))
                    dense.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
            dense = np.vstack(dense)
            row, col = self.cell_of(dense[:, 0], dense[:, 1])
            ok = self.active[row, col]
            target = self.platte if name == "Platte River" else self.creek
            for r, c, (px, py) in zip(row[ok], col[ok], dense[ok]):
                target.setdefault((r, c), []).append((px, py))
        # along-channel distance for Platte cells, flowing north -> south-east
        cells = sorted(self.platte, key=lambda rc: rc[0])
        gx, gy = geo.to_xy(-96.324514, 41.062256)
        ref = self.cell_of(gx, gy)
        pos = {rc: i for i, rc in enumerate(cells)}
        self.platte_cells = cells
        self.platte_dist_ft = np.array([(pos[rc] - pos.get(ref, np.argmin([abs(r - ref[0]) for r, _ in cells]))) * geo.CELL_FT
                                        for rc in cells])

    def _wells(self):
        w = pd.read_csv(os.path.join(DATA, "wells.csv"), dtype={"site_no": str})
        x, y = geo.to_xy(w.dec_long_va, w.dec_lat_va)
        w["row"], w["col"] = self.cell_of(x, y)
        w["x"], w["y"] = x, y
        self.wells = w


def forcing_weekly(start, end):
    """Weekly (7-day) stress-period forcing: river stage elev at gauge (ft), precip & ET0 (ft/d)."""
    rv = pd.read_csv(os.path.join(DATA, "river.csv"), parse_dates=["date"]).set_index("date")
    wx = pd.read_csv(os.path.join(DATA, "weather.csv"), parse_dates=["date"]).set_index("date")
    idx = pd.date_range(start, end, freq="D")
    f = pd.DataFrame(index=idx)
    f["stage"] = rv.stage_ft.reindex(idx).interpolate(limit_direction="both") + GAUGE_DATUM_FT
    f["p"] = wx.precip_mm.reindex(idx).fillna(0) / 304.8
    f["et0"] = wx.et0_mm.reindex(idx).fillna(0) / 304.8
    return f


def build_and_run(params, forcing, step_days=7, pumping=None, ws=None, start_heads=None, keep=False,
                  riv_flux=False):
    """Run the model. forcing: daily DataFrame. pumping: dict {(row,col): daily Series of ft3/d}.

    Returns (dates at the end of each stress period, heads array [nper, nrow, ncol]); with riv_flux=True also
    the net Platte/creek-to-aquifer flow per stress period (ft3/d, positive = river losing water to the aquifer).
    """
    g = grid()
    p = {**DEFAULT, **params}
    n = len(forcing) // step_days
    fw = forcing.iloc[: n * step_days].groupby(np.arange(n * step_days) // step_days).mean()
    dates = forcing.index[step_days - 1: n * step_days: step_days]
    ws = ws or tempfile.mkdtemp(prefix="mf6_")
    name = "m"
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=ws, exe_name=MF6, verbosity_level=0)
    flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=n, perioddata=[(step_days, 1, 1.0)] * n)
    flopy.mf6.ModflowIms(sim, complexity="MODERATE", inner_dvclose=1e-4, outer_dvclose=1e-4)
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=riv_flux)
    flopy.mf6.ModflowGwfdis(gwf, length_units="FEET", nlay=1, nrow=g.nrow, ncol=g.ncol,
                            delr=geo.CELL_FT, delc=geo.CELL_FT, top=g.top, botm=g.botm[None],
                            idomain=g.active[None].astype(int))
    sh = start_heads if start_heads is not None else np.where(g.active, g.top - p["d_bnd"], g.top)
    flopy.mf6.ModflowGwfic(gwf, strt=sh[None])
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=1, k=p["k"])
    flopy.mf6.ModflowGwfsto(gwf, iconvert=1, sy=p["sy"], ss=1e-6,
                            transient={0: True}, steady_state=None)
    # Platte River stage varies along channel and with time
    riv = {}
    for i in range(n):
        stage = fw.stage.iloc[i]
        riv[i] = [[(0, r, c), stage - RIVER_SLOPE * d, p["cond_riv"], stage - RIVER_SLOPE * d - 8.0]
                  for (r, c), d in zip(g.platte_cells, g.platte_dist_ft)]
    creek = [[(0, r, c), g.top[r, c] - 3.0, p["cond_creek"], g.top[r, c] - 8.0] for (r, c) in g.creek]
    for i in range(n):
        riv[i] = riv[i] + [x for x in creek if (x[0][1], x[0][2]) not in g.platte]
    flopy.mf6.ModflowGwfriv(gwf, stress_period_data=riv)
    # open valley ends: active cells on the outer rim of the grid
    rim = np.zeros_like(g.active)
    rim[0, :] = rim[-1, :] = rim[:, 0] = rim[:, -1] = True
    ghb = [[(0, r, c), g.top[r, c] - p["d_bnd"], p["cond_ghb"]]
           for r, c in zip(*np.where(rim & g.active)) if (r, c) not in g.platte]
    flopy.mf6.ModflowGwfghb(gwf, stress_period_data={0: ghb})
    rch = {i: np.where(g.active, p["alpha"] * fw.p.iloc[i], 0.0) for i in range(n)}
    flopy.mf6.ModflowGwfrcha(gwf, recharge=rch)
    flopy.mf6.ModflowGwfevta(gwf, surface=g.top, rate={i: p["gamma"] * fw.et0.iloc[i] for i in range(n)},
                             depth=6.0)
    if pumping:
        wel = {i: [[(0, r, c), -float(q.iloc[i * step_days:(i + 1) * step_days].mean())]
                   for (r, c), q in pumping.items()] for i in range(n)}
        flopy.mf6.ModflowGwfwel(gwf, stress_period_data=wel)
    rec = [("HEAD", "LAST")] + ([("BUDGET", "LAST")] if riv_flux else [])
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord=f"{name}.hds", budget_filerecord=f"{name}.cbc" if riv_flux else None,
                           saverecord={i: rec for i in range(n)})
    sim.write_simulation(silent=True)
    ok, buf = sim.run_simulation(silent=True)
    if not ok:
        raise RuntimeError("MODFLOW failed:\n" + "\n".join(buf[-15:]))
    h = flopy.utils.HeadFile(os.path.join(ws, f"{name}.hds")).get_alldata()[:, 0]
    h = np.where(h > 1e20, np.nan, h)
    flux = None
    if riv_flux:
        cbc = flopy.utils.CellBudgetFile(os.path.join(ws, f"{name}.cbc"))
        flux = np.array([sum(float(np.sum(a["q"])) for a in cbc.get_data(kstpkper=kk, text="RIV"))
                         for kk in cbc.get_kstpkper()])
    if not keep:
        shutil.rmtree(ws, ignore_errors=True)
    return (dates, h, flux) if riv_flux else (dates, h)


_GRID = None


def grid():
    global _GRID
    if _GRID is None:
        _GRID = Grid()
    return _GRID


if __name__ == "__main__":
    import time
    g = grid()
    print("active cells", g.active.sum(), "| river cells", len(g.platte_cells), "| creek cells", len(g.creek))
    f = forcing_weekly("2023-01-01", "2023-12-31")
    t0 = time.time()
    dates, h = build_and_run({}, f)
    print("run", round(time.time() - t0, 1), "s;", h.shape, "head range", np.nanmin(h), np.nanmax(h))
