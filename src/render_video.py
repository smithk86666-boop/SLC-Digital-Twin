"""Render the LinkedIn video (1080x1080, 24 fps) from the twin outputs.

    python render_video.py --still SCENE FRACTION out.png     # one frame, for checking layout
    python render_video.py --render out.mp4                   # full video

Scenes: title, live twin, expected-vs-actual (pumping events), what-if, scorecard, end card.
Colours follow the dataviz palette on the dark surface: blue / orange / aqua for the three series,
a one-hue blue ramp for depth to water, blue<->red with a neutral midpoint for changes.
"""
import glob
import json
import os
import subprocess
import sys
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap

import geo
import model

for f in glob.glob("/usr/share/fonts/opentype/inter/Inter-*.otf"):
    fm.fontManager.addfont(f)
plt.rcParams["font.family"] = "Inter"

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
SIZE, DPI, FPS = 10.8, 100, 24

BG, INK, INK2, MUTED, GRID, AXIS = "#1a1a19", "#f0efec", "#c3c2b7", "#898781", "#2c2c2a", "#383835"
BLUE, ORANGE, AQUA = "#3987e5", "#d95926", "#199e70"
SEQ = LinearSegmentedColormap.from_list("seq", ["#0d366b", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"])
DIV = LinearSegmentedColormap.from_list("div", ["#e66767", "#383835", "#3987e5"])

# ---------------------------------------------------------------- data (loaded once, shared by forked workers)
G = model.grid()
TW = dict(np.load(os.path.join(ROOT, "twin_out.npz"), allow_pickle=True))
LD = dict(np.load(os.path.join(ROOT, "live_daily.npz"), allow_pickle=True))
SC = dict(np.load(os.path.join(ROOT, "scenarios.npz"), allow_pickle=True))
EV = json.load(open(os.path.join(ROOT, "events.json")))
MET = json.load(open(os.path.join(ROOT, "twin_metrics.json")))
GOOD = set(json.load(open(os.path.join(ROOT, "calibration_wells.json"))))
WELLS = [str(w) for w in TW["wells"]]
WK = pd.to_datetime(TW["dates"])
DAYS = pd.to_datetime(LD["days"])
NW, ND = len(WK), len(DAYS)
RIV = pd.read_csv(os.path.join(ROOT, "river.csv"), parse_dates=["date"]).set_index("date").stage_ft.reindex(DAYS).interpolate()
WX = pd.read_csv(os.path.join(ROOT, "weather.csv"), parse_dates=["date"]).set_index("date").precip_mm.reindex(DAYS).fillna(0)
WELL_XY = G.wells.set_index("site_no").loc[WELLS][["x", "y"]].values
SEL = np.array([w in GOOD for w in WELLS])
EXTENT = (0, G.ncol * geo.CELL_FT, 0, G.nrow * geo.CELL_FT)


def _streams():
    out = {"Platte River": [], "Salt Creek": [], "Wahoo Creek": []}
    for f in json.load(open(os.path.join(ROOT, "streams.json")))["features"]:
        g = f["geometry"]
        for p in (g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]):
            a = np.array(p)
            out[f["properties"]["gnis_name"]].append(np.c_[geo.to_xy(a[:, 0], a[:, 1])])
    return out


STREAMS = _streams()
MEAN_FIELD = TW["an"].mean(axis=0)
GAIN = (np.nanmean(TW["obs"], axis=0))


# ---------------------------------------------------------------- helpers
def fig_new():
    return plt.figure(figsize=(SIZE, SIZE), dpi=DPI, facecolor=BG)


def text(fig, x, y, s, size=15, color=INK, weight="regular", ha="left", va="top", alpha=1.0):
    fig.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, alpha=alpha)


def header(fig, step, title, sub=None):
    text(fig, 0.05, 0.955, step, 14, MUTED, "medium")
    text(fig, 0.05, 0.925, title, 30, INK, "bold")
    if sub:
        text(fig, 0.05, 0.868, sub, 16, INK2)


def footer(fig, s="Data: USGS NWIS (wells, Platte River gauge) · Open-Meteo · USGS NHD/3DEP   |   Model: MODFLOW 6 via FloPy"):
    text(fig, 0.05, 0.028, s, 11.5, MUTED, va="bottom")


def style(ax, left=True):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["left"].set_color(AXIS if left else BG)
    ax.tick_params(colors=MUTED, labelsize=12, length=3)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def month_ticks(ax, every=2):
    idx = [i for i, d in enumerate(DAYS) if d.day == 1][::every]
    ax.set_xticks(idx)
    ax.set_xticklabels([DAYS[i].strftime("%b") for i in idx])
    ax.set_xlim(0, ND - 1)


def draw_map(ax, field, cmap, vmin, vmax, wells=True, well_style="ring", mark=None):
    ax.set_facecolor(BG)
    f = np.where(G.active, field, np.nan)
    im = ax.imshow(f, extent=EXTENT, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    for name, lines in STREAMS.items():
        for ln in lines:
            if name == "Platte River":
                ax.plot(ln[:, 0], ln[:, 1], color=INK, lw=1.6, alpha=0.9, solid_capstyle="round")
            else:
                ax.plot(ln[:, 0], ln[:, 1], color=INK2, lw=0.8, alpha=0.7)
    if wells:
        if well_style == "split":
            ax.scatter(*WELL_XY[SEL].T, s=70, facecolor=BG, edgecolor=INK, lw=1.6, zorder=5)
            ax.scatter(*WELL_XY[~SEL].T, s=70, facecolor=BG, edgecolor=ORANGE, lw=1.6, marker="D", zorder=5)
        else:
            ax.scatter(*WELL_XY.T, s=60, facecolor="none", edgecolor=INK, lw=1.4, zorder=5)
    if mark is not None:
        ax.scatter([mark[0]], [mark[1]], s=200, marker="*", facecolor=AQUA, edgecolor=BG, lw=1.2, zorder=6)
    ax.set_xlim(0, EXTENT[1])
    ax.set_ylim(0, EXTENT[3])
    ax.set_aspect("equal")
    ax.axis("off")
    return im


def colorbar_h(fig, im, rect, label, ticks):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=ticks)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelsize=12, length=0)
    cb.set_label(label, color=INK2, fontsize=12.5)
    return cb


def ease(t):
    return t * t * (3 - 2 * t)


def week_field(k):
    """Twin analysis at fractional week index k, blended between weeks."""
    k = float(np.clip(k, 0, NW - 1))
    k0 = int(np.floor(k))
    k1 = min(k0 + 1, NW - 1)
    f = k - k0
    return (1 - f) * TW["an"][k0] + f * TW["an"][k1]


# ---------------------------------------------------------------- scenes
def scene_title(p):
    fig = fig_new()
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    draw_map(ax, G.top - week_field(NW - 1), SEQ, 0, 30, wells=True)
    ax.images[0].set_alpha(0.55)
    ax.set_xlim(-4000, EXTENT[1] + 4000)
    a = float(np.clip(p * 3, 0, 1))
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=BG, alpha=0.62, zorder=20))
    text(fig, 0.07, 0.60, "A digital twin of a river aquifer", 46, INK, "bold", alpha=a)
    text(fig, 0.07, 0.50, "Platte River alluvial aquifer · Ashland, Nebraska", 24, INK2, alpha=a)
    text(fig, 0.07, 0.40, "23 live USGS wells  +  1 river gauge  +  daily weather\nfeeding a MODFLOW 6 groundwater model", 20, INK2, alpha=a)
    text(fig, 0.07, 0.12, "Built with open data and open-source tools", 17, MUTED, alpha=a, va="bottom")
    return fig


def scene_live(p):
    k = p * (NW - 1)
    day = int(np.clip(round(p * (ND - 1)), 0, ND - 1))
    fig = fig_new()
    date = DAYS[day]
    header(fig, "1 · THE LIVE MODEL", "Fed by real data, every week",
           f"Twin state, week of {date.strftime('%b %d, %Y')}")
    ax = fig.add_axes([0.03, 0.20, 0.56, 0.62])
    im = draw_map(ax, week_field(k) - MEAN_FIELD, DIV, -2.0, 2.0)
    colorbar_h(fig, im, [0.08, 0.145, 0.40, 0.014], "Twin's water level vs its 1-year average (ft)", [-2, -1, 0, 1, 2])
    # right column: the data feeding the model
    x = np.arange(ND)
    a1 = fig.add_axes([0.67, 0.64, 0.30, 0.15])
    a1.plot(x[: day + 1], RIV.values[: day + 1], color=INK, lw=2)
    style(a1); month_ticks(a1, 3); a1.set_ylim(RIV.min() - 0.3, RIV.max() + 0.3)
    a1.axvline(day, color=MUTED, lw=1)
    a1.set_title("Platte River stage (ft)", loc="left", color=INK2, fontsize=14, pad=6)
    a2 = fig.add_axes([0.67, 0.41, 0.30, 0.15])
    wk_p = pd.Series(WX.values, index=range(ND)).groupby(np.arange(ND) // 7).sum()
    n = min(day // 7 + 1, len(wk_p))
    a2.bar(np.arange(n) * 7 + 3, wk_p.values[:n], width=5.2, color=BLUE)
    style(a2); month_ticks(a2, 3); a2.set_ylim(0, wk_p.max() * 1.05)
    a2.axvline(day, color=MUTED, lw=1)
    a2.set_title("Rain (mm per week)", loc="left", color=INK2, fontsize=14, pad=6)
    a3 = fig.add_axes([0.67, 0.18, 0.30, 0.15])
    o = TW["obs"][:, SEL] - np.nanmean(TW["obs"][:, SEL], axis=0)
    t = TW["fc_w"][:, SEL] - np.nanmean(TW["obs"][:, SEL], axis=0)
    wx_idx = np.arange(NW) * 7 + 6
    kk = min(int(k) + 1, NW)
    a3.plot(wx_idx[:kk], np.nanmean(o, axis=1)[:kk], color=INK2, lw=2)
    a3.plot(wx_idx[:kk], np.nanmean(t, axis=1)[:kk], color=BLUE, lw=2)
    style(a3); month_ticks(a3, 3); a3.set_ylim(-1.8, 1.8)
    a3.axvline(day, color=MUTED, lw=1)
    a3.set_title("River-connected wells (ft)", loc="left", color=INK2, fontsize=14, pad=6)
    a3.text(0.02, 0.06, "Observed", color=INK2, fontsize=12.5, transform=a3.transAxes, fontweight="medium")
    a3.text(0.40, 0.06, "Twin forecast", color=BLUE, fontsize=12.5, transform=a3.transAxes, fontweight="medium")
    text(fig, 0.05, 0.095, "Rings = USGS monitoring wells. Line = Platte River.", 13.5, INK2)
    footer(fig)
    return fig


def scene_events(p):
    q = ease(float(np.clip(p / 0.78, 0, 1)))
    day = int(q * (ND - 1))
    fig = fig_new()
    header(fig, "2 · EXPECTED VS ACTUAL", "Spotting pumping from the gap",
           "Where the twin expects the water level to be, and where it actually is")
    x = np.arange(ND)
    obs, exp = LD["obs"], LD["expected"]
    rows = [("A river-connected well: tracks the twin", "410159096181001"),
            ("A pumped well: repeatedly drops below the twin", max(EV["per_well"], key=EV["per_well"].get))]
    for i, (ttl, w) in enumerate(rows):
        j = WELLS.index(w)
        ax = fig.add_axes([0.08, 0.66 - i * 0.20, 0.88, 0.145])
        ax.plot(x[: day + 1], obs[: day + 1, j], color=INK2, lw=1.6)
        ax.plot(x[: day + 1], exp[: day + 1, j], color=BLUE, lw=2.4)
        below = (obs[:, j] - exp[:, j] < -2.5) & (x <= day)
        ax.fill_between(x, np.where(below, obs[:, j], np.nan), np.where(below, exp[:, j], np.nan),
                        color=ORANGE, alpha=0.9, lw=0)
        style(ax); month_ticks(ax, 2)
        lo, hi = np.nanmin(np.r_[obs[:, j], exp[:, j]]), np.nanmax(np.r_[obs[:, j], exp[:, j]])
        ax.set_ylim(lo - 0.5, hi + 0.5)
        ax.set_title(ttl, loc="left", color=INK, fontsize=15, pad=6, fontweight="medium")
        ax.set_ylabel("Water level (ft)", color=MUTED, fontsize=12)
    fig.text(0.96, 0.805, "Observed", color=INK2, fontsize=13, ha="right", fontweight="medium")
    fig.text(0.96, 0.783, "Twin expected", color=BLUE, fontsize=13, ha="right", fontweight="medium")
    # event raster: wells (rows) x days
    order = sorted(range(len(WELLS)), key=lambda j: -EV["per_well"][WELLS[j]])
    dep = (obs - exp) < -2.5
    ras = np.where(dep[:, order].T, 1.0, 0.0)
    ras[:, day + 1:] = 0.0
    ax = fig.add_axes([0.08, 0.15, 0.88, 0.21])
    ax.set_facecolor(BG)
    ax.imshow(ras, aspect="auto", cmap=LinearSegmentedColormap.from_list("e", [GRID, ORANGE]), vmin=0, vmax=1,
              interpolation="nearest", extent=(0, ND - 1, len(WELLS), 0))
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_yticks([])
    ax.tick_params(colors=MUTED, labelsize=12, length=3)
    month_ticks(ax, 2)
    ax.set_title("All 23 wells: days when the level is 2.5+ ft below the twin", loc="left", color=INK, fontsize=15,
                 pad=6, fontweight="medium")
    n_ev = sum(1 for j in range(len(WELLS)) for s in LD_EVENT_STARTS[WELLS[j]] if s <= day)
    text(fig, 0.96, 0.098, f"{n_ev} drawdown events flagged", 17, ORANGE, "bold", ha="right")
    text(fig, 0.05, 0.098, "Departures are likely pumping; the model has no pumping data.", 13, INK2)
    footer(fig)
    return fig


def scene_whatif(p):
    q = ease(float(np.clip(p / 0.72, 0, 1)))
    w = int(round(q * (len(SC["dates"]) - 1)))
    fig = fig_new()
    header(fig, "3 · WHAT IF?", "Ask the twin about the next 13 weeks",
           f"Starting from today's state ({SC['start']}); change in water level vs a typical season")
    r, c = SC["well_cell"]
    mark = ((c + 0.5) * geo.CELL_FT, (G.nrow - r - 0.5) * geo.CELL_FT)
    d_dry = SC["drought"][w] - SC["typical"][w]
    d_well = SC["new_well"][w] - SC["typical"][w]
    ax = fig.add_axes([0.03, 0.40, 0.45, 0.40])
    im = draw_map(ax, d_dry, DIV, -1.5, 1.5, well_style="ring")
    colorbar_h(fig, im, [0.08, 0.375, 0.34, 0.012], "Water-level change (ft)", [-1.5, 0, 1.5])
    ax.set_title("Drought: 30% of normal rain, river 1 ft low", color=ORANGE, fontsize=15, pad=6, loc="left",
                 fontweight="bold")
    ax = fig.add_axes([0.52, 0.40, 0.45, 0.40])
    im = draw_map(ax, d_well, DIV, -0.6, 0.6, well_style="ring", mark=mark)
    colorbar_h(fig, im, [0.57, 0.375, 0.34, 0.012], "Water-level change (ft), note finer scale", [-0.6, 0, 0.6])
    ax.set_title("New 1 MGD well (hypothetical, ★)", color=AQUA, fontsize=15, pad=6, loc="left", fontweight="bold")
    # river share of the new well's pumping
    q_well = 133680.6
    share = (SC["new_well_riv_flux"] - SC["typical_riv_flux"]) / q_well * 100
    ax = fig.add_axes([0.08, 0.115, 0.50, 0.13])
    wk = np.arange(1, len(share) + 1)
    ax.plot(wk[: w + 1], share[: w + 1], color=AQUA, lw=2.6)
    ax.scatter([wk[w]], [share[w]], s=60, color=AQUA, zorder=5, edgecolor=BG, lw=1.5)
    style(ax)
    ax.set_xlim(1, len(share)); ax.set_ylim(0, 100)
    ax.set_xticks([1, 5, 9, 13]); ax.set_xticklabels(["wk 1", "5", "9", "13"])
    ax.set_yticks([0, 50, 100]); ax.set_yticklabels(["0%", "50%", "100%"])
    ax.set_title("Share of the new well's water drawn from the river", loc="left", color=INK2, fontsize=13.5, pad=6)
    text(fig, 0.66, 0.255, f"{share[w]:.0f}%", 54, AQUA, "bold")
    text(fig, 0.66, 0.165, "of the pumped water comes\nfrom the river by week 13\n(model estimate)", 13.5, INK2)
    footer(fig)
    return fig


def scene_score(p):
    a = float(np.clip(p * 4, 0, 1))
    fig = fig_new()
    header(fig, "4 · HOW GOOD IS IT?", "An honest scorecard", "Forecast error, wells where the river drives the aquifer (7 of 23)")
    m = MET["river-connected wells"]
    leads = [("4 weeks ahead", m["lead_4wk"]), ("8 weeks ahead", m["lead_8wk"])]
    ax = fig.add_axes([0.08, 0.42, 0.88, 0.36])
    xs = np.arange(len(leads))
    wdt = 0.24
    series = [("Twin", "twin_rmse_ft", BLUE), ("Persistence (last value)", "persistence_rmse_ft", ORANGE),
              ("Climatology (average)", "climatology_rmse_ft", AQUA)]
    for i, (lab, key, col) in enumerate(series):
        vals = [l[1][key] * a for l in leads]
        bars = ax.bar(xs + (i - 1) * (wdt + 0.03), vals, width=wdt, color=col)
        for xb, v in zip(xs + (i - 1) * (wdt + 0.03), vals):
            ax.text(xb, v + 0.03, f"{v:.2f}", ha="center", color=INK, fontsize=13.5, fontweight="medium", alpha=a)
    style(ax)
    ax.set_xticks(xs); ax.set_xticklabels([l[0] for l in leads], fontsize=14, color=INK2)
    ax.set_ylim(0, 1.5); ax.set_yticks([0, 0.5, 1.0, 1.5])
    ax.set_ylabel("RMSE (ft), lower is better", color=MUTED, fontsize=12.5)
    ax.tick_params(axis="x", length=0)
    for i, (lab, key, col) in enumerate(series):
        fig.text(0.10 + i * 0.30, 0.375, "■ ", color=col, fontsize=15, va="center")
        fig.text(0.125 + i * 0.30, 0.375, lab, color=INK2, fontsize=13.5, va="center")
    rows = ["Works where the river and rain drive the water table: beats both baselines at 4–8 weeks.",
            "Elsewhere pumping dominates and the twin is no better than an average, so it flags events instead.",
            "Tested on 2025–26 weather it never saw; wells chosen from 2021–24 data only.",
            "Hindcast with observed weather, not a true forecast. Single-layer model, 800 ft cells."]
    for i, r_ in enumerate(rows):
        text(fig, 0.06, 0.315 - i * 0.062, "•  " + r_, 14.5, INK if i < 2 else INK2, alpha=float(np.clip(p * 5 - i * 0.6, 0, 1)))
    footer(fig)
    return fig


def scene_end(p):
    a = float(np.clip(p * 4, 0, 1))
    fig = fig_new()
    text(fig, 0.07, 0.62, "Digital twin ≠ just a model", 44, INK, "bold", alpha=a)
    text(fig, 0.07, 0.50, "A model answers once.\nA twin stays connected to the real aquifer,\nupdates as the data arrives, and tells you what changed.", 22, INK2, alpha=a)
    text(fig, 0.07, 0.16, "Open data: USGS NWIS · Open-Meteo · USGS NHD & 3DEP\nOpen source: MODFLOW 6 · FloPy · Python", 16, MUTED, alpha=a, va="bottom")
    return fig


LD_EVENT_STARTS = {w: EV_s for w, EV_s in zip(WELLS, [[] for _ in WELLS])}


def _event_starts():
    obs, exp = LD["obs"], LD["expected"]
    out = {}
    for j, w in enumerate(WELLS):
        below = ((obs[:, j] - exp[:, j]) < -2.5) & ~np.isnan(obs[:, j])
        out[w] = [int(s) for s in np.where(below & ~np.r_[False, below[:-1]])[0]]
    return out


LD_EVENT_STARTS = _event_starts()

SCENES = [(scene_title, 5.0), (scene_live, 20.0), (scene_events, 17.0), (scene_whatif, 17.0), (scene_score, 12.0),
          (scene_end, 5.0)]
BOUNDS = np.cumsum([0] + [int(s * FPS) for _, s in SCENES])
NFRAMES = int(BOUNDS[-1])


def frame_fig(i):
    s = int(np.searchsorted(BOUNDS, i, side="right") - 1)
    fn = SCENES[s][0]
    n = BOUNDS[s + 1] - BOUNDS[s]
    return fn((i - BOUNDS[s]) / max(n - 1, 1))


def frame_bytes(i):
    fig = frame_fig(i)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].tobytes()
    plt.close(fig)
    return buf


if __name__ == "__main__":
    if "--still" in sys.argv:
        scene, frac, out = int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
        fig = SCENES[scene][0](frac)
        fig.savefig(out, dpi=DPI, facecolor=BG)
    else:
        out = sys.argv[sys.argv.index("--render") + 1]
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                               "-s", f"{int(SIZE * DPI)}x{int(SIZE * DPI)}", "-r", str(FPS), "-i", "-",
                               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
                               "-movflags", "+faststart", out], stdin=subprocess.PIPE)
        with Pool(4) as pool:
            for k, b in enumerate(pool.imap(frame_bytes, range(NFRAMES), chunksize=6)):
                ff.stdin.write(b)
                if k % 120 == 0:
                    print(f"frame {k}/{NFRAMES}", flush=True)
        ff.stdin.close()
        ff.wait()
        print("wrote", out)
