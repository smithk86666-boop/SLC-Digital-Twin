# Groundwater digital twin: Platte River aquifer near Ashland, Nebraska

A small, fully open-source digital twin of an alluvial aquifer. A MODFLOW 6 model (built with FloPy) is kept in
sync with the real aquifer by weekly assimilation of USGS well data, forced by the real river stage and weather, and
used to ask "what if?" questions.

> Repo note: the project started as a Salt Lake Valley twin. That area has too few live wells, so it moved to
> Nebraska. The repository name has not been changed.

## What a twin adds over a model

| | Model | This twin |
|---|---|---|
| Data | historical snapshot | live USGS wells, river gauge, daily weather |
| State | fixed after calibration | updated weekly by optimal interpolation of well observations |
| Use | one study | expected-vs-actual monitoring and what-if scenarios from today's state |

## Pipeline

```
src/fetch_data.py   USGS NWIS wells + Platte gauge (06801000) + Open-Meteo weather  -> data/*.csv
src/geo.py          grid, USGS 3DEP elevations, USGS NHD streams
src/model.py        MODFLOW 6 model: Platte (RIV, gauge-driven), creeks, open ends (GHB), recharge, ET, wells
src/calibrate.py    fits K, Sy, recharge, ET, riverbed and boundary parameters (2021-2024)
src/evaluate.py     skill on the calibration period vs the held-out 2025-26 period
src/twin.py         weekly forecast -> assimilate -> next week; multi-week hindcasts vs baselines
src/analysis.py     expected-vs-actual and drawdown-event detection
src/scenarios.py    13-week what-ifs: typical, drought, hypothetical 1 MGD well
src/render_video.py the explainer video
```

Run order: `fetch_data` → `calibrate` → `evaluate` → `twin` → `analysis` → `scenarios` → `render_video`.
Requires the `mf6` executable (`get-modflow` from FloPy, or set `MF6=/path/to/mf6`), Python 3, FloPy, NumPy, SciPy,
pandas, matplotlib, and ffmpeg for the video.

## Results, and their limits

Hindcast over 52 weekly cycles (Nov 2025 - Oct 2026) using observed weather, so these are not true forecasts.
RMSE of ambient water level, in feet (lower is better):

| Wells | Lead | Twin | Persistence | Climatology |
|---|---|---|---|---|
| 7 river-connected | 4 weeks | 0.80 | 1.03 | 0.93 |
| 7 river-connected | 8 weeks | 0.83 | 1.25 | 1.03 |
| 16 others | 4 weeks | 2.99 | 3.78 | 2.96 |
| 16 others | 8 weeks | 3.10 | 4.23 | 3.14 |

- **Where it works:** wells whose levels are driven by the river and rain. The twin beats both baselines at 4-8 weeks.
  At 1 week, simply persisting the last value is better.
- **Where it doesn't:** the other 16 wells are dominated by pumping that the model does not know about. There the twin
  is no better than a long-term average. The twin instead flags departures below its expected level as likely
  pumping events (146 over the year; the river-connected wells are almost silent).
- The river-connected wells were selected using 2021-2024 data only; 2025-26 was held out of calibration and
  selection. Free-running held-out skill on those wells is modest (median NSE 0.38, correlation 0.63).
- Calibration used a pumping-filtered "ambient" level (rolling upper percentile). One parameter (boundary
  depth `d_bnd`) ended on its lower bound, a sign of structural simplification.
- Single layer, 800 ft cells, 90 ft uniform thickness, uniform properties. Individual wells in a pumping field are
  not resolved.
- The what-if results (river supplying 56% of a new well's water in week 1, rising to 94% by week 13) are model
  estimates under these simplifications, for a hypothetical well, and have not been validated against observed
  stream depletion.
