# gaspred — U.S. gasoline price: oil vs. actual vs. forecast

A small end-to-end system that forecasts U.S. retail gasoline prices and shows, per
**granularity (national + each PADD region)**, three+ lines: **crude oil price**, **actual gas
price**, and **model forecasts**. It **compares two models** (Ridge vs. gradient boosting)
against a naive baseline, **opens zoomed to the near future** (the primary use case), and has a
**time-range slider** to zoom out.

It's built to run unattended: a single script regenerates the data, and a **GitHub Action**
runs it on a schedule and republishes the static site via **GitHub Pages** — no server to host.

```
EIA + FRED + yfinance ──▶ build_panel.py ──▶ model.py ──▶ update.py ──▶ site/data/predictions.json
   (data sources)          (weekly panel)    (forecast)   (writes JSON)          │
                                                                                 ▼
                              GitHub Action (cron) commits the JSON ──▶ GitHub Pages serves site/index.html
```

---

## ✅ Inputs needed from you

Everything below is free. Two API keys, and (for deployment) a few GitHub settings.

**1. Get two free API keys**

| Key | Where | Used for |
|---|---|---|
| `EIA_API_KEY` | https://www.eia.gov/opendata/register.php | gas prices (target), stocks, refinery utilization, demand |
| `FRED_API_KEY` | https://fredaccount.stlouisfed.org/apikeys | WTI, Brent, USD index |

**2. To run locally:** copy `.env.example` → `.env` and paste both keys.

**3. To deploy (GitHub Actions + Pages):**
- Push this folder to a GitHub repo.
- Add both keys as **repository secrets**: repo *Settings → Secrets and variables → Actions → New repository secret*, named exactly `EIA_API_KEY` and `FRED_API_KEY`.
- Enable Pages: *Settings → Pages → Source: “Deploy from a branch” → branch `main`, folder `/site`*.
- Ensure Actions can push commits: *Settings → Actions → General → Workflow permissions → “Read and write permissions”* (the workflow also requests this itself).
- Run it once: *Actions → “Update predictions” → Run workflow* (or wait for the daily cron).

That's it — nothing else is required from your side. Optional tweaks: the cron schedule in
`.github/workflows/update.yml` and the forecast horizon (`--horizon`, default 4 weeks).

---

## Quick start (local)

```bash
pip install -r requirements.txt

# Option A — preview the UI immediately with synthetic data (no keys):
python gen_sample.py
python -m http.server 8765 --directory site      # open http://localhost:8765

# Option B — real data (needs the two keys in .env):
cp .env.example .env        # paste keys
python update.py -v         # writes site/data/predictions.json
python -m http.server 8765 --directory site
```

Tests (no keys/network needed):
```bash
python test_align.py    # look-ahead-safe weekly alignment
python test_model.py    # model + JSON payload on synthetic data
```

---

## Granularities

The dropdown switches between **National (U.S.)** and the five **PADD** regions (East Coast,
Midwest, Gulf Coast, Rocky Mountain, West Coast). This is the finest level where EIA provides
the target *and* supply-side features for free; West Coast (PADD 5 / California) is the region
that decouples most from the national trend, so it's worth viewing on its own.

## Webapp

Single static page (`site/index.html`, Chart.js, no build step):
- Lines for actual gas, **each model's forecast**, and WTI oil (right axis), with a `now` marker
  where actuals end and pure forecast begins.
- **Opens zoomed to the near future** (3-month window — the primary use case), with a **time-range
  slider** to zoom out (1M → 3M → 6M → 1Y → 2Y → All). The y-axis auto-fits the visible window.
- A **model-comparison table** (MAE / RMSE / MAPE / skill) below the chart; the best model is highlighted.
- Region dropdown switches granularity.

## The models

Per region, two models predict the price `horizon` weeks ahead from one leak-free feature set
(crude levels & lags, **asymmetric** crude moves — rockets-and-feathers "up"/"down" 4-week sums —
a crack-spread proxy, gasoline stocks, refinery utilization, demand, the dollar index, AR lags of
the price, and seasonality):

- `ridge` — standardized `RidgeCV` (linear, interpretable).
- `gbm` — `HistGradientBoostingRegressor` (nonlinear, captures interactions).

Both are evaluated identically and compared against a **naive random-walk** (last-price) baseline:

- **Historical lines** = strictly out-of-sample, expanding-window backtest with an `embargo` of
  `horizon` weeks (training targets are never observable after the decision date).
- **Forward lines** = fit on all data, predict the last `horizon` weeks → the lines extend past "now".
- Per-model metrics: MAE / RMSE / MAPE and **skill vs. naive** (error reduction vs. "tomorrow ≈
  today"). On real data, whichever model wins wins — surfacing that is the point of the comparison.

Add or swap models in `model_factories()` in `model.py` without touching the rest. Threads are
pinned to 1 there (the backtest does many tiny refits, so parallelism would only add overhead).

## Look-ahead safety

Each series is aligned with `align.to_weekly_asof`, which only exposes a value once it would
actually have been public (retail +1 day, weekly WPSR stocks/refinery/demand +5 days, daily
market data +1 day). A row dated week *t* contains only information knowable by that Friday, so
the backtest doesn't peek into the future.

## Output schema — `site/data/predictions.json`

```jsonc
{
  "generated_utc": "2026-06-17T16:56:53Z",
  "horizon_weeks": 4,
  "region_order": ["US", "PADD1", ...],
  "regions": {
    "US": {
      "name": "National (U.S.)",
      "models": ["ridge", "gbm"],
      "model_labels": { "ridge": "Ridge (linear)", "gbm": "Gradient boosting" },
      "metrics": {
        "ridge": { "n": 432, "mae": 0.064, "rmse": 0.08, "mape_pct": 2.37,
                   "naive_mae": 0.089, "skill_vs_naive_pct": 28.0 },
        "gbm":   { "n": 432, "mae": 0.095, "rmse": 0.121, "mape_pct": 3.6,
                   "naive_mae": 0.089, "skill_vs_naive_pct": -7.0 },
        "naive": { "n": 432, "mae": 0.089, "rmse": 0.111, "mape_pct": 3.3 }
      },
      "series": [ { "date": "2026-06-12", "oil": 41.7, "actual": 2.696, "ridge": 2.62, "gbm": 2.70 }, ... ]
    }
  }
}
```
`oil` and `actual` are `null` on future dates; each model's prediction continues `horizon` weeks past them.

## Files

```
config.py        regions, series IDs, release lags, weekly anchor
clients.py       EIA / FRED / yfinance fetchers -> tidy Series
align.py         to_weekly_asof, panel assembly, supervised builder
build_panel.py   orchestrates fetch + align -> weekly national+PADD panel
model.py         feature engineering + Ridge & GBM backtest & forecast
update.py        build panel -> run models -> write predictions.json   (the scheduled job)
gen_sample.py    synthetic predictions.json for previewing without keys
run.py           CLI to dump just the panel (parquet/csv) for analysis
site/index.html  static webapp (Chart.js): model lines + comparison table + time-range slider
.github/workflows/update.yml   scheduled refresh + commit (deployable as-is)
test_align.py, test_model.py   tests (no keys needed)
```

## Caveats

- Weekly granularity is national + PADD only (per-state isn't built here, by design).
- `product_supplied`, crude, and macro/FX are national and broadcast to every region.
- yfinance is unofficial; continuous front-month futures carry roll effects.
- EIA revises history, so re-pulls can shift older values slightly.
- The crack spread here is a simple gasoline crack, not a 3-2-1 spread.
- The baseline model is intentionally simple — treat the metrics as a floor to beat.
