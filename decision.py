"""Decision layer: turn the short-horizon forecast into a 'fill up now vs. wait' signal,
and backtest its real-world value — cents/gallon saved vs. always-filling-now.

This reframes the problem from "what's the price" to "given where prices are headed over the
next few weeks, should I fill now or wait?" The forecast is used only for its direction within a
practical wait window; the payoff metric is money saved, not MAE.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as C
from model import engineer, feature_columns


def _ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 9)))


def horizon_oos(df_region, horizon, min_train=104, refit_every=13):
    """Out-of-sample forecasts for one horizon, keyed by decision date t.
    Columns: t, p0 (price known at t), yhat{h} (predicted price at t+h), ytrue{h} (actual at t+h)."""
    embargo = horizon
    feat = engineer(df_region, horizon)
    cols = feature_columns(feat)
    known = feat.dropna(subset=cols).reset_index(drop=True)
    known = known[known["y"].notna()].reset_index(drop=True)
    X, y = known[cols].to_numpy(float), known["y"].to_numpy(float)
    p0, dt = known["retail_t"].to_numpy(float), known["date"].to_numpy()
    n, rows, i = len(known), [], max(min_train, embargo + 10)
    while i < n:
        train_end = i - embargo
        if train_end >= 30:
            model = _ridge().fit(X[:train_end], y[:train_end])
            j = min(i + refit_every, n)
            pred = model.predict(X[i:j])
            for k in range(i, j):
                rows.append((dt[k], p0[k], pred[k - i], y[k]))
            i = j
        else:
            i += refit_every
    return pd.DataFrame(rows, columns=["t", "p0", f"yhat{horizon}", f"ytrue{horizon}"])


def decision_value(df_region, max_wait=2, tau=0.02, min_train=104, refit_every=13):
    """Backtest 'fill now vs. wait up to max_wait weeks for a predicted dip'.

    Rule: if the cheapest predicted price within the window is below today's price by more than
    `tau`, wait until that predicted-cheapest week (and pay the actual price then); else fill now.
    """
    tabs = [horizon_oos(df_region, h, min_train, refit_every) for h in range(1, max_wait + 1)]
    m = tabs[0]
    for t in tabs[1:]:
        m = m.merge(t.drop(columns=["p0"]), on="t", how="inner")
    m = m.dropna().reset_index(drop=True)
    if len(m) < 30:
        return None

    yhat = m[[f"yhat{h}" for h in range(1, max_wait + 1)]].to_numpy()
    ytrue = m[[f"ytrue{h}" for h in range(1, max_wait + 1)]].to_numpy()
    p0 = m["p0"].to_numpy()
    idx = np.arange(len(m))

    pred_min, pred_arg = yhat.min(axis=1), yhat.argmin(axis=1)
    wait = pred_min < (p0 - tau)                                   # recommend waiting?
    model_cost = np.where(wait, ytrue[idx, pred_arg], p0)          # pay actual at chosen week, else now
    always_now = p0
    perfect = np.minimum(p0, ytrue.min(axis=1))                    # best you could have done

    waited = int(wait.sum())
    wait_win = float((ytrue[idx, pred_arg] < p0)[wait].mean()) if waited else float("nan")
    save_per_gal = always_now.mean() - model_cost.mean()
    headroom = always_now.mean() - perfect.mean()
    return {
        "n": int(len(m)),
        "pct_weeks_wait": round(100 * wait.mean(), 1),
        "save_cents_per_gal": round(100 * save_per_gal, 2),
        "capture_pct": round(100 * save_per_gal / headroom, 1) if headroom > 1e-9 else None,
        "wait_win_rate_pct": round(100 * wait_win, 1) if waited else None,
        "always_now_avg": round(float(always_now.mean()), 3),
        "model_avg": round(float(model_cost.mean()), 3),
        "perfect_avg": round(float(perfect.mean()), 3),
    }


def forward_path(df_region, max_w=4, min_train=104):
    """Fit Ridge per horizon on all known data; forecast the next 1..max_w weeks from the latest
    fully-observed week. Returns as_of date, current price, and the forward path."""
    base = engineer(df_region, 1)
    cols = feature_columns(base)
    valid = base.dropna(subset=cols).reset_index(drop=True)
    if len(valid) < min_train + max_w + 5:
        return None
    latest = valid.iloc[-1]
    t, p0 = pd.Timestamp(latest["date"]), float(latest["retail_t"])
    Xlast = latest[cols].to_numpy(float).reshape(1, -1)
    path = []
    for h in range(1, max_w + 1):
        fe = engineer(df_region, h).dropna(subset=cols)
        fe = fe[fe["y"].notna()]
        model = _ridge().fit(fe[cols].to_numpy(float), fe["y"].to_numpy(float))
        path.append({"weeks_ahead": h, "date": (t + pd.Timedelta(weeks=h)).strftime("%Y-%m-%d"),
                     "yhat": round(float(model.predict(Xlast)[0]), 3)})
    return {"as_of": t.strftime("%Y-%m-%d"), "price_now": round(p0, 3), "path": path}


def _reco_for_window(path, p0, W, tau):
    """Recommendation if willing to wait up to W weeks, given the forward path."""
    yhats = [p["yhat"] for p in path[:W]]
    pred_min, arg = min(yhats), yhats.index(min(yhats))
    drop, rise = p0 - pred_min, max(yhats) - p0
    if drop > tau:
        return {"action": "wait", "expected_change_cents": -round(100 * drop, 1),
                "best_week": path[arg]["weeks_ahead"]}
    if rise > tau:
        return {"action": "fill_now", "expected_change_cents": round(100 * rise, 1), "best_week": 0}
    return {"action": "flat", "expected_change_cents": round(100 * (yhats[0] - p0), 1), "best_week": 0}


def _oos_table(df_region, max_w, min_train, refit_every):
    """One merged out-of-sample table with yhat/ytrue for every horizon 1..max_w (computed once)."""
    tabs = [horizon_oos(df_region, h, min_train, refit_every) for h in range(1, max_w + 1)]
    m = tabs[0]
    for tab in tabs[1:]:
        m = m.merge(tab.drop(columns=["p0"]), on="t", how="inner")
    return m.dropna().reset_index(drop=True)


def _value_for_window(m, W, tau):
    """Decision-value backtest for the 'wait up to W weeks' policy, from the precomputed table."""
    if len(m) < 30:
        return None
    yhat = m[[f"yhat{h}" for h in range(1, W + 1)]].to_numpy()
    ytrue = m[[f"ytrue{h}" for h in range(1, W + 1)]].to_numpy()
    p0, idx = m["p0"].to_numpy(), np.arange(len(m))
    pred_arg = yhat.argmin(axis=1)
    wait = yhat.min(axis=1) < (p0 - tau)
    model_cost = np.where(wait, ytrue[idx, pred_arg], p0)
    perfect = np.minimum(p0, ytrue.min(axis=1))
    waited = int(wait.sum())
    win = float((ytrue[idx, pred_arg] < p0)[wait].mean()) if waited else float("nan")
    save, head = p0.mean() - model_cost.mean(), p0.mean() - perfect.mean()
    return {"n": int(len(m)), "pct_weeks_wait": round(float(100 * wait.mean()), 1),
            "save_cents_per_gal": round(float(100 * save), 2),
            "capture_pct": round(float(100 * save / head), 1) if head > 1e-9 else None,
            "wait_win_rate_pct": round(float(100 * win), 1) if waited else None}


def build_decision(df_region, max_w=4, tau=0.02, min_train=104, refit_every=13):
    """Recommendation for every wait window 1..max_w, each with its own backtest track record."""
    fp = forward_path(df_region, max_w, min_train)
    if fp is None:
        return None
    table = _oos_table(df_region, max_w, min_train, refit_every)
    windows = {str(W): {**_reco_for_window(fp["path"], fp["price_now"], W, tau),
                        "track_record": _value_for_window(table, W, tau)}
               for W in range(1, max_w + 1)}
    return {"as_of": fp["as_of"], "price_now": fp["price_now"], "path": fp["path"],
            "max_wait": max_w, "windows": windows}


def _load_panel():
    pq = Path("data/panel.parquet")
    if pq.exists():
        panel = pd.read_parquet(pq)
        print(f"loaded cached panel {panel.shape}")
        return panel
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    from build_panel import build
    panel = build()
    print(f"built panel {panel.shape}")
    return panel


if __name__ == "__main__":
    panel = _load_panel()
    for W in (1, 2):
        print(f"\n=== max wait = {W} week(s) · threshold = 2 cents/gal ===")
        print(f"{'region':6} {'n':>4} {'%wait':>6} {'save c/gal':>10} {'capture%':>9} {'wait_win%':>9}")
        for r in C.REGIONS:
            dfr = panel[panel["region"] == r].sort_values("date").reset_index(drop=True)
            d = decision_value(dfr, max_wait=W, tau=0.02)
            if d:
                print(f"{r:6} {d['n']:>4} {d['pct_weeks_wait']:>6} {d['save_cents_per_gal']:>10} "
                      f"{str(d['capture_pct']):>9} {str(d['wait_win_rate_pct']):>9}")
