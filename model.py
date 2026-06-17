"""Forecasting models: predict retail gasoline price ``horizon`` weeks ahead, per region.

Two models are compared (plus a naive random-walk baseline computed in update.py):
- ``ridge`` — standardized RidgeCV (linear, interpretable).
- ``gbm``   — HistGradientBoostingRegressor (nonlinear, captures interactions).

Both share one leak-free feature set and one expanding-window backtest:
- ``walk_forward`` -> strictly out-of-sample historical predictions (embargo = ``horizon`` weeks,
  so training targets are never observable after the decision date).
- ``forecast``     -> fit on all known data, predict the final ``horizon`` weeks (the future path).
"""
import os as _os
# Pin to single-thread BEFORE importing numpy/sklearn: the walk-forward does hundreds of tiny
# refits, and OpenMP thread-spawn overhead per fit otherwise dwarfs the actual compute.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_EXOG = ["wti", "brent", "rbob_front", "crack_spread",
         "gasoline_stocks", "refinery_util", "product_supplied", "usd_index"]
_SEASONAL = ["woy_sin", "woy_cos", "summer_blend"]

MODEL_KEYS = ("ridge", "gbm")


def model_factories():
    """Fresh model instances per fit (required for a clean walk-forward)."""
    return {
        "ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 9))),
        "gbm": lambda: HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.07, max_depth=3, min_samples_leaf=20,
            l2_regularization=1.0, early_stopping=False, random_state=0),
    }


def engineer(df, horizon, lags=(1, 2, 3, 4)):
    """Build the feature frame for one region. Columns: date, target_date, y, + features."""
    df = df.sort_values("date").reset_index(drop=True)
    feat = pd.DataFrame({"date": df["date"]})

    for col in _EXOG + _SEASONAL:
        if col in df:
            feat[col] = df[col].values

    feat["retail_t"] = df["retail_regular"].values  # most recent known price (AR term)
    for L in lags:
        feat[f"retail_lag{L}"] = df["retail_regular"].shift(L).values
        feat[f"wti_lag{L}"] = df["wti"].shift(L).values

    # Asymmetric crude/RBOB moves over the last 4 weeks (rockets-and-feathers).
    dwti = df["wti"].diff()
    feat["wti_up_4w"] = dwti.clip(lower=0).rolling(4).sum().values
    feat["wti_down_4w"] = dwti.clip(upper=0).rolling(4).sum().values
    if "rbob_front" in df:
        drbob = df["rbob_front"].diff()
        feat["rbob_up_4w"] = drbob.clip(lower=0).rolling(4).sum().values
        feat["rbob_down_4w"] = drbob.clip(upper=0).rolling(4).sum().values

    feat["y"] = df["retail_regular"].shift(-horizon).values
    feat["target_date"] = df["date"] + pd.to_timedelta(horizon * 7, unit="D")
    return feat


def feature_columns(feat):
    return [c for c in feat.columns if c not in ("date", "y", "target_date")]


def walk_forward(known, cols, factories, min_train=104, refit_every=13, embargo=4):
    """Expanding-window out-of-sample predictions for every model. Returns date + one col/model."""
    X = known[cols].to_numpy(dtype=float)
    y = known["y"].to_numpy(dtype=float)
    td = known["target_date"].to_numpy()
    n = len(known)
    out = {k: [] for k in factories}
    dates = []
    i = max(min_train, embargo + 10)
    while i < n:
        train_end = i - embargo
        if train_end >= 30:
            j = min(i + refit_every, n)
            for k, factory in factories.items():
                preds = factory().fit(X[:train_end], y[:train_end]).predict(X[i:j])
                out[k].extend(preds.tolist())
            dates.extend(td[i:j].tolist())
            i = j
        else:
            i += refit_every
    res = pd.DataFrame({"date": dates})
    for k in factories:
        res[k] = out[k]
    return res


def forecast(known, future, cols, factories):
    """Fit on all known data; predict the future rows (the forward path), per model."""
    res = pd.DataFrame({"date": future["target_date"].values})
    if len(future) == 0:
        return pd.DataFrame(columns=["date", *factories])
    Xk, yk = known[cols].to_numpy(dtype=float), known["y"].to_numpy(dtype=float)
    Xf = future[cols].to_numpy(dtype=float)
    for k, factory in factories.items():
        res[k] = factory().fit(Xk, yk).predict(Xf)
    return res


def run_region(df_region, horizon=4, min_train=104, refit_every=13):
    """Full pipeline for one region. Returns DataFrame (date, <one col per model>, kind), or None.

    ``date`` is the *target* date; ``kind`` is 'oos' (historical, out-of-sample) or 'forecast'.
    """
    factories = model_factories()
    feat = engineer(df_region, horizon)
    cols = feature_columns(feat)
    valid = feat.dropna(subset=cols).reset_index(drop=True)
    known = valid[valid["y"].notna()].reset_index(drop=True)
    future = valid[valid["y"].isna()].reset_index(drop=True)
    if len(known) < min_train + 20:
        return None

    oos = walk_forward(known, cols, factories, min_train, refit_every, embargo=horizon)
    oos["kind"] = "oos"
    fc = forecast(known, future, cols, factories)
    fc["kind"] = "forecast"
    out = pd.concat([oos, fc], ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date").reset_index(drop=True)
