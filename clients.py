"""Data-source clients. Each fetcher returns a tidy ``pd.Series`` indexed by date.

- EIA   : v2 /seriesid/ path (accepts full APIv1 ids); needs EIA_API_KEY.
- FRED  : series/observations endpoint; needs FRED_API_KEY.
- Futures: yfinance (no key; optional dependency).
"""
import os

import pandas as pd
import requests

EIA_SERIESID_BASE = "https://api.eia.gov/v2/seriesid/"
FRED_OBS_URL = "https://api.stlouisfed.org/fred/series/observations"

# Non-numeric metadata keys that may appear in an EIA record alongside the value.
_EIA_META_KEYS = {"period", "series", "series-description", "units", "area-name",
                  "product", "product-name", "process", "process-name", "duoarea"}


def _need(key, name):
    if not key:
        raise RuntimeError(f"Missing {name}; set it in the environment or a .env file.")
    return key


def fetch_eia(series_id, api_key=None, length=5000, timeout=30):
    """Fetch a weekly EIA series via the v2 /seriesid/ compatibility path."""
    api_key = _need(api_key or os.getenv("EIA_API_KEY"), "EIA_API_KEY")
    resp = requests.get(EIA_SERIESID_BASE + series_id,
                        params={"api_key": api_key, "length": length}, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    data = (payload.get("response") or {}).get("data") or []
    if not data:
        raise RuntimeError(f"EIA returned no rows for {series_id}: {payload.get('error', payload)}")
    out = {}
    for rec in data:
        period = rec.get("period")
        if period is None:
            continue
        val = rec.get("value")
        if val is None:  # fall back to the first numeric non-metadata field
            for k, v in rec.items():
                if k not in _EIA_META_KEYS and isinstance(v, (int, float)):
                    val = v
                    break
        if val in (None, ""):
            continue
        try:
            out[period] = float(val)
        except (TypeError, ValueError):
            continue
    s = pd.Series(out, name=series_id).sort_index()
    s.index = pd.to_datetime(s.index)
    return s


def fetch_fred(series_id, api_key=None, start=None, timeout=30):
    """Fetch a FRED series. Missing observations (value '.') are dropped."""
    api_key = _need(api_key or os.getenv("FRED_API_KEY"), "FRED_API_KEY")
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    if start:
        params["observation_start"] = start
    resp = requests.get(FRED_OBS_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    obs = resp.json().get("observations", [])
    out = {o["date"]: float(o["value"]) for o in obs if o.get("value") not in (None, ".", "")}
    s = pd.Series(out, name=series_id).sort_index()
    s.index = pd.to_datetime(s.index)
    return s


def fetch_futures(ticker, start=None):
    """Fetch daily closing prices for a futures ticker via yfinance."""
    try:
        import yfinance as yf
    except ImportError as e:
        raise RuntimeError("yfinance not installed; run `pip install yfinance`.") from e
    df = yf.download(ticker, start=start, progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance may return MultiIndex columns
        close = close.iloc[:, 0]
    close = close.dropna()
    close.index = pd.to_datetime(close.index)
    close.name = ticker
    return close
