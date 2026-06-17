"""Look-ahead-safe weekly alignment and panel assembly.

The central idea (``to_weekly_asof``): a value observed for period ``P`` only becomes
usable at ``P + release_lag_days``. For each weekly decision date we take the most recent
value that was already public, so no feature uses information published after the decision.
"""
import numpy as np
import pandas as pd


def to_weekly_asof(s, release_lag_days, anchor="W-FRI"):
    """Resample to a weekly anchor using only values known by each anchor date.

    Each observation's "known date" = period-end + ``release_lag_days``. We forward-fill on a
    daily grid by known date, then sample on the weekly anchor — giving the latest publicly
    available value as of each weekly decision date (no look-ahead).
    """
    s = s.dropna().sort_index()
    if s.empty:
        return s
    known = s.copy()
    known.index = known.index + pd.Timedelta(days=int(release_lag_days))
    daily = known.resample("D").last().ffill()
    return daily.resample(anchor).last()


def master_index(series_list, anchor="W-FRI"):
    """A common weekly DatetimeIndex spanning all provided (already-weekly) series."""
    idxs = [s.index for s in series_list if s is not None and len(s)]
    if not idxs:
        raise RuntimeError("No data available to build a panel index from.")
    start = min(i.min() for i in idxs)
    end = max(i.max() for i in idxs)
    return pd.date_range(start=start, end=end, freq=anchor)


def build_long_panel(idx, target, region_feats, national_feats, regions):
    """Assemble a long (date x region) panel.

    target:         {region -> weekly Series}             retail price (the label)
    region_feats:   {feature -> {region -> weekly Series}} per-region features
    national_feats: {feature -> weekly Series}             broadcast to every region
    """
    frames = []
    for r in regions:
        df = pd.DataFrame(index=idx)
        df.index.name = "date"
        df["region"] = r
        df["retail_regular"] = target[r].reindex(idx) if r in target else np.nan
        for feat, by_region in region_feats.items():
            df[feat] = by_region[r].reindex(idx) if r in by_region else np.nan
        for feat, s in national_feats.items():
            df[feat] = s.reindex(idx)
        frames.append(df.reset_index())
    return pd.concat(frames, ignore_index=True)


def add_calendar_features(panel, date_col="date"):
    """Add seasonal features: cyclical week-of-year, month, and the summer-blend window."""
    d = pd.to_datetime(panel[date_col])
    woy = d.dt.isocalendar().week.astype(int)
    panel["woy_sin"] = np.sin(2 * np.pi * woy / 52.0)
    panel["woy_cos"] = np.cos(2 * np.pi * woy / 52.0)
    panel["month"] = d.dt.month
    # Summer (low-RVP) blend sold at retail roughly Jun 1 - Sep 15.
    doy = d.dt.dayofyear
    year = d.dt.year.astype(str)
    jun1 = pd.to_datetime(year + "-06-01").dt.dayofyear
    sep15 = pd.to_datetime(year + "-09-15").dt.dayofyear
    panel["summer_blend"] = ((doy >= jun1) & (doy <= sep15)).astype(int)
    return panel


def make_supervised(panel, horizon=4, target="retail_regular",
                    date_col="date", region_col="region"):
    """Build a supervised frame for h-week-ahead forecasting.

    ``y`` is ``target`` ``horizon`` weeks in the future, computed within each region; every
    other column is information known as of ``date``. Rows whose future target is unknown are
    dropped. NOTE: do not feed contemporaneous ``retail_regular`` as a predictor — use lags of
    it only, otherwise the label leaks.
    """
    p = panel.sort_values([region_col, date_col]).copy()
    p["y"] = p.groupby(region_col)[target].shift(-horizon)
    p["target_date"] = p.groupby(region_col)[date_col].shift(-horizon)
    return p.dropna(subset=["y"]).reset_index(drop=True)
