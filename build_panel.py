"""Orchestrate fetching + alignment into a weekly national + PADD panel."""
import logging

import pandas as pd

import config as C
from clients import fetch_eia, fetch_fred, fetch_futures
from align import to_weekly_asof, master_index, build_long_panel, add_calendar_features

log = logging.getLogger("gaspred")


def _splice_tail(spot, fut):
    """Fill a spot series' trailing NaNs using the futures path, anchored to spot's last known
    level. FRED crude spot publishes ~a week late; the futures front-month is current and nearly
    co-moves, so this carries the latest moves onto the spot level without a basis jump. Only the
    trailing gap is touched — no look-ahead (futures values are known daily)."""
    if fut is None:
        return spot
    lv = spot.last_valid_index()
    if lv is None:
        return spot
    fut_to_lv = fut.loc[:lv].dropna()
    if fut_to_lv.empty:
        return spot
    base = fut_to_lv.iloc[-1]
    out = spot.copy()
    for t in spot.index[(spot.index > lv) & spot.isna()]:
        if t in fut.index and pd.notna(fut[t]):
            out[t] = spot[lv] + (fut[t] - base)
    return out


def _safe(fetch, label):
    """Run a fetch, returning None (and logging) on failure so one bad series can't abort the run."""
    try:
        s = fetch()
        log.info("  ok: %-28s %d rows", label, len(s))
        return s
    except Exception as e:
        log.warning("  FAILED: %-25s %s", label, e)
        return None


def build(start=C.DEFAULT_START, anchor=C.WEEK_ANCHOR):
    log.info("Fetching EIA retail prices (target)...")
    target = {}
    for r, sid in C.EIA_RETAIL.items():
        s = _safe(lambda sid=sid: fetch_eia(sid), f"retail {r}")
        if s is not None:
            target[r] = to_weekly_asof(s, C.LAG_RETAIL_DAYS, anchor)
    if not target:
        raise RuntimeError("No retail target series fetched; cannot build a panel.")

    log.info("Fetching EIA region features (stocks, refinery utilization)...")
    region_feats = {"gasoline_stocks": {}, "refinery_util": {}}
    for r, sid in C.EIA_GAS_STOCKS.items():
        s = _safe(lambda sid=sid: fetch_eia(sid), f"stocks {r}")
        if s is not None:
            region_feats["gasoline_stocks"][r] = to_weekly_asof(s, C.LAG_WPSR_DAYS, anchor)
    for r, sid in C.EIA_REFINERY_UTIL.items():
        s = _safe(lambda sid=sid: fetch_eia(sid), f"refinery_util {r}")
        if s is not None:
            region_feats["refinery_util"][r] = to_weekly_asof(s, C.LAG_WPSR_DAYS, anchor)

    log.info("Fetching national drivers (crude, USD, futures, demand)...")
    # (raw series, release-lag) keyed by feature name; all broadcast to every region.
    national_raw = {
        "wti":              (_safe(lambda: fetch_fred(C.FRED_WTI, start=start), "FRED WTI"), C.LAG_DAILY_DAYS),
        "brent":            (_safe(lambda: fetch_fred(C.FRED_BRENT, start=start), "FRED Brent"), C.LAG_DAILY_DAYS),
        "usd_index":        (_safe(lambda: fetch_fred(C.FRED_USD_INDEX, start=start), "FRED USD index"), C.LAG_DAILY_DAYS),
        "product_supplied": (_safe(lambda: fetch_eia(C.EIA_PRODUCT_SUPPLIED), "product supplied (US)"), C.LAG_WPSR_DAYS),
    }
    for name, tk in C.YF_TICKERS.items():
        national_raw[name] = (_safe(lambda tk=tk: fetch_futures(tk, start=start), f"futures {name}"), C.LAG_DAILY_DAYS)

    national_feats = {name: to_weekly_asof(s, lag, anchor)
                      for name, (s, lag) in national_raw.items() if s is not None}

    all_series = list(target.values())
    for d in region_feats.values():
        all_series += list(d.values())
    all_series += list(national_feats.values())
    idx = master_index(all_series, anchor)
    if start:
        idx = idx[idx >= pd.Timestamp(start)]

    # Freshness: FRED crude spot publishes ~a week late, leaving the most recent weeks blank.
    # Extend the spot tail with the (current) futures front-month, then carry any remaining national
    # daily series (e.g. USD index) forward a couple of weeks so the latest week is decision-ready.
    # Still leak-free: ffill only propagates the last *published* value, never a future one.
    for name in national_feats:
        national_feats[name] = national_feats[name].reindex(idx)
    for spot, fut in (("wti", "wti_front"), ("brent", "brent_front")):
        if spot in national_feats and fut in national_feats:
            national_feats[spot] = _splice_tail(national_feats[spot], national_feats[fut])
    if "rbob_front" in national_feats and "wti" in national_feats:
        national_feats["crack_spread"] = national_feats["rbob_front"] * 42.0 - national_feats["wti"]
    for name in list(national_feats):
        national_feats[name] = national_feats[name].ffill(limit=2)

    panel = build_long_panel(idx, target, region_feats, national_feats, C.REGIONS)
    panel = add_calendar_features(panel)
    return panel.sort_values(["region", "date"]).reset_index(drop=True)
