"""Tests for the look-ahead-safe alignment. Run: python test_align.py

These exercise the pure-pandas logic and need no API keys or network.
"""
import pandas as pd

from align import (to_weekly_asof, build_long_panel, add_calendar_features,
                   make_supervised)


def test_no_lookahead():
    # Weekly Friday observations; a spike on the obs dated 2020-03-13 (a Friday).
    idx = pd.date_range("2020-02-07", "2020-04-10", freq="W-FRI")
    s = pd.Series(1.0, index=idx)
    s.loc["2020-03-13"] = 99.0
    w = to_weekly_asof(s, release_lag_days=5, anchor="W-FRI")
    # Known date = 2020-03-13 + 5 = 2020-03-18 (Wed) -> first visible at the 2020-03-20 anchor.
    assert w.loc["2020-03-13"] != 99.0, "value leaked on its own period date (before release)"
    assert w.loc["2020-03-20"] == 99.0, "value should be known by the following Friday"
    print("ok test_no_lookahead")


def test_asof_uses_latest_known():
    idx = pd.date_range("2021-01-01", "2021-02-26", freq="W-FRI")
    s = pd.Series(range(len(idx)), index=idx, dtype=float)
    w = to_weekly_asof(s, release_lag_days=1, anchor="W-FRI")
    # Lag 1 day: a Friday obs is known Saturday, so it surfaces at the *next* Friday anchor.
    assert w.loc[idx[1]] == 0.0
    assert w.loc[idx[2]] == 1.0
    print("ok test_asof_uses_latest_known")


def test_supervised_horizon():
    idx = pd.date_range("2022-01-07", periods=10, freq="W-FRI")
    target = {"US": pd.Series(range(10), index=idx, dtype=float)}
    panel = build_long_panel(idx, target, {}, {}, ["US"])
    panel = add_calendar_features(panel)
    sup = make_supervised(panel, horizon=3)
    assert sup.iloc[0]["y"] == 3.0, "y should be the target 3 weeks ahead"
    assert len(sup) == 7, "10 rows minus a 3-week horizon = 7 usable rows"
    print("ok test_supervised_horizon")


def test_calendar_summer_blend():
    idx = pd.to_datetime(["2023-01-06", "2023-07-07", "2023-10-06"])
    panel = build_long_panel(idx, {"US": pd.Series([1.0, 2.0, 3.0], index=idx)}, {}, {}, ["US"])
    panel = add_calendar_features(panel)
    flags = dict(zip(panel["date"].dt.strftime("%Y-%m-%d"), panel["summer_blend"]))
    assert flags["2023-07-07"] == 1 and flags["2023-01-06"] == 0 and flags["2023-10-06"] == 0
    print("ok test_calendar_summer_blend")


if __name__ == "__main__":
    test_no_lookahead()
    test_asof_uses_latest_known()
    test_supervised_horizon()
    test_calendar_summer_blend()
    print("all tests passed")
