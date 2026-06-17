"""Regenerate the webapp's data file: build panel -> run models per region -> write JSON.

    python update.py --out site/data/predictions.json -v

Designed to run unattended (e.g. from a GitHub Action). Needs EIA_API_KEY and FRED_API_KEY.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")  # avoid thread oversubscription across many small refits

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from model import run_region, MODEL_KEYS
from decision import build_decision

log = logging.getLogger("gaspred")

REGION_NAMES = {
    "US": "National (U.S.)",
    "PADD1": "East Coast (PADD 1)",
    "PADD2": "Midwest (PADD 2)",
    "PADD3": "Gulf Coast (PADD 3)",
    "PADD4": "Rocky Mountain (PADD 4)",
    "PADD5": "West Coast (PADD 5)",
}
MODEL_LABELS = {"ridge": "Ridge (linear)", "gbm": "Gradient boosting"}


def _num(x, nd=4):
    """JSON-safe number: None for NaN, else a rounded float."""
    if x is None or pd.isna(x):
        return None
    return round(float(x), nd)


def _err_metrics(pred, actual):
    err = pred - actual
    return {"mae": _num(err.abs().mean()),
            "rmse": _num(np.sqrt((err ** 2).mean())),
            "mape_pct": _num((err.abs() / actual).mean() * 100, 2)}


def _region_metrics(merged, horizon):
    """Per-model out-of-sample error + skill vs. a naive random-walk baseline."""
    oos = merged[(merged["kind"] == "oos") & merged["actual"].notna()]
    naive_oos = oos[oos["naive"].notna()]
    naive_mae = _num((naive_oos["naive"] - naive_oos["actual"]).abs().mean()) if len(naive_oos) >= 5 else None

    metrics = {}
    for k in MODEL_KEYS:
        mk = oos[oos[k].notna()]
        if len(mk) >= 5:
            em = _err_metrics(mk[k], mk["actual"])
            em["n"] = int(len(mk))
            em["naive_mae"] = naive_mae
            if naive_mae:
                em["skill_vs_naive_pct"] = _num((1 - em["mae"] / naive_mae) * 100, 1)
            metrics[k] = em
        else:
            metrics[k] = {"n": int(len(mk))}

    if len(naive_oos) >= 5:
        nm = _err_metrics(naive_oos["naive"], naive_oos["actual"])
        nm["n"] = int(len(naive_oos))
        metrics["naive"] = nm
    else:
        metrics["naive"] = {"n": int(len(naive_oos))}
    return metrics


def make_payload(panel, horizon=4, min_train=104, refit_every=13):
    """Build the JSON-serializable payload (all regions, all models) from a weekly panel."""
    regions_out = {}
    for r in [x for x in C.REGIONS if x in set(panel["region"])]:
        dfr = panel[panel["region"] == r].sort_values("date").reset_index(drop=True)
        preds = run_region(dfr, horizon, min_train, refit_every)
        actual = dfr[["date", "retail_regular", "wti"]].rename(
            columns={"retail_regular": "actual", "wti": "oil"})

        if preds is None:
            merged = actual.copy()
            for k in MODEL_KEYS:
                merged[k] = np.nan
            merged["kind"] = None
            metrics = {k: {"n": 0} for k in (*MODEL_KEYS, "naive")}
        else:
            merged = actual.merge(preds, on="date", how="outer")
            merged["naive"] = merged.sort_values("date")["actual"].shift(horizon)
            metrics = _region_metrics(merged, horizon)

        merged = merged.sort_values("date").reset_index(drop=True)
        series = []
        for _, row in merged.iterrows():
            rec = {"date": row["date"].strftime("%Y-%m-%d"),
                   "oil": _num(row["oil"], 2), "actual": _num(row["actual"], 3)}
            for k in MODEL_KEYS:
                rec[k] = _num(row[k], 3)
            series.append(rec)

        regions_out[r] = {"name": REGION_NAMES.get(r, r), "models": list(MODEL_KEYS),
                          "model_labels": MODEL_LABELS, "metrics": metrics,
                          "decision": build_decision(dfr, max_w=4) or {}, "series": series}
        log.info("  %-6s rows=%d  ridge_mae=%s  gbm_mae=%s  naive_mae=%s", r, len(series),
                 metrics.get("ridge", {}).get("mae"), metrics.get("gbm", {}).get("mae"),
                 metrics.get("naive", {}).get("mae"))

    return {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "horizon_weeks": horizon,
        "region_order": [r for r in C.REGIONS if r in regions_out],
        "regions": regions_out,
    }


def main():
    ap = argparse.ArgumentParser(description="Build the webapp's predictions.json")
    ap.add_argument("--out", default="site/data/predictions.json")
    ap.add_argument("--start", default=None)
    ap.add_argument("--horizon", type=int, default=4, help="forecast horizon in weeks")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    logging.getLogger("gaspred").setLevel(logging.INFO)

    from build_panel import build
    panel = build(start=args.start or C.DEFAULT_START)
    payload = make_payload(panel, horizon=args.horizon)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")))
    n = sum(len(v["series"]) for v in payload["regions"].values())
    print(f"Wrote {out} — {len(payload['regions'])} regions, {n:,} points, "
          f"models {', '.join(MODEL_KEYS)}, horizon {args.horizon}w")


def _load_env():
    """Load .env for local runs (CI passes secrets as env vars directly). Works without dotenv."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return
    except ImportError:
        pass
    from pathlib import Path
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


if __name__ == "__main__":
    _load_env()
    main()
