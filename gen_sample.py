"""Generate a synthetic site/data/predictions.json so the webapp previews without API keys.

This is a DEV/PREVIEW utility only — the numbers are fabricated (retail is built from a lagged
RBOB proxy so the model shows realistic skill). The first real `update.py` or workflow run
overwrites it with live EIA + FRED data.

    python gen_sample.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from update import make_payload

_REGION_OFFSET = {"US": 0.0, "PADD1": 0.05, "PADD2": -0.02,
                  "PADD3": -0.15, "PADD4": 0.10, "PADD5": 0.95}


def synthetic_panel(seed=0):
    dates = pd.date_range("2016-01-01", "2026-06-12", freq="W-FRI")
    n = len(dates)
    rng = np.random.default_rng(seed)
    wti = np.clip(55 + np.cumsum(rng.normal(0, 1.6, n)), 25, 130)
    rbob = wti * 0.024 + 0.65 + rng.normal(0, 0.04, n)
    woy = dates.isocalendar().week.to_numpy()
    season = np.sin(2 * np.pi * woy / 52.0)
    rbob_s = pd.Series(rbob)

    frames = []
    for r in C.REGIONS:
        retail = (0.95 + 1.0 * rbob_s.shift(2).to_numpy() + 0.12 * season
                  + _REGION_OFFSET[r] + rng.normal(0, 0.03, n))
        frames.append(pd.DataFrame({
            "date": dates, "region": r, "retail_regular": retail,
            "wti": wti, "brent": wti + 4, "rbob_front": rbob, "crack_spread": rbob * 42 - wti,
            "gasoline_stocks": 230000 + rng.normal(0, 5000, n),
            "refinery_util": 90 + 5 * season + rng.normal(0, 2, n),
            "product_supplied": 9000 + 300 * season + rng.normal(0, 150, n),
            "usd_index": 110 + np.cumsum(rng.normal(0, 0.1, n)),
            "woy_sin": np.sin(2 * np.pi * woy / 52.0),
            "woy_cos": np.cos(2 * np.pi * woy / 52.0),
            "summer_blend": ((dates.dayofyear >= 152) & (dates.dayofyear <= 258)).astype(int),
        }))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    payload = make_payload(synthetic_panel(), horizon=4)
    payload["synthetic"] = True
    out = Path("site/data/predictions.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"Wrote synthetic {out} — {len(payload['regions'])} regions.")
