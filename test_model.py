"""Model + payload tests on synthetic data (no keys/network). Run: python test_model.py"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from gen_sample import synthetic_panel
from update import make_payload
from model import MODEL_KEYS


def test_payload_shape_forecast_and_models():
    # refit_every=26 keeps this fast; the orchestration default is finer.
    payload = make_payload(synthetic_panel(), horizon=4, min_train=104, refit_every=26)
    assert {"US", "PADD5"}.issubset(payload["regions"]), "expected national + PADD regions"

    us = payload["regions"]["US"]
    assert us["models"] == list(MODEL_KEYS)
    assert len(us["series"]) > 200

    # Every model (plus naive) has metrics; each series row carries every model's prediction.
    for k in (*MODEL_KEYS, "naive"):
        assert k in us["metrics"], f"missing metrics for {k}"
    for k in MODEL_KEYS:
        assert k in us["series"][300], f"missing series column for {k}"
        assert us["metrics"][k]["skill_vs_naive_pct"] is not None, f"{k} missing skill metric"

    # Forecast extends past the last actual.
    actual_dates = [x["date"] for x in us["series"] if x["actual"] is not None]
    pred_dates = [x["date"] for x in us["series"] if x["ridge"] is not None]
    assert max(pred_dates) > max(actual_dates), "forecast should extend past the last actual"

    # The synthetic DGP is linear, so Ridge should clearly beat the naive baseline.
    # (GBM may not on linear data — that's exactly what the comparison surfaces.)
    assert us["metrics"]["ridge"]["skill_vs_naive_pct"] > 0, \
        f"ridge should beat naive on linear synthetic data: {us['metrics']['ridge']}"
    print("ok test_payload_shape_forecast_and_models",
          {k: us["metrics"][k]["mae"] for k in (*MODEL_KEYS, "naive")})


if __name__ == "__main__":
    test_payload_shape_forecast_and_models()
    print("all model tests passed")
