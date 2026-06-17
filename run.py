"""CLI entrypoint: build the weekly national + PADD gasoline-price panel and save it.

    python run.py --start 2015-01-01 --out data/panel.parquet -v
"""
import argparse
import logging
import os
from pathlib import Path


def _load_dotenv():
    """Load .env via python-dotenv if present, else a tiny built-in parser."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return
    except ImportError:
        pass
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def main():
    ap = argparse.ArgumentParser(description="Build a weekly U.S. national + PADD gasoline-price panel.")
    ap.add_argument("--start", default=None, help="ISO start date (default from config)")
    ap.add_argument("--out", default="data/panel.parquet", help="output path (.parquet or .csv)")
    ap.add_argument("--anchor", default=None, help="weekly anchor, e.g. W-FRI")
    ap.add_argument("-v", "--verbose", action="store_true", help="log each series fetch")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    logging.getLogger("gaspred").setLevel(logging.INFO)

    _load_dotenv()
    import config as C
    from build_panel import build

    panel = build(start=args.start or C.DEFAULT_START, anchor=args.anchor or C.WEEK_ANCHOR)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".csv":
        panel.to_csv(out, index=False)
    else:
        panel.to_parquet(out, index=False)

    print(f"\nSaved {len(panel):,} rows x {panel.shape[1]} cols -> {out}")
    print(f"Date range : {panel['date'].min().date()} .. {panel['date'].max().date()}")
    print(f"Regions    : {', '.join(panel['region'].unique())}")
    print(f"Columns    : {', '.join(panel.columns)}")
    cov = panel.groupby("region")["retail_regular"].apply(lambda s: round(s.notna().mean() * 100, 1))
    print("\nTarget (retail_regular) coverage by region (%):")
    print(cov.to_string())


if __name__ == "__main__":
    main()
