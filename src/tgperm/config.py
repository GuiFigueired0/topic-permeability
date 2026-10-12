"""Paths and fixed analysis settings (see docs/PROJECT_CONTEXT.md, section 5)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tgdata"
DERIVED = DATA / "derived"
LABELS = DATA / "labels"
RESULTS = ROOT / "results"

FORWARDS = (DATA / "forwards" / "*.parquet").as_posix()
TEXTS = (DATA / "texts" / "*.parquet").as_posix()
CHANNELS = (DATA / "channels" / "*.parquet").as_posix()
CHANNEL_MONTH = (DATA / "channel_month" / "*.parquet").as_posix()

SEED = 42
LEIDEN_RESOLUTION = 1.0

# RQ3 windows on the forward's `date` (UTC), inclusive start, exclusive end
INVASION = "2022-02-24"
WINDOW_BEFORE = ("2021-09-24", "2022-02-24")
WINDOW_AFTER = ("2022-02-24", "2022-08-01")
