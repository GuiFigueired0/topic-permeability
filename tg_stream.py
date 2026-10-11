"""
tg_stream.py - stream TGDataset straight from Zenodo and keep only what the
topic-permeability project needs. Nothing but the small outputs touches disk.

Pipeline for each of the four archives (TGDataset_N.tar.gz):
    HTTPS bytes  ->  gzip  ->  tar members (one JSON file per ~1,000 channels)
                 ->  incremental JSON parser (ijson)  ->  rows  ->  Parquet

Outputs (one set of files per JSON member, so work is resumable):
    out/forwards/       every forwarded message, all channels (text + media):
                        dest_id, msg_id, kind, date, src_id, orig_date
    out/texts/          text of forwarded TEXT messages posted in English channels:
                        dest_id, msg_id, date, src_id, orig_date, text
    out/channels/       one row per channel: metadata + message/forward counts
    out/channel_month/  messages and forwards per channel per month (UTC)
    out/_done/          one marker per finished member (used to resume)
    out/_logs/          progress log per archive

Restarting is safe: finished members are skipped (the archive is still
streamed, because a .tar.gz can only be read front to back, but their JSON is
not parsed). A dropped connection is resumed in place with an HTTP Range
request, so the decompressor never notices.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path

import ijson
import pyarrow as pa
import pyarrow.parquet as pq
import requests

ZENODO_RECORD = "https://zenodo.org/records/7640712/files"
PARTS = {
    1: f"{ZENODO_RECORD}/TGDataset_1.tar.gz?download=1",
    2: f"{ZENODO_RECORD}/TGDataset_2.tar.gz?download=1",
    3: f"{ZENODO_RECORD}/TGDataset_3.tar.gz?download=1",
    4: f"{ZENODO_RECORD}/TGDataset_4.tar.gz?download=1",
}
PART_SIZES_GB = {1: 19.7, 2: 20.4, 3: 21.1, 4: 9.5}  # compressed, from Zenodo

LABELS_URL = "https://raw.githubusercontent.com/SystemsLab-Sapienza/TGDataset/main/labeled_data"
LABEL_FILES = [
    "channel_to_language_mapping.csv",
    "ch_to_topic_mapping.csv",
    "conspiracy_channels.csv",
    "sabmyk_network.csv",
]

FLUSH_ROWS = 200_000  # rows buffered in memory before a Parquet row group is written


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------

def download_labels(out_dir: str | Path) -> Path:
    """Download the four small label CSVs from the TGDataset GitHub repository."""
    label_dir = Path(out_dir) / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    for name in LABEL_FILES:
        target = label_dir / name
        if target.exists() and target.stat().st_size > 0:
            continue
        r = requests.get(f"{LABELS_URL}/{name}", timeout=60)
        r.raise_for_status()
        target.write_bytes(r.content)
    return label_dir


def load_language_ids(label_dir: str | Path, language: str = "en") -> set[int]:
    """Channel ids whose detected language is `language` (tab-separated file)."""
    path = Path(label_dir) / "channel_to_language_mapping.csv"
    with open(path, newline="", encoding="utf-8") as f:
        return {int(r["ch_id"]) for r in csv.DictReader(f, delimiter="\t") if r["language"] == language}


# --------------------------------------------------------------------------
# A byte stream over HTTPS that survives dropped connections
# --------------------------------------------------------------------------

class ResumableHTTPStream:
    """File-like, read-only stream of a remote file.

    If the connection drops, it reconnects with `Range: bytes=<pos>-` and
    continues exactly where it stopped. If a server ignores Range (answers 200),
    the already-read bytes are re-downloaded and discarded, which is slower but
    still correct.
    """

    def __init__(self, url: str, chunk: int = 1 << 20, max_retries: int = 50,
                 timeout: float = 120, log: logging.Logger | None = None):
        self.url, self.chunk, self.max_retries, self.timeout = url, chunk, max_retries, timeout
        self.log = log or logging.getLogger(__name__)
        self.pos = 0          # bytes handed to the reader
        self.received = 0     # bytes received from the network (resume point)
        self.total: int | None = None
        self._buf = b""
        self._iter = None
        self._session = requests.Session()
        self._connect()

    def _connect(self) -> None:
        headers = {"Range": f"bytes={self.received}-"} if self.received else {}
        resp = self._session.get(self.url, headers=headers, stream=True,
                                 timeout=self.timeout, allow_redirects=True)
        resp.raise_for_status()
        if self.total is None:
            length = resp.headers.get("Content-Length")
            self.total = int(length) if length else None
        it = resp.iter_content(chunk_size=self.chunk)
        self._skip = self.received if (self.received and resp.status_code == 200) else 0
        if self._skip:  # Range ignored: the bytes we already have will be dropped in _next_block
            self.log.warning("server ignored Range; re-reading %d bytes to resume", self._skip)
        self._iter = it

    def _next_block(self) -> bytes:
        """Next block of new bytes from the network; b'' only at the true end."""
        failures = 0
        while True:
            try:
                block = next(self._iter)
            except StopIteration:
                if self.total is None or self.received >= self.total:
                    return b""
                problem = f"stream ended early at byte {self.received} of {self.total}"
            except (requests.RequestException, ConnectionError, OSError) as exc:
                problem = f"{type(exc).__name__} at byte {self.received}"
            else:
                if self._skip:  # server ignored Range: drop bytes we already have
                    drop = min(self._skip, len(block))
                    self._skip -= drop
                    block = block[drop:]
                    if not block:
                        continue
                self.received += len(block)
                return block

            failures += 1
            if failures > self.max_retries:
                raise IOError(f"gave up after {self.max_retries} retries: {problem}")
            wait = min(60, 2 ** min(failures, 6))
            self.log.warning("%s; reconnecting in %ds (attempt %d)", problem, wait, failures)
            time.sleep(wait)
            try:
                self._connect()
            except (requests.RequestException, ConnectionError, OSError) as exc2:
                self.log.warning("reconnect failed: %s", exc2)

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            parts = [self._buf]
            self._buf = b""
            while block := self._next_block():
                parts.append(block)
            data = b"".join(parts)
            self.pos += len(data)
            return data
        while len(self._buf) < n:
            block = self._next_block()
            if not block:
                break
            self._buf += block
        data, self._buf = self._buf[:n], self._buf[n:]
        self.pos += len(data)
        return data

    def readable(self) -> bool:
        return True

    def close(self) -> None:
        self._session.close()


# --------------------------------------------------------------------------
# Parquet writing
# --------------------------------------------------------------------------

SCHEMAS = {
    "forwards": pa.schema([
        ("dest_id", pa.int64()), ("msg_id", pa.int64()), ("kind", pa.string()),
        ("date", pa.int64()), ("src_id", pa.int64()), ("orig_date", pa.int64()),
    ]),
    "texts": pa.schema([
        ("dest_id", pa.int64()), ("msg_id", pa.int64()), ("date", pa.int64()),
        ("src_id", pa.int64()), ("orig_date", pa.int64()), ("text", pa.string()),
    ]),
    "channels": pa.schema([
        ("channel_id", pa.int64()), ("username", pa.string()), ("title", pa.string()),
        ("description", pa.string()), ("scam", pa.bool_()), ("verified", pa.bool_()),
        ("n_subscribers", pa.int64()), ("creation_date", pa.int64()),
        ("n_text", pa.int64()), ("n_media", pa.int64()),
        ("n_text_fwd", pa.int64()), ("n_media_fwd", pa.int64()),
        ("n_fwd_hidden_src", pa.int64()),
        ("first_date", pa.int64()), ("last_date", pa.int64()),
        ("part", pa.int8()), ("member", pa.string()),
    ]),
    "channel_month": pa.schema([
        ("channel_id", pa.int64()), ("month", pa.int32()),
        ("n_msgs", pa.int64()), ("n_fwd", pa.int64()),
    ]),
}


class TableBuffer:
    """Collects rows column-wise and writes them as Parquet row groups."""

    def __init__(self, name: str, path: Path):
        self.schema = SCHEMAS[name]
        self.cols = {f.name: [] for f in self.schema}
        self.path = path
        self.writer: pq.ParquetWriter | None = None
        self.rows = 0

    def add(self, *values) -> None:
        for col, v in zip(self.cols.values(), values):
            col.append(v)
        self.rows += 1
        if len(next(iter(self.cols.values()))) >= FLUSH_ROWS:
            self.flush()

    def flush(self) -> None:
        if not next(iter(self.cols.values())):
            return
        table = pa.table({k: pa.array(v, type=self.schema.field(k).type) for k, v in self.cols.items()},
                         schema=self.schema)
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")
        self.writer.write_table(table)
        for v in self.cols.values():
            v.clear()

    def close(self) -> None:
        self.flush()
        if self.writer is None:  # always leave a (possibly empty) file
            self.path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(self.schema.empty_table(), self.path, compression="zstd")
        else:
            self.writer.close()


def _int(x):
    return None if x is None else int(x)


# --------------------------------------------------------------------------
# The parser: one JSON member -> rows
# --------------------------------------------------------------------------

@dataclass
class MemberStats:
    channels: int = 0
    messages: int = 0
    forwards: int = 0
    texts: int = 0


def parse_member(fobj, part: int, member: str, out_dir: Path, english_ids: set[int]) -> MemberStats:
    """Stream one TGDataset JSON file ({channel_id: {...}}) and write its rows.

    The JSON is never loaded whole: ijson emits one event at a time and a small
    state machine turns them into rows, so memory stays flat whatever the size
    of a channel.
    """
    stem = f"part{part}__{Path(member).stem}"
    tmp = out_dir / "_tmp" / stem
    bufs = {name: TableBuffer(name, tmp / f"{name}.parquet") for name in SCHEMAS}
    fwd_add, txt_add = bufs["forwards"].add, bufs["texts"].add
    stats = MemberStats()

    day_to_month: dict[int, int] = {}

    def month_of(ts: float) -> int:
        day = int(ts // 86400)
        m = day_to_month.get(day)
        if m is None:
            t = time.gmtime(day * 86400)
            m = day_to_month[day] = t.tm_year * 100 + t.tm_mon
        return m

    depth = 0          # current map depth (1 = root, 2 = channel, 3 = section, 4 = message)
    skip = 0           # depth of containers we ignore (arrays, unexpected nesting)
    k1 = k2 = k3 = k4 = None
    ch_id = None
    meta: dict = {}
    msg: dict = {}
    counts: dict = {}
    months: dict = {}
    is_en = False

    for ev, val in ijson.basic_parse(fobj, use_float=True, buf_size=1 << 20):
        if skip:
            if ev in ("start_map", "start_array"):
                skip += 1
            elif ev in ("end_map", "end_array"):
                skip -= 1
            continue

        if ev == "map_key":
            if depth == 4:
                k4 = val
            elif depth == 3:
                k3 = val
            elif depth == 2:
                k2 = val
            else:
                k1 = val

        elif ev == "start_map":
            if depth >= 4:            # nested object inside a message: ignore it
                skip = 1
                continue
            depth += 1
            if depth == 2:            # a new channel begins
                ch_id = int(k1)
                is_en = ch_id in english_ids
                meta = {}
                counts = {"text_messages": 0, "generic_media": 0, "fwd_text_messages": 0,
                          "fwd_generic_media": 0, "hidden": 0, "first": None, "last": None}
                months = {}
            elif depth == 4:
                msg = {}

        elif ev == "end_map":
            if depth == 4:            # a message is complete
                section = k2
                counts[section] = counts.get(section, 0) + 1
                stats.messages += 1
                ts = msg.get("date")
                if ts is not None:
                    if counts["first"] is None or ts < counts["first"]:
                        counts["first"] = ts
                    if counts["last"] is None or ts > counts["last"]:
                        counts["last"] = ts
                    m = month_of(ts)
                    mc = months.get(m)
                    if mc is None:
                        mc = months[m] = [0, 0]
                    mc[0] += 1
                if msg.get("is_forwarded"):
                    counts["fwd_" + section] = counts.get("fwd_" + section, 0) + 1
                    src = msg.get("forwarded_from_id")
                    if src is None:
                        counts["hidden"] += 1
                    if ts is not None:
                        mc[1] += 1
                    kind = "text" if section == "text_messages" else "media"
                    fwd_add(ch_id, int(k3), kind, _int(ts), _int(src), _int(msg.get("forwarded_message_date")))
                    stats.forwards += 1
                    if is_en and kind == "text":
                        text = msg.get("message")
                        if text:
                            txt_add(ch_id, int(k3), _int(ts), _int(src),
                                    _int(msg.get("forwarded_message_date")), text)
                            stats.texts += 1
            elif depth == 2:          # a channel is complete
                bufs["channels"].add(
                    ch_id, meta.get("username"), meta.get("title"), meta.get("description"),
                    meta.get("scam"), meta.get("verified"), _int(meta.get("n_subscribers")),
                    _int(meta.get("creation_date")),
                    counts["text_messages"], counts["generic_media"],
                    counts["fwd_text_messages"], counts["fwd_generic_media"], counts["hidden"],
                    _int(counts["first"]), _int(counts["last"]), part, member,
                )
                cm_add = bufs["channel_month"].add
                for m, (n_msgs, n_fwd) in months.items():
                    cm_add(ch_id, m, n_msgs, n_fwd)
                stats.channels += 1
            depth -= 1

        elif ev == "start_array":
            skip = 1

        else:                          # a scalar value
            if depth == 4:
                msg[k4] = val
            elif depth == 2:
                meta[k2] = val

    for b in bufs.values():
        b.close()

    # move finished files into place, then mark the member as done
    for name in SCHEMAS:
        final = out_dir / name / f"{stem}.parquet"
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp / f"{name}.parquet", final)
    tmp.rmdir()
    done = out_dir / "_done" / f"{stem}.json"
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps({"part": part, "member": member, **stats.__dict__}))
    return stats


# --------------------------------------------------------------------------
# One archive, front to back
# --------------------------------------------------------------------------

def _logger(out_dir: Path, part: int) -> logging.Logger:
    log = logging.getLogger(f"tgdataset.part{part}")
    if not log.handlers:
        (out_dir / "_logs").mkdir(parents=True, exist_ok=True)
        h = logging.FileHandler(out_dir / "_logs" / f"part{part}.log", encoding="utf-8")
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log


def process_part(part: int, out_dir: str, url: str | None = None) -> dict:
    """Stream archive `part` (1-4) and write rows for every JSON member in it.

    Safe to call again after an interruption: finished members are skipped.
    Returns a summary dict. Runs in its own process when called in parallel.
    """
    out = Path(out_dir)
    label_dir = download_labels(out)
    english_ids = load_language_ids(label_dir, "en")
    log = _logger(out, part)
    # TGDATASET_URL_TEMPLATE (e.g. "https://mirror/TGDataset_{part}.tar.gz") overrides Zenodo
    template = os.environ.get("TGDATASET_URL_TEMPLATE")
    url = url or (template.format(part=part) if template else PARTS[part])
    log.info("start part %d: %s", part, url)

    stream = ResumableHTTPStream(url, log=log)
    totals = MemberStats()
    t0 = time.time()
    n_members = n_skipped = 0
    try:
        with tarfile.open(fileobj=stream, mode="r|gz") as tar:
            for member in tar:
                if not (member.isfile() and member.name.endswith(".json")):
                    continue
                n_members += 1
                stem = f"part{part}__{Path(member.name).stem}"
                if (out / "_done" / f"{stem}.json").exists():
                    n_skipped += 1
                    log.info("skip %s (already done)", member.name)
                    continue  # tar streams past it without parsing
                tm = time.time()
                fobj = tar.extractfile(member)
                st = parse_member(fobj, part, member.name, out, english_ids)
                for k in totals.__dict__:
                    setattr(totals, k, getattr(totals, k) + getattr(st, k))
                gb = stream.pos / 1e9
                rate = gb * 1000 / max(time.time() - t0, 1)
                log.info("done %s: %d channels, %d messages, %d forwards, %d texts in %.0fs | "
                         "%.2f / %s GB read (%.1f MB/s)", member.name, st.channels, st.messages,
                         st.forwards, st.texts, time.time() - tm, gb, PART_SIZES_GB[part], rate)
    finally:
        stream.close()
    summary = {"part": part, "members": n_members, "skipped": n_skipped,
               "hours": round((time.time() - t0) / 3600, 2), **totals.__dict__}
    log.info("finished part %d: %s", part, summary)
    (out / "_done" / f"part{part}__ALL.json").write_text(json.dumps(summary))
    return summary


if __name__ == "__main__":  # command-line use: python tg_stream.py OUT_DIR [PART ...]
    import sys
    from concurrent.futures import ProcessPoolExecutor

    out_dir = sys.argv[1] if len(sys.argv) > 1 else "tgdata"
    parts = [int(p) for p in sys.argv[2:]] or [1, 2, 3, 4]
    download_labels(out_dir)
    with ProcessPoolExecutor(max_workers=len(parts)) as ex:
        for s in ex.map(process_part, parts, [out_dir] * len(parts)):
            print(s)
