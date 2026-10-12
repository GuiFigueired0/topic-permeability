"""DuckDB access to the extracted data: views over the Parquet folders, labels, edge tables.

Nothing here loads `forwards/` or `texts/` into pandas; aggregation happens in DuckDB.
"""
from pathlib import Path

import duckdb
import pandas as pd

from . import config as cfg


def connect(memory_limit: str = "8GB", threads: int | None = None) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with views `forwards`, `texts`, `channels`, `channel_month` and label tables
    `lang`, `ch_topic`, `consp`, `sabmyk` (ids as BIGINT, one row per channel)."""
    con = duckdb.connect()
    con.sql(f"SET memory_limit = '{memory_limit}'")
    con.sql(f"SET temp_directory = '{(cfg.DATA / '_tmp').as_posix()}'")
    if threads:
        con.sql(f"SET threads = {threads}")
    con.sql(f"CREATE VIEW forwards AS SELECT * FROM '{cfg.FORWARDS}'")
    con.sql(f"CREATE VIEW texts AS SELECT * FROM '{cfg.TEXTS}'")
    con.sql(f"CREATE VIEW channels AS SELECT * FROM '{cfg.CHANNELS}'")
    con.sql(f"CREATE VIEW channel_month AS SELECT * FROM '{cfg.CHANNEL_MONTH}'")

    L = cfg.LABELS.as_posix()
    # Each label file has its own id column name and separator (see CLAUDE.md, "Data gotchas").
    con.sql(f"""CREATE TABLE lang AS
        SELECT DISTINCT ch_id AS channel_id, nullif(trim(language), '') AS language
        FROM read_csv('{L}/channel_to_language_mapping.csv', delim='\t', header=true,
                      columns={{'ch_id': 'BIGINT', 'language': 'VARCHAR'}})""")
    con.sql(f"""CREATE TABLE ch_topic AS
        SELECT DISTINCT ch_ID AS channel_id, topic AS ch_topic
        FROM read_csv('{L}/ch_to_topic_mapping.csv', header=true,
                      columns={{'ch_ID': 'BIGINT', 'topic': 'VARCHAR'}})""")
    con.sql(f"""CREATE TABLE consp AS
        SELECT DISTINCT ch_id AS channel_id
        FROM read_csv('{L}/conspiracy_channels.csv', header=true,
                      columns={{'ch_id': 'BIGINT', 'username': 'VARCHAR'}})""")
    con.sql(f"""CREATE TABLE sabmyk AS
        SELECT DISTINCT ch_ID AS channel_id
        FROM read_csv('{L}/sabmyk_network.csv', header=true,
                      columns={{'ch_ID': 'BIGINT', 'username': 'VARCHAR'}})""")
    for t in ("lang", "ch_topic", "consp", "sabmyk"):
        n, n_ids = con.sql(f"SELECT count(*), count(DISTINCT channel_id) FROM {t}").fetchone()
        assert n == n_ids, f"label table {t} has {n - n_ids} channels with more than one row"
    return con


def channel_table(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """One row per channel with metadata and the published labels (unchanged)."""
    return con.sql("""
        SELECT c.channel_id, c.username, c.title, c.n_subscribers,
               c.n_text + c.n_media AS n_msgs, c.n_text_fwd + c.n_media_fwd AS n_fwd,
               l.language, t.ch_topic,
               k.channel_id IS NOT NULL AS conspiracy,
               s.channel_id IS NOT NULL AS sabmyk
        FROM channels c
        LEFT JOIN lang l USING (channel_id)
        LEFT JOIN ch_topic t USING (channel_id)
        LEFT JOIN consp k USING (channel_id)
        LEFT JOIN sabmyk s USING (channel_id)
        ORDER BY c.channel_id""").df()


def build_edges(con: duckdb.DuckDBPyConnection, overwrite: bool = False) -> dict[str, Path]:
    """Write `edges_full.parquet` and `edges_en.parquet` to tgdata/derived/.

    Edge = src_id -> dest_id (content flow), aggregated over forwards whose source is a dataset
    channel other than the destination. Columns: src_id, dest_id, weight (= n_text + n_media),
    n_text, n_media, first_date, last_date (forward `date`, Unix seconds).
    """
    cfg.DERIVED.mkdir(parents=True, exist_ok=True)
    full, en = cfg.DERIVED / "edges_full.parquet", cfg.DERIVED / "edges_en.parquet"
    if overwrite or not full.exists():
        con.sql(f"""COPY (
            SELECT src_id, dest_id,
                   count(*) AS weight,
                   count(*) FILTER (WHERE kind = 'text') AS n_text,
                   count(*) FILTER (WHERE kind = 'media') AS n_media,
                   min(date) AS first_date, max(date) AS last_date
            FROM forwards
            WHERE src_id IN (SELECT channel_id FROM channels) AND src_id <> dest_id
            GROUP BY src_id, dest_id
            ORDER BY src_id, dest_id
        ) TO '{full.as_posix()}' (FORMAT parquet, COMPRESSION zstd)""")
    if overwrite or not en.exists():
        # Aggregation is per (src, dest) pair, so filtering the full edge table is equivalent
        # to filtering the forwards first.
        con.sql(f"""COPY (
            SELECT e.* FROM '{full.as_posix()}' e
            WHERE e.src_id IN (SELECT channel_id FROM lang WHERE language = 'en')
              AND e.dest_id IN (SELECT channel_id FROM lang WHERE language = 'en')
            ORDER BY src_id, dest_id
        ) TO '{en.as_posix()}' (FORMAT parquet, COMPRESSION zstd)""")
    return {"full": full, "en": en}
