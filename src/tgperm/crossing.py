"""Per-topic crossing rates with a permutation null and a source-channel bootstrap (DECISIONS D15, D18).

Input is an aggregated table, one row per (source, destination[, window]) with
  topic  - integer topic code of the forwards in the row
  source - source channel id (the bootstrap cluster)
  n      - forwards in the measure's population
  x      - of those, forwards with the event (e.g. crossing a community boundary)
A topic is constant within a row, so the aggregation loses nothing the measures need.
"""
import warnings

import numpy as np
import pandas as pd


def permutation_null(n_t: np.ndarray, k: int, B: int = 1000, seed: int = 42) -> np.ndarray:
    """Event counts per topic when topic labels are permuted across the population's forwards.

    Permuting labels across N forwards, of which k have the event, puts a multivariate
    hypergeometric number of event forwards in each topic (sizes n_t). Drawing it directly is exactly
    equivalent and avoids shuffling millions of labels. Returns an array of shape (B, T).
    """
    rng = np.random.default_rng(seed)
    return rng.multivariate_hypergeometric(np.asarray(n_t, dtype=np.int64), int(k), size=B, method="marginals")


def two_sided_p(obs: np.ndarray, null: np.ndarray) -> np.ndarray:
    """(1 + #{|x_b - mu| >= |x_obs - mu|}) / (B + 1), mu = null mean, per column."""
    mu = null.mean(axis=0)
    return (1 + (np.abs(null - mu) >= np.abs(obs - mu)).sum(axis=0)) / (null.shape[0] + 1)


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    q = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def source_bootstrap(rows: pd.DataFrame, T: int, B: int = 1000, seed: int = 42,
                     strata: pd.Series | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Cluster bootstrap over source channels: resample sources with replacement (all their forwards).

    With `strata` (source channel id -> stratum, e.g. its community), sources are resampled within each
    stratum, so every stratum keeps its number of sources in every resample (DECISIONS D23).
    Returns (rate, excess), each (B, T): the topic's event rate in the resample and its difference from
    the overall rate of the same resample. A topic absent from a resample gives NaN.
    """
    src = rows.groupby(["source", "topic"], as_index=False)[["n", "x"]].sum()
    sid, uniq = pd.factorize(src.source)
    S = len(uniq)
    topic, n, x = src.topic.to_numpy(), src.n.to_numpy(float), src.x.to_numpy(float)
    groups = None
    if strata is not None:
        st = pd.Series(strata).reindex(uniq).to_numpy()
        assert not pd.isna(st).any(), "every source needs a stratum"
        groups = [np.flatnonzero(st == k) for k in pd.unique(st)]
    rng = np.random.default_rng(seed)
    rate, excess = np.empty((B, T)), np.empty((B, T))
    for b in range(B):
        if groups is None:
            counts = rng.multinomial(S, np.full(S, 1 / S))
        else:
            counts = np.zeros(S, dtype=np.int64)
            for idx in groups:
                counts[idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)))
        w = counts[sid]
        n_t = np.bincount(topic, w * n, minlength=T)
        x_t = np.bincount(topic, w * x, minlength=T)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate[b] = x_t / n_t
        excess[b] = rate[b] - x_t.sum() / n_t.sum()
    return rate, excess


def topic_table(rows: pd.DataFrame, topics: list[str], B: int = 1000, seed: int = 42,
                alpha: float = 0.05, strata: pd.Series | None = None) -> pd.DataFrame:
    """Per topic: forwards, sources, event rate, excess over the overall rate (with bootstrap 95%
    interval), permutation-null band, two-sided p, BH q and the verdict of D15.
    `strata` switches the bootstrap to resampling within strata (D23)."""
    T = len(topics)
    n_t = np.bincount(rows.topic, rows.n, minlength=T).astype(np.int64)
    x_t = np.bincount(rows.topic, rows.x, minlength=T).astype(np.int64)
    overall = x_t.sum() / n_t.sum()
    null = permutation_null(n_t, x_t.sum(), B, seed)
    p = two_sided_p(x_t, null)
    q = bh(p)
    rate_bs, excess_bs = source_bootstrap(rows, T, B, seed, strata=strata)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        # a topic with no forwards in the population (possible inside an RQ3 window) gives all-NaN columns
        warnings.simplefilter("ignore", RuntimeWarning)
        tab = _assemble(topics, rows, n_t, x_t, overall, null, p, q, rate_bs, excess_bs)
    robust = (tab.q < alpha) & ((tab.excess_low_pp > 0) | (tab.excess_high_pp < 0))
    tab["verdict"] = np.where(robust, np.where(tab.excess_pp > 0, "more", "less"), "not distinguishable")
    tab.attrs.update({"overall_rate": overall, "population": int(n_t.sum()), "events": int(x_t.sum()),
                      "B": B, "seed": seed})
    return tab


def _assemble(topics, rows, n_t, x_t, overall, null, p, q, rate_bs, excess_bs) -> pd.DataFrame:
    T = len(topics)
    rate = x_t / n_t
    null_rate = null / n_t
    return pd.DataFrame({
        "topic": topics,
        "forwards": n_t,
        "sources": rows[rows.n > 0].groupby("topic").source.nunique().reindex(range(T), fill_value=0).to_numpy(),
        "events": x_t,
        "rate": rate,
        "excess_pp": 100 * (rate - overall),
        "ratio": rate / overall,
        "excess_low_pp": 100 * np.nanquantile(excess_bs, 0.025, axis=0),
        "excess_high_pp": 100 * np.nanquantile(excess_bs, 0.975, axis=0),
        "rate_low": np.nanquantile(rate_bs, 0.025, axis=0),
        "rate_high": np.nanquantile(rate_bs, 0.975, axis=0),
        "null_rate_low": np.quantile(null_rate, 0.025, axis=0),
        "null_rate_high": np.quantile(null_rate, 0.975, axis=0),
        "p": p,
        "q": q,
    })
