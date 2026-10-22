"""Describing partitions: composition, conspiracy impurity, crossings and conspiracy blocks."""
import numpy as np
import pandas as pd


def mix(s: pd.Series, k: int = 3) -> str:
    """'ru 94%, unknown 4%, en 1%': the k most common values with their shares."""
    vc = s.fillna("unknown").value_counts(normalize=True).head(k)
    return ", ".join(f"{i} {v:.0%}" for i, v in vc.items())


def composition(ch: pd.DataFrame, part_col: str = "community") -> pd.DataFrame:
    """Per community: size, conspiracy count and share, Gini impurity 2p(1-p), English count, languages."""
    comp = ch.groupby(part_col).agg(channels=("channel_id", "size"), conspiracy=("conspiracy", "sum"),
                                    english=("language", lambda s: int((s == "en").sum())),
                                    languages=("language", mix))
    comp["conspiracy_share"] = comp.conspiracy / comp.channels
    comp["gini_impurity"] = 2 * comp.conspiracy_share * (1 - comp.conspiracy_share)
    return comp


def impurity_summary(ch: pd.DataFrame, comp: pd.DataFrame) -> dict:
    """Channel-weighted Gini impurity within communities vs. the whole set (DECISIONS D10)."""
    p = ch.conspiracy.mean()
    within = (comp.gini_impurity * comp.channels).sum() / comp.channels.sum()
    c = comp[comp.conspiracy > 0]
    return {"channels": len(ch), "conspiracy_channels": int(ch.conspiracy.sum()), "conspiracy_share": p,
            "gini_whole_graph": 2 * p * (1 - p), "gini_within_communities": within,
            "impurity_reduction": 1 - within / (2 * p * (1 - p)),
            "consp_in_majority_consp_communities": c.conspiracy[c.conspiracy_share >= 0.5].sum() / c.conspiracy.sum(),
            "consp_in_communities_under_10pct": c.conspiracy[c.conspiracy_share < 0.1].sum() / c.conspiracy.sum()}


def purity(community: np.ndarray, label: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Clustering purity of `label` within `community` (both integer codes, one entry per channel).

    Returns (overall purity = sum_c max_t n_ct / N, per-community purity, per-community main label),
    per-community arrays indexed by community code 0..max.
    """
    C, T = community.max() + 1, label.max() + 1
    table = np.bincount(community * T + label, minlength=C * T).reshape(C, T)
    size = table.sum(axis=1)
    top = table.max(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        per = top / size
    return top.sum() / size.sum(), per, table.argmax(axis=1)


def purity_null(community: np.ndarray, label: np.ndarray, B: int = 1000,
                seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Purity when labels are permuted across channels (community sizes and label counts fixed).
    Returns (overall purity per permutation (B,), per-community purity per permutation (B, C))."""
    rng = np.random.default_rng(seed)
    overall, per = np.empty(B), np.empty((B, community.max() + 1))
    for b in range(B):
        overall[b], per[b], _ = purity(community, rng.permutation(label))
    return overall, per


def crossing_mask(edges: pd.DataFrame, part: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """(cross, covered) per edge row. `covered` = both ends have a community in `part`
    (channel_id -> community); `cross` is only meaningful where `covered` is True."""
    s, d = edges.src_id.map(part), edges.dest_id.map(part)
    covered = (s.notna() & d.notna()).to_numpy()
    return (s.to_numpy() != d.to_numpy()) & covered, covered


def blocks(edges: pd.DataFrame, col: str, part: pd.Series, cons_of: pd.Series,
           unit: str = "forwards") -> pd.DataFrame:
    """Conspiracy label of source x destination (C->C, C->N, N->C, N->N) for all, within-community and
    across-community forwards (or edges). Rows not covered by `part` are left out."""
    w = edges[col].to_numpy()
    w = (w > 0).astype("int64") if unit == "edges" else w
    sc, dc = edges.src_id.map(cons_of).to_numpy(), edges.dest_id.map(cons_of).to_numpy()
    cross, covered = crossing_mask(edges, part)
    out = {}
    for name, mask in [("all", covered), ("within community", covered & ~cross), ("across communities", cross)]:
        out[name] = {f"{'C' if a else 'N'}→{'C' if b else 'N'}": int(w[mask & (sc == a) & (dc == b)].sum())
                     for a in (True, False) for b in (True, False)}
    return pd.DataFrame(out).T
