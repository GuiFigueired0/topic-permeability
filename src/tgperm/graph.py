"""Graph construction, descriptives and community detection on the forwarding graph."""
import time

import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd

from . import config as cfg


def to_igraph(edges: pd.DataFrame) -> ig.Graph:
    """Directed weighted igraph from an edge table (src_id -> dest_id, content flow).

    Vertex attribute `channel_id` holds the Telegram id; vertex i is the i-th smallest id, so the
    mapping is deterministic. Only channels with at least one edge become vertices.
    """
    ids = np.union1d(edges.src_id.to_numpy(), edges.dest_id.to_numpy())
    src = np.searchsorted(ids, edges.src_id.to_numpy())
    dst = np.searchsorted(ids, edges.dest_id.to_numpy())
    g = ig.Graph(n=len(ids), edges=np.column_stack([src, dst]).tolist(), directed=True)
    g.vs["channel_id"] = ids.tolist()
    g.es["weight"] = edges.weight.astype(float).tolist()
    return g


def node_table(edges: pd.DataFrame) -> pd.DataFrame:
    """Per channel: out/in degree (distinct partners) and out/in strength (forwards).

    Content-flow direction: out-strength = how often the channel's posts were reposted by other
    channels; in-strength = how many forwards the channel itself reposted.
    """
    out = edges.groupby("src_id").agg(out_degree=("dest_id", "size"), out_strength=("weight", "sum"))
    inn = edges.groupby("dest_id").agg(in_degree=("src_id", "size"), in_strength=("weight", "sum"))
    nodes = out.join(inn, how="outer").fillna(0).astype("int64")
    nodes.index.name = "channel_id"
    return nodes.reset_index()


def weighted_reciprocity(edges: pd.DataFrame) -> float:
    """Share of forward volume that is reciprocated: sum_ij min(w_ij, w_ji) / sum_ij w_ij."""
    rev = edges[["src_id", "dest_id", "weight"]].rename(
        columns={"src_id": "dest_id", "dest_id": "src_id", "weight": "w_rev"})
    m = edges[["src_id", "dest_id", "weight"]].merge(rev, on=["src_id", "dest_id"], how="left")
    return float(np.minimum(m.weight, m.w_rev.fillna(0)).sum() / m.weight.sum())


def graph_summary(edges: pd.DataFrame, g: ig.Graph) -> dict:
    n, m = g.vcount(), g.ecount()
    wcc = g.connected_components(mode="weak").sizes()
    scc = g.connected_components(mode="strong").sizes()
    return {
        "nodes": n,
        "edges": m,
        "forwards": int(edges.weight.sum()),
        "text_forwards_share": edges.n_text.sum() / edges.weight.sum(),
        "density": m / (n * (n - 1)),
        "mean_out_degree": m / n,
        "reciprocity_edges": g.reciprocity(ignore_loops=True, mode="default"),
        "reciprocity_weighted": weighted_reciprocity(edges),
        "wcc_count": len(wcc),
        "largest_wcc_share": max(wcc) / n,
        "scc_count": len(scc),
        "largest_scc_share": max(scc) / n,
    }


def top_share(values: np.ndarray, fractions=(0.01, 0.05, 0.10)) -> dict:
    """Share of the total carried by the top x% of units (values sorted in descending order)."""
    v = np.sort(np.asarray(values, dtype=float))[::-1]
    total = v.sum()
    return {f: v[: max(1, int(round(f * len(v))))].sum() / total for f in fractions}


def ccdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empirical complementary CDF P(X >= x) over the positive values."""
    v = np.sort(np.asarray(values)[np.asarray(values) > 0])
    x, idx = np.unique(v, return_index=True)
    return x, 1.0 - idx / len(v)


def leiden(g: ig.Graph, resolution: float = cfg.LEIDEN_RESOLUTION, seed: int = cfg.SEED,
           n_iterations: int = -1) -> tuple[np.ndarray, dict]:
    """Leiden with RBConfigurationVertexPartition on the directed weighted graph.

    Returns the membership renumbered by community size (0 = largest; ties broken by the
    smallest member vertex, so the numbering is deterministic) and a dict of run metadata.
    """
    t0 = time.time()
    part = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight",
                             resolution_parameter=resolution, seed=seed, n_iterations=n_iterations)
    raw = np.asarray(part.membership)
    sizes = np.bincount(raw)
    first = np.full(len(sizes), len(raw))
    np.minimum.at(first, raw, np.arange(len(raw)))
    order = np.lexsort((first, -sizes))           # largest first, then earliest vertex
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    membership = rank[raw]
    meta = {
        "algorithm": "leidenalg.RBConfigurationVertexPartition",
        "resolution": resolution,
        "seed": seed,
        "n_iterations": n_iterations,
        "directed": g.is_directed(),
        "weighted": True,
        "nodes": g.vcount(),
        "edges": g.ecount(),
        "n_communities": int(len(sizes)),
        "quality": float(part.quality()),
        "modularity_directed": float(g.modularity(membership.tolist(), weights="weight", directed=True)),
        "modularity_undirected": float(g.modularity(membership.tolist(), weights="weight", directed=False)),
        "seconds": round(time.time() - t0, 1),
        "leidenalg_version": la.version,
        "igraph_version": ig.__version__,
    }
    return membership, meta


def mixing_matrix(edges: pd.DataFrame, community: pd.Series, keep: list[int],
                  other_label: str = "other") -> pd.DataFrame:
    """Forwards from source community (rows) to destination community (columns).

    `community` maps channel_id -> community id. Communities outside `keep` are pooled.
    """
    src_c = edges.src_id.map(community)
    dst_c = edges.dest_id.map(community)
    def pool(c):
        return c.where(c.isin(keep), other_label).astype(str)
    mat = (pd.DataFrame({"src": pool(src_c), "dst": pool(dst_c), "w": edges.weight})
           .pivot_table(index="src", columns="dst", values="w", aggfunc="sum", fill_value=0))
    labels = [str(k) for k in keep] + ([other_label] if other_label in mat.index.union(mat.columns) else [])
    return mat.reindex(index=labels, columns=labels, fill_value=0)
