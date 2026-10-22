"""Backbone extraction for the directed weighted forwarding graph (see docs/DECISIONS.md, D14).

Both functions take an edge table with `src_id, dest_id, weight` (content flow, src -> dest) and return
per-edge scores aligned with its rows; thresholding is separate so the scores can be saved once.
"""
import numpy as np
import pandas as pd

NC_DELTA = 1.64     # noise-corrected: keep if score - delta * sd > 0 (one-tailed 5%)
DF_ALPHA = 0.05     # disparity filter: keep if min(alpha_out, alpha_in) < alpha


def noise_corrected(edges: pd.DataFrame) -> pd.DataFrame:
    """Noise-corrected backbone (Coscia & Neffke 2017), directed.

    Follows the authors' reference implementation (michelecoscia/network_backboning, `noise_corrected`):
    the lift of the observed weight over the expectation n_i. * n_.j / n.. is transformed to a symmetric
    score in (-1, 1), and its standard deviation comes from a Beta prior on the edge probability updated
    with the observed weight (Bayesian variance, then the delta method).
    Returns columns `nc_score`, `nc_sdev`.
    """
    nij = edges.weight.to_numpy(dtype=float)
    ni = edges.groupby("src_id").weight.transform("sum").to_numpy(dtype=float)   # out-strength of src
    nj = edges.groupby("dest_id").weight.transform("sum").to_numpy(dtype=float)  # in-strength of dest
    n = nij.sum()

    prior_mean = (ni * nj / n) / n
    kappa = n / (ni * nj)
    score = (kappa * nij - 1) / (kappa * nij + 1)
    prior_var = (1 / n ** 2) * (ni * nj * (n - ni) * (n - nj)) / (n ** 2 * (n - 1))
    alpha_prior = (prior_mean ** 2 / prior_var) * (1 - prior_mean) - prior_mean
    beta_prior = (prior_mean / prior_var) * (1 - prior_mean ** 2) - (1 - prior_mean)
    alpha_post = alpha_prior + nij
    beta_post = n - nij + beta_prior
    expected_pij = alpha_post / (alpha_post + beta_post)
    var_nij = expected_pij * (1 - expected_pij) * n
    d = 1.0 / (ni * nj) - n * ((ni + nj) / (ni * nj) ** 2)
    var_cij = var_nij * ((2 * (kappa + nij * d)) / (kappa * nij + 1) ** 2) ** 2
    return pd.DataFrame({"nc_score": score, "nc_sdev": np.sqrt(var_cij)}, index=edges.index)


def disparity_filter(edges: pd.DataFrame) -> pd.DataFrame:
    """Disparity filter (Serrano, Boguna & Vespignani 2009), directed.

    alpha_out tests the edge against the source's outgoing weights, alpha_in against the destination's
    incoming weights: alpha = (1 - w / s) ** (k - 1). A node with a single edge on that side gives
    alpha = 1 (never significant from that side).
    Returns columns `df_alpha_out`, `df_alpha_in`.
    """
    w = edges.weight.to_numpy(dtype=float)
    g_src, g_dst = edges.groupby("src_id").weight, edges.groupby("dest_id").weight
    s_out, k_out = g_src.transform("sum").to_numpy(dtype=float), g_src.transform("size").to_numpy()
    s_in, k_in = g_dst.transform("sum").to_numpy(dtype=float), g_dst.transform("size").to_numpy()
    a_out = np.power(1 - w / s_out, k_out - 1)
    a_in = np.power(1 - w / s_in, k_in - 1)
    return pd.DataFrame({"df_alpha_out": a_out, "df_alpha_in": a_in}, index=edges.index)


def keep_nc(scores: pd.DataFrame, delta: float = NC_DELTA) -> np.ndarray:
    return (scores.nc_score - delta * scores.nc_sdev > 0).to_numpy()


def keep_df(scores: pd.DataFrame, alpha: float = DF_ALPHA) -> np.ndarray:
    return (np.minimum(scores.df_alpha_out, scores.df_alpha_in) < alpha).to_numpy()
