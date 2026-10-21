"""Paired cluster bootstrap; test examples are not independent training seeds."""
from __future__ import annotations
import numpy as np


def cluster_bootstrap(values, cluster_ids, *, unit: str, repeats: int = 5000,
                      seed: int | None = None, rng=None, batch_size: int = 32) -> dict:
    """Paired differences, one row per item and one column per contrast.

    Resample independent clusters, carrying all their items together. The
    estimand is the item-weighted mean, including when clusters have unequal
    sizes. IDs and their scientific unit are mandatory; tokens or variants must
    not silently become independent observations. Shared draws pair columns.

    First-seen cluster order and the supplied Generator's draw stream are
    preserved. A supplied Generator cannot also have a seed. Returned draws
    support derived contrasts without a second statistical implementation.
    """
    values = np.asarray(values, dtype=np.float64)
    ids = list(cluster_ids)
    if (values.ndim != 2 or not values.shape[1] or len(values) != len(ids)
            or not np.isfinite(values).all()):
        raise ValueError("Finite item-by-contrast values and matching cluster IDs required.")
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("Name the independent resampling unit.")
    if (not isinstance(repeats, int) or repeats < 100
            or not isinstance(batch_size, int) or batch_size < 1):
        raise ValueError("Use at least 100 draws and a positive batch size.")
    if rng is not None and (seed is not None or not isinstance(rng, np.random.Generator)):
        raise ValueError("Supply either an integer seed or a NumPy Generator.")
    supplied_rng = rng is not None
    if rng is None:
        seed = 17 if seed is None else seed
        rng = np.random.default_rng(seed)
    groups = {}
    membership = []
    for cluster in ids:
        if not isinstance(cluster, (str, int, np.integer)) or isinstance(cluster, bool) or cluster == "":
            raise ValueError("Cluster IDs must be nonempty strings or integers.")
        if cluster not in groups:
            groups[cluster] = len(groups)
        membership.append(groups[cluster])
    if len(groups) < 2:
        raise ValueError("At least two independent clusters required.")
    sums = np.zeros((len(groups), values.shape[1]), dtype=np.float64)
    np.add.at(sums, membership, values)
    sizes = np.bincount(membership).astype(np.float64)
    draws = []
    for first in range(0, repeats, batch_size):
        indices = rng.integers(0, len(groups), size=(min(batch_size, repeats-first), len(groups)))
        draws.append(sums[indices].sum(1) / sizes[indices].sum(1)[:, None])
    draws = np.concatenate(draws)
    return dict(mean=sums.sum(0)/sizes.sum(), CI95=np.quantile(draws, [.025, .975], axis=0).T,
                draws=draws, metadata=dict(resampling_unit=unit, independent_clusters=len(groups),
                    items=len(values), bootstrap_draws=repeats,
                    bootstrap_seed=None if supplied_rng else seed,
                    random_source="supplied Generator" if supplied_rng else "seeded Generator",
                    estimand="item-weighted mean; all items in a sampled cluster travel together"))


def paired_cluster_bootstrap(rows_a: list[dict], rows_b: list[dict], *,
                             repeats: int = 5000, seed: int = 17,
                             unit: str = "document_id cluster") -> dict:
    """Rows: item_id, document_id, score; one pre-aggregated score per item.

    Higher score is better (convert loss to negative loss). Pairing is strict:
    missing/duplicate items fail rather than silently dropping hard examples.
    Resample the document_id clusters with replacement; each item's weight is
    retained. A task may use that field for source documents or related program
    families; pass unit to name the actual independent unit in new reports.
    """
    def index(rows):
        d = {}
        for row in rows:
            key = row["item_id"]
            if key in d or not np.isfinite(float(row["score"])):
                raise ValueError("Duplicate item or nonfinite score.")
            d[key] = row
        return d
    a, b = index(rows_a), index(rows_b)
    if not a or a.keys() != b.keys():
        raise ValueError("Comparisons require identical nonempty item sets.")
    if repeats < 100:
        raise ValueError("Use at least 100 bootstrap draws.")
    differences, cluster_ids = [], []
    for key in sorted(a):
        if a[key]["document_id"] != b[key]["document_id"]:
            raise ValueError("Cluster IDs disagree for a paired item.")
        differences.append([float(b[key]["score"])-float(a[key]["score"])])
        cluster_ids.append(a[key]["document_id"])
    result = cluster_bootstrap(differences, cluster_ids, unit=unit, repeats=repeats, seed=seed)
    lo, hi = result["CI95"][0]
    return {"difference_b_minus_a": float(result["mean"][0]),
            "ci95": [float(lo), float(hi)], "items": len(a),
            "documents": result["metadata"]["independent_clusters"],
            "bootstrap_draws": repeats, "bootstrap_seed": seed,
            "resampling_unit": unit,
            "qualification": "Sampling uncertainty conditional on these checkpoints; not training-seed uncertainty."}
