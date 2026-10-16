"""Paired cluster bootstrap; test examples are not independent training seeds."""
from __future__ import annotations
from collections import defaultdict
import numpy as np


def paired_cluster_bootstrap(rows_a: list[dict], rows_b: list[dict], *,
                             repeats: int = 5000, seed: int = 17) -> dict:
    """Rows: item_id, document_id, score; one pre-aggregated score per item.

    Higher score is better (convert loss to negative loss). Pairing is strict:
    missing/duplicate items fail rather than silently dropping hard examples.
    Re-sample DOCUMENTS with replacement; each item's weight is retained.
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
    groups = defaultdict(list)
    for key in sorted(a):
        if a[key]["document_id"] != b[key]["document_id"]:
            raise ValueError("Cluster IDs disagree for a paired item.")
        groups[a[key]["document_id"]].append(float(b[key]["score"])-float(a[key]["score"]))
    if len(groups) < 2:
        raise ValueError("At least two independent documents are required.")
    sums = np.array([sum(v) for v in groups.values()], dtype=np.float64)
    sizes = np.array([len(v) for v in groups.values()], dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = []
    # Bounded memory, even for a large final held-out set.
    for _ in range(repeats):
        idx = rng.integers(0, len(sums), size=len(sums))
        means.append(sums[idx].sum()/sizes[idx].sum())
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"difference_b_minus_a": float(sums.sum()/sizes.sum()),
            "ci95": [float(lo), float(hi)], "items": len(a), "documents": len(groups),
            "bootstrap_draws": repeats, "bootstrap_seed": seed,
            "qualification": "Sampling uncertainty conditional on these checkpoints; not training-seed uncertainty."}
