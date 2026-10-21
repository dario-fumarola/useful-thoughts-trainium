"""Replay released selector-optimization scores; NumPy only, no model execution.

This verifies projected document scores, routing, intervals and decisions. It
does not reconstruct omitted hidden features, source text or optimizer updates.
The separate private-cache checker covers feature/gate/mixture arithmetic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message):
    require(np.allclose(actual, expected, rtol=0, atol=1e-9), message)


def replay(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        require(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest,
                "Evidence digest mismatch: " + name)
    report = json.loads((root / "readout.json").read_text())
    identities = json.loads((root / "validation-identities.json").read_text())
    require(len(identities) == 1024, "All validation documents required")
    require(len({r["source_hash"] for r in identities}) == 1024,
            "Each row must be a unique source document")
    require(manifest["all_heads_frozen_before_validation"], "Freeze contract")
    scores = np.load(root / "document-scores.npz", allow_pickle=False)
    routes = np.load(root / "gates.npz", allow_pickle=False)
    orders = range(2026102141, 2026102147)
    reports = {}
    for kind in ("calculation", "ordinary"):
        ix = [i for i, row in enumerate(identities) if row["kind"] == kind]
        require(len(ix) == 512, "Exactly 512 documents per stratum")
        for name in scores.files:
            require(scores[name].shape == (1024,), "Document-score shape")
            close(scores[name][ix].mean(),
                  report["absolute_mean_log_probability_per16"][kind][name],
                  "Absolute mean: " + name)
        columns = {}
        for gate in ("hard", "smooth"):
            for order in orders:
                name = f"T_note-{gate}-order{order}"
                for label, baseline in (("B0", "B0"), ("global", "T_note:global"),
                        ("position", "T_note:position"),
                        ("raw-ensemble", f"T_raw-{gate}-order{order}")):
                    columns[name + ":minus-" + label] = (scores[name] - scores[baseline])[ix]
        for role in ("T_note", "T_raw"):
            for order in orders:
                columns[f"{role}-order{order}:smooth-minus-hard"] = (
                    scores[f"{role}-smooth-order{order}"] -
                    scores[f"{role}-hard-order{order}"])[ix]
        require(set(columns) == set(report["reports"][kind]), "Every contrast required")
        values = np.column_stack(list(columns.values()))
        # Independent numerical implementation, deliberately not the production
        # statistics helper. One draw selects complete source documents.
        rng = np.random.default_rng(2026102147 + (kind == "ordinary"))
        draws = np.array([values[rng.integers(0, 512, 512)].mean(0)
                          for _ in range(5000)])
        reports[kind] = {}
        for i, name in enumerate(columns):
            actual = dict(mean=float(values[:, i].mean()),
                          CI95=np.quantile(draws[:, i], [.025, .975]).tolist())
            expected = report["reports"][kind][name]
            close([actual["mean"], *actual["CI95"]],
                  [expected["mean"], *expected["CI95"]], "Paired interval: " + name)
            reports[kind][name] = actual
        for name in routes.files:
            g = routes[name]
            require(g.shape == (1024, 16) and np.isfinite(g).all()
                    and ((g >= 0) & (g <= 1)).all(), "Routing probabilities")
            for key, value in (("mean_gate", g[ix].mean()),
                    ("closed_fraction", (g[ix] == 0).mean()),
                    ("near_closed_fraction", (g[ix] <= .001).mean())):
                close(value, report["routing"][name][kind][key], "Routing: " + name)
    decisions = {}
    for gate in ("hard", "smooth"):
        decisions[gate] = {}
        for order in orders:
            name = f"T_note-{gate}-order{order}"
            c = reports["calculation"]
            base = c[name + ":minus-B0"]
            ordinary = reports["ordinary"][name + ":minus-B0"]
            decisions[gate][str(order)] = dict(
                base=base["mean"] >= .25 and base["CI95"][0] > 0,
                global_control=c[name + ":minus-global"]["CI95"][0] > 0,
                raw_ensemble=c[name + ":minus-raw-ensemble"]["CI95"][0] > 0,
                ordinary_guard=-ordinary["CI95"][0] / 16 <= .03,
                position_secondary=c[name + ":minus-position"]["CI95"][0] > 0,
                effectively_stalled=report["routing"][name]["calculation"]
                    ["near_closed_fraction"] >= .99 and base["mean"] <= .01)
    require(decisions == report["decisions"], "All declared decisions")
    all_pass = {g: all(all(v[k] for k in ("base", "global_control", "raw_ensemble",
                    "ordinary_guard")) for v in outcomes.values())
                for g, outcomes in decisions.items()}
    require(all_pass == report["all_six_primary_pass"], "Joint decisions")
    return dict(status="PASS_PUBLIC_SELECTOR_OPTIMIZATION_NUMERICAL_REPLAY",
                verified_files=len(manifest["files"]), document_rows=1024,
                heads=24, source_document_bootstrap_draws=5000,
                all_six_primary_pass=all_pass,
                limits="Projected numerical evidence, not model execution, optimizer replication or fresh confirmation.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.evidence)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
