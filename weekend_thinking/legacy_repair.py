"""Narrow, fail-closed patch for the uploaded trainer's last-token target defect.

Default writes a separate repaired tree and diff, never alters the input repo.
It does NOT certify the old training recipe, fix other known bugs, or turn the
old best-of trainer into the proposed on-policy experiment.
"""
from __future__ import annotations
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import numpy as np

FILES = ("pre-training/train/think_oracle.py", "pre-training/train/think_bestof.py",
         "pre-training/train/think_hintmix.py")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def patched_sources(sources: dict[str, str]) -> dict[str, str]:
    out = dict(sources)
    key = FILES[0]; s = out[key]
    signature = "def buffer(self, x, anchors, Z, lens):"
    target = "        tgt[:T - 1] = x[1:]"
    if s.count(signature) != 1 or s.count(target) != 1:
        raise ValueError("OGeo signature/target pattern differs from reviewed source; inspect manually.")
    s = s.replace(signature, "def buffer(self, x, anchors, Z, lens, *, next_targets=None):", 1)
    s = s.replace(target, '''        tgt[:T - 1] = x[1:]
        # next_targets is the complete shifted T-token vector from row[1:T+1].
        # Legacy read-only probes can omit it; all three training callers pass it.
        if next_targets is not None:
            next_targets = np.asarray(next_targets)
            if next_targets.shape != (T,) or not np.array_equal(next_targets[:-1], x[1:]):
                raise ValueError("next_targets must be the complete, aligned next-token vector")
            tgt[:T] = next_targets''', 1)
    out[key] = s
    calls = {
        FILES[0]: ["geo.buffer(x, anchors, Z, lens)"],
        FILES[1]: ["sgeo.buffer(x, anchors, Z[sl], lens[sl])", "ggeo.buffer(x, anchors, Zsel, lsel)"],
        FILES[2]: ["geo.buffer(x, anchors, Z, lens)", "hgeo.buffer(x, anchors, Zh, lens_h)"],
    }
    for key, patterns in calls.items():
        for old in patterns:
            if out[key].count(old) != 1:
                raise ValueError(f"Caller differs from reviewed source: {key}: {old}")
            out[key] = out[key].replace(old, old[:-1] + ", next_targets=y)", 1)
        ast.parse(out[key])
    return out


def load_geometry(source: str):
    """Execute ONLY the pure-NumPy OGeo AST, avoiding torch_xla imports."""
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OGeo")
    module = ast.Module(body=[cls], type_ignores=[])
    namespace = {"np": np, "T_OPEN": 1000, "T_CLOSE": 1001, "NEG": -1e9}
    exec(compile(module, "<isolated OGeo>", "exec"), namespace)
    return namespace["OGeo"]


def verify_target_repair(original: str, repaired: str) -> dict:
    old, new = load_geometry(original), load_geometry(repaired)
    cases = 0
    for T in (16, 32, 64):
        for L in (2, 5):
            for K in (1, 4):
                row = np.arange(10, 10+T+1, dtype=np.int64)
                x, y = row[:-1], row[1:]
                anchors = np.array([2, 5]); Z = np.full((2, 2, L), 500, dtype=np.int64)
                lens = np.full((2, 2), L, dtype=np.int64)
                a = old(T, 2, 2, L, K).buffer(x, anchors, Z, lens)
                b = new(T, 2, 2, L, K).buffer(x, anchors, Z, lens, next_targets=y)
                assert a[3][T-1] == 0 and b[3][T-1] == y[-1]
                assert np.array_equal(b[3][:T], y)
                for j in (0, 1, 2, 4):
                    assert np.array_equal(a[j], b[j]), "Patch unexpectedly changed geometry."
                assert np.array_equal(a[3][T:], b[3][T:])
                cases += 1
    return {"passed_cases": cases, "change": "Only the final clean-token target changes; buffer, attention, branch labels and thought masks are unchanged."}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--expected-manifest", type=Path)
    a = p.parse_args()
    if a.out.resolve() == a.repo.resolve() or a.repo.resolve() in a.out.resolve().parents:
        raise SystemExit("Use a separate output directory, outside the source repo.")
    sources = {key: (a.repo/key).read_text() for key in FILES}
    if a.expected_manifest:
        expected = json.loads(a.expected_manifest.read_text())["source_sha256"]
        for key, src in sources.items():
            if expected.get(key) != sha(src):
                raise SystemExit(f"Source fingerprint differs for {key}; do not apply blindly.")
    patched = patched_sources(sources)
    result = verify_target_repair(sources[FILES[0]], patched[FILES[0]])
    a.out.mkdir(parents=True, exist_ok=True)
    diff = []
    for key in FILES:
        dest = a.out/key; dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists(): raise SystemExit(f"Refusing to overwrite {dest}.")
        dest.write_text(patched[key])
        diff += difflib.unified_diff(sources[key].splitlines(True), patched[key].splitlines(True),
                                    fromfile="a/"+key, tofile="b/"+key)
    (a.out/"target-repair.patch").write_text("".join(diff))
    manifest = {"source_sha256": {k: sha(v) for k, v in sources.items()},
                "patched_sha256": {k: sha(v) for k, v in patched.items()},
                "verification": result,
                "limitation": "Training callers outside these three files are not updated; read-only probes that omit next_targets retain the legacy last label. Do not assign NTP weight to that label."}
    (a.out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__": main()
