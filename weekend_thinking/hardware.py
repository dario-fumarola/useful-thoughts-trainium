"""Read-only host report. Never installs software or queries cloud credentials."""
from __future__ import annotations
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time


def _command(argv: list[str]) -> dict:
    if not shutil.which(argv[0]):
        return {"available": False, "command": argv}
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=15, check=False)
        return {"available": True, "command": argv, "returncode": p.returncode,
                "stdout": p.stdout[:200_000], "stderr": p.stderr[:10_000]}
    except subprocess.TimeoutExpired:
        return {"available": True, "command": argv, "error": "15-second timeout"}


def report() -> dict:
    versions = {}
    for name in ("torch", "torch-xla", "torch-neuronx", "transformers", "neuronx-cc",
                 "neuronx-distributed", "numpy", "safetensors", "datasets"):
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    # Explicit allowlist: never dump environment variables containing secrets.
    env = {k: os.environ.get(k) for k in (
        "NEURON_LOGICAL_NC_CONFIG", "NEURON_RT_VISIBLE_CORES", "NEURON_RT_NUM_CORES",
        "PJRT_DEVICE", "WORLD_SIZE", "XLA_USE_BF16", "XLA_DOWNCAST_BF16")}
    mem = Path("/proc/meminfo")
    return {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": platform.python_version(), "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "versions": versions, "environment_allowlist": env,
            "memory": mem.read_text() if mem.exists() else None,
            "disk_free_bytes": shutil.disk_usage(Path.cwd()).free,
            "neuron_ls": _command(["neuron-ls", "--json-output"]),
            "neuron_topology": _command(["neuron-ls", "--topology"]),
            "interpretation": (
                "Verify instance type in the EC2 console. trn2.48xlarge is 16 chips, not 48. "
                "At default LNC2 it exposes 64 logical NeuronCores, ordinarily 24 GB per "
                "logical core, not 1.5 TB per process. Do not infer device layout from the name alone."),
            "production_qualified": False}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report(), indent=2) + "\n")
    print(f"Saved read-only host report to {out}.")
