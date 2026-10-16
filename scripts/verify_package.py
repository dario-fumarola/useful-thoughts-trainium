#!/usr/bin/env python3
"""Verify files in the original delivery; intentional later edits will differ."""
from pathlib import Path
import hashlib,json,sys


def main():
    root=Path(__file__).resolve().parents[1]
    manifest=root/'DELIVERY_MANIFEST.json'
    if not manifest.exists():raise SystemExit('No delivery manifest found; cannot verify this package.')
    data=json.loads(manifest.read_text());failed=[]
    for name,item in data['files'].items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts:raise SystemExit('Unsafe path in package manifest.')
        p=root/p
        if p.is_symlink() or not p.is_file():failed.append(name+' (missing or symlink)');continue
        h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
        if h.hexdigest()!=item['sha256'] or p.stat().st_size!=item['bytes']:failed.append(name+' (changed)')
    if failed:
        print('\n'.join(failed[:40]))
        raise SystemExit('Delivery differs. On first extraction, do not proceed blindly. After intentional edits, inspect git diff and use scripts/run_checks.sh instead.')
    print(f"Verified {len(data['files'])} delivered files. This verifies bytes, not Neuron readiness.")
if __name__=='__main__':main()
