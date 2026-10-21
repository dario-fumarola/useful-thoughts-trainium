"""Freeze V30 questions, labels and design before any new model call."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from train.candidate_utility_v30 import cohort
from train.real_trace_contexts import question_projection
from weekend_thinking.artifacts import atomic_json, sha256_file, utc_now


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'audit', 'binding', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); root = Path(__file__).resolve().parents[1]
    design_file = root/'configs/candidate_utility_v30_design.json'
    design = json.loads(design_file.read_text())
    pool_file = a.data/design['data']['pool']
    assert sha256_file(pool_file) == design['data']['pool_sha256']
    audit = json.loads(a.audit.read_text())
    assert audit['status'] == 'PASS_ALL_FRESH_POOL_RAW_PARQUET_NATIVE_PREFIX_LABEL_VALUE_AND_EXCLUSION_CHECKS'
    assert audit['pool_sha256'] == sha256_file(pool_file) and audit['documents'] == 390
    assert audit['source_sha256'] == sha256_file(root/'scripts/check_candidate_utility_v30_data.py')
    binding = json.loads(a.binding.read_text())
    assert binding['id'] == 'computation-seed2026102029' and binding['step'] == 1280
    assert binding['checkpoint_sha256'] == 'bacd0f2522d5998c4b355c0f15209fcb36b5b0ddd361606e8fef89cefea714d4'
    assert sha256_file(binding['checkpoint']) == binding['checkpoint_sha256']
    qualification, diagnostic = cohort([json.loads(s) for s in pool_file.open()])
    a.output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, values in (('qualification', qualification), ('diagnostic', diagnostic)):
        for kind, rows in (('questions', [question_projection(r) for r in values]), ('labels', values)):
            path = a.output/f'{name}-{kind}.jsonl'
            with path.open('x') as stream:
                for row in rows:
                    stream.write(json.dumps(row, separators=(',', ':'))+'\n')
            files[path.name] = sha256_file(path)
    atomic_json(a.output/'protocol.json', dict(status='FROZEN_V30_BEFORE_MODEL_CALLS',
        created_utc=utc_now(), design_file=str(design_file), design_sha256=sha256_file(design_file),
        pool_file=str(pool_file), pool_sha256=sha256_file(pool_file),
        audit_file=str(a.audit.resolve()), audit_sha256=sha256_file(a.audit),
        binding_file=str(a.binding.resolve()), binding_sha256=sha256_file(a.binding),
        model=binding, files=files, qualification_documents=8, diagnostic_documents=256,
        model_calls_so_far=0, learning_released=False))
    print('FROZEN_8_QUALIFICATION_256_DIAGNOSTIC_NO_MODEL_CALLS', flush=True)


if __name__ == '__main__':
    main()
