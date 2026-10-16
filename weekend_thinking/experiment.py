"""Freeze an experiment against real local assets; never launches hardware."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from .artifacts import atomic_json, canonical_hash, sha256_file, source_fingerprint, utc_now, require_utc_deadline
from .core import Format
from .data import load_records


def _local(root: Path, value, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Resolve {label} in configs/site.local.json before freezing.")
    path = Path(value).expanduser()
    if not path.is_absolute(): path = root / path
    path = path.resolve()
    if not path.exists(): raise FileNotFoundError(f"{label}: {path}")
    return path


def freeze(root: str | Path, proposal: dict, site: dict, *, experiment_id: str, steps: int) -> dict:
    root = Path(root).resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,80}", experiment_id):
        raise ValueError("Use a simple unique experiment ID, not a path.")
    if steps not in proposal['optimization']['step_budget_candidates']:
        raise ValueError("Step count must be a predeclared, timing-selected budget.")
    model = _local(root, site.get('model_dir'), 'model_dir')
    records_path = _local(root, site.get('records_file'), 'records_file')
    format_path = _local(root, site.get('format_file'), 'format_file')
    snapshot_manifest = model / 'ASSET_MANIFEST.json'
    asset = json.loads(snapshot_manifest.read_text())
    if asset['model_id'] != proposal['model']['id']:
        raise ValueError("Model ID differs from proposal. Use a new explicit proposal/experiment identity for fallback.")
    if not re.fullmatch(r'[0-9a-f]{40,64}', asset.get('resolved_revision', '')):
        raise ValueError("Model revision must be an immutable commit.")
    for filename, item in asset['files'].items():
        if Path(filename).is_absolute() or '..' in Path(filename).parts:
            raise ValueError("Unsafe model manifest path.")
        path = model / filename
        if path.is_symlink() or not path.is_file() or sha256_file(path) != item['sha256']:
            raise ValueError(f"Model asset changed or missing: {filename}")
    if 'config.json' not in asset['files'] or not any(x.endswith('.safetensors') for x in asset['files']):
        raise ValueError("Manifest does not contain config and weights.")
    model_config = json.loads((model / 'config.json').read_text())
    if model_config.get('model_type') != 'qwen3':
        raise ValueError("This proposal expects a dense Qwen3 model.")
    data_manifest = json.loads(records_path.with_suffix('.manifest.json').read_text())
    if sha256_file(records_path) != data_manifest['output_sha256']:
        raise ValueError("Record-file checksum differs from preparation manifest.")
    identity = data_manifest.get('tokenizer_identity', '')
    if asset['resolved_revision'] not in identity or asset['model_id'] not in identity:
        raise ValueError("Record tokenizer identity must include this exact model ID and revision.")
    rows = load_records(records_path)
    counts = Counter(r.split for r in rows)
    b = proposal['optimization']['global_unique_prefixes_per_update']
    if counts['train'] < steps * b:
        raise ValueError(f"Need {steps*b} fresh training records, found {counts['train']}. No silent cycling.")
    evaluation = proposal.get('evaluation', {})
    for split, key in [('dev','dev_documents'), ('test','final_local_documents')]:
        if counts[split] < evaluation.get(key, 0):
            raise ValueError(f'Insufficient {split} documents for the frozen local evaluation.')
    prefix = proposal['data']['prefix_tokens']; future = proposal['data']['stored_future_tokens']
    if any(len(r.prefix) != prefix or len(r.future) != future for r in rows):
        raise ValueError("Record lengths differ from the fixed proposal.")
    if max(t for r in rows for t in r.prefix+r.future) >= model_config['vocab_size']:
        raise ValueError("Record token ID exceeds the model vocabulary.")
    formatted = json.loads(format_path.read_text())
    fmt = Format.from_dict(formatted['format'])
    tokens = fmt.header_ids+fmt.open_ids+fmt.close_ids+fmt.additional_stop_ids+(fmt.stop_id,fmt.pad_id)
    if max(tokens) >= model_config['vocab_size']:
        raise ValueError("Prompt token ID exceeds the model vocabulary.")
    fmt_tokenizer = Path(formatted.get('tokenizer_dir', '')).expanduser()
    if not fmt_tokenizer.is_absolute(): fmt_tokenizer = root / fmt_tokenizer
    if fmt_tokenizer.resolve() != model:
        raise ValueError("Prompt format was resolved with a different tokenizer directory.")
    world = site.get('world_size_per_arm'); accum = site.get('gradient_accumulation')
    micro = proposal['optimization']['micro_prefixes_per_rank']
    if type(world) is not int or type(accum) is not int or world < 1 or accum < 1 or world*accum*micro != b:
        raise ValueError("world × accumulation × micro_prefixes must equal the global prefix batch.")
    deadline = require_utc_deadline(site.get('deadline_utc', ''))
    result = {
        'schema': 1, 'experiment_id': experiment_id, 'created_utc': utc_now(),
        'status': 'FROZEN_ASSETS_NOT_HARDWARE_QUALIFIED', 'terminal_steps': steps,
        'proposal': proposal, 'site': site, 'deadline_utc': deadline.isoformat(),
        'assets': {'model_dir': str(model), 'model_manifest_sha256': sha256_file(snapshot_manifest),
                   'model_id': asset['model_id'], 'model_revision': asset['resolved_revision'],
                   'records_file': str(records_path), 'records_sha256': sha256_file(records_path),
                   'record_counts': dict(counts), 'record_manifest_sha256': sha256_file(records_path.with_suffix('.manifest.json')),
                   'format_file': str(format_path), 'format_sha256': sha256_file(format_path)},
        'source': source_fingerprint(root),
        'training_exposure_per_arm': {'prefix_visits': steps*b, 'source_token_visits': steps*b*(prefix+proposal['thought_arm']['reward_horizon'])},
        'qualification': 'A frozen specification is not evidence of device correctness. Check the real qualification receipt.'}
    result['spec_hash'] = canonical_hash(result)
    return result


REQUIRED_CHECKS = ('pretrained_loading', 'target_alignment', 'prefix_only_generation',
                   'cache_replay_parity', 'distributed_update', 'checkpoint_resume',
                   'data_audit', 'full_step_throughput', 'downstream_eval_path')


def validate_qualification(root: str | Path, frozen: dict, receipt: dict) -> None:
    root = Path(root).resolve()
    if canonical_hash({k:v for k,v in frozen.items() if k != 'spec_hash'}) != frozen.get('spec_hash'):
        raise ValueError('Frozen specification was edited after hashing.')
    if require_utc_deadline(frozen['deadline_utc']) <= datetime.now(timezone.utc):
        raise ValueError('Operator deadline has passed.')
    assets = frozen['assets']
    for path_key, hash_key in [('records_file','records_sha256'),('format_file','format_sha256')]:
        if sha256_file(assets[path_key]) != assets[hash_key]:
            raise ValueError(f'Frozen asset changed: {path_key}')
    manifest = Path(assets['model_dir']) / 'ASSET_MANIFEST.json'
    if sha256_file(manifest) != assets['model_manifest_sha256']:
        raise ValueError('Model manifest changed after freezing.')
    for name, item in json.loads(manifest.read_text())['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe model manifest path.')
        target = manifest.parent / name
        if target.is_symlink() or sha256_file(target) != item['sha256']:
            raise ValueError(f'Model weight/config changed: {name}')
    if receipt.get('platform') != 'neuron' or receipt.get('synthetic') is not False:
        raise ValueError("CPU/synthetic validation does not qualify a Neuron run.")
    if receipt.get('spec_hash') != frozen['spec_hash']:
        raise ValueError("Qualification belongs to a different frozen specification.")
    if receipt.get('source_sha256') != frozen['source']['sha256'] or source_fingerprint(root)['sha256'] != receipt.get('source_sha256'):
        raise ValueError("Active source changed after qualification.")
    checks = receipt.get('checks', {})
    for name in REQUIRED_CHECKS:
        row = checks.get(name, {})
        if row.get('passed') is not True or not row.get('command') or not row.get('observations'):
            raise ValueError(f"Unqualified check: {name}")
        evidence = _local(root, row.get('evidence_file'), name)
        if sha256_file(evidence) != row.get('evidence_sha256'):
            raise ValueError(f"Evidence changed for {name}")
    if len(receipt.get('full_step_seconds', [])) < 10:
        raise ValueError("Need ten complete measured steady-state optimizer steps.")
    if any(not isinstance(x,(float,int)) or not 0 < x < 86400 for x in receipt['full_step_seconds']):
        raise ValueError("Invalid full-step measurements.")
    # This validates recorded evidence and bindings, not the truth of observations.
    # Never fill a 'passed' field without actually running the named hardware check.
