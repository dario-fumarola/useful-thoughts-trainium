#!/usr/bin/env python3
"""Explicit network action: download one allowlisted model at a resolved commit.

No authentication tokens are printed or saved. No model code is executed.
Requires the existing environment's huggingface_hub; no packages are installed.
"""
from pathlib import Path
import argparse, json, re, sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from weekend_thinking.artifacts import atomic_json,sha256_file,utc_now

ALLOWED=('Qwen/Qwen3-1.7B-Base','Qwen/Qwen3-0.6B-Base')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-id',choices=ALLOWED,default=ALLOWED[0])
    p.add_argument('--revision',default='main',help='Resolved once to an immutable commit before download.')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-workers',type=int,default=4)
    a=p.parse_args()
    if (a.out/'ASSET_MANIFEST.json').exists():
        raise SystemExit('Existing asset manifest found. Reuse and verify it; do not silently replace a pinned snapshot.')
    try:
        from huggingface_hub import HfApi,snapshot_download
    except ImportError:
        raise SystemExit('huggingface_hub is missing. Use the qualified environment; do not replace Torch/Neuron.')
    info=HfApi().model_info(a.model_id,revision=a.revision)
    commit=info.sha
    if not commit or not re.fullmatch(r'[0-9a-f]{40,64}',commit):
        raise SystemExit('Hub did not return an immutable commit.')
    request={'model_id':a.model_id,'resolved_revision':commit}
    request_path=a.out/'DOWNLOAD_REQUEST.json'
    if request_path.exists():
        if json.loads(request_path.read_text())!=request:
            raise SystemExit('Interrupted download belongs to a different snapshot; use a fresh directory.')
    else:
        if a.out.exists() and any(a.out.iterdir()):
            raise SystemExit('Output directory is nonempty and has no matching download request; use a fresh directory.')
        atomic_json(request_path,request)
    print(f'Downloading {a.model_id}@{commit} into {a.out}. No training is launched.')
    snapshot_download(repo_id=a.model_id,revision=commit,local_dir=str(a.out),
                      allow_patterns=['*.json','*.safetensors','*.model','*.tiktoken','*.txt','README.md','LICENSE*'],
                      max_workers=a.max_workers)
    if not (a.out/'config.json').exists() or not list(a.out.glob('*.safetensors')):
        raise SystemExit('Incomplete snapshot: expected config.json and safetensors weights.')
    files={str(f.relative_to(a.out)):{'sha256':sha256_file(f),'bytes':f.stat().st_size}
           for f in sorted(a.out.rglob('*')) if f.is_file() and '.cache' not in f.parts}
    manifest={'model_id':a.model_id,'resolved_revision':commit,'requested_revision':a.revision,
              'created_utc':utc_now(),'files':files,'trust_remote_code':False,
              'qualification':'Downloaded bytes only. Weight loading and device parity are not qualified.'}
    atomic_json(a.out/'ASSET_MANIFEST.json',manifest)
    print(json.dumps({'model_id':a.model_id,'resolved_revision':commit,'files':len(files)},indent=2))
if __name__=='__main__':main()
