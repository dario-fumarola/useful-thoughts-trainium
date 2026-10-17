"""Fixed-note reader probe: retain the entire original passage immediately before targets."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

from train.evaluate_local import load_selected, setup_model
from train.evaluate_likelihood import score_continuations
from train.evaluation_integrity import evaluation_dependencies
from weekend_thinking.artifacts import atomic_json, sha256_file, utc_now

ROOT = Path(__file__).resolve().parents[1]
MODES = ('empty','own','filler','shuffled','other')


def sources():
    return {**evaluation_dependencies(),'train/prepend_probe.py':sha256_file(__file__)}


def contexts(prefix, content, p):
    fmt = p['format']
    return dict(original=fmt['header_ids']+prefix+fmt['open_ids']+content+fmt['close_ids'],
        prepend_plain=content+p['separator_ids']+prefix,
        prepend_labeled=p['notes_label_ids']+content+p['passage_label_ids']+prefix,
        reanchor512=fmt['header_ids']+prefix+fmt['open_ids']+content+fmt['close_ids']+prefix)


def prepare(output):
    from transformers import AutoTokenizer
    path=ROOT/'local/nvme/diagnostics/prompt-interface-dev32-v1/protocol.json'
    pilot=json.loads(path.read_text())
    tok=AutoTokenizer.from_pretrained(pilot['model_dir'],local_files_only=True)
    fmt=json.loads(Path(pilot['paths']['format']).read_text())['format']
    inputs=dict(pilot_protocol=path,reference_scores=path.parent/'scores.jsonl',
                controls=Path(pilot['paths']['controls']),format=Path(pilot['paths']['format']))
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'protocol.json',dict(experiment_id=output.name,prepared_utc=utc_now(),
        selection=pilot['selection'],strata=pilot['strata'],model_dir=pilot['model_dir'],
        records=pilot['records'],format=fmt,source_files=sources(),
        paths={k:str(v) for k,v in inputs.items()},hashes={k:sha256_file(v) for k,v in inputs.items()},
        interfaces=['original','prepend_plain','prepend_labeled','reanchor512'],
        separator_ids=tok.encode('\n\n',add_special_tokens=False),
        notes_label_ids=tok.encode('Working notes:\n',add_special_tokens=False),
        passage_label_ids=tok.encode('\n\nPassage:\n',add_special_tokens=False),
        width=2304,horizons=[1,16],compute='fp32',candidates_per_item=1,
        limitation='Adaptive fixed DEV32 reader-only probe on original saved G1 notes. Prepending reorders reader context; writer remains prefix-only. No policy training or deployment claim.'))


def run(output):
    p=json.loads((output/'protocol.json').read_text())
    if p['source_files']!=sources():raise ValueError('Source changed')
    for key,path in p['paths'].items():
        if sha256_file(path)!=p['hashes'][key]:raise ValueError('Input changed')
    controls={r['item_id']:r for r in map(json.loads,Path(p['paths']['controls']).read_text().splitlines())}
    if any(controls[k]['failures'] for k in p['selection']['item_ids']):raise ValueError('Failed original note')
    expected_keys={(key,'raw','raw') for key in p['selection']['item_ids']}
    expected_keys|={(key,'original',mode) for key in p['selection']['item_ids'] for mode in MODES}
    ref={}
    for row in map(json.loads,Path(p['paths']['reference_scores']).read_text().splitlines()):
        key=(row['item_id'],row['interface'],row['mode'])
        if key not in expected_keys:continue
        values=row.get('token_log_likelihoods',[])
        if key in ref or row['status']!='ok' or len(values)!=16 or not all(math.isfinite(v) for v in values):
            raise ValueError('Duplicate/failed/incomplete/nonfinite required reference')
        ref[key]=row
    if set(ref)!=expected_keys:raise ValueError('Missing required raw/original reference')
    records=load_selected(Path(p['records']),p['selection'],include_future=True)
    backend,_,checkpoint,model,_=setup_model(SimpleNamespace(model_dir=Path(p['model_dir']),checkpoint=None,device='xla'))
    if checkpoint is not None:raise ValueError('Require unchanged B0')
    atomic_json(output/'started.json',dict(time_utc=utc_now(),model=model))
    count=0;maximum=0.
    with (output/'scores.jsonl').open('x') as stream:
        for record in records:
            key=record['item_id'];prefix=record['prefix']
            cases=[('raw','raw',prefix),('raw','repeat512',prefix+prefix)]
            cases += [(interface,mode,contexts(prefix,controls[key]['content_ids'][mode],p)[interface])
                      for interface in p['interfaces'] for mode in MODES]
            for interface,mode,ids in cases:
                start=time.monotonic();row=dict(item_id=key,document_id=record['document_id'],interface=interface,mode=mode)
                try:
                    values=score_continuations(backend,ids,[record['future']],total_width=p['width'],future_width=16,pad_id=p['format']['pad_id'])[0].token_log_likelihoods
                    if len(values)!=16 or not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite/incomplete likelihood')
                    row.update(status='ok',token_log_likelihoods=values)
                    expected=ref[(key,interface,mode)] if interface=='original' or (interface,mode)==('raw','raw') else None
                    if expected is not None:
                        error=max(abs(a-b) for a,b in zip(values,expected['token_log_likelihoods']));maximum=max(maximum,error)
                        row['reference_max_error']=error
                        if error>.002:raise ValueError('FP32 reference mismatch')
                except Exception as exc:row.update(status='error',error=f'{type(exc).__name__}: {exc}')
                row['wall_seconds']=time.monotonic()-start
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();count+=1
                if row['status']!='ok':raise RuntimeError('Failed reader context retained; no retry')
            print(json.dumps(dict(completed_items=count//22,utc=utc_now())),flush=True)
    atomic_json(output/'summary.json',dict(status='complete',items=len(records),rows=count,failures=0,
        completed_utc=utc_now(),protocol_sha256=sha256_file(output/'protocol.json'),
        scores_sha256=sha256_file(output/'scores.jsonl'),maximum_reference_error=maximum))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run']);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if a.command=='prepare':prepare(a.output)
    else:run(a.output)
