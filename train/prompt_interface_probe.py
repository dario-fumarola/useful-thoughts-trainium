"""Reader-only DEV probe of published RLP instructions and plain boundaries.

The published prompt has an apparent delimiter typo and lacks verified runtime
serialization. Test its transcription and a clearly labeled correction. Saved
original-writer notes stay fixed; this is not an RLP replication or new policy.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

from train.evaluate_local import load_selected, setup_model
from train.evaluate_likelihood import score_continuations
from train.evaluation_integrity import evaluation_dependencies
from weekend_thinking.artifacts import atomic_json, sha256_file, utc_now

ROOT=Path(__file__).resolve().parents[1]
MODES=('empty','own','filler','shuffled','other')


def sources():
    return dict(evaluation_dependencies(), **{'train/prompt_interface_probe.py':sha256_file(__file__)})


def context(prefix, content, config):
    # Every token in the repeated suffix is already present in the prefix.
    suffix=prefix[-config['repeat']:] if config['repeat'] else []
    if content is None:
        return list(prefix)+list(suffix)
    return config['header']+list(prefix)+config['open']+list(content)+config['close']+list(suffix)


def prepare(output):
    from transformers import AutoTokenizer
    pilot_path=ROOT/'local/nvme/diagnostics/interface-tax-dev32-v1/protocol.json'
    pilot=json.loads(pilot_path.read_text())
    source_path=ROOT/'local/analysis/new-direction-review-20261017/rlp-prompt-retrieval.json'
    research=json.loads(source_path.read_text())
    # Typographic quotation marks surround the displayed prompt, not its text.
    printed=research['transcription'].strip('”')
    corrected=printed.replace('between </think> and </think>','between <think> and </think>',1)
    assert corrected!=printed
    tok=AutoTokenizer.from_pretrained(pilot['model_dir'],local_files_only=True)
    enc=lambda s:tok.encode(s,add_special_tokens=False)
    old=json.loads(Path(pilot['paths']['format']).read_text())['format']
    def variant(header,repeat=0):
        return dict(header=enc(header+'\n\nPassage:\n'),open=old['open_ids'],close=old['close_ids'],repeat=repeat)
    configurations=dict(
        original=dict(header=old['header_ids'],open=old['open_ids'],close=old['close_ids'],repeat=0),
        published_RLP_instruction=variant(printed),
        corrected_RLP_instruction=variant(corrected),
        corrected_RLP_reanchor32=variant(corrected,32),
        plain_reanchor32=dict(header=[],open=enc('\n'),close=enc('\n'),repeat=32))
    paths=dict(pilot_protocol=pilot_path,controls=Path(pilot['paths']['controls']),
        reference_scores=pilot_path.parent/'scores.jsonl',prompt_source=source_path,
        format=Path(pilot['paths']['format']))
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'protocol.json',dict(experiment_id=output.name,prepared_utc=utc_now(),
        selection=pilot['selection'],strata=pilot['strata'],model_dir=pilot['model_dir'],
        records=pilot['paths']['records'],configurations=configurations,width=1792,horizons=[1,16],
        pad_id=old['pad_id'],compute='fp32',candidates_per_item=1,
        paths={key:str(path) for key,path in paths.items()},
        hashes={key:sha256_file(path) for key,path in paths.items()},source_files=sources(),
        prompt_provenance={key:research[key] for key in ('source','source_location','source_license','caveat')},
        serialization='Plain instruction+two newlines+Passage:+prefix, existing open/close; not a verified paper serializer',
        limitation='Exploratory reader interventions on original saved G1 notes; no new writer, training or deployment'))
    print(json.dumps({'prepared':str(output),'interfaces':list(configurations)}),flush=True)


def run(output):
    protocol=json.loads((output/'protocol.json').read_text())
    if protocol['source_files']!=sources():raise ValueError('Source changed')
    for key,path in protocol['paths'].items():
        if sha256_file(path)!=protocol['hashes'][key]:raise ValueError(f'Input changed: {key}')
    controls={r['item_id']:r for r in map(json.loads,Path(protocol['paths']['controls']).read_text().splitlines())}
    for key in protocol['selection']['item_ids']:
        if controls[key]['failures']:raise ValueError('Original note/control failure')
    records=load_selected(Path(protocol['records']),protocol['selection'],include_future=True)
    reference={(r['item_id'],r['interface'],r['mode']):r for r in
        map(json.loads,Path(protocol['paths']['reference_scores']).read_text().splitlines())}
    backend,_,_,signature,_=setup_model(SimpleNamespace(
        model_dir=Path(protocol['model_dir']),checkpoint=None,device='xla'))
    atomic_json(output/'started.json',dict(time_utc=utc_now(),model=signature,device=str(backend.device)))
    rows=[];maximum_error=0.
    with (output/'scores.jsonl').open('x') as stream:
        for record in records:
            key=record['item_id']
            conditions=[('raw',mode,list(record['prefix'])+(record['prefix'][-repeat:] if repeat else []))
                        for mode,repeat in [('raw',0),('repeat32',32)]]
            conditions += [(name,mode,context(record['prefix'],controls[key]['content_ids'][mode],config))
                for name,config in protocol['configurations'].items() for mode in MODES]
            for name,mode,ids in conditions:
                started=time.monotonic()
                row=dict(item_id=key,document_id=record['document_id'],interface=name,mode=mode)
                try:
                    score=score_continuations(backend,ids,[record['future']],total_width=protocol['width'],
                        future_width=16,pad_id=protocol['pad_id'])[0]
                    values=score.token_log_likelihoods
                    if len(values)!=16 or not all(math.isfinite(v) for v in values):
                        raise ValueError('Invalid/nonfinite token likelihoods')
                    row.update(status='ok',token_log_likelihoods=values)
                    if name in ('raw','original'):
                        expected=reference[(key,name,mode)]
                        error=max(abs(x-y) for x,y in zip(values,expected['token_log_likelihoods']))
                        row['reference_max_error']=error;maximum_error=max(maximum_error,error)
                        if error>.002:raise ValueError(f'FP32 reference mismatch {error}')
                except Exception as exc:
                    row.update(status='error',error=f'{type(exc).__name__}: {exc}')
                row['wall_seconds']=time.monotonic()-started
                rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                if row['status']!='ok':raise RuntimeError(f'Failure retained: {row}')
            print(json.dumps(dict(time_utc=utc_now(),completed_items=len(rows)//27,total_items=len(records))),flush=True)
    groups={}
    for group in ['all',*sorted(set(protocol['strata'].values()))]:
        selected=[r['item_id'] for r in records if group=='all' or protocol['strata'][r['item_id']]==group]
        cells={}
        for h in protocol['horizons']:
            index={(r['item_id'],r['interface'],r['mode']):math.fsum(r['token_log_likelihoods'][:h]) for r in rows}
            for name,config in protocol['configurations'].items():
                def values(interface,mode):return [index[(key,interface,mode)] for key in selected]
                own=values(name,'own');empty=values(name,'empty')
                comparisons={'empty_minus_raw':(empty,values('raw','raw')),
                    'own_minus_raw':(own,values('raw','raw')),
                    'own_minus_raw_repeat':(own,values('raw','repeat32' if config['repeat'] else 'raw')),
                    **{f'own_minus_{mode}':(own,values(name,mode)) for mode in MODES if mode!='own'}}
                cells[f'{name}/K{h}']=dict(items=len(selected),
                    absolute_means={mode:statistics.mean(values(name,mode)) for mode in MODES},
                    contrasts={label:dict(mean_nats=statistics.mean(x-y for x,y in zip(a,b)),
                        positive_documents=sum(x>y for x,y in zip(a,b))) for label,(a,b) in comparisons.items()})
        groups[group]=cells
    atomic_json(output/'summary.json',dict(status='complete',experiment_id=protocol['experiment_id'],
        completed_utc=utc_now(),rows=len(rows),items=len(records),groups=groups,
        maximum_reference_error=maximum_error,protocol_sha256=sha256_file(output/'protocol.json'),
        scores_sha256=sha256_file(output/'scores.jsonl'),limitation=protocol['limitation']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','run']);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='prepare':prepare(a.output)
    else:run(a.output)
