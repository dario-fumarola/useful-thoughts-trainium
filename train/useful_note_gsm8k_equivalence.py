"""Fixed first32/96-path GSM topology equivalence, never an accuracy summary.

Uses ALL512-prepared common shapes and the exact ordinary generate_item policy.
No gold is read; this distinct schema cannot pass the full1536-path loader.
"""
import argparse,json,time
from pathlib import Path
from train import useful_note_gsm8k as gsm
from train import useful_note_evaluation as shared
from weekend_thinking.artifacts import atomic_json,canonical_hash,sha256_file,utc_now

SCHEMA='gsm8k-first32-native96-equivalence-v1'
ITEMS,SHARDS=32,16
REFERENCE_PROTOCOL_SHA256='7d67b7bf1c1ba4a2d55d6ba26757ce0cb76f96625388995cbb1f515099db9760'
REFERENCE_ARCHIVE_SHA256='5080542c82522e491e65975cb23bfe7d5037c2125438fbaf513d4b6eea4faf3d'
FIELDS=('item_id','document_id','task','split','mode','selection_policy','status','question_prompt_ids','writer_prompt_ids',
 'reader_suffix_ids','answer_prompt_ids','thought_actions','thought_content_ids','thought_text','thought_stop_type',
 'answer_actions','answer_content_ids','generated_text','raw_generated_text','answer_stop_type','n_generated_tokens',
 'n_answer_tokens','n_thought_tokens','answer_cap','thought_cap')

def sources():return {**gsm.sources(),'train/useful_note_gsm8k_equivalence.py':sha256_file(__file__)}
def key(v):return f'{v["item_id"]}/{v["mode"]}'
def projection(v):return {name:v.get(name) for name in FIELDS}


def reference_rows(full,questions,original,archive):
    original=Path(original).resolve();archive=Path(archive).resolve()
    shared.verify_pins({str(archive):REFERENCE_ARCHIVE_SHA256,str(original/'protocol.json'):REFERENCE_PROTOCOL_SHA256});a=shared.read(archive)
    if a['status']!='complete_SHA256_source_and_destination_readback_verified':raise ValueError('Verified original B0 archive required')
    entries={v['source']:v['sha256'] for v in a['entries']};pins={str(archive):sha256_file(archive)}
    def bound(path):
        path=Path(path);digest=entries[str(path)];shared.verify_pins({str(path):digest});pins[str(path)]=digest;return path
    old=shared.read(bound(original/'protocol.json'))
    if (old['schema']!='useful-note-gsm8k-train512-v1' or old['items']!=512 or old['shards']!=4
        or old['model']!={'id':'B0','checkpoint':None,'checkpoint_sha256':None}
        or old['ordered_item_ids']!=full['ordered_item_ids'] or old['ordered_document_ids']!=full['ordered_document_ids']
        or old['projection_hashes']!=full['projection_hashes'] or old['assets']!=full['assets']
        or any(old[k]!=full[k] for k in ('modes','thought_cap','max_answer_tokens','compute','head_precision','parser','selection_policy',
            'question_prefill_width','answer_prefill_width','question_prompt_lengths','reader_suffix_lengths'))
        or {k:v for k,v in old['source_files'].items() if k!='train/useful_note_gsm8k.py'}!=
           {k:v for k,v in full['source_files'].items() if k!='train/useful_note_gsm8k.py'}):
        raise ValueError('Original full512 B0 policy/projections/shapes and unchanged numerical dependencies required')
    values=[];identity=None
    for shard in range(4):
        d=original/'generation'/f'shard-{shard}';r=shared.read(bound(d/'run.json'));vv=shared.rows(bound(d/'predictions.jsonl'))
        expected=[(q['item_id'],mode) for mode in gsm.MODES for q in gsm.evaluator.shard_questions(questions,shard,4)]
        if (r['status']!='complete_attempts' or r['rows']!=384 or r['protocol_sha256']!=sha256_file(original/'protocol.json')
            or r['predictions_sha256']!=sha256_file(d/'predictions.jsonl') or [(v['item_id'],v['mode']) for v in vv]!=expected
            or r['model']['id']!='B0' or r['model']['checkpoint'] is not None or identity is not None and identity!=r['model']):
            raise ValueError('All original closed B0 reference paths required')
        identity=r['model'];values.extend(vv)
    ids=full['ordered_item_ids'][:ITEMS];selected={key(v):projection(v) for v in values if v['item_id'] in ids}
    if len(values)!=1536 or len(selected)!=96 or any(v['status']=='error' for v in selected.values()):raise ValueError('All96 fixed reference paths required')
    return selected,pins,identity


def prepare(output,full_preparation,original,archive,until,control_dir):
    shared.check(until);full_preparation=Path(full_preparation).resolve();p,questions=gsm.load(full_preparation)
    if p['shards']!=SHARDS or p['model']!={'id':'B0','checkpoint':None,'checkpoint_sha256':None}:raise ValueError('Actual full512 B0 sixteen-shard preparation required')
    refs,pins,identity=reference_rows(p,questions,original,archive)
    out=Path(output);out.mkdir(parents=True,exist_ok=False);atomic_json(out/'references.json',refs)
    q=dict(schema=SCHEMA,created_utc=utc_now(),purpose='Native action/stop/status equivalence only; no gold, accuracy or full-suite claim',
        source_files=sources(),full_preparation=str(full_preparation),full_protocol_sha256=sha256_file(full_preparation/'protocol.json'),
        input_hashes={**pins,**shared.pin([full_preparation/'protocol.json',full_preparation/'questions.jsonl',full_preparation/'prompt.json',full_preparation/'format.json'])},
        reference_model=identity,reference_sha256=sha256_file(out/'references.json'),ordered_item_ids=p['ordered_item_ids'][:ITEMS],
        items=ITEMS,paths=96,shards=SHARDS,full_shape_items=512,common_widths={k:p[k] for k in ('question_prefill_width','answer_prefill_width','question_prompt_lengths','reader_suffix_lengths')},
        deadline_utc=until,control_dir=str(Path(control_dir).resolve()),native_reference_indices=[i%4 for i in range(SHARDS)])
    atomic_json(out/'protocol.json',q);return q


def load(output):
    out=Path(output);q=shared.read(out/'protocol.json')
    if q['schema']!=SCHEMA or q['source_files']!=sources() or (q['items'],q['paths'],q['shards'],q['full_shape_items'])!=(32,96,16,512):raise ValueError('Frozen fixed96 equivalence schema required')
    shared.verify_pins(q['input_hashes']);shared.verify_pins({str(out/'references.json'):q['reference_sha256']})
    p,questions=gsm.load(Path(q['full_preparation']))
    if (sha256_file(Path(q['full_preparation'])/'protocol.json')!=q['full_protocol_sha256'] or p['shards']!=16
        or p['model']!={'id':'B0','checkpoint':None,'checkpoint_sha256':None} or q['ordered_item_ids']!=p['ordered_item_ids'][:32]
        or q['common_widths']!={k:p[k] for k in q['common_widths']} or q['native_reference_indices']!=[i%4 for i in range(16)]):raise ValueError('Original full512 shapes and first32 changed')
    return q,p,questions


def generate(output,shard,until,device='xla'):
    import torch
    out=Path(output);q,p,questions=load(out);shared.check(until)
    if type(shard)is not int or shard not in range(16) or until!=q['deadline_utc']:raise ValueError('Exact shard/deadline required')
    d=out/'generation'/f'shard-{shard}';d.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    r=dict(status='started',rows=0,shard=shard,native_reference_index=shard%4,started_utc=utc_now(),protocol_sha256=sha256_file(out/'protocol.json'));atomic_json(d/'run.json',r)
    try:
        backend,tok,actual=shared.load_model(p,p['model'],device);gsm.validate_identity(actual);r['model']=actual
        if actual!=q['reference_model'] or any(v.dtype!=torch.float32 for v in backend.model.parameters()):raise ValueError('Exact initial FP32 B0 required')
        r['gate']=shared.base_gate(backend,p,shard%4,d,actual)
        full=Path(q['full_preparation']);fmt=gsm.validate_format(tok,shared.read(full/'format.json'));prompt=shared.read(full/'prompt.json')
        if gsm.measure_widths(questions,prompt,fmt,tok)!=q['common_widths']:raise ValueError('Measure ALL512 native common shapes, never only32')
        decoder=gsm.evaluator.GreedyDecoder(backend.model,tok,sync=backend.sync,banned_ids=backend.banned_ids)
        local=[v for v in gsm.evaluator.shard_questions(questions,shard,16) if v['item_id'] in q['ordered_item_ids']]
        if len(local)!=2:raise ValueError('Two first32 questions per strided worker required')
        with (d/'predictions.jsonl').open('x') as stream:
            for mode in gsm.MODES:
                for question in local:
                    shared.check(until);v=gsm.evaluator.generate_item(question,decoder=decoder,prompt_spec=prompt,fmt=fmt,mode=mode,max_answer_tokens=gsm.ANSWER_CAP,thought_cap=gsm.THOUGHT_CAP,
                        question_prefill_width=p['question_prefill_width'],answer_prefill_width=p['answer_prefill_width'])
                    stream.write(json.dumps(v,allow_nan=False)+'\n');stream.flush();r['rows']+=1
        r.update(status='complete_equivalence_attempts',predictions_sha256=sha256_file(d/'predictions.jsonl'))
    except BaseException as error:r.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:r.update(finished_utc=utc_now(),stage_wall_seconds=time.monotonic()-started);atomic_json(d/'run.json',r,overwrite=True)


def reduce_rows(values,references,ids):
    expected={f'{i}/{mode}' for i in ids for mode in gsm.MODES};index={key(v):v for v in values}
    if len(ids)!=32 or len(set(ids))!=32 or len(values)!=96 or len(index)!=96 or set(index)!=expected or set(references)!=expected:raise ValueError('Exactly all96 distinct paths required; no partial PASS')
    return [dict(item_id=v['item_id'],mode=v['mode'],differing_fields=[k for k in FIELDS if projection(v)[k]!=references[name][k]]) for name,v in index.items() if projection(v)!=references[name]]


def compare(output):
    out=Path(output);q,p,questions=load(out);closure=Path(q['control_dir'])/'generate.closure.json';c=shared.read(closure)
    if (c['status']!='COMPLETE_PHYSICALLY_CLOSED' or not c['all16_exit0'] or not c['all48_owned_PID_start_identities_absent'] or not c['all_sessions_absent']
        or not c['assigned_devices_empty'] or c['owned_group_survivors'] or c['protocol_sha256']!=sha256_file(out/'protocol.json')):raise ValueError('All sixteen jobs physically closed before comparison')
    values=[];pins={}
    for shard in range(16):
        d=out/'generation'/f'shard-{shard}';r=shared.read(d/'run.json');vv=shared.rows(d/'predictions.jsonl')
        ids=q['ordered_item_ids'][shard::16];expected=[(i,mode) for mode in gsm.MODES for i in ids]
        if (r['status']!='complete_equivalence_attempts' or r['rows']!=6 or r['shard']!=shard or r['native_reference_index']!=shard%4
            or r['model']!=q['reference_model'] or r['protocol_sha256']!=sha256_file(out/'protocol.json') or r['predictions_sha256']!=sha256_file(d/'predictions.jsonl')
            or [(v['item_id'],v['mode']) for v in vv]!=expected):raise ValueError('All sixteen original fixed96 path receipts required')
        values.extend(vv);pins.update(shared.pin([d/'run.json',d/'predictions.jsonl']))
    differences=reduce_rows(values,shared.read(out/'references.json'),q['ordered_item_ids'])
    result=dict(schema=SCHEMA,status='FAIL_NATIVE_PATH_DIFFERENCES' if differences else 'PASS_FIXED96_NATIVE_PATHS',paths=96,items=32,
        exact_fields=list(FIELDS),differences=differences,protocol_sha256=sha256_file(out/'protocol.json'),input_hashes={**pins,**shared.pin([closure])},
        scope='All native action/prompt/stop/status/text fields, fixed first32 only; no gold or accuracy read, and no full1536 suite claim.')
    atomic_json(out/'comparison.json',result)
    if differences:raise RuntimeError('Fixed96 native equivalence failed; original paths retained')
    return result


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('command',choices=('prepare','generate','compare'));a.add_argument('--output',type=Path,required=True)
    for name in ('full-preparation','original','archive','control-dir'):a.add_argument('--'+name,type=Path)
    a.add_argument('--deadline-utc');a.add_argument('--shard',type=int);a.add_argument('--device',choices=('cpu','xla'),default='xla');v=a.parse_args()
    if v.command=='prepare':prepare(v.output,v.full_preparation,v.original,v.archive,v.deadline_utc,v.control_dir)
    elif v.command=='generate':generate(v.output,v.shard,v.deadline_utc,v.device)
    else:print(json.dumps(compare(v.output),indent=2))

if __name__=='__main__':main()
