"""Secondary synthetic reader diagnostic; never replaces real-data endpoints.

All256 prefix-only note attempts physically close before global donors or
future scoring. Exactly4096 full-span likelihood cells, no answer decoder.
Full supplied notes reveal the answer; partial notes omit the root operation.
"""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import math
import time

from train import controlled_arithmetic_data as dataset
from train import general_note_evaluation as base
from train import useful_note_evaluation as shared
from train.interface_oracle_probe import reader_context
from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file, utc_now
from weekend_thinking.stats import paired_cluster_bootstrap

SCHEMA = 'controlled-compound-arithmetic256-reader-v1'
DATA_MANIFEST_SHA256 = '7e067c86315a83476b80fc1c9369ea4f78bc08c11a2db2d9013768085dabda50'
N, SHARDS, CAP, WIDTH = 256, 4, 128, 1024
MODES = ('raw', 'empty', 'full_correct', 'full_wrong', 'partial_correct', 'partial_wrong', 'own', 'other')
PATHS = ('I3', 'I6w0.375')
NOTE_KEYS = dict(full_correct='correct', full_wrong='wrong', partial_correct='partial_correct', partial_wrong='partial_wrong')
CONTRASTS = (('raw', 'empty'), ('raw', 'full_correct'), ('empty', 'full_correct'),
    ('full_wrong', 'full_correct'), ('raw', 'partial_correct'), ('empty', 'partial_correct'),
    ('partial_wrong', 'partial_correct'), ('raw', 'own'), ('empty', 'own'), ('other', 'own'), ('own', 'full_correct'))


def sources():
    return {**base.sources(), 'train/controlled_arithmetic_data.py': sha256_file(dataset.__file__),
        'train/controlled_arithmetic_probe.py': sha256_file(__file__)}


def validate_prefixes(rows, ids, docs):
    fields = {'item_id', 'document_id', 'split', 'prefix_ids', 'reanchor_ids'}
    if (len(rows) != N or len(set(ids)) != N or len(set(docs)) != N
            or [r['item_id'] for r in rows] != ids or [r['document_id'] for r in rows] != docs
            or any(set(r) != fields or r['split'] != 'synthetic_dev' or not r['prefix_ids'] or not r['reanchor_ids'] for r in rows)):
        raise ValueError('All256 exact future-free synthetic prefixes required')


def writer_identity(checkpoint=None, checkpoint_sha256=None, complete_sha256=None, training_protocol=None):
    values=(checkpoint,checkpoint_sha256,complete_sha256,training_protocol)
    if all(v is None for v in values):return dict(base.B0)
    if any(v is None for v in values):raise ValueError('All actual trained checkpoint lineage pins required')
    return shared.trained_identity(*values)


def validate_writer(writer):
    if writer.get('id')=='B0':
        if writer!=base.B0:raise ValueError('Unchanged B0 identity required')
    elif (writer.get('id') not in ('W','Cprime','T') or any(not writer.get(k) for k in
            ('checkpoint','checkpoint_sha256','provenance','checkpoint_files','training_source_files','run_id','step','spec_hash'))):
        raise ValueError('Actual terminal trained writer identity required')
    return writer


def prepare(output, data_dir, native_reference, until, control_dir, writer=None):
    from transformers import AutoTokenizer
    from train.interface_oracle_probe import load as load_native, reference
    from train.useful_note_sft_data import writer_prompt, writer_stop_ids
    shared.check(until)
    writer=validate_writer(dict(base.B0) if writer is None else writer)
    if writer['id']!='B0':shared.verify_pins(writer['provenance'])
    data_dir, native_reference = Path(data_dir).resolve(), Path(native_reference).resolve()
    shared.verify_pins({str(data_dir/'manifest.json'): DATA_MANIFEST_SHA256})
    manifest = shared.read(data_dir/'manifest.json')
    if (manifest['schema'] != dataset.SCHEMA or manifest['status'] != 'FROZEN_BEFORE_ANY_MODEL_OUTCOME'
            or manifest['items'] != N or manifest['families'] != {key: 64 for key in dataset.FAMILIES}
            or manifest['source_files'] != {'train/controlled_arithmetic_data.py': sha256_file(dataset.__file__)}):
        raise ValueError('Frozen complete synthetic data publication required')
    shared.verify_pins(manifest['input_hashes'])
    shared.verify_pins({str(data_dir/'prefixes.jsonl'): manifest['prefixes_sha256'],
        str(data_dir/'reader-inputs.json'): manifest['reader_inputs_sha256']})
    prefixes = shared.rows(data_dir/'prefixes.jsonl')
    rows = shared.read(data_dir/'reader-inputs.json')['rows']
    ids, docs = manifest['ordered_item_ids'], [r['document_id'] for r in prefixes]
    validate_prefixes(prefixes, ids, docs)
    if [dataset.prefix_projection(r) for r in rows] != prefixes:
        raise ValueError('Writer and reader native prefixes differ')
    old, native = load_native(native_reference)
    ref, _ = reference(native_reference, old, native)
    tok = AutoTokenizer.from_pretrained(old['model_dir'], local_files_only=True, trust_remote_code=False)
    if (Path(manifest['model_dir']).resolve() != Path(old['model_dir']).resolve()
            or manifest['tokenizer_vocab_sha256'] != canonical_hash(tok.get_vocab())
            or writer_stop_ids('I3', native['parts']) != (151668, 151643)):
        raise ValueError('Exact native tokenizer/I3/first-close/EOS contract required')
    defined = set(tok.get_vocab().values())
    enc = lambda text: tok.encode(text, add_special_tokens=False)
    for row in rows:
        if (set(row['supplied_note_ids']) != set(NOTE_KEYS.values())
                or any(row[key+'_ids'] != enc(row[key+'_text']) for key in ('prefix', 'reanchor', 'target'))
                or row['prefix_ids'] + row['target_ids'] != enc(row['prefix_text'] + row['target_text'])
                or any(row['supplied_note_ids'][key] != enc(row['supplied_notes'][key]) for key in NOTE_KEYS.values())
                or not 1 <= len(row['target_ids']) <= 16
                or len(row['numeric_only_mask']) != len(row['target_ids'])
                or any(type(v) is not bool for v in row['numeric_only_mask'])
                or row['numeric_only_supported'] != any(row['numeric_only_mask'])
                or any(t not in defined for t in row['prefix_ids'] + row['target_ids'] + row['reanchor_ids'])):
            raise ValueError('Exact supplied native surfaces, targets and descriptive mask required')
    widths = [len(writer_prompt(r['prefix_ids'], 'I3', native['parts'])) for r in prefixes]
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    (output/'prefixes.jsonl').write_bytes((data_dir/'prefixes.jsonl').read_bytes())
    inherited = {key: old[key] for key in ('model_dir', 'assets')}
    p = dict(schema=SCHEMA, created_utc=utc_now(), **inherited, source_files=sources(), writer=writer,
        model_manifest_sha256=old['assets']['manifest_sha256'],
        base_signature=ref['signature'], native_reference=str(native_reference), parts=native['parts'],
        native_inputs=shared.pin([native_reference/'protocol.json', native_reference/'inputs.json',
            native_reference/'cpu-reference/reference.json', native_reference/'cpu-reference/next-logps.pt']),
        input_hashes=shared.pin([data_dir/'manifest.json', data_dir/'prefixes.jsonl']),
        reader_inputs_file=str(data_dir/'reader-inputs.json'), reader_inputs_sha256=manifest['reader_inputs_sha256'],
        prefixes_sha256=sha256_file(output/'prefixes.jsonl'), ordered_item_ids=ids, ordered_document_ids=docs,
        items=N, shards=SHARDS, max_actions=CAP, width=WIDTH, interface='I3', paths=list(PATHS), modes=list(MODES),
        scoring_positions=16, unused_target_pad_id=151643, score_target_token_lengths=dict(Counter(len(r['target_ids']) for r in rows)),
        writer_prefill_width=((max(widths)+31)//32)*32, writer_stop_ids=[151668,151643],
        vocab_size=old['vocab_size'], banned_ids=sorted(set(range(old['vocab_size'])) - defined), precision='fp32', selection_policy='single_greedy',
        control_dir=str(Path(control_dir).resolve()), deadline_utc=until, bootstrap=dict(repeats=5000, seed=17),
        primary='Exact signed-integer plus leading space/newline full-span LL. Reader gain and correct-minus-wrong CIs; raw headroom reported, no universal +2nat gate.',
        limitations=manifest['limitations'] + ['All eight conditions and failures retained; no outcome-based replacement.',
            'I6=.375 interpolates full logits; no likelihood-only mixture or fit.',
            'Own is prefix-only greedy cap128 TOTAL actions; intact cyclic unrelated donors retain empty/error outcomes.',
            'Numeric-only is descriptive, digit-only native-token support; no silent subset or error imputation.',
            'All head/mix computations use16 positions; only actual native target slots enter saved vectors/metrics, never unused EOS filler.',
            'Same frozen256 supports B0 and later terminal writers; scoring uses that SAME model as reader, no crossed-reader matrix.',
            'No generated scalar answers, deployed arithmetic accuracy or generalization claim.'])
    atomic_json(output/'protocol.json', p); return p


def load(output):
    output = Path(output); p = shared.read(output/'protocol.json')
    if (p['schema'] != SCHEMA or p['source_files'] != sources() or (p['items'],p['shards'],p['max_actions'],p['width']) != (N,SHARDS,CAP,WIDTH)
            or p['modes'] != list(MODES) or p['paths'] != list(PATHS)
            or p['scoring_positions'] != 16 or p['unused_target_pad_id'] != 151643
            or p['interface'] != 'I3' or p['precision'] != 'fp32' or p['selection_policy'] != 'single_greedy'
            or p['writer_stop_ids'] != [151668,151643] or p['bootstrap'] != dict(repeats=5000,seed=17)):
        raise ValueError('Frozen synthetic schema/source/recipe required')
    validate_writer(p['writer'])
    shared.verify_pins(p['input_hashes']); shared.verify_pins(p['native_inputs'])
    shared.verify_pins({str(output/'prefixes.jsonl'):p['prefixes_sha256']})
    prefixes = shared.rows(output/'prefixes.jsonl')
    validate_prefixes(prefixes,p['ordered_item_ids'],p['ordered_document_ids'])
    return p,prefixes


def generate(output, shard, until, device='xla'):
    import torch
    from train.evaluate import GreedyDecoder, GenerationFailure
    from train.useful_note_sft_data import writer_prompt, writer_stop_ids
    output=Path(output); p,prefixes=load(output); shared.check(until)
    if type(shard) is not int or shard not in range(SHARDS) or until!=p['deadline_utc']:raise ValueError('Fixed shard/deadline required')
    d=output/'notes'/f'shard-{shard}'; d.mkdir(parents=True,exist_ok=False); started=time.monotonic()
    r=dict(status='started',rows=0,shard=shard,started_utc=utc_now(),protocol_sha256=sha256_file(output/'protocol.json')); atomic_json(d/'run.json',r)
    try:
        before=time.monotonic(); backend,tok,actual=shared.load_model(p,p['writer'],device); r.update(writer=actual,model_setup_seconds=time.monotonic()-before)
        before=time.monotonic(); r['gate']=shared.base_gate(backend,p,shard,d,actual); r['native_gate_seconds']=time.monotonic()-before
        decoder=GreedyDecoder(backend.model,tok,sync=backend.sync,banned_ids=backend.banned_ids)
        with (d/'notes.jsonl').open('x') as stream:
            for q in prefixes[shard::SHARDS]:
                shared.check(until); before=time.monotonic(); prompt=writer_prompt(q['prefix_ids'],'I3',p['parts'])
                v=dict(item_id=q['item_id'],document_id=q['document_id'],status='error',actions=[],content_ids=[],prefix_sha256=canonical_hash(q['prefix_ids']),writer_prompt_sha256=canonical_hash(prompt))
                try:
                    with torch.no_grad(): value=decoder.decode(prompt,max_new_tokens=CAP,prefill_width=p['writer_prefill_width'],pad_id=151643,stop_ids=writer_stop_ids('I3',p['parts']),answer_line_stop=False)
                    v.update(status='ok',**asdict(value))
                except Exception as error:
                    if isinstance(error,GenerationFailure):v.update(**asdict(error.partial))
                    v['error']=f'{type(error).__name__}: {error}'
                v.update(wall_seconds=time.monotonic()-before,n_actions=len(v['actions']),n_content_tokens=len(v['content_ids']))
                stream.write(json.dumps(v,allow_nan=False)+'\n'); stream.flush(); r['rows']+=1
        r.update(status='complete_attempts',notes_sha256=sha256_file(d/'notes.jsonl'))
    except BaseException as error:r.update(status='failed',error=f'{type(error).__name__}: {error}'); raise
    finally:r.update(finished_utc=utc_now(),stage_wall_seconds=time.monotonic()-started); atomic_json(d/'run.json',r,overwrite=True)


def controls(output,p,prefixes,create=False):
    closed=base.closure(output,p,'generate'); notes,pins,writer=shared.complete_notes(output,p,prefixes)
    value=dict(status='complete',protocol_sha256=sha256_file(output/'protocol.json'),generation_closure=closed,note_pins=pins,writer=writer,**shared.other_controls(p,notes))
    if create:atomic_json(output/'controls.json',value)
    elif shared.read(output/'controls.json')!=value:raise ValueError('All-global intact note/donor publication changed')
    return notes,value


def reader_rows(p,prefixes):
    shared.verify_pins({p['reader_inputs_file']:p['reader_inputs_sha256']})
    rows=shared.read(p['reader_inputs_file'])['rows']
    if [dataset.prefix_projection(r) for r in rows]!=prefixes:raise ValueError('Reader/writer prefixes differ')
    return rows


def contexts(row,p,notes,control):
    key=row['item_id']; payloads={mode:row['supplied_note_ids'][name] for mode,name in NOTE_KEYS.items()}
    payloads.update(empty=[],own=notes[key]['content_ids'],other=control['other_content_ids'][key])
    result=dict(raw=list(row['prefix_ids']),**{mode:reader_context(row,note,'I3',p['parts']) for mode,note in payloads.items()})
    if set(result)!=set(MODES) or any(len(v)+16>WIDTH for v in result.values()):raise ValueError('All eight bounded contexts required, never truncate')
    return result


def metrics(lp,row):
    if len(lp)!=len(row['target_ids']) or any(type(v) not in (float,int) or not math.isfinite(v) or v>0 for v in lp):raise ValueError('Every finite nonpositive full-span token LP required')
    mask=row['numeric_only_mask']
    if len(mask)!=len(lp) or any(type(v) is not bool for v in mask) or row['numeric_only_supported']!=any(mask):raise ValueError('Original descriptive native mask required')
    return dict(fullspan_ll=sum(lp),numeric_only_ll=sum(v for v,on in zip(lp,mask) if on) if any(mask) else None)


def score(output,shard,until,device='xla'):
    import torch
    from train.interface_mixer_probe import full_logits,mixed_token_logps,token_logps
    output=Path(output); p,prefixes=load(output); shared.check(until)
    if type(shard) is not int or shard not in range(SHARDS) or until!=p['deadline_utc']:raise ValueError('Fixed shard/deadline required')
    notes,control=controls(output,p,prefixes); rows=reader_rows(p,prefixes)
    d=output/'scores'/f'shard-{shard}'; d.mkdir(parents=True,exist_ok=False); started=time.monotonic()
    r=dict(status='started',rows=0,shard=shard,started_utc=utc_now(),protocol_sha256=sha256_file(output/'protocol.json'),controls_sha256=sha256_file(output/'controls.json')); atomic_json(d/'run.json',r)
    try:
        before=time.monotonic(); backend,_,actual=shared.load_model(p,p['writer'],device); r.update(reader=actual,model_setup_seconds=time.monotonic()-before)
        before=time.monotonic(); r['gate']=shared.base_gate(backend,p,shard,d,actual); r['native_gate_seconds']=time.monotonic()-before
        with torch.no_grad(),(d/'scores.jsonl').open('x') as stream:
            for row in rows[shard::SHARDS]:
                key=row['item_id']; ctx=contexts(row,p,notes,control); before=time.monotonic(); raw=raw_lp=y=None; failed=None
                try:
                    raw=full_logits(backend,ctx['raw'],row['target_ids'])
                    y=torch.tensor([row['target_ids']+[151643]*(16-len(row['target_ids']))],device=backend.device)
                    raw_lp=token_logps(raw,y); backend.sync(); raw_values=raw_lp.cpu()[0].tolist()[:len(row['target_ids'])]; metrics(raw_values,row)
                except Exception as error:failed=f'{type(error).__name__}: {error}'
                raw_seconds=time.monotonic()-before
                for mode in MODES:
                    shared.check(until); before=time.monotonic(); cell=dict(item_id=key,document_id=row['document_id'],mode=mode,target_ids=row['target_ids'],context_sha256=canonical_hash(ctx[mode]))
                    try:
                        if failed:raise RuntimeError(failed)
                        if mode=='own' and notes[key]['status']!='ok' or mode=='other' and notes[control['other_donors'][key]]['status']!='ok':raise ValueError('Failed genuine note/donor retained')
                        if mode=='raw':vectors=[raw_values,raw_values]
                        else:
                            note=full_logits(backend,ctx[mode],row['target_ids'])
                            mixed,_,end=mixed_token_logps(raw,note,y,[.375],raw_lp); backend.sync(); vectors=[end.cpu()[0].tolist()[:len(row['target_ids'])],mixed.cpu()[0].tolist()[:len(row['target_ids'])]]
                        emitted=[dict(cell,path=path,status='ok',token_logps=lp,**metrics(lp,row)) for path,lp in zip(PATHS,vectors)]
                    except Exception as error:emitted=[dict(cell,path=path,status='error',error=f'{type(error).__name__}: {error}') for path in PATHS]
                    elapsed=time.monotonic()-before+(raw_seconds if mode=='raw' else 0.)
                    for v in emitted:
                        v.update(wall_seconds=elapsed,timing_group=f'{key}/{mode}',shared_by_paths=list(PATHS),shared_raw_forward_seconds=raw_seconds if mode=='raw' else 0.)
                        stream.write(json.dumps(v,allow_nan=False)+'\n'); stream.flush(); r['rows']+=1
                del raw,raw_lp,y
        r.update(status='complete_attempts',scores_sha256=sha256_file(d/'scores.jsonl'))
    except BaseException as error:r.update(status='failed',error=f'{type(error).__name__}: {error}'); raise
    finally:r.update(finished_utc=utc_now(),stage_wall_seconds=time.monotonic()-started); atomic_json(d/'run.json',r,overwrite=True)


def reduce_rows(values,ids,docs):
    index={(v['item_id'],v['path'],v['mode']):v for v in values}
    if not ids or len(ids)!=len(set(ids)) or len(docs)!=len(set(docs)) or len(ids)!=len(docs) or len(values)!=len(index) or set(index)!={(i,path,mode) for i in ids for path in PATHS for mode in MODES}:raise ValueError('Every document/path/mode exactly once required')
    result={}
    for path in PATHS:
        result[path]={}
        for metric in ('fullspan_ll','numeric_only_ll'):
            absolute={};good={};contrasts={}
            for mode in MODES:
                rows=[index[i,path,mode] for i in ids]
                if any(v['document_id']!=doc or v['status'] not in ('ok','error') for v,doc in zip(rows,docs)):raise ValueError('Document/status mismatch')
                errors=sum(v['status']=='error' for v in rows); unsupported=sum(v['status']=='ok' and v[metric] is None for v in rows)
                if not errors and not unsupported:
                    if any(not math.isfinite(v[metric]) for v in rows):raise ValueError('Nonfinite metric')
                    good[mode]=[dict(item_id=i,document_id=doc,score=v[metric]) for i,doc,v in zip(ids,docs,rows)]
                absolute[mode]=dict(items=len(ids),errors=errors,unsupported=unsupported,mean=sum(v[metric] for v in rows)/len(ids) if mode in good else None)
            for a,b in CONTRASTS:
                contrasts[f'{b}_minus_{a}']=(paired_cluster_bootstrap(good[a],good[b],repeats=5000,seed=17) if a in good and b in good else dict(available=False,items=len(ids),reason='Errors/unsupported retained; no complete-case or zero imputation'))
            result[path][metric]=dict(absolute=absolute,contrasts=contrasts,raw_mean_maximum_possible_gain=-absolute['raw']['mean'] if absolute['raw']['mean'] is not None else None)
    return result


def summarize(output):
    output=Path(output); p,prefixes=load(output); closed=base.closure(output,p,'score'); notes,control=controls(output,p,prefixes)
    rows=reader_rows(p,prefixes); byid={v['item_id']:v for v in rows}; values=[];pins={};reports=[];identity=None
    for shard in range(SHARDS):
        d=output/'scores'/f'shard-{shard}'; r=shared.read(d/'run.json'); vv=shared.rows(d/'scores.jsonl')
        if (r['status']!='complete_attempts' or r['rows']!=N//SHARDS*len(MODES)*len(PATHS) or r['shard']!=shard or r['protocol_sha256']!=sha256_file(output/'protocol.json')
                or r['controls_sha256']!=sha256_file(output/'controls.json') or r['scores_sha256']!=sha256_file(d/'scores.jsonl')
                or any(r['reader'].get(k)!=v for k,v in p['writer'].items()) or r['reader']!=control['writer'] or identity is not None and r['reader']!=identity):raise ValueError('All four unchanged same-model score jobs required')
        if [(v['item_id'],v['mode'],v['path']) for v in vv]!=[(q['item_id'],m,path) for q in rows[shard::SHARDS] for m in MODES for path in PATHS]:raise ValueError('Full original shard order required')
        for v in vv:
            row=byid[v['item_id']]
            if (v['document_id']!=row['document_id'] or v['target_ids']!=row['target_ids'] or v['context_sha256']!=canonical_hash(contexts(row,p,notes,control)[v['mode']])
                    or v['status']=='ok' and any(v[k]!=a for k,a in metrics(v['token_logps'],row).items())):raise ValueError('Actual context/target/vector changed')
        values.extend(vv); reports.append(r); identity=r['reader']; pins.update(shared.pin([d/'run.json',d/'scores.jsonl']))
    counts=dict(attempts=N,empty=sum(v['status']=='ok' and not v['content_ids'] for v in notes.values()),caps=sum(v.get('stop_type')=='length_cap' for v in notes.values()),
        eos=sum(v.get('stop_type')=='token:151643' for v in notes.values()),first_close=sum(v.get('stop_type')=='token:151668' for v in notes.values()),
        generation_errors=sum(v['status']=='error' for v in notes.values()),scoring_errors=sum(v['status']=='error' for v in values),numeric_supported=sum(v['numeric_only_supported'] for v in rows))
    result=dict(schema=SCHEMA,status='complete_all_attempts',finished_utc=utc_now(),items=N,cells=len(values),reader=identity,counts=counts,
        groups=reduce_rows(values,p['ordered_item_ids'],p['ordered_document_ids']),costs=shared.saved_costs(notes,values),score_stages=reports,
        generation_stages=[shared.read(output/'notes'/f'shard-{i}'/'run.json') for i in range(SHARDS)],protocol_sha256=sha256_file(output/'protocol.json'),
        input_hashes={**pins,**control['note_pins'],**closed,**shared.pin([output/'controls.json'])},limitations=p['limitations'])
    shared.verify_pins(result['input_hashes']); atomic_json(output/'summary.json',result); return result


def main():
    a=argparse.ArgumentParser(description=__doc__); a.add_argument('command',choices=('prepare','generate','controls','score','summarize')); a.add_argument('--output',type=Path,required=True)
    for name in ('dataset','native-reference','control-dir','checkpoint','training-protocol'):a.add_argument('--'+name,type=Path)
    a.add_argument('--checkpoint-sha256');a.add_argument('--complete-sha256');a.add_argument('--deadline-utc');a.add_argument('--shard',type=int);a.add_argument('--device',choices=('cpu','xla'),default='xla');v=a.parse_args()
    if v.command=='prepare':prepare(v.output,v.dataset,v.native_reference,v.deadline_utc,v.control_dir,writer_identity(v.checkpoint,v.checkpoint_sha256,v.complete_sha256,v.training_protocol))
    elif v.command=='generate':generate(v.output,v.shard,v.deadline_utc,v.device)
    elif v.command=='controls':p,rows=load(v.output);controls(v.output,p,rows,create=True)
    elif v.command=='score':score(v.output,v.shard,v.deadline_utc,v.device)
    else:print(json.dumps(summarize(v.output),indent=2))


if __name__=='__main__':main()
