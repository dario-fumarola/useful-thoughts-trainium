"""Separate prefix-only I3/few-shot policies; whole notes and fixed first lines.

No real continuation enters either writer API. All512 attempts close before
future scoring. Fewshot+firstline fixedI6 is the sole prospective primary;
I3 and the other three writer/extraction combinations are secondary controls.
"""
from dataclasses import asdict
from pathlib import Path
import argparse, json, math, time
from train import general_note_evaluation as base
from train import useful_note_evaluation as shared
from train.firstline_note_probe import truncate_content
from train.interface_oracle_probe import reader_context
from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file, utc_now

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='general-prefix-helper-held256-v1'
POLICIES=('plain','fewshot');EXTRACTIONS=('full','firstline')
VARIANTS=tuple(f'{p}/{e}' for p in POLICIES for e in EXTRACTIONS)
N,SHARDS,CAP,WIDTH=256,4,64,1024
DEMO_SHA='6060b9cf1740d34abb48d490ef97e90ab3156b7d49cc2fdec75b0d2de51483bf'
INSTRUCTION=('Predict one concise helper note useful for continuing the passage, using only the passage and your knowledge. '
 'Give only one line, without a final-answer field, extra examples, or another Passage.')


def sources():
    return {**base.sources(),**{n:sha256_file(ROOT/n) for n in ('train/general_prefix_helper_probe.py','train/firstline_note_probe.py')}}


def writer_prompt(prefix_ids,parts):
    """Exactly two arguments; no row/future field can enter either policy."""
    past=list(prefix_ids)
    return dict(plain=past+parts['plain_open'],fewshot=parts['fewshot_initial']+past+parts['fewshot_final'])


def prepare(output,base_preparation,declaration,until,control_dir):
    from transformers import AutoTokenizer
    shared.check(until);original=Path(base_preparation).resolve();old,prefixes=base.load(original)
    declaration=Path(declaration).resolve();shared.verify_pins({str(declaration):DEMO_SHA})
    text=shared.read(declaration)['teacher']['fixed_initial_text'];separator='\n\nPassage:\n'
    demos=separator+text.split(separator,1)[1]
    if demos.count('Observed continuation:\n')!=3 or demos.count('Helper note:')!=3 or not demos.endswith(separator):
        raise ValueError('Exact original three synthetic demonstrations required')
    tok=AutoTokenizer.from_pretrained(old['model_dir'],local_files_only=True,trust_remote_code=False)
    enc=lambda v:tok.encode(v,add_special_tokens=False)
    parts=dict(plain_open=old['parts']['old_open'],fewshot_initial=enc(INSTRUCTION+demos),fewshot_final=enc('\nHelper note:'))
    widths={policy:max(len(writer_prompt(r['prefix_ids'],parts)[policy]) for r in prefixes) for policy in POLICIES}
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    with (out/'prefixes.jsonl').open('x') as stream:
        for row in prefixes:stream.write(json.dumps(row)+'\n')
    inherited={k:old[k] for k in ('assets','model_dir','model_manifest_sha256','base_signature','native_reference','native_inputs',
        'parts','vocab_size','banned_ids','ordered_item_ids','ordered_document_ids','reader_inputs_file','reader_inputs_sha256')}
    p=dict(schema=SCHEMA,created_utc=utc_now(),**inherited,source_files=sources(),writer=base.B0,items=N,shards=SHARDS,
        max_actions=CAP,width=WIDTH,precision='fp32',policies=list(POLICIES),extractions=list(EXTRACTIONS),paths=list(base.PATHS),modes=list(base.MODES),
        writer_parts=parts,writer_prefill_widths={k:((v+31)//32)*32 for k,v in widths.items()},actual_prompt_tokens=widths,
        stop_ids=[151668,151643],selection_policy='single_greedy',deadline_utc=until,control_dir=str(Path(control_dir).resolve()),
        bootstrap=dict(repeats=5000,seed=17),prefixes_sha256=sha256_file(out/'prefixes.jsonl'),
        input_hashes={**old['input_hashes'],**shared.pin([original/'protocol.json',original/'prefixes.jsonl',declaration])},
        demo_provenance=dict(declaration_sha256=DEMO_SHA,exact_demo_text_sha256=canonical_hash(demos),query_future_access=False,
            caveat='Original synthetic demonstrations retain their observed-continuation examples; the actual query has ONLY original prefix then Helper note, and the instruction explicitly predicts from past/knowledge.'),
        primary='fewshot/firstline fixedI6=.375 own-minus-raw AND own-minus-intact-other lower95%>0; I3 and other three variants secondary.',
        limitations=['Separate inference policy, no SFT/RL gate change. Same B0-surprisal-conditioned developmental256 reused.',
            'All512 whole generated attempts retained; first-line extraction is token-only and never discounts actual writer cost.',
            'Intact cyclic donor notes preserve each policy/extraction population lengths; per-item lengths differ.',
            'Target/corrupt copies are actual16 from own/next document: reader plumbing, not generated utility.',
            'Prefix-only fewshot versus plain is a new prompt-policy comparison; without a matched future-aware teacher on this held panel it cannot causally partition prior TRAIN teacher gains.'])
    atomic_json(out/'protocol.json',p);return p


def load(output):
    out=Path(output);p=shared.read(out/'protocol.json')
    if (p['schema']!=SCHEMA or p['source_files']!=sources() or p['writer']!=base.B0 or p['items']!=N or p['shards']!=SHARDS
        or p['policies']!=list(POLICIES) or p['extractions']!=list(EXTRACTIONS) or p['max_actions']!=CAP or p['stop_ids']!=[151668,151643]
        or p['paths']!=list(base.PATHS) or p['modes']!=list(base.MODES) or p['width']!=WIDTH or p['precision']!='fp32'
        or p['bootstrap']!=dict(repeats=5000,seed=17) or p['selection_policy']!='single_greedy'):
        raise ValueError('Frozen new helper-policy source/recipe required')
    shared.verify_pins(p['input_hashes']);shared.verify_pins(p['native_inputs']);shared.verify_pins({str(out/'prefixes.jsonl'):p['prefixes_sha256']})
    prefixes=shared.rows(out/'prefixes.jsonl');base.validate_prefixes(prefixes,p['ordered_item_ids'],p['ordered_document_ids'])
    if p['writer_prefill_widths']!={k:((max(len(writer_prompt(r['prefix_ids'],p['writer_parts'])[k]) for r in prefixes)+31)//32)*32 for k in POLICIES}:
        raise ValueError('Actual prefix widths changed')
    return p,prefixes


def generate(output,shard,until,device='xla'):
    import torch
    from train.evaluate import GreedyDecoder,GenerationFailure
    shared.check(until);out=Path(output);p,rows=load(out)
    if type(shard)is not int or shard not in range(SHARDS) or until!=p['deadline_utc']:raise ValueError('Exact shard/deadline required')
    d=out/'notes'/f'shard-{shard}';d.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    report=dict(status='started',rows=0,shard=shard,started_utc=utc_now(),protocol_sha256=sha256_file(out/'protocol.json'));atomic_json(d/'run.json',report)
    try:
        before=time.monotonic();backend,tok,identity=shared.load_model(p,base.B0,device);report.update(writer=identity,model_setup_seconds=time.monotonic()-before)
        before=time.monotonic();report['gate']=shared.base_gate(backend,p,shard,d,identity);report['native_gate_seconds']=time.monotonic()-before
        decoder=GreedyDecoder(backend.model,tok,sync=backend.sync,banned_ids=backend.banned_ids)
        with (d/'notes.jsonl').open('x') as stream:
            for row in rows[shard::SHARDS]:
                prompts=writer_prompt(row['prefix_ids'],p['writer_parts'])
                for policy in POLICIES:
                    shared.check(until);before=time.monotonic();prompt=prompts[policy]
                    v=dict(item_id=row['item_id'],document_id=row['document_id'],policy=policy,status='error',actions=[],content_ids=[],
                        prefix_sha256=canonical_hash(row['prefix_ids']),writer_prompt_sha256=canonical_hash(prompt))
                    try:
                        with torch.no_grad():value=decoder.decode(prompt,max_new_tokens=CAP,prefill_width=p['writer_prefill_widths'][policy],pad_id=151643,stop_ids=(151668,151643),answer_line_stop=False)
                        v.update(status='ok',**asdict(value))
                    except Exception as error:
                        if isinstance(error,GenerationFailure):v.update(**asdict(error.partial))
                        v['error']=f'{type(error).__name__}: {error}'
                    v.update(wall_seconds=time.monotonic()-before,n_actions=len(v['actions']),n_content_tokens=len(v['content_ids']))
                    try:v['extraction']=truncate_content(v['content_ids'],tok)
                    except Exception as error:v.update(status='error',extraction=None,extraction_error=f'{type(error).__name__}: {error}')
                    stream.write(json.dumps(v,allow_nan=False)+'\n');stream.flush();report['rows']+=1
        report.update(status='complete_attempts',notes_sha256=sha256_file(d/'notes.jsonl'))
    except BaseException as error:report.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:report.update(finished_utc=utc_now(),stage_wall_seconds=time.monotonic()-started);atomic_json(d/'run.json',report,overwrite=True)


def validate_note(v,row,p,tok):
    a=v['actions'];stop=a[-1] if a and a[-1] in (151668,151643) else None;content=a[:-1] if stop is not None else a
    if (v['policy'] not in POLICIES or v['status'] not in ('ok','error') or v['item_id']!=row['item_id'] or v['document_id']!=row['document_id']
        or v['prefix_sha256']!=canonical_hash(row['prefix_ids']) or v['writer_prompt_sha256']!=canonical_hash(writer_prompt(row['prefix_ids'],p['writer_parts'])[v['policy']])
        or len(a)>CAP or any(type(t)is not int or not 0<=t<p['vocab_size'] or t in p['banned_ids'] for t in a)
        or v['content_ids']!=content or v.get('raw_text','')!=tok.decode(content,skip_special_tokens=False,clean_up_tokenization_spaces=False) or any(t in (151668,151643) for t in content) or v['n_actions']!=len(a) or v['n_content_tokens']!=len(content)
        or (v['extraction']!=truncate_content(content,tok) if v['extraction'] is not None else v['status']!='error' or not v.get('extraction_error')) or not math.isfinite(v['wall_seconds']) or v['wall_seconds']<0
        or v['status']=='ok' and (not a or (v['stop_type']!=f'token:{stop}' if stop is not None else v['stop_type']!='length_cap' or len(a)!=CAP))):
        raise ValueError('Original native actions/stop/prompt/first-line contract changed')


def complete_notes(output,p,rows):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(p['model_dir'],local_files_only=True,trust_remote_code=False);notes={};pins={};identity=None
    for shard in range(SHARDS):
        d=Path(output)/'notes'/f'shard-{shard}';r=shared.read(d/'run.json');values=shared.rows(d/'notes.jsonl');expected=[(q,policy) for q in rows[shard::SHARDS] for policy in POLICIES]
        if (r['status']!='complete_attempts' or r['rows']!=N//SHARDS*2 or r['protocol_sha256']!=sha256_file(Path(output)/'protocol.json')
            or r['notes_sha256']!=sha256_file(d/'notes.jsonl') or any(r['writer'].get(k)!=v for k,v in base.B0.items())
            or identity is not None and identity!=r['writer'] or [(v['item_id'],v['policy']) for v in values]!=[(q['item_id'],policy) for q,policy in expected]):
            raise ValueError('All512 original attempts in exact shard order required')
        for v,(q,policy) in zip(values,expected):validate_note(v,q,p,tok);notes[f'{policy}/{q["item_id"]}']=v
        identity=r['writer'];pins.update(shared.pin([d/'run.json',d/'notes.jsonl']))
    return notes,pins,identity


def note_variants(p,notes):
    return {variant:{key:dict(notes[f'{policy}/{key}'],content_ids=list(notes[f'{policy}/{key}']['content_ids'] if extraction=='full' else (notes[f'{policy}/{key}']['extraction'] or {'content_ids':[]})['content_ids']))
        for key in p['ordered_item_ids']} for variant in VARIANTS for policy,extraction in [variant.split('/')]}


def controls(output,p,rows,create=False):
    closed=base.closure(output,p,'generate');notes,pins,writer=complete_notes(output,p,rows);variants=note_variants(p,notes)
    value=dict(status='complete',protocol_sha256=sha256_file(Path(output)/'protocol.json'),generation_closure=closed,note_pins=pins,writer=writer,
        variants={key:shared.other_controls(p,v) for key,v in variants.items()})
    file=Path(output)/'controls.json'
    if create:atomic_json(file,value)
    elif shared.read(file)!=value:raise ValueError('Intact policy/extraction global controls changed')
    return notes,variants,value


def score(output,shard,until,device='xla'):
    import torch
    from train.interface_mixer_probe import full_logits,mixed_token_logps,token_logps
    shared.check(until);out=Path(output);p,prefixes=load(out)
    if type(shard)is not int or shard not in range(SHARDS) or until!=p['deadline_utc']:raise ValueError('Exact shard/deadline required')
    notes,variants,control=controls(out,p,prefixes);data=base.reader_rows(p,prefixes);byid={r['item_id']:r for r in data}
    d=out/'scores'/f'shard-{shard}';d.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    report=dict(status='started',rows=0,shard=shard,started_utc=utc_now(),protocol_sha256=sha256_file(out/'protocol.json'),controls_sha256=sha256_file(out/'controls.json'));atomic_json(d/'run.json',report)
    try:
        before=time.monotonic();backend,_,actual=shared.load_model(p,base.B0,device);report.update(reader=actual,model_setup_seconds=time.monotonic()-before)
        before=time.monotonic();report['gate']=shared.base_gate(backend,p,shard,d,actual);report['native_gate_seconds']=time.monotonic()-before
        with torch.no_grad(),(d/'scores.jsonl').open('x') as stream:
            for row in data[shard::SHARDS]:
                key=row['item_id'];cache={};before=time.monotonic();raw=raw_lp=y=None;raw_error=None
                try:
                    raw=full_logits(backend,row['prefix_ids'],row['natural_future16_ids']);y=torch.tensor([row['natural_future16_ids']],device=backend.device)
                    raw_lp=token_logps(raw,y);backend.sync();raw_values=raw_lp.cpu()[0].tolist();base.likelihood(raw_values)
                except Exception as error:raw_error=f'{type(error).__name__}: {error}'
                raw_seconds=time.monotonic()-before
                for variant in VARIANTS:
                    ctx=base.contexts(row,p,variants[variant],control['variants'][variant],byid)
                    for mode in base.MODES:
                        shared.check(until);digest=canonical_hash(ctx[mode]);timing=f'{key}/{digest}';before=time.monotonic()
                        cell=dict(item_id=key,document_id=row['document_id'],variant=variant,mode=mode,target_ids=row['natural_future16_ids'],context_sha256=digest)
                        try:
                            if raw_error:raise RuntimeError(raw_error)
                            donor=control['variants'][variant]['other_donors'][key]
                            if mode=='own' and variants[variant][key]['status']!='ok' or mode=='other' and variants[variant][donor]['status']!='ok':
                                raise ValueError('Failed genuine note/donor retained; no partial substitution')
                            if digest not in cache:
                                try:
                                    if mode=='raw':vectors=[raw_values,raw_values]
                                    else:
                                        note=full_logits(backend,ctx[mode],row['natural_future16_ids']);mixed,_,end=mixed_token_logps(raw,note,y,[.375],raw_lp)
                                        backend.sync();vectors=[end.cpu()[0].tolist(),mixed.cpu()[0].tolist()]
                                    for lp in vectors:base.likelihood(lp)
                                    cache[digest]=(vectors,time.monotonic()-before+(raw_seconds if mode=='raw' else 0.),None)
                                except Exception as error:cache[digest]=(None,time.monotonic()-before,f'{type(error).__name__}: {error}')
                            vectors,elapsed,failed=cache[digest]
                            if failed:raise RuntimeError(failed)
                            emitted=[dict(cell,path=path,status='ok',token_logps=lp,natural16_ll=base.likelihood(lp)) for path,lp in zip(base.PATHS,vectors)]
                        except Exception as error:
                            if digest in cache:_,elapsed,_=cache[digest]
                            elif raw_error and mode=='raw':elapsed=raw_seconds;timing=f'{key}/raw-error'
                            else:elapsed=time.monotonic()-before;timing=f'{variant}/{key}/{mode}/error'
                            emitted=[dict(cell,path=path,status='error',error=f'{type(error).__name__}: {error}') for path in base.PATHS]
                        for v in emitted:
                            v.update(wall_seconds=elapsed,timing_group=timing,shared_raw_forward_seconds=raw_seconds if mode=='raw' else 0.,shared_by_paths=list(base.PATHS),shared_context_reuse='Identical actual contexts cached within item across policies/extractions; elapsed counted once per timing_group.')
                            stream.write(json.dumps(v,allow_nan=False)+'\n');stream.flush();report['rows']+=1
                del raw,raw_lp,y
        report.update(status='complete_attempts',scores_sha256=sha256_file(d/'scores.jsonl'))
    except BaseException as error:report.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:report.update(finished_utc=utc_now(),stage_wall_seconds=time.monotonic()-started);atomic_json(d/'run.json',report,overwrite=True)


def reduce_rows(values,ids,docs):
    if len(values)!=len(ids)*len(VARIANTS)*len(base.MODES)*len(base.PATHS) or any(v['variant'] not in VARIANTS for v in values):
        raise ValueError('All four complete writer/extraction variants required')
    groups={variant:base.reduce_rows([v for v in values if v['variant']==variant],ids,docs) for variant in VARIANTS}
    return dict(variants=groups,primary=dict(variant='fewshot/firstline',path='I6w0.375',
        passed=groups['fewshot/firstline']['I6w0.375']['gates']['useful_content'],
        definition='Both paired document lower95% own-minus-raw and own-minus-intact-other >0; other conditions secondary.'))


def summarize(output):
    out=Path(output);p,prefixes=load(out);closed=base.closure(out,p,'score');notes,variants,control=controls(out,p,prefixes)
    data=base.reader_rows(p,prefixes);byid={r['item_id']:r for r in data};values=[];pins={};reports=[];identity=None
    for shard in range(SHARDS):
        d=out/'scores'/f'shard-{shard}';r=shared.read(d/'run.json');vv=shared.rows(d/'scores.jsonl')
        if (r['status']!='complete_attempts' or r['rows']!=N//SHARDS*len(VARIANTS)*len(base.MODES)*len(base.PATHS)
            or r['protocol_sha256']!=sha256_file(out/'protocol.json') or r['controls_sha256']!=sha256_file(out/'controls.json')
            or r['scores_sha256']!=sha256_file(d/'scores.jsonl') or any(r['reader'].get(k)!=v for k,v in base.B0.items())
            or identity is not None and identity!=r['reader']):raise ValueError('All complete unchanged B0 shards required')
        expected=[(q['item_id'],variant,mode,path) for q in data[shard::SHARDS] for variant in VARIANTS for mode in base.MODES for path in base.PATHS]
        if [(v['item_id'],v['variant'],v['mode'],v['path']) for v in vv]!=expected:raise ValueError('Exact full score order required')
        for v in vv:
            row=byid[v['item_id']];variant=v['variant'];ctx=base.contexts(row,p,variants[variant],control['variants'][variant],byid)[v['mode']]
            if (v['document_id']!=row['document_id'] or v['target_ids']!=row['natural_future16_ids'] or v['context_sha256']!=canonical_hash(ctx)
                or v['status']=='ok' and v['natural16_ll']!=base.likelihood(v['token_logps'])):raise ValueError('Context/target/vector changed')
        values.extend(vv);identity=r['reader'];reports.append(r);pins.update(shared.pin([d/'run.json',d/'scores.jsonl']))
    counts={policy:dict(attempts=N,errors=sum(v['status']=='error' for v in notes.values() if v['policy']==policy),
        first_close=sum(v.get('stop_type')=='token:151668' for v in notes.values() if v['policy']==policy),
        eos=sum(v.get('stop_type')=='token:151643' for v in notes.values() if v['policy']==policy),
        note_generation_seconds_sum=sum(v['wall_seconds'] for v in notes.values() if v['policy']==policy),
        caps=sum(v.get('stop_type')=='length_cap' for v in notes.values() if v['policy']==policy),
        empty=sum(v['status']=='ok' and not v['content_ids'] for v in notes.values() if v['policy']==policy),
        total_actions=sum(v['n_actions'] for v in notes.values() if v['policy']==policy),
        kept_firstline_tokens=sum(len((v['extraction'] or {'content_ids':[]})['content_ids']) for v in notes.values() if v['policy']==policy)) for policy in POLICIES}
    result=dict(schema=SCHEMA,status='complete_all_attempts',finished_utc=utc_now(),items=N,attempts=512,cells=len(values),reader=identity,
        groups=reduce_rows(values,p['ordered_item_ids'],p['ordered_document_ids']),counts=counts,costs=shared.saved_costs(notes,values),
        generation_stages=[shared.read(out/'notes'/f'shard-{s}'/'run.json') for s in range(SHARDS)],score_stages=reports,
        protocol_sha256=sha256_file(out/'protocol.json'),input_hashes={**pins,**control['note_pins'],**closed,**shared.pin([out/'controls.json'])},
        bootstrap='5000 paired document draws seed17; conditional on checkpoint/panel.',limitations=p['limitations'])
    shared.verify_pins(result['input_hashes']);atomic_json(out/'summary.json',result);return result


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('command',choices=('prepare','generate','controls','score','summarize'));a.add_argument('--output',type=Path,required=True)
    for name in ('base-preparation','declaration','control-dir'):a.add_argument('--'+name,type=Path)
    a.add_argument('--deadline-utc');a.add_argument('--shard',type=int);a.add_argument('--device',choices=('cpu','xla'),default='xla');v=a.parse_args()
    if v.command=='prepare':prepare(v.output,v.base_preparation,v.declaration,v.deadline_utc,v.control_dir)
    elif v.command=='generate':generate(v.output,v.shard,v.deadline_utc,v.device)
    elif v.command=='controls':p,rows=load(v.output);controls(v.output,p,rows,create=True)
    elif v.command=='score':score(v.output,v.shard,v.deadline_utc,v.device)
    else:print(json.dumps(summarize(v.output),indent=2))


if __name__=='__main__':main()
