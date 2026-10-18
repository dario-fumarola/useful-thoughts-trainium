"""Frozen-B0 DEV target-copy plumbing oracle; no writer policy or training.

One raw score/document is reused across six interfaces. Notes deliberately see
future targets. Full-span log likelihood is primary; K1 is secondary only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
import json
import math
from pathlib import Path
import random
import time
from types import SimpleNamespace

import torch

from train.evaluate import asset_provenance
from train.evaluate_likelihood import score_continuations
from train.evaluate_local import load_selected, read_selection, setup_model
from train.evaluation_integrity import evaluation_dependencies
from weekend_thinking.artifacts import atomic_json, canonical_hash, require_utc_deadline, sha256_file, utc_now
from weekend_thinking.pretrained import load_base, sampled_fingerprint
from weekend_thinking.stats import paired_cluster_bootstrap

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='phase1-interface-target-copy-oracle-v1'
COHORTS=('general','arithmetic'); INTERFACES=tuple(f'I{i}' for i in range(6))
MODES=('empty','correct','corrupt','shuffled','other')
N=256; WIDTH=1024; FUTURE=16; PAD=151643; SHARDS=4
MODEL='Qwen/Qwen3-1.7B-Base'; REVISION='ea980cb0a6c2ae4b936e82123acc929f1cec04c1'
THRESHOLDS={'empty_minus_raw':-.5,'correct_minus_raw':2.,'correct_minus_corrupt':2.}


def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def decode(tok,ids):return tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
def deadline(value):
    if datetime.now(timezone.utc)>=require_utc_deadline(value):raise RuntimeError('Finite oracle deadline reached; no automatic retry')
def sources():
    return {**evaluation_dependencies(), **{name:sha256_file(ROOT/name) for name in
        ('train/interface_oracle_probe.py','train/computable_equations.py','train/computable_equations_supervised.py','train/calculation_boundaries.py','train/scalar_completion.py')}}
def conditions():return [('raw','raw')]+[(i,m) for i in INTERFACES for m in MODES]


def prepare_general(output,selection_path,records,model_dir,until):
    """First256 of existing ordered DEV512; no model or surprisal selection."""
    from transformers import AutoTokenizer
    deadline(until); selection=read_selection(selection_path,qualification=False,split='dev')
    if selection['count']!=512:raise ValueError('Original fixed DEV512 required')
    selected=dict(selection,item_ids=selection['item_ids'][:N],document_ids=selection['document_ids'][:N],count=N)
    selected['item_ids_sha256']=canonical_hash(selected['item_ids'])
    actual=load_selected(records,selected,include_future=True)
    tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
    defined=set(tok.get_vocab().values()); result=[]
    for r in actual:
        if len(r['prefix'])!=512 or len(r['future'])!=16 or any(t not in defined for t in r['prefix']+r['future']):
            raise ValueError('Complete native512+16 DEV record required')
        result.append(dict(cohort='general',item_id=r['item_id'],document_id=r['document_id'],split='dev',
            prefix_ids=r['prefix'],target_ids=r['future'],copy_note_ids=r['future'],
            reanchor_ids=r['prefix'][-24:],primary_mask=[True]*16))
    deadline(until);output.mkdir(parents=True,exist_ok=False)
    (output/'prepare_source.py').write_bytes(Path(__file__).read_bytes())
    atomic_json(output/'inputs.json',dict(rows=result,selection=selected))
    atomic_json(output/'manifest.json',dict(status='complete_CPU_cohort_only',count=N,
        selection_rule='First256 existing fixed DEV512 IDs; no outcome-dependent exclusions',
        inputs_sha256=sha256_file(output/'inputs.json'),tokenizer_vocab_sha256=canonical_hash(tok.get_vocab()),
        input_hashes={str(p.resolve()):sha256_file(p) for p in (selection_path,records,output/'prepare_source.py')},created_utc=utc_now()))


def serialization(tok,fmt,published):
    enc=lambda s:tok.encode(s,add_special_tokens=False)
    if (decode(tok,fmt['header_ids'])!=fmt['header_text'] or decode(tok,fmt['open_ids'])!='\n<think>'
            or decode(tok,fmt['close_ids'])!='</think>\n'):
        raise ValueError('Exact old format tokens/text required')
    text=published['prompt_text']
    if (text!=published['transcription'][1:-1] or text.count('</think>')!=3
            or '<think>' in text or text[:1] in ('”','“','"')):
        raise ValueError('Exact published closing/closing typo; decorative quotes excluded')
    return dict(old_header=fmt['header_ids'],old_open=fmt['open_ids'],old_close=fmt['close_ids'],
        inline_open=enc(' <think>'),inline_close=enc('</think> '),
        published_header=enc(text+'\n\n'),scratch_open=enc('\n(scratch: '),scratch_close=enc(')\n'))


def reader_context(row,note,interface,parts):
    past=list(row['prefix_ids']); note=list(note); anchor=list(row['reanchor_ids'])
    if interface=='raw':return past
    if interface=='I0':return parts['old_header']+past+parts['old_open']+note+parts['old_close']
    if interface in ('I1','I3'):
        return past+parts['old_open']+note+parts['old_close']+(anchor if interface=='I3' else [])
    if interface=='I2':return past+parts['inline_open']+note+parts['inline_close']
    if interface=='I4':return parts['published_header']+past+parts['old_open']+note+parts['old_close']
    if interface=='I5':return past+parts['scratch_open']+note+parts['scratch_close']+anchor
    raise ValueError('Unknown fixed interface')


def make_controls(selected):
    """Fixed global donors within cohort, before any scoring/sharding."""
    controls={}
    for cohort in COHORTS:
        group=[r for r in selected if r['cohort']==cohort]
        if len(group)!=N or len({r['document_id'] for r in group})!=N:raise ValueError('Complete unique256/cohort required')
        for index,r in enumerate(group):
            note=list(r['copy_note_ids']); donor=group[(index+1)%N]; other=group[(index+129)%N]
            seed=int(canonical_hash(dict(seed=17,cohort=cohort,item_id=r['item_id']))[:15],16)
            shuffled=list(note);random.Random(seed).shuffle(shuffled)
            corrupt=list(donor['target_ids'] if cohort=='general' else r['corrupt_note_ids'])
            content=list(other['copy_note_ids']); matched=(content*((len(note)+len(content)-1)//len(content)))[:len(note)]
            controls[f'{cohort}:{r["item_id"]}']=dict(values=dict(empty=[],correct=note,corrupt=corrupt,shuffled=shuffled,other=matched),
                corrupt_donor_item_id=donor['item_id'] if cohort=='general' else None,
                other_donor_item_id=other['item_id'],shuffle_seed=seed,
                shuffled_equals_correct=shuffled==note,corrupt_equals_correct=corrupt==note,other_equals_correct=matched==note)
    return controls


def load_arithmetic(directory,tok):
    """Adapt the independently published relaxed curator; no RHS filtering here."""
    from train import computable_equations as curator
    from train import computable_equations_supervised as supervised
    m=read(directory/'manifest.json'); accepted=rows(directory/'accepted.jsonl')
    if m.get('schema')==curator.SCHEMA:
        selector=curator;status='complete_CPU_preparation_not_model_or_training_qualification'
    elif m.get('schema')==supervised.SCHEMA:
        selector=supervised;status='complete_supervised_CPU_preparation_not_model_or_training_qualification'
    else:raise ValueError('Only explicitly reviewed causal-v1 or supervised-v2 publication supported')
    if (m.get('selection_salt')!=selector.SALT
            or m.get('source_sha256')!=curator.SOURCE_SHA or m.get('source_files')!=selector.sources()
            or type(m.get('ready_for_requested_pools')) is not bool
            or m.get('status')!=status or m.get('tokenizer_vocab_sha256')!=canonical_hash(tok.get_vocab())
            or m.get('accepted_sha256')!=sha256_file(directory/'accepted.jsonl')):
        raise ValueError('Complete exact native arithmetic publication required')
    if m['selection']!=selector.select_pools(accepted):raise ValueError('Frozen arithmetic hash pools changed')
    ids=m['selection']['dev']['selected_item_ids']
    if not N<=len(ids)<=512 or len(set(ids))!=len(ids):raise ValueError('At least the complete fixed Phase1 DEV256 required')
    if not m['ready_for_requested_pools'] and selector is not supervised:raise ValueError('Only complete supervised-v2 supports a reported optional-pool shortfall')
    index={r['item_id']:r for r in accepted}; selected=[]; defined=set(tok.get_vocab().values())
    if len(index)!=len(accepted) or len({r['document_id'] for r in accepted})!=len(accepted):raise ValueError('Unique arithmetic documents required')
    if m['phase1_dev_item_ids']!=ids[:N] or m['held_post_sft_dev_item_ids']!=ids[N:512]:raise ValueError('Fixed prospective DEV halves required')
    for key in ids[:N]:
        r=index[key]
        target=curator.source_target(dict(cut=len(r['prefix_ids']),prefix_text=r['prefix_text']),r['original_source_ids'],tok)
        bad=curator.corrupt_surface(target,tok)
        if (str(curator.safe.parse_numeric(r['boundary']['expression']).value)!=r['numeric_value']
                or r['boundary']['prefix_ids']!=r['prefix_ids'] or r['boundary']['reanchor_text']!=r['reanchor_text']
                or target!=r['verified_numeric_rhs'] or target['value']!=r['numeric_value']
                or r['copy_note_ids']!=r['target_ids'] or r['correct_note_ids']!=r['target_ids']
                or r['correct_note_text']!=decode(tok,r['target_ids']) or r['target_ids']!=target['token_ids']
                or r['corrupt_note_ids']!=bad['ids'] or r['corrupt_numeric_value']!=bad['value']
                or r['source_sha256']!=m['source_sha256'] or Path(r['source_file']).resolve()!=Path(m['source_file']).resolve()
                or len(r['original_source_ids'])!=528 or r['original_source_ids'][:len(r['prefix_ids'])]!=r['prefix_ids']
                or r['natural_future16_ids']!=r['original_source_ids'][len(r['prefix_ids']):len(r['prefix_ids'])+16]
                or r['target_ids']!=r['natural_future16_ids'][:len(r['target_ids'])]
                or len(r['natural_future16_ids'])!=16 or r['split']!='dev' or r['numeric_value']==r['corrupt_numeric_value']
                or Fraction(r['numeric_value'])==Fraction(r['corrupt_numeric_value'])
                or not r['copy_note_ids'] or not r['corrupt_note_ids'] or not r['reanchor_ids']
                or not 1<=len(r['target_ids'])<=FUTURE
                or r['numeric_token_mask']!=target['numeric_token_mask']
                or len(r['numeric_token_mask'])!=len(r['target_ids']) or not any(r['numeric_token_mask'])
                or r['closing_boundary_ids']!=target['closing_boundary_ids']
                or r['closing_boundary_token_start']!=target['closing_boundary_token_start']
                or not r['closing_boundary_ids'] or r['target_ids'][-len(r['closing_boundary_ids']):]!=r['closing_boundary_ids']
                or decode(tok,r['prefix_ids'])!=r['prefix_text'] or not r['prefix_text'].endswith(r['reanchor_text'])
                or decode(tok,r['reanchor_ids'])!=r['reanchor_text']
                or decode(tok,r['copy_note_ids'])!=r['copy_note_text']
                or any(type(t)is not int or t not in defined for t in r['prefix_ids']+r['target_ids']+r['copy_note_ids']+r['corrupt_note_ids']+r['reanchor_ids'])):
            raise ValueError('Invalid complete numeric-span/boundary/copy/corruption contract')
        selected.append(dict(r,cohort='arithmetic',primary_mask=[True]*len(r['target_ids'])))
    return m,selected


def prepare(output,general,arithmetic,model_dir,format_path,published_path,until,
            exclusions_path=ROOT/'local/diagnostics/postsft-arithmetic-held-dev-exclusions-v1.json'):
    from transformers import AutoTokenizer
    deadline(until); assets=asset_provenance(model_dir)
    if (assets['model_id'],assets['revision'])!=(MODEL,REVISION):raise ValueError('Released B0 required')
    tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
    gm=read(general/'manifest.json')
    if gm['count']!=N or gm['inputs_sha256']!=sha256_file(general/'inputs.json') or gm['tokenizer_vocab_sha256']!=canonical_hash(tok.get_vocab()):
        raise ValueError('Fixed general cohort changed')
    for f,h in gm['input_hashes'].items():
        if sha256_file(f)!=h:raise ValueError('General source selection changed')
    am,math_rows=load_arithmetic(arithmetic,tok); selected=read(general/'inputs.json')['rows']+math_rows
    gd={r['document_id'] for r in selected if r['cohort']=='general'};ad={r['document_id'] for r in math_rows}
    accepted=rows(arithmetic/'accepted.jsonl');ai={r['item_id']:r for r in accepted}
    dev_ids=am['selection']['dev']['selected_item_ids']+am['selection']['dev']['reserve_item_ids']
    reserved=set(dev_ids[N:2*N]);rd={ai[key]['document_id'] for key in reserved}
    exclusions=read(exclusions_path)
    if exclusions['count']!=len(set(exclusions['exclusions'])):raise ValueError('Bound holdout exclusions required')
    post_docs={ai[key]['document_id'] for key in dev_ids[N:]}
    fresh=post_docs-set(exclusions['exclusions'])-ad
    coverage=dict(curator_dev_requested=am['selection']['dev']['requested'],curator_dev_available=len(dev_ids),
        curator_ready_for_requested_pools=am['ready_for_requested_pools'],phase1_documents=N,phase1_ready=True,
        actual_post_phase1_remainder=len(post_docs),fresh_after_sealed_exclusions=len(fresh),
        required_future_postsft_documents=256,future_postsft_shortfall=max(0,256-len(fresh)),
        future_postsft_ready=len(fresh)>=256,training_or_future_held_release=False)
    overlap=dict(general_with_phase1_arithmetic=len(gd&ad),general_with_raw_next256_arithmetic=len(gd&rd),
        raw_next256_not_claimed_unexposed=True,later_postsft='Separate sealed exclusions and inventory; current pools unchanged')
    fmt=read(format_path);parts=serialization(tok,dict(fmt['format'],header_text=fmt['header_text']),read(published_path))
    controls=make_controls(selected); contexts={}
    for r in selected:
        key=f'{r["cohort"]}:{r["item_id"]}'; values=controls[key]['values']; context_meta={}
        for interface,mode in conditions():
            c=reader_context(r,[] if interface=='raw' else values[mode],interface,parts)
            if not c or len(c)+FUTURE>WIDTH:raise ValueError('Overwidth oracle context; no truncation')
            context_meta[f'{interface}/{mode}']=dict(tokens=len(c),sha256=canonical_hash(c))
        contexts[key]=context_meta
    refs=[]
    for cohort,index,interface,mode in (('general',0,'raw','raw'),('general',1,'I4','correct'),('arithmetic',0,'raw','raw'),('arithmetic',1,'I3','correct')):
        r=[r for r in selected if r['cohort']==cohort][index];key=f'{cohort}:{r["item_id"]}'
        refs.append(dict(cohort=cohort,item_id=r['item_id'],interface=interface,mode=mode,
            context_ids=reader_context(r,[] if interface=='raw' else controls[key]['values'][mode],interface,parts)))
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'inputs.json',dict(rows=selected,controls=controls,context_hashes=contexts,parts=parts,qualification_refs=refs))
    input_files=(general/'inputs.json',general/'manifest.json',arithmetic/'manifest.json',arithmetic/'accepted.jsonl',arithmetic/'audit.jsonl',format_path,published_path,exclusions_path)
    if sha256_file(arithmetic/'audit.jsonl')!=am['audit_sha256']:raise ValueError('Arithmetic complete audit changed')
    p=dict(schema=SCHEMA,identity='phase1-interface-oracle-v1',created_utc=utc_now(),deadline_utc=until,
        source_files=sources(),input_hashes={str(f.resolve()):sha256_file(f) for f in input_files},
        inputs_sha256=sha256_file(output/'inputs.json'),model_dir=str(model_dir.resolve()),assets=assets,
        counts={c:N for c in COHORTS},interfaces=list(INTERFACES),modes=list(MODES),shards=SHARDS,
        width=WIDTH,future_width=FUTURE,pad_id=PAD,vocab_size=read(model_dir/'config.json')['vocab_size'],
        expected_rows=2*N*len(conditions()),expected_raw_rows=2*N,precision='fp32',forward_atol=.002,
        thresholds=THRESHOLDS,primary='Full16 general; complete original numeral AND closing-boundary arithmetic joint token log likelihood',
        secondary='K1 and arithmetic numeric-only LL; never substitute for full span',
        control_semantics='GENERAL correct/corrupt = actual future16 copy versus different-document real future16, not isolated numeric-error sensitivity; other uses a distinct donor. Arithmetic corrupt = native different-value numeral.',
        selection='Fixed first256 DEV of each frozen512 pool; no outcome exclusion; raw arithmetic next256 not claimed unexposed',
        arithmetic_selection_policy=am['boundary_selection'],cohort_document_overlap=overlap,arithmetic_publication_coverage=coverage,
        qualification='Four native full-head next-logp refs: general raw/I4correct, arithmetic raw/I3correct',
        serialization='Native array concatenation; original prefix/target IDs never jointly retokenized. I2 added inline spaces retained. I4 published text/typo, not claimed author implementation.',
        selection_rule='Per cohort mean thresholds inclusive, then greatest correct-minus-raw fullspan gain; exact ties lowest interface ID. Report cohort agreement; no posthoc subset.',
        bootstrap=dict(unit='document',repeats=5000,seed=17),
        limitations=['Deliberate future-visible target-copy plumbing oracle, never useful generated-note policy or training benefit.',
            'Known DEV cohorts; six-interface selection is exploratory, not independent confirmation.',
            'Every full-span gain is bounded by minus raw LL; +2 is a COHORT MEAN condition, not a per-item demand.',
            'Arithmetic closing boundary is included in primary; numeric-only score remains separate.',
            'Raw measured once/document; every interface contrast reuses that same actual vector.'])
    atomic_json(output/'protocol.json',p);return p


def load(output):
    p=read(output/'protocol.json');x=read(output/'inputs.json')
    if (p['schema']!=SCHEMA or p['source_files']!=sources() or p['counts']!={c:N for c in COHORTS}
            or p['thresholds']!=THRESHOLDS or p['interfaces']!=list(INTERFACES) or p['modes']!=list(MODES)
            or p['inputs_sha256']!=sha256_file(output/'inputs.json') or p['precision']!='fp32'
            or (p['width'],p['future_width'],p['shards'],p['expected_rows'])!=(WIDTH,FUTURE,SHARDS,2*N*31)):
        raise ValueError('Frozen batch/source/recipe changed')
    for f,h in [*p['input_hashes'].items(),(Path(p['model_dir'])/'ASSET_MANIFEST.json',p['assets']['manifest_sha256'])]:
        if sha256_file(f)!=h:raise ValueError('Bound input/model bytes changed')
    if (len(x['rows'])!=2*N or len({(r['cohort'],r['item_id']) for r in x['rows']})!=2*N
            or any(sum(r['cohort']==c for r in x['rows'])!=N for c in COHORTS)):
        raise ValueError('Complete frozen cohort inventories required')
    return p,x


@torch.no_grad()
def cpu_reference(output,threads,until):
    if type(threads)is not int or not 1<=threads<=16:raise ValueError('CPU threads1..16 required')
    deadline(until);p,x=load(output)
    if until!=p['deadline_utc'] or asset_provenance(Path(p['model_dir']))!=p['assets']:raise ValueError('Frozen deadline/assets required')
    torch.set_num_threads(threads);start=time.monotonic()
    model,_=load_base(p['model_dir']);model.eval()
    if any(t.dtype!=torch.float32 for t in model.parameters()) or model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
        raise ValueError('Tied native FP32 B0 required')
    setup=time.monotonic()-start;values=[];times=[]
    for r in x['qualification_refs']:
        deadline(until);start=time.monotonic();h=model.model(input_ids=torch.tensor([r['context_ids']]),use_cache=False).last_hidden_state
        lp=model.lm_head(h[:,-1]).float().log_softmax(-1).cpu()
        if not torch.isfinite(lp).all():raise ValueError('Nonfinite native reference')
        values.append(lp);times.append(time.monotonic()-start)
    d=output/'cpu-reference';d.mkdir(exist_ok=False);v=torch.cat(values);torch.save(v,d/'next-logps.pt')
    atomic_json(d/'reference.json',dict(status='complete',protocol_sha256=sha256_file(output/'protocol.json'),
        signature=sampled_fingerprint(model),refs=x['qualification_refs'],tied=True,dtype='float32',shape=list(v.shape),
        logps_sha256=sha256_file(d/'next-logps.pt'),model_setup_seconds=setup,forward_seconds=times,finished_utc=utc_now()))


def reference(output,p,x):
    ref=read(output/'cpu-reference/reference.json');v=torch.load(output/'cpu-reference/next-logps.pt',weights_only=True,map_location='cpu')
    if (ref['status']!='complete' or ref['protocol_sha256']!=sha256_file(output/'protocol.json') or ref['refs']!=x['qualification_refs']
            or not ref['tied'] or ref['dtype']!='float32' or v.dtype!=torch.float32 or not torch.isfinite(v).all()
            or list(v.shape)!=ref['shape'] or list(v.shape)!=[SHARDS,p['vocab_size']] or ref['signature']['parameters']!=1720574976
            or ref['logps_sha256']!=sha256_file(output/'cpu-reference/next-logps.pt')):
        raise ValueError('Exact complete native reference required')
    return ref,v


@torch.no_grad()
def score(output,shard,until):
    if type(shard)is not int or shard not in range(SHARDS):raise ValueError('Four fixed shards required')
    deadline(until);p,x=load(output)
    if until!=p['deadline_utc'] or asset_provenance(Path(p['model_dir']))!=p['assets']:raise ValueError('Frozen deadline/assets required')
    ref,native=reference(output,p,x)
    d=output/'scores'/f'shard-{shard}';d.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',rows=0,shard=shard,protocol_sha256=sha256_file(output/'protocol.json'),started_utc=utc_now())
    atomic_json(d/'run.json',report)
    try:
        start=time.monotonic();backend,_,checkpoint,signature,_=setup_model(SimpleNamespace(model_dir=Path(p['model_dir']),checkpoint=None,device='xla'))
        report.update(model_setup_seconds=time.monotonic()-start,signature=signature)
        if checkpoint is not None or signature!=ref['signature'] or any(t.dtype!=torch.float32 for t in backend.model.parameters()) or backend.model.get_input_embeddings().weight is not backend.model.get_output_embeddings().weight:
            raise ValueError('Actual unchanged tied B0 differs from native reference')
        native_start=time.monotonic();q=x['qualification_refs'][shard];ids=torch.full((1,WIDTH),PAD,dtype=torch.long);ids[0,:len(q['context_ids'])]=torch.tensor(q['context_ids'])
        h=backend.hidden(ids.to(backend.device))
        if h.dtype!=torch.float32:raise ValueError('Actual FP32 trunk required')
        actual=backend.model.lm_head(h[:,len(q['context_ids'])-1]).float().log_softmax(-1);backend.sync();actual=actual.cpu()
        error=float((actual-native[shard:shard+1]).abs().max());torch.save(actual,d/'forward-logps.pt')
        report['native_gate_wall_seconds']=time.monotonic()-native_start
        report['native_max_full_head_logp_error']=error;report['forward_values_sha256']=sha256_file(d/'forward-logps.pt')
        if not math.isfinite(error) or error>p['forward_atol']:raise RuntimeError('Native CPU/device full-head qualification failed before rows')
        del h,actual
        with (d/'scores.jsonl').open('x') as stream:
            for r in x['rows'][shard::SHARDS]:
                key=f'{r["cohort"]}:{r["item_id"]}'
                for interface,mode in conditions():
                    deadline(until);start=time.monotonic();context=reader_context(r,[] if interface=='raw' else x['controls'][key]['values'][mode],interface,x['parts'])
                    metadata=x['context_hashes'][key][f'{interface}/{mode}']
                    v=dict(cohort=r['cohort'],item_id=r['item_id'],document_id=r['document_id'],interface=interface,mode=mode,
                        context_sha256=canonical_hash(context),target_sha256=canonical_hash(r['target_ids']),context_tokens=len(context))
                    try:
                        if metadata!={'sha256':v['context_sha256'],'tokens':len(context)}:raise ValueError('Frozen exact context changed')
                        value=score_continuations(backend,context,[r['target_ids']],total_width=WIDTH,future_width=FUTURE,pad_id=PAD)[0]
                        lp=value.token_log_likelihoods
                        if len(lp)!=len(r['target_ids']) or not all(math.isfinite(a) for a in lp):raise ValueError('Complete finite target span required')
                        v.update(status='ok',**asdict(value),full_span_ll=sum(lp),k1_ll=lp[0],
                            numeric_only_ll=sum(a for a,on in zip(lp,r.get('numeric_token_mask',[])) if on) if r['cohort']=='arithmetic' else None)
                    except Exception as e:
                        v.update(status='error',error=f'{type(e).__name__}: {e}',wall_seconds=time.monotonic()-start)
                        stream.write(json.dumps(v,allow_nan=False)+'\n');stream.flush();report['rows']+=1;raise
                    v['wall_seconds']=time.monotonic()-start;stream.write(json.dumps(v,allow_nan=False)+'\n');stream.flush();report['rows']+=1
                print(json.dumps(dict(utc=utc_now(),shard=shard,rows=report['rows'])),flush=True)
        report.update(status='complete',scores_sha256=sha256_file(d/'scores.jsonl'))
    except BaseException as e:report.update(status='failed',error=f'{type(e).__name__}: {e}');raise
    finally:report['finished_utc']=utc_now();atomic_json(d/'run.json',report,overwrite=True)


def choose_interface(contrasts,documents):
    passing=[i for i in INTERFACES if documents>=256 and all(
        contrasts[i][k]['difference_b_minus_a']>=bound for k,bound in THRESHOLDS.items())]
    chosen=min(passing,key=lambda i:(-contrasts[i]['correct_minus_raw']['difference_b_minus_a'],i)) if passing else None
    return passing,chosen


def summarize_rows(p,x,actual):
    expected={(r['cohort'],r['item_id'],i,m) for r in x['rows'] for i,m in conditions()};index={}
    records={(r['cohort'],r['item_id']):r for r in x['rows']}
    for v in actual:
        key=(v['cohort'],v['item_id'],v['interface'],v['mode'])
        if key in index or key not in expected or v['status']!='ok':raise ValueError('Duplicate/unexpected/failed cell; no complete result')
        r=records[key[:2]];lp=v['token_log_likelihoods']
        meta=x['context_hashes'][f'{r["cohort"]}:{r["item_id"]}'][f'{key[2]}/{key[3]}']
        if (v['document_id']!=r['document_id'] or v['context_sha256']!=meta['sha256']
                or v['target_sha256']!=canonical_hash(r['target_ids']) or v['target_token_ids']!=r['target_ids']
                or len(lp)!=len(r['target_ids']) or not all(math.isfinite(a) and a<=0 for a in lp)
                or v['full_span_ll']!=sum(lp) or v['k1_ll']!=lp[0]
                or (r['cohort']=='arithmetic' and v['numeric_only_ll']!=sum(a for a,on in zip(lp,r['numeric_token_mask']) if on))):raise ValueError('Incomplete/nonfinite/changed full span')
        index[key]=v
    if set(index)!=expected:raise ValueError('Complete batch required; never shrink failure denominator')
    results={}
    for cohort in COHORTS:
        rr=[r for r in x['rows'] if r['cohort']==cohort];absolute={};contrasts={};passing=[]
        for metric in (('full_span_ll','k1_ll','numeric_only_ll') if cohort=='arithmetic' else ('full_span_ll','k1_ll')):
            def values(i,m):return [dict(item_id=r['item_id'],document_id=r['document_id'],score=index[(cohort,r['item_id'],i,m)][metric]) for r in rr]
            absolute[metric]={f'{i}/{m}':sum(z['score'] for z in values(i,m))/len(rr) for i,m in conditions()}
            contrasts[metric]={}
            for i in INTERFACES:
                cc={}
                for a,b in (('raw','empty'),('raw','correct'),('empty','correct'),('corrupt','correct'),('shuffled','correct'),('other','correct')):
                    cc[f'{b}_minus_{a}']=paired_cluster_bootstrap(values('raw','raw') if a=='raw' else values(i,a),values(i,b),repeats=p['bootstrap']['repeats'],seed=p['bootstrap']['seed'])
                contrasts[metric][i]=cc
        passing,chosen=choose_interface(contrasts['full_span_ll'],len(rr))
        lengths=[len(r['copy_note_ids']) for r in rr]
        corrupt_equal=sum(len(x['controls'][f'{cohort}:{r["item_id"]}']['values']['correct'])==len(x['controls'][f'{cohort}:{r["item_id"]}']['values']['corrupt']) for r in rr)
        results[cohort]=dict(correct_corrupt_note_token_lengths=dict(equal=corrupt_equal,unequal=len(rr)-corrupt_equal),
            control_note_token_means={mode:sum(len(x['controls'][f'{cohort}:{r["item_id"]}']['values'][mode]) for r in rr)/len(rr) for mode in MODES},
            note_tokens=dict(min=min(lengths),max=max(lengths),mean=sum(lengths)/len(lengths)),
            control_coincidences={mode:sum(x['controls'][f'{cohort}:{r["item_id"]}'][f'{mode}_equals_correct'] for r in rr) for mode in ('shuffled','corrupt','other')},
            scoring_wall_seconds=sum(v['wall_seconds'] for key,v in index.items() if key[0]==cohort),
            raw_fullspan_surprisal_headroom=-absolute['full_span_ll']['raw/raw'],
            mean_gain2_possible_by_probability_bound=-absolute['full_span_ll']['raw/raw']>=2.,
            documents=len(rr),absolute=absolute,contrasts=contrasts,passing_interfaces=passing,chosen_interface=chosen,
            raw_reused_exactly_once_per_document=True)
    return dict(status='complete',cohorts=results,rows=len(actual),failures=0,thresholds=THRESHOLDS,
        cohort_choice_agreement=results['general']['chosen_interface'] is not None and results['general']['chosen_interface']==results['arithmetic']['chosen_interface'],
        cohort_document_overlap=p['cohort_document_overlap'],arithmetic_publication_coverage=p['arithmetic_publication_coverage'],
        interpretation='Future-visible target-copy plumbing and interface selection only; not useful policy, training, or held-out confirmation.')


def summarize(output):
    p,x=load(output);ref,_=reference(output,p,x);actual=[];pins={}
    for shard in range(SHARDS):
        d=output/'scores'/f'shard-{shard}';r=read(d/'run.json')
        if (r['status']!='complete' or r['rows']!=2*N*31//SHARDS
                or r['protocol_sha256']!=sha256_file(output/'protocol.json') or r['signature']!=ref['signature']
                or not math.isfinite(r['native_max_full_head_logp_error']) or r['native_max_full_head_logp_error']>p['forward_atol']
                or r['forward_values_sha256']!=sha256_file(d/'forward-logps.pt') or r['scores_sha256']!=sha256_file(d/'scores.jsonl')):
            raise ValueError('All four complete immutable shards required')
        actual.extend(rows(d/'scores.jsonl'));pins[str(d/'run.json')]=sha256_file(d/'run.json');pins[str(d/'scores.jsonl')]=r['scores_sha256']
    result=summarize_rows(p,x,actual);result.update(protocol_sha256=sha256_file(output/'protocol.json'),input_hashes=pins,finished_utc=utc_now())
    atomic_json(output/'summary.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare-general','prepare','cpu-reference','score','summarize'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--deadline-utc');parser.add_argument('--general',type=Path)
    parser.add_argument('--arithmetic',type=Path);parser.add_argument('--selection',type=Path);parser.add_argument('--records',type=Path)
    parser.add_argument('--model-dir',type=Path,default=ROOT/'local/assets/qwen3-1.7b-base')
    parser.add_argument('--format',type=Path,default=ROOT/'local/resolved_format.json');parser.add_argument('--published',type=Path,default=ROOT/'configs/rlp_published_prompt.json')
    parser.add_argument('--holdout-exclusions',type=Path,default=ROOT/'local/diagnostics/postsft-arithmetic-held-dev-exclusions-v1.json')
    parser.add_argument('--cpu-threads',type=int,default=16);parser.add_argument('--shard-index',type=int)
    a=parser.parse_args()
    if a.command!='summarize' and a.deadline_utc is None:parser.error('--deadline-utc required')
    if a.command=='prepare-general':prepare_general(a.output,a.selection,a.records,a.model_dir,a.deadline_utc)
    elif a.command=='prepare':prepare(a.output,a.general,a.arithmetic,a.model_dir,a.format,a.published,a.deadline_utc,a.holdout_exclusions)
    elif a.command=='cpu-reference':cpu_reference(a.output,a.cpu_threads,a.deadline_utc)
    elif a.command=='score':score(a.output,a.shard_index,a.deadline_utc)
    else:summarize(a.output)


if __name__=='__main__':main()
