"""Independently reparse closed fixed-panel answers; never generate or train.

The input contract is the immutable v6 TRAIN128-model DEV experiment. Require
the actual four-panel controller closures before reading any gold or outcomes.
"""
import argparse
from collections import Counter
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
from datetime import datetime, timezone

import numpy as np

PANELS=('DEV_interpolation','DEV_structure','DEV_range','DEV_composition')


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def parse_independent(text,*,cot=False,eos=False):
    """Separate whole-first-line implementation, no later-answer rescue."""
    if cot:
        label=re.search(r'(?im)^[ \t]*Answer:[ \t]*(?:\r?\n)?',text)
        if label is None:return None
        text=text[label.end():]
    else:
        text=text.lstrip()
        label=re.match(r'(?i)Answer:[ \t]*(?:\r?\n)?',text)
        if label:text=text[label.end():]
    text=text.lstrip()
    if '\n' not in text and not eos:return None
    line=text.split('\n',1)[0].strip()
    if line.endswith('.'):line=line[:-1].rstrip()
    line=line.replace('−','-')
    # Validate comma groups before removing them; Decimal alone is too lenient.
    if ',' in line:
        integer=re.split(r'[.eE]',line.lstrip('+-'),maxsplit=1)[0]
        groups=integer.split(',')
        if (not 1<=len(groups[0])<=3 or not all(g.isdecimal() for g in groups)
                or any(len(g)!=3 for g in groups[1:]) or ',' in line[len(line)-len(line.lstrip('+-'))+len(integer):]):
            return None
    cleaned=line.replace(',','')
    if re.fullmatch(r'[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?',cleaned) is None:return None
    value=Decimal(cleaned)
    return value if value.is_finite() else None


def independent_score(value,gold):
    stop=value['stop_type'];parsed=parse_independent(value['text'],cot=value['mode']=='cot',eos=stop=='token:151643')
    if stop not in ('error','length_cap','number_line','token:151643') or value['status'] not in ('ok','error'):
        raise ValueError('Unknown original generation outcome')
    status=('error' if value['status']=='error' or stop=='error' else
            'truncated' if stop=='length_cap' else 'format_error' if parsed is None else 'ok')
    return dict(score=int(status=='ok' and parsed==Decimal(str(gold))),status=status)


def exact_trace(tree):
    if type(tree)is int:return tree,[],str(tree)
    op,left,right=tree;a,la,ls=exact_trace(left);b,lb,rs=exact_trace(right)
    if op=='+':value=a+b
    elif op=='-':value=a-b
    elif op=='*':value=a*b
    else:raise ValueError('Unknown prefix computation')
    return value,la+lb+[(a,op,b,value)],f'({ls} {op} {rs})'


def exact_tree(tree):
    return exact_trace(tree)[0]


def independent_state(note,tree):
    """Re-execute prefix arithmetic and parse saved text without producer scores."""
    statements=[];lines=note['text'].strip().splitlines();supported=bool(lines)
    for line in lines:
        match=re.fullmatch(r'\s*([+-]?\d+)\s*([+*\-])\s*([+-]?\d+)\s*=\s*([+-]?\d+)\s*',line)
        if match is None:
            supported=False;break
        statements.append((int(match[1]),match[2],int(match[3]),int(match[4])))
    if not supported:statements=[]
    trace=exact_trace(tree)[1];canonical=supported and statements==trace[:-1]
    correct=supported and all((a+b if op=='+' else a-b if op=='-' else a*b)==v for a,op,b,v in statements)
    return dict(canonical_nonroot_exact=canonical,
                canonical_and_natural_stop=canonical and note['status']=='ok' and note['stop_type']=='token:151643',
                equations_supported=supported,all_stated_equations_correct=correct,
                root_operation_transmitted=supported and trace[-1] in statements,
                expected_nonroot_statements=len(trace)-1,observed_statements=len(statements),
                stop_type=note['stop_type'],status=note['status'])


def paired(delta):
    delta=np.asarray(delta,float);n=len(delta)
    picks=np.random.default_rng(17).integers(0,n,size=(5000,n))
    return dict(difference=float(delta.mean()),ci95=np.quantile(delta[picks].mean(1),[.025,.975]).tolist(),items=n,repeats=5000,seed=17,unit='accuracy_fraction')


def numerical_readback(path,shard,*,notes=False,reference_root=None):
    import torch
    from train.useful_note_precision import compare_distributions
    d=path/('notes' if notes else 'answers')/f'shard-{shard}'
    reference=torch.load((reference_root or path)/'native-reference.pt',map_location='cpu',weights_only=True)
    actual=torch.load(d/'native-logps.pt',map_location='cpu',weights_only=True)
    native=compare_distributions(reference[shard:shard+1],actual)[0]
    cache={}
    for mode in (['prefill'] if notes else ['direct','cot','teacher']):
        vectors=torch.load(d/f'cache-{mode}-logps.pt',map_location='cpu',weights_only=True)
        cache[mode]=compare_distributions(vectors['ordinary'],vectors['cached'])[0]
    if not native['passed'] or any(v['max_logp_error']>.001 for v in cache.values()):
        raise ValueError('Actual saved numerical vectors do not reproduce the gates')
    return dict(shard=shard,native=native,cache=cache,actual_vectors_recomputed=True)


def audit(root):
    root=Path(root).resolve();controls=root/'local/diagnostics';evaluation=root/'local/nvme/evaluation'
    control_dirs={arm:controls/f'partial-state-v6-fixed-DEV512-{arm}128-v1' for arm in ('writer','cot','reader')}
    # No partial-panel peeking: close the entire fixed question before review.
    for directory in control_dirs.values():
        if read(directory/'completion.json')['status']!='COMPLETE_ALL_FOUR_FIXED_DEV_PANELS':
            raise ValueError('All original four-panel jobs must close before review')
    from transformers import AutoTokenizer
    from train.partial_state_evaluation import prompt_tokens,reader_tokens
    from train.partial_state_answers import score_answer
    from train.partial_state_note_driver import load_bundle
    tok=AutoTokenizer.from_pretrained(root/'local/assets/qwen3-1.7b-base',local_files_only=True,trust_remote_code=False)
    public=root/'validation/partial_state_v6_DEV512_frozen128_20261018';public.mkdir(exist_ok=False)
    archive=root/'local/backups/evaluation/partial-state-v6-fixed-DEV512-closed';archive.mkdir(exist_ok=False,parents=True)
    panels={};payloads={};archived={};answer_count=0
    def archive_tree(label,source):
        destination=archive/label;shutil.copytree(source,destination)
        for f in source.rglob('*'):
            if f.is_file():
                out=destination/f.relative_to(source);h=sha(f)
                if sha(out)!=h:raise ValueError('Archive readback differs')
                archived[str(out.relative_to(archive))]=dict(bytes=f.stat().st_size,sha256=h)
    for arm,directory in control_dirs.items():archive_tree('control-'+arm,directory)
    for index,panel in enumerate(PANELS):
        paths=dict(writer=evaluation/f'partial-state-v6-{panel}-writer128-XLA-notes-v2',
                   reader=evaluation/f'partial-state-v6-{panel}-reader128-answers-v4',
                   cot=evaluation/f'partial-state-v6-{panel}-cot128-answers-v4')
        for arm,path in paths.items():
            completion=read(control_dirs[arm]/f'stage-{index}/completion.json')
            result=path/('notes-bundle.json' if arm=='writer' else 'summary.json')
            if (completion['status']!='COMPLETE_FIXED_PANEL_ALL512_ATTEMPTS' or completion['output']!=str(path)
                    or sha(result)!=completion['result_sha256']
                    or read(path/'generation-closure.json')['status']!='ALL_FOUR_GENERATION_SESSIONS_AND_DEVICE_PHYSICALLY_CLOSED'):
                raise ValueError('Complete physical closure/result binding required')
            archive_tree(panel+'/'+arm,path)
        rp,cp=read(paths['reader']/'protocol.json'),read(paths['cot']/'protocol.json')
        qs=rows(paths['reader']/'questions.jsonl');cs=rows(paths['reader']/'controls.jsonl')
        ids=[q['item_id'] for q in qs]
        if len(ids)!=512 or len(set(ids))!=512 or ids!=cp['ordered_item_ids'] or ids!=rp['ordered_item_ids']:
            raise ValueError('Same complete512-question cohort required')
        question={q['item_id']:q for q in qs};control={c['item_id']:c for c in cs}
        gold={q['item_id']:exact_tree(q['expression_tree']) for q in qs}
        if gold!={r['item_id']:int(r['answer']) for r in rows(paths['reader']/'gold.jsonl')}:
            raise ValueError('Stored gold differs from independent prefix interpreter')
        bundle=load_bundle(paths['writer']/'notes-bundle.json',ids)
        if bundle['execution_device']!='xla' or rp['writer']!=bundle['writer']:
            raise ValueError('Actual fixed Trainium writer required')
        teacher_ids={}
        for q,c in zip(qs,cs):
            _,trace,text=exact_trace(q['expression_tree']);partial=trace[:-1]
            teacher=''.join(f'{a} {op} {b} = {v}\n' for a,op,b,v in partial)
            a,op,b,v=partial[-1];wrong=partial[:-1]+[(a,op,b,v+1)]
            corrupt=''.join(f'{a} {op} {b} = {v}\n' for a,op,b,v in wrong)
            expected={'empty':[],'teacher':tok.encode(teacher,add_special_tokens=False),
                      'corrupt':tok.encode(corrupt,add_special_tokens=False),
                      'restatement':tok.encode(text+'\n',add_special_tokens=False)}
            if any(c['notes'][m]!=v for m,v in expected.items()):raise ValueError('Supplied control differs from independent prefix computation')
            teacher_ids[q['item_id']]=expected['teacher']
        if sorted(c['other_teacher_item_id'] for c in cs)!=sorted(ids):raise ValueError('Teacher donors must preserve every note exactly once')
        for c in cs:
            if c['other_teacher_item_id']==c['item_id'] or c['notes']['other_teacher']!=teacher_ids[c['other_teacher_item_id']]:
                raise ValueError('Other-teacher note was changed or self-paired')
        for c,n in zip(cs,bundle['controls']):
            for mode in ('own','other'):
                if c['notes'][mode]!=n[mode]['content_ids'] or c['note_generation'][mode]['item_id']!=n[mode]['item_id']:
                    raise ValueError('Reader did not consume every intact actual note/donor')
        wp=read(paths['writer']/'protocol.json');writer_reference=Path(wp['model_preparation'])
        archive_tree(panel+'/writer-native-preparation',writer_reference)
        writer_checks=[]
        for shard in range(4):
            stage=control_dirs['writer']/f'stage-{index}';exit=read(stage/f'shard-{shard}/exit.json')
            if exit['exit_code']!=0 or exit['original_session_remaining']:raise ValueError('Original writer sessions must close')
            writer_checks.append(numerical_readback(paths['writer'],shard,notes=True,reference_root=writer_reference))
        cells={};metrics={};checks={'writer':writer_checks};raw=[]
        for arm in ('reader','cot'):
            path=paths[arm];p=rp if arm=='reader' else cp;summary=read(path/'summary.json');gates=[]
            for shard in range(4):
                stage=control_dirs[arm]/f'stage-{index}';exit=read(stage/f'shard-{shard}/exit.json');d=path/f'answers/shard-{shard}';run=read(d/'run.json')
                if exit['exit_code']!=0 or exit['original_session_remaining'] or run['status']!='complete_all_attempts' or sha(d/'answers.jsonl')!=run['answers_sha256']:
                    raise ValueError('Original answer shards must be complete and closed')
                native=read(d/'native.json');cache={m:read(d/f'cache-{m}.json') for m in ('direct','cot','teacher')}
                if not native['passed'] or any(v['max_logp_error']>.001 for v in cache.values()):
                    raise ValueError('Actual numerical gates failed')
                gates.append(numerical_readback(path,shard))
                for v in rows(d/'answers.jsonl'):
                    key=(arm,v['item_id'],v['mode'])
                    if key in cells or v['item_id'] not in gold or v['mode'] not in p['modes']:raise ValueError('Unexpected duplicate/unknown outcome')
                    q=question[v['item_id']];mode=v['mode']
                    prompt=(prompt_tokens(q,tok,cot=mode=='cot') if mode in ('direct','cot') else reader_tokens(q,control[q['item_id']]['notes'][mode],tok))
                    expected_cap=80 if mode in ('direct','cot') else 16
                    if v['prompt_ids']!=prompt or v['cap']!=expected_cap or v['prefill_width']!=(32 if mode in ('direct','cot') else 128) or v['n_actions']!=len(v['actions']):raise ValueError('Actual prefix/condition/budget changed')
                    independent=independent_score(v,gold[v['item_id']]);reference=score_answer(v,gold[v['item_id']],cot=mode=='cot')
                    if independent!={k:reference[k] for k in ('score','status')}:raise ValueError('Independent complete-answer parsers disagree')
                    cells[key]={**v,**independent};raw.append(dict(arm=arm,**cells[key]));answer_count+=1
            result={}
            for mode in p['modes']:
                vs=[cells[arm,k,mode] for k in ids]
                result[mode]=dict(items=512,correct=sum(v['score'] for v in vs),unsupported=sum(v['status']!='ok' for v in vs),status_counts=dict(Counter(v['status'] for v in vs)),generated_actions=sum(v['n_actions'] for v in vs),writer_actions=sum(v['n_writer_actions'] for v in vs),decode_wall_seconds=sum(v['wall_seconds'] for v in vs),writer_wall_seconds=sum(0 if v['source_note'] is None else v['source_note']['wall_seconds'] for v in vs))
                for key,value in result[mode].items():
                    if summary['results'][mode][key]!=value:raise ValueError('Original summary does not reproduce')
            for name,value in summary['contrasts'].items():
                b,a=name.split('_minus_');actual=paired([cells[arm,k,b]['score']-cells[arm,k,a]['score'] for k in ids])
                if actual['difference']!=value['difference'] or actual['ci95']!=value['ci95']:raise ValueError('Original paired interval does not reproduce')
            metrics[arm]=result;checks[arm]=gates
        if len(cells)!=512*9:raise ValueError('Every original outcome must stay in denominator')
        contrasts={}
        for b,a in [('own','direct'),('own','empty'),('own','other'),('teacher','own'),('teacher','direct'),('teacher','corrupt'),('teacher','other_teacher'),('own','restatement')]:
            contrasts[b+'_minus_'+a]=paired([cells['reader',k,b]['score']-cells['reader',k,a]['score'] for k in ids])
        contrasts['student_split_minus_compact_CoT']=paired([cells['reader',k,'own']['score']-cells['cot',k,'cot']['score'] for k in ids])
        notes=[v['own'] for v in bundle['controls']]
        state={q['item_id']:independent_state(note,q['expression_tree']) for q,note in zip(qs,notes)}
        if state!=bundle['state_diagnostics']:raise ValueError('Writer-state diagnostics differ from independent prefix/text computation')
        states=dict(items=512,canonical_nonroot_exact=sum(v['canonical_nonroot_exact'] for v in state.values()),canonical_and_natural_stop=sum(v['canonical_and_natural_stop'] for v in state.values()),root_operation_transmitted=sum(v['root_operation_transmitted'] for v in state.values()),stop_counts=dict(Counter(v['stop_type'] for v in notes)),status_counts=dict(Counter(v['status'] for v in notes)))
        file=public/f'{panel}-all4608-original-answers.jsonl.gz'
        with gzip.GzipFile(filename=str(file),mode='wb',mtime=0) as stream:
            for value in raw:stream.write((json.dumps(value,separators=(',',':'))+'\n').encode())
        note_file=public/f'{panel}-all512-original-notes.jsonl.gz'
        with gzip.GzipFile(filename=str(note_file),mode='wb',mtime=0) as stream:
            for value in notes:stream.write((json.dumps(value,separators=(',',':'))+'\n').encode())
        for f,n in [(file,4608),(note_file,512)]:payloads[str(f.relative_to(root))]=dict(rows=n,bytes=f.stat().st_size,sha256=sha(f))
        examples=[dict(question=q,writer=notes[i],reader_direct=cells['reader',q['item_id'],'direct'],reader_own=cells['reader',q['item_id'],'own'],ordinary_CoT=cells['cot',q['item_id'],'cot']) for i,q in enumerate(qs[:2])]
        panels[panel]=dict(results=metrics,contrasts=contrasts,writer_state_diagnostics=states,numerical_checks=checks,examples_fixed_first_two=examples,reader=rp['reader'],writer=bundle['writer'],CoT=cp['reader'])
    if answer_count!=18432:raise ValueError('All18,432answers required')
    write(archive/'ARCHIVE.json',dict(status='COMPLETE_FIXED_DEV_EBS_READBACK_VERIFIED',files=archived))
    report=dict(schema='partial-state-fixed-DEV512-frozen128-complete-v1',status='COMPLETE_ALL_FOUR_PANELS_NO_FILTERING',finished_utc=datetime.now(timezone.utc).isoformat(),panels=panels,answer_attempts=answer_count,writer_attempts=2048,independent_prefix_gold_and_answer_reparse=True,independent_writer_state_reparse=True,all_original_intervals_recomputed=True,all_original_sessions_devices_closed=True,audit_script_sha256=sha(__file__),public_payloads=payloads,EBS_archive_receipt_sha256=sha(archive/'ARCHIVE.json'),limitations=['128TRAIN fitting models,32passes, one seed. DEV is diagnostic, not an official final test.','Separate512-item panels; do not pool generalization or claim seed robustness from item intervals.','No generation failures, truncations, unsupported answers or bad notes removed.','Ordinary compact CoT uses one model/128training updates; split has two models/256total updates. Equal80generated-action caps do not equal measured FLOPs.','Per-row timing includes first-call compilation in separate jobs; not an end-to-end matched latency benchmark.','These are synthetic exact-computation diagnostics, not broad language capability or pretraining gains.'])
    write(root/'validation/partial_state_v6_DEV512_frozen128_20261018.json',report)
    return {panel:dict(writer=panels[panel]['writer_state_diagnostics'],answers={arm:{mode:(v['correct'],v['unsupported']) for mode,v in result.items()} for arm,result in panels[panel]['results'].items()},contrasts=panels[panel]['contrasts']) for panel in PANELS}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    print(json.dumps(audit(parser.parse_args().root),indent=2,allow_nan=False))
