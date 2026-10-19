"""Frozen TRAIN-only analysis of pointer errors and gold-prefix field closure."""
import argparse,json,hashlib,os,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
from collections import Counter
from types import SimpleNamespace
from train import document_qa_evaluate as engine
from train.document_evidence_plan_evaluate import prompt_text
from weekend_thinking.artifacts import atomic_json,sha256_file,utc_now

R=Path('/home/ubuntu/dario');W=R/'local/worktrees/document-evidence-eval-v23'
OLD=R/'local/diagnostics/document-evidence-weight-v25-ablation-v1'
T=OLD/'TRAIN-fit';O=R/'local/diagnostics/selector-TRAIN-decisions-CPU-v1'
def sources():
    return {**engine.sources(),**{n:sha256_file(W/n) for n in ('train/document_evidence_plan_evaluate.py','train/document_evidence_plan_packing.py')},str(Path(__file__).relative_to(R)):sha256_file(__file__)}
def category(v,gold):
    if v['selection']['status']!='complete':return 'unsupported_selection'
    ids=list(v['selection']['indices']);gold=list(gold)
    if set(ids)==set(gold):return 'exact_annotated_set'
    if len(ids)<len(gold) and ids==gold[:len(ids)]:return 'canonical_prefix_then_early_closure'
    if set(ids)<set(gold):return 'annotated_subset_with_earlier_omission'
    if any(i not in gold for i in ids):return 'contains_nonannotated_ID'
    return 'other_sequence_deviation'
def prepare():
    from transformers import AutoTokenizer
    assert engine.ROOT==W and not O.exists()
    p=engine.read(R/'local/nvme-cpt/evaluation/document-evidence-weight-v25-TRAIN256-v1/protocol.json')
    binding=OLD/'frozen-models/evidence_select-weight1-seed2026101923.json';model=engine.read(binding)
    assert model['step']==256 and model['checkpoint_sha256']==p['model']['checkpoint_sha256']==sha256_file(model['checkpoint'])
    assert engine.read(T/'independent-check.json')['status'].startswith('PASS_')
    questions=engine.rows(T/'TRAIN256-questions-only.jsonl');labels={v['item_id']:v for v in engine.rows(T/'TRAIN256-labels.jsonl')}
    native={v['native']['item_id']:v for v in engine.rows(T/'native-and-scored.jsonl')}
    assert len(questions)==len(labels)==len(native)==256 and {q['item_id'] for q in questions}==set(labels)==set(native)
    tok=AutoTokenizer.from_pretrained(p['model_dir'],local_files_only=True,trust_remote_code=False)
    items=[]
    for q in questions:
        assert set(q)=={'item_id','question','contexts'}
        ids=labels[q['item_id']]['support_indices'];assert ids==sorted(ids) and len(ids)==len(set(ids))
        text=prompt_text(q,'selection');prefix=tok.encode(text,add_special_tokens=False)
        full=tok.encode(text+'Evidence:'+''.join(f' [{i}]' for i in ids)+'\n',add_special_tokens=False)
        assert full[:len(prefix)]==prefix and len(full)<=3584
        branches=[]
        for count in range(1,len(ids)):
            closed=tok.encode(text+'Evidence:'+''.join(f' [{i}]' for i in ids[:count])+'\n',add_special_tokens=False)
            shared=0
            for a,b in zip(full,closed):
                if a!=b:break
                shared+=1
            assert len(prefix)<=shared<min(len(full),len(closed)) and full[shared]!=closed[shared]
            branches.append(dict(after_ID_count=count,shared_prefix_tokens=shared,required_token=full[shared],closure_token=closed[shared],required_surface=tok.decode([full[shared]]),closure_surface=tok.decode([closed[shared]])))
        items.append(dict(item_id=q['item_id'],support_count=len(ids),ids=full,prefix_tokens=len(prefix),branches=branches,native_category=category(native[q['item_id']],ids)))
    assert Counter(v['support_count'] for v in items)==Counter({2:128,3:96,4:32})
    O.mkdir();path=O/'inputs.jsonl'
    with path.open('x') as f:
        for v in items:f.write(json.dumps(v)+'\n')
    atomic_json(O/'design.json',dict(status='PREPARED_FROZEN_TRAIN_DECISION_DIAGNOSTIC_CPU_ONLY',utc=utc_now(),deadline_utc=(datetime.now(timezone.utc)+timedelta(minutes=40)).isoformat(),
        question='Is incomplete annotated selection already preferred under correct preceding IDs, or preceded by another free-running selection deviation?',
        model=model,model_dir=p['model_dir'],signature=p['signature'],sources=sources(),input_sha256=sha256_file(path),items=256,
        inputs={str(z):sha256_file(z) for z in [binding,T/'TRAIN256-questions-only.jsonl',T/'TRAIN256-labels.jsonl',T/'native-and-scored.jsonl']},
        native_reference_dir=str(R/'local/nvme-cpt/evaluation/document-evidence-weight-v25-TRAIN256-v1'),
        endpoints=['Native annotation-set errors classified before new forwards.','For each required additional ID under canonical correct prefixes, logp(field closure) minus logp(required continuation first differing token).','Descriptive per-question/stratum counts; no pooled branches as independent observations, gate or held-out inference.'],
        limits=['Teacher-forced annotation inputs are explicitly privileged TRAIN diagnostic conditions, not prospective self-generated thoughts.','Correct set means author annotation agreement, not semantic entailment. Alternate valid subsets/order are not adjudicated.','CPU FP32 versus native BF16 can change near-tied decisions. This is not a causal attribution of original native errors.','No training, new held data, prompt rescue or V26 source/score changes.'],CPU_threads=16,accelerator_calls=0,no_automatic_retry=True))
    print('PREPARED_FROZEN_TRAIN_DECISION_DIAGNOSTIC_CPU_ONLY',flush=True)
def run():
    import torch
    from train.evaluate_local import setup_model
    from train.useful_note_precision import compare_distributions
    assert engine.ROOT==W
    p=engine.read(O/'design.json');assert p['sources']==sources() and sha256_file(O/'inputs.jsonl')==p['input_sha256']
    assert all(sha256_file(path)==digest for path,digest in p['inputs'].items())
    assert sha256_file(p['model']['checkpoint'])==p['model']['checkpoint_sha256']
    deadline=datetime.fromisoformat(p['deadline_utc']);torch.set_num_threads(p['CPU_threads'])
    backend,tok,loaded,signature,_=setup_model(SimpleNamespace(model_dir=Path(p['model_dir']),checkpoint=Path(p['model']['checkpoint']),device='cpu'))
    assert all(loaded[k]==p['model'][k] for k in ('run_id','step','spec_hash')) and signature==p['signature']
    old=Path(p['native_reference_dir']);ref=engine.read(old/'native-contexts.json')['refs'][0]['context_ids']
    x=torch.full((1,1024),151643,dtype=torch.long);x[0,:len(ref)]=torch.tensor(ref)
    with torch.no_grad():
        hidden=backend.hidden(x);actual=backend.model.lm_head(hidden[:,len(ref)-1]).float().log_softmax(-1).cpu()
    expected=torch.load(old/'native-reference.pt',map_location='cpu',weights_only=True)
    check=compare_distributions(expected[:1],actual)[0];assert check['passed'];atomic_json(O/'actual-CPU-model-check.json',check)
    del x,hidden,actual,expected
    items=engine.rows(O/'inputs.jsonl');start=time.monotonic();done=[]
    with (O/'observations.jsonl').open('x') as f,torch.no_grad():
        for item in items:
            if datetime.now(timezone.utc)>=deadline:raise TimeoutError('Bounded CPU diagnostic; preserve partial observations, no retry')
            before=time.monotonic();x=torch.tensor([item['ids']],dtype=torch.long);hidden=backend.hidden(x)
            positions=[v['shared_prefix_tokens']-1 for v in item['branches']]
            lp=backend.model.lm_head(hidden[0,positions]).float().log_softmax(-1)
            branches=[]
            for index,v in enumerate(item['branches']):
                a,b=float(lp[index,v['required_token']]),float(lp[index,v['closure_token']])
                branches.append(dict(**v,required_logp=a,closure_logp=b,closure_minus_required_logp=b-a,closure_preferred=b>a))
            value=dict(item_id=item['item_id'],support_count=item['support_count'],native_category=item['native_category'],branches=branches,wall_seconds=time.monotonic()-before)
            f.write(json.dumps(value)+'\n');f.flush();done.append(value);del x,hidden,lp
            print(json.dumps(dict(rows=len(done),item_id=item['item_id'],seconds=value['wall_seconds'])),flush=True)
    strata={}
    for n in (2,3,4):
        sub=[v for v in done if v['support_count']==n];margins=[b['closure_minus_required_logp'] for v in sub for b in v['branches']]
        strata[str(n)]=dict(items=len(sub),native_categories=dict(Counter(v['native_category'] for v in sub)),required_continuation_decisions=len(margins),closure_preferred_decisions=sum(x>0 for x in margins),questions_with_any_closure_preferred=sum(any(b['closure_preferred'] for b in v['branches']) for v in sub))
    atomic_json(O/'result.json',dict(status='COMPLETE_TRAIN_TEACHER_FORCED_CLOSURE_AND_NATIVE_ERROR_DIAGNOSTIC_CPU_ONLY',utc=utc_now(),items=len(done),strata=strata,wall_seconds=time.monotonic()-start,source_sha256=sha256_file(__file__),design_sha256=sha256_file(O/'design.json'),observations_sha256=sha256_file(O/'observations.jsonl'),actual_CPU_reference_pass=check['passed'],limits=p['limits'],accelerator_calls=0))
    print(json.dumps(strata,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','run'));args=parser.parse_args();(prepare if args.command=='prepare' else run)()
