"""TRAIN-only natural-stop integration of the unchanged b25560a decoder.

One frozen V17 trained model, eight prefix-only inputs, production128 action cap.
No model update, held data, scoring or change to either live training source.
"""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
import argparse
import hashlib
import json
import subprocess
import time
from statistics import median

import torch

from train.batched_causal_decoder import BatchedGreedyDecoder, BatchGenerationFailure
from train.evaluate import GreedyDecoder, GenerationFailure, asset_provenance
from train.evaluate_local import setup_model
from train.neuron_backend import CausalCache
from train.real_trace_contexts import question_projection, writer_prompt, boundary_stop_ids
from train.real_trace_packing import prefix_calculation
from train.useful_note_precision import compare_distributions
from weekend_thinking.artifacts import atomic_json, sha256_file

R=Path('/home/ubuntu/dario')
W=R/'local/worktrees/causal-decoder-batching-v1'
C=R/'local/diagnostics/causal-decoder-trained-natural-v2'
ASSET=R/'local/assets/qwen3-1.7b-base'
TRAIN=R/'local/nvme-study/data/real-prefix-partial-labels-v9/TRAIN.jsonl'
CHECKPOINT=R/'local/nvme-cpt/runs/real-source-balanced-v17-source_cot-TRAIN5120-seed2026101913-v1/step-0000320/model.pt'
CHECKPOINT_SHA='9fb43e0faa8180c8859a50b722489821225379f8dc8424115f3ccdbbee0c0439'
read=lambda p:json.loads(Path(p).read_text())


def unchanged(plan):
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()==plan['candidate_commit']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip()
    assert all(sha256_file(path)==value for path,value in plan['sources'].items())
    assert sha256_file(CHECKPOINT)==plan['checkpoint_sha256']
    assert asset_provenance(ASSET)==plan['assets']


def prepare():
    C.mkdir(exist_ok=False)
    assert sha256_file(TRAIN)=='437c64bc9ce4b451f4d5da07fccca9e108ee9add75b3c955bc54d2a2eca6b915'
    assert sha256_file(CHECKPOINT)==CHECKPOINT_SHA
    torch.set_num_threads(16)
    backend,tok,binding,signature,_=setup_model(SimpleNamespace(model_dir=ASSET,checkpoint=CHECKPOINT,device='cpu'))
    assert binding['sha256']==CHECKPOINT_SHA and binding['step']==320
    candidates=[]
    for line in TRAIN.open():
        row=json.loads(line)
        assert row['split']=='train'
        q=question_projection(row)
        if len(q['prefix_ids'])!=512:continue
        work,_=prefix_calculation(q['prefix_ids'],tok)
        depth=len(work.splitlines())
        if depth in (2,3,4,5):candidates.append((depth,q))
    selected=[]
    for depth in (2,3,4,5):
        values=[q for d,q in candidates if d==depth]
        values.sort(key=lambda q:hashlib.sha256(('trained-natural-v2:'+q['item_id']).encode()).hexdigest())
        assert len(values)>=2
        selected += [dict(item_id=q['item_id'],document_id=q['document_id'],prefix_only_operation_count=depth,
            prompt_ids=writer_prompt(q,tok)) for q in values[:2]]
    assert len({len(q['prompt_ids']) for q in selected})==1
    atomic_json(C/'inputs.json',selected)
    vectors=[]
    with torch.no_grad():
        for row in selected:
            ids=row['prompt_ids'];x=torch.full((1,544),151643,dtype=torch.long)
            x[0,:len(ids)]=torch.tensor(ids)
            h=backend.hidden(x);vectors.append(backend.model.lm_head(h[:,len(ids)-1]).float().log_softmax(-1))
    torch.save(torch.cat(vectors),C/'CPU-native.pt')
    names=[Path(__file__),W/'train/batched_causal_decoder.py',W/'train/evaluate.py',
        W/'train/evaluate_local.py',W/'train/neuron_backend.py',W/'weekend_thinking/core.py',
        W/'train/real_trace_contexts.py',W/'train/real_trace_packing.py']
    plan=dict(status='PREPARED_TRAIN_ONLY_TRAINED_NATURAL_STOP_ENGINEERING_CHECK',
        candidate_commit='b25560a9618970a43afab349edc785d772a02604',
        sources={str(path):sha256_file(path) for path in names},assets=asset_provenance(ASSET),
        checkpoint=str(CHECKPOINT),checkpoint_sha256=CHECKPOINT_SHA,checkpoint_binding=binding,
        trained_signature=signature,train_sha256=sha256_file(TRAIN),inputs_sha256=sha256_file(C/'inputs.json'),
        cpu_native_sha256=sha256_file(C/'CPU-native.pt'),prompts=8,prefix_tokens=len(selected[0]['prompt_ids']),
        prefill_width=544,max_new_tokens=128,stop_ids=list(boundary_stop_ids(tok)),
        batch_sizes=[1,4,8],timed_repeats=3,physical_device=8,logical_core=32,maximum_reserved_chip_hours=.5,
        gate='All24 native vectors pass standard distributions plus max_logp_error<=.001;every action/content/text/stop equals original serial decoder in all conditions/repeats;>=6/8 natural stops and>=2 distinct action lengths.',
        limits='One frozen trained model/eight TRAIN-only equal-length prompts selected by prefix-derived operation strata2/3/4/5,not quality. FP32 inference,production128 cap,native Answer/EOS stops. No sampling,ragged batches,new science evaluation,learning or global-equivalence claim. Timing under concurrent separate training compilation is descriptive.')
    atomic_json(C/'plan.json',plan);unchanged(plan)
    print('PREPARED_TRAINED_MODEL_NATIVE_REFERENCE_AND_FIXED_TRAIN8',flush=True)


def run():
    plan=read(C/'plan.json');unchanged(plan)
    assert sha256_file(C/'inputs.json')==plan['inputs_sha256'] and sha256_file(C/'CPU-native.pt')==plan['cpu_native_sha256']
    rows=read(C/'inputs.json');prompts=[r['prompt_ids'] for r in rows]
    backend,tok,binding,signature,_=setup_model(SimpleNamespace(model_dir=ASSET,checkpoint=CHECKPOINT,device='xla'))
    assert signature==plan['trained_signature'] and binding['sha256']==CHECKPOINT_SHA
    cpu=torch.load(C/'CPU-native.pt',map_location='cpu',weights_only=True)
    old=GreedyDecoder(backend.model,tok,sync=backend.sync,banned_ids=backend.banned_ids)
    new=BatchedGreedyDecoder(backend.model,tok,sync=backend.sync,banned_ids=backend.banned_ids)
    native={}
    for size in (1,4,8):
        values=[]
        with torch.no_grad():
            for offset in range(0,8,size):
                prefix=prompts[offset:offset+size];x=torch.full((size,544),151643,dtype=torch.long)
                x[:,:len(prefix[0])]=torch.tensor(prefix)
                cache=CausalCache(backend.model.config.num_hidden_layers,672)
                z=(old if size==1 else new)._next(x.to(backend.device),torch.arange(544,device=backend.device),
                    cache,torch.tensor([len(prefix[0])-1],device=backend.device))
                lp=z.log_softmax(-1);backend.sync();values.append(lp.cpu())
        actual=torch.cat(values);torch.save(actual,C/f'XLA-native-B{size}.pt')
        checks=compare_distributions(cpu,actual);native[str(size)]=checks
        atomic_json(C/f'native-B{size}.json',checks)
        assert all(v['passed'] and v['max_logp_error']<=.001 for v in checks)
        print(f'NATIVE_B{size}_ALL8_PASS',flush=True)

    def measure(size):
        backend.sync();started=time.monotonic();output=[]
        for offset in range(0,8,size):
            prefix=prompts[offset:offset+size]
            try:
                if size==1:
                    output.append(asdict(old.decode(prefix[0],max_new_tokens=128,prefill_width=544,
                        pad_id=151643,stop_ids=tuple(plan['stop_ids']),answer_line_stop=False)))
                else:
                    output.extend(asdict(v) for v in new.decode_batch(prefix,max_new_tokens=128,
                        prefill_width=544,pad_id=151643,stop_ids=tuple(plan['stop_ids'])))
            except (GenerationFailure,BatchGenerationFailure) as error:
                partials=getattr(error,'partials',[getattr(error,'partial',None)])
                atomic_json(C/f'failure-B{size}-offset{offset}.json',dict(error=str(error),
                    completed=output,partials=[asdict(v) for v in partials if v is not None]));raise
        backend.sync();return output,time.monotonic()-started

    expected,warm1=measure(1);atomic_json(C/'original-serial-attempts.json',expected)
    natural=sum(v['stop_type'].startswith('token:') for v in expected)
    diverse=len({len(v['actions']) for v in expected})
    warm={'1':warm1}
    for size in (4,8):
        actual,seconds=measure(size);atomic_json(C/f'batched-B{size}-attempts.json',actual)
        assert actual==expected,'Preserve mismatch; no parser/precision/tolerance rescue'
        warm[str(size)]=seconds
    timings={str(s):[] for s in (1,4,8)}
    for order in ((1,4,8),(4,8,1),(8,1,4)):
        for size in order:
            actual,seconds=measure(size);assert actual==expected
            timings[str(size)].append(seconds);atomic_json(C/'timings.json',timings,overwrite=True)
            print(json.dumps(dict(batch=size,completed_repeats=len(timings[str(size)]),seconds=seconds)),flush=True)
    coverage=natural>=6 and diverse>=2
    medians={k:median(v) for k,v in timings.items()}
    atomic_json(C/'worker-result.json',dict(status='PASS_TRAINED_NATURAL_STOP_EXACT_BATCH_PARITY' if coverage
        else 'EXACT_PARITY_BUT_INSUFFICIENT_NATURAL_STOP_COVERAGE',plan_sha256=sha256_file(C/'plan.json'),
        native=native,natural_stops=natural,distinct_action_lengths=diverse,action_lengths=[len(v['actions']) for v in expected],
        warm_seconds=warm,timed_seconds=timings,median_seconds=medians,
        speedup_vs_serial={k:medians['1']/v for k,v in medians.items()},limits=plan['limits']))
    assert coverage,'Exact parity retained but natural-stop integration gate fails'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','run'))
    args=parser.parse_args();(prepare if args.stage=='prepare' else run)()
