"""One complete readout after ALL nine fixed models/panels physically close."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import gzip
import hashlib
import json
import math
import os
import shutil

from train.real_trace_answers import ROOT, MODES, classify, sources as scalar_sources
from weekend_thinking.stats import paired_cluster_bootstrap

R = Path('/home/ubuntu/dario')
W = R/'local/worktrees/real-target-eval-v15'
assert ROOT == W
C = R/'local/diagnostics/real-target-eval-v15-study-v1'
read = lambda p:json.loads(Path(p).read_text())
rows = lambda p:[json.loads(v) for v in Path(p).read_text().splitlines()]
sha = lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan = read(C/'plan.json')
assert len(plan['models']) == 9
assert read(C/'score-queue-completion.json')['status']=='ALL_NINE_FIXED_NATIVE_SCORE_JOBS_CLOSED'
# Check completeness before opening ANY scalar or likelihood outcome values.
registries = {}
for item in plan['models']:
    registry = read(item['registry'])
    assert registry['model_id']==item['id']
    control = Path(registry['control'])
    assert read(control/'completion.json')['status']=='COMPLETE_FIXED_V15_MODEL_NOTES_AND_SCALAR'
    assert read(C/'scores'/item['id']/'completion.json')['status']=='BOTH_FIXED_NATIVE_LIKELIHOOD_PANELS_COMPLETE'
    registries[item['id']] = registry
S = R/'local/diagnostics/real-target-sft-v15-study-v1'
assert read(S/'terminal-pipeline-completion.json')['status']=='ALL_FOUR_MODELS_AUDITED_PROTECTED_AND_FROZEN'
inputs = Path(next(iter(registries.values()))['inputs'])
assert sha(inputs/'manifest.json')==plan['input_manifest_sha256']
questions = rows(inputs/'DEV_fresh384-questions.jsonl')
assert len(questions)==len({v['document_id'] for v in questions})==384
ids = [v['item_id'] for v in questions]
gold_path = Path(next(iter(registries.values()))['gold'])
gold = {v['item_id']:v['numeric_value'] for v in rows(gold_path)}
assert set(gold)==set(ids)

public = R/'validation/real_target_sft_v15_complete_20261019'
public.mkdir(exist_ok=False)
A = R/'local/backups/evaluation/real-target-sft-v15-complete'
A.mkdir(exist_ok=False)
payloads, archived, receipts = {},{},{}


def protect(source, destination):
    if source.is_dir():
        for file in sorted(source.rglob('*')):
            if file.is_file():
                protect(file,destination/file.relative_to(source))
        return
    destination.parent.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(R).free > 35*2**30+source.stat().st_size
    shutil.copyfile(source,destination)
    with destination.open('rb') as stream:
        os.fsync(stream.fileno())
    assert sha(source)==sha(destination)
    archived[str(destination.relative_to(A))]=dict(bytes=destination.stat().st_size,sha256=sha(destination))


def payload(name, values):
    file = public/name
    file.write_bytes(gzip.compress((''.join(json.dumps(v,separators=(',',':'))+'\n' for v in values)).encode(),mtime=0))
    payloads[str(file.relative_to(R))]=dict(rows=len(values),sha256=sha(file))


for name in ['plan.json','baseline-release.json','new-model-release.json','score-queue-completion.json','score-queue-state.json']:
    protect(C/name,A/'study-control'/name)
protect(inputs,A/'fixed-native-inputs')
protect(gold_path.parent,A/'scoring-only-gold')
scalar, likelihood, ordinary, tables, absolute, note_costs = {},{},{},{},{},{}
execution_errors = 0
for model, registry in registries.items():
    control, note_root, answer_root = [Path(registry[k]) for k in ['control','notes','scalar']]
    summary, p = read(answer_root/'summary.json'),read(answer_root/'protocol.json')
    assert p['source_files']==scalar_sources() and p['items']==384
    assert p['ordered_item_ids']==ids and p['gold_sha256']==sha(gold_path)
    assert summary['rows']==1920 and summary['model_id']==model
    assert summary['protocol_sha256']==sha(answer_root/'protocol.json')
    assert sha(answer_root/'scored.jsonl')==summary['scored_sha256']
    values = rows(answer_root/'scored.jsonl')
    assert len(values)==len({(v['item_id'],v['mode']) for v in values})==1920
    scalar[model],tables[model] = {},{}
    for value in values:
        actual=classify(value,gold[value['item_id']])
        assert all(value[k]==v for k,v in actual.items())
    for mode in MODES:
        selected=[v for v in values if v['mode']==mode]
        assert len(selected)==len({v['document_id'] for v in selected})==384
        assert {v['item_id'] for v in selected}==set(ids)
        table=dict(documents=384,correct=sum(v['correct'] for v in selected),
            categories=dict(Counter(v['category'] for v in selected)),
            capped=sum(v['stop_type']=='length_cap' for v in selected),
            generated_actions=sum(v['n_actions'] for v in selected),
            wall_seconds_sum=sum(v['wall_seconds'] for v in selected))
        assert table==summary['modes'][mode]
        tables[model][mode]=table
        execution_errors+=table['categories'].get('execution_error',0)
        scalar[model][mode]=[dict(item_id=v['item_id'],document_id=v['document_id'],score=int(v['correct'])) for v in selected]
    payload(f'{model}-all1920-scalar-continuations.jsonl.gz',
            [{**{k:v for k,v in item.items() if k!='prompt_ids'},'expected_numeric_value':gold[item['item_id']]} for item in values])
    note_protocol = read(note_root/'protocol.json')
    bundle=read(note_root/'notes-bundle.json')
    assert bundle['model_id']==model and bundle['ordered_item_ids']==ids
    for file,digest in bundle['source_pins'].items():
        assert sha(file)==digest
    notes=[v for file in bundle['source_pins'] if Path(file).name=='notes.jsonl' for v in rows(file)]
    assert len(notes)==len({v['item_id'] for v in notes})==384
    note_costs[model]=dict(documents=384,generated_actions=sum(v['n_actions'] for v in notes),
        mean_actions=sum(v['n_actions'] for v in notes)/384,
        capped=sum(v['stop_type']=='length_cap' for v in notes),empty=sum(not v['content_ids'] for v in notes),
        errors=sum(v['status']!='ok' for v in notes),stop_types=dict(Counter(v['stop_type'] for v in notes)),
        generation_wall_seconds_sum=sum(v['wall_seconds'] for v in notes),
        forced_reader_closing_native_tokens_per_document=2,
        limits='Generation wall sums are measured in concurrent FP32 worker jobs;not controlled serving latency. Given closing tokens are separate from sampled actions.')
    payload(f'{model}-all384-original-notes.jsonl.gz',[{k:v for k,v in note.items() if k!='prompt_ids'} for note in notes])
    absolute[model]={}
    likelihood[model],ordinary[model] = {},{}
    score_receipts={}
    for panel, expected in [('DEV_fresh384',1920),('DEV_ordinary256',256)]:
        folder=Path(registry['score_output_root'])/panel
        result=read(folder/'run.json')
        assert result['status']=='COMPLETE_ALL_FIXED_DOCUMENT_LIKELIHOODS' and result['rows']==expected
        assert result['model_id']==model and result['manifest_sha256']==plan['input_manifest_sha256']
        assert sha(folder/'logps.jsonl')==result['logps_sha256']
        assert result['fixed_first4_per_mode_parity']['passed']
        for file,digest in result['source_files'].items():
            assert sha(W/file)==digest
        logps=rows(folder/'logps.jsonl')
        assert len(logps)==len({(v['item_id'],v['mode']) for v in logps})==expected
        for value in logps:
            assert len(value['logps16'])==16 and all(math.isfinite(v) for v in value['logps16'])
            assert abs(math.fsum(value['logps16'])-value['span16_logprob'])<1e-8
            if panel=='DEV_fresh384':
                assert value['component_status']=='exact_native_partition'
                assert abs(sum(value['component_logprobs'].values())-value['span16_logprob'])<1e-8
        absolute[model][panel]={}
        for mode in (MODES if panel=='DEV_fresh384' else ('raw',)):
            selected=[v for v in logps if v['mode']==mode]
            n=384 if panel=='DEV_fresh384' else 256
            assert len(selected)==len({v['document_id'] for v in selected})==n
            absolute[model][panel][mode]=dict(documents=n,
                mean_span16_logprob=math.fsum(v['span16_logprob'] for v in selected)/n,
                mean_token_NLL=-math.fsum(v['span16_logprob'] for v in selected)/n/16,
                generation_error_rows=sum(v['note_execution_status']!='ok' for v in selected))
            (likelihood[model] if panel=='DEV_fresh384' else ordinary[model])[mode]=selected
        payload(f'{model}-{panel}-all{expected}-logps.jsonl.gz',logps)
        score_receipts[panel]=dict(result_sha256=sha(folder/'run.json'),logps_sha256=result['logps_sha256'],
            compute=result['compute'],wall_seconds=result['wall_seconds'],
            CPU_reference_including_load_seconds=result['CPU_reference_including_load_seconds'],
            fixed_first4_per_mode_parity=result['fixed_first4_per_mode_parity'])
        protect(folder,A/model/'likelihood'/panel)
    receipts[model]=dict(model_binding=p['model'],scalar_protocol_sha256=sha(answer_root/'protocol.json'),
        scalar_summary_sha256=sha(answer_root/'summary.json'),original_scored_sha256=sha(answer_root/'scored.jsonl'),
        note_protocol_sha256=sha(note_root/'protocol.json'),original_note_bundle_sha256=sha(note_root/'notes-bundle.json'),
        scores=score_receipts)
    protect(control,A/model/'control')
    protect(note_root,A/model/'notes')
    protect(answer_root,A/model/'scalar')
    protect(C/'scores'/model,A/model/'score-control')


def compare(a,b,am='raw',bm='raw',metric='accuracy',part=None):
    if metric=='accuracy':
        first,second=scalar[a][am],scalar[b][bm]
    else:
        source=ordinary if metric=='ordinary_NLL' else likelihood
        def project(values):
            return [dict(item_id=v['item_id'],document_id=v['document_id'],
                score=-v['span16_logprob']/16 if metric=='ordinary_NLL' else
                    v['component_logprobs'][part] if part else v['span16_logprob']) for v in values]
        first,second=project(source[a][am]),project(source[b][bm])
    return paired_cluster_bootstrap(first,second,repeats=20000,seed=2026101917)


seeds={}
for seed in ['2026101913','2026101914']:
    direct,cot,raw,prior=[f'{arm}-seed{seed}-step320' for arm in ['target_direct','target_cot','real_raw','real_prior']]
    pairs={
        'cot_own_minus_direct_raw':(direct,cot,'raw','own'),
        'cot_own_minus_cot_empty':(cot,cot,'empty','own'),
        'cot_own_minus_cot_raw':(cot,cot,'raw','own'),
        'cot_own_minus_cot_other':(cot,cot,'other','own'),
        'cot_own_minus_B0raw':('B0',cot,'raw','own'),
        'direct_raw_minus_B0raw':('B0',direct,'raw','raw'),
        'direct_raw_minus_CPTraw':(raw,direct,'raw','raw'),
        'cot_own_minus_CPTprior_own':(prior,cot,'own','own'),
        'CPTprior_own_minus_CPTraw_raw':(raw,prior,'raw','own'),
    }
    accuracies={k:compare(a,b,am,bm) for k,(a,b,am,bm) in pairs.items()}
    full16={k:compare(a,b,am,bm,metric='LL') for k,(a,b,am,bm) in pairs.items()}
    components={part:compare(direct,cot,'raw','own',metric='LL',part=part)
                for part in ['leading','numeric','closing','remainder']}
    quality={
        'cot_minus_direct':compare(direct,cot,metric='ordinary_NLL'),
        'cot_minus_B0':compare('B0',cot,metric='ordinary_NLL'),
        'direct_minus_B0':compare('B0',direct,metric='ordinary_NLL')}
    primary=accuracies['cot_own_minus_direct_raw']
    passed=(primary['difference_b_minus_a']>=.03 and primary['ci95'][0]>0
            and accuracies['cot_own_minus_cot_empty']['ci95'][0]>0
            and accuracies['cot_own_minus_B0raw']['ci95'][0]>0 and execution_errors==0)
    seeds[seed]=dict(accuracy_contrasts=accuracies,secondary_full16_contrasts=full16,
        prospective_component_contrasts=components,ordinary_token_NLL_regression=quality,
        declared_primary_gate_pass=bool(passed))

meta=rows(inputs/'DEV_fresh384-targets.jsonl')
held=rows(R/'local/nvme-study/data/real-target-sft-v15-held-v1/DEV_fresh384.jsonl')
annotations={v['item_id']:v for v in held}
payload('fixed-source-target-metadata.jsonl.gz',[dict(v,expected_numeric_value=gold[v['item_id']],
    normalized_expression=annotations[v['item_id']]['normalized_expression'],
    numeric_surface_text=annotations[v['item_id']]['numeric_surface_text'],
    teacher_full_calculation=annotations[v['item_id']]['full_calculation_text']) for v in meta])
protect(Path(__file__),A/'analysis-source'/Path(__file__).name)
archive=dict(status='ALL_NINE_FIXED_ORIGINALS_READBACK_VERIFIED_FSYNCED',utc=datetime.now(timezone.utc).isoformat(),files=archived)
(A/'ARCHIVE.json').write_text(json.dumps(archive,indent=2))
with (A/'ARCHIVE.json').open('rb') as stream:
    os.fsync(stream.fileno())
result=dict(status='COMPLETE_NINE_MODEL_TWO_SEED_TARGET_SFT_REFERENCE',utc=datetime.now(timezone.utc).isoformat(),
    source_commit='41a5bfd',training_source_commit='829d9a2',design_sha256=plan['design_sha256'],
    primary_documents=384,ordinary_documents=256,scalar_attempts=17280,original_notes=3456,likelihood_rows=19584,
    tables=tables,absolute_likelihood=absolute,note_costs=note_costs,source_receipts=receipts,seeds=seeds,
    decisions=dict(practical_reference_pass_both_seeds=all(v['declared_primary_gate_pass'] for v in seeds.values()),
                   execution_error_rows_all_fixed_scalar_attempts=execution_errors),
    payloads=payloads,archive_sha256=sha(A/'ARCHIVE.json'),analysis_source_sha256=sha(__file__),
    limits=['New sourceTRAIN-derived development,not officialTEST or pretraining-novelty evidence. Exact ordered value/operator ASTs excludeTRAINandearlier arithmetic panels;patterns can recur.',
            'Scalarprimary/frozen3pp gate and controls unchanged. All failures retained;zeroexecutionerror guard is conservatively checked across all fixed attempts.',
            '20k paired document bootstraps withseed2026101917,unadjusted checkpoint-conditional intervals;not training-seed uncertainty;seeds never pooled.',
            'No automatic extension,retry,checkpoint selection or new inference model. Supplied oracle contains root answer;own writer sees prefix only.',
            'TargetCoT receives trace supervision onbothpasses;CPTprior received one plain and one augmented occurrence. Different active target totals and inference budgets are explicit costs.',
            'A supervised arithmetic reference is not thought-augmented pretraining benefit,emergence,broad reasoning or superiority to an optimized/few-shot base CoT prompt.',
            'Native full16 uses same-model first4 CPU references per mode;not all-panel CPU parity for each model. No controlled end-to-end service latency or RL-step claim.'])
public.with_suffix('.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(status=result['status'],decisions=result['decisions'],seeds=result['seeds']),indent=2),flush=True)
