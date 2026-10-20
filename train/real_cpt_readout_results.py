"""Complete V29 inventories and prospectively declared document-paired gates.

No model execution, outcome-based cohort changes, parser rescue, or seed pooling.
Every original failed generation remains in its denominator. Missing required
jobs or likelihood rows cause an incomplete report rather than a scientific pass.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path

from train.real_cpt_readout import MATH_MODES, SCALAR_MODES, numeric_masks
from train.scalar_completion import parse_initial_scalar_v2
from weekend_thinking.artifacts import atomic_json,sha256_file,utc_now
from weekend_thinking.stats import paired_cluster_bootstrap

SEEDS=(2026102029,2026102030)


def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(s) for s in Path(path).open() if s.strip()]


def paired(reference,treatment,*,name,metric):
    """One fixed mean per unique source document; no repeated-sample inflation."""
    value=paired_cluster_bootstrap(reference,treatment,repeats=10000,seed=2026102029)
    value.update(name=name,metric=metric)
    return value


def gain_pass(value):
    return bool(value['difference_b_minus_a']>=.25 and value['ci95'][0]>0)


def decisions(by_seed):
    if set(by_seed)!=set(map(str,SEEDS)):raise ValueError('Both original orders required; never pool seeds')
    outputs={}
    for seed in map(str,SEEDS):
        c=by_seed[seed]
        recipe=all(gain_pass(c[k]) for k in ('raw_control','format_control'))
        absolute=gain_pass(c['base'])
        retention=c['ordinary_base']['ci95'][1]<=.03
        inference=bool(c['inference_execution_complete'] and all(gain_pass(c[k])
            for k in ('own_raw','own_empty','own_raw_control')))
        outputs[seed]=dict(training_recipe_pass=recipe,absolute_base_pass=absolute,
            retention_target_met=retention,overall_CPT_criterion=recipe and absolute and retention,
            inference_gate_pass=inference)
    return dict(by_order=outputs,
        training_recipe_pass_both_orders=all(v['training_recipe_pass'] for v in outputs.values()),
        absolute_base_pass_both_orders=all(v['absolute_base_pass'] for v in outputs.values()),
        retention_target_met_both_orders=all(v['retention_target_met'] for v in outputs.values()),
        overall_CPT_criterion=all(v['overall_CPT_criterion'] for v in outputs.values()),
        inference_gate_pass_both_orders=all(v['inference_gate_pass'] for v in outputs.values()),
        limits='Covered FineMath calculation-targeted CPT; not general LM improvement, autonomous discovery, or training-seed population inference.')


def checked_job(directory,model,panel,release,release_sha,pins,shard=None):
    directory=Path(directory);run=read(directory/'run.json');closure=read(directory/'closure.json')
    if (run['status']!='COMPLETE_ALL_DECLARED_ATTEMPTS' or run['qualify_only']
            or run['model']!=model or run['panel']!=panel or run['shard']!=(shard if shard is not None else 0)
            or run['release_sha256']!=release_sha or run['source_files']!=release['source_files']
            or closure['status']!='SESSION_AND_PHYSICAL_DEVICE_CLOSED' or closure['exit_code']!=0
            or not closure['owned_session_absent'] or not closure['physical_device_idle']
            or not closure['observed_expected_live_owners'] or closure['physical_chips']!=1
            or any(code!=0 for code in closure['child_exit_codes'])):
        raise ValueError('Complete original fixed-model process and physical closure required')
    reference=Path(run['reference']);ref=read(reference/'reference.json')
    if (sha256_file(reference/'reference.json')!=run['reference_sha256']
            or ref['model']!=model or ref['purpose']!='science'
            or ref['source_files']!=release['source_files']
            or sha256_file(reference/'TRAIN-inputs.json')!=ref['inputs_sha256']
            or sha256_file(reference/'CPU-native.pt')!=ref['native_sha256']):
        raise ValueError('Original actual-model TRAIN reference binding changed')
    for name in ('reference.json','TRAIN-inputs.json','CPU-native.pt'):
        pins[str((reference/name).resolve())]=sha256_file(reference/name)
    for name,digest in run['output_files'].items():
        if sha256_file(directory/name)!=digest:raise ValueError('Original native output bytes changed')
    for f in directory.iterdir():
        if f.is_file():pins[str(f.resolve())]=sha256_file(f)
    return run


def finite_logps(row,target):
    lp=row['token_logps']
    if (row['target_ids']!=target or row['tokens']!=len(target) or len(lp)!=len(target)
            or any(type(v) not in (float,int) or not math.isfinite(v) or v>1e-5 for v in lp)
            or abs(math.fsum(lp)-row['full_logp'])>1e-7
            or row['K1_logp']!=lp[0]):
        raise ValueError('Every original finite native target score and causal target required')
    return lp


def project(values,mode,field):
    return [dict(item_id=r['item_id'],document_id=r['document_id'],score=r[field])
            for r in values if r['mode']==mode]


def summarize(index,output):
    config=read(index);release_path=Path(config['release']);release=read(release_path)
    if sha256_file(release_path)!=config['release_sha256'] or release['status']!='RELEASED_ALL_SEVEN_V29_FIXED_TERMINAL_READOUTS':
        raise ValueError('Original all-seven-terminal readout release required')
    if set(config['models'])!=set(release['models']) or len(config['models'])!=7:
        raise ValueError('All seven models required')
    from transformers import AutoTokenizer
    from train.evaluate import asset_provenance
    base_reference=read(Path(release['references']['B0'])/'reference.json')
    if base_reference['model']!=release['models']['B0'] or asset_provenance(Path(base_reference['model_dir']))!=base_reference['assets']:
        raise ValueError('Original released tokenizer/model assets changed')
    tokenizer=AutoTokenizer.from_pretrained(base_reference['model_dir'],local_files_only=True)
    data=Path(config['data']);manifest=read(data/'manifest.json')
    if sha256_file(data/'manifest.json')!='77c8774d3fdeb283728878c44522dab541ec61a213d3f89ece471b3dcc45fb3f':
        raise ValueError('Original source cohorts required')
    for name in ('DEV_calculation.jsonl','DEV_ordinary.jsonl'):
        if sha256_file(data/name)!=manifest['files'][name]['sha256']:raise ValueError('Original source cohort changed')
    math_data=rows(data/'DEV_calculation.jsonl');ordinary_data=rows(data/'DEV_ordinary.jsonl')
    math_by_id={r['item_id']:r for r in math_data};ordinary_by_id={r['item_id']:r for r in ordinary_data}
    if len(math_by_id)!=768 or len(ordinary_by_id)!=4096:raise ValueError('Complete unique source inventories required')
    pins={str(index.resolve()):sha256_file(index),str(release_path.resolve()):sha256_file(release_path)}
    inventory={};absolute=[];scalar_rows=[]
    for key,jobs in config['models'].items():
        model=release['models'][key]
        for panel in ('writer','math','ordinary'):
            checked_job(jobs[panel],model,panel,release,config['release_sha256'],pins)
        notes=rows(Path(jobs['writer'])/'notes.jsonl')
        if len(notes)!=768 or {v['item_id'] for v in notes}!=set(math_by_id):raise ValueError('Every original writer attempt required')
        math_rows=rows(Path(jobs['math'])/'logps.jsonl');ordinary_rows=rows(Path(jobs['ordinary'])/'logps.jsonl')
        if len(math_rows)!=768*6 or len({(r['item_id'],r['mode']) for r in math_rows})!=768*6:
            raise ValueError('Complete unique all-six-mode likelihood inventory required')
        for row in math_rows:
            if row['item_id'] not in math_by_id or row['mode'] not in MATH_MODES:raise ValueError('Unplanned likelihood identity')
            source=math_by_id[row['item_id']]
            if row['document_id']!=source['document_id']:raise ValueError('Source pairing changed')
            lp=finite_logps(row,source['continuation16_ids']);masks=numeric_masks(source,tokenizer)
            if row['component_masks']!=masks or set(row['component_logps'])!=set(masks):
                raise ValueError('Original native numeral/boundary partition changed')
            for component,mask in masks.items():
                total=math.fsum(v for v,use in zip(lp,mask) if use)
                if abs(total-row['component_logps'][component])>1e-7:
                    raise ValueError('Component score differs from original token subset')
                row['component:'+component]=total
        if len(ordinary_rows)!=4096 or {v['item_id'] for v in ordinary_rows}!=set(ordinary_by_id):
            raise ValueError('Every original ordinary source document required')
        for row in ordinary_rows:
            source=ordinary_by_id[row['item_id']];ids=source['prefix_ids']+source['continuation16_ids']
            lp=finite_logps(row,ids[1:])
            if (row['document_id']!=source['document_id'] or row['context_ids']!=ids[:1] or row['mode']!='raw'
                    or abs(row['mean_NLL']+math.fsum(lp)/527)>1e-9
                    or row['suffix16_logps']!=lp[-16:] or abs(row['suffix16_logp']-math.fsum(lp[-16:]))>1e-8):
                raise ValueError('Full527 ordinary alignment changed')
        generated=[]
        if len(jobs['scalar'])!=4:raise ValueError('All four scalar processes required')
        for shard,path in enumerate(jobs['scalar']):
            checked_job(path,model,'scalar',release,config['release_sha256'],pins,shard)
            values=rows(Path(path)/'answers.jsonl')
            if len(values)!=192*3:raise ValueError('Original scalar shard incomplete')
            expected=[q['item_id'] for q in math_data[shard::4]]
            if [(v['item_id'],v['mode']) for v in values]!=[(i,m) for i in expected for m in SCALAR_MODES]:
                raise ValueError('Original scalar question/mode order changed')
            generated.extend(values)
        for value in generated:
            parsed=parse_initial_scalar_v2(value['text'],value['stop_type'])
            if parsed!=value['parsed']:raise ValueError('Frozen parser result differs from original text')
            good=(value['status']=='ok' and value['source_note_status']=='ok' and parsed['status']=='valid'
                  and Fraction(parsed['value'])==Fraction(math_by_id[value['item_id']]['numeric_value']))
            value['correct']=bool(good)
        inference_ok=all(v['status']=='ok' for v in notes)
        inventory[key]=dict(math=math_rows,ordinary=ordinary_rows,notes=notes,scalar=generated,inference_ok=inference_ok)
        for mode in MATH_MODES:
            values=[r for r in math_rows if r['mode']==mode]
            absolute.append(dict(model=key,mode=mode,documents=768,
                mean_span16_logp=math.fsum(v['full_logp'] for v in values)/768,
                mean_K1_logp=math.fsum(v['K1_logp'] for v in values)/768,
                mean_component_logps={c:math.fsum(v['component_logps'][c] for v in values)/768
                    for c in ('numeral','numeral_and_boundary','boundary_and_whitespace','surrounding')},
                note_execution_errors=sum(v['note_execution_status']!='ok' for v in values)))
        for mode in SCALAR_MODES:
            values=[r for r in generated if r['mode']==mode]
            scalar_rows.append(dict(model=key,mode=mode,attempts=768,correct=sum(v['correct'] for v in values),
                unsupported=sum(v['parsed']['status']=='unsupported' for v in values),
                incomplete=sum(v['parsed']['status']=='incomplete' for v in values),
                format_errors=sum(v['parsed']['status']=='format_error' for v in values),
                execution_errors=sum(v['status']!='ok' or v['source_note_status']!='ok' for v in values),
                source_note_errors=sum(v['source_note_status']!='ok' for v in values),
                parser_errors=sum(v['parsed']['status']=='error' for v in values),
                capped=sum(v['stop_type']=='length_cap' for v in values),
                generated_actions=sum(v['n_actions'] for v in values)))
    contrasts={};all_contrasts=[]
    def comparison(a,b,mode_a='raw',mode_b='raw',field='full_logp',panel='math',name=''):
        metric=('mean_document_NLL_delta' if field=='mean_NLL' else
                'mean_document_span16_logp_gain' if field=='full_logp' else
                'mean_document_'+field.removeprefix('component:')+'_logp_gain')
        result=paired(project(inventory[a][panel],mode_a,field),project(inventory[b][panel],mode_b,field),
                      name=name,metric=metric)
        result.update(reference_model=a,treatment_model=b,reference_mode=mode_a,treatment_mode=mode_b)
        all_contrasts.append(result);return result
    for seed in SEEDS:
        t=f'computation-seed{seed}';r=f'raw-seed{seed}';f=f'format-seed{seed}'
        contrasts[str(seed)]=dict(
            raw_control=comparison(r,t,name=f'{seed}:training-vs-raw'),
            format_control=comparison(f,t,name=f'{seed}:training-vs-format'),
            base=comparison('B0',t,name=f'{seed}:absolute-base'),
            ordinary_base=comparison('B0',t,field='mean_NLL',panel='ordinary',name=f'{seed}:ordinary-base-cost'),
            own_raw=comparison(t,t,mode_b='own',name=f'{seed}:own-raw'),
            own_empty=comparison(t,t,mode_a='empty',mode_b='own',name=f'{seed}:own-empty'),
            own_raw_control=comparison(r,t,mode_b='own',name=f'{seed}:own-raw-control'),
            inference_execution_complete=inventory[t]['inference_ok'])
        for a in (r,f):comparison(a,t,field='mean_NLL',panel='ordinary',name=f'{seed}:ordinary-vs-{a}')
        comparison('B0',t,mode_b='own',name=f'{seed}:secondary-own-absolute-base')
        comparison(f,t,mode_b='own',name=f'{seed}:secondary-own-format-control')
        for a in (r,f,'B0'):
            for component in ('numeral','numeral_and_boundary','boundary_and_whitespace','surrounding'):
                comparison(a,t,field='component:'+component,name=f'{seed}:secondary-{component}-vs-{a}')
        for a,b in (('other','own'),('raw','empty'),('raw','oracle'),('corrupt','oracle')):
            comparison(t,t,mode_a=a,mode_b=b,name=f'{seed}:{b}-minus-{a}')
    report=dict(status='COMPLETE_V29_ALL_FIXED_MODEL_READOUT',utc=utc_now(),
        decision=decisions(contrasts),by_order=contrasts,all_paired_contrasts=all_contrasts,
        absolute_math=absolute,scalar_secondary=scalar_rows,
        ordinary_absolute=[dict(model=k,documents=4096,transitions_per_document=527,
            mean_NLL=math.fsum(r['mean_NLL'] for r in v['ordinary'])/4096) for k,v in inventory.items()],
        writer_costs=[dict(model=k,attempts=768,actions=sum(n['n_actions'] for n in v['notes']),
            capped=sum(n['stop_type']=='length_cap' for n in v['notes']),
            empty=sum(not n['content_ids'] for n in v['notes']),
            errors=sum(n['status']!='ok' for n in v['notes'])) for k,v in inventory.items()],
        source_pins=pins,analysis_source_sha256=sha256_file(__file__),
        bootstrap_source_sha256=sha256_file(Path(__file__).resolve().parents[1]/'weekend_thinking/stats.py'),
        limits=['Two separate data-order results, not a population estimate over training runs.',
            'Supervised prefix-computable traces and source-RHS curation, not autonomous discovery.',
            'Source-matched complete recipes differ in work labels and answer contexts; no pure auxiliary-gradient attribution.',
            'Oracle includes the answer; supplied pauses are not a baseline in this study.',
            'Latency and whole-chip reservation require the separate execution ledger; generated actions alone are not FLOPs.'])
    output.mkdir(parents=True,exist_ok=False);atomic_json(output/'readout.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=summarize(a.index,a.output);print(json.dumps(report['decision'],allow_nan=False))


if __name__=='__main__':main()
