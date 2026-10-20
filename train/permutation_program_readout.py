"""Offline strict coverage, complete-node scores and separate trace diagnostics."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
from statistics import median

from train.permutation_program_answers import score_answer,execution_diagnostic
from train.permutation_program_data import question_from_dict
from train.permutation_program_science_evaluate import SCHEMA,sources,MODEL_IDS,PANELS,model_modes
from weekend_thinking.artifacts import sha256_file,atomic_json
from weekend_thinking.stats import paired_cluster_bootstrap


def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]


def score(output):
    output=Path(output);p=read(output/'protocol.json')
    if p['schema']!=SCHEMA or p['source_files']!=sources() or sha256_file(output/'gold.jsonl')!=p['gold_file_sha256']:
        raise ValueError('Frozen evaluation source and offline gold required')
    gold=rows(output/'gold.jsonl');lookup={v['item_id']:v for v in gold}
    if len(lookup)!=p['items'] or [v['item_id'] for v in gold]!=p['ordered_item_ids']:
        raise ValueError('Complete unique original gold cohort required')
    families=Counter(v['family_id'] for v in gold)
    if set(families.values())!={2}:raise ValueError('Both original counterfactual variants required')
    # Independent dictionary interpreter, not the producer's execute helper.
    for g in gold:
        q=g['question'];node=q['start']
        for op in q['program']:node=q[op][node]
        if g['answer']!='ABCDEFGHIJKLMNOP'[node]:raise ValueError('Offline gold contradicts available program')
    by_mode={mode:{} for mode in p['modes']}
    for shard in range(p['shards']):
        d=output/'answers'/f'shard-{shard}';run=read(d/'run.json')
        if (run['status']!='complete_all_attempts' or run['protocol_sha256']!=sha256_file(output/'protocol.json')
                or run['attempts_sha256']!=sha256_file(d/'attempts.jsonl')):
            raise ValueError('Every original shard must close before scientific scoring')
        attempts=rows(d/'attempts.jsonl')
        expected={(mode,p['ordered_item_ids'][i]) for mode in p['modes'] for i in range(shard,p['items'],p['shards'])}
        if len(attempts)!=len(expected) or {(v['mode'],v['item_id']) for v in attempts}!=expected:
            raise ValueError('No missing/duplicate/foreign generation attempts may leave denominators')
        for value in attempts:
            mode=value['mode'];item=value['item_id'];g=lookup[item]
            if item in by_mode[mode]:raise ValueError('Duplicate original item/mode')
            result=score_answer(value,g['answer'],mode)
            row=dict(item_id=item,document_id=g['family_id'],family_id=g['family_id'],
                model=p['model']['id'],panel=p['panel'],mode=mode,
                depth=len(g['question']['program']),program=g['question']['program'],
                expected=g['answer'],**result,n_actions=value['n_actions'],
                supplied_positions=(6*len(g['question']['program']) if mode=='pause' else 0),
                prompt_tokens=value['prompt_tokens'],wall_seconds=value['wall_seconds'],stop_type=value['stop_type'])
            if mode=='trace':row['execution']=execution_diagnostic(value['text'],question_from_dict(g['question']))
            by_mode[mode][item]=row
    summaries={}
    for mode,values in by_mode.items():
        if set(values)!=set(lookup):raise ValueError('Complete common paired item set required')
        data=[values[item] for item in p['ordered_item_ids']];paired=defaultdict(list)
        for value in data:paired[value['family_id']].append(value['score'])
        both=sum(int(all(pair)) for pair in paired.values())
        summaries[mode]=dict(items=len(data),families=len(families),correct=sum(v['score'] for v in data),
            accuracy=sum(v['score'] for v in data)/len(data),both_variants_correct=both,
            statuses=dict(Counter(v['status'] for v in data)),stop_types=dict(Counter(v['stop_type'] for v in data)),
            generated_actions=sum(v['n_actions'] for v in data),supplied_pause_positions=sum(v['supplied_positions'] for v in data),
            prompt_tokens=sum(v['prompt_tokens'] for v in data),
            measured_wall_seconds=sum(v['wall_seconds'] for v in data),
            median_row_wall_seconds=median(v['wall_seconds'] for v in data))
    return dict(status='COMPLETE_ALL_ORIGINAL_ATTEMPTS_SCORED',model=p['model']['id'],panel=p['panel'],diagnostic=p.get('diagnostic',False),
                protocol_sha256=sha256_file(output/'protocol.json'),summaries=summaries,
                rows={mode:[values[item] for item in p['ordered_item_ids']] for mode,values in by_mode.items()},
                limits=['Errors,caps and unsupported fields remain scored0.','Paired variants share one family; resample families for confidence intervals.','Trace syntax/execution never changes the complete-answer score.','Measured local timings under concurrent jobs are not a controlled serving benchmark.'])


def contrast(control,treatment):
    """Strict pairing and whole-family uncertainty, with absolute fix/harm counts."""
    estimate=paired_cluster_bootstrap(control,treatment,repeats=5000,seed=17)
    a={v['item_id']:v for v in control};b={v['item_id']:v for v in treatment}
    if (any(v['score'] not in (0,1) for v in [*control,*treatment])
            or any(v!=2 for v in Counter(row['document_id'] for row in control).values())):
        raise ValueError('Binary complete answers and two original variants per family required')
    return dict(**estimate,fixed=sum(a[k]['score']==0 and b[k]['score']==1 for k in a),
        harmed=sum(a[k]['score']==1 and b[k]['score']==0 for k in a),
        control_correct=sum(v['score'] for v in control),treatment_correct=sum(v['score'] for v in treatment),
        meets_practical_and_interval_rule=estimate['difference_b_minus_a']>=.03 and estimate['ci95'][0]>0)


def descriptive(rows,field):
    grouped=defaultdict(list)
    for row in rows:grouped[row[field]].append(row)
    return {str(key):dict(items=len(values),correct=sum(v['score'] for v in values),
        accuracy=sum(v['score'] for v in values)/len(values),
        statuses=dict(Counter(v['status'] for v in values)),
        complete_correct_execution=sum(v.get('execution',{}).get('complete_correct_execution',False) for v in values)
            if values[0]['mode']=='trace' else None) for key,values in sorted(grouped.items())}


def aggregate(results):
    """Read the complete fixed study once; never choose a seed or panel by score."""
    expected={(model,panel,False) for model in {'B0',*MODEL_IDS} for panel in PANELS}
    expected|={(model,'DEV_composition',True) for model in MODEL_IDS if '-trace-' in model}
    lookup={}
    for result in results:
        key=(result['model'],result['panel'],result['diagnostic'])
        if key in lookup or result['status']!='COMPLETE_ALL_ORIGINAL_ATTEMPTS_SCORED':
            raise ValueError('Exactly one complete original result per fixed model/panel/condition required')
        lookup[key]=result
    if set(lookup)!=expected:raise ValueError('All six models, all three panels, B0 and both fixed partial-control panels must close')
    for (model,panel,diagnostic),result in lookup.items():
        spec=dict(id=model,arm=model.split('-')[1] if model!='B0' else None)
        modes=model_modes(spec,diagnostic)
        n=1024 if diagnostic else 2048 if model=='B0' or panel!='DEV_composition' else 18432
        if set(result['rows'])!=set(modes):raise ValueError('Fixed model-specific modes required')
        common=None
        for mode,values in result['rows'].items():
            ids=[v['item_id'] for v in values]
            if (len(values)!=n or len(set(ids))!=n
                    or any(v['model']!=model or v['panel']!=panel or v['mode']!=mode for v in values)
                    or set(Counter(v['document_id'] for v in values).values())!={2}):
                raise ValueError('Complete original paired cohort and row lineage required')
            if common is not None and ids!=common:raise ValueError('All modes require the same ordered questions')
            common=ids
    def cell(model,panel,mode):return lookup[model,panel,False]['rows'][mode]
    inference={};training={};secondary={}
    for seed in (2026102028,2026102029):
        d=f'program-direct-seed{seed}-v1';t=f'program-trace-seed{seed}-v1';p=f'program-pause-seed{seed}-v1'
        inference[str(seed)]={};training[str(seed)]={}
        for panel in PANELS:
            pairs={'trace_minus_direct_model':((d,'direct'),(t,'trace')),
                   'trace_minus_trained_pause':((p,'pause'),(t,'trace')),
                   'trace_minus_same_direct':((t,'direct'),(t,'trace')),
                   'trained_direct_minus_direct_model':((d,'direct'),(t,'direct')),
                   'trained_direct_minus_pause_direct':((p,'direct'),(t,'direct')),
                   'trace_minus_same_empty':((t,'empty'),(t,'trace')),
                   'direct_empty_minus_direct':((d,'direct'),(d,'empty'))}
            comparisons={name:contrast(cell(a[0],panel,a[1]),cell(b[0],panel,b[1])) for name,(a,b) in pairs.items()}
            secondary[f'{seed}/{panel}']=comparisons
            if panel=='DEV_composition':
                inference[str(seed)]={k:v for k,v in comparisons.items() if k in ('trace_minus_direct_model','trace_minus_trained_pause','trace_minus_same_direct')}
                training[str(seed)]={k:v for k,v in comparisons.items() if k.startswith('trained_direct_')}
    gates=dict(inference=all(v['meets_practical_and_interval_rule'] for seed in inference.values() for v in seed.values()),
               training=all(v['meets_practical_and_interval_rule'] for seed in training.values() for v in seed.values()))
    absolute=[];strata={};diagnostics={}
    for key,result in sorted(lookup.items()):
        model,panel,diagnostic=key
        for mode,summary in result['summaries'].items():
            absolute.append(dict(model=model,panel=panel,diagnostic=diagnostic,mode=mode,**summary))
            strata[f'{model}/{panel}/{mode}']=dict(depth=descriptive(result['rows'][mode],'depth'),program=descriptive(result['rows'][mode],'program'))
        if diagnostic:
            diagnostics[model]={name:contrast(result['rows'][control],result['rows']['partial']) for name,control in [('partial_minus_corrupt','corrupt'),('partial_minus_donor','donor')]}
            subset={v['item_id'] for v in result['rows']['partial']}
            for mode in ('direct','empty','trace'):
                original=[v for v in cell(model,panel,mode) if v['item_id'] in subset]
                diagnostics[model]['partial_minus_'+mode]=contrast(original,result['rows']['partial'])
    return dict(status='COMPLETE_FIXED_V28_ALL_TERMINAL_READOUT',gates=gates,primary_inference=inference,
        primary_training=training,all_panel_contrasts=secondary,absolute=absolute,strata=strata,partial_controls=diagnostics,
        source_files=sources(),input_protocols={f'{k[0]}/{k[1]}/{k[2]}':v['protocol_sha256'] for k,v in lookup.items()},
        limits=['Primary gates use composition only; interpolation and length are distinct secondary endpoints.',
            'Training and inference claims are evaluated independently. No averaging of two training orders into a new gate.',
            'All intervals resample two-variant families and are conditional on the checkpoints; no broad seed uncertainty.',
            'Fixed B0 demonstration reference is not an optimized or primary matched-training baseline.',
            'Complete training recipes differ in targets and some answer-training contexts; no pure gradient, internalization or autonomous-discovery claim.',
            'Provided pause positions and generated actions have distinct execution costs; local summed row times are not reserved chip-hours.',
            'A failed gate does not prove zero effect. This supervised laboratory does not establish general pretraining improvement.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path,required=True,help='JSON list of the23 fixed prepared output directories')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();paths=read(args.index)
    if type(paths) is not list or len(paths)!=23 or len(set(paths))!=23:parser.error('Exactly23 distinct original execution directories required')
    results=[score(path) for path in paths]
    atomic_json(args.output,aggregate(results))


if __name__=='__main__':main()
