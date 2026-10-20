"""Same-question B0 reference tables after complete fixed evaluation.

These secondary comparisons do not alter the frozen primary gates. In particular,
the18432-item composition result is never compared unpaired with B0's2048 items.
"""
from pathlib import Path
import argparse,json
from train.permutation_program_readout import score,contrast
from train.permutation_program_science_evaluate import PANELS
from weekend_thinking.artifacts import atomic_json,sha256_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('index','primary-readout','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();primary=json.loads(args.primary_readout.read_text());paths=json.loads(args.index.read_text())
    assert primary['status']=='COMPLETE_FIXED_V28_ALL_TERMINAL_READOUT' and len(paths)==len(set(paths))==23
    results=[score(path) for path in paths]
    identities={f"{v['model']}/{v['panel']}/{v['diagnostic']}":v['protocol_sha256'] for v in results}
    assert len(identities)==23 and identities==primary['input_protocols']
    lookup={(v['model'],v['panel']):v for v in results if not v['diagnostic']}
    tables=[];comparisons={};base_within={};additional_controls={}
    for panel in PANELS:
        base=lookup['B0',panel];ids={v['item_id'] for v in base['rows']['trace']}
        assert len(ids)==2048
        base_within[panel]=contrast(base['rows']['direct'],base['rows']['trace'])
        base_within[panel].pop('meets_practical_and_interval_rule')
        for (model,p),result in sorted(lookup.items()):
            if p!=panel:continue
            for mode,rows in result['rows'].items():
                common=[v for v in rows if v['item_id'] in ids]
                assert len(common)==2048 and {v['item_id'] for v in common}==ids
                tables.append(dict(panel=panel,model=model,mode=mode,items=2048,families=1024,
                    correct=sum(v['score'] for v in common),accuracy=sum(v['score'] for v in common)/2048,
                    statuses={s:sum(v['status']==s for v in common) for s in sorted({v['status'] for v in common})}))
        for seed in (2026102028,2026102029):
            for arm,mode in [('direct','direct'),('trace','direct'),('trace','trace'),('pause','pause')]:
                model=f'program-{arm}-seed{seed}-v1'
                common=[v for v in lookup[model,panel]['rows'][mode] if v['item_id'] in ids]
                comparisons[f'{panel}/{model}/{mode}_minus_B0_trace']=contrast(base['rows']['trace'],common)
                comparisons[f'{panel}/{model}/{mode}_minus_B0_trace'].pop('meets_practical_and_interval_rule')
            d=lookup[f'program-direct-seed{seed}-v1',panel]['rows']
            t=lookup[f'program-trace-seed{seed}-v1',panel]['rows']
            p=lookup[f'program-pause-seed{seed}-v1',panel]['rows']
            for name,control,treatment in [('trace_minus_direct_empty',d['empty'],t['trace']),
                                          ('trace_minus_pause_direct',p['direct'],t['trace']),
                                          ('trained_direct_minus_direct_empty',d['empty'],t['direct'])]:
                value=contrast(control,treatment);value.pop('meets_practical_and_interval_rule')
                additional_controls[f'{panel}/seed{seed}/{name}']=value
    atomic_json(args.output,dict(status='COMPLETE_SAME_QUESTION_SECONDARY_B0_REFERENCES',
        primary_readout_sha256=sha256_file(args.primary_readout),index_sha256=sha256_file(args.index),
        source_sha256=sha256_file(__file__),matched_absolute=tables,
        checkpoint_conditional_secondary_contrasts=comparisons,B0_trace_minus_B0_direct=base_within,
        additional_same_question_control_contrasts=additional_controls,
        limits=['All base comparisons use the same2048 original questions/1024 paired families per panel.',
            'No primary gate uses B0 or these secondary contrasts. No significance-based selection of comparisons.',
            'B0 uses one fixed TRAIN depth1 demonstration; it is not an optimized pretrained-capability ceiling.',
            'B0 and trained paths use different declared task-format contexts. This is a practical reference,not a pure training intervention.',
            'Full primary composition results remain separately reported on9216 families; no convenient pooling of panels or seeds.']))


if __name__=='__main__':main()
