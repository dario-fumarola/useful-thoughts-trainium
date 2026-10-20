"""Synthetic score fixtures on exact cohort identities; zero model outcomes.

Checks independent training/inference gates using deliberately known outcomes.
These fixture scores must never be reported as learned performance.
"""
from pathlib import Path
import json,hashlib
from train.permutation_program_science_evaluate import MODEL_IDS,PANELS,cohort,model_modes,sources
from train.permutation_program_readout import aggregate

R=Path('/home/ubuntu/dario');D=R/'local/nvme-study/data/permutation-program-v28-study-v1'
C=R/'local/diagnostics/permutation-program-v28-evaluation-CPU-v1'
def fixture(training_wins):
    results=[]
    for model in sorted({'B0',*MODEL_IDS}):
        arm=None if model=='B0' else model.split('-')[1]
        spec=dict(id=model,arm=arm)
        for panel in PANELS:
            for diagnostic in ([False,True] if panel=='DEV_composition' and arm=='trace' else [False]):
                items=cohort(D,panel,base=model=='B0',diagnostic=diagnostic);data={};summaries={}
                for mode in model_modes(spec,diagnostic):
                    correct=int((arm=='trace' and ((training_wins and mode in ('direct','empty','trace')) or mode=='trace')) or mode=='partial')
                    values=[dict(item_id=v['item_id'],document_id=v['family_id'],family_id=v['family_id'],model=model,panel=panel,mode=mode,
                                 score=correct,status='complete',depth=len(v['question']['program']),program=v['question']['program']) for v in items]
                    data[mode]=values;summaries[mode]=dict(items=len(items),families=len(items)//2,correct=correct*len(items),accuracy=float(correct))
                results.append(dict(status='COMPLETE_ALL_ORIGINAL_ATTEMPTS_SCORED',model=model,panel=panel,diagnostic=diagnostic,
                                    protocol_sha256='EXPLICIT_SYNTHETIC_SCORE_FIXTURE_NOT_A_MODEL_OUTCOME',rows=data,summaries=summaries))
    return results

reports={}
for training_wins in (True,False):
    result=aggregate(fixture(training_wins));expected=dict(training=training_wins,inference=not training_wins)
    assert result['gates']==expected
    assert all(v['items']==18432 and v['documents']==9216 for group in (result['primary_inference'],result['primary_training']) for seed in group.values() for v in seed.values())
    assert len(result['absolute'])==60 # 18 mode cells per panel plus6 partial controls.
    reports[str(training_wins)]=dict(expected_gates=expected,observed_gates=result['gates'],primary_family_count=9216,primary_question_count=18432)
path=C/'readout-structure-fixture.json'
with path.open('x') as f:json.dump(dict(status='PASS_KNOWN_SYNTHETIC_SCORE_FIXTURES_ONLY',model_forwards=0,
    source_files=sources(),checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),fixtures=reports,
    limits=['Artificial all-correct/all-wrong scores test gate independence and exact cohort coverage,not learning.','No held model outcome is generated or inspected.']),f,indent=2)
print('PASS: full-sized paired-cohort fixtures independently separate both claim families; zero model forwards.')
