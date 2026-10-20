import json
from pathlib import Path
import pytest
from train.permutation_program_readout import score,contrast,aggregate
from train.permutation_program_science_evaluate import SCHEMA,sources
from weekend_thinking.artifacts import sha256_file


def fixture(root):
    ids=[f'family{i//2}:{i%2}' for i in range(4)];r=list(range(1,16))+[0]
    gold=[dict(item_id=item,family_id=f'family{i//2}',question=dict(r=r,s=list(range(16)),start=i,program='r'),answer='BCDE'[i]) for i,item in enumerate(ids)]
    (root/'gold.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in gold))
    p=dict(schema=SCHEMA,source_files=sources(),gold_file_sha256=sha256_file(root/'gold.jsonl'),items=4,
           ordered_item_ids=ids,modes=['direct'],shards=4,model=dict(id='fixture'),panel='CPU-fixture')
    (root/'protocol.json').write_text(json.dumps(p))
    outcomes=[('B\n','ok','qa_answer_line'),('C','ok','length_cap'),('D\n','error','error'),('Wrong\n','ok','qa_answer_line')]
    for shard,(text,status,stop) in enumerate(outcomes):
        d=root/'answers'/f'shard-{shard}';d.mkdir(parents=True)
        v=dict(item_id=ids[shard],mode='direct',text=text,status=status,stop_type=stop,n_actions=2,prompt_tokens=90,wall_seconds=.1)
        (d/'attempts.jsonl').write_text(json.dumps(v)+'\n')
        (d/'run.json').write_text(json.dumps(dict(status='complete_all_attempts',protocol_sha256=sha256_file(root/'protocol.json'),attempts_sha256=sha256_file(d/'attempts.jsonl'))))


def test_caps_errors_and_format_failures_stay_in_full_paired_denominator(tmp_path):
    fixture(tmp_path);r=score(tmp_path);v=r['summaries']['direct']
    assert v['items']==4 and v['families']==2 and v['correct']==1 and v['accuracy']==.25
    assert v['both_variants_correct']==0
    assert v['statuses']==dict(complete=1,capped=1,error=1,unsupported=1)


def test_incomplete_shards_cannot_be_reported_as_completed_science(tmp_path):
    fixture(tmp_path);path=tmp_path/'answers/shard-2/run.json';v=json.loads(path.read_text());v['status']='failed';path.write_text(json.dumps(v))
    with pytest.raises(ValueError):score(tmp_path)


def test_uncertainty_resamples_whole_counterfactual_families():
    a=[dict(item_id=str(i),document_id=str(i//2),score=int(i>=2)) for i in range(4)]
    b=[dict(v,score=1-v['score']) for v in a]
    result=contrast(a,b)
    assert result['items']==4 and result['documents']==2
    assert result['fixed']==2 and result['harmed']==2
    assert result['ci95']==[-1.,1.]
    assert not result['meets_practical_and_interval_rule']
    with pytest.raises(ValueError):contrast(a,b[:-1])


def test_no_partial_study_can_receive_a_primary_gate():
    with pytest.raises(ValueError,match='All six models'):
        aggregate([])
