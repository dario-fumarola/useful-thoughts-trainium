"""Teacher selection keeps fixed denominators, order and declared span."""
import json

import pytest
import torch

from train import filter_symbolic_notes as filter_notes
from weekend_thinking.artifacts import sha256_file


def fixture(tmp_path, monkeypatch, *, requested=2):
    # r0's teacher improves the numeral+boundary but worsens the later prose.
    # r1 ties exactly; r2 improves; selection must NOT rank by gain magnitude.
    rows = [dict(item_id=f'r{i}',document_id=f'd{i}',target_ids=[10,11],
                 natural_future16_ids=list(range(10,26)),numeric_token_mask=[True,False]) for i in range(3)]
    hashes = {r['item_id']:{m:f"{r['item_id']}-{m}" for m in filter_notes.MODES} for r in rows}
    p = dict(shards=1,base_signature='base',requested_training_items=requested)
    x = dict(rows=rows,context_hashes=hashes)
    (tmp_path/'protocol.json').write_text('{}')
    d = tmp_path/'scores'/'shard-0'; d.mkdir(parents=True)
    (d/'forward-logps.pt').write_bytes(b'native-qualified-values')
    cells = []
    for i,r in enumerate(rows):
        for mode in filter_notes.MODES:
            lp = [-2.]*16
            if mode == 'teacher' and i in (0,2):
                lp[:2] = [-2.5,-.5] if i==0 else [-.5,-.5]
                if i==0:lp[2:]=[-5.]*14
            cells.append(dict(status='complete',item_id=r['item_id'],document_id=r['document_id'],mode=mode,
                context_sha256=hashes[r['item_id']][mode],target_token_ids=r['natural_future16_ids'],
                token_log_likelihoods=lp,log_likelihood=sum(lp),mean_token_log_likelihood=sum(lp)/16,
                primary_span_tokens=2,primary_span_ll=sum(lp[:2]),numeric_only_ll=lp[0]))
    def write(cells, error=0.):
        (d/'scores.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in cells))
        (d/'run.json').write_text(json.dumps(dict(status='complete',rows=len(cells),signature='base',
            protocol_sha256=sha256_file(tmp_path/'protocol.json'),
            scores_sha256=sha256_file(d/'scores.jsonl'),native_max_full_head_logp_error=error,
            forward_values_sha256=sha256_file(d/'forward-logps.pt'))))
    write(cells);monkeypatch.setattr(filter_notes,'load',lambda output:(p,x))
    return p,x,cells,write


def test_selection_uses_declared_complete_scalar_span_and_original_order(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch)
    result=filter_notes.summarize(tmp_path)
    assert result['positive']==2 and result['natural16_positive']==1
    assert result['selected_item_ids']==['r0','r2']
    decisions=json.loads((tmp_path/'decisions.json').read_text())['rows']
    assert decisions[0]['teacher_minus_empty']>0
    assert decisions[0]['teacher_minus_empty_natural16']<0
    assert decisions[0]['numeric_teacher_minus_raw']<0  # Boundary repair is not numeric benefit.
    assert decisions[1]['teacher_minus_empty']==0 and not decisions[1]['accepted']
    assert result['status']=='complete_filter_training_ready'


def test_training_shortfall_never_creates_a_smaller_training_file(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch,requested=3)
    result=filter_notes.summarize(tmp_path)
    assert result['shortfall']==1 and result['selected_item_ids']==[]
    assert result['attempted']==3 and result['positive']==2
    assert not (tmp_path/'training.jsonl').exists()


@pytest.mark.parametrize('defect',['missing','duplicate','changed_span','native_nan'])
def test_no_complete_result_from_partial_or_corrupt_shards(tmp_path,monkeypatch,defect):
    p,x,cells,write=fixture(tmp_path,monkeypatch)
    if defect=='missing':cells.pop()
    if defect=='duplicate':cells[-1]=cells[0]
    if defect=='changed_span':cells[2]['primary_span_ll']+=1
    write(cells,float('nan') if defect=='native_nan' else 0.)
    with pytest.raises(ValueError):filter_notes.summarize(tmp_path)
    assert not (tmp_path/'summary.json').exists()


@pytest.mark.parametrize('value',[.5,float('nan')])
def test_native_failure_keeps_actual_values_and_json_evidence(tmp_path,value):
    actual=torch.tensor([[value,0.]])
    report=dict(status='started',rows=0)
    with pytest.raises(RuntimeError,match='Native full-head'):
        filter_notes.record_native_check(tmp_path,report,actual,torch.zeros_like(actual))
    saved=torch.load(tmp_path/'forward-logps.pt',weights_only=True)
    torch.testing.assert_close(saved,actual,equal_nan=True)
    receipt=json.loads((tmp_path/'run.json').read_text())
    assert receipt['rows']==0
    assert receipt['native_max_full_head_logp_error']==(.5 if value==.5 else None)
    assert receipt['forward_values_sha256']==sha256_file(tmp_path/'forward-logps.pt')
