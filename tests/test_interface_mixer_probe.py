"""CPU semantics of the full-logit grid and TRAIN-only constrained fit."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from train import interface_mixer_probe as m


def test_full_vocabulary_mixture_matches_dense_logits_not_average_token_logps():
    raw=torch.tensor([[2.,-3.,1.,.5],[1.,2.,-1.,0.]])
    note=torch.tensor([[-2.,3.,.5,1.],[0.,-.5,3.,1.]])
    target=torch.tensor([[0,2]])
    result,raw_lp,note_lp=m.mixed_token_logps(raw,note,target,m.GRID)
    expected=torch.cat([((1-w)*raw+w*note).log_softmax(-1).gather(-1,target.T).T for w in m.GRID])
    torch.testing.assert_close(result,expected,atol=5e-7,rtol=1e-6)
    assert result[0].numpy().tobytes()==raw_lp[0].numpy().tobytes()
    torch.testing.assert_close(result[-1],note_lp[0],atol=5e-7,rtol=1e-6)
    wrong=.5*(raw_lp+note_lp)
    assert not torch.allclose(result[m.GRID.index(.5)],wrong[0],atol=1e-3)


def test_zero_endpoint_returns_actual_materialized_raw_values_exactly():
    raw=torch.randn(3,11);note=torch.randn_like(raw);target=torch.tensor([[1,2,3]])
    materialized=m.token_logps(raw,target)
    # A representable perturbation distinguishes actual reuse from recomputing.
    supplied=torch.nextafter(materialized,torch.full_like(materialized,-float('inf')))
    values,actual,_=m.mixed_token_logps(raw,note,target,m.GRID,supplied)
    assert values[0].contiguous().numpy().tobytes()==supplied[0].numpy().tobytes()
    assert actual.data_ptr()==supplied.data_ptr()


class TinyBackend:
    device=torch.device('cpu')
    def __init__(self):
        self.model=torch.nn.Module();self.model.lm_head=torch.nn.Linear(5,11,bias=False)
        self.embedding=torch.nn.Embedding(151644,5)
        self.syncs=0
    def hidden(self,ids):return self.embedding(ids).cumsum(1)/100
    def sync(self):self.syncs+=1


def test_real_target_positions_match_unpadded_causal_dense_reference():
    torch.manual_seed(5);b=TinyBackend();past=[1,2,3,4];target=[7,8,9]
    actual=m.full_logits(b,past,target)
    h=b.hidden(torch.tensor([past+target]))
    expected=b.model.lm_head(h[:,len(past)-1:len(past)+len(target)-1])[0]
    # CPU GEMM changes its row blocking between3 and16 head rows.
    torch.testing.assert_close(actual[:len(target)],expected,atol=1e-8,rtol=1e-6)
    assert actual.shape==(16,11) and b.syncs==1


@pytest.mark.parametrize('past,target',[(list(range(1010)),[1]),([1],[]),([1],list(range(17)))])
def test_overwidth_or_truncated_targets_rejected(past,target):
    with pytest.raises(ValueError,match='Bounded'):m.full_logits(TinyBackend(),past,target)


def fit_fixture():
    items=[dict(item_id=f'train{i}',document_id=f'doc{i}',target_ids=[i+1]*16) for i in range(3)]
    rows=[]
    for r in items:
        base=dict(cohort='train',item_id=r['item_id'],document_id=r['document_id'],target_ids=r['target_ids'],
            target_sha256=m.canonical_hash(r['target_ids']),status='ok')
        rows.append(dict(base,mode='raw',context_sha256=f'{r["item_id"]}/raw',token_logps=[-2.]*16))
        for mode in ('empty','correct'):
            values=[]
            for j,w in enumerate(m.GRID):
                value=(-2. if j==0 else -2.-(.25 if j<4 else .5)/16) if mode=='empty' else -2.+j*.02
                values.append([value]*16)
            rows.append(dict(base,mode=mode,context_sha256=f'{r["item_id"]}/{mode}',weights=list(m.GRID),token_logps=values,
                zero_raw_bitwise_equal=True,one_note_max_error=0.))
    return items,rows


def select(rows,items):
    expected={f'train:{r["item_id"]}':{mode:f'{r["item_id"]}/{mode}' for mode in m.MODES[:3]} for r in items}
    return m.select_weight(rows,items,expected)


def test_constrained_fit_uses_all_training_rows_and_highest_feasible_objective():
    items,rows=fit_fixture();result=select(rows,items)
    assert result['selected_weight']==1/8
    assert [z['eligible'] for z in result['grid']]==[True]*4+[False]*5
    assert result['grid'][0]['empty_minus_raw']==0


def test_exact_tie_chooses_smaller_weight_and_identity_remains_available():
    items,rows=fit_fixture()
    for v in rows:
        if v['mode']=='correct':v['token_logps']=[[-2.]*16 for _ in m.GRID]
    assert select(rows,items)['selected_weight']==0
    for v in rows:
        if v['mode']=='empty':v['token_logps']=[[-2.]*16]+[[-3.]*16 for _ in m.GRID[1:]]
    assert select(rows,items)['selected_weight']==0


@pytest.mark.parametrize('mutation',['missing','duplicate','dev','endpoint0','endpoint1','endpoint_nan','wrong_target','nonfinite','bad_grid','wrong_context'])
def test_fit_refuses_changed_denominator_or_failed_endpoint(mutation):
    items,rows=fit_fixture()
    if mutation=='missing':rows.pop()
    elif mutation=='duplicate':rows.append(copy.deepcopy(rows[0]))
    elif mutation=='dev':rows[0]['cohort']='general'
    elif mutation=='endpoint0':rows[1]['zero_raw_bitwise_equal']=False
    elif mutation=='endpoint1':rows[1]['one_note_max_error']=.001
    elif mutation=='endpoint_nan':rows[1]['one_note_max_error']=float('nan')
    elif mutation=='wrong_target':rows[0]['target_ids']=[9]*16
    elif mutation=='nonfinite':rows[1]['token_logps'][3][0]=float('nan')
    elif mutation=='bad_grid':rows[1]['weights']=[0,1]
    elif mutation=='wrong_context':rows[1]['context_sha256']='different'
    with pytest.raises((ValueError,RuntimeError)):select(rows,items)


def test_first_training_panel_uses_literal_unique_document_order_and_no_dev(tmp_path):
    p=tmp_path/'records.jsonl';source=[]
    def record(identifier,doc,split):return dict(id=identifier,document_id=doc,split=split,prefix=[1]*512,future=[2]*16)
    source.append(record('ignored','dev','dev'))
    for i in range(256):
        source.append(record(f'i{i}',f'd{i}','train'))
        if i==0:source.append(record('duplicate-document','d0','train'))
    p.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in source))
    actual=m.first_training_rows(p)
    assert len(actual)==256 and actual[0]['item_id']=='i0' and actual[1]['item_id']=='i1'
    assert actual[-1]['item_id']=='i255' and actual[0]['reanchor_ids']==[1]*24
    assert all(r['split']=='train' for r in actual)


def test_train_context_uses_only_empty_or_full_target_copy_and_exact_I3():
    row=dict(prefix_ids=[1,2],target_ids=[7,8],reanchor_ids=[1,2])
    inputs=dict(parts={'old_open':[10],'old_close':[11]})
    assert m.context(row,'raw',inputs,training=True)==[1,2]
    assert m.context(row,'empty',inputs,training=True)==[1,2,10,11,1,2]
    assert m.context(row,'correct',inputs,training=True)==[1,2,10,7,8,11,1,2]
    with pytest.raises(ValueError,match='no wrong-note'):m.context(row,'corrupt',inputs,training=True)


def test_complete_summarize_path_accepts_real_bootstrap_metadata(tmp_path,monkeypatch):
    from weekend_thinking import stats
    protocol=dict(bootstrap={'repeats':5000,'seed':17,'unit':'document'},limitations=[])
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    (tmp_path/'fit.json').write_text(json.dumps({'selected_weight':0.,'grid':[]}))
    inputs=dict(training_rows=[],dev_rows=[],context_hashes={})
    actual=[]
    for cohort in ('general','arithmetic'):
        for i in range(256):
            key=f'{cohort}:i{i}';row=dict(cohort=cohort,item_id=f'i{i}',document_id=f'{cohort}-d{i}',
                target_ids=[1,2],numeric_token_mask=[True,False])
            inputs['dev_rows'].append(row)
            inputs['context_hashes'][key]={mode:f'{key}/{mode}' for mode in m.MODES}
            for mode in m.MODES:
                actual.append(dict(row,mode=mode,status='ok',weights=[0.],context_sha256=f'{key}/{mode}',
                    token_logps=[-2.,-3.] if mode=='raw' else [[-2.,-3.]]))
    monkeypatch.setattr(m,'load',lambda output:(protocol,inputs))
    monkeypatch.setattr(m,'complete_rows',lambda output,stage,p:(actual if stage=='dev' else [],{}))
    monkeypatch.setattr(m,'select_weight',lambda *args:{'selected_weight':0.,'grid':[]})
    calls=[]
    def bootstrap(a,b,*,repeats,seed):
        assert repeats==5000 and seed==17 and len(a)==len(b)==256
        calls.append((repeats,seed))
        return {'difference_b_minus_a':0.,'ci95':[0.,0.]}
    monkeypatch.setattr(stats,'paired_cluster_bootstrap',bootstrap)
    result=m.summarize(tmp_path)
    assert result['status']=='complete' and result['rows']==3072
    assert len(calls)==30 and (tmp_path/'summary.json').exists()
