import json
from dataclasses import replace
import numpy as np
import pytest
from weekend_thinking.core import Record
from weekend_thinking.data import digest,split_document,make_record,write_records,load_records,data_index
from weekend_thinking.stats import paired_cluster_bootstrap
from weekend_thinking.qualification import project


def test_exact_dedup_hash_normalizes_whitespace():
    assert digest('one   two\nthree') == digest('one two three')


def test_anchor_is_deterministic_and_not_entropy_dependent():
    tokens=list(range(100))
    a=make_record(tokens,'doc',prefix_length=16,future_length=4)
    b=make_record(tokens,'doc',prefix_length=16,future_length=4)
    assert a==b and a.future[0] == a.prefix[-1]+1
    assert a.split==split_document('doc')
    assert make_record([1,2],'short',prefix_length=16,future_length=4) is None


def test_distributed_ids_are_unique_and_full():
    ids=[data_index(s,m,r,4,2) for s in range(5) for m in range(2) for r in range(4)]
    assert sorted(ids)==list(range(40))


def test_split_roundtrip_and_no_split_crossing(tmp_path):
    r=Record('a','doc','train',(1,2),(3,4))
    path=tmp_path/'records.jsonl'; assert write_records(path,[r])==1
    assert load_records(path)==[r]
    write_records(path,[r,replace(r,id='b',split='test')])
    with pytest.raises(ValueError):load_records(path)


def test_duplicate_ids_rejected(tmp_path):
    r=Record('a','doc','train',(1,2),(3,4))
    with pytest.raises(ValueError):write_records(tmp_path/'bad.jsonl',[r,r])


def test_bootstrap_sign_and_pairing():
    a=[{'item_id':str(i),'document_id':str(i//2),'score':0.0} for i in range(10)]
    b=[dict(x,score=0.25) for x in a]
    r=paired_cluster_bootstrap(a,b,repeats=100)
    assert r['difference_b_minus_a']==0.25 and r['ci95']==[0.25,0.25]
    assert r['documents']==5
    with pytest.raises(ValueError):paired_cluster_bootstrap(a,b[:-1],repeats=100)
    with pytest.raises(ValueError):paired_cluster_bootstrap(a,b+[b[0]],repeats=100)


def test_bootstrap_rejects_changed_clusters():
    a=[{'item_id':str(i),'document_id':str(i),'score':0.0} for i in range(5)]
    b=[dict(x) for x in a];b[0]['document_id']='not-the-same-story'
    with pytest.raises(ValueError):paired_cluster_bootstrap(a,b,repeats=100)


def test_budget_uses_full_measured_steps_not_peak_flops():
    p=project([10.0]*12,training_hours=10,unique_prefixes_per_step=64,prefix_tokens=512,
              max_thought_tokens=1024,group_size=4,logical_cores=32)
    assert p['projected_steps']==3060
    assert p['source_token_visits']==3060*64*513
    assert p['logical_core_hours']==320
    with pytest.raises(ValueError):project([0.0]*12,training_hours=10,unique_prefixes_per_step=64,
         prefix_tokens=512,max_thought_tokens=1024,group_size=4,logical_cores=32)


def test_record_preparation_rejects_fractional_token_ids():
    with pytest.raises(ValueError):make_record([1,2.5,3,4],'doc',prefix_length=2,future_length=1)
