"""Native GENERAL dispatch, closed-order proof and old/new source separation."""
import copy
import json
from pathlib import Path

import pytest
import torch

from train import general_hindsight_filtered_data as native
from train import prepare_general_sft_data as prep
from train import useful_note_sft_data as packing
from train import useful_note_sft_driver as driver
from weekend_thinking.artifacts import canonical_hash, sha256_file


class Tok:
    all_special_ids=[151643,151644]
    eos_token_id=151643
    def get_vocab(self):return {str(i):i for i in range(1,11)}
    def decode(self,ids,**kwargs):return ''.join({1:'past ',2:'future ',3:'a useful cue ',4:'else '}.get(t,str(t)) for t in ids)
    def encode(self,*args,**kwargs):raise AssertionError('Native labels must never be re-encoded')


PARTS=dict(old_open=[151667,198],old_close=[151668,198])


def example(i=0):
    row=dict(item_id=str(i),document_id='doc'+str(i),split='train',teacher_index=i,pool_index=i+10,
        prefix_ids=[1]*512,natural_future16_ids=[2]*16,reanchor_ids=[1]*24,source_line=i+1,
        original_source_ids=[1]*512+[2]*16,source528_sha256=canonical_hash([1]*512+[2]*16),
        teacher_note_ids=[3]*3,teacher_note_text=Tok().decode([3]*3),teacher_kind=prep.KIND,
        selection_uses_likelihood=True,teacher_prompt_sha256='prompt',teacher_protocol_sha256='protocol')
    return row


def decision(i,positive=True):
    # Teacher still loses to raw/other: those were never acceptance requirements.
    path=dict(absolute_ll=dict(teacher=-2.,empty=-3. if positive else -1.,raw=0.,other=0.),
        statuses=dict(teacher='ok',empty='ok',raw='ok',other='ok'),
        strict_teacher_positive=dict(empty=positive,raw=False,other=False))
    return dict(teacher_index=i,pool_index=i+10,item_id=str(i),document_id='doc'+str(i),
        eligible_plain_label=dict(eligible=True),accepted=positive,paths=dict(I3=path))


def tiny(monkeypatch):
    monkeypatch.setattr(prep,'ITEMS',2);monkeypatch.setattr(native,'TEACHER_N',4);monkeypatch.setattr(native,'RAW_N',4)


def selected_fixture():
    dd=[decision(i,i!=1) for i in range(4)]
    rr=[dict(example(i),label_decision=dd[i]) for i in (0,2)]
    return rr,dd


def test_first_original_accepted_order_is_preserved_without_stricter_new_gate(monkeypatch):
    tiny(monkeypatch);rows,dd=selected_fixture()
    result=prep.selected_rows(rows,dd,Tok())
    assert [r['item_id'] for r in result]==['0','2']
    assert all(r['native_window_cut']==512 and 'absolute_source_cut' not in r for r in result)
    assert [dict(r,native_window_cut=512) for r in rows]==result
    with pytest.raises(ValueError,match='first5120'):prep.selected_rows(list(reversed(rows)),dd,Tok())


def test_quota_failure_or_forged_original_acceptance_cannot_be_prepared(monkeypatch):
    tiny(monkeypatch);rows,dd=selected_fixture()
    for d in dd[1:]:
        d['paths']['I3']['absolute_ll']['empty']=-1.;d['paths']['I3']['strict_teacher_positive']['empty']=False;d['accepted']=False
    with pytest.raises(ValueError,match='quota'):prep.selected_rows(rows[:1],dd,Tok())
    rows,dd=selected_fixture();dd[0]['accepted']=False
    with pytest.raises(ValueError,match='strict gain'):prep.selected_rows(rows,dd,Tok())


@pytest.mark.parametrize('change',[
    dict(split='dev'),dict(teacher_kind='exact-postorder-v1'),dict(selection_uses_likelihood=False),
    dict(teacher_note_ids=[151667]),dict(teacher_note_ids=[2]*16),dict(teacher_note_text='rewritten'),
    dict(original_source_ids=[1]*528),dict(source528_sha256='wrong')])
def test_native_original_label_and_causal_lineage_cannot_be_changed(monkeypatch,change):
    tiny(monkeypatch);rows,dd=selected_fixture();rows[0].update(change)
    with pytest.raises(ValueError):prep.selected_rows(rows,dd,Tok())


@pytest.mark.parametrize('component',packing.COMPONENTS)
def test_dispatch_is_exact_qualified_native_adapter_with_explicit_padding(component):
    row=example();tok=Tok();shape=prep.SHAPES[component]
    actual=packing.pack_example(row,tok,component,interface='I3',parts=PARTS,**shape)
    expected=native.pack_native(row,tok,component,parts=PARTS,**shape)
    assert all(torch.equal(actual[k],expected[k]) for k in actual)
    active=actual['note_close_mask']|actual['future_mask'];labels=actual['labels'][active].tolist()
    if component=='source':assert labels==row['original_source_ids'][1:]
    if component=='reader':assert labels==row['natural_future16_ids']
    if component=='writer':
        assert labels==row['teacher_note_ids']+[151668] and not actual['future_mask'].any()
        changed=dict(row,natural_future16_ids=[4]*16,original_source_ids=[1]*512+[4]*16)
        after=packing.pack_example(changed,tok,component,interface='I3',parts=PARTS,**shape)
        assert all(torch.equal(actual[k],after[k]) for k in actual)
    assert torch.equal(actual['labels'][active],actual['input_ids'].gather(1,actual['target_positions'])[active])


def test_unknown_teacher_kind_is_not_reinterpreted_as_symbolic():
    with pytest.raises(ValueError,match='Unknown explicit teacher'):
        packing.pack_example(dict(example(),teacher_kind='other'),Tok(),'writer',interface='I3',parts=PARTS,**prep.SHAPES['writer'])


def inventory(rows):
    p=dict(items=len(rows),purpose='qualification',data_sha256='data',packing=prep.SHAPES,interface='I3',parts=PARTS)
    m=dict(schema=prep.SCHEMA,status='complete_CPU_inventory_NOT_Neuron_training_qualification',
        teacher_kind=prep.KIND,purpose='qualification',split='train',selection_uses_likelihood=True,
        items=len(rows),unique_documents=len(rows),training_sha256='data',packing=prep.SHAPES,interface='I3',parts=PARTS,
        ordered_item_ids_sha256=canonical_hash([r['item_id'] for r in rows]),ordered_document_ids_sha256=canonical_hash([r['document_id'] for r in rows]))
    return p,m


def test_driver_honest_general_schema_requires_64_subset_unique_docs_and_cut512():
    rows=[dict(example(i),native_window_cut=512) for i in range(64)];p,m=inventory(rows)
    assert driver.validate_inventory(p,m,rows)==64
    for field,value in [('selection_uses_likelihood',False),('schema','useful-note-supervised-train-v3'),('teacher_kind','symbolic')]:
        bad=dict(m);bad[field]=value
        with pytest.raises(ValueError):driver.validate_inventory(p,bad,rows)
    bad=copy.deepcopy(rows);bad[0]['native_window_cut']=511
    with pytest.raises(ValueError):driver.validate_inventory(p,m,bad)
    p['purpose']=m['purpose']='science'
    with pytest.raises(ValueError,match='GENERAL'):driver.validate_inventory(p,m,rows)


def test_original_filter_source_pins_resolve_at_main_not_modified_worktree(tmp_path,monkeypatch):
    tiny(monkeypatch);old=tmp_path/'old-main';(old/'train').mkdir(parents=True)
    old_file=old/'train/useful_note_sft_data.py';old_file.write_text('old exact filter packer')
    directory=tmp_path/'closed';directory.mkdir();control=tmp_path/'control';control.mkdir()
    source={'train/useful_note_sft_data.py':sha256_file(old_file)}
    p=dict(schema=native.SCHEMA,control_dir=str(control),source_files=source)
    (directory/'protocol.json').write_text(json.dumps(p));protocol_sha=sha256_file(directory/'protocol.json')
    rows,decisions=selected_fixture()
    for r in rows:r['teacher_protocol_sha256']=protocol_sha
    data=directory/'training.jsonl';data.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    choices=directory/'decisions.jsonl';choices.write_text(''.join(json.dumps(d)+'\n' for d in decisions))
    status='COMPLETE_FIXED5120_NATIVE_LABELS_NOT_TRAINING_RELEASE'
    s=dict(status=status,counts=dict(raw_documents=4,teacher_attempts=4,reader_attempts=32,selected=2,requested=2),
        protocol_sha256=protocol_sha,source_files=source,input_hashes={},training_file=str(data),training_sha256=sha256_file(data),
        decisions_file=str(choices),decisions_sha256=sha256_file(choices))
    m=dict(schema='general-hindsight-native-labels-v1',status=status,items=2,selection_uses_likelihood=True,
        source_files=source,protocol_sha256=protocol_sha,examples_file=str(data),examples_sha256=sha256_file(data),
        decisions_sha256=sha256_file(choices),item_ids_sha256=canonical_hash([r['item_id'] for r in rows]))
    for name,value in [('summary.json',s),('training.manifest.json',m),('manifest.json',{})]:
        (directory/name).write_text(json.dumps(value))
    for phase in ('raw','teacher','reader'):
        (control/f'{phase}.closure.json').write_text(json.dumps(dict(status='COMPLETE_PHYSICALLY_CLOSED',protocol_sha256=protocol_sha,
            all16_exit0=True,all48_owned_PID_start_identities_absent=True,all_sessions_absent=True,assigned_devices_empty=True,owned_group_survivors=[])))
    monkeypatch.setattr(native,'load',lambda *a:(_ for _ in ()).throw(AssertionError('Cannot reinterpret old filter with changed worktree hashes')))
    _,actual,_,pins=prep.closed_publication(directory,old)
    assert actual==rows and pins[str(old_file)]==source['train/useful_note_sft_data.py']
    old_file.write_text('changed')
    with pytest.raises(ValueError,match='provenance changed'):prep.closed_publication(directory,old)


def test_CPU_preparation_counts_all_four_native_components_without_model(tmp_path,monkeypatch):
    from transformers import AutoTokenizer
    tiny(monkeypatch);rows,dd=selected_fixture()
    monkeypatch.setattr(prep,'closed_publication',lambda *a:(dict(model_dir='unused',parts=PARTS),rows,dd,{}))
    monkeypatch.setattr(AutoTokenizer,'from_pretrained',lambda *a,**k:Tok())
    monkeypatch.setattr(driver,'source_files',lambda:{'prepared-source':'fixed'})
    out=tmp_path/'new'
    m=prep.prepare('unused','unused',out,'2099-01-01T00:00:00+00:00')
    assert m['schema']==prep.SCHEMA and m['selection_uses_likelihood'] is True and m['teacher_kind']==prep.KIND
    assert m['packing']==prep.SHAPES and m['items']==m['unique_documents']==2
    counts=m['component_counts']
    assert counts['source']['active_labels']==2*527
    assert counts['writer']['active_labels']==counts['writer']['note_close_labels']==2*4
    assert counts['writer']['non_note_labels']==0
    assert counts['reader']['active_labels']==counts['reader']['non_note_labels']==2*16
    assert counts['reader']['note_close_labels']==0
    written=[json.loads(line) for line in (out/'training.jsonl').read_text().splitlines()]
    assert written==[dict(r,native_window_cut=512) for r in rows]
    assert m['training_sha256']==sha256_file(out/'training.jsonl')


@pytest.mark.parametrize('size,duplicate',[(5000,False),(5001,False),(5120,True),(5120,False)])
def test_qualification_loader_checks_actual_full_parent_quota_and_unique_documents(tmp_path,monkeypatch,size,duplicate):
    from transformers import AutoTokenizer
    import train.evaluate as evaluate
    rows=[dict(example(i),native_window_cut=512) for i in range(size)]
    if duplicate:rows[-1]['document_id']=rows[0]['document_id']
    parent_data=tmp_path/'science-training.jsonl'
    lines=[json.dumps(r)+'\n' for r in rows];parent_data.write_text(''.join(lines))
    _,science=inventory(rows);science.update(purpose='science',unique_documents=5120,
        training_file=str(parent_data),training_sha256=sha256_file(parent_data))
    parent=tmp_path/'science-manifest.json';parent.write_text(json.dumps(science))
    selected=rows[:64];p,m=inventory(selected)
    data=tmp_path/'qualification-training.jsonl';data.write_text(''.join(lines[:64]))
    m.update(training_sha256=sha256_file(data),parent_science_manifest=str(parent),
        parent_science_manifest_sha256=sha256_file(parent),parent_science_training_sha256=sha256_file(parent_data),
        preparation_source_sha256=sha256_file(driver.__file__),filter_input_hashes={},source_files={},
        tokenizer_vocab_sha256=canonical_hash(Tok().get_vocab()),writer_stop_ids=[151668,151643])
    manifest=tmp_path/'qualification-manifest.json';manifest.write_text(json.dumps(m))
    p.update(data_file=str(data),data_sha256=sha256_file(data),dataset_file=str(manifest),dataset_sha256=sha256_file(manifest),
        source_files={'train/useful_note_sft_driver.py':sha256_file(driver.__file__)},model_dir='not-a-model',model_manifest_sha256='assets')
    reached=[]
    def assets(*args):
        reached.append(True);return dict(model_id='Qwen/Qwen3-1.7B-Base',manifest_sha256='assets')
    monkeypatch.setattr(evaluate,'asset_provenance',assets)
    monkeypatch.setattr(AutoTokenizer,'from_pretrained',lambda *a,**k:Tok())
    if size!=5120 or duplicate:
        with pytest.raises(ValueError):driver.load_inputs(p)
        assert not reached
    else:
        actual,_,_,_=driver.load_inputs(p)
        assert actual==selected and reached==[True]
