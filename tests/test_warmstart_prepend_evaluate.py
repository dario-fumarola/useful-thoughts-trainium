"""Complete G1 bundles, exact prepend semantics, raw parity and paired reduction."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from train import warmstart_prepend_evaluate as probe
from weekend_thinking.artifacts import canonical_hash, sha256_file
from weekend_thinking.core import Format


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def write_rows(path,values):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text("".join(json.dumps(v)+"\n" for v in values))


@pytest.fixture
def bundle(tmp_path,monkeypatch):
    import transformers
    directory=tmp_path/"original";directory.mkdir();ids=[f"item-{i:02d}" for i in range(64)]
    manifest="f"*64;identity=dict(**probe.original.model_identity("B0"),loaded_checkpoint=None,
                                 model={"parameters":3,"sampled_parameter_sha256":"a"*64},base_manifest_sha256=manifest)
    selection=dict(split="dev",count=64,item_ids=ids,document_ids=["doc:"+k for k in ids],
                   item_ids_sha256=canonical_hash(ids),strata={key:f"Q{i//16+1}" for i,key in enumerate(ids)},records_file="unused")
    p=dict(selection=selection,writer=probe.original.model_identity("B0"),controls_seed=17,filler_text=" ",
           cap=256,groups_per_prefix=1,temperature=.7,batch_size=4,model_dir=str(tmp_path/"model"),
           model_manifest_sha256=manifest,format_sha256="b"*64)
    fmt=Format((2,),(3,),5,0);write(directory/"protocol.json",p);notes={}
    for shard in range(4):
        local=[];shard_ids=probe.original.shard_ids(p,shard)
        for i,key in enumerate(shard_ids):
            global_index=shard*16+i;content=[] if global_index==0 else ([8]*256 if global_index==63 else [100+global_index,200+global_index])
            actions=content+([] if len(content)==256 else [5]);batch=shard_ids[i//4*4:i//4*4+4]
            row=dict(item_id=key,document_id="doc:"+key,split="dev",status="ok",actions=actions,
                action_mask=[True]*len(actions)+[False]*(256-len(actions)),content_ids=content,
                behavior_logps=[-1.]*len(actions),n_actions=len(actions),n_content_tokens=len(content),
                prefix_sha256=canonical_hash([10+global_index]*512),stop_type="length_cap" if len(actions)==256 else "token:5",
                batch_item_ids=batch,batch_seed=probe.original.batch_seed(batch),batch_row=i%4,
                cache_replay_max_error=0. if i<4 else None)
            local.append(row);notes[key]=row
        d=directory/"notes"/f"shard-{shard}";write_rows(d/"notes.jsonl",local)
        write(d/"run.json",dict(status="complete",writer=identity,shard=shard,notes=16,sampling_batches=4,
            started_utc="2026-10-17T00:00:00+00:00",finished_utc="2026-10-17T00:00:10+00:00",
            batch_sampling_wall_seconds_total=4.,batch_replay_audit_wall_seconds_total=1.,
            protocol_sha256=sha256_file(directory/"protocol.json"),notes_sha256=sha256_file(d/"notes.jsonl"),item_ids=shard_ids))
    controls=probe.original.make_controls(p,notes,[220]);write_rows(directory/"controls.jsonl",controls)
    write(directory/"controls.complete.json",dict(status="complete",writer=identity,items=64,filler_ids=[220],
        protocol_sha256=sha256_file(directory/"protocol.json"),notes_sha256=canonical_hash(notes),controls_sha256=sha256_file(directory/"controls.jsonl")))
    for shard in range(4):
        d=directory/"readers/B0"/f"shard-{shard}";values=[]
        for key in probe.original.shard_ids(p,shard):
            index=ids.index(key)
            for mode in ("raw","raw_repeat32",*probe.original.MODES):
                values.append(dict(item_id=key,document_id="doc:"+key,mode=mode,status="ok",writer_id="B0",reader_id="B0",
                                   token_log_likelihoods=[-3-index*.01]*16))
        write_rows(d/"scores.jsonl",values)
        write(d/"run.json",dict(status="complete",shard=shard,rows=112,expected_rows=112,writer=identity,reader=identity,
            item_ids=probe.original.shard_ids(p,shard),notes_sha256=canonical_hash(notes),
            controls_sha256=sha256_file(directory/"controls.jsonl"),protocol_sha256=sha256_file(directory/"protocol.json"),scores_sha256=sha256_file(d/"scores.jsonl")))
    monkeypatch.setattr(probe.original,"read_protocol",lambda _: (p,fmt))
    monkeypatch.setattr(transformers.AutoTokenizer,"from_pretrained",lambda *a,**k:SimpleNamespace(encode=lambda text,**k:[271] if text=="\n\n" else [220]))
    return directory,p,fmt,notes,identity


def mock_scoring(monkeypatch,bundle,reader=None,raw_offset=0.):
    directory,p,_,_,identity=bundle;reader=identity if reader is None else reader
    monkeypatch.setattr(probe.original,"load_selected",lambda path,selected,**kw:[dict(item_id=k,document_id="doc:"+k,
        prefix=[10+p["selection"]["item_ids"].index(k)]*512,future=[300]*16) for k in selected["item_ids"]])
    monkeypatch.setattr(probe.original,"load_model",lambda *args:(None,None,reader,None))
    def score(_,context,targets,**kwargs):
        assert kwargs["total_width"]==1024 and kwargs["future_width"]==16
        index=context[-1]-10;value=-3-index*.01+(raw_offset if len(context)==512 else -.25)
        return [SimpleNamespace(token_log_likelihoods=[value]*16)]
    monkeypatch.setattr(probe,"score_continuations",score)


def test_contexts_prepend_complete_prefix_without_old_header_close_or_repeat():
    prefix=list(range(512));control={"content_ids":{m:[800,801] for m in probe.MODES[1:]}}
    control["content_ids"]["empty"]=[]
    values=dict(probe.contexts(prefix,control))
    assert values["raw"]==prefix and values["empty"]==[271]+prefix
    assert values["own"]==[800,801,271]+prefix and len(values["own"])==515
    assert set(values)==set(probe.MODES) and control["content_ids"]["own"]==[800,801]
    with pytest.raises(ValueError,match="original512"):
        probe.contexts(prefix[:-1],control)


@pytest.mark.parametrize("damage",["missing_note","wrong_donor","wrong_writer","partial_count"])
def test_invalid_closed_bundle_blocks_new_futures_and_models(tmp_path,monkeypatch,bundle,damage):
    d,p,_,_,_=bundle
    if damage=="missing_note":(d/"notes/shard-3/notes.jsonl").unlink()
    elif damage=="wrong_donor":
        path=d/"controls.jsonl";values=probe.rows(path);values[15]["donor_item_id"]="wrong";write_rows(path,values)
        receipt=probe.read(d/"controls.complete.json");receipt["controls_sha256"]=sha256_file(path);write(d/"controls.complete.json",receipt)
    else:
        path=d/"notes/shard-3/run.json";run=probe.read(path)
        if damage=="wrong_writer":run["writer"]["base_manifest_sha256"]="bad"
        else:run["notes"]=15
        write(path,run)
    monkeypatch.setattr(probe.original,"load_selected",lambda *a,**k:pytest.fail("New future access before full bundle"))
    monkeypatch.setattr(probe.original,"load_model",lambda *a,**k:pytest.fail("Model before full bundle"))
    with pytest.raises(ValueError):probe.prepare(tmp_path/"new",d,d,probe.original.model_identity("B0"))
    assert not (tmp_path/"new").exists()


def test_baseline_requires_full_original_scoring_matrix_and_bound_identity(bundle):
    d,_,_,_,_=bundle;p,raw,identity,_=probe.raw_baseline(d)
    assert len(raw)==64 and identity["id"]=="B0"
    path=d/"readers/B0/shard-2/scores.jsonl";values=probe.rows(path);values[0]["token_log_likelihoods"]=[-1.]*15
    write_rows(path,values);receipt=probe.read(path.parent/"run.json");receipt["scores_sha256"]=sha256_file(path);write(path.parent/"run.json",receipt)
    with pytest.raises(ValueError,match="reference matrix"):probe.raw_baseline(d)


def test_preparation_binds_all_original_receipts_and_no_reference_outcomes_changed(tmp_path,bundle):
    d,p,_,_,identity=bundle;out=tmp_path/"prepend"
    prepared=probe.prepare(out,d,d,probe.original.model_identity("B0"))
    assert prepared["writer"]==identity and prepared["selection"]==p["selection"]
    assert prepared["expected_rows"]==384 and prepared["reader"]["checkpoint_sha256"] is None
    assert "K16 fresh G1" in prepared["primary"] and "K1 secondary" in prepared["primary"]
    assert prepared["source_files"]["train/warmstart_evaluate.py"]==sha256_file(probe.original.__file__)
    assert str((d/"notes/shard-3/notes.jsonl").resolve()) in prepared["input_hashes"]
    assert probe.read_protocol(out)==prepared
    path=d/"controls.complete.json";path.write_text(path.read_text()+" ")
    with pytest.raises(ValueError,match="Changed original"):probe.read_protocol(out)


def test_score_and_summary_preserve_full_coverage_cost_lengths_and_quartiles(tmp_path,monkeypatch,bundle):
    d,_,_,_,_=bundle;out=tmp_path/"prepend";probe.prepare(out,d,d,probe.original.model_identity("B0"))
    mock_scoring(monkeypatch,bundle)
    for shard in range(4):probe.score(out,shard,"cpu")
    bootstrap=probe.paired_cluster_bootstrap
    monkeypatch.setattr(probe,"paired_cluster_bootstrap",lambda a,b,**kw:bootstrap(a,b,repeats=100,seed=kw["seed"]))
    summary=probe.summarize(out)
    assert summary["status"]=="complete" and summary["rows"]==384 and summary["documents"]==64
    assert summary["fixed_B0_reader_primary"] and summary["raw_reference_max_token_error"]==0
    assert summary["note_context"]["empty"]==1 and summary["note_context"]["caps"]==1
    assert summary["cost"]["reused_sampling_batches"]==16 and summary["cost"]["reused_sampling_batch_seconds_counted_once"]==16
    assert set(summary["groups"])=={"all","Q1","Q2","Q3","Q4"}
    assert summary["groups"]["all"]["horizons"]["K1"]["contrasts"]["own_minus_empty"]["difference_b_minus_a"]==0
    for mode in probe.MODES:
        absolute=summary["groups"]["Q1"]["horizons"]["K16"]["absolute_likelihoods"][mode]
        assert absolute["documents"]==16 and absolute["mean_log_likelihood_per_target_token"]==absolute["mean_log_likelihood"]/16
    assert len(summary["per_document"])==64


@pytest.mark.parametrize("value",[float("nan"),-.01])
def test_failed_raw_or_nonfinite_row_is_preserved_and_blocks_summary(tmp_path,monkeypatch,bundle,value):
    d,_,_,_,_=bundle;out=tmp_path/"prepend";probe.prepare(out,d,d,probe.original.model_identity("B0"))
    mock_scoring(monkeypatch,bundle,raw_offset=value)
    with pytest.raises(RuntimeError,match="Failed prepend score retained"):probe.score(out,0,"cpu")
    report=probe.read(out/"scoring/shard-0/run.json");scored=probe.rows(out/"scoring/shard-0/scores.jsonl")
    assert report["status"]=="failed" and report["rows"]==1 and scored[0]["status"]=="error"
    with pytest.raises(ValueError,match="complete bound prepend"):probe.summarize(out)


def test_adapted_reader_is_explicit_and_does_not_use_B0_raw_parity(tmp_path,monkeypatch,bundle):
    d,p,_,_,_=bundle;checkpoint=tmp_path/"student/model.pt";checkpoint.parent.mkdir();checkpoint.write_bytes(b"new weights")
    reader=probe.original.model_identity("student",checkpoint,sha256_file(checkpoint))
    actual=dict(**reader,model={"parameters":3,"sampled_parameter_sha256":"c"*64},
                base_manifest_sha256=p["model_manifest_sha256"],loaded_checkpoint={"sha256":reader["checkpoint_sha256"]})
    out=tmp_path/"adapted";probe.prepare(out,d,d,reader);mock_scoring(monkeypatch,bundle,actual,raw_offset=1.)
    probe.score(out,0,"cpu");report=probe.read(out/"scoring/shard-0/run.json")
    assert report["status"]=="complete" and report["reader"]==actual
    assert report["raw_reference_rows"]==0 and report["raw_reference_max_token_error"] is None


@pytest.mark.parametrize("damage",["missing","duplicate","nonfinite","document"])
def test_reducer_refuses_any_missing_duplicate_nonfinite_or_mispaired_row(bundle,damage):
    _,p,_,notes,_=bundle;values=[]
    for key in p["selection"]["item_ids"]:
        for mode in probe.MODES:
            values.append(dict(item_id=key,document_id="doc:"+key,mode=mode,status="ok",token_log_likelihoods=[-1.]*16))
    if damage=="missing":values.pop()
    elif damage=="duplicate":values[-1]=values[0]
    elif damage=="nonfinite":values[0]["token_log_likelihoods"][0]=float("nan")
    else:values[0]["document_id"]="different doc"
    with pytest.raises(ValueError,match="exactly once"):probe.reduce_rows(p,values,notes,repeats=100)
