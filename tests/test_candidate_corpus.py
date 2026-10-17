"""CPU-only shard/global-barrier semantics; no actual corpus or model outcomes."""
import json

import pytest

from train import candidate_corpus as corpus
from train import candidate_probe as shared
from weekend_thinking.artifacts import canonical_hash, sha256_file
from weekend_thinking.core import Format


def protocol():
    ids=[f"train:{i}" for i in range(128)]
    return dict(schema=corpus.SCHEMA,experiment_id=corpus.SCHEMA,
        selection=dict(split="train",count=128,item_ids=ids,
            document_ids=[f"doc:{i}" for i in range(128)],item_ids_sha256=canonical_hash(ids)),
        seed=1729,control_seed=17,filler_ids=[7],horizons=[1,16],
        interfaces=dict(original=[True,0],reanchor32=[True,32]),
        strata={key:"TRAIN_selected" for key in ids},limitations=["TRAIN future-scored; not deployed utility"])


def fixture_generation(tmp_path, p, *, shards=8):
    (tmp_path/"protocol.json").write_text(json.dumps(p))
    for shard in range(shards):
        directory=tmp_path/"generation"/f"shard-{shard:02d}";directory.mkdir(parents=True)
        rows=[]
        for key in corpus.shard_selection(p["selection"],shard)["item_ids"]:
            global_index=p["selection"]["item_ids"].index(key)
            for i in range(16):
                row=dict(item_id=key,document_id=f"doc:{global_index}",split="train",shard=shard,
                    candidate_index=i,group=i//4,group_sample=i%4,status="ok",
                    group_seed=corpus.seed_for(1729,key,i//4),actions=[10+global_index+i,5],
                    behavior_logps=[-2.,-1.],action_mask=[True,True]+[False]*254,
                    content_ids=[10+global_index+i],n_actions=2,n_content_tokens=1,
                    prefix_sha256=canonical_hash([10,11,12]),stop_type="token:5",decode_iterations=32)
                if global_index==0 and i==0:row.update(cache_replay_status="ok",cache_replay_max_error=0.)
                rows.append(row)
        corpus.write_rows(directory/"candidates.jsonl",rows)
        (directory/"run.json").write_text(json.dumps(dict(schema=corpus.SCHEMA,status="complete",shard=shard,
            protocol_sha256=sha256_file(tmp_path/"protocol.json"),model={"fixture":"B0"},checkpoint=None,
            candidates=len(rows),expected_candidates=256,candidates_sha256=sha256_file(directory/"candidates.jsonl"))))


def test_eight_stable_partitions_cover_each_id_once_and_have_new_seed_namespace():
    p=protocol();parts=[corpus.shard_selection(p["selection"],s) for s in range(8)]
    assert all(len(v["item_ids"])==16 for v in parts)
    flat=[key for v in parts for key in v["item_ids"]]
    assert len(flat)==len(set(flat))==128 and set(flat)==set(p["selection"]["item_ids"])
    assert parts[0]["item_ids"]==p["selection"]["item_ids"][0::8]
    seeds=[corpus.seed_for(1729,key,g) for key in p["selection"]["item_ids"] for g in range(4)]
    assert len(set(seeds))==512
    assert corpus.seed_for(1729,"train:0",0)!=shared.group_seed(1729,"train:0",0)
    with pytest.raises(ValueError,match="index"):
        corpus.shard_selection(p["selection"],8)


def test_incomplete_generation_refuses_models_and_future_access(tmp_path,monkeypatch):
    p=protocol();fmt=Format((2,),(3,),5,0);fixture_generation(tmp_path,p,shards=7)
    monkeypatch.setattr(corpus,"read_protocol",lambda _: (p,fmt))
    monkeypatch.setattr(corpus,"setup_model",lambda *a,**k:pytest.fail("Model loaded before global barrier"))
    monkeypatch.setattr(corpus,"load_selected",lambda *a,**k:pytest.fail("Future read before global barrier"))
    with pytest.raises(FileNotFoundError):corpus.score(tmp_path,0,"cpu")
    assert not (tmp_path/"scoring").exists()


def test_global_controls_use_donors_across_shards_and_tampering_is_rejected(tmp_path,monkeypatch):
    p=protocol();fmt=Format((2,),(3,),5,0);fixture_generation(tmp_path,p)
    monkeypatch.setattr(corpus,"read_protocol",lambda _: (p,fmt))
    corpus.controls(tmp_path)
    notes,controls,receipt=corpus.global_inputs(tmp_path,p,fmt)
    assert len(notes)==len(controls)==receipt["candidates"]==2048
    first=controls[("train:0",0)]
    assert first["donor_item_id"]=="train:1" and first["content_ids"]["other"]==[11]
    assert "train:1" not in corpus.shard_selection(p["selection"],0)["item_ids"]
    path=tmp_path/"global/controls.jsonl"
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["donor_item_id"]="train:8"  # A local-only donor must not pass.
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    receipt["controls_sha256"]=sha256_file(path)
    (tmp_path/"global/COMPLETE.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match="global same-index"):
        corpus.global_inputs(tmp_path,p,fmt)


def test_missing_or_failed_candidate_cannot_be_filtered_at_global_barrier(tmp_path):
    p=protocol();fmt=Format((2,),(3,),5,0);fixture_generation(tmp_path,p)
    directory=tmp_path/"generation/shard-00";path=directory/"candidates.jsonl"
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    rows[3]["status"]="error"
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    report=json.loads((directory/"run.json").read_text());report["candidates_sha256"]=sha256_file(path)
    (directory/"run.json").write_text(json.dumps(report))
    with pytest.raises(ValueError,match="Invalid sampled"):
        corpus.load_all_candidates(tmp_path,p,fmt)


def test_reused_statistics_report_real_train_identity_and_keep_failed_denominator():
    p=protocol();ids=["train:0","train:1"]
    p["selection"]=dict(p["selection"],item_ids=ids,document_ids=["doc:0","doc:1"],count=2)
    p["interfaces"]={"original":[True,0]};p["strata"]={key:"TRAIN_selected" for key in ids}
    rows=[]
    for key in ids:
        def add(interface,mode,i,value):
            rows.append(dict(item_id=key,document_id="doc:"+key.split(":")[1],split="train",
                interface=interface,mode=mode,candidate_index=i,status="ok",
                token_log_likelihoods=[value]+[0.]*15,n_note_content_tokens=1,note_stop_type="token:5"))
        for mode in shared.RAW_REPEATS:add("raw",mode,None,-1.)
        add("original","empty",None,-2.)
        for i in range(16):
            for mode in shared.CONTENT_MODES:add("original",mode,i,-1. if mode=="own" and i==2 else -3.)
    result=corpus.reduce_rows(p,rows)
    assert result["schema"]==corpus.SCHEMA and result["split"]=="train"
    assert all(r["stratum"]=="TRAIN_selected" and "quartile" not in r for r in result["per_document"])
    assert all(r["oracle"]["candidate_index"]==2 for r in result["per_document"])
    failed=next(r for r in rows if r["item_id"]=="train:0" and r["mode"]=="own")
    failed["status"]="error";failed.pop("token_log_likelihoods")
    result=corpus.reduce_rows(p,rows)
    entry=result["groups"]["all"]["original/K1/oracle_max_own_minus_empty"]
    assert entry["expected_documents"]==2 and entry["complete_documents"]==1
    assert "document_bootstrap" not in entry
