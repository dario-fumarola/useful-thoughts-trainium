"""TRAIN ID selection, honest failure denominators and native CPU likelihoods."""
import hashlib
import json
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

import train.anchor_scan as scan
from train.neuron_backend import NeuronCausalBackend
from weekend_thinking.artifacts import canonical_hash, sha256_file


def header(item, document, split="train"):
    # Deliberately undecodable arrays prove that preparation uses headers only.
    return (json.dumps(dict(id=item, document_id=document, split=split))[:-1]
            + ', "prefix": [NOT_JSON], "future": [NEVER_DECODE]}\n')


def test_pool_is_canonical_ordered_document_distinct_and_future_blind(tmp_path):
    rows=[("a","same"),("b","same"),("c","c"),("d","d")]
    path=tmp_path/"rows.jsonl"
    path.write_text("".join(header(i,d) for i,d in rows)+header("dev","dev","dev"))
    first=scan.train_pool(path,sha256_file(path),count=3)
    expected=[];seen=set()
    for item,doc in sorted(rows,key=lambda pair:hashlib.sha256(
            json.dumps(dict(seed=20261017,item_id=pair[0]),sort_keys=True,separators=(",",":")).encode()).hexdigest()):
        if doc not in seen:
            expected.append(item);seen.add(doc)
    assert first["item_ids"]==expected
    assert len(set(first["document_ids"]))==3 and first["eligible_train_items"]==4
    assert first["pool_preparation_arrays_decoded"] is False
    path.write_text("".join(header(i,d) for i,d in reversed(rows)))
    assert scan.train_pool(path,sha256_file(path),count=3)["item_ids"]==expected
    with pytest.raises(ValueError,match="manifest"):
        scan.train_pool(path,"0"*64,count=3)
    with pytest.raises(ValueError,match="Insufficient"):
        scan.train_pool(path,sha256_file(path),count=4)


def test_prepare_never_loads_model_or_decodes_arrays_and_keeps_dev_separate(tmp_path,monkeypatch):
    records=tmp_path/"rows.jsonl"
    records.write_text("".join(header(f"train:{i}",f"doc:{i}") for i in range(1024))+header("dev:0","devdoc","dev"))
    records.with_suffix(".manifest.json").write_text(json.dumps(dict(prefix_length=512,future_length=16,
        output_sha256=sha256_file(records),source_revision="fixture")))
    model=tmp_path/"model";model.mkdir()
    (model/"ASSET_MANIFEST.json").write_text(json.dumps(dict(model_id="Qwen/Qwen3-1.7B-Base",resolved_revision="fixture")))
    selection=tmp_path/"dev.ids.json"
    selection.write_text(json.dumps(dict(records_sha256=sha256_file(records),split="dev",count=1,
        item_ids=["dev:0"],document_ids=["devdoc"],item_ids_sha256=canonical_hash(["dev:0"]))))
    parity=tmp_path/"saved-scores.jsonl";parity.write_text("unopened saved score bytes\n")
    monkeypatch.setattr(scan,"setup_model",lambda *a,**k:pytest.fail("Preparation loaded a model"))
    monkeypatch.setattr(scan,"load_selected",lambda *a,**k:pytest.fail("Preparation decoded arrays"))
    monkeypatch.setattr(scan,"sources",lambda:{"fixture":"0"*64})
    output=tmp_path/"prepared"
    p=scan.prepare(output,records=records,model_dir=model,selection_file=selection,parity_scores=parity)
    pool=json.loads((output/"pool.ids.json").read_text())
    assert pool["split"]=="train" and pool["count"]==1024 and "dev:0" not in pool["item_ids"]
    assert p["parity"]["selection"]["item_ids"]==["dev:0"] and p["threshold_surprisal_nats"]==1.7
    assert p["pool_sha256"]==sha256_file(output/"pool.ids.json")
    assert not (output/"scoring").exists()


def pool_rows(surprisals):
    ids=[f"a:{i}" for i in range(len(surprisals))]
    pool=dict(count=len(ids),item_ids=ids,document_ids=["doc:"+key for key in ids],
              item_ids_sha256=canonical_hash(ids),split="train")
    rows=[dict(item_id=key,document_id="doc:"+key,split="train",status="ok",
        token_log_likelihoods=[-s]+[0.]*15,raw_K1_surprisal_nats=s,
        raw_K1_log_likelihood=-s,raw_K16_log_likelihood=-s) for key,s in zip(ids,surprisals)]
    return pool,rows


def test_threshold_is_inclusive_first128_frozen_order_without_fallback():
    pool,rows=pool_rows([1.69,1.7,2.]+[9.]*140)
    summary,selected=scan.summarize(pool,list(reversed(rows)))
    assert summary["qualifying_items"]==142 and selected["item_ids"]==pool["item_ids"][1:129]
    pool,rows=pool_rows([1.7,.1,1.8])
    summary,selected=scan.summarize(pool,rows)
    assert summary["fewer_than_requested"] and selected["item_ids"]==["a:0","a:2"]
    pool,rows=pool_rows([.1,.2])
    summary,selected=scan.summarize(pool,rows)
    assert selected["count"]==0 and summary["selected_items"]==0


def test_any_failed_unknown_qualifier_withholds_selection_and_full_pool_is_required():
    pool,rows=pool_rows([2.,2.,2.])
    rows[0]=dict(item_id="a:0",document_id="doc:a:0",split="train",status="error",error="fixture")
    summary,selected=scan.summarize(pool,rows)
    assert summary["rows"]==summary["expected_items"]==3 and summary["failures"]==1
    assert summary["status"]=="incomplete" and selected is None
    with pytest.raises(ValueError,match="entire fixed pool"):
        scan.summarize(pool,rows[1:])
    with pytest.raises(ValueError,match="entire fixed pool"):
        scan.summarize(pool,rows+[rows[1]])


def test_fixed_width_raw_score_matches_native_full_causal_cpu_likelihood():
    torch.manual_seed(111)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=31,hidden_size=24,intermediate_size=40,
        num_hidden_layers=2,num_attention_heads=3,num_key_value_heads=1,head_dim=8,
        tie_word_embeddings=True,attn_implementation="eager")).eval()
    record=dict(prefix=[10,11,12],future=list(range(10,26)))
    actual=scan.raw_score(NeuronCausalBackend(model,checkpoint_trunk=False),record,0)
    with torch.no_grad():
        logits=model(torch.tensor([record["prefix"]+record["future"]])).logits
        lp=logits[:,2:18].log_softmax(-1).gather(-1,torch.tensor([record["future"]])[:,:,None])[0,:,0]
    assert torch.allclose(torch.tensor(actual["token_log_likelihoods"]),lp,atol=1e-5,rtol=0.)
    assert actual["raw_K1_surprisal_nats"]==-actual["token_log_likelihoods"][0]


def test_saved_reference_rejects_failed_or_missing_outcome(tmp_path):
    path=tmp_path/"reference.jsonl"
    row=dict(item_id="dev:a",mode="raw",horizon=16,status="ok",token_log_likelihoods=[-2.]*16)
    path.write_text("unrelated invalid saved row\n"+json.dumps(row)+"\n")
    assert scan.reference_logps(path,"dev:a")==[-2.]*16
    row["status"]="error";path.write_text(json.dumps(row)+"\n")
    with pytest.raises(ValueError,match="Failed"):
        scan.reference_logps(path,"dev:a")
