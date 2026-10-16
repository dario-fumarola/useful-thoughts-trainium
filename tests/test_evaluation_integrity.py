"""Artifact swaps and final-test bypass regressions, entirely CPU/file fixtures."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

from train import evaluation_integrity as integrity
from weekend_thinking.artifacts import canonical_hash, sha256_file
from weekend_thinking.numeric_projection import jsonl_bytes, project_rows


def write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def final_run(tmp_path):
    records = tmp_path/"records.jsonl"
    # The evaluator must not parse any held-out future to establish train order.
    records.write_text('{"id":"train-a","document_id":"a","split":"train","prefix":[1],"future":[2]}\n'
                       '{"id":"test-b","document_id":"b","split":"test","prefix":UNREAD,"future":UNREAD}\n'
                       '{"id":"train-c","document_id":"c","split":"train","prefix":[3],"future":[4]}\n')
    assets = dict(sha256="base-files", manifest_sha256="manifest", model_id="Qwen/base", revision="a"*40)
    spec = dict(terminal_steps=8, site=dict(world_size_per_arm=2),
        assets=dict(records_file=str(records), records_sha256=sha256_file(records),
                    model_manifest_sha256="manifest", model_id="Qwen/base", model_revision="a"*40),
        proposal=dict(optimization=dict(seeds=[1234,5678], global_unique_prefixes_per_update=128,
                                        compute_dtype="fp32")))
    spec["spec_hash"] = canonical_hash(spec)
    checkpoint = tmp_path/"run"/"step-0000008"/"model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"own model fixture")
    for rank in range(2):
        (checkpoint.parent/f"rank-{rank:05d}.pt").write_bytes(f"optimizer {rank}".encode())
    files = {p.name:dict(bytes=p.stat().st_size,sha256=sha256_file(p)) for p in checkpoint.parent.iterdir()}
    metadata = dict(arm="thought",seed=1234,spec_hash=spec["spec_hash"],
        order_hash=integrity.training_order_hash(spec,1234),global_batch=128,accumulation=64,
        parameter_dtype="fp32",compute_dtype="fp32",counters=dict(source_records=8*128))
    complete = dict(schema="neuron-v1",step=8,world_size=2,spec_hash=spec["spec_hash"],
                    run_id="run",metadata=metadata,files=files)
    write_json(checkpoint.parent/"COMPLETE.json",complete)
    identity = integrity.final_model_identity(checkpoint,assets,spec,arm="thought",seed=1234)
    b0 = integrity.final_model_identity(None,assets,spec,arm="B0",seed=None)
    frozen = dict(schema=integrity.FREEZE_SCHEMA,status="frozen",frozen_utc="2026-10-16T22:00:00Z",
                  training_spec=spec,models=[identity,b0],protocols=[])
    path=tmp_path/"frozen.json";write_json(path,frozen)
    return path,checkpoint,assets,frozen,complete


def test_exact_terminal_and_b0_pass_but_arbitrary_model_file_does_not(final_run,tmp_path):
    path,checkpoint,assets,frozen,_=final_run
    expected=frozen["models"][0]
    assert integrity.require_final_model(path,checkpoint,assets,arm="thought",seed=1234)==expected
    assert integrity.require_final_model(path,None,assets,arm="B0",seed=None)==frozen["models"][1]
    loose=tmp_path/"model.pt";loose.write_bytes(checkpoint.read_bytes())
    with pytest.raises(FileNotFoundError):
        integrity.require_final_model(path,loose,assets,arm="thought",seed=1234)
    with pytest.raises(ValueError,match="B0"):
        integrity.require_final_model(path,checkpoint,assets,arm="B0",seed=None)
    with pytest.raises(ValueError,match="pretrained"):
        integrity.require_final_model(path,None,dict(assets,revision="b"*40),arm="B0",seed=None)


@pytest.mark.parametrize("change",["early-step","wrong-arm","wrong-seed","wrong-order","wrong-spec","missing-rank"])
def test_reject_lineage_or_exposure_swaps_even_with_a_fresh_complete_file(final_run,change):
    path,checkpoint,assets,_,complete=final_run
    if change=="early-step":complete["step"]=7
    if change=="wrong-arm":complete["metadata"]["arm"]="vanilla"
    if change=="wrong-seed":complete["metadata"]["seed"]=5678
    if change=="wrong-order":complete["metadata"]["order_hash"]="other order"
    if change=="wrong-spec":complete["spec_hash"]="other spec"
    if change=="missing-rank":complete["files"].pop("rank-00001.pt")
    write_json(checkpoint.parent/"COMPLETE.json",complete)
    with pytest.raises(ValueError,match="terminal|lineage"):
        integrity.require_final_model(path,checkpoint,assets,arm="thought",seed=1234)


@pytest.mark.parametrize("filename",["model.pt","rank-00001.pt"])
def test_hash_every_file_including_unused_inference_optimizer_shards(final_run,filename):
    path,checkpoint,assets,_,_=final_run
    target=checkpoint.parent/filename
    original=target.read_bytes();target.write_bytes(bytes([original[0]^1])+original[1:])
    with pytest.raises(ValueError,match="integrity"):
        integrity.require_final_model(path,checkpoint,assets,arm="thought",seed=1234)


def test_inventory_training_spec_and_internal_weight_metadata_are_bound(final_run):
    path,checkpoint,assets,frozen,_=final_run
    identity=frozen["models"][0]
    integrity.validate_loaded_checkpoint({key:identity[key] for key in ("step","run_id","spec_hash")},identity)
    with pytest.raises(ValueError,match="metadata"):
        integrity.validate_loaded_checkpoint(dict(step=7,run_id="run",spec_hash=identity["spec_hash"]),identity)
    frozen["models"]=[frozen["models"][1]];write_json(path,frozen)
    with pytest.raises(ValueError,match="inventory"):
        integrity.require_final_model(path,checkpoint,assets,arm="thought",seed=1234)
    frozen["training_spec"]["terminal_steps"]=9;write_json(path,frozen)
    with pytest.raises(ValueError,match="specification hash"):
        integrity.read_evaluation_freeze(path)


def test_dependency_closure_binds_likelihood_imported_decoder_and_parser(tmp_path):
    root=Path(__file__).resolve().parents[1]
    current=integrity.evaluation_dependencies()
    assert {"train/evaluate.py","train/neuron_backend.py","train/runtime_compat.py",
            "weekend_thinking/core.py","weekend_thinking/pretrained.py",
            "weekend_thinking/evaluation.py","weekend_thinking/numeric_projection.py"} <= current.keys()
    for name in current:
        target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,target)
    helper=tmp_path/"train/evaluate.py";helper.write_text(helper.read_text()+"\n# changed decoder\n")
    assert integrity.evaluation_dependencies(tmp_path)["train/evaluate.py"] != current["train/evaluate.py"]


@pytest.fixture
def numeric_run(final_run,monkeypatch,tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    frozen_path,_,_,frozen,_=final_run
    source=tmp_path/"canonical.parquet"
    source_rows=[dict(question="One plus two?",answer="Work.\n#### 3"),
                 dict(question="Four less one?",answer="Work.\n#### 3")]
    pq.write_table(pa.Table.from_pylist(source_rows),source)
    monkeypatch.setattr(integrity,"GSM8K_TEST_SHA256",sha256_file(source))
    monkeypatch.setattr(integrity,"evaluation_dependencies",lambda:{"fixture.py":"locked"})
    questions,gold=project_rows(source_rows)
    q=tmp_path/"questions.jsonl";q.write_bytes(jsonl_bytes(questions))
    g=tmp_path/"gold.jsonl";g.write_bytes(jsonl_bytes(gold))
    manifest=dict(schema="numeric-scoring-v1",task="gsm8k",task_revision=integrity.GSM8K_REVISION,
        split="test",parser=integrity.PARSER,dependencies=integrity.evaluation_dependencies(),
        canonical_source_file=str(source),canonical_source_sha256=sha256_file(source),
        question_file=str(q),question_file_sha256=sha256_file(q),gold_file_sha256=sha256_file(g),
        item_ids_sha256=canonical_hash([row["item_id"] for row in questions]),
        training_spec_hash=frozen["training_spec"]["spec_hash"])
    identity=frozen["models"][0]
    protocol=dict(task="gsm8k",mode="one_thought",task_revision=manifest["task_revision"],task_file_sha256=sha256_file(q),
        item_ids_sha256=manifest["item_ids_sha256"],dependencies=integrity.evaluation_dependencies())
    frozen.update(scoring_manifests=[manifest],protocols=[protocol]);write_json(frozen_path,frozen)
    manifest_path=tmp_path/"scoring.json";write_json(manifest_path,manifest)
    predictions=tmp_path/"predictions.jsonl"
    rows=[dict(item_id=row["item_id"],split="test",run_id="generation",status="ok" if index==0 else "error",
        generated_text="#### 3",protocol_sha256=canonical_hash(protocol),model_sha256=identity["base_assets_sha256"],
        checkpoint_sha256=identity["model_sha256"],model_identity_sha256=canonical_hash(identity))
        for index,row in enumerate(questions)]
    predictions.write_bytes(jsonl_bytes(rows))
    report=dict(qualification=False,status="complete",run_id="generation",protocol=protocol,
        protocol_sha256=canonical_hash(protocol),model_identity=identity,
        model=dict(sha256=identity["base_assets_sha256"]),checkpoint=dict(sha256=identity["model_sha256"]),
        expected_items=2,predictions_sha256=sha256_file(predictions))
    write_json(tmp_path/"run.json",report)
    return frozen_path,manifest_path,g,predictions,manifest,frozen,report


def test_locked_canonical_numeric_manifest_and_failures_retained(numeric_run,monkeypatch,tmp_path):
    frozen,manifest,gold,predictions,*_=numeric_run
    location=Path(__file__).resolve().parents[1]/"scripts/score_numeric_answers.py"
    spec=importlib.util.spec_from_file_location("numeric_cli_fixture",location)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    out=tmp_path/"scored.jsonl"
    monkeypatch.setattr(sys,"argv",[str(location),"--gold",str(gold),"--predictions",str(predictions),
        "--out",str(out),"--frozen-protocol",str(frozen),"--scoring-manifest",str(manifest)])
    module.main()
    summary=json.loads(out.with_suffix(".summary.json").read_text())
    assert summary["items"]==2 and summary["accuracy"]==.5 and summary["qualification"] is False
    assert summary["scores_sha256"]==sha256_file(out)
    rows=[json.loads(line) for line in out.read_text().splitlines()]
    assert all(row["condition"]==summary["condition"] for row in rows)
    assert summary["condition"]["arm"]=="thought" and summary["condition"]["mode"]=="one_thought"


def test_arbitrary_gold_cannot_be_relocked_as_canonical(numeric_run):
    frozen_path,manifest_path,gold,_,manifest,frozen,_=numeric_run
    gold.write_text(gold.read_text().replace("#### 3","#### 99"))
    manifest["gold_file_sha256"]=sha256_file(gold)
    frozen["scoring_manifests"]=[manifest];write_json(frozen_path,frozen);write_json(manifest_path,manifest)
    with pytest.raises(ValueError,match="canonical benchmark projection"):
        integrity.require_numeric_scoring_manifest(frozen_path,manifest_path,gold)


def test_unlocked_parser_or_prediction_model_is_rejected(numeric_run):
    frozen_path,manifest_path,gold,predictions,manifest,frozen,report=numeric_run
    manifest["dependencies"]={"fixture.py":"different"}
    frozen["scoring_manifests"]=[manifest];write_json(frozen_path,frozen);write_json(manifest_path,manifest)
    with pytest.raises(ValueError,match="dependency binding"):
        integrity.require_numeric_scoring_manifest(frozen_path,manifest_path,gold)
    report["model_identity"]=dict(report["model_identity"],arm="vanilla")
    write_json(predictions.with_name("run.json"),report)
    with pytest.raises(ValueError,match="frozen generation"):
        integrity.validate_numeric_predictions(frozen,manifest,predictions)


@pytest.mark.parametrize("kind",["gsm8k","arc_challenge"])
def test_final_entrypoints_cannot_use_protocol_only_receipt_before_model_or_gold_load(tmp_path,monkeypatch,kind):
    from train import evaluate,evaluate_likelihood
    from weekend_thinking.core import Format
    from dataclasses import asdict
    module=evaluate if kind=="gsm8k" else evaluate_likelihood
    question=tmp_path/"task.jsonl"
    item=dict(item_id=f"{kind}:test:00000",task=kind,split="test")
    item.update(dict(question="Question?") if kind=="gsm8k" else dict(context="Question?",choices=[" A"," B"]))
    question.write_bytes(jsonl_bytes([item]))
    fmt=tmp_path/"format.json";write_json(fmt,dict(format=asdict(Format((2,),(3,),4,0))))
    old_receipt=tmp_path/"old-protocol-only.json";write_json(old_receipt,dict(status="frozen",protocols=[]))
    assets=dict(files={"config.json":dict(sha256="a",bytes=1)},sha256="fixture")
    monkeypatch.setattr(module,"asset_provenance",lambda path:assets)
    # Even matching the old protocol-only gate must not bypass the model inventory gate.
    monkeypatch.setattr(module,"require_frozen_protocol",lambda *args:None)
    def forbidden(*args,**kwargs):
        pytest.fail("Final model/gold was loaded before the complete freeze gate")
    monkeypatch.setattr(module,"load_base",forbidden)
    argv=[kind,"--model-dir",str(tmp_path),"--task-file",str(question),"--format",str(fmt),
          "--task-revision",integrity.GSM8K_REVISION if kind=="gsm8k" else module.REVISIONS[kind],
          "--frozen-protocol",str(old_receipt),"--output",str(tmp_path/"out")]
    if kind=="gsm8k":
        prompt=tmp_path/"prompt.json";write_json(prompt,dict(instruction="Finish #### <number>.",demonstrations=[]))
        argv += ["--prompt-spec",str(prompt),"--mode","direct"]
    else:
        gold=tmp_path/"gold.jsonl";gold.write_text("GOLD MUST NOT BE PARSED")
        monkeypatch.setattr(module,"read_gold",forbidden)
        argv.insert(1,"run");argv += ["--task",kind,"--gold-file",str(gold)]
    monkeypatch.setattr(sys,"argv",argv)
    with pytest.raises(ValueError,match="evaluation-freeze-v2"):
        module.main()
