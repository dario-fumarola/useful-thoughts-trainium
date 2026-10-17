"""CPU semantics for candidate multiplicity, causal boundaries and document reduction."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import train.candidate_probe as probe
from weekend_thinking.artifacts import canonical_hash, sha256_file
from weekend_thinking.core import Format


@pytest.fixture
def fmt():
    return Format((2,), (3, 4), 5, 0, header_ids=(9,), additional_stop_ids=(0,))


class Sampler:
    device = torch.device("cpu")

    def __init__(self):
        self.events = []

    def sync(self):
        self.events.append("sync")

    def seed(self, value):
        self.events.append("seed")
        torch.manual_seed(value)

    def sample(self, prefix, *, fmt, group_size, max_actions, temperature, policy_version):
        self.events.append("sample")
        assert prefix.tolist() == [[10, 11, 12]]
        assert (group_size, max_actions, temperature, policy_version) == (4, 256, .7, 0)
        a = torch.randint(10, 29, (1, 4, 256))
        mask = torch.zeros_like(a, dtype=torch.bool)
        for index, length in enumerate([1, 3, 256, 2]):
            mask[0, index, :length] = True
            if length != 256:
                a[0, index, length-1] = fmt.additional_stop_ids[0] if index == 3 else fmt.stop_id
        return SimpleNamespace(actions=a, action_mask=mask,
            behavior_logps=torch.full(a.shape, -2.), decode_iterations=256)

    def replay(self, prefix, rollout, *, fmt, temperature):
        self.events.append("replay")
        return rollout.behavior_logps.clone()


def test_four_sequential_group_seeds_preserve_empty_stops_caps_and_reproducibility(fmt):
    backend = Sampler()
    seed = probe.group_seed(1729, "a", 0)
    first = probe.generate_group(backend, [10, 11, 12], fmt=fmt, seed=seed, set_seed=backend.seed)
    assert backend.events == ["sync", "seed", "sample", "sync"]
    second = probe.generate_group(backend, [10, 11, 12], fmt=fmt, seed=seed, set_seed=backend.seed)
    assert [r["actions"] for r in first] == [r["actions"] for r in second]
    third = probe.generate_group(backend, [10, 11, 12], fmt=fmt,
        seed=probe.group_seed(1729, "a", 1), set_seed=backend.seed)
    assert first[2]["actions"] != third[2]["actions"]
    assert len({probe.group_seed(1729, key, group) for key in ["a", "b"] for group in range(4)}) == 8
    assert first[0]["content_ids"] == [] and first[0]["stop_type"] == "token:5"
    assert first[2]["n_actions"] == 256 and first[2]["stop_type"] == "length_cap"
    assert first[3]["stop_type"] == "token:0"
    assert all(len(r["actions"]) == len(r["behavior_logps"]) for r in first)
    audited=probe.generate_group(backend,[10,11,12],fmt=fmt,seed=seed,set_seed=backend.seed,audit_replay=True)
    assert audited[0]["cache_replay_max_error"]==0. and audited[0]["cache_replay_status"]=="ok"
    backend.replay=lambda *a,**k: torch.full((1,4,256),-2.01)
    failed=probe.generate_group(backend,[10,11,12],fmt=fmt,seed=seed,set_seed=backend.seed,audit_replay=True)
    assert all(r["status"]=="error" and r["actions"] for r in failed)
    assert [r["actions"] for r in failed]==[r["actions"] for r in first]
    assert [r["behavior_logps"] for r in failed]==[r["behavior_logps"] for r in first]
    assert all(r["group_replay_audit_wall_seconds"]>=0 for r in failed)


def protocol(ids=("a", "b", "c")):
    return dict(selection=dict(item_ids=list(ids), document_ids=["doc:"+key for key in ids]),
        seed=1729, control_seed=17, horizons=[1, 16],
        interfaces=dict(original=[True, 0], reanchor32=[True, 32]),
        strata={key: "Q1" if index % 2 == 0 else "Q2" for index,key in enumerate(ids)},
        limitations=["Cap256 is a bounded search; a zero finite oracle does not prove impossibility"])


def test_controls_keep_global_same_candidate_donor_and_exact_content_lengths():
    p = protocol()
    notes = {(key,i):dict(content_ids=([10+i, 40+i, 70+i] if key=="a"
                else ([] if key=="b" and i==0 else [100+i])))
             for key in p["selection"]["item_ids"] for i in range(16)}
    controls = probe.construct_controls(p, notes, [7, 8])
    assert controls == probe.construct_controls(p, notes, [7, 8])
    assert controls[("a", 0)]["content_ids"]["other"] == [7, 8, 7]
    assert controls[("a", 1)]["content_ids"]["other"] == [101, 101, 101]
    for (key,i), row in controls.items():
        own = notes[(key,i)]["content_ids"]
        assert row["donor_item_id"] != key and row["donor_candidate_index"] == i
        assert all(len(values)==len(own) for values in row["content_ids"].values())
        assert sorted(row["content_ids"]["shuffled"]) == sorted(own)


def complete_score_rows(p):
    rows = []
    for key in p["selection"]["item_ids"]:
        def add(interface, mode, candidate, value):
            rows.append(dict(item_id=key, document_id="doc:"+key, interface=interface,
                mode=mode, candidate_index=candidate, status="ok", token_log_likelihoods=[value]+[0.]*15))
        for mode in probe.RAW_REPEATS:
            add("raw", mode, None, -1.)
        for interface in p["interfaces"]:
            add(interface, "empty", None, -10. if interface=="original" else -20.)
            for i in range(16):
                own = (-8. if i<15 else -40.) if key=="a" else -11.
                for mode in probe.CONTENT_MODES:
                    add(interface, mode, i, own if mode=="own" else -12.)
    return rows


def test_oracle_and_mean_reduce_within_documents_and_use_same_interface_empty():
    p=protocol(("a", "b"));rows=complete_score_rows(p);result=probe.summarize(p,rows)
    oracle=result["groups"]["all"]["original/K1/oracle_max_own_minus_empty"]
    mean=result["groups"]["all"]["original/K1/mean16_own_minus_empty"]
    assert oracle["mean_complete_documents_only"] == .5  # Per-document maxima: +2 and -1.
    assert mean["mean_complete_documents_only"] == -.5  # Per-document 16-sample means: 0 and -1.
    assert oracle["positive_documents"] == 1 and oracle["expected_documents"] == 2
    assert oracle["document_bootstrap"]["documents"] == oracle["document_bootstrap"]["items"] == 2
    repaired=result["groups"]["all"]["reanchor32/K1/oracle_max_own_minus_empty"]
    assert repaired["mean_complete_documents_only"] == 10.5
    control=next(r for r in rows if r["item_id"]=="a" and r["interface"]=="original"
                 and r["mode"]=="other" and r["candidate_index"]==1)
    control["token_log_likelihoods"][0]=0.
    checked=probe.summarize(p,rows)
    document=next(r for r in checked["per_document"] if r["item_id"]=="a"
                  and r["interface"]=="original" and r["horizon"]==1)
    assert document["oracle"]["candidate_index"]==0
    assert document["metrics"]["oracle_selected_own_minus_other"]==4.
    assert document["metrics"]["max16_own_minus_max16_other"]==-8.
    failed=next(r for r in rows if r["item_id"]=="a" and r["interface"]=="original" and r["mode"]=="own")
    failed.update(status="error");failed.pop("token_log_likelihoods")
    result=probe.summarize(p,rows)
    entry=result["groups"]["all"]["original/K1/oracle_max_own_minus_empty"]
    assert result["failures"] == 1 and entry["expected_documents"] == 2 and entry["complete_documents"] == 1
    assert entry["incomplete_item_ids"] == ["a"] and "document_bootstrap" not in entry
    with pytest.raises(ValueError,match="coverage"):
        probe.summarize(p,rows[:-1])


def test_incomplete_sampling_blocks_model_and_future_access(tmp_path, fmt, monkeypatch):
    p=protocol(("a", "b"));(tmp_path/"protocol.json").write_text("{}")
    d=tmp_path/"generation";d.mkdir();(d/"candidates.jsonl").write_text("")
    (d/"run.json").write_text(json.dumps(dict(status="complete",protocol_sha256=sha256_file(tmp_path/"protocol.json"),
        candidates_sha256=sha256_file(d/"candidates.jsonl"),
        first_group_replay_check=dict(status="ok",max_active_logp_error=0.))))
    monkeypatch.setattr(probe,"read_protocol",lambda _: (p,fmt))
    monkeypatch.setattr(probe,"setup_model",lambda _: pytest.fail("Model loading before complete sampling"))
    monkeypatch.setattr(probe,"load_selected",lambda *a,**k: pytest.fail("Future loading before complete sampling"))
    with pytest.raises(ValueError,match="all 16"):
        probe.score(tmp_path,"cpu")


def test_prepare_reuses_exact_fixed_pilot_ids_and_default_reader_choice(tmp_path, monkeypatch):
    ids=[f"dev:{i}" for i in range(32)];selection=dict(item_ids=ids,document_ids=[f"doc:{i}" for i in range(32)],
        split="dev",count=32,item_ids_sha256=canonical_hash(ids),records_sha256="a"*64)
    model=tmp_path/"model";model.mkdir();(model/"ASSET_MANIFEST.json").write_text("{}")
    pilot=dict(selection=selection,per_quartile=8,strata={key:f"Q{i//8+1}" for i,key in enumerate(ids)},
        paths=dict(records="unopened-records.jsonl",format="unopened-format.json",strata="unopened-strata.json"),
        sha256=dict(format="b"*64,strata="c"*64),model_dir=str(model))
    path=tmp_path/"pilot.json";path.write_text(json.dumps(pilot));monkeypatch.setattr(probe,"sources",lambda: {"fixture":"0"*64})
    (tmp_path/"scores.jsonl").write_text("unused fixed pilot score bytes\n")
    (tmp_path/"summary.json").write_text(json.dumps(dict(status="complete",failures=0,
        scores_sha256=sha256_file(tmp_path/"scores.jsonl"))))
    result=probe.prepare(tmp_path/"new",path)
    assert result["selection"] == selection and tuple(result["interfaces"]) == ("original", "reanchor32")
    assert result["cap"] == 256 and result["candidates_per_item"] == 16 and result["group_size"] == 4
    pilot["strata"][ids[0]]="Q2";path.write_text(json.dumps(pilot))
    with pytest.raises(ValueError,match="fixed DEV32"):
        probe.prepare(tmp_path/"bad",path)


def test_pilot_baselines_require_exact_complete_finite_same_interface_rows(tmp_path):
    p=protocol(("a", "b"));path=tmp_path/"pilot-scores.jsonl";p["pilot_scores"]=str(path)
    rows=complete_score_rows(p)
    def write(values):path.write_text("".join(json.dumps(r)+"\n" for r in values))
    write(rows);reference=probe.pilot_baselines(p)
    assert len(reference)==6
    assert reference[("a","original","empty")][0]==-10.
    assert reference[("a","reanchor32","empty")][0]==-20.
    raw=next(r for r in rows if r["item_id"]=="a" and r["interface"]=="raw" and r["mode"]=="raw")
    write([r for r in rows if r is not raw])
    with pytest.raises(ValueError,match="Missing required"):
        probe.pilot_baselines(p)
    raw["token_log_likelihoods"][0]=float("nan");write(rows)
    with pytest.raises(ValueError,match="nonfinite"):
        probe.pilot_baselines(p)
