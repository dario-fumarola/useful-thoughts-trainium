"""Rescore the closed TRAIN128/G16 bundle with notes before the full prefix.

No generation, training or selection occurs here. All eight score shards must
close before a TRAIN oracle/ordinary-mean summary can be produced.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

from train import candidate_corpus as corpus, prepend_probe, candidate_probe as shared
from train.evaluate_local import load_selected, setup_model
from train.evaluate_likelihood import score_continuations
from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file, utc_now
from weekend_thinking.core import Format

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "candidate-prepend-train128-g16-v1"
CORPUS = ROOT/"local/nvme/diagnostics/candidate-corpus-train128-g16-v1"
PILOT = ROOT/"local/nvme/diagnostics/prepend-reader-dev32-v1"


def sources():
    return {**corpus.sources(), "train/prepend_probe.py": sha256_file(prepend_probe.__file__),
            "train/candidate_prepend.py": sha256_file(__file__)}


def closed_corpus(directory):
    """Reuse the actual all2048-candidate/global-control barrier, never rebuild it."""
    p,fmt = corpus.read_protocol(directory)
    notes,controls,barrier = corpus.global_inputs(directory,p,fmt)
    summary = json.loads((directory/"summary.json").read_text())
    if (summary["schema"] != corpus.SCHEMA or summary["split"] != "train"
            or summary["status"] != "complete" or summary["failures"] != 0
            or summary["rows"] != p["expected_global_score_rows"]
            or summary["protocol_sha256"] != sha256_file(directory/"protocol.json")
            or [r["shard"] for r in summary["scoring"]] != list(range(8))):
        raise ValueError("Require the completed original TRAIN corpus, not partial outcomes")
    for receipt in summary["scoring"]:
        d = directory/"scoring"/f"shard-{receipt['shard']:02d}"
        run = json.loads((d/"run.json").read_text())
        if (receipt["run_sha256"] != sha256_file(d/"run.json")
                or receipt["scores_sha256"] != sha256_file(d/"scores.jsonl")
                or run["status"] != "complete" or run["schema"] != corpus.SCHEMA
                or run["protocol_sha256"] != sha256_file(directory/"protocol.json")
                or run["barrier_sha256"] != sha256_file(directory/"global/COMPLETE.json")
                or run["model"] != barrier["model"] or run["checkpoint"] is not None):
            raise ValueError("Original corpus scoring/source identity mismatch")
    return p,notes,controls,barrier,summary


def prepare(output, corpus_dir=CORPUS, pilot_dir=PILOT):
    from transformers import AutoTokenizer
    old,notes,_,barrier,summary = closed_corpus(corpus_dir)
    pilot = json.loads((pilot_dir/"protocol.json").read_text())
    pilot_summary = json.loads((pilot_dir/"summary.json").read_text())
    if (pilot_summary["status"] != "complete" or pilot_summary["failures"] != 0
            or pilot_summary["rows"] != 704
            or pilot_summary["protocol_sha256"] != sha256_file(pilot_dir/"protocol.json")
            or pilot_summary["scores_sha256"] != sha256_file(pilot_dir/"scores.jsonl")):
        raise ValueError("Require the completed prepend interface development probe")
    tok = AutoTokenizer.from_pretrained(old["model_dir"],local_files_only=True,trust_remote_code=False)
    separator = tok.encode("\n\n",add_special_tokens=False)
    if not separator or separator != pilot["separator_ids"]:
        raise ValueError("Prepend separator differs from the measured interface")
    paths = dict(corpus_protocol=corpus_dir/"protocol.json",corpus_summary=corpus_dir/"summary.json",
                 corpus_barrier=corpus_dir/"global/COMPLETE.json",controls=corpus_dir/"global/controls.jsonl",
                 pilot_protocol=pilot_dir/"protocol.json",pilot_summary=pilot_dir/"summary.json",
                 pilot_scores=pilot_dir/"scores.jsonl")
    p = dict(schema=SCHEMA,experiment_id=output.name,prepared_utc=utc_now(),purpose="Adaptive TRAIN reader-interface rescore before warmstart label choice; no new samples or held-out claim",
        corpus_dir=str(corpus_dir.resolve()),paths={k:str(v.resolve()) for k,v in paths.items()},
        hashes={k:sha256_file(v) for k,v in paths.items()},generation=barrier["generation"],
        original_scoring=summary["scoring"],model=barrier["model"],model_dir=old["model_dir"],
        model_manifest_sha256=old["model_manifest_sha256"],source_files=sources(),selection=old["selection"],
        thought_format=old["thought_format"],strata=old["strata"],separator_ids=separator,
        interfaces={"prepend_plain":dict(content_then_separator_then_full_prefix=True,separator_ids=separator)},
        serialization="content_ids + encode('\\n\\n') + all512 prefix_ids; empty uses separator + full prefix",
        horizons=[1,16],width=1024,compute="fp32",shards=8,shard_rule=old["shard_rule"],
        candidates_per_item=16,cap=256,temperature=.7,raw_repeats=shared.RAW_REPEATS,content_modes=shared.CONTENT_MODES,
        expected_global_candidates=len(notes),expected_global_score_rows=128*(3+1+16*4),expected_shard_score_rows=16*(3+1+16*4),
        raw_reference_atol=.002,raw_reference_rule="raw/raw versus TRAIN scanner; repeat32/64 versus identical completed original-corpus raw-repeat contexts",
        timing_unit="No new sampling; reused generation times counted once per(item_id,group), new score rows timed separately",
        limitations=["All128 selected on TRAIN target surprisal; not held-out utility.","Same2048 previously sampled candidates/global controls; no new generation.","Future-selected oracle is potential TRAIN supervision, not a deployed prefix-only policy.","Adaptive interface choice follows DEV32 development; no confirmatory claim."])
    output.mkdir(parents=True,exist_ok=False);atomic_json(output/"protocol.json",p)
    return p


def read_protocol(output):
    p = json.loads((output/"protocol.json").read_text())
    if (p["schema"] != SCHEMA or p["source_files"] != sources() or p["selection"]["split"] != "train"
            or p["selection"]["count"] != 128 or p["compute"] != "fp32" or p["width"] != 1024
            or set(p["interfaces"]) != {"prepend_plain"}):
        raise ValueError("Prepend source/recipe/selection changed")
    for key,path in p["paths"].items():
        if sha256_file(path) != p["hashes"][key]:raise ValueError(f"Changed prepend input: {key}")
    return p,Format.from_dict(p["thought_format"])


def inputs(p):
    old,notes,controls,barrier,summary = closed_corpus(Path(p["corpus_dir"]))
    if (old["selection"] != p["selection"] or barrier["generation"] != p["generation"]
            or barrier["model"] != p["model"] or summary["scoring"] != p["original_scoring"]):
        raise ValueError("Changed complete corpus lineage")
    return old,notes,controls,barrier


def raw_references(p,old):
    ids = set(p["selection"]["item_ids"]);result = {}
    def add(key,values,status):
        if key in result or status != "ok" or len(values) != 16 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in values):
            raise ValueError("Invalid/duplicate raw reference")
        result[key]=values
    for line in Path(old["paths"]["raw_references"]).read_text().splitlines():
        row=json.loads(line)
        if row["item_id"] in ids:add((row["item_id"],"raw"),row.get("token_log_likelihoods",[]),row["status"])
    for shard in range(8):
        path=Path(p["corpus_dir"])/"scoring"/f"shard-{shard:02d}"/"scores.jsonl"
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if row["interface"]=="raw" and row["mode"] in {"repeat32","repeat64"}:
                add((row["item_id"],row["mode"]),row.get("token_log_likelihoods",[]),row["status"])
    if set(result)!={(key,mode) for key in ids for mode in shared.RAW_REPEATS}:
        raise ValueError("Every raw/raw-repeat context requires its identical reference")
    return result


def contexts(prefix,controls,p):
    values=[("raw",mode,None,list(prefix)+(list(prefix[-n:]) if n else [])) for mode,n in shared.RAW_REPEATS.items()]
    values.append(("prepend_plain","empty",None,p["separator_ids"]+list(prefix)))
    values.extend(("prepend_plain",mode,i,controls[i]["content_ids"][mode]+p["separator_ids"]+list(prefix)) for i in range(16) for mode in shared.CONTENT_MODES)
    return values


def score(output,shard,device):
    p,fmt=read_protocol(output);selected=corpus.shard_selection(p["selection"],shard)
    old,notes,controls,barrier=inputs(p)  # All2048 notes/global controls before models/futures.
    references=raw_references(p,old)
    backend,_,checkpoint,model,_=setup_model(SimpleNamespace(model_dir=Path(p["model_dir"]),checkpoint=None,device=device))
    if checkpoint is not None or model!=barrier["model"]:raise ValueError("Require unchanged B0 reader")
    records=load_selected(Path(selected["records_file"]),selected,include_future=True)
    directory=output/"scoring"/f"shard-{shard:02d}";directory.mkdir(parents=True,exist_ok=False)
    report=dict(schema=SCHEMA,status="running",started_utc=utc_now(),shard=shard,model=model,checkpoint=None,
        protocol_sha256=sha256_file(output/"protocol.json"),corpus_barrier_sha256=p["hashes"]["corpus_barrier"],expected_rows=p["expected_shard_score_rows"])
    atomic_json(directory/"run.json",report);count=0;maximum=0.;raw_count=0
    try:
        with (directory/"scores.jsonl").open("x") as stream:
            for record in records:
                key=record["item_id"]
                if len(record["prefix"])!=512 or len(record["future"])!=16 or any(notes[key,i]["prefix_sha256"]!=canonical_hash(record["prefix"]) for i in range(16)):
                    raise ValueError("Saved writer/full512 reader prefix or target width mismatch")
                for interface,mode,i,context in contexts(record["prefix"],{n:controls[key,n] for n in range(16)},p):
                    started=time.monotonic();row=dict(item_id=key,document_id=record["document_id"],split="train",shard=shard,interface=interface,mode=mode,candidate_index=i,context_tokens=len(context))
                    if i is not None:row.update(n_note_content_tokens=notes[key,i]["n_content_tokens"],n_note_actions=notes[key,i]["n_actions"],note_stop_type=notes[key,i]["stop_type"])
                    try:
                        values=score_continuations(backend,context,[record["future"]],total_width=1024,future_width=16,pad_id=fmt.pad_id)[0].token_log_likelihoods
                        if len(values)!=16 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in values):raise ValueError("Require sixteen finite token likelihoods")
                        row.update(status="ok",token_log_likelihoods=values)
                        if interface=="raw":
                            error=max(abs(a-b) for a,b in zip(values,references[key,mode]));maximum=max(maximum,error);raw_count+=1
                            row["raw_reference_max_token_error"]=error
                            if error>.002:raise ValueError(f"Raw context parity failed: {error}")
                    except Exception as exc:row.update(status="error",error=f"{type(exc).__name__}: {exc}")
                    row["wall_seconds"]=time.monotonic()-started;stream.write(json.dumps(row,allow_nan=False)+"\n");stream.flush();count+=1
                    if row["status"]!="ok":raise RuntimeError("Failed actual reader row retained; no retry/filtering")
                print(json.dumps(dict(shard=shard,rows=count,utc=utc_now())),flush=True)
        if count!=p["expected_shard_score_rows"] or raw_count!=48:raise ValueError("Incomplete fixed score shard")
        report.update(status="complete",scores_sha256=sha256_file(directory/"scores.jsonl"))
    except BaseException as exc:report.update(status="failed",error=f"{type(exc).__name__}: {exc}");raise
    finally:
        report.update(rows=count,raw_reference_rows=raw_count,raw_reference_max_token_error=maximum,finished_utc=utc_now())
        atomic_json(directory/"run.json",report,overwrite=True)


def reduce_rows(p,rows):
    result=corpus.reduce_rows(p,rows)
    result.update(schema=SCHEMA,experiment_id=p["experiment_id"],split="train",
        statistics_implementation="candidate_corpus.reduce_rows numerical reducer; actual prepend_plain/TRAIN schema and experiment identity")
    return result


def summarize(output):
    p,_=read_protocol(output);_,notes,_,barrier=inputs(p);rows=[];receipts=[];runs=[];maximum=0.
    for shard in range(8):
        d=output/"scoring"/f"shard-{shard:02d}";run=json.loads((d/"run.json").read_text());path=d/"scores.jsonl"
        if (run["status"]!="complete" or run["schema"]!=SCHEMA or run["shard"]!=shard
                or run["protocol_sha256"]!=sha256_file(output/"protocol.json")
                or run["corpus_barrier_sha256"]!=p["hashes"]["corpus_barrier"] or run["model"]!=barrier["model"]
                or run["checkpoint"] is not None or run["scores_sha256"]!=sha256_file(path)
                or run["rows"]!=p["expected_shard_score_rows"] or run["raw_reference_rows"]!=48
                or not 0<=run["raw_reference_max_token_error"]<=.002):raise ValueError("All eight complete bound scoring shards are required")
        local=[json.loads(s) for s in path.read_text().splitlines()];ids=set(corpus.shard_selection(p["selection"],shard)["item_ids"])
        if len(local)!=run["rows"] or set(r["item_id"] for r in local)!=ids or any(r["shard"]!=shard or r["split"]!="train" or r["status"]!="ok" or len(r.get("token_log_likelihoods",[]))!=16 or not all(math.isfinite(v) for v in r["token_log_likelihoods"]) for r in local):raise ValueError("Invalid complete shard coverage/status")
        rows.extend(local);maximum=max(maximum,run["raw_reference_max_token_error"])
        runs.append(run)
        receipts.append(dict(shard=shard,run_sha256=sha256_file(d/"run.json"),scores_sha256=sha256_file(path)))
    result=reduce_rows(p,rows)
    groups={(r["item_id"],r["group"]):r for r in notes.values()}
    cost=dict(new_sampling_seconds=0,reused_sampling_group_count=len(groups),
        reused_sampling_group_seconds_counted_once=sum(r["group_sampling_wall_seconds"] for r in groups.values()),
        reused_replay_audit_group_seconds_counted_once=sum(r["group_replay_audit_wall_seconds"] for r in groups.values()),
        new_score_rows=len(rows),new_score_row_wall_seconds_sum=sum(r["wall_seconds"] for r in rows),
        new_score_process_wall_seconds_sum=sum((datetime.fromisoformat(r["finished_utc"])-datetime.fromisoformat(r["started_utc"])).total_seconds() for r in runs),
        new_score_stage_span_seconds=(max(datetime.fromisoformat(r["finished_utc"]) for r in runs)-min(datetime.fromisoformat(r["started_utc"]) for r in runs)).total_seconds())
    result.update(protocol_sha256=sha256_file(output/"protocol.json"),corpus_summary_sha256=p["hashes"]["corpus_summary"],scoring=receipts,completed_utc=utc_now(),sampling_counts=dict(Counter(r["stop_type"] for r in notes.values())),empty_candidates=sum(r["n_content_tokens"]==0 for r in notes.values()),raw_reference_rows=384,raw_reference_max_token_error=maximum,cost=cost)
    atomic_json(output/"summary.json",result)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("command",choices=("prepare","score","summarize"))
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--corpus",type=Path,default=CORPUS)
    parser.add_argument("--pilot",type=Path,default=PILOT);parser.add_argument("--shard-index",type=int,choices=range(8))
    parser.add_argument("--device",choices=("cpu","xla"),default="xla");a=parser.parse_args()
    if a.command=="prepare":
        p=prepare(a.output,a.corpus,a.pilot);print(json.dumps(dict(prepared=str(a.output),items=128,candidates=2048,expected_score_rows=p["expected_global_score_rows"])))
    elif a.command=="summarize":summarize(a.output)
    elif a.shard_index is None:parser.error("score requires shard-index0..7")
    else:score(a.output,a.shard_index,a.device)


if __name__=="__main__":main()
