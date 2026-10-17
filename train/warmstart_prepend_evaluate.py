"""Rescore fixed fresh DEV64/G1 notes with content + separator + full prefix.

No sampling, oracle selection or training occurs here. Original notes, global
controls, writer provenance and the unchanged B0 raw reference remain fixed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import time

from train import warmstart_evaluate as original
from train.evaluate_likelihood import score_continuations
from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file, utc_now
from weekend_thinking.stats import paired_cluster_bootstrap

SCHEMA = "warmstart-prepend-dev64-g1-v1"
MODES = ("raw", "empty", "own", "filler", "shuffled", "other")
ROWS_PER_SHARD = 16*len(MODES)
BASELINE = Path("local/nvme/diagnostics/warmstart-dev64-cap256-B0-v1")


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def sources():
    return {**original.sources(), "train/warmstart_prepend_evaluate.py":sha256_file(__file__)}


def finite_vector(value):
    return isinstance(value,list) and len(value)==16 and all(
        type(v) in (int,float) and math.isfinite(v) for v in value)


def check_identity(actual, expected, manifest):
    if (any(actual.get(k)!=v for k,v in expected.items())
            or actual.get("base_manifest_sha256")!=manifest or not actual.get("model")):
        raise ValueError("Writer/reader model identity or base manifest differs")
    loaded = actual.get("loaded_checkpoint")
    if ((expected["id"]=="B0" and loaded is not None)
            or expected["id"]!="B0" and (not loaded or loaded.get("sha256")!=expected["checkpoint_sha256"])):
        raise ValueError("Loaded checkpoint SHA differs from the explicit model identity")


def closed_bundle(directory):
    """Validate all64 notes/global donors before any new future or model access."""
    p,fmt = original.read_protocol(directory)
    selected=p["selection"]; ids=selected["item_ids"]
    if (selected["split"]!="dev" or selected["count"]!=64 or len(set(ids))!=64
            or len(ids)!=64 or len(set(selected["document_ids"]))!=64
            or selected["item_ids_sha256"]!=canonical_hash(ids)
            or set(selected["strata"])!=set(ids)
            or sorted(Counter(selected["strata"].values()).values())!=[16]*4
            or (p["cap"],p["groups_per_prefix"],p["temperature"],p["batch_size"])!=(256,1,.7,4)):
        raise ValueError("Require the fixed DEV64/quartiles and fresh G1/cap256 policy")
    notes,writer=original.load_notes(directory,p,fmt)
    check_identity(writer,p["writer"],p["model_manifest_sha256"])
    receipt=read(directory/"controls.complete.json"); path=directory/"controls.jsonl"
    controls=original.make_controls(p,notes,receipt["filler_ids"])
    if (receipt["status"]!="complete" or receipt["items"]!=64 or receipt["writer"]!=writer
            or receipt["protocol_sha256"]!=sha256_file(directory/"protocol.json")
            or receipt["notes_sha256"]!=canonical_hash(notes)
            or receipt["controls_sha256"]!=sha256_file(path) or rows(path)!=controls):
        raise ValueError("Require the original complete global-control receipt/content")
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(p["model_dir"],local_files_only=True,trust_remote_code=False)
    if tok.encode(p["filler_text"],add_special_tokens=False)!=receipt["filler_ids"]:
        raise ValueError("Global filler differs from the original native-token text")
    if tok.encode("\n\n",add_special_tokens=False)!=[271]:
        raise ValueError("Prepend separator must be the native double-newline token271")
    paths=[directory/"protocol.json",directory/"controls.complete.json",path]
    generation=[]
    for shard in range(4):
        d=directory/"notes"/f"shard-{shard}"; run=read(d/"run.json")
        if (run["shard"]!=shard or run["notes"]!=16 or run["sampling_batches"]!=4
                or any(not isinstance(run.get(k),(int,float)) or not math.isfinite(run[k]) or run[k]<0
                       for k in ("batch_sampling_wall_seconds_total","batch_replay_audit_wall_seconds_total"))):
            raise ValueError("Incomplete original generation count/cost receipt")
        generation.append(run);paths.extend([d/"run.json",d/"notes.jsonl"])
    return p,fmt,notes,writer,{r["item_id"]:r for r in controls},generation,paths


def raw_baseline(directory):
    p,_,notes,writer,_,_,paths=closed_bundle(directory)
    if p["writer"]!=original.model_identity("B0"):
        raise ValueError("Raw reference must use the existing unchanged B0 DEV64 bundle")
    reference={}; identity=None
    for shard in range(4):
        d=directory/"readers/B0"/f"shard-{shard}"; run=read(d/"run.json"); path=d/"scores.jsonl"
        check_identity(run["reader"],original.model_identity("B0"),p["model_manifest_sha256"])
        if (run["status"]!="complete" or run["shard"]!=shard or run["rows"]!=112
                or run["expected_rows"]!=112 or run["item_ids"]!=original.shard_ids(p,shard)
                or run["writer"]!=writer or run["notes_sha256"]!=canonical_hash(notes)
                or run["controls_sha256"]!=sha256_file(directory/"controls.jsonl")
                or run["protocol_sha256"]!=sha256_file(directory/"protocol.json")
                or run["scores_sha256"]!=sha256_file(path)):
            raise ValueError("All four completed, bound B0 raw-reference score shards required")
        if identity is not None and run["reader"]!=identity:
            raise ValueError("Raw-reference reader differs across shards")
        identity=run["reader"]; values=rows(path)
        expected=[(key,mode) for key in original.shard_ids(p,shard)
                  for mode in ("raw","raw_repeat32",*original.MODES)]
        documents=dict(zip(p["selection"]["item_ids"],p["selection"]["document_ids"]))
        if ([(r["item_id"],r["mode"]) for r in values]!=expected or any(
            r["status"]!="ok" or r["document_id"]!=documents[r["item_id"]]
            or r["writer_id"]!="B0" or r["reader_id"]!="B0"
            or not finite_vector(r.get("token_log_likelihoods")) for r in values)):
            raise ValueError("Incomplete or invalid original B0 reference matrix")
        reference.update({r["item_id"]:r["token_log_likelihoods"] for r in values if r["mode"]=="raw"})
        paths.extend([d/"run.json",path])
    return p,reference,identity,paths


def prepare(output,bundle,baseline,reader):
    if reader!=original.model_identity(reader["id"],reader["checkpoint"],reader["checkpoint_sha256"]):
        raise ValueError("Require the explicit canonical reader identity/checkpoint SHA")
    p,_,_,writer,_,_,paths=closed_bundle(bundle)
    ref,_,baseline_identity,reference_paths=raw_baseline(baseline)
    if any(p[k]!=ref[k] for k in ("selection","model_dir","model_manifest_sha256","format_sha256")):
        raise ValueError("Writer bundle and B0 reference require identical DEV64/base/format")
    protocol=dict(schema=SCHEMA,experiment_id=output.name,prepared_utc=utc_now(),
        purpose="Fresh G1 student-writer development, fixed global controls; no oracle or final-test claim",
        bundle=str(bundle.resolve()),baseline=str(baseline.resolve()),writer=writer,reader=reader,
        baseline_reader=baseline_identity,selection=p["selection"],model_dir=p["model_dir"],
        model_manifest_sha256=p["model_manifest_sha256"],source_files=sources(),
        input_hashes={str(path.resolve()):sha256_file(path) for path in paths+reference_paths},
        interface="prepend_plain",serialization="content_ids + [271] + full original512 prefix; raw=prefix",
        separator_ids=[271],modes=list(MODES),width=1024,horizons=[1,16],compute="fp32",
        shards=4,shard_size=16,expected_shard_rows=ROWS_PER_SHARD,expected_rows=384,
        raw_reference_atol=.002,raw_reference_rule="Only unchanged B0 reader must match existing B0 raw token logps",
        primary="Fixed B0 reader, K16 fresh G1 own-minus-empty/filler/shuffled/other; K1 secondary",
        bootstrap=dict(unit="document",repeats=5000,seed=17),
        limitations=["Exploratory fixed DEV64, not official final/test or training-seed uncertainty.",
            "Notes and global controls are reused unchanged; no future-selected winner.",
            "Adapted readers are secondary crossed-reader diagnostics, not the fixed-reader primary.",
            "Prepending changes the reader serialization; no reanchor32/header/close is inserted."])
    output.mkdir(parents=True,exist_ok=False);atomic_json(output/"protocol.json",protocol)
    return protocol


def read_protocol(output):
    p=read(output/"protocol.json")
    if (p["schema"]!=SCHEMA or p["source_files"]!=sources() or p["modes"]!=list(MODES)
            or p["separator_ids"]!=[271] or p["width"]!=1024 or p["compute"]!="fp32"):
        raise ValueError("Prepend evaluation source/recipe changed")
    for path,digest in p["input_hashes"].items():
        if sha256_file(path)!=digest:raise ValueError(f"Changed original note/control/reference input: {path}")
    return p


def contexts(prefix,control):
    if len(prefix)!=512:raise ValueError("Require the full original512-token prefix")
    return [("raw",list(prefix))]+[(mode,control["content_ids"][mode]+[271]+list(prefix)) for mode in MODES[1:]]


def score(output,shard,device):
    p=read_protocol(output);old,fmt,notes,writer,control,_,_=closed_bundle(Path(p["bundle"]))
    _,references,baseline_identity,_=raw_baseline(Path(p["baseline"]))
    if writer!=p["writer"] or baseline_identity!=p["baseline_reader"]:
        raise ValueError("Original writer/reference identity changed")
    ids=original.shard_ids(old,shard)
    selected=dict(p["selection"],item_ids=ids,document_ids=p["selection"]["document_ids"][shard*16:(shard+1)*16],count=16,item_ids_sha256=canonical_hash(ids))
    directory=output/"scoring"/f"shard-{shard}";directory.mkdir(parents=True,exist_ok=False)
    report=dict(status="running",schema=SCHEMA,started_utc=utc_now(),shard=shard,item_ids=ids,
        protocol_sha256=sha256_file(output/"protocol.json"),writer=writer,expected_rows=ROWS_PER_SHARD)
    atomic_json(directory/"run.json",report);count=0;maximum=0.;raw_count=0
    try:
        records=original.load_selected(Path(selected["records_file"]),selected,include_future=True)
        before=time.monotonic();backend,_,identity,_=original.load_model(old,p["reader"],device)
        check_identity(identity,p["reader"],p["model_manifest_sha256"])
        if p["reader"]["id"]=="B0" and identity!=baseline_identity:
            raise ValueError("Actual B0 reader differs from the existing raw-reference model")
        report.update(reader=identity,model_setup_seconds=time.monotonic()-before)
        with (directory/"scores.jsonl").open("x") as stream:
            for record in records:
                key=record["item_id"]
                if len(record["future"])!=16 or notes[key]["prefix_sha256"]!=canonical_hash(record["prefix"]):
                    raise ValueError("Original writer/reader prefix or future16 width differs")
                for mode,context in contexts(record["prefix"],control[key]):
                    before=time.monotonic();row=dict(item_id=key,document_id=record["document_id"],split="dev",shard=shard,
                        interface="prepend_plain" if mode!="raw" else "raw",mode=mode,context_tokens=len(context),
                        writer_identity_sha256=canonical_hash(writer),reader_identity_sha256=canonical_hash(identity),
                        writer_checkpoint_sha256=writer["checkpoint_sha256"],reader_checkpoint_sha256=identity["checkpoint_sha256"])
                    try:
                        values=score_continuations(backend,context,[record["future"]],total_width=1024,future_width=16,pad_id=fmt.pad_id)[0].token_log_likelihoods
                        if not finite_vector(values):raise ValueError("Expected sixteen finite target log likelihoods")
                        row.update(status="ok",token_log_likelihoods=values)
                        if mode=="raw" and p["reader"]["id"]=="B0":
                            error=max(abs(a-b) for a,b in zip(values,references[key]));maximum=max(maximum,error);raw_count+=1
                            row["raw_reference_max_token_error"]=error
                            if error>.002:raise ValueError("Unchanged B0 raw-reference parity failed")
                    except Exception as exc:row.update(status="error",error=f"{type(exc).__name__}: {exc}")
                    row["wall_seconds"]=time.monotonic()-before
                    stream.write(json.dumps(row,allow_nan=False)+"\n");stream.flush();count+=1
                    if row["status"]!="ok":raise RuntimeError("Failed prepend score retained; no retry/filtering")
                print(json.dumps(dict(shard=shard,rows=count,utc=utc_now())),flush=True)
        if count!=ROWS_PER_SHARD:raise ValueError("Incomplete fixed prepend shard")
        report.update(status="complete",scores_sha256=sha256_file(directory/"scores.jsonl"))
    except BaseException as exc:report.update(status="failed",error=f"{type(exc).__name__}: {exc}");raise
    finally:
        report.update(rows=count,raw_reference_rows=raw_count,raw_reference_max_token_error=maximum if raw_count else None,finished_utc=utc_now())
        atomic_json(directory/"run.json",report,overwrite=True)


def reduce_rows(p,values,notes,*,repeats=5000):
    ids=p["selection"]["item_ids"];documents=dict(zip(ids,p["selection"]["document_ids"]))
    expected={(key,mode) for key in ids for mode in MODES};keys=[(r["item_id"],r["mode"]) for r in values]
    if (len(keys)!=len(expected) or set(keys)!=expected or any(r["status"]!="ok"
            or r["document_id"]!=documents[r["item_id"]] or not finite_vector(r.get("token_log_likelihoods")) for r in values)):
        raise ValueError("Summary requires every finite document/context exactly once; no failures dropped")
    indexed={(r["item_id"],r["mode"]):r for r in values};groups={"all":ids}
    groups.update({label:[key for key in ids if p["selection"]["strata"][key]==label]
                   for label in sorted(set(p["selection"]["strata"].values()))})
    result={};per_document=[]
    for key in ids:
        per_document.append(dict(item_id=key,document_id=documents[key],quartile=p["selection"]["strata"][key],
            n_actions=notes[key]["n_actions"],n_content_tokens=notes[key]["n_content_tokens"],stop_type=notes[key]["stop_type"],
            absolute_log_likelihoods={f"K{k}":{m:sum(indexed[key,m]["token_log_likelihoods"][:k]) for m in MODES} for k in (1,16)}))
    for label,items in groups.items():
        horizons={}
        for k in (1,16):
            scores={m:[dict(item_id=key,document_id=documents[key],score=sum(indexed[key,m]["token_log_likelihoods"][:k])) for key in items] for m in MODES}
            absolute={m:dict(documents=len(items),mean_log_likelihood=statistics.mean(r["score"] for r in scores[m]),mean_log_likelihood_per_target_token=statistics.mean(r["score"] for r in scores[m])/k) for m in MODES}
            contrasts={}
            for a,b in (("empty","own"),("filler","own"),("shuffled","own"),("other","own"),("raw","own"),("raw","empty")):
                contrast=paired_cluster_bootstrap(scores[a],scores[b],repeats=repeats,seed=17)
                differences=[y["score"]-x["score"] for x,y in zip(scores[a],scores[b])]
                contrasts[f"{b}_minus_{a}"]={**contrast,"median_nats":statistics.median(differences),"positive_documents":sum(v>0 for v in differences)}
            horizons[f"K{k}"]=dict(absolute_likelihoods=absolute,contrasts=contrasts)
        result[label]=dict(documents=len(items),horizons=horizons)
    return dict(status="complete",documents=len(ids),rows=len(values),failures=0,groups=result,per_document=per_document)


def summarize(output):
    p=read_protocol(output);_,_,notes,writer,control,generation,_=closed_bundle(Path(p["bundle"]))
    _,references,_,_=raw_baseline(Path(p["baseline"]));values=[];runs=[];receipts=[];identity=None;maximum=0.
    for shard in range(4):
        d=output/"scoring"/f"shard-{shard}";run=read(d/"run.json");path=d/"scores.jsonl"
        if run["status"]!="complete":raise ValueError("All four complete bound prepend shards required")
        check_identity(run["reader"],p["reader"],p["model_manifest_sha256"])
        if (run["schema"]!=SCHEMA or run["shard"]!=shard or run["writer"]!=writer
                or run["item_ids"]!=original.shard_ids(p,shard) or run["rows"]!=ROWS_PER_SHARD
                or run["expected_rows"]!=ROWS_PER_SHARD
                or run["protocol_sha256"]!=sha256_file(output/"protocol.json") or run["scores_sha256"]!=sha256_file(path)
                or identity is not None and identity!=run["reader"]):raise ValueError("All four complete bound prepend shards required")
        identity=run["reader"];local=rows(path)
        if identity["id"]=="B0" and identity!=p["baseline_reader"]:
            raise ValueError("Actual B0 summary reader differs from its frozen raw reference")
        if [(r["item_id"],r["mode"]) for r in local]!=[(key,m) for key in original.shard_ids(p,shard) for m in MODES]:
            raise ValueError("Prepend score shard coverage/order differs")
        for row in local:
            expected_width=512 if row["mode"]=="raw" else 513+len(control[row["item_id"]]["content_ids"][row["mode"]])
            if (row["split"]!="dev" or row["shard"]!=shard or row["writer_identity_sha256"]!=canonical_hash(writer)
                    or row["context_tokens"]!=expected_width
                    or row["interface"]!=("raw" if row["mode"]=="raw" else "prepend_plain")
                    or not isinstance(row["wall_seconds"],(int,float)) or not math.isfinite(row["wall_seconds"]) or row["wall_seconds"]<0
                    or row["reader_identity_sha256"]!=canonical_hash(identity)
                    or row["writer_checkpoint_sha256"]!=writer["checkpoint_sha256"]
                    or row["reader_checkpoint_sha256"]!=identity["checkpoint_sha256"]):raise ValueError("Score provenance differs")
            if p["reader"]["id"]=="B0" and row["mode"]=="raw":
                if not finite_vector(row.get("token_log_likelihoods")):raise ValueError("Invalid B0 raw reference vector")
                error=max(abs(a-b) for a,b in zip(row["token_log_likelihoods"],references[row["item_id"]]));maximum=max(maximum,error)
                if error>.002 or row["raw_reference_max_token_error"]!=error:raise ValueError("B0 raw parity failed")
        if p["reader"]["id"]=="B0" and (run["raw_reference_rows"]!=16 or not 0<=run["raw_reference_max_token_error"]<=.002):
            raise ValueError("Incomplete B0 raw-reference parity receipt")
        values.extend(local);runs.append(run);receipts.append(dict(shard=shard,run_sha256=sha256_file(d/"run.json"),scores_sha256=sha256_file(path)))
    result=reduce_rows(p,values,notes);lengths=[n["n_content_tokens"] for n in notes.values()]
    result.update(schema=SCHEMA,experiment_id=p["experiment_id"],completed_utc=utc_now(),writer=writer,reader=identity,
        protocol_sha256=sha256_file(output/"protocol.json"),scoring=receipts,
        fixed_B0_reader_primary=identity["id"]=="B0",primary=p["primary"] if identity["id"]=="B0" else None,
        raw_reference_max_token_error=maximum if identity["id"]=="B0" else None,
        note_context=dict(empty=sum(n==0 for n in lengths),caps=sum(n["stop_type"]=="length_cap" for n in notes.values()),
            content_length_counts=dict(Counter(str(n) for n in lengths)),actions_total=sum(n["n_actions"] for n in notes.values()),
            content_tokens_total=sum(lengths),content_tokens_mean=statistics.mean(lengths),content_tokens_median=statistics.median(lengths),stop_counts=dict(Counter(n["stop_type"] for n in notes.values()))),
        cost=dict(new_sampling_seconds=0,reused_sampling_batches=16,
            reused_sampling_batch_seconds_counted_once=sum(r["batch_sampling_wall_seconds_total"] for r in generation),
            reused_replay_audit_batch_seconds_counted_once=sum(r["batch_replay_audit_wall_seconds_total"] for r in generation),
            new_score_context_seconds=sum(r["wall_seconds"] for r in values),
            new_score_process_seconds=sum((datetime.fromisoformat(r["finished_utc"])-datetime.fromisoformat(r["started_utc"])).total_seconds() for r in runs),
            stage_span_seconds=(max(datetime.fromisoformat(r["finished_utc"]) for r in runs)-min(datetime.fromisoformat(r["started_utc"]) for r in runs)).total_seconds(),
            qualification="Batch generation counted once; isolated reader/parallel process walls are not training throughput"),limitations=p["limitations"])
    atomic_json(output/"summary.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("command",choices=("prepare","score","summarize"))
    parser.add_argument("--output",required=True,type=Path);parser.add_argument("--bundle",type=Path,default=BASELINE)
    parser.add_argument("--baseline",type=Path,default=BASELINE);parser.add_argument("--reader-id",default="B0")
    parser.add_argument("--reader-checkpoint",type=Path);parser.add_argument("--reader-sha256")
    parser.add_argument("--shard-index",type=int,choices=range(4));parser.add_argument("--device",choices=("cpu","xla"),default="xla")
    a=parser.parse_args()
    if a.command=="prepare":prepare(a.output,a.bundle,a.baseline,original.model_identity(a.reader_id,a.reader_checkpoint,a.reader_sha256))
    elif a.command=="summarize":summarize(a.output)
    elif a.shard_index is None:parser.error("score requires shard-index0..3")
    else:score(a.output,a.shard_index,a.device)


if __name__=="__main__":main()
