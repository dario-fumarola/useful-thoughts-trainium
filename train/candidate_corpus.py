"""Finite TRAIN128/G16 corpus stages; no training or automatic job orchestration.

Eight prefix-only generation shards must finish before global controls are
published. Only then may eight scoring shards read TRAIN futures. Oracle choices
are training supervision, never a prefix-only deployed selector or DEV result.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

from train import anchor_scan, candidate_probe as shared
from train.evaluate_local import load_selected, read_selection, setup_model
from train.evaluate_likelihood import score_continuations
from train.interface_probe import INTERFACES, reader_context
from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file, utc_now
from weekend_thinking.core import Format, content_from_actions

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "candidate-corpus-train128-g16-v1"
SHARDS, ITEMS = 8, 128
SELECTION = ROOT/"local/nvme/diagnostics/anchors-train1024-v1/candidate-selection.ids.json"


def sources():
    return {**shared.sources(), "train/anchor_scan.py": sha256_file(anchor_scan.__file__),
            "train/candidate_corpus.py": sha256_file(__file__)}


def seed_for(seed, item_id, group):
    return int(canonical_hash(dict(schema=SCHEMA, seed=seed, item_id=item_id, group=group))[:15], 16)


def shard_selection(selection, shard):
    if type(shard) is not int or not 0 <= shard < SHARDS:
        raise ValueError("Require shard index0..7")
    ids = selection["item_ids"][shard::SHARDS]
    return dict(selection, item_ids=ids, document_ids=selection["document_ids"][shard::SHARDS],
                count=len(ids), item_ids_sha256=canonical_hash(ids))


def prepare(output, selection_path=SELECTION):
    from transformers import AutoTokenizer
    selection = read_selection(selection_path, qualification=True, split="train")
    if selection["count"] != ITEMS:
        raise ValueError("Use the actual frozen TRAIN128 selection; no fallback or DEV")
    scan = selection_path.parent
    scan_protocol = json.loads((scan/"protocol.json").read_text())
    run = json.loads((scan/"scoring/run.json").read_text())
    pool = json.loads((scan/"pool.ids.json").read_text())
    raw_path = scan/"scoring/scores.jsonl"
    raw_rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    summary, expected = anchor_scan.summarize(pool, raw_rows)
    if (run["status"] != "complete" or summary["failures"] or expected != selection
            or run["candidate_selection_sha256"] != sha256_file(selection_path)
            or run["protocol_sha256"] != sha256_file(scan/"protocol.json")
            or scan_protocol["pool_sha256"] != sha256_file(scan/"pool.ids.json")
            or json.loads((scan/"scoring/summary.json").read_text())["scores_sha256"] != sha256_file(raw_path)):
        raise ValueError("TRAIN selection must be the complete scanner's exact first128 qualifiers")
    format_path = ROOT/"local/science/resolved_format.absolute.json"
    fmt = json.loads(format_path.read_text())["format"]
    tok = AutoTokenizer.from_pretrained(scan_protocol["model_dir"], local_files_only=True)
    filler = tok.encode(" ", add_special_tokens=False)
    if not filler:
        raise ValueError("Nonempty fixed ASCII-space filler required")
    inputs = dict(selection=selection_path, scan_protocol=scan/"protocol.json", scan_run=scan/"scoring/run.json",
                  scan_summary=scan/"scoring/summary.json", scan_pool=scan/"pool.ids.json",
                  raw_references=raw_path, format=format_path)
    seeds = [seed_for(1729, key, group) for key in selection["item_ids"] for group in range(4)]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Generation seed collision")
    protocol = dict(schema=SCHEMA, experiment_id=SCHEMA, prepared_utc=utc_now(), selection=selection,
        purpose="Future-scored TRAIN candidate corpus for possible later prefix-only student training; no student implemented",
        source_files=sources(), paths={k:str(p.resolve()) for k,p in inputs.items()},
        hashes={k:sha256_file(p) for k,p in inputs.items()}, model_dir=scan_protocol["model_dir"],
        model_manifest_sha256=scan_protocol["model_manifest_sha256"], thought_format=fmt,
        shards=SHARDS, shard_rule="Global ordered item index modulo8; complete IDs validated before slicing",
        groups=4, group_size=4, candidates_per_item=16, cap=256, temperature=.7, compute="fp32",
        seed=1729, seed_namespace=SCHEMA, control_seed=17, filler_ids=filler,
        expected_global_candidates=ITEMS*16,
        expected_global_score_rows=ITEMS*(3+2*(1+16*4)),
        timing_unit="Group times repeat on four candidate rows; sum once per(item_id,group)",
        control_seed_namespace=shared.SCHEMA,
        control_rng="Deliberately reuse candidate_probe.construct_controls and its DEV-named PRNG domain; global TRAIN donors/content are newly constructed",
        writer="Original prefix-only header+prefix+open; no TRAIN future or score in sampling inputs",
        strata={key:"TRAIN_selected_raw_K1_ge1.7" for key in selection["item_ids"]},
        interfaces={name:INTERFACES[name] for name in ("original", "reanchor32")},
        horizons=[1,16], width=1024, raw_repeats=shared.RAW_REPEATS, content_modes=shared.CONTENT_MODES,
        raw_reference_atol=.002, empty_reference="No previous TRAIN empty-format scores; measure fresh same-format empty",
        first_global_group_replay_atol=.001,
        limitations=["All128 were selected using observed TRAIN target surprisal; not held-out evaluation.",
            "All2048 samples and global same-index donors must precede new future scoring.",
            "Oracle selection sees TRAIN futures; a later student must use prefix-only inputs.",
            "Cap256/G16 is a finite candidate search, not an impossibility bound."])
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/"protocol.json", protocol)
    return protocol


def read_protocol(output):
    p = json.loads((output/"protocol.json").read_text())
    if (p["schema"] != SCHEMA or p["source_files"] != sources()
            or p["selection"]["split"] != "train" or p["selection"]["count"] != ITEMS
            or (p["shards"],p["groups"],p["group_size"],p["cap"],p["temperature"],p["compute"])
               != (8,4,4,256,.7,"fp32")):
        raise ValueError("Corpus source/recipe/selection changed")
    for key,path in p["paths"].items():
        if sha256_file(path) != p["hashes"][key]:
            raise ValueError(f"Changed corpus input {key}")
    if sha256_file(Path(p["model_dir"])/"ASSET_MANIFEST.json") != p["model_manifest_sha256"]:
        raise ValueError("B0 asset manifest changed")
    return p, Format.from_dict(p["thought_format"])


def write_rows(path, rows):
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False)+"\n")


def generate(output, shard, device):
    p,fmt = read_protocol(output)
    selected = shard_selection(p["selection"], shard)
    records = load_selected(Path(selected["records_file"]), selected, include_future=False)
    directory = output/"generation"/f"shard-{shard:02d}"
    directory.mkdir(parents=True, exist_ok=False)
    backend,tok,checkpoint,signature,set_seed = setup_model(SimpleNamespace(
        model_dir=Path(p["model_dir"]), checkpoint=None, device=device))
    report = dict(schema=SCHEMA, status="running", started_utc=utc_now(), shard=shard,
        protocol_sha256=sha256_file(output/"protocol.json"), expected_candidates=len(records)*16,
        model=signature, checkpoint=checkpoint)
    atomic_json(directory/"run.json", report)
    count = 0
    try:
        with (directory/"candidates.jsonl").open("x") as stream:
            for record in records:
                for group in range(4):
                    seed = seed_for(p["seed"], record["item_id"], group)
                    first = record["item_id"]==p["selection"]["item_ids"][0] and group==0
                    try:
                        values = shared.generate_group(backend,record["prefix"],fmt=fmt,
                            seed=seed,set_seed=set_seed,audit_replay=first)
                    except Exception as exc:
                        values = [dict(status="error",error=f"{type(exc).__name__}: {exc}",group_seed=seed)
                                  for _ in range(4)]
                    for local,value in enumerate(values):
                        row = dict(item_id=record["item_id"],document_id=record["document_id"],split="train",
                            shard=shard,candidate_index=group*4+local,group=group,group_sample=local,**value)
                        if "actions" in row:
                            row.update(text=tok.decode(row["actions"],skip_special_tokens=False),
                                       content_text=tok.decode(row["content_ids"],skip_special_tokens=False))
                        stream.write(json.dumps(row,allow_nan=False)+"\n");stream.flush();count+=1
                    if any(v["status"]!="ok" for v in values):
                        raise RuntimeError("Failed group retained; no retry or incomplete global scoring")
                print(json.dumps(dict(shard=shard,item_id=record["item_id"],candidates=count,utc=utc_now())),flush=True)
        report.update(status="complete",candidates=count,candidates_sha256=sha256_file(directory/"candidates.jsonl"))
    except BaseException as exc:
        report.update(status="failed",error=f"{type(exc).__name__}: {exc}");raise
    finally:
        report.update(candidates=count,finished_utc=utc_now())
        atomic_json(directory/"run.json",report,overwrite=True)


def load_all_candidates(output, p, fmt):
    """Validate all eight global shards before any new canonical future read."""
    notes,receipts,identity = {},[],None
    docs = dict(zip(p["selection"]["item_ids"],p["selection"]["document_ids"]))
    for shard in range(SHARDS):
        directory = output/"generation"/f"shard-{shard:02d}"
        run = json.loads((directory/"run.json").read_text())
        path = directory/"candidates.jsonl"
        if (run["status"]!="complete" or run["schema"]!=SCHEMA or run["shard"]!=shard
                or run["protocol_sha256"]!=sha256_file(output/"protocol.json")
                or run["candidates_sha256"]!=sha256_file(path) or run["checkpoint"] is not None):
            raise ValueError("Every generation shard must be complete/hash-bound unchanged B0")
        if identity is None:identity=run["model"]
        if run["model"]!=identity:raise ValueError("Generation shards mix models")
        expected_ids = set(shard_selection(p["selection"],shard)["item_ids"])
        local = []
        for line in path.read_text().splitlines():
            row=json.loads(line);key=(row["item_id"],row["candidate_index"])
            if (key in notes or key[0] not in expected_ids or row["document_id"]!=docs[key[0]]
                    or row["split"]!="train" or row["shard"]!=shard
                    or type(key[1]) is not int or not 0<=key[1]<16
                    or row["group"]!=key[1]//4 or row["group_sample"]!=key[1]%4
                    or row["group_seed"]!=seed_for(p["seed"],key[0],key[1]//4)):
                raise ValueError("Candidate shard/ID/document/seed mismatch")
            a,lp,mask = row.get("actions",[]),row.get("behavior_logps",[]),row.get("action_mask",[])
            content=content_from_actions(a,fmt.stop_id,fmt.additional_stop_ids)
            stop_type = f"token:{a[-1]}" if a and a[-1] in {fmt.stop_id,*fmt.additional_stop_ids} else "length_cap"
            if (row["status"]!="ok" or not a or len(a)>256 or len(lp)!=len(a)
                    or any(type(v) is not int or v<0 for v in a) or not all(math.isfinite(v) for v in lp)
                    or row["content_ids"]!=content or len(content)<len(a)-1
                    or (len(content)==len(a) and len(a)!=256) or row["n_actions"]!=len(a)
                    or row["n_content_tokens"]!=len(content) or len(mask)!=256
                    or row["stop_type"]!=stop_type or not len(a)<=row["decode_iterations"]<=256
                    or any(type(v) is not bool for v in mask)
                    or mask!=[True]*len(a)+[False]*(256-len(a))):
                raise ValueError("Invalid sampled actions/logps/stops; no filtering")
            if key==(p["selection"]["item_ids"][0],0):
                if row.get("cache_replay_status")!="ok" or not 0<=row.get("cache_replay_max_error",math.inf)<=.001:
                    raise ValueError("First global group replay failed")
            notes[key]=row;local.append(key)
        if (set(local)!={(key,i) for key in expected_ids for i in range(16)}
                or len(local)!=run["candidates"] or len(local)!=run["expected_candidates"]):
            raise ValueError("Full generation shard coverage required")
        receipts.append(dict(shard=shard,run_sha256=sha256_file(directory/"run.json"),candidates_sha256=sha256_file(path)))
    if len(notes)!=ITEMS*16:raise ValueError("All2048 candidates are required before scoring")
    return notes,dict(model=identity,generation=receipts)


def controls(output):
    p,fmt=read_protocol(output)
    notes,receipt=load_all_candidates(output,p,fmt)
    values=shared.construct_controls(p,notes,p["filler_ids"])
    directory=output/"global";directory.mkdir(exist_ok=False)
    write_rows(directory/"controls.jsonl",values.values())
    atomic_json(directory/"COMPLETE.json",dict(schema=SCHEMA,status="complete",
        protocol_sha256=sha256_file(output/"protocol.json"),candidates=ITEMS*16,
        controls_sha256=sha256_file(directory/"controls.jsonl"),**receipt))


def global_inputs(output,p,fmt):
    notes,receipt=load_all_candidates(output,p,fmt)
    complete=json.loads((output/"global/COMPLETE.json").read_text())
    path=output/"global/controls.jsonl"
    if (complete["schema"]!=SCHEMA or complete["status"]!="complete" or complete["candidates"]!=ITEMS*16
            or complete["protocol_sha256"]!=sha256_file(output/"protocol.json")
            or complete["controls_sha256"]!=sha256_file(path)
            or complete["generation"]!=receipt["generation"] or complete["model"]!=receipt["model"]):
        raise ValueError("Global generation/control barrier mismatch")
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    values={(r["item_id"],r["candidate_index"]):r for r in rows}
    if len(values)!=len(rows) or values!=shared.construct_controls(p,notes,p["filler_ids"]):
        raise ValueError("Controls must use all128 global same-index donors before sharding")
    return notes,values,complete


def score(output,shard,device):
    p,fmt=read_protocol(output)
    selected=shard_selection(p["selection"],shard)
    notes,values,barrier=global_inputs(output,p,fmt)  # Before models or new TRAIN futures.
    backend,_,checkpoint,signature,_=setup_model(SimpleNamespace(model_dir=Path(p["model_dir"]),checkpoint=None,device=device))
    if checkpoint is not None or signature!=barrier["model"]:raise ValueError("Reader differs from unchanged writer B0")
    references={r["item_id"]:r["token_log_likelihoods"] for r in map(json.loads,
        Path(p["paths"]["raw_references"]).read_text().splitlines()) if r["item_id"] in selected["item_ids"]}
    if set(references)!=set(selected["item_ids"]):raise ValueError("Missing TRAIN raw scanner reference")
    records=load_selected(Path(selected["records_file"]),selected,include_future=True)
    directory=output/"scoring"/f"shard-{shard:02d}";directory.mkdir(parents=True,exist_ok=False)
    report=dict(schema=SCHEMA,status="running",started_utc=utc_now(),shard=shard,
        protocol_sha256=sha256_file(output/"protocol.json"),barrier_sha256=sha256_file(output/"global/COMPLETE.json"),
        model=signature,checkpoint=checkpoint,expected_items=len(records))
    atomic_json(directory/"run.json",report);count=0;max_error=0.
    try:
        with (directory/"scores.jsonl").open("x") as stream:
            for record in records:
                key=record["item_id"]
                if any(notes[(key,i)]["prefix_sha256"]!=canonical_hash(record["prefix"]) for i in range(16)):
                    raise ValueError("Candidate prefix differs from reader prefix")
                contexts=[("raw",mode,None,reader_context(record["prefix"],None,fmt,header=False,reanchor=n)) for mode,n in shared.RAW_REPEATS.items()]
                for interface,(header,reanchor) in p["interfaces"].items():
                    contexts.append((interface,"empty",None,reader_context(record["prefix"],[],fmt,header=header,reanchor=reanchor)))
                    contexts.extend((interface,mode,i,reader_context(record["prefix"],values[(key,i)]["content_ids"][mode],fmt,header=header,reanchor=reanchor)) for i in range(16) for mode in shared.CONTENT_MODES)
                for interface,mode,i,context in contexts:
                    started=time.monotonic();row=dict(item_id=key,document_id=record["document_id"],split="train",shard=shard,interface=interface,mode=mode,candidate_index=i)
                    if i is not None:row.update(n_note_content_tokens=notes[(key,i)]["n_content_tokens"],note_stop_type=notes[(key,i)]["stop_type"],n_note_actions=notes[(key,i)]["n_actions"])
                    try:
                        result=score_continuations(backend,context,[record["future"]],total_width=1024,future_width=16,pad_id=fmt.pad_id)[0]
                        row.update(status="ok",token_log_likelihoods=result.token_log_likelihoods)
                        if (interface,mode)==("raw","raw"):
                            error=max(abs(a-b) for a,b in zip(result.token_log_likelihoods,references[key]))
                            row["scanner_reference_max_error"]=error;max_error=max(error,max_error)
                            if not math.isfinite(error) or error>.002:raise ValueError(f"Raw width parity failed: {error}")
                    except Exception as exc:row.update(status="error",error=f"{type(exc).__name__}: {exc}")
                    row["wall_seconds"]=time.monotonic()-started
                    stream.write(json.dumps(row,allow_nan=False)+"\n");stream.flush();count+=1
                    if row["status"]!="ok":raise RuntimeError("Failure retained; shard incomplete, no filtering/retry")
                print(json.dumps(dict(shard=shard,item_id=key,rows=count,utc=utc_now())),flush=True)
        report.update(status="complete",scores_sha256=sha256_file(directory/"scores.jsonl"))
    except BaseException as exc:report.update(status="failed",error=f"{type(exc).__name__}: {exc}");raise
    finally:
        report.update(rows=count,raw_parity_max_error=max_error,finished_utc=utc_now())
        atomic_json(directory/"run.json",report,overwrite=True)


def reduce_rows(p, rows):
    """The shared reducer is numerical code; its DEV identity is not inherited."""
    result=shared.summarize(p,rows)
    result.update(schema=SCHEMA,experiment_id=p["experiment_id"],split="train",
        statistics_implementation="candidate_probe.summarize numerical reducer; TRAIN schema/stratum explicitly adapted")
    for row in result["per_document"]:row["stratum"]=row.pop("quartile")
    return result


def summarize(output):
    p,fmt=read_protocol(output);notes,_,barrier=global_inputs(output,p,fmt)
    rows=[];receipts=[]
    for shard in range(SHARDS):
        directory=output/"scoring"/f"shard-{shard:02d}";run=json.loads((directory/"run.json").read_text());path=directory/"scores.jsonl"
        if (run["status"]!="complete" or run["schema"]!=SCHEMA or run["shard"]!=shard
                or run["protocol_sha256"]!=sha256_file(output/"protocol.json")
                or run["barrier_sha256"]!=sha256_file(output/"global/COMPLETE.json")
                or run["model"]!=barrier["model"] or run["checkpoint"] is not None
                or run["scores_sha256"]!=sha256_file(path)):
            raise ValueError("All scoring shards must complete before a global summary")
        local=[json.loads(line) for line in path.read_text().splitlines()]
        ids=set(shard_selection(p["selection"],shard)["item_ids"])
        if (set(r["item_id"] for r in local)!=ids or any(r["split"]!="train" or r["shard"]!=shard or r["status"]!="ok" for r in local)
                or len(local)!=run["rows"]):raise ValueError("Score shard IDs/status/count mismatch")
        rows.extend(local);receipts.append(dict(shard=shard,run_sha256=sha256_file(directory/"run.json"),scores_sha256=sha256_file(path)))
    result=reduce_rows(p,rows)
    result.update(protocol_sha256=sha256_file(output/"protocol.json"),scoring=receipts,
        completed_utc=utc_now(),sampling_counts=dict(Counter(n["stop_type"] for n in notes.values())))
    atomic_json(output/"summary.json",result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","generate","controls","score","summarize"))
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--selection",type=Path,default=SELECTION)
    parser.add_argument("--shard-index",type=int);parser.add_argument("--device",choices=("cpu","xla"),default="xla")
    args=parser.parse_args()
    if args.command in {"generate","score"} and args.shard_index is None:parser.error("Stage requires explicit --shard-index0..7")
    if args.command=="prepare":prepare(args.output,args.selection)
    elif args.command=="generate":generate(args.output,args.shard_index,args.device)
    elif args.command=="controls":controls(args.output)
    elif args.command=="score":score(args.output,args.shard_index,args.device)
    else:summarize(args.output)


if __name__=="__main__":main()
