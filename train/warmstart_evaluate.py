"""Fresh held DEV64 G1 notes in fixed B4 batches, then global-control scoring.

This is exploratory development. Writer and reader may differ, but their exact
identities are always retained; no oracle selection or final-test gate is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from train.evaluate_local import load_selected, setup_model
from train.evaluate_likelihood import score_continuations
from train.evaluation_integrity import evaluation_dependencies
from train.interface_probe import reader_context
from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file, utc_now
from weekend_thinking.core import Format, content_from_actions, matched_controls

SCHEMA = "warmstart-dev64-cap256-v1"
CAP, BATCH, SHARDS, SHARD_SIZE = 256, 4, 4, 16
MODES = ("empty", "own", "filler", "shuffled", "other")


def sources():
    root = Path(__file__).resolve().parents[1]
    values = evaluation_dependencies()
    for name in ("train/warmstart_evaluate.py", "train/interface_probe.py"):
        values[name] = sha256_file(root/name)
    return values


def model_identity(identifier, checkpoint=None, expected_sha256=None):
    if not identifier or identifier in {".",".."} or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in identifier):
        raise ValueError("Use an explicit safe writer/reader ID")
    if (checkpoint is None) != (expected_sha256 is None):
        raise ValueError("Checkpoint and expected model.pt SHA must be supplied together")
    if (identifier == "B0") != (checkpoint is None):
        raise ValueError("Only unchanged B0 uses no checkpoint")
    if checkpoint is not None and (Path(checkpoint).name != "model.pt" or len(expected_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_sha256)):
        raise ValueError("Expected explicit model.pt and SHA256")
    return dict(id=identifier, checkpoint=str(Path(checkpoint).resolve()) if checkpoint else None,
                checkpoint_sha256=expected_sha256)


def prepare(output, selection_path, model_dir, format_path, writer):
    selection = json.loads(selection_path.read_text())
    ids = selection["item_ids"]
    if (selection["split"] != "dev" or selection["count"] != 64 or len(ids) != 64
            or len(set(ids)) != 64 or len(selection["document_ids"]) != 64 or len(set(selection["document_ids"])) != 64
            or selection["item_ids_sha256"] != canonical_hash(ids)
            or set(selection["strata"]) != set(ids)
            or sorted(Counter(selection["strata"].values()).values()) != [16]*4):
        raise ValueError("Require the prospectively fixed DEV64, sixteen per frozen quartile")
    protocol = dict(schema=SCHEMA, experiment_id=output.name, created_utc=utc_now(),
        purpose="nonfinal held development; fresh G1 B0/student/cross-reader utility, no oracle",
        selection=selection, selection_path=str(selection_path.resolve()), selection_sha256=sha256_file(selection_path),
        model_dir=str(model_dir.resolve()), model_manifest_sha256=sha256_file(model_dir/"ASSET_MANIFEST.json"),
        format_path=str(format_path.resolve()), format_sha256=sha256_file(format_path), writer=writer,
        cap=CAP, temperature=.7, seed=1729, batch_size=BATCH, groups_per_prefix=1,
        shards=SHARDS, shard_size=SHARD_SIZE, batch_seed_rule="SHA256(namespace,seed,ordered_batch_item_ids), first15hex; same across writers",
        timing_unit="Four distinct-prefix notes share one batch wall time; count each batch once",
        controls_seed=17, filler_text=" ", horizons=[1,16], width=1024,
        reader_contexts="raw,raw_repeat32,reanchor32(empty,own,filler,shuffled,other)",
        first_batch_replay_atol=.001, source_sha256=sources(),
        limitations=["Fixed exploratory DEV64; not official final/test", "G1/cap256 bounded stochastic policy; no oracle", "Batch RNG is explicit and fixed; not equivalent to four sequential item-seeded calls"])
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/"protocol.json", protocol)
    return protocol


def read_protocol(output):
    p = json.loads((output/"protocol.json").read_text())
    if p["schema"] != SCHEMA or p["source_sha256"] != sources():
        raise ValueError("Warmstart source/protocol changed")
    for path,digest in ((p["selection_path"],p["selection_sha256"]),(p["format_path"],p["format_sha256"]),
                        (str(Path(p["model_dir"])/"ASSET_MANIFEST.json"),p["model_manifest_sha256"])):
        if sha256_file(path) != digest:
            raise ValueError("Warmstart input changed")
    return p, Format.from_dict(json.loads(Path(p["format_path"]).read_text())["format"])


def shard_ids(protocol, shard):
    if type(shard) is not int or not 0 <= shard < SHARDS:
        raise ValueError("Shard must be0..3")
    return protocol["selection"]["item_ids"][shard*SHARD_SIZE:(shard+1)*SHARD_SIZE]


def batch_seed(ids):
    return int(canonical_hash(dict(namespace=SCHEMA,seed=1729,ordered_batch_item_ids=ids))[:15],16)


def load_model(protocol, identity, device):
    checkpoint = Path(identity["checkpoint"]) if identity["checkpoint"] else None
    if checkpoint is not None and (checkpoint.is_symlink() or sha256_file(checkpoint) != identity["checkpoint_sha256"]):
        raise ValueError("Explicit checkpoint SHA mismatch")
    backend,tok,loaded,signature,seed = setup_model(SimpleNamespace(
        model_dir=Path(protocol["model_dir"]),checkpoint=checkpoint,device=device))
    if checkpoint is not None and loaded["sha256"] != identity["checkpoint_sha256"]:
        raise ValueError("Loaded checkpoint differs from the expected model SHA")
    return backend,tok,dict(**identity,loaded_checkpoint=loaded,model=signature,
                           base_manifest_sha256=protocol["model_manifest_sha256"]),seed


@torch.no_grad()
def generate_batch(backend, prefixes, *, fmt, seed, set_seed, audit):
    if len(prefixes) != BATCH or len({len(x) for x in prefixes}) != 1:
        raise ValueError("Exactly four equal-width distinct-prefix rows required")
    value = torch.tensor(prefixes,device=backend.device)
    backend.sync(); set_seed(seed)
    started = time.monotonic()
    rollout = backend.sample(value,fmt=fmt,group_size=1,max_actions=CAP,temperature=.7,policy_version=0)
    backend.sync()
    actions,masks,lps = (x.cpu()[:,0] for x in (rollout.actions,rollout.action_mask,rollout.behavior_logps))
    sample_wall = time.monotonic()-started; error = None; audit_error = None
    audit_start = time.monotonic()
    if audit:
        try:
            replay = backend.replay(value,rollout,fmt=fmt,temperature=.7); backend.sync()
            error = torch.where(rollout.action_mask,(replay-rollout.behavior_logps).abs(),0.).max().cpu().item()
        except Exception as exc:
            audit_error = f"{type(exc).__name__}: {exc}"
    audit_wall = time.monotonic()-audit_start if audit else 0.
    rows = []
    for i in range(BATCH):
        a = actions[i][masks[i]].tolist(); lp = lps[i][masks[i]].tolist()
        content = content_from_actions(a,fmt.stop_id,fmt.additional_stop_ids)
        valid = bool(a) and all(math.isfinite(x) for x in lp) and (not audit or error is not None and math.isfinite(error) and error<=.001)
        rows.append(dict(status="ok" if valid else "error",actions=a,action_mask=masks[i].tolist(),
            behavior_logps=[x if math.isfinite(x) else repr(x) for x in lp],content_ids=content,
            n_actions=len(a),n_content_tokens=len(content),prefix_sha256=canonical_hash(prefixes[i]),
            stop_type=f"token:{a[-1]}" if a and a[-1] in {fmt.stop_id,*fmt.additional_stop_ids} else "length_cap",
            decode_iterations=rollout.decode_iterations,batch_seed=seed,batch_row=i,
            batch_sampling_wall_seconds=sample_wall,batch_replay_audit_wall_seconds=audit_wall,
            cache_replay_max_error=error if error is None or math.isfinite(error) else repr(error),
            **({} if valid else dict(error=audit_error or "Invalid sample or first-batch replay mismatch"))))
    return rows


def generate(output, shard, device):
    p,fmt = read_protocol(output); ids = shard_ids(p,shard)
    selected = dict(p["selection"],item_ids=ids,document_ids=p["selection"]["document_ids"][shard*16:(shard+1)*16],count=16,item_ids_sha256=canonical_hash(ids))
    records = load_selected(Path(selected["records_file"]),selected,include_future=False)
    backend,tok,identity,set_seed = load_model(p,p["writer"],device)
    directory = output/"notes"/f"shard-{shard}"; directory.mkdir(parents=True,exist_ok=False)
    report = dict(status="running",started_utc=utc_now(),writer=identity,protocol_sha256=sha256_file(output/"protocol.json"),item_ids=ids,shard=shard)
    atomic_json(directory/"run.json",report); count=0; sampling_wall=0.; audit_wall=0.
    try:
        with (directory/"notes.jsonl").open("x") as stream:
            for offset in range(0,SHARD_SIZE,BATCH):
                batch = records[offset:offset+BATCH]; batch_ids = [r["item_id"] for r in batch]
                values = generate_batch(backend,[r["prefix"] for r in batch],fmt=fmt,seed=batch_seed(batch_ids),set_seed=set_seed,audit=offset==0)
                sampling_wall+=values[0]["batch_sampling_wall_seconds"];audit_wall+=values[0]["batch_replay_audit_wall_seconds"]
                for record,value in zip(batch,values):
                    row = dict(item_id=record["item_id"],document_id=record["document_id"],split="dev",batch_item_ids=batch_ids,**value)
                    row.update(text=tok.decode(row["actions"],skip_special_tokens=False),content_text=tok.decode(row["content_ids"],skip_special_tokens=False))
                    stream.write(json.dumps(row,allow_nan=False)+"\n"); stream.flush(); count+=1
                if any(r["status"]!="ok" for r in values):
                    raise RuntimeError("Failed actual batch retained; no resampling")
                print(json.dumps(dict(shard=shard,notes=count,utc=utc_now())),flush=True)
        report.update(status="complete",finished_utc=utc_now(),notes=count,notes_sha256=sha256_file(directory/"notes.jsonl"),sampling_batches=4,batch_sampling_wall_seconds_total=sampling_wall,batch_replay_audit_wall_seconds_total=audit_wall)
    except BaseException as exc:
        report.update(status="failed",notes=count,error=f"{type(exc).__name__}: {exc}"); raise
    finally:
        atomic_json(directory/"run.json",report,overwrite=True)


def load_notes(output,p,fmt):
    result = {}; signatures = []
    for shard in range(SHARDS):
        directory=output/"notes"/f"shard-{shard}"
        if not (directory/"run.json").is_file() or not (directory/"notes.jsonl").is_file():
            raise ValueError("All four complete hash-bound note shards required before controls/futures")
        report=json.loads((directory/"run.json").read_text())
        if (report["status"]!="complete" or report["protocol_sha256"]!=sha256_file(output/"protocol.json")
                or report["notes_sha256"]!=sha256_file(directory/"notes.jsonl") or report["item_ids"]!=shard_ids(p,shard)):
            raise ValueError("All four complete hash-bound note shards required before controls/futures")
        signatures.append(report["writer"])
        rows=[json.loads(x) for x in (directory/"notes.jsonl").read_text().splitlines()]
        if [r["item_id"] for r in rows]!=shard_ids(p,shard):raise ValueError("Note shard coverage/order mismatch")
        documents=dict(zip(p["selection"]["item_ids"],p["selection"]["document_ids"]))
        for i,row in enumerate(rows):
            a=row["actions"]; content=content_from_actions(a,fmt.stop_id,fmt.additional_stop_ids)
            batch=shard_ids(p,shard)[i//4*4:i//4*4+4]
            stop=f"token:{a[-1]}" if a and a[-1] in {fmt.stop_id,*fmt.additional_stop_ids} else "length_cap"
            if (row["status"]!="ok" or row["document_id"]!=documents[row["item_id"]] or row["split"]!="dev"
                    or not a or len(a)>CAP or any(type(v) is not int or v<0 for v in a)
                    or row["content_ids"]!=content or len(content)<len(a)-1
                    or len(content)==len(a) and len(a)!=CAP or len(row["behavior_logps"])!=len(a)
                    or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in row["behavior_logps"])
                    or row["n_actions"]!=len(a) or row["n_content_tokens"]!=len(content) or row["stop_type"]!=stop
                    or any(type(v) is not bool for v in row["action_mask"])
                    or row["action_mask"]!=[True]*len(a)+[False]*(CAP-len(a)) or row["batch_item_ids"]!=batch
                    or row["batch_seed"]!=batch_seed(batch) or row["batch_row"]!=i%4
                    or i<4 and (not isinstance(row["cache_replay_max_error"],(int,float)) or not 0<=row["cache_replay_max_error"]<=.001)):
                raise ValueError("Invalid complete G1 note/batch/replay evidence")
            result[row["item_id"]]=row
    if any(s!=signatures[0] for s in signatures[1:]):raise ValueError("Writer model differs across shards")
    if any(signatures[0][k]!=p["writer"][k] for k in p["writer"]):raise ValueError("Unexpected writer identity")
    return result,signatures[0]


def make_controls(p,notes,filler):
    ids=p["selection"]["item_ids"];contents=[notes[k]["content_ids"] for k in ids]
    values=matched_controls(contents,filler_ids=filler,seed=p["controls_seed"])
    return [dict(item_id=k,donor_item_id=ids[(i+1)%len(ids)],content_ids=dict(empty=[],own=contents[i],**{m:values[m][i] for m in ("filler","shuffled","other")})) for i,k in enumerate(ids)]


def controls(output):
    p,fmt=read_protocol(output);notes,writer=load_notes(output,p,fmt)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(p["model_dir"],local_files_only=True,trust_remote_code=False)
    filler=tok.encode(p["filler_text"],add_special_tokens=False)
    if not filler or set(filler)&{fmt.stop_id,*fmt.additional_stop_ids}:raise ValueError("Invalid filler")
    with (output/"controls.jsonl").open("x") as stream:
        for row in make_controls(p,notes,filler):stream.write(json.dumps(row)+"\n")
    atomic_json(output/"controls.complete.json",dict(status="complete",writer=writer,filler_ids=filler,
        protocol_sha256=sha256_file(output/"protocol.json"),notes_sha256=canonical_hash(notes),controls_sha256=sha256_file(output/"controls.jsonl"),items=64))


def score(output,shard,device,reader):
    p,fmt=read_protocol(output);notes,writer=load_notes(output,p,fmt)
    receipt=json.loads((output/"controls.complete.json").read_text());path=output/"controls.jsonl"
    expected=make_controls(p,notes,receipt["filler_ids"])
    if (receipt["status"]!="complete" or receipt["writer"]!=writer or receipt["items"]!=64 or receipt["protocol_sha256"]!=sha256_file(output/"protocol.json")
            or receipt["notes_sha256"]!=canonical_hash(notes) or receipt["controls_sha256"]!=sha256_file(path)
            or [json.loads(s) for s in path.read_text().splitlines()]!=expected):raise ValueError("Complete global controls required")
    control={r["item_id"]:r for r in expected};ids=shard_ids(p,shard)
    selected=dict(p["selection"],item_ids=ids,document_ids=p["selection"]["document_ids"][shard*16:(shard+1)*16],count=16,item_ids_sha256=canonical_hash(ids))
    records=load_selected(Path(selected["records_file"]),selected,include_future=True)
    backend,_,identity,_=load_model(p,reader,device)
    directory=output/"readers"/reader["id"]/f"shard-{shard}";directory.mkdir(parents=True,exist_ok=False)
    report=dict(status="running",started_utc=utc_now(),writer=writer,reader=identity,shard=shard,item_ids=ids,
        protocol_sha256=sha256_file(output/"protocol.json"),controls_sha256=receipt["controls_sha256"],notes_sha256=receipt["notes_sha256"],expected_rows=112)
    atomic_json(directory/"run.json",report);count=0
    try:
        with (directory/"scores.jsonl").open("x") as stream:
            for record in records:
                key=record["item_id"]
                if notes[key]["prefix_sha256"]!=canonical_hash(record["prefix"]):raise ValueError("Writer/reader prefix mismatch")
                if len(record["future"])<16:raise ValueError("All sixteen future tokens are required")
                contexts=[("raw",list(record["prefix"])),("raw_repeat32",list(record["prefix"])+record["prefix"][-32:])]
                contexts += [(mode,reader_context(record["prefix"],control[key]["content_ids"][mode],fmt,header=True,reanchor=32)) for mode in MODES]
                for mode,context in contexts:
                    started=time.monotonic();row=dict(item_id=key,document_id=record["document_id"],mode=mode,
                        writer_id=writer["id"],reader_id=reader["id"],context_tokens=len(context),note_status=notes[key]["status"],n_note_actions=notes[key]["n_actions"],n_note_content_tokens=notes[key]["n_content_tokens"],note_stop_type=notes[key]["stop_type"])
                    try:
                        value=score_continuations(backend,context,[record["future"][:16]],total_width=p["width"],future_width=16,pad_id=fmt.pad_id)[0]
                        if len(value.token_log_likelihoods)!=16 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in value.token_log_likelihoods):
                            raise ValueError("Expected exactly sixteen finite token log likelihoods")
                        row.update(status="ok",token_log_likelihoods=value.token_log_likelihoods)
                    except Exception as exc:row.update(status="error",error=f"{type(exc).__name__}: {exc}")
                    row["wall_seconds"]=time.monotonic()-started;stream.write(json.dumps(row,allow_nan=False)+"\n");stream.flush();count+=1
                    if row["status"]!="ok":raise RuntimeError("Scoring failure retained; no retry")
                print(json.dumps(dict(shard=shard,rows=count,utc=utc_now())),flush=True)
        if count!=112:raise ValueError("Incomplete seven-context shard matrix")
        report.update(status="complete",finished_utc=utc_now(),rows=count,scores_sha256=sha256_file(directory/"scores.jsonl"))
    except BaseException as exc:report.update(status="failed",rows=count,error=f"{type(exc).__name__}: {exc}");raise
    finally:atomic_json(directory/"run.json",report,overwrite=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("command",choices=("prepare","generate","controls","score"))
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--selection",type=Path)
    parser.add_argument("--model-dir",type=Path);parser.add_argument("--format",type=Path)
    parser.add_argument("--writer-id");parser.add_argument("--writer-checkpoint",type=Path);parser.add_argument("--writer-sha256")
    parser.add_argument("--reader-id");parser.add_argument("--reader-checkpoint",type=Path);parser.add_argument("--reader-sha256")
    parser.add_argument("--shard-index",type=int,choices=range(4));parser.add_argument("--device",choices=("cpu","xla"),default="xla")
    a=parser.parse_args()
    if a.command=="prepare":
        if any(v is None for v in (a.selection,a.model_dir,a.format,a.writer_id)):parser.error("prepare requires selection/model-dir/format/writer-id")
        p=prepare(a.output,a.selection,a.model_dir,a.format,model_identity(a.writer_id,a.writer_checkpoint,a.writer_sha256));print(json.dumps(dict(prepared=str(a.output),writer=p["writer"],items=64)))
    elif a.command=="controls":controls(a.output)
    elif a.shard_index is None:parser.error("generate/score require shard-index")
    elif a.command=="generate":generate(a.output,a.shard_index,a.device)
    else:
        if a.reader_id is None:parser.error("score requires explicit reader-id/checkpoint identity")
        score(a.output,a.shard_index,a.device,model_identity(a.reader_id,a.reader_checkpoint,a.reader_sha256))


if __name__=="__main__":main()
