"""Separate DEV32 candidate search: prefix-only G16, then fixed-reader scoring.

This is a finite oracle-availability diagnostic, not a deployed best-of policy.
The 256-action search does not bound what longer notes or other policies can do.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch

from train.evaluate_local import load_selected, setup_model
from train.evaluate_likelihood import score_continuations
from train.evaluation_integrity import evaluation_dependencies
from train.interface_probe import INTERFACES, reader_context
from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file, utc_now
from weekend_thinking.core import Format, content_from_actions, matched_controls
from weekend_thinking.stats import paired_cluster_bootstrap

SCHEMA = "candidate-oracle-dev32-v1"
GROUPS, GROUP_SIZE, CAP = 4, 4, 256
SAMPLES = GROUPS * GROUP_SIZE
CONTENT_MODES = ("own", "filler", "shuffled", "other")
RAW_REPEATS = {"raw": 0, "repeat32": 32, "repeat64": 64}
DEFAULT_INTERFACES = ("original", "reanchor32")


def group_seed(seed: int, item_id: str, group: int) -> int:
    return int(canonical_hash(dict(schema=SCHEMA, seed=seed, item_id=item_id,
                                   group=group))[:15], 16)


def sources() -> dict:
    result = evaluation_dependencies()
    root = Path(__file__).resolve().parents[1]
    for name in ("train/candidate_probe.py", "train/interface_probe.py"):
        result[name] = sha256_file(root/name)
    return result


def prepare(output: Path, pilot_path: Path, *, interfaces=DEFAULT_INTERFACES,
            seed: int = 1729, control_seed: int = 17) -> dict:
    pilot = json.loads(pilot_path.read_text())
    selection = pilot["selection"]
    ids, documents = selection["item_ids"], selection["document_ids"]
    counts = Counter(pilot["strata"][key] for key in ids)
    if (selection["split"] != "dev" or len(ids) != 32 or len(set(ids)) != 32
            or len(documents) != 32 or len(set(documents)) != 32
            or selection["count"] != 32 or selection["item_ids_sha256"] != canonical_hash(ids)
            or pilot["per_quartile"] != 8 or len(counts) != 4 or set(counts.values()) != {8}):
        raise ValueError("Use the existing fixed DEV32: eight items per frozen quartile")
    if not interfaces or len(set(interfaces)) != len(interfaces) or set(interfaces)-set(INTERFACES):
        raise ValueError("Unknown or duplicate reader interface")
    pilot_scores = pilot_path.parent/"scores.jsonl"
    pilot_summary = json.loads((pilot_path.parent/"summary.json").read_text())
    if (pilot_summary["status"] != "complete" or pilot_summary["failures"] != 0
            or sha256_file(pilot_scores) != pilot_summary["scores_sha256"]):
        raise ValueError("Use the completed, failure-free pilot score identity")
    seeds = [group_seed(seed, key, group) for key in ids for group in range(GROUPS)]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Candidate group seeds collide")
    paths = {key: pilot["paths"][key] for key in ("records", "format", "strata")}
    protocol = dict(schema=SCHEMA, experiment_id=output.name, created_utc=utc_now(),
        purpose="exploratory held DEV candidate availability; no final test or training",
        pilot_protocol=str(pilot_path.resolve()), pilot_sha256=sha256_file(pilot_path),
        pilot_scores=str(pilot_scores.resolve()), pilot_scores_sha256=sha256_file(pilot_scores),
        selection=selection, strata={key: pilot["strata"][key] for key in ids},
        paths=paths, input_sha256={key: pilot["sha256"][key] for key in ("format", "strata")},
        model_dir=pilot["model_dir"], model_manifest_sha256=sha256_file(
            Path(pilot["model_dir"])/"ASSET_MANIFEST.json"),
        writer="original prefix-only interface", groups=GROUPS, group_size=GROUP_SIZE,
        candidates_per_item=SAMPLES, cap=CAP, temperature=.7, seed=seed,
        support="tokenizer-defined IDs only; original stop semantics",
        group_seed_rule="SHA256(schema,seed,item_id,group), first 15 hex digits",
        timing_unit="Four candidates sampled together; group wall time is shared, never summed four times",
        control_seed=control_seed, filler_text=" ", horizons=[1, 16], width=1024,
        interfaces={name: INTERFACES[name] for name in interfaces},
        content_modes=CONTENT_MODES, raw_repeats=RAW_REPEATS, compute="fp32",
        expected_score_rows=len(ids)*(len(RAW_REPEATS)+len(interfaces)*(1+SAMPLES*len(CONTENT_MODES))),
        replay_atol=.001, baseline_atol=.002,
        source_sha256=sources(), limitations=[
            "G16 is newly sampled candidates; K1/K16 are future-token horizons.",
            "Cap256 is a bounded search, not a long-note oracle ceiling.",
            "Oracle selection uses the true future only after all sampling; it is not deployed policy utility.",
            "A zero finite-sample oracle cannot establish impossibility."])
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/"protocol.json", protocol)
    return protocol


def read_protocol(output: Path) -> tuple[dict, Format]:
    protocol = json.loads((output/"protocol.json").read_text())
    if protocol["schema"] != SCHEMA or protocol["source_sha256"] != sources():
        raise ValueError("Candidate protocol/source changed")
    if (protocol["groups"],protocol["group_size"],protocol["candidates_per_item"],
            protocol["cap"],protocol["temperature"],protocol["compute"]) != (4,4,16,256,.7,"fp32"):
        raise ValueError("Candidate protocol differs from the declared sampling recipe")
    if sha256_file(protocol["pilot_protocol"]) != protocol["pilot_sha256"]:
        raise ValueError("Fixed pilot identity changed")
    if sha256_file(protocol["pilot_scores"]) != protocol["pilot_scores_sha256"]:
        raise ValueError("Completed pilot scores changed")
    for key, digest in protocol["input_sha256"].items():
        if sha256_file(protocol["paths"][key]) != digest:
            raise ValueError(f"Changed {key} input")
    if sha256_file(Path(protocol["model_dir"])/"ASSET_MANIFEST.json") != protocol["model_manifest_sha256"]:
        raise ValueError("B0 asset manifest changed")
    return protocol, Format.from_dict(json.loads(Path(protocol["paths"]["format"]).read_text())["format"])


@torch.no_grad()
def generate_group(backend, prefix, *, fmt, seed, set_seed, audit_replay=False) -> list[dict]:
    """No record, future, score or target is accepted by the writer API."""
    value = torch.tensor([prefix], device=backend.device)
    backend.sync()                 # Settle all previous device work before reset.
    set_seed(seed)                  # No additional sync/MarkStep before sampling.
    started = time.monotonic()
    rollout = backend.sample(value, fmt=fmt, group_size=GROUP_SIZE,
                             max_actions=CAP, temperature=.7, policy_version=0)
    backend.sync()
    actions, masks, logps = (x.cpu()[0] for x in
        (rollout.actions, rollout.action_mask, rollout.behavior_logps))
    sampling_elapsed = time.monotonic()-started
    audit = {}
    audit_started = time.monotonic()
    if audit_replay:
        try:
            replay = backend.replay(value, rollout, fmt=fmt, temperature=.7)
            backend.sync()
            error = torch.where(rollout.action_mask,
                (replay-rollout.behavior_logps).abs(), 0.).max().cpu().item()
            audit = dict(cache_replay_max_error=error if math.isfinite(error) else repr(error),
                         cache_replay_status="ok" if math.isfinite(error) and error<=.001 else "error")
        except Exception as exc:
            audit = dict(cache_replay_max_error=None, cache_replay_status="error",
                         cache_replay_error=f"{type(exc).__name__}: {exc}")
    elapsed = time.monotonic()-started
    audit_elapsed = time.monotonic()-audit_started if audit_replay else 0.
    result = []
    for index in range(GROUP_SIZE):
        a, lp = actions[index][masks[index]].tolist(), logps[index][masks[index]].tolist()
        content = content_from_actions(a, fmt.stop_id, fmt.additional_stop_ids)
        finite = all(math.isfinite(v) for v in lp)
        valid = bool(a) and finite and audit.get("cache_replay_status","ok")=="ok"
        result.append(dict(status="ok" if valid else "error", actions=a,
            behavior_logps=[v if math.isfinite(v) else repr(v) for v in lp],
            action_mask=masks[index].tolist(), content_ids=content, n_actions=len(a),
            n_content_tokens=len(content), group_seed=seed,
            stop_type=(f"token:{a[-1]}" if a and a[-1] in {fmt.stop_id,*fmt.additional_stop_ids}
                       else "length_cap"), decode_iterations=rollout.decode_iterations,
            prefix_sha256=canonical_hash(prefix), group_wall_seconds=elapsed,
            group_sampling_wall_seconds=sampling_elapsed,
            group_replay_audit_wall_seconds=audit_elapsed,
            **audit, **({} if valid else {"error": "Invalid actions/logps or cached-sampling replay mismatch"})))
    return result


def generate(output: Path, device: str) -> None:
    protocol, fmt = read_protocol(output)
    records = load_selected(Path(protocol["paths"]["records"]), protocol["selection"], include_future=False)
    directory = output/"generation"
    directory.mkdir(exist_ok=False)
    backend, tok, checkpoint, signature, set_seed = setup_model(SimpleNamespace(
        model_dir=Path(protocol["model_dir"]), checkpoint=None, device=device))
    report = dict(status="running", started_utc=utc_now(), protocol_sha256=sha256_file(output/"protocol.json"),
                  model=signature, checkpoint=checkpoint, expected_candidates=len(records)*SAMPLES)
    atomic_json(directory/"run.json", report)
    count = 0
    started = time.monotonic()
    try:
        with (directory/"candidates.jsonl").open("x") as stream:
            for item_index,record in enumerate(records):
                for group in range(GROUPS):
                    seed = group_seed(protocol["seed"], record["item_id"], group)
                    try:
                        values = generate_group(backend, record["prefix"], fmt=fmt, seed=seed,
                            set_seed=set_seed, audit_replay=item_index==0 and group==0)
                    except Exception as exc:
                        values = [dict(status="error", error=f"{type(exc).__name__}: {exc}",
                                       group_seed=seed, prefix_sha256=canonical_hash(record["prefix"]))
                                  for _ in range(GROUP_SIZE)]
                    for local, value in enumerate(values):
                        row = dict(item_id=record["item_id"], document_id=record["document_id"], split="dev",
                                   candidate_index=group*GROUP_SIZE+local, group=group, group_sample=local, **value)
                        if "actions" in row:
                            row.update(text=tok.decode(row["actions"], skip_special_tokens=False),
                                       content_text=tok.decode(row["content_ids"], skip_special_tokens=False))
                        stream.write(json.dumps(row, allow_nan=False)+"\n"); stream.flush(); count += 1
                    if item_index==0 and group==0:
                        report["first_group_replay_check"] = dict(item_id=record["item_id"],group=group,
                            seed=seed,atol=.001,max_active_logp_error=values[0].get("cache_replay_max_error"),
                            status=values[0].get("cache_replay_status","error"))
                    if any(value["status"] != "ok" for value in values):
                        raise RuntimeError("Failed candidate group preserved; no automatic retry")
                print(json.dumps(dict(item_id=record["item_id"], candidates_written=count, utc=utc_now())), flush=True)
        report.update(status="complete", finished_utc=utc_now(), candidates=count,
                      sampling_wall_seconds=time.monotonic()-started,
                      candidates_sha256=sha256_file(directory/"candidates.jsonl"))
    except BaseException as exc:
        report.update(status="failed", finished_utc=utc_now(), candidates=count, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        atomic_json(directory/"run.json", report, overwrite=True)


def load_candidates(output: Path, protocol: dict, fmt: Format) -> dict:
    report = json.loads((output/"generation/run.json").read_text())
    path = output/"generation/candidates.jsonl"
    if (report["status"] != "complete" or report["protocol_sha256"] != sha256_file(output/"protocol.json")
            or report.get("candidates_sha256") != sha256_file(path)):
        raise ValueError("Complete, hash-bound sampling must precede scoring/future access")
    check=report.get("first_group_replay_check",{})
    if check.get("status")!="ok" or not isinstance(check.get("max_active_logp_error"),(float,int)) or not 0<=check["max_active_logp_error"]<=.001:
        raise ValueError("First actual candidate group must pass its cached/replay check")
    ids = protocol["selection"]["item_ids"]
    documents = dict(zip(ids, protocol["selection"]["document_ids"]))
    result = {}
    for line in path.read_text().splitlines():
        row = json.loads(line); key = (row["item_id"], row["candidate_index"])
        if key in result or key[0] not in documents or row["document_id"] != documents[key[0]] or row["split"] != "dev":
            raise ValueError("Duplicate/unexpected candidate or document")
        index = key[1]
        if (type(index) is not int or not 0 <= index < SAMPLES or row["group"] != index//GROUP_SIZE
                or row["group_sample"] != index%GROUP_SIZE
                or row["group_seed"] != group_seed(protocol["seed"], key[0], index//GROUP_SIZE)):
            raise ValueError("Candidate group/seed identity mismatch")
        a, lp = row.get("actions", []), row.get("behavior_logps", [])
        content = content_from_actions(a, fmt.stop_id, fmt.additional_stop_ids)
        stop_type=(f"token:{a[-1]}" if a and a[-1] in {fmt.stop_id,*fmt.additional_stop_ids}
                   else "length_cap")
        if (row["status"] != "ok" or not a or len(a)>CAP or len(lp)!=len(a)
                or any(type(token) is not int or token<0 for token in a)
                or not all(isinstance(v,(float,int)) and math.isfinite(v) for v in lp)
                or row["content_ids"] != content or len(content)<len(a)-1
                or (len(content)==len(a) and len(a)!=CAP)
                or row["n_actions"] != len(a) or row["n_content_tokens"] != len(content)
                or row.get("stop_type")!=stop_type
                or len(row.get("action_mask",[]))!=CAP
                or any(type(value) is not bool for value in row["action_mask"])
                or row["action_mask"] != [True]*len(a)+[False]*(CAP-len(a))):
            raise ValueError("Invalid candidate actions/logps/stopping")
        result[key] = row
    if set(result) != {(key,i) for key in ids for i in range(SAMPLES)}:
        raise ValueError("Every document needs all 16 candidates; no incomplete bundle scoring")
    return result


def construct_controls(protocol: dict, notes: dict, filler_ids: list[int]) -> dict:
    """Global same-index donors; no prefix, future or likelihood is accepted."""
    ids = protocol["selection"]["item_ids"]
    result = {}
    for candidate in range(SAMPLES):
        contents = [notes[(key,candidate)]["content_ids"] for key in ids]
        values = matched_controls(contents, filler_ids=filler_ids,
            seed=group_seed(protocol["control_seed"], "controls", candidate))
        for index,key in enumerate(ids):
            donor = ids[(index+1)%len(ids)]
            result[(key,candidate)] = dict(item_id=key, candidate_index=candidate,
                donor_item_id=donor, donor_candidate_index=candidate,
                donor_fit="cyclic repeat/trim; empty donor uses fixed filler",
                content_ids=dict(own=contents[index], **{name:values[name][index]
                    for name in ("filler","shuffled","other")}))
    return result


def pilot_baselines(protocol: dict) -> dict:
    expected={(key,"raw","raw") for key in protocol["selection"]["item_ids"]}
    expected|={(key,interface,"empty") for key in protocol["selection"]["item_ids"]
               for interface in protocol["interfaces"]}
    result={}
    for line in Path(protocol["pilot_scores"]).read_text().splitlines():
        row=json.loads(line);key=(row["item_id"],row["interface"],row["mode"])
        if key not in expected:continue
        values=row.get("token_log_likelihoods",[])
        if (key in result or row["status"]!="ok" or len(values)!=16
                or not all(isinstance(v,(float,int)) and math.isfinite(v) for v in values)):
            raise ValueError("Missing, duplicate or nonfinite pilot baseline")
        result[key]=values
    if set(result)!=expected:raise ValueError("Missing required pilot baseline")
    return result


def summarize(protocol: dict, rows: list[dict]) -> dict:
    ids = protocol["selection"]["item_ids"]
    expected = {(key,"raw",mode,None) for key in ids for mode in RAW_REPEATS}
    expected |= {(key,interface,"empty",None) for key in ids for interface in protocol["interfaces"]}
    expected |= {(key,interface,mode,i) for key in ids for interface in protocol["interfaces"]
                 for mode in CONTENT_MODES for i in range(SAMPLES)}
    keys = [(r["item_id"],r["interface"],r["mode"],r["candidate_index"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Full document/candidate/control coverage required")
    documents = dict(zip(ids,protocol["selection"]["document_ids"]))
    if any(row["document_id"]!=documents[row["item_id"]] for row in rows):
        raise ValueError("Candidate score document identities disagree")
    index = {key:row for key,row in zip(keys,rows)}
    per_document = []
    def value(key,horizon):
        row=index[key];lp=row.get("token_log_likelihoods",[])
        return sum(lp[:horizon]) if row["status"]=="ok" and len(lp)==16 and all(math.isfinite(v) for v in lp) else None
    for key in ids:
        for interface in protocol["interfaces"]:
            for horizon in protocol["horizons"]:
                own=[value((key,interface,"own",i),horizon) for i in range(SAMPLES)]
                empty=value((key,interface,"empty",None),horizon)
                metrics={}
                oracle=None
                if all(v is not None for v in own):
                    metrics["mean16_own_log_likelihood"]=statistics.mean(own)
                if empty is not None:metrics["empty_log_likelihood"]=empty
                if empty is not None and all(v is not None for v in own):
                    diffs=[v-empty for v in own]
                    chosen=max(range(SAMPLES),key=lambda i:diffs[i])
                    selected_row=index[(key,interface,"own",chosen)]
                    oracle=dict(candidate_index=chosen,
                        nonempty=selected_row.get("n_note_content_tokens",0)>0,
                        capped=selected_row.get("note_stop_type")=="length_cap",
                        candidates_beating_empty=sum(v>0 for v in diffs))
                    metrics.update(oracle_max_own_minus_empty=max(diffs),
                                   oracle_selected_log_likelihood=own[chosen],
                                   mean16_own_minus_empty=statistics.mean(diffs))
                for mode in (*RAW_REPEATS,*CONTENT_MODES[1:]):
                    other=([value((key,"raw",mode,None),horizon)]*SAMPLES if mode in RAW_REPEATS
                           else [value((key,interface,mode,i),horizon) for i in range(SAMPLES)])
                    if all(v is not None for v in other):
                        metrics[(f"{mode}_log_likelihood" if mode in RAW_REPEATS
                                 else f"mean16_{mode}_log_likelihood")]=statistics.mean(other)
                    if all(v is not None for v in [*own,*other]):
                        metrics[f"mean16_own_minus_{mode}"]=statistics.mean(a-b for a,b in zip(own,other))
                        if mode not in RAW_REPEATS:
                            metrics[f"max16_own_minus_max16_{mode}"]=max(own)-max(other)
                        if oracle is not None:
                            metrics[f"oracle_selected_own_minus_{mode}"]=own[chosen]-other[chosen]
                per_document.append(dict(item_id=key,document_id=documents[key],
                    quartile=protocol["strata"][key],interface=interface,horizon=horizon,
                    complete_candidates=sum(v is not None for v in own),oracle=oracle,metrics=metrics))
    groups={}
    for group in ["all",*sorted(set(protocol["strata"].values()))]:
        subset=[r for r in per_document if group=="all" or r["quartile"]==group]
        result={}
        for interface in protocol["interfaces"]:
            for horizon in protocol["horizons"]:
                selected=[r for r in subset if r["interface"]==interface and r["horizon"]==horizon]
                names={"oracle_max_own_minus_empty","mean16_own_minus_empty",
                       "mean16_own_log_likelihood","empty_log_likelihood","oracle_selected_log_likelihood",
                       *[f"{mode}_log_likelihood" for mode in RAW_REPEATS],
                       *[f"mean16_{mode}_log_likelihood" for mode in CONTENT_MODES[1:]],
                       *[f"max16_own_minus_max16_{mode}" for mode in CONTENT_MODES[1:]],
                       *[f"{kind}_own_minus_{mode}" for kind in ("mean16","oracle_selected")
                         for mode in (*RAW_REPEATS,*CONTENT_MODES[1:])]}
                for name in sorted(names):
                    valid=[r for r in selected if name in r["metrics"]]
                    values=[r["metrics"][name] for r in valid]
                    entry=dict(expected_documents=len(selected),complete_documents=len(valid),
                        incomplete_item_ids=[r["item_id"] for r in selected if name not in r["metrics"]],
                        mean_complete_documents_only=statistics.mean(values) if values else None,
                        median_complete_documents_only=statistics.median(values) if values else None,
                        positive_documents=sum(v>0 for v in values))
                    if len(valid)==len(selected) and len(valid)>=2 and name in {"oracle_max_own_minus_empty","mean16_own_minus_empty"}:
                        a=[dict(item_id=r["item_id"],document_id=r["document_id"],score=0.) for r in valid]
                        b=[dict(row,score=value) for row,value in zip(a,values)]
                        entry["document_bootstrap"]=paired_cluster_bootstrap(a,b,repeats=5000,seed=17)
                    result[f"{interface}/K{horizon}/{name}"]=entry
        groups[group]=result
    return dict(schema=SCHEMA, status="complete", rows=len(rows), failures=sum(r["status"]!="ok" for r in rows),
        groups=groups,per_document=per_document,limitations=protocol["limitations"],
        uncertainty="Candidates reduced within documents first; no sample-as-independent or training-seed CI")


def score(output: Path, device: str) -> None:
    protocol,fmt=read_protocol(output)
    notes=load_candidates(output,protocol,fmt)  # Must close all sampling before any future access.
    backend,tok,checkpoint,signature,_=setup_model(SimpleNamespace(
        model_dir=Path(protocol["model_dir"]),checkpoint=None,device=device))
    writer=json.loads((output/"generation/run.json").read_text())
    if signature!=writer["model"] or checkpoint!=writer["checkpoint"] or checkpoint is not None:
        raise ValueError("Candidate writer and reader must use the same unchanged B0")
    controls=construct_controls(protocol,notes,tok.encode(protocol["filler_text"],add_special_tokens=False))
    directory=output/"scoring";directory.mkdir(exist_ok=False)
    with (directory/"controls.jsonl").open("x") as stream:
        for value in controls.values():stream.write(json.dumps(value)+"\n")
    reference=pilot_baselines(protocol)
    records=load_selected(Path(protocol["paths"]["records"]),protocol["selection"],include_future=True)
    report=dict(status="running",started_utc=utc_now(),protocol_sha256=sha256_file(output/"protocol.json"),
                candidates_sha256=sha256_file(output/"generation/candidates.jsonl"),
                controls_sha256=sha256_file(directory/"controls.jsonl"),model=signature,checkpoint=checkpoint,
                expected_rows=protocol["expected_score_rows"])
    atomic_json(directory/"run.json",report)
    rows=[]
    maximum_reference_error=0.;reference_rows=0
    try:
        with (directory/"scores.jsonl").open("x") as stream:
            for record in records:
                key=record["item_id"]
                for i in range(SAMPLES):
                    if notes[(key,i)]["prefix_sha256"]!=canonical_hash(record["prefix"]):
                        raise ValueError("Candidate and scoring prefix hashes disagree")
                contexts=[("raw",mode,None,reader_context(record["prefix"],None,fmt,header=False,reanchor=n))
                          for mode,n in RAW_REPEATS.items()]
                for interface,(header,reanchor) in protocol["interfaces"].items():
                    contexts.append((interface,"empty",None,reader_context(record["prefix"],[],fmt,header=header,reanchor=reanchor)))
                    for i in range(SAMPLES):
                        for mode in CONTENT_MODES:
                            contexts.append((interface,mode,i,reader_context(record["prefix"],
                                controls[(key,i)]["content_ids"][mode],fmt,header=header,reanchor=reanchor)))
                for interface,mode,candidate,context in contexts:
                    started=time.monotonic();row=dict(item_id=key,document_id=record["document_id"],
                        interface=interface,mode=mode,candidate_index=candidate,context_tokens=len(context))
                    if candidate is not None:
                        note=notes[(key,candidate)]
                        row.update(n_note_content_tokens=note["n_content_tokens"],
                                   n_note_actions=note["n_actions"],note_stop_type=note["stop_type"])
                    try:
                        values=score_continuations(backend,context,[record["future"][:16]],
                            total_width=protocol["width"],future_width=16,pad_id=fmt.pad_id)[0]
                        row.update(status="ok",token_log_likelihoods=values.token_log_likelihoods)
                        baseline=reference.get((key,interface,mode)) if candidate is None else None
                        if baseline is not None:
                            error=max(abs(a-b) for a,b in zip(values.token_log_likelihoods,baseline))
                            row["pilot_reference_max_token_error"]=error
                            maximum_reference_error=max(maximum_reference_error,error);reference_rows+=1
                            if not math.isfinite(error) or error>.002:
                                raise RuntimeError(f"Pilot baseline parity failed: {error}")
                    except Exception as exc:
                        row.update(status="error",error=f"{type(exc).__name__}: {exc}")
                    row["wall_seconds"]=time.monotonic()-started;rows.append(row)
                    stream.write(json.dumps(row,allow_nan=False)+"\n");stream.flush()
                    if row["status"]!="ok":raise RuntimeError("Reader failure preserved; no automatic retry")
                print(json.dumps(dict(item_id=key,rows=len(rows),utc=utc_now())),flush=True)
        if reference_rows!=len(reference):raise ValueError("Not every pilot baseline was compared")
        report.update(pilot_baseline_max_token_error=maximum_reference_error,
                      pilot_baseline_rows_compared=reference_rows,pilot_baseline_atol=.002)
        result=summarize(protocol,rows);result.update(report);result.update(completed_utc=utc_now(),
            status="complete",scores_sha256=sha256_file(directory/"scores.jsonl"),
            sampling_counts=dict(Counter(row["stop_type"] for row in notes.values())),
            empty_candidates=sum(row["n_content_tokens"]==0 for row in notes.values()))
        atomic_json(directory/"summary.json",result)
        report.update(status="complete",finished_utc=utc_now(),rows=len(rows),summary_sha256=sha256_file(directory/"summary.json"))
    except BaseException as exc:
        report.update(status="failed",finished_utc=utc_now(),rows=len(rows),
                      pilot_baseline_max_token_error=maximum_reference_error,
                      pilot_baseline_rows_compared=reference_rows,error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        atomic_json(directory/"run.json",report,overwrite=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","generate","score"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--pilot-protocol",type=Path)
    parser.add_argument("--interfaces",nargs="+",choices=tuple(INTERFACES),default=DEFAULT_INTERFACES)
    parser.add_argument("--seed",type=int,default=1729)
    parser.add_argument("--control-seed",type=int,default=17)
    parser.add_argument("--device",choices=("cpu","xla"),default="xla")
    args=parser.parse_args()
    if args.command=="prepare":
        if args.pilot_protocol is None:parser.error("prepare requires the existing --pilot-protocol")
        protocol=prepare(args.output,args.pilot_protocol,interfaces=args.interfaces,seed=args.seed,control_seed=args.control_seed)
        print(json.dumps(dict(prepared=str(args.output),items=32,candidates=32*SAMPLES,
                              expected_score_rows=protocol["expected_score_rows"],interfaces=args.interfaces)))
    elif args.command=="generate":generate(args.output,args.device)
    else:score(args.output,args.device)


if __name__=="__main__":main()
