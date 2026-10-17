"""Fixed-DEV FP32 reader/replay cost and truncation diagnostic; never trains.

Preparation reads saved B0 notes, not likelihood outcomes. Execution uses the
unchanged backend with full 1024-wide and right-truncated action buffers. The
first call and its immediate repeat are reported separately; an existing disk
cache means that "first" does not imply a fresh compiler invocation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

from weekend_thinking.artifacts import (
    atomic_json, canonical_hash, require_utc_deadline, sha256_file, utc_now,
)

ROOT = Path(__file__).resolve().parents[1]
CAP = 1024
BUCKETS = (1, 32, 64, 128, 256, 512, 1024)


def action_bucket(actions, stop_ids):
    """Keep every active action, including its first real terminator."""
    if not actions or len(actions) > CAP or any(type(t) is not int or t < 0 for t in actions):
        raise ValueError("Expected 1..1024 saved token IDs")
    stops = [i for i, token in enumerate(actions) if token in stop_ids]
    if stops and stops[0] != len(actions) - 1:
        raise ValueError("Saved active actions contain tokens after their first stop")
    if not stops and len(actions) != CAP:
        raise ValueError("An uncapped saved note must end in a real stop")
    return next(n for n in BUCKETS if n >= len(actions))


def choose_cases(selection, notes, stop_ids):
    """First DEV empty control + first naturally stopped short saved note.

    Length is a prospective performance condition, never a likelihood filter.
    If no 2..128-action saved note exists, retain the first saved note instead.
    """
    ids = selection["item_ids"]
    if selection["split"] != "dev":
        raise ValueError("This diagnostic accepts the frozen DEV selection only")
    if set(notes) != set(ids):
        raise ValueError("Saved notes must cover the entire fixed selection")
    for key in ids:
        row = notes[key]
        if row["status"] != "ok" or row["split"] != "dev":
            raise ValueError("Do not silently drop a failed or non-DEV note")
        action_bucket(row["actions"], stop_ids)
    first = ids[0]
    short = next((key for key in ids if 2 <= len(notes[key]["actions"]) <= 128
                  and notes[key]["actions"][-1] in stop_ids), None)
    saved = short or first
    return [dict(name="first_DEV_empty", item_id=first,
                 actions=[min(stop_ids)], synthetic_empty_control=True),
            dict(name="first_short_saved" if short else "first_saved_no_short_available",
                 item_id=saved, actions=notes[saved]["actions"],
                 synthetic_empty_control=False)]


def prepare(args):
    # Keep XLA imports out of preparation and CPU test collection.
    from train.evaluate import asset_provenance
    from train.evaluate_local import load_selected, read_selection
    from weekend_thinking.core import Format

    selection = read_selection(args.selection, qualification=True, split="dev")
    note_run = json.loads(args.notes.with_name("run.json").read_text())
    if (note_run["checkpoint"] is not None or note_run["status"] != "complete"
            or note_run["protocol"]["compute"] != "fp32"
            or note_run["protocol"]["split"] != "dev"
            or note_run["protocol"]["selection_item_ids_sha256"] != selection["item_ids_sha256"]
            or note_run["notes_sha256"] != sha256_file(args.notes)):
        raise ValueError("Expected the complete unchanged-B0 FP32 DEV writer bundle")
    fmt = Format.from_dict(json.loads(args.format.read_text())["format"])
    notes = {}
    for line in args.notes.read_text().splitlines():
        row = json.loads(line)
        if row["item_id"] in notes:
            raise ValueError("Duplicate saved note")
        notes[row["item_id"]] = row
    stop_ids = {fmt.stop_id, *fmt.additional_stop_ids}
    cases = choose_cases(selection, notes, stop_ids)
    # The empty control uses the primary stop, not whichever ID sorts first.
    cases[0]["actions"] = [fmt.stop_id]
    ids = list(dict.fromkeys(case["item_id"] for case in cases))
    documents = dict(zip(selection["item_ids"], selection["document_ids"]))
    subset = dict(selection, item_ids=ids, document_ids=[documents[key] for key in ids],
                  count=len(ids), item_ids_sha256=canonical_hash(ids))
    records = {r["item_id"]: r for r in load_selected(args.records, subset, include_future=True)}
    for case in cases:
        record = records[case["item_id"]]
        if canonical_hash(record["prefix"]) != notes[case["item_id"]]["prefix_sha256"]:
            raise ValueError("Saved note and selected real prefix differ")
        case.update(prefix=record["prefix"], future=record["future"][:1],
                    document_id=record["document_id"], split="dev",
                    active_actions=len(case["actions"]),
                    shortened_width=action_bucket(case["actions"], stop_ids))
        case["projection_sha256"] = canonical_hash(case)
    model = asset_provenance(args.model_dir)
    if model != note_run["model"]:
        raise ValueError("B0 weights/assets differ from the saved-note writer")
    deps = [ROOT/name for name in ("train/profile_replay.py", "train/neuron_backend.py",
        "train/runtime_compat.py", "train/evaluate.py", "train/evaluate_local.py",
        "weekend_thinking/artifacts.py", "weekend_thinking/backend_contract.py",
        "weekend_thinking/core.py", "weekend_thinking/pretrained.py",
        "weekend_thinking/runtime_policy.py")]
    paths = dict(notes=args.notes.resolve(), note_run=args.notes.with_name("run.json").resolve(),
                 selection=args.selection.resolve(), format=args.format.resolve())
    protocol = dict(schema="fixed-DEV-replay-profile-v1", prepared_utc=utc_now(),
        experiment_id=args.output.name, model=model, model_dir=str(args.model_dir.resolve()),
        source_files={str(p.resolve()): sha256_file(p) for p in deps},
        input_files={name:dict(path=str(p), sha256=sha256_file(p)) for name,p in paths.items()},
        records_path=str(args.records.resolve()), records_sha256=selection["records_sha256"],
        cases=cases, cap=CAP, buckets=BUCKETS, advantages=[1.0, -1.0],
        temperature=.7, divisor=16, repeats=2, compute="fp32", checkpoint_trunk=True,
        criteria=dict(loss_atol=.001, reader_atol=.001, full_gradient_relative_l2_max=.001,
                      full_gradient_cosine_min=.999999),
        no_optimizer=True, no_sampling=True, no_final_test=True,
        limits=["Only right padding is removed; sampling cap, active actions, stop and normalization are unchanged.",
                "These isolated fixed-note phases are not end-to-end training throughput.",
                "First-observed/immediate-repeat times include explicit sync; disk-cache hits can load executables.",
                "Full unique-parameter gradient comparison/export is outside timed forward/backward phases."])
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output/"protocol.json", protocol)
    print(json.dumps(dict(protocol=str(args.output/"protocol.json"),
                          sha256=sha256_file(args.output/"protocol.json"),
                          cases=[{k:v for k,v in row.items() if k in
                                  {"name","item_id","active_actions","shortened_width"}} for row in cases])), flush=True)


def gradient_geometry(reference, actual):
    """Full FP64 accumulation, chunked to bound host memory for the embedding."""
    import torch
    if reference.shape != actual.shape or reference.dtype != torch.float32 or actual.dtype != torch.float32:
        raise ValueError("Gradient shape/FP32 dtype mismatch")
    totals = dict(reference_squared=0., actual_squared=0., error_squared=0., dot=0., max_abs=0.)
    a, b = reference.reshape(-1), actual.reshape(-1)
    for start in range(0, a.numel(), 1024*1024):
        x = a[start:start+1024*1024].double()
        y = b[start:start+1024*1024].double()
        if not bool(torch.isfinite(x).all() & torch.isfinite(y).all()):
            raise ValueError("Nonfinite full gradient")
        error = y-x
        totals["reference_squared"] += float(x.square().sum())
        totals["actual_squared"] += float(y.square().sum())
        totals["error_squared"] += float(error.square().sum())
        totals["dot"] += float((x*y).sum())
        totals["max_abs"] = max(totals["max_abs"], float(error.abs().max()))
    return dict(elements=a.numel(), **totals,
        reference_bytes_sha256=hashlib.sha256(memoryview(reference.contiguous().numpy()).cast("B")).hexdigest(),
        actual_bytes_sha256=hashlib.sha256(memoryview(actual.contiguous().numpy()).cast("B")).hexdigest())


def geometry_summary(totals):
    a, b, e = (totals[k] for k in ("reference_squared", "actual_squared", "error_squared"))
    return dict(**totals, relative_l2=math.sqrt(e/a) if a else (0. if not e else None),
                cosine=totals["dot"]/math.sqrt(a*b) if a and b else (1. if not a and not b else None))


def run(args):
    import torch
    from train.neuron_backend import NeuronCausalBackend
    from train.runtime_compat import configure_runtime
    from train.evaluate import asset_provenance
    from weekend_thinking.backend_contract import Rollouts
    from weekend_thinking.core import Format, action_mask
    from weekend_thinking.pretrained import load_base
    from weekend_thinking.runtime_policy import runtime_policy
    import torch_xla
    from torch_xla.debug import metrics

    pp = args.output/"protocol.json"
    if sha256_file(pp) != args.protocol_sha256:
        raise ValueError("Protocol hash differs from the reviewed preparation")
    protocol = json.loads(pp.read_text())
    for path, digest in protocol["source_files"].items():
        if sha256_file(path) != digest:
            raise ValueError(f"Changed source {path}")
    for value in protocol["input_files"].values():
        if sha256_file(value["path"]) != value["sha256"]:
            raise ValueError(f"Changed input {value['path']}")
    if asset_provenance(Path(protocol["model_dir"])) != protocol["model"]:
        raise ValueError("Changed released B0 assets")
    deadline = require_utc_deadline(args.deadline_utc)
    def check_deadline():
        if datetime.now(timezone.utc) >= deadline:
            raise TimeoutError("Diagnostic deadline reached; no automatic retry")
    check_deadline()
    atomic_json(args.output/"started.json", dict(started_utc=utc_now(),
        protocol_sha256=args.protocol_sha256, deadline_utc=deadline.isoformat()))
    report = dict(status="running", protocol_sha256=args.protocol_sha256, phases=[], comparisons=[])
    def publish():
        atomic_json(args.output/"result.json", report, overwrite=True)
    def snapshot():
        return {"counters":{n:metrics.counter_value(n) or 0 for n in
                            ("CachedCompile","UncachedCompile","ExecuteComputation","MarkStep")},
                "metric_totals_raw":{n:list(metrics.metric_data(n)[:2]) if metrics.metric_data(n) else [0,0]
                                     for n in ("CompileTime","ExecuteTime","TransferFromDeviceTime")}}
    sync = lambda: torch_xla.sync(wait=True)
    def timed(label, fn):
        check_deadline(); sync(); before=snapshot(); started=time.monotonic()
        value=fn(); sync(); elapsed=time.monotonic()-started; after=snapshot()
        row=dict(label=label, wall_seconds=elapsed, xla_before=before, xla_after=after,
                 counters_delta={k:after["counters"][k]-before["counters"][k] for k in before["counters"]})
        report["phases"].append(row); publish()
        print(json.dumps(dict(time_utc=utc_now(), **row)), flush=True)
        return value
    try:
        report["runtime_compatibility"] = configure_runtime()
        report["runtime_policy"] = runtime_policy()
        tempdir=Path(tempfile.gettempdir()).resolve()
        report["host_io"] = dict(effective_tempdir=str(tempdir), tempdir_device=tempdir.stat().st_dev,
                                  compiler_cache_url=os.environ.get("NEURON_COMPILE_CACHE_URL"))
        model, tok = load_base(Path(protocol["model_dir"]))
        device = torch_xla.device(); model.to(device); model.tie_weights(); model.eval()
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("Expected actual FP32 B0 parameters")
        if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
            raise ValueError("Actual released embedding/head tie lost")
        valid = set(tok.get_vocab().values())
        backend = NeuronCausalBackend(model, sync=sync,
            banned_ids=tuple(i for i in range(model.config.vocab_size) if i not in valid))
        fmt = Format.from_dict(json.loads(Path(protocol["input_files"]["format"]["path"]).read_text())["format"])
        sync(); publish()
        for case in protocol["cases"]:
            prefix=torch.tensor([case["prefix"]],device=device)
            future=torch.tensor([case["future"]],device=device)
            widths={"full":CAP,"shortened":case["shortened_width"]}
            reader={}
            for variant,width in widths.items():
                active=case["actions"]
                actions=torch.tensor([active+[fmt.pad_id]*(width-len(active))],device=device)
                rollout=Rollouts(actions[:,None], action_mask(actions[:,None],fmt.stop_id,fmt.additional_stop_ids),
                                 torch.zeros((1,1,width),device=device),0)
                reader[variant]=[]
                for repeat in range(protocol["repeats"]):
                    with torch.no_grad():
                        value=timed(f"{case['name']}/reader/{variant}/{repeat}",
                            lambda:backend.score(prefix,future,rollout,fmt=fmt,horizon=1))
                    reader[variant].append(float(value.cpu()[0,0]))
                    if not math.isfinite(reader[variant][-1]):
                        raise ValueError("Nonfinite reader likelihood")
            reader_error=max(abs(a-b) for a,b in zip(reader["full"],reader["shortened"]))
            report["comparisons"].append(dict(case=case["name"],phase="reader",values=reader,
                max_abs=reader_error,passed=reader_error<=protocol["criteria"]["reader_atol"]))
            for advantage in protocol["advantages"]:
                reference=None; losses={}; geometry=[]
                for variant,width in widths.items():
                    actions=torch.tensor([case["actions"]+[fmt.pad_id]*(width-case["active_actions"])],device=device)
                    mask=action_mask(actions,fmt.stop_id,fmt.additional_stop_ids)
                    adv=torch.tensor(advantage,device=device)
                    losses[variant]=[]
                    for repeat in range(protocol["repeats"]):
                        model.zero_grad(set_to_none=True)
                        label=f"{case['name']}/adv{advantage}/{variant}/{repeat}"
                        loss=timed(label+"/forward",lambda:backend.replay_one_loss(prefix,actions,mask,adv,
                            fmt=fmt,temperature=protocol["temperature"],divisor=protocol["divisor"]))
                        losses[variant].append(float(loss.detach().cpu()))
                        if not math.isfinite(losses[variant][-1]):
                            raise ValueError("Nonfinite signed replay loss")
                        timed(label+"/backward",loss.backward)
                        # Export only AFTER the immediate compute repeat. A
                        # large gradient D2H pass between repeats would perturb
                        # the LRU2 working set we are trying to observe.
                        if repeat==protocol["repeats"]-1:
                            exported=time.monotonic()
                            if variant=="full":
                                reference={name:p.grad.detach().cpu() for name,p in model.named_parameters()}
                            else:
                                for name,p in model.named_parameters():
                                    row=gradient_geometry(reference[name],p.grad.detach().cpu())
                                    geometry.append(dict(parameter=name,**geometry_summary(row)))
                            report["phases"].append(dict(label=label+"/gradient_export_and_comparison",
                                wall_seconds=time.monotonic()-exported,excluded_from_compute_timing=True))
                total={k:sum(r[k] for r in geometry) for k in
                       ("elements","reference_squared","actual_squared","error_squared","dot")}
                total["max_abs"]=max(r["max_abs"] for r in geometry)
                summary=geometry_summary(total); criteria=protocol["criteria"]
                loss_error=max(abs(a-b) for a,b in zip(losses["full"],losses["shortened"]))
                passed=(summary["relative_l2"] is not None and summary["relative_l2"]<=criteria["full_gradient_relative_l2_max"]
                        and summary["cosine"] is not None and summary["cosine"]>=criteria["full_gradient_cosine_min"]
                        and loss_error<=criteria["loss_atol"])
                report["comparisons"].append(dict(case=case["name"],phase="replay",advantage=advantage,
                    losses=losses,loss_abs_error=loss_error,
                    full_gradient=summary,per_parameter=geometry,passed=passed))
                del reference; model.zero_grad(set_to_none=True); sync(); publish()
        report.update(status="passed" if all(x["passed"] for x in report["comparisons"]) else "failed_agreement",
                      finished_utc=utc_now(),raw_gradient_files_saved=False)
        (args.output/"xla-metrics.txt").write_text(metrics.metrics_report())
        publish()
        if report["status"]!="passed":
            raise RuntimeError("Full-width/shortened semantic agreement failed; see preserved result")
    except BaseException as exc:
        report.update(status="failed" if report["status"]=="running" else report["status"],
                      error=f"{type(exc).__name__}: {exc}",finished_utc=utc_now())
        publish(); raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("prepare","run"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--model-dir",type=Path,default=ROOT/"local/assets/qwen3-1.7b-base")
    p.add_argument("--selection",type=Path,default=ROOT/"local/data/finemath/fixed-local-evaluation-v1.ids.json")
    p.add_argument("--records",type=Path,default=ROOT/"local/data/finemath/production-500000.audit-v2.records.jsonl")
    p.add_argument("--notes",type=Path,default=ROOT/"local/nvme/evaluation/local-dev512-v1-B0-001/notes/notes.jsonl")
    p.add_argument("--format",type=Path,default=ROOT/"local/science/resolved_format.absolute.json")
    p.add_argument("--protocol-sha256")
    p.add_argument("--deadline-utc")
    args=p.parse_args()
    if args.command=="prepare":
        prepare(args)
    else:
        if not args.protocol_sha256 or not args.deadline_utc:
            p.error("run requires the reviewed --protocol-sha256 and finite --deadline-utc")
        run(args)


if __name__=="__main__":
    main()
