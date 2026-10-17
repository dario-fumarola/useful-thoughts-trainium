"""DEV8 diagnostic of tokenizer-undefined output probability, without sampling.

Both likelihood supports come from the same FP32 logits. Renormalization is a
diagnostic intervention; it neither changes B0 nor demonstrates useful thought.
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
from train.evaluation_integrity import evaluation_dependencies
from train.interface_probe import reader_context
from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file, utc_now
from weekend_thinking.core import Format

SCHEMA = "reader-support-dev8-v1"
MODES = ("empty", "own", "filler", "shuffled", "other")
INTERFACES = ("original", "reanchor32", "plain_reanchor32")


def sources():
    root = Path(__file__).resolve().parents[1]
    result = evaluation_dependencies()
    for name in ("train/reader_support_probe.py", "train/interface_probe.py"):
        result[name] = sha256_file(root/name)
    return result


def prepare(output: Path, pilot_path: Path, prompt_path: Path) -> dict:
    pilot = json.loads(pilot_path.read_text())
    prompt = json.loads(prompt_path.read_text())
    if (pilot["selection"]["split"] != "dev" or pilot["selection"]["count"] != 32
            or pilot["selection"] != prompt["selection"]):
        raise ValueError("Use the same fixed DEV32 completed probes")
    counts, ids = Counter(), []
    for key in pilot["selection"]["item_ids"]:
        group = pilot["strata"][key]
        if counts[group] < 2:
            ids.append(key); counts[group] += 1
    if len(counts) != 4 or set(counts.values()) != {2}:
        raise ValueError("Expected first two fixed IDs per frozen quartile")
    documents = dict(zip(pilot["selection"]["item_ids"], pilot["selection"]["document_ids"]))
    selection = dict(pilot["selection"], item_ids=ids, document_ids=[documents[k] for k in ids],
                     count=len(ids), item_ids_sha256=canonical_hash(ids))
    paths = {k: pilot["paths"][k] for k in ("records", "format", "strata", "controls")}
    paths.update(pilot_protocol=str(pilot_path.resolve()), prompt_protocol=str(prompt_path.resolve()),
                 pilot_scores=str((pilot_path.parent/"scores.jsonl").resolve()),
                 prompt_scores=str((prompt_path.parent/"scores.jsonl").resolve()))
    for directory in (pilot_path.parent, prompt_path.parent):
        summary = json.loads((directory/"summary.json").read_text())
        rows = [json.loads(line) for line in (directory/"scores.jsonl").read_text().splitlines()]
        if (summary["status"] != "complete" or any(r["status"] != "ok" for r in rows)
                or summary["scores_sha256"] != sha256_file(directory/"scores.jsonl")):
            raise ValueError("Require complete, failure-free reference probes")
    plain = prompt["configurations"]["plain_reanchor32"]
    if plain != dict(header=[], open=[198], close=[198], repeat=32):
        raise ValueError("Plain-newline interface differs from completed reference")
    protocol = dict(schema=SCHEMA, experiment_id=output.name, created_utc=utc_now(),
        purpose="Exploratory held DEV output-support mass diagnostic; no sampling or training",
        selection=selection, strata={k: pilot["strata"][k] for k in ids},
        selection_rule="First two IDs per frozen B0 quartile, in the completed DEV32 order",
        paths=paths, sha256={k: sha256_file(v) for k,v in paths.items() if k != "records"},
        model_dir=pilot["model_dir"], model_manifest_sha256=sha256_file(Path(pilot["model_dir"])/"ASSET_MANIFEST.json"),
        compute="fp32", interfaces=INTERFACES, modes=MODES, width=1664, horizons=[1,16],
        newline_ids=[198], expected_rows=len(ids)*(1+len(INTERFACES)*len(MODES)),
        supports=["full_model_vocabulary", "tokenizer_defined_renormalized"],
        formula="logP_defined(y)=logP_full(y)-logP_full(defined); all from identical logits",
        reference_atol=.002, source_sha256=sources(), limitations=[
            "Original G1 prefix-only notes and global controls; no new writer.",
            "Renormalizing output support does not create reasoning or establish content utility.",
            "Eight fixed development documents; not a confirmatory or final-test result."])
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/"protocol.json", protocol)
    return protocol


def read_protocol(output):
    p = json.loads((output/"protocol.json").read_text())
    if p["schema"] != SCHEMA or p["source_sha256"] != sources():
        raise ValueError("Support probe source/protocol changed")
    for key,digest in p["sha256"].items():
        if sha256_file(p["paths"][key]) != digest:
            raise ValueError(f"Changed support probe input: {key}")
    if sha256_file(Path(p["model_dir"])/"ASSET_MANIFEST.json") != p["model_manifest_sha256"]:
        raise ValueError("B0 asset manifest changed")
    return p, Format.from_dict(json.loads(Path(p["paths"]["format"]).read_text())["format"])


def support_from_logits(logits, targets, defined):
    """Full and renormalized likelihoods plus log defined mass, without mutation."""
    full = logits.log_softmax(-1).gather(-1, targets[..., None])[..., 0]
    full_normalizer = logits.logsumexp(-1)
    defined_normalizer = logits.masked_fill(~defined, -torch.inf).logsumexp(-1)
    log_mass = defined_normalizer-full_normalizer
    return full, full-log_mass, log_mass


@torch.no_grad()
def score_support(backend, context, future, *, defined, width, pad_id):
    if not context or len(future) != 16 or len(context)+16 > width:
        raise ValueError("Require one real context and all sixteen future tokens within static width")
    if any(p.dtype != torch.float32 for p in backend.model.parameters() if p.is_floating_point()):
        raise ValueError("Support diagnostic requires FP32 weights")
    ids = torch.full((1,width),pad_id,dtype=torch.long)
    ids[0,:len(context)+16] = torch.tensor(context+future)
    targets = torch.tensor([future],device=backend.device)
    positions = (torch.arange(16)+len(context)-1)[None].to(backend.device)
    hidden = backend.hidden(ids.to(backend.device))
    if hidden.dtype != torch.float32:
        raise ValueError("Support diagnostic requires FP32 trunk")
    selected = hidden.gather(1,positions[:,:,None].expand(-1,-1,hidden.shape[-1]))
    logits = backend.model.lm_head(selected)
    if logits.dtype != torch.float32:
        raise ValueError("Support diagnostic requires FP32 head")
    result = support_from_logits(logits,targets,defined)
    backend.sync()
    values = [v.cpu()[0].tolist() for v in result]
    if not all(math.isfinite(x) for v in values for x in v):
        raise ValueError("Nonfinite full/support mass likelihood")
    return dict(token_log_likelihoods=values[0], defined_token_log_likelihoods=values[1],
                token_log_defined_mass=values[2])


def references(protocol):
    ids = set(protocol["selection"]["item_ids"])
    expected = {(key,"raw","raw") for key in ids}
    expected |= {(key,interface,mode) for key in ids for interface in INTERFACES for mode in MODES}
    result = {}
    for source in ("pilot_scores","prompt_scores"):
        for line in Path(protocol["paths"][source]).read_text().splitlines():
            row = json.loads(line); key = (row["item_id"],row["interface"],row["mode"])
            # The interface pilot supplies raw/original/reanchor; prompt supplies plain.
            if key not in expected or (source == "prompt_scores") != (key[1] == "plain_reanchor32"):
                continue
            values = row.get("token_log_likelihoods",[])
            if key in result or row["status"] != "ok" or len(values) != 16 or not all(math.isfinite(v) for v in values):
                raise ValueError("Missing, duplicate or invalid full-support reference")
            result[key] = values
    if set(result) != expected:
        raise ValueError("Every support context requires an existing full-support reference")
    return result


def contexts(prefix, control, fmt, newline):
    result = [("raw","raw",list(prefix))]
    for interface in INTERFACES:
        for mode in MODES:
            content = control["content_ids"][mode]
            value = (list(prefix)+newline+content+newline+list(prefix[-32:])
                     if interface == "plain_reanchor32" else
                     reader_context(prefix,content,fmt,header=True,reanchor=32 if interface == "reanchor32" else 0))
            result.append((interface,mode,value))
    return result


def summarize(protocol, rows):
    groups = {}
    for group in ("all",*sorted(set(protocol["strata"].values()))):
        ids = [k for k in protocol["selection"]["item_ids"] if group == "all" or protocol["strata"][k] == group]
        selected = [r for r in rows if r["item_id"] in ids]
        index = {(r["item_id"],r["interface"],r["mode"]): r for r in selected}
        results = {}
        for horizon in protocol["horizons"]:
            for interface,modes in [("raw",("raw",)),*((i,MODES) for i in INTERFACES)]:
                for mode in modes:
                    values = [index[k,interface,mode] for k in ids]
                    masses = [x for r in values for x in r["token_log_defined_mass"][:horizon]]
                    item = dict(documents=len(ids), mean_log_defined_mass_per_token=statistics.mean(masses),
                                mean_undefined_mass_per_token=statistics.mean(-math.expm1(x) for x in masses),
                                max_undefined_mass=max(-math.expm1(x) for x in masses))
                    for support,field in (("full","token_log_likelihoods"),("defined","defined_token_log_likelihoods")):
                        item[support+"_mean_log_likelihood"] = statistics.mean(sum(r[field][:horizon]) for r in values)
                    results[f"{interface}/{mode}/K{horizon}"] = item
            for interface in INTERFACES:
                for support,field in (("full","token_log_likelihoods"),("defined","defined_token_log_likelihoods")):
                    for treatment,baseline in (("empty","raw"),("own","raw"),("own","empty"),("own","filler"),("own","shuffled"),("own","other")):
                        diffs = [sum(index[k,interface,treatment][field][:horizon])-sum(index[k,"raw" if baseline == "raw" else interface,baseline][field][:horizon]) for k in ids]
                        results[f"{interface}/{treatment}_minus_{baseline}/K{horizon}/{support}"] = dict(
                            documents=len(ids),mean_nats=statistics.mean(diffs),positive_documents=sum(x>0 for x in diffs))
        groups[group] = results
    return groups


def run(output, device):
    p,fmt = read_protocol(output)
    reference = references(p)
    controls = {r["item_id"]: r for r in map(json.loads,Path(p["paths"]["controls"]).read_text().splitlines())}
    if any(controls[k].get("failures") for k in p["selection"]["item_ids"]):
        raise ValueError("Original controls contain failures; do not filter them")
    # All original global G1 notes/controls were frozen before any target access.
    records = load_selected(Path(p["paths"]["records"]),p["selection"],include_future=True)
    backend,tok,checkpoint,signature,_ = setup_model(SimpleNamespace(model_dir=Path(p["model_dir"]),checkpoint=None,device=device))
    if tok.encode("\n",add_special_tokens=False) != p["newline_ids"] or checkpoint is not None:
        raise ValueError("Use unchanged B0 and the pinned newline tokenization")
    defined_ids = set(tok.get_vocab().values())
    vocab = backend.model.config.vocab_size
    if not defined_ids or min(defined_ids)<0 or max(defined_ids)>=vocab:
        raise ValueError("Invalid tokenizer/model support")
    mask = torch.zeros(vocab,dtype=torch.bool); mask[list(defined_ids)] = True
    mask = mask.to(backend.device)
    report = dict(status="running",started_utc=utc_now(),protocol_sha256=sha256_file(output/"protocol.json"),
                  model=signature,checkpoint=checkpoint,defined_ids=sorted(defined_ids),undefined_ids=sorted(set(range(vocab))-defined_ids))
    # Store the long support list separately, keeping run metadata compact.
    atomic_json(output/"support.json",dict(defined_ids=report.pop("defined_ids"),undefined_ids=report.pop("undefined_ids")))
    report["support_sha256"] = sha256_file(output/"support.json")
    atomic_json(output/"run.json",report)
    rows = []; maximum = 0.
    try:
        with (output/"scores.jsonl").open("x") as stream:
            for n,record in enumerate(records):
                if len(record["future"]) != 16 or not set(record["future"]) <= defined_ids:
                    raise ValueError("Expected sixteen tokenizer-defined target tokens")
                for interface,mode,context in contexts(record["prefix"],controls[record["item_id"]],fmt,p["newline_ids"]):
                    started = time.monotonic()
                    row = dict(item_id=record["item_id"],document_id=record["document_id"],interface=interface,
                               mode=mode,context_tokens=len(context),context_sha256=canonical_hash(context))
                    try:
                        row.update(score_support(backend,context,record["future"],defined=mask,width=p["width"],pad_id=fmt.pad_id))
                        error = max(abs(a-b) for a,b in zip(row["token_log_likelihoods"],reference[record["item_id"],interface,mode]))
                        maximum = max(maximum,error)
                        row.update(status="ok",reference_max_token_error=error)
                        if error > p["reference_atol"]:
                            raise ValueError(f"Full-support reference mismatch: {error}")
                    except Exception as exc:
                        row.update(status="error",error=f"{type(exc).__name__}: {exc}")
                    row["wall_seconds"] = time.monotonic()-started
                    rows.append(row); stream.write(json.dumps(row,allow_nan=False)+"\n"); stream.flush()
                    if row["status"] != "ok":
                        raise RuntimeError("Support probe failed; partial rows retained, no retry")
                print(json.dumps(dict(items_complete=n+1,rows=len(rows),utc=utc_now())),flush=True)
        if len(rows) != p["expected_rows"]:
            raise ValueError("Incomplete fixed support matrix")
        result = dict(status="complete",schema=SCHEMA,completed_utc=utc_now(),rows=len(rows),failures=0,
                      maximum_reference_error=maximum,groups=summarize(p,rows),limitations=p["limitations"],
                      protocol_sha256=sha256_file(output/"protocol.json"),scores_sha256=sha256_file(output/"scores.jsonl"))
        atomic_json(output/"summary.json",result)
        report.update(status="complete",finished_utc=utc_now(),rows=len(rows),summary_sha256=sha256_file(output/"summary.json"))
    except BaseException as exc:
        report.update(status="failed",finished_utc=utc_now(),rows=len(rows),error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        atomic_json(output/"run.json",report,overwrite=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","run"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--pilot-protocol",type=Path)
    parser.add_argument("--prompt-protocol",type=Path)
    parser.add_argument("--device",choices=("cpu","xla"),default="xla")
    args = parser.parse_args()
    if args.command == "prepare":
        if args.pilot_protocol is None or args.prompt_protocol is None:
            parser.error("prepare requires both completed probe protocols")
        p = prepare(args.output,args.pilot_protocol,args.prompt_protocol)
        print(json.dumps(dict(prepared=str(args.output),items=8,expected_rows=p["expected_rows"])))
    else:
        run(args.output,args.device)


if __name__ == "__main__":
    main()
