"""No-training TRAIN32 calculation diagnostic with notes frozen before gold.

Answer generation accepts only question-derived, sealed contexts. Gold is
projected in a later command after every answer shard closes. Forced numeric
likelihood is a separate diagnostic; it is not free-running answer accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from train.evaluate import GreedyDecoder, GenerationFailure, question_prompt, validate_prompt_spec
from train.evaluate_likelihood import score_continuations
from train.evaluate_local import setup_model
from train.evaluation_integrity import evaluation_dependencies
from weekend_thinking.artifacts import atomic_json, canonical_hash, require_utc_deadline, sha256_file, utc_now
from weekend_thinking.core import matched_controls
from weekend_thinking.evaluation import numeric_answer, score_predictions
from weekend_thinking.stats import paired_cluster_bootstrap

SCHEMA = "gsm8k-calculation-reader-train32-v1"
MODES = ("raw", "empty", "generated", "filler", "shuffled", "other",
         "verified", "corrupted", "verified_shuffled", "verified_other")
CONTRASTS = (("raw", "empty"), ("raw", "generated"), ("empty", "generated"),
             ("other", "generated"), ("shuffled", "generated"), ("filler", "generated"),
             ("raw", "verified"), ("empty", "verified"), ("corrupted", "verified"),
             ("verified_other", "verified"), ("verified_shuffled", "verified"))
TRAIN_SHA = "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d"


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def sources():
    root = Path(__file__).resolve().parents[1]
    return {**evaluation_dependencies(), **{f: sha256_file(root/f) for f in
            ("train/calculation_pilot.py", "train/calculation_teacher.py", "train/evaluate_local.py",
             "train/evaluate_likelihood.py", "weekend_thinking/core.py")}}


def check_deadline(deadline):
    if datetime.now(timezone.utc) >= require_utc_deadline(deadline):
        raise RuntimeError("Finite diagnostic deadline reached; no automatic retry")


def make_contexts(prefix, generated, verified, corrupted, generated_controls, verified_controls):
    content = dict(empty=[], generated=list(generated), verified=list(verified), corrupted=list(corrupted),
                   **{k: generated_controls[k] for k in ("filler", "shuffled", "other")},
                   verified_shuffled=verified_controls["shuffled"], verified_other=verified_controls["other"])
    return {"raw": list(prefix), **{mode: list(ids)+[271]+list(prefix) for mode, ids in content.items()}}


def prepare(output, teacher_dir, verified_file, prompt_file, model_dir):
    from transformers import AutoTokenizer
    from train.calculation_teacher import load_complete
    tp, inputs, notes, receipts = load_complete(teacher_dir)
    questions = inputs["rows"]
    ids = [q["item_id"] for q in questions]
    if len(ids) != 32 or len(set(ids)) != 32:
        raise ValueError("Exactly32 unique TRAIN questions required")
    verified_rows = rows(verified_file)
    verified = {r["item_id"]: r for r in verified_rows}
    if len(verified_rows) != 32 or set(verified) != set(ids):
        raise ValueError("Keep all32 verified and corrupted notes; no item selection")
    if any(verified[q["item_id"]]["question_sha256"] != hashlib.sha256(q["question"].encode()).hexdigest()
           for q in questions):
        raise ValueError("Verified calculations must bind the exact current question text")
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    if tok.encode("\n\n", add_special_tokens=False) != [271]:
        raise ValueError("Released newline separator changed")
    spec = read(prompt_file); validate_prompt_spec(spec, set(ids))
    # Question-only notes and the reader context are fixed before any gold access.
    prefixes = [tok.encode(question_prompt(q["question"], spec), add_special_tokens=False) for q in questions]
    generated = [notes[k]["content_ids"] for k in ids]
    correct = [tok.encode(verified[k]["verified_text"], add_special_tokens=False) for k in ids]
    corrupted = [tok.encode(verified[k]["corrupted_text"], add_special_tokens=False) for k in ids]
    if any(not a or len(a) != len(b) or a == b for a, b in zip(correct, corrupted)):
        raise ValueError("Corruptions must change each nonempty note at identical token length")
    gcontrols = matched_controls(generated, filler_ids=tok.encode(" ", add_special_tokens=False), seed=17)
    vcontrols = matched_controls(correct, filler_ids=tok.encode(" ", add_special_tokens=False), seed=17)
    data = []
    for i, q in enumerate(questions):
        contexts = make_contexts(prefixes[i], generated[i], correct[i], corrupted[i],
                    {k:v[i] for k,v in gcontrols.items()}, {k:v[i] for k,v in vcontrols.items()})
        data.append(dict(**q, contexts=contexts, generated_note=notes[ids[i]],
                         verified_note=verified[ids[i]], donor_item_id=ids[(i+1)%32]))
    max_context = max(len(v) for r in data for v in r["contexts"].values())
    width = ((max_context+127)//128)*128
    suffix = tok.encode("\n#### ", add_special_tokens=False)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/"inputs.json", dict(questions=data, numeric_suffix_ids=suffix))
    pins = [Path(teacher_dir)/"protocol.json", Path(teacher_dir)/"inputs.json", verified_file,
            prompt_file, Path(model_dir)/"ASSET_MANIFEST.json"]
    # Retain actual generation receipt paths supplied by the strict loader.
    pins += [Path(p) for p in receipts]
    protocol = dict(schema=SCHEMA, created_utc=utc_now(), source_files=sources(),
        input_hashes={str(p.resolve()):sha256_file(p) for p in pins}, inputs_sha256=sha256_file(output/"inputs.json"),
        teacher_protocol_sha256=sha256_file(Path(teacher_dir)/"protocol.json"),
        model_dir=str(Path(model_dir).resolve()), items=32, item_ids=ids, modes=list(MODES),
        shards=4, answer_cap=256, answer_prefill_width=width, compute="fp32", controls_seed=17,
        score_width=((max_context+len(suffix)+16+127)//128)*128, target_width=16,
        interface="raw=locked4shotQA; otherwise note+[271]+identical locked4shotQA",
        likelihood="Forced final numeric answer: context+'\\n#### '+canonical number; no latent CoT; full released head",
        correctness="One greedy calculation+answer from Answer:, cap256, existing strict hash4 parser",
        control_construction="Seed17 token shuffle; cyclic next-item donor repeated/trimmed to recipient length; empty donor uses space filler",
        gold_barrier="All notes/controls fixed and all4 answer shards complete before gold projection",
        bootstrap=dict(unit="question", repeats=5000, seed=17),
        limitations=["Curated QA feasibility, not FineMath transfer or self-discovered reasoning.",
            "Manual verified calculations use question only and omit final answer; external assistance.",
            "32 unfiltered TRAIN items; broad conditional intervals and ceiling effects expected.",
            "Released models may have encountered these public TRAIN problems before; this is not an uncontaminated capability benchmark.",
            "Keep capped/empty teacher notes and every answer failure in the denominator.",
            "Different note lengths and calculation quality; filler-only gains do not establish utility."])
    atomic_json(output/"protocol.json", protocol)
    return protocol


def load(output):
    p = read(output/"protocol.json")
    if p["schema"] != SCHEMA or p["source_files"] != sources() or p["modes"] != list(MODES):
        raise ValueError("Reader protocol/source changed")
    for file, expected in [(output/"inputs.json", p["inputs_sha256"]), *p["input_hashes"].items()]:
        if sha256_file(file) != expected:
            raise ValueError(f"Changed sealed diagnostic input: {file}")
    inputs=read(output/"inputs.json")
    if (p["items"]!=32 or len(inputs["questions"])!=32 or len(set(p["item_ids"]))!=32
        or [r["item_id"] for r in inputs["questions"]]!=p["item_ids"]
        or any(r["split"]!="train" or r["item_id"]!=r["document_id"]
               or not r["item_id"].startswith("gsm8k:train:") for r in inputs["questions"])):
        raise ValueError("Sealed questions must be all32 canonical TRAIN items")
    return p, inputs


@torch.no_grad()
def answers(output, shard, deadline):
    check_deadline(deadline); p, inputs = load(output)
    selected = inputs["questions"][shard::4]
    if shard not in range(4) or len(selected) != 8:
        raise ValueError("Exactly four strided shards of8")
    directory = output/"answers"/f"shard-{shard}"; directory.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", started_utc=utc_now(), protocol_sha256=sha256_file(output/"protocol.json"),
                  shard=shard, deadline_utc=deadline, rows=0)
    atomic_json(directory/"run.json", report)
    try:
        backend, tok, _, signature, _ = setup_model(SimpleNamespace(model_dir=Path(p["model_dir"]), checkpoint=None, device="xla"))
        decoder = GreedyDecoder(backend.model, tok, sync=backend.sync, banned_ids=tuple(backend.banned_ids))
        report["model_signature"] = signature
        with (directory/"predictions.jsonl").open("x") as stream:
            for q in selected:
                for mode in MODES:
                    check_deadline(deadline); start=time.monotonic()
                    row = dict(item_id=q["item_id"], document_id=q["document_id"], split="train", mode=mode,
                               status="error", generated_text="", n_generated_tokens=0,
                               context_sha256=canonical_hash(q["contexts"][mode]))
                    try:
                        result = decoder.decode(q["contexts"][mode], max_new_tokens=256, prefill_width=p["answer_prefill_width"],
                            pad_id=151643, stop_ids=(151643,), answer_line_stop=True)
                        status = "truncated" if result.stop_type == "length_cap" else "ok"
                        if status=="ok" and numeric_answer(result.text) is None: status="format_error"
                        row.update(status=status, generated_text=result.text, raw_generated_text=result.raw_text,
                                   actions=result.actions, stop_type=result.stop_type, n_generated_tokens=len(result.actions))
                    except Exception as exc:
                        row["error"]=f"{type(exc).__name__}: {exc}"
                        if isinstance(exc,GenerationFailure):
                            row.update(generated_text=exc.partial.text, actions=exc.partial.actions,
                                       n_generated_tokens=len(exc.partial.actions))
                    row["wall_seconds"]=time.monotonic()-start
                    stream.write(json.dumps(row,allow_nan=False)+"\n"); stream.flush(); report["rows"]+=1
                print(json.dumps(dict(shard=shard, rows=report["rows"], utc=utc_now())),flush=True)
        report.update(status="complete", predictions_sha256=sha256_file(directory/"predictions.jsonl"))
    except BaseException as exc:
        report.update(status="failed",error=f"{type(exc).__name__}: {exc}"); raise
    finally:
        report["finished_utc"]=utc_now(); atomic_json(directory/"run.json",report,overwrite=True)


def closed_answers(output, p):
    result=[]; signatures=[]
    for shard in range(4):
        d=output/"answers"/f"shard-{shard}"; r=read(d/"run.json"); values=rows(d/"predictions.jsonl")
        expected={(k,m) for k in p["item_ids"][shard::4] for m in MODES}
        if (r["status"]!="complete" or r["rows"]!=80 or len(values)!=80
            or r["protocol_sha256"]!=sha256_file(output/"protocol.json")
            or r["predictions_sha256"]!=sha256_file(d/"predictions.jsonl")
            or {(v["item_id"],v["mode"]) for v in values}!=expected):
            raise ValueError("All4 whole answer shards must close before gold access")
        result.extend(values); signatures.append(r["model_signature"])
    if len({canonical_hash(s) for s in signatures})!=1: raise ValueError("Reader model identities differ")
    return result


def project_gold(output, source):
    p, _=load(output); closed_answers(output,p)
    if sha256_file(source)!=TRAIN_SHA: raise ValueError("Use pinned TRAIN source only")
    import pyarrow.parquet as pq
    answers = pq.read_table(source,columns=["answer"]).column("answer")
    gold=[]
    for k in p["item_ids"]:
        if not k.startswith("gsm8k:train:"): raise ValueError("TRAIN only")
        answer=answers[int(k.rsplit(":",1)[1])].as_py(); number=numeric_answer(answer)
        if number is None: raise ValueError("Invalid canonical TRAIN answer")
        gold.append(dict(item_id=k, document_id=k, split="train", answer=answer, number=number))
    atomic_json(output/"gold.json",dict(source_sha256=TRAIN_SHA, protocol_sha256=sha256_file(output/"protocol.json"),
                                      created_utc=utc_now(),rows=gold))


@torch.no_grad()
def likelihood(output, shard, deadline):
    check_deadline(deadline); p, inputs=load(output); closed_answers(output,p)
    g=read(output/"gold.json")
    if g["protocol_sha256"]!=sha256_file(output/"protocol.json"): raise ValueError("Gold/protocol differs")
    gold={v["item_id"]:v["number"] for v in g["rows"]}
    if set(gold)!=set(p["item_ids"]): raise ValueError("Gold item set differs")
    selected=inputs["questions"][shard::4]
    if shard not in range(4) or len(selected)!=8: raise ValueError("Expected four shards")
    d=output/"likelihood"/f"shard-{shard}"; d.mkdir(parents=True,exist_ok=False)
    r=dict(status="running",started_utc=utc_now(),rows=0,shard=shard,
           protocol_sha256=sha256_file(output/"protocol.json"),gold_sha256=sha256_file(output/"gold.json"))
    atomic_json(d/"run.json",r)
    try:
        backend,tok,_,signature,_=setup_model(SimpleNamespace(model_dir=Path(p["model_dir"]),checkpoint=None,device="xla"))
        r["model_signature"]=signature
        with (d/"scores.jsonl").open("x") as stream:
            for q in selected:
                target=tok.encode(gold[q["item_id"]],add_special_tokens=False)
                for mode in MODES:
                    check_deadline(deadline); start=time.monotonic()
                    value=score_continuations(backend,q["contexts"][mode]+inputs["numeric_suffix_ids"],[target],
                        total_width=p["score_width"],future_width=p["target_width"],pad_id=151643)[0]
                    row=dict(item_id=q["item_id"],document_id=q["document_id"],mode=mode,status="ok",
                             **asdict(value),wall_seconds=time.monotonic()-start)
                    stream.write(json.dumps(row,allow_nan=False)+"\n"); stream.flush(); r["rows"]+=1
        r.update(status="complete",scores_sha256=sha256_file(d/"scores.jsonl"))
    except BaseException as exc:
        r.update(status="failed",error=f"{type(exc).__name__}: {exc}"); raise
    finally:
        r["finished_utc"]=utc_now(); atomic_json(d/"run.json",r,overwrite=True)


def summarize(output):
    p,_=load(output); predictions=closed_answers(output,p); g=read(output/"gold.json")
    scores=[]
    for shard in range(4):
        d=output/"likelihood"/f"shard-{shard}"; r=read(d/"run.json"); values=rows(d/"scores.jsonl")
        expected={(k,m) for k in p["item_ids"][shard::4] for m in MODES}
        if (r["status"]!="complete" or len(values)!=80 or r["rows"]!=80
            or r["scores_sha256"]!=sha256_file(d/"scores.jsonl")
            or r["protocol_sha256"]!=sha256_file(output/"protocol.json")
            or r["gold_sha256"]!=sha256_file(output/"gold.json")
            or r["model_signature"]!=read(output/"answers"/f"shard-{shard}"/"run.json")["model_signature"]
            or {(v["item_id"],v["mode"]) for v in values}!=expected):
            raise ValueError("All likelihood shards must close without dropping rows")
        scores.extend(values)
    accuracy={}; accuracy_rows={}; ll_rows={}
    for mode in MODES:
        accuracy_rows[mode],accuracy[mode]=score_predictions(g["rows"],[r for r in predictions if r["mode"]==mode])
        values=[r for r in scores if r["mode"]==mode]
        if len(values)!=32 or {r["item_id"] for r in values}!=set(p["item_ids"]): raise ValueError("Incomplete LL condition")
        if any(r["status"]!="ok" or not r["target_token_ids"]
               or not all(math.isfinite(v) for v in r["token_log_likelihoods"])
               or len(r["token_log_likelihoods"])!=len(r["target_token_ids"])
               or not math.isclose(sum(r["token_log_likelihoods"]),r["log_likelihood"],rel_tol=0,abs_tol=1e-6)
               for r in values): raise ValueError("Invalid numeric LL; never drop failed items")
        ll_rows[mode]=[dict(item_id=r["item_id"],document_id=r["document_id"],score=r["log_likelihood"]) for r in values]
    contrasts={f"{b}_minus_{a}":dict(accuracy=paired_cluster_bootstrap(accuracy_rows[a],accuracy_rows[b]),
        numeric_ll=paired_cluster_bootstrap(ll_rows[a],ll_rows[b])) for a,b in CONTRASTS}
    result=dict(schema=SCHEMA,status="complete",protocol_sha256=sha256_file(output/"protocol.json"),
        items=32,accuracy=accuracy,mean_numeric_ll={m:sum(r["score"] for r in ll_rows[m])/32 for m in MODES},
        contrasts=contrasts,limitations=p["limitations"],score_rows=scores,accuracy_rows=accuracy_rows,
        answer_statuses=dict(Counter(r["status"] for r in predictions)),created_utc=utc_now())
    atomic_json(output/"summary.json",result); return result


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("command",choices=("prepare","answers","project-gold","likelihood","summarize"))
    p.add_argument("--output",type=Path,required=True); p.add_argument("--teacher",type=Path)
    p.add_argument("--verified",type=Path); p.add_argument("--prompt",type=Path,default=Path("local/evaluation/held-dev128-v1/prompt.json"))
    p.add_argument("--model-dir",type=Path,default=Path("local/assets/qwen3-1.7b-base")); p.add_argument("--source",type=Path)
    p.add_argument("--shard-index",type=int); p.add_argument("--deadline-utc")
    a=p.parse_args()
    if a.command=="prepare": prepare(a.output,a.teacher,a.verified,a.prompt,a.model_dir)
    elif a.command=="answers": answers(a.output,a.shard_index,a.deadline_utc)
    elif a.command=="project-gold": project_gold(a.output,a.source)
    elif a.command=="likelihood": likelihood(a.output,a.shard_index,a.deadline_utc)
    else: summarize(a.output)


if __name__=="__main__": main()
