"""Exploratory unchanged-B0 short-note test, averaging every generated path.

The writer uses a separate calculation prompt. Each note is then consumed by the
locked QA reader. Four sampled notes are observations within a question, never
four independent questions or an answer-selected deployment policy.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time
from types import SimpleNamespace

import torch

from train.evaluate import GreedyDecoder, GenerationFailure, question_prompt, validate_prompt_spec
from train.evaluate_likelihood import score_continuations
from train.evaluate_local import setup_model
from train.evaluation_integrity import evaluation_dependencies
from weekend_thinking.artifacts import atomic_json, canonical_hash, require_utc_deadline, sha256_file, utc_now
from weekend_thinking.evaluation import numeric_answer, score_predictions
from weekend_thinking.stats import paired_cluster_bootstrap

SCHEMA = "gsm8k-self-note-reader-train32-g4-v1"
MODES = ("own", "shuffled", "other", "empty192", "direct256")
CONTRASTS = (("direct256", "own"), ("empty192", "own"),
             ("other", "own"), ("shuffled", "own"), ("direct256", "empty192"))


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def sources():
    root = Path(__file__).resolve().parents[1]
    return {**evaluation_dependencies(), **{name: sha256_file(root/name) for name in
        ("train/self_note_pilot.py", "train/calculation_self_notes.py", "train/evaluate_local.py",
         "train/evaluate_likelihood.py")}}


def check_deadline(deadline):
    if datetime.now(timezone.utc) >= require_utc_deadline(deadline):
        raise RuntimeError("Finite self-note diagnostic deadline reached; no automatic retry")


def controls(note, donor, question_id, sample):
    """Question-only deterministic controls; unrelated donor is same sample index."""
    seed = int.from_bytes(hashlib.sha256(f"{question_id}:{sample}:17".encode()).digest()[:8], "big")
    shuffled = list(note); random.Random(seed).shuffle(shuffled)
    # Empty notes remain empty in every content control. An empty donor uses
    # the documented space-token fallback, not a search for a better donor.
    donor = list(donor) or [220]
    other = (donor*((len(note)+len(donor)-1)//len(donor)))[:len(note)]
    return dict(own=list(note), shuffled=shuffled, other=other)


def build_cells(questions, notes, prefixes):
    cells = []
    for index, q in enumerate(questions):
        key=q["item_id"]; prefix=prefixes[key]
        donor=questions[(index+1)%len(questions)]["item_id"]
        # The exact direct256 baseline already closed in the preceding pilot.
        # Reuse those sealed paths and numeric scores only during summarization.
        for mode, cap, context in (("empty192",192,prefix+[198,198]),):
            cells.append(dict(item_id=f"{key}::{mode}", document_id=key, question_id=key,
                mode=mode, sample_index=None, answer_cap=cap, context_ids=list(context)))
        for sample in range(4):
            note=notes[(key,sample)]["content_ids"]
            other=notes[(donor,sample)]["content_ids"]
            for mode, content in controls(note,other,key,sample).items():
                cells.append(dict(item_id=f"{key}::sample-{sample}::{mode}", document_id=key,
                    question_id=key, sample_index=sample, mode=mode, answer_cap=192,
                    context_ids=list(prefix)+[198]+content+[198], donor_question_id=donor))
    return cells


def prepare(output, writer, prompt, model_dir, previous_pilot):
    from transformers import AutoTokenizer
    from train.calculation_self_notes import load_complete
    wp, inputs, notes, receipts = load_complete(writer)
    questions=inputs["rows"]; ids=[q["item_id"] for q in questions]
    if len(ids)!=32 or len(set(ids))!=32 or set(notes)!={(k,s) for k in ids for s in range(4)}:
        raise ValueError("All32 questions and128 unselected notes are required")
    from train.calculation_pilot import load as previous_load
    pp,previous_inputs=previous_load(previous_pilot)
    if pp["item_ids"]!=ids or pp["model_dir"]!=str(model_dir.resolve()):
        raise ValueError("Exploratory question reuse and unchanged reader identity must match")
    spec=read(prompt); validate_prompt_spec(spec,set(ids))
    tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
    if tok.encode("\n",add_special_tokens=False)!=[198] or tok.encode(" ",add_special_tokens=False)!=[220]:
        raise ValueError("Released separator or filler token changed")
    prefixes={q["item_id"]:tok.encode(question_prompt(q["question"],spec),add_special_tokens=False) for q in questions}
    previous_contexts={q["item_id"]:q["contexts"]["raw"] for q in previous_inputs["questions"]}
    if prefixes!=previous_contexts:raise ValueError("Reused direct256 baseline contexts must be byte-identical")
    cells=build_cells(questions,notes,prefixes)
    width=((max(len(c["context_ids"]) for c in cells)+127)//128)*128
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/"inputs.json",dict(questions=questions,cells=cells,
        numeric_suffix_ids=tok.encode("\n#### ",add_special_tokens=False)))
    pins=[Path(writer)/"protocol.json",Path(writer)/"inputs.json",prompt,
          model_dir/"ASSET_MANIFEST.json",previous_pilot/"protocol.json",previous_pilot/"inputs.json",
          previous_pilot/"summary.json",*map(Path,receipts)]
    for phase,file in (("answers","predictions.jsonl"),("likelihood","scores.jsonl")):
        for shard in range(4):pins.extend([previous_pilot/phase/f"shard-{shard}"/file,previous_pilot/phase/f"shard-{shard}"/"run.json"])
    protocol=dict(schema=SCHEMA,created_utc=utc_now(),source_files=sources(),
        inputs_sha256=sha256_file(output/"inputs.json"),
        input_hashes={str(p.resolve()):sha256_file(p) for p in pins},
        writer=str(writer.resolve()),writer_protocol_sha256=sha256_file(Path(writer)/"protocol.json"),
        model_dir=str(model_dir.resolve()),items=32,notes=128,cells=416,reused_direct_cells=32,
        previous_pilot=str(previous_pilot.resolve()),shards=4,item_ids=ids,modes=list(MODES),
        answer_prefill_width=width,score_width=((width+32+127)//128)*128,target_width=16,
        compute="fp32",writer_cap=64,reader_cap=192,direct_cap=256,controls_seed=17,
        primary="Monte Carlo mean over4 notes estimates single sampled-note policy accuracy minus direct256; no selected path or ensemble",
        interface="Locked4shotQA+[198]+note+[198]; empty192 preserves both separators; direct256 is unwrapped QA",
        gold_source=str((previous_pilot/"gold.json").resolve()),
        gold_source_sha256=read(previous_pilot/"likelihood/shard-0/run.json")["gold_sha256"],
        gold_access="Existing TRAIN32 answers are known from the preceding pilot. No gold in writer/reader generation; parse only after all answer receipts close.",
        bootstrap=dict(unit="question_after_averaging_all4_samples",repeats=5000,seed=17),
        limitations=["Exploratory reuse of32 TRAIN questions after the previous calculation pilot, not blinded confirmation or official TEST.",
            "Unchanged released B0; public TRAIN questions may be present in pretraining.",
            "64 writer plus192 reader is an equal maximum generated-token budget to direct256, not equal measured FLOPs or latency.",
            "empty192 is the generate-then-discard-note control: generation cost is charged although one identical greedy reader path per question is deduplicated.",
            "No oracle selection, voting or best-of4: G4 is a Monte Carlo estimate of a single-note policy; intervals are conditional on these fixed sampled notes.",
            "direct256 reuses the exact32 closed B0 raw paths/scores from the preceding pilot; no repeated baseline decoding.",
            "Separate few-shot writer prompt changes context; final-answer/empty/capped notes remain and need content inspection.",
            "Cyclic unrelated donors repeated/trimmed to recipient length and token shuffling are imperfect linguistic controls.",
            "Forced canonical numeric likelihood measures a different task from free-running calculation and answer accuracy."])
    atomic_json(output/"protocol.json",protocol);return protocol


def load(output):
    p=read(output/"protocol.json")
    if p["schema"]!=SCHEMA or p["source_files"]!=sources() or p["modes"]!=list(MODES):
        raise ValueError("Self-note reader source/protocol changed")
    for f,h in [(output/"inputs.json",p["inputs_sha256"]),*p["input_hashes"].items()]:
        if sha256_file(f)!=h:raise ValueError(f"Sealed self-note reader input changed: {f}")
    x=read(output/"inputs.json");cells=x["cells"];ids=p["item_ids"]
    if (len(ids)!=32 or len(set(ids))!=32 or [q["item_id"] for q in x["questions"]]!=ids
        or any(q["split"]!="train" or q["document_id"]!=q["item_id"] for q in x["questions"])
        or len(cells)!=416 or len({c["item_id"] for c in cells})!=416):
        raise ValueError("Keep exact32 TRAIN questions and416 unique fresh cells")
    expected={(k,m,s) for k in ids for m in MODES[:-1] for s in (range(4) if m in MODES[:3] else [None])}
    if {(c["question_id"],c["mode"],c["sample_index"]) for c in cells}!=expected:
        raise ValueError("Every question needs every baseline and all4 paths/controls")
    if any(c["document_id"]!=c["question_id"] or c["answer_cap"]!=(256 if c["mode"]=="direct256" else 192)
           or not c["context_ids"] or len(c["context_ids"])>p["answer_prefill_width"] for c in cells):
        raise ValueError("Changed causal context, cluster identity or answer cap")
    return p,x


def selected_cells(p,x,shard):
    if shard not in range(4):raise ValueError("Four shards only")
    ids=set(p["item_ids"][shard::4])
    return [c for c in x["cells"] if c["question_id"] in ids]


def completed_rows(output,p,x,phase):
    result=[];signatures=[]
    filename="predictions.jsonl" if phase=="answers" else "scores.jsonl"
    for shard in range(4):
        d=output/phase/f"shard-{shard}";r=read(d/"run.json");values=rows(d/filename)
        expected={c["item_id"]:c for c in selected_cells(p,x,shard)}
        if (r["status"]!="complete" or r["rows"]!=104 or len(values)!=104
            or r["protocol_sha256"]!=sha256_file(output/"protocol.json")
            or r["rows_sha256"]!=sha256_file(d/filename)
            or {v["item_id"] for v in values}!=set(expected)):
            raise ValueError("All4 full104-cell shards must close; no dropped or duplicate rows")
        for v in values:
            c=expected[v["item_id"]]
            if any(v[k]!=c[k] for k in ("question_id","document_id","sample_index","mode")) or v["context_sha256"]!=canonical_hash(c["context_ids"]):
                raise ValueError("Sealed context or pairing changed")
        signatures.append(r["model_signature"]);result.extend(values)
    if len({canonical_hash(s) for s in signatures})!=1:raise ValueError("Reader model identities differ")
    return result,signatures[0]


@torch.no_grad()
def run(output,shard,deadline,phase):
    check_deadline(deadline);p,x=load(output);gold=None
    if phase=="likelihood":
        _,answer_signature=completed_rows(output,p,x,"answers")
        g=read(output/"gold.json");gold={r["item_id"]:r["number"] for r in g["rows"]}
        if set(gold)!=set(p["item_ids"]) or g["protocol_sha256"]!=sha256_file(output/"protocol.json"):
            raise ValueError("Late gold coverage/protocol mismatch")
    cells=selected_cells(p,x,shard);d=output/phase/f"shard-{shard}";d.mkdir(parents=True,exist_ok=False)
    filename="predictions.jsonl" if phase=="answers" else "scores.jsonl"
    r=dict(status="running",started_utc=utc_now(),rows=0,shard=shard,deadline_utc=deadline,
        protocol_sha256=sha256_file(output/"protocol.json"))
    if gold is not None:r["gold_sha256"]=sha256_file(output/"gold.json")
    atomic_json(d/"run.json",r)
    try:
        backend,tok,_,signature,_=setup_model(SimpleNamespace(model_dir=Path(p["model_dir"]),checkpoint=None,device="xla"))
        r["model_signature"]=signature
        if gold is not None and signature!=answer_signature:raise ValueError("Answer/LL models differ")
        decoder=GreedyDecoder(backend.model,tok,sync=backend.sync,banned_ids=tuple(backend.banned_ids))
        with (d/filename).open("x") as stream:
            for c in cells:
                check_deadline(deadline);start=time.monotonic()
                v={k:c[k] for k in ("item_id","document_id","question_id","sample_index","mode")}
                v["context_sha256"]=canonical_hash(c["context_ids"])
                if gold is None:
                    v.update(status="error",generated_text="",n_generated_tokens=0,answer_cap=c["answer_cap"])
                    try:
                        a=decoder.decode(c["context_ids"],max_new_tokens=c["answer_cap"],
                            prefill_width=p["answer_prefill_width"],pad_id=151643,stop_ids=(151643,),answer_line_stop=True)
                        status="truncated" if a.stop_type=="length_cap" else "ok"
                        if status=="ok" and numeric_answer(a.text) is None:status="format_error"
                        v.update(status=status,generated_text=a.text,raw_generated_text=a.raw_text,
                            actions=a.actions,stop_type=a.stop_type,n_generated_tokens=len(a.actions))
                    except Exception as exc:
                        v["error"]=f"{type(exc).__name__}: {exc}"
                        if isinstance(exc,GenerationFailure):v.update(generated_text=exc.partial.text,
                            actions=exc.partial.actions,n_generated_tokens=len(exc.partial.actions))
                else:
                    target=tok.encode(gold[c["question_id"]],add_special_tokens=False)
                    score=score_continuations(backend,c["context_ids"]+x["numeric_suffix_ids"],[target],
                        total_width=p["score_width"],future_width=p["target_width"],pad_id=151643)[0]
                    v.update(status="ok",**asdict(score))
                v["wall_seconds"]=time.monotonic()-start
                stream.write(json.dumps(v,allow_nan=False)+"\n");stream.flush();r["rows"]+=1
                if r["rows"]%13==0:print(json.dumps(dict(phase=phase,shard=shard,rows=r["rows"],utc=utc_now())),flush=True)
        r.update(status="complete",rows_sha256=sha256_file(d/filename))
    except BaseException as exc:r.update(status="failed",error=f"{type(exc).__name__}: {exc}");raise
    finally:r["finished_utc"]=utc_now();atomic_json(d/"run.json",r,overwrite=True)


def project_gold(output):
    p,x=load(output);completed_rows(output,p,x,"answers")
    previous=Path(p["previous_pilot"])
    expected={read(previous/"likelihood"/f"shard-{s}"/"run.json")["gold_sha256"] for s in range(4)}
    if expected!={p["gold_source_sha256"]} or sha256_file(p["gold_source"])!=p["gold_source_sha256"]:
        raise ValueError("Previous gold must match all4 pinned prior likelihood receipts")
    g=read(p["gold_source"])
    if {v["item_id"] for v in g["rows"]}!=set(p["item_ids"]):raise ValueError("Previous TRAIN32 gold differs")
    atomic_json(output/"gold.json",dict(rows=g["rows"],protocol_sha256=sha256_file(output/"protocol.json"),
        source_sha256=sha256_file(p["gold_source"]),source=p["gold_source"],created_utc=utc_now()))


def aggregate_questions(values,ids,field):
    groups=defaultdict(list)
    for v in values:groups[(v["mode"],v["question_id"])].append(float(v[field]))
    result={}
    for mode in MODES:
        count=4 if mode in MODES[:3] else 1;result[mode]=[]
        for key in ids:
            scores=groups[(mode,key)]
            if len(scores)!=count or not all(math.isfinite(v) for v in scores):
                raise ValueError("Require all4 finite paths or the single baseline for every question")
            result[mode].append(dict(item_id=key,document_id=key,score=sum(scores)/count))
    return result


def reused_direct(output,p,signature):
    """Use complete pinned prior paths; this function is called after generation."""
    from train.calculation_pilot import load as previous_load, closed_answers
    previous=Path(p["previous_pilot"]);pp,_=previous_load(previous)
    predictions=[v for v in closed_answers(previous,pp) if v["mode"]=="raw"]
    summary=read(previous/"summary.json")
    scores=[v for v in summary["score_rows"] if v["mode"]=="raw"]
    if (summary["status"]!="complete" or len(predictions)!=32 or len(scores)!=32
        or {v["item_id"] for v in predictions}!=set(p["item_ids"])
        or {v["item_id"] for v in scores}!=set(p["item_ids"])
        or summary["protocol_sha256"]!=sha256_file(previous/"protocol.json")):
        raise ValueError("Complete pinned raw256 baseline required")
    for shard in range(4):
        for phase in ("answers","likelihood"):
            if read(previous/phase/f"shard-{shard}"/"run.json")["model_signature"]!=signature:
                raise ValueError("Reused baseline must be the identical B0 model")
    def renamed(v):
        return dict(v,item_id=f"{v['item_id']}::direct256",question_id=v["item_id"],
            document_id=v["item_id"],sample_index=None,mode="direct256",reused=True,
            reuse_source=str(previous),reuse_protocol_sha256=sha256_file(previous/"protocol.json"))
    return list(map(renamed,predictions)),list(map(renamed,scores))


def summarize(output):
    p,x=load(output);pred,signature=completed_rows(output,p,x,"answers");ll,ll_signature=completed_rows(output,p,x,"likelihood")
    if signature!=ll_signature:raise ValueError("Answer and likelihood model signatures differ")
    baseline_predictions,baseline_scores=reused_direct(output,p,signature)
    pred+=baseline_predictions;ll+=baseline_scores
    g=read(output/"gold.json");gold={v["item_id"]:v for v in g["rows"]}
    if (g["source_sha256"]!=p["gold_source_sha256"] or g["protocol_sha256"]!=sha256_file(output/"protocol.json")
        or set(gold)!=set(p["item_ids"])):raise ValueError("Late gold identity or coverage differs")
    for shard in range(4):
        if read(output/"likelihood"/f"shard-{shard}"/"run.json")["gold_sha256"]!=sha256_file(output/"gold.json"):
            raise ValueError("LL did not score the sealed late gold")
    scored,_=score_predictions([dict(gold[v["question_id"]],item_id=v["item_id"],document_id=v["question_id"]) for v in pred],pred)
    meta={v["item_id"]:v for v in pred}
    acc=[dict(r,question_id=meta[r["item_id"]]["question_id"],mode=meta[r["item_id"]]["mode"],
              sample_index=meta[r["item_id"]]["sample_index"]) for r in scored]
    for v in ll:
        if (v["status"]!="ok" or not v["target_token_ids"]
            or len(v["target_token_ids"])!=len(v["token_log_likelihoods"])
            or not all(math.isfinite(t) for t in v["token_log_likelihoods"])
            or not math.isclose(sum(v["token_log_likelihoods"]),v["log_likelihood"],rel_tol=0,abs_tol=1e-6)):
            raise ValueError("Every numeric LL cell must be finite and correctly summed")
    a=aggregate_questions(acc,p["item_ids"],"score");l=aggregate_questions(ll,p["item_ids"],"log_likelihood")
    contrasts={f"{b}_minus_{c}":dict(accuracy=paired_cluster_bootstrap(a[c],a[b]),numeric_ll=paired_cluster_bootstrap(l[c],l[b])) for c,b in CONTRASTS}
    result=dict(schema=SCHEMA,status="complete",created_utc=utc_now(),protocol_sha256=sha256_file(output/"protocol.json"),
        questions=32,notes=128,answer_cells=448,likelihood_cells=448,fresh_cells_per_phase=416,
        reused_direct_cells_per_phase=32,primary="own_minus_direct256",
        mean_accuracy={m:sum(v["score"] for v in a[m])/32 for m in MODES},
        mean_numeric_ll={m:sum(v["score"] for v in l[m])/32 for m in MODES},contrasts=contrasts,
        statuses={m:dict(Counter(v["status"] for v in acc if v["mode"]==m)) for m in MODES},
        question_accuracy=a,question_numeric_ll=l,accuracy_cells=acc,score_cells=ll,
        costs=dict(reader_tokens={m:sum(v["n_generated_tokens"] for v in pred if v["mode"]==m) for m in MODES},
            reader_wall_seconds={m:sum(v["wall_seconds"] for v in pred if v["mode"]==m) for m in MODES},
            qualification="Actual writer costs are in the separately pinned128-note bundle. empty192 deduplicates four identical discarded-note reader paths. direct256 tokens/wall are historical reused baseline costs, not new work. G4 wall divided by4 is not measured G1 latency."),
        limitations=p["limitations"])
    atomic_json(output/"summary.json",result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("prepare","answers","project-gold","likelihood","summarize"))
    p.add_argument("--output",type=Path,required=True);p.add_argument("--writer",type=Path)
    p.add_argument("--prompt",type=Path,default=Path("local/evaluation/held-dev128-v1/prompt.json"))
    p.add_argument("--model-dir",type=Path,default=Path("local/assets/qwen3-1.7b-base"))
    p.add_argument("--previous-pilot",type=Path,default=Path("local/nvme/diagnostics/gsm8k-calculation-reader-train32-v1"))
    p.add_argument("--shard-index",type=int);p.add_argument("--deadline-utc")
    a=p.parse_args()
    if a.command=="prepare":result=prepare(a.output,a.writer,a.prompt,a.model_dir,a.previous_pilot)
    elif a.command in ("answers","likelihood"):result=run(a.output,a.shard_index,a.deadline_utc,a.command)
    elif a.command=="project-gold":result=project_gold(a.output)
    else:result=summarize(a.output)
    if result is not None:print(json.dumps({k:v for k,v in result.items() if k not in ("accuracy_cells","score_cells")},allow_nan=False))


if __name__=="__main__":main()
