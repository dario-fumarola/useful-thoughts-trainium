"""Paired fixed-B0-reader comparisons of three closed fresh-G1 DEV64 evaluations.

Inputs are explicit directories (or summary.json paths), never a checkpoint
search. This command loads no model or target records and writes only its output.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import math
from pathlib import Path
import statistics

from train import warmstart_prepend_evaluate as evaluate
from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file, utc_now
from weekend_thinking.stats import paired_cluster_bootstrap

SCHEMA = "warmstart-fixed-reader-comparison-v1"
BASELINES = ("empty", "raw", "filler", "shuffled", "other")
POLICY_KEYS = ("selection", "selection_sha256", "model_dir", "model_manifest_sha256", "format_sha256",
    "cap", "temperature", "seed", "batch_size", "groups_per_prefix", "shards", "shard_size",
    "batch_seed_rule", "controls_seed", "filler_text", "first_batch_replay_atol", "source_sha256")
READER_KEYS = ("baseline", "baseline_reader", "selection", "model_dir", "model_manifest_sha256",
    "interface", "serialization", "separator_ids", "modes", "width", "horizons", "compute", "shards",
    "shard_size", "expected_shard_rows", "expected_rows", "raw_reference_atol", "raw_reference_rule", "primary", "bootstrap", "source_files")


def sources():
    return {**evaluate.sources(), "train/warmstart_compare.py":sha256_file(__file__)}


def document_rows(p, values, notes):
    ids = p["selection"]["item_ids"]; documents = dict(zip(ids,p["selection"]["document_ids"]))
    keys = [(r["item_id"],r["mode"]) for r in values]
    if (len(keys) != 384 or len(set(keys)) != 384 or set(keys) != {(k,m) for k in ids for m in evaluate.MODES}
            or any(r["status"] != "ok" or r["document_id"] != documents[r["item_id"]]
                   or not evaluate.finite_vector(r.get("token_log_likelihoods")) for r in values)):
        raise ValueError("Require every finite DEV64/context exactly once; no dropped failures")
    index = {(r["item_id"],r["mode"]):r for r in values}
    return [dict(item_id=k,document_id=documents[k],quartile=p["selection"]["strata"][k],
        n_actions=notes[k]["n_actions"],n_content_tokens=notes[k]["n_content_tokens"],stop_type=notes[k]["stop_type"],
        absolute_log_likelihoods={f"K{h}":{m:sum(index[k,m]["token_log_likelihoods"][:h]) for m in evaluate.MODES} for h in (1,16)}) for k in ids]


def note_context(notes):
    lengths = [n["n_content_tokens"] for n in notes.values()]
    return dict(empty=sum(n==0 for n in lengths),caps=sum(n["stop_type"]=="length_cap" for n in notes.values()),
        content_length_counts=dict(Counter(str(n) for n in lengths)),actions_total=sum(n["n_actions"] for n in notes.values()),
        content_tokens_total=sum(lengths),content_tokens_mean=statistics.mean(lengths),content_tokens_median=statistics.median(lengths),
        stop_counts=dict(Counter(n["stop_type"] for n in notes.values())))


def cost_context(generation, runs, values):
    return dict(new_sampling_seconds=0,reused_sampling_batches=16,
        reused_sampling_batch_seconds_counted_once=sum(r["batch_sampling_wall_seconds_total"] for r in generation),
        reused_replay_audit_batch_seconds_counted_once=sum(r["batch_replay_audit_wall_seconds_total"] for r in generation),
        new_score_context_seconds=sum(r["wall_seconds"] for r in values),
        new_score_process_seconds=sum((datetime.fromisoformat(r["finished_utc"])-datetime.fromisoformat(r["started_utc"])).total_seconds() for r in runs),
        stage_span_seconds=(max(datetime.fromisoformat(r["finished_utc"]) for r in runs)-min(datetime.fromisoformat(r["started_utc"]) for r in runs)).total_seconds(),
        qualification="Batch generation counted once; isolated reader/parallel process walls are not training throughput")


def load_complete(path):
    """No-write validator: reuse closed-note helpers, independently verify all score rows."""
    directory = Path(path)
    if directory.name == "summary.json": directory = directory.parent
    p = evaluate.read_protocol(directory); summary = evaluate.read(directory/"summary.json")
    if (p["reader"] != evaluate.original.model_identity("B0") or summary["fixed_B0_reader_primary"] is not True
            or summary["status"] != "complete" or summary["failures"] != 0 or summary["documents"] != 64
            or summary["rows"] != 384 or summary["schema"] != evaluate.SCHEMA
            or summary["experiment_id"] != p["experiment_id"]
            or summary["protocol_sha256"] != sha256_file(directory/"protocol.json")
            or p["interface"] != "prepend_plain" or p["horizons"] != [1,16] or p["shards"] != 4
            or p["shard_size"] != 16 or p["expected_rows"] != 384 or p["expected_shard_rows"] != 96
            or p["raw_reference_atol"] != .002 or p["bootstrap"] != dict(unit="document",repeats=5000,seed=17)
            or p["primary"] != "Fixed B0 reader, K16 fresh G1 own-minus-empty/filler/shuffled/other; K1 secondary"
            or summary["primary"] != p["primary"]):
        raise ValueError("Require a complete fixed-B0-reader K16-primary DEV64 evaluation")
    old,_,notes,writer,controls,generation,bundle_paths = evaluate.closed_bundle(Path(p["bundle"]))
    if (old["seed"],old["controls_seed"],old["first_batch_replay_atol"],old["shards"],old["shard_size"]) != (1729,17,.001,4,16):
        raise ValueError("Original note/control seed, replay or shard policy changed")
    reference, raw, identity, reference_paths = evaluate.raw_baseline(Path(p["baseline"]))
    if (old["selection"] != p["selection"] or any(old[k] != reference[k] for k in ("selection","model_dir","model_manifest_sha256","format_sha256"))
            or writer != p["writer"] or summary["writer"] != writer or summary["reader"] != identity
            or p["baseline_reader"] != identity or p["model_manifest_sha256"] != old["model_manifest_sha256"]):
        raise ValueError("Writer/panel/base/reference identity mismatch")
    expected_writer = evaluate.original.model_identity(writer["id"],writer["checkpoint"],writer["checkpoint_sha256"])
    evaluate.check_identity(writer,expected_writer,p["model_manifest_sha256"])
    evaluate.check_identity(identity,evaluate.original.model_identity("B0"),p["model_manifest_sha256"])
    baseline_pins = {str(file.resolve()):sha256_file(file) for file in reference_paths}
    required_pins = {str(file.resolve()):sha256_file(file) for file in bundle_paths+reference_paths}
    if any(p["input_hashes"].get(file) != digest for file,digest in required_pins.items()):
        raise ValueError("Writer bundle/raw-reference files are not bound in the evaluation input map")
    if [r["shard"] for r in summary["scoring"]] != list(range(4)):
        raise ValueError("Require all four unique ordered score receipts")
    values=[];runs=[];maximum=0.;artifacts={str((directory/name).resolve()):sha256_file(directory/name) for name in ("protocol.json","summary.json")}
    for shard,receipt in enumerate(summary["scoring"]):
        d=directory/"scoring"/f"shard-{shard}";run=evaluate.read(d/"run.json");file=d/"scores.jsonl"
        if (run["status"] != "complete" or run["schema"] != evaluate.SCHEMA or run["shard"] != shard
                or run["writer"] != writer or run["reader"] != identity or run["rows"] != 96 or run["expected_rows"] != 96
                or run["item_ids"] != evaluate.original.shard_ids(p,shard)
                or run["protocol_sha256"] != sha256_file(directory/"protocol.json")
                or run["scores_sha256"] != sha256_file(file) or receipt["scores_sha256"] != run["scores_sha256"]
                or receipt["run_sha256"] != sha256_file(d/"run.json") or run["raw_reference_rows"] != 16):
            raise ValueError("Incomplete or changed score shard/receipt/identity")
        local=evaluate.rows(file)
        if [(r["item_id"],r["mode"]) for r in local] != [(k,m) for k in run["item_ids"] for m in evaluate.MODES]:
            raise ValueError("Score shard coverage/order differs")
        shard_error=0.
        for row in local:
            mode=row["mode"];k=row["item_id"]
            if (row["status"] != "ok" or not evaluate.finite_vector(row.get("token_log_likelihoods"))
                    or row["split"] != "dev" or row["shard"] != shard
                    or row["writer_identity_sha256"] != canonical_hash(writer) or row["reader_identity_sha256"] != canonical_hash(identity)
                    or row["writer_checkpoint_sha256"] != writer["checkpoint_sha256"] or row["reader_checkpoint_sha256"] is not None
                    or row["interface"] != ("raw" if mode=="raw" else "prepend_plain")
                    or row["context_tokens"] != (512 if mode=="raw" else 513+len(controls[k]["content_ids"][mode]))
                    or type(row["wall_seconds"]) not in (int,float) or not math.isfinite(row["wall_seconds"]) or row["wall_seconds"]<0):
                raise ValueError("Failed row or mismatched score provenance")
            if mode=="raw":
                error=max(abs(a-b) for a,b in zip(row["token_log_likelihoods"],raw[k]));shard_error=max(shard_error,error)
                if error>.002 or row["raw_reference_max_token_error"] != error:
                    raise ValueError("Fixed B0 raw-reference parity mismatch")
        if run["raw_reference_max_token_error"] != shard_error:
            raise ValueError("Raw-reference maximum receipt differs")
        maximum=max(maximum,shard_error);values.extend(local);runs.append(run)
        artifacts.update({str((d/name).resolve()):sha256_file(d/name) for name in ("run.json","scores.jsonl")})
    per_document=document_rows(p,values,notes);lengths=note_context(notes);cost=cost_context(generation,runs,values)
    if (summary["per_document"] != per_document or summary["note_context"] != lengths or summary["cost"] != cost
            or summary["raw_reference_max_token_error"] != maximum):
        raise ValueError("Saved summary disagrees with original scores, lengths or costs")
    artifacts.update(p["input_hashes"])
    return dict(protocol=p,bundle_policy={k:old[k] for k in POLICY_KEYS},reader=identity,writer=writer,
                baseline_pins=baseline_pins,prefix_hashes={k:notes[k]["prefix_sha256"] for k in p["selection"]["item_ids"]},
                per_document=per_document,note_context=lengths,cost=cost,artifacts=artifacts)


def compare(arms):
    """Explicit writerBC versus B0/raw-control; no checkpoint selection or pooled samples."""
    b0=arms["B0"]
    if (b0["writer"]["id"] != "B0" or any(arms[a]["writer"]["id"]=="B0" for a in ("writerBC","raw_control"))
            or arms["writerBC"]["writer"]["id"] == arms["raw_control"]["writer"]["id"]
            or arms["writerBC"]["writer"]["checkpoint"] == arms["raw_control"]["writer"]["checkpoint"]
            or arms["writerBC"]["writer"]["checkpoint_sha256"] == arms["raw_control"]["writer"]["checkpoint_sha256"]):
        raise ValueError("Supply three distinct explicit B0/writerBC/raw-control identities")
    if any(arms[name]["writer"]["loaded_checkpoint"].get("step") != 64 for name in ("writerBC","raw_control")):
        raise ValueError("This comparison requires the predeclared step64 checkpoints; no checkpoint selection")
    for arm in arms.values():
        if (arm["bundle_policy"] != b0["bundle_policy"] or arm["reader"] != b0["reader"] or arm["baseline_pins"] != b0["baseline_pins"]
                or arm["prefix_hashes"] != b0["prefix_hashes"]
                or any(arm["protocol"][k] != b0["protocol"][k] for k in READER_KEYS)):
            raise ValueError("Comparison requires identical DEV64/documents/quartiles, policy, fixed reader and raw references")
    selection=b0["protocol"]["selection"];ids=selection["item_ids"];documents=dict(zip(ids,selection["document_ids"]))
    indexed={name:{r["item_id"]:r for r in arm["per_document"]} for name,arm in arms.items()}
    if any([r["item_id"] for r in arm["per_document"]] != ids or any(r["document_id"]!=documents[r["item_id"]]
            or r["quartile"]!=selection["strata"][r["item_id"]] for r in arm["per_document"]) for arm in arms.values()):
        raise ValueError("Per-document panel pairing differs")
    panels={"all":ids,**{q:[k for k in ids if selection["strata"][k]==q] for q in sorted(set(selection["strata"].values()))}}
    groups={};per_document=[]
    for key in ids:
        per_document.append(dict(item_id=key,document_id=documents[key],quartile=selection["strata"][key],
            arms={name:{**row, "utility":{h:{f"own_minus_{m}":v["own"]-v[m] for m in BASELINES}
                    for h,v in row["absolute_log_likelihoods"].items()}} for name,arm in indexed.items() for row in [arm[key]]}))
    for panel,keys in panels.items():
        horizons={}
        for h in (16,1):
            absolute={name:{mode:[arm[k]["absolute_log_likelihoods"][f"K{h}"][mode] for k in keys] for mode in evaluate.MODES} for name,arm in indexed.items()}
            utilities={name:{f"own_minus_{m}":[a-b for a,b in zip(v["own"],v[m])] for m in BASELINES} for name,v in absolute.items()}
            comparisons={}
            for baseline in ("B0","raw_control"):
                comparisons[f"writerBC_minus_{baseline}"]={}
                for metric in utilities["writerBC"]:
                    def paired(name):
                        return [dict(item_id=k,document_id=documents[k],score=value) for k,value in zip(keys,utilities[name][metric])]
                    result=paired_cluster_bootstrap(paired(baseline),paired("writerBC"),repeats=5000,seed=17)
                    differences=[b-a for a,b in zip(utilities[baseline][metric],utilities["writerBC"][metric])]
                    comparisons[f"writerBC_minus_{baseline}"][metric]={**result,"median_nats":statistics.median(differences),"positive_documents":sum(v>0 for v in differences)}
            horizons[f"K{h}"]=dict(arms={name:dict(absolute_mean_nats={m:statistics.mean(v) for m,v in absolute[name].items()},
                utility_mean_nats={m:statistics.mean(v) for m,v in utilities[name].items()}) for name in arms},paired_comparisons=comparisons)
        groups[panel]=dict(documents=len(keys),horizons=horizons)
    return dict(schema=SCHEMA,status="complete",created_utc=utc_now(),primary="K16",secondary="K1",documents=64,
        selection=selection,reader=b0["reader"],groups=groups,per_document=per_document,
        arms={name:dict(writer=arm["writer"],training_checkpoint_metadata={k:arm["writer"]["loaded_checkpoint"].get(k) for k in ("run_id","step","spec_hash")}
            if arm["writer"]["loaded_checkpoint"] is not None else None,
            note_context=arm["note_context"],cost=arm["cost"],input_sha256=arm["artifacts"]) for name,arm in arms.items()},
        source_files=sources(),limitations=["Exploratory paired DEV64 document uncertainty conditional on explicit checkpoints; not training-seed uncertainty.",
            "Quartile/control/horizon comparisons are descriptive and multiple; no best-checkpoint selection or multiplicity correction.",
            "Length and matched-control contexts differ by writer; utility changes need their absolute likelihood and cost context.",
            "Likelihood utility is not mathematical correctness; no oracle selection, final-test or deployment claim."])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("b0","writer","control","output"):parser.add_argument(f"--{name}",type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=compare({name:load_complete(path) for name,path in (("B0",a.b0),("writerBC",a.writer),("raw_control",a.control))})
    atomic_json(a.output,result)


if __name__=="__main__":main()
