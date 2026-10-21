"""Read the complete, fixed V30 diagnostic once; never call or update a model."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from train.candidate_utility_v30 import correctness
from train.real_cpt_readout import numeric_masks
from weekend_thinking.artifacts import atomic_json, sha256_file
from weekend_thinking.stats import paired_cluster_bootstrap


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).open() if s.strip()]


def contrast(values):
    """One already-aggregated paired difference per unique source document."""
    if len(values) < 2:
        return dict(status='unidentifiable',documents=len(values),passed=False)
    a = [dict(item_id=k,document_id=k,score=0.) for k,_ in values]
    b = [dict(item_id=k,document_id=k,score=v) for k,v in values]
    return paired_cluster_bootstrap(a,b,repeats=5000,seed=2026102130)


def positive(value, minimum=.25):
    return (value.get('difference_b_minus_a',-math.inf) >= minimum
            and value.get('ci95',[-math.inf])[0] > 0)


def decide(contrasts, *, helpful_documents, mixed_documents, alignment):
    return dict(complete_execution=True,
        opportunity=all(positive(contrasts[k]) for k in
            ('best8_minus_raw','best8_minus_B0_raw','best8_minus_greedy')),
        correct_helpful_coverage=helpful_documents>=64,
        alignment=mixed_documents>=32 and positive(alignment),
        plumbing=positive(contrasts['oracle_minus_corrupt'],minimum=0.))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args(); protocol = read(a.protocol/'protocol.json')
    design = read(protocol['design_file'])
    assert sha256_file(protocol['design_file']) == protocol['design_sha256']
    assert design['experiment_id'] == 'candidate-utility-v30'
    for name,digest in protocol['files'].items(): assert sha256_file(a.protocol/name) == digest
    labels = rows(a.protocol/'diagnostic-labels.jsonl'); assert len(labels) == 256
    notes = {}; scores = {}; source_pins = {}
    for shard in range(4):
        directory = a.runs/f'generate-{shard}'; complete = read(directory/'complete.json')
        assert complete['status'] == 'COMPLETE_V30_ORIGINAL_CANDIDATE_SHARD'
        assert complete['protocol_sha256'] == sha256_file(a.protocol/'protocol.json')
        assert sha256_file(directory/'notes.jsonl') == complete['notes_sha256']
        source_pins[str(directory/'notes.jsonl')] = complete['notes_sha256']
        for r in rows(directory/'notes.jsonl'):
            assert r['status'] == 'ok'
            key = (r['item_id'],r['mode']); assert key not in notes; notes[key] = r
            n = len(r['actions'])
            assert 1 <= n <= 128 and len(r['behavior_logps']) == n
            assert r['action_mask'] == [True]*n+[False]*(128-n)
            assert all(math.isfinite(v) for v in r['behavior_logps'])
        for prefix in ('score-T','score-B0'):
            directory = a.runs/f'{prefix}-{shard}'; complete = read(directory/'complete.json')
            assert complete['status'] == 'COMPLETE_V30_FIXED_READER_SCORE_SHARD'
            assert complete['protocol_sha256'] == sha256_file(a.protocol/'protocol.json')
            assert sha256_file(directory/'scores.jsonl') == complete['scores_sha256']
            source_pins[str(directory/'scores.jsonl')] = complete['scores_sha256']
            for r in rows(directory/'scores.jsonl'):
                assert r['status'] == 'ok' and r['tokens'] == len(r['token_logps']) == 16
                assert all(math.isfinite(v) for v in r['token_logps'])
                assert abs(sum(r['token_logps'])-r['full_logp']) < 1e-9
                key=(r['item_id'],r['mode']);assert key not in scores;scores[key]=r
    ids = {r['item_id'] for r in labels}
    assert len(notes) == 256*9 and len(scores) == 256*15
    assert {key[0] for key in notes} == {key[0] for key in scores} == ids
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('/home/ubuntu/dario/local/assets/qwen3-1.7b-base',local_files_only=True)
    aggregates=[]; alignment=[]; categories=Counter(); examples=[]
    deltas = {name:[] for name in ('best8_minus_raw','best8_minus_B0_raw','best8_minus_greedy',
        'mean8_minus_raw','greedy_minus_raw','oracle_minus_corrupt','oracle_minus_raw')}
    components = {name:[] for name in ('numeral','boundary_and_whitespace','surrounding')}
    helpful=0; valid_canonical_documents=0; canonical_only=[]
    for row in labels:
        item=row['item_id']; s={mode:r for (key,mode),r in scores.items() if key==item}
        n={mode:r for (key,mode),r in notes.items() if key==item}
        assert set(s) == {'raw','empty','greedy','oracle','corrupt','other','B0_raw'}|{f'sample{i}' for i in range(8)}
        assert all(r['target_ids'] == row['continuation16_ids'] for r in s.values())
        checks={mode:correctness(r['text'],row['full_calculation_text'],row['numeric_value']) for mode,r in n.items()}
        sampled=[f'sample{i}' for i in range(8)]
        for mode in sampled: categories[checks[mode]['category']]+=1
        best=max(sampled,key=lambda mode:s[mode]['full_logp'])
        raw=s['raw']['full_logp']; b0=s['B0_raw']['full_logp']; greedy=s['greedy']['full_logp']
        full=s[best]['full_logp']
        for name,value in dict(best8_minus_raw=full-raw,best8_minus_B0_raw=full-b0,
            best8_minus_greedy=full-greedy,mean8_minus_raw=statistics.mean(s[m]['full_logp'] for m in sampled)-raw,
            greedy_minus_raw=greedy-raw,oracle_minus_corrupt=s['oracle']['full_logp']-s['corrupt']['full_logp'],
            oracle_minus_raw=s['oracle']['full_logp']-raw).items(): deltas[name].append((item,value))
        correct=[m for m in sampled if checks[m]['category']=='verified_canonical']
        wrong=[m for m in sampled if checks[m]['category']=='arithmetic_error']
        if correct:
            valid_canonical_documents+=1
            canonical_only.append((item,statistics.mean(s[m]['full_logp'] for m in correct)-raw))
        helpful+=any(s[m]['full_logp'] > max(raw,b0) for m in correct)
        if correct and wrong:
            alignment.append((item,statistics.mean(s[m]['full_logp'] for m in correct)
                                   -statistics.mean(s[m]['full_logp'] for m in wrong)))
        masks=numeric_masks(row,tok)
        for name in components:
            components[name].append((item,sum((x-y) for x,y,active in zip(
                s[best]['token_logps'],s['raw']['token_logps'],masks[name]) if active)))
        aggregate=dict(item_id=item,document_id=row['document_id'],best8=best,
            best8_category=checks[best]['category'],correct_samples=len(correct),arithmetic_error_samples=len(wrong),
            unique_sampled_contents=len({tuple(n[m]['content_ids']) for m in sampled}),
            best8_minus_raw=full-raw,best8_minus_B0_raw=full-b0,
            correctness=checks)
        aggregates.append(aggregate)
        if len(examples)<8:
            examples.append(dict(**aggregate,greedy=n['greedy']['text'],
                sampled=[dict(mode=m,text=n[m]['text'],full16_gain_vs_raw=s[m]['full_logp']-raw,
                    category=checks[m]['category'],stop_type=n[m]['stop_type']) for m in sampled]))
    contrasts={k:contrast(v) for k,v in deltas.items()}
    align=contrast(alignment)
    gates=decide(contrasts,helpful_documents=helpful,mixed_documents=len(alignment),alignment=align)
    passed=all(gates.values())
    result=dict(experiment_id='candidate-utility-v30',status='COMPLETE_FIXED_TRAIN_DIAGNOSTIC',
        documents=256,candidate_attempts=len(notes),score_rows=len(scores),gates=gates,all_required_passed=passed,
        decision=('Oracle opportunity only; no learning released.' if passed else
            'Close this reward branch. No RL under this reward; no temperature/prompt/checkpoint rescue sweep.'),
        contrasts=contrasts,alignment=dict(mixed_documents=len(alignment),conditional_contrast=align),
        verified_canonical_documents=valid_canonical_documents,correct_helpful_documents=helpful,
        correct_helpful_fraction_interval=contrast([(r['item_id'],float(any(
            r['correctness'][f'sample{i}']['category']=='verified_canonical' and
            scores[(r['item_id'],f'sample{i}')]['full_logp']>max(
                scores[(r['item_id'],'raw')]['full_logp'],scores[(r['item_id'],'B0_raw')]['full_logp'])
            for i in range(8)))) for r in aggregates]),
        canonical_mean_gain_vs_raw_conditional=contrast(canonical_only),
        candidate_categories=dict(categories),best8_categories=dict(Counter(r['best8_category'] for r in aggregates)),
        best8_native_components_vs_raw={k:contrast(v) for k,v in components.items()},
        caps=dict(Counter(r['mode'] for r in notes.values() if r['stop_type']=='length_cap')),
        empty=dict(Counter(r['mode'] for r in notes.values() if not r['content_ids'])),
        mean_actions=statistics.mean(len(r['actions']) for r in notes.values()),
        source_pins=source_pins,protocol_sha256=sha256_file(a.protocol/'protocol.json'),
        design_sha256=protocol['design_sha256'],analysis_source_sha256=sha256_file(__file__),
        limits=design['statistics']['limits']+' Best-of8 sees the actual future and is not a deployed selection policy.')
    a.output.mkdir(parents=True,exist_ok=False)
    atomic_json(a.output/'readout.json',result)
    atomic_json(a.output/'document-aggregates.json',aggregates)
    atomic_json(a.output/'fixed-first-eight-examples.json',examples)
    print(json.dumps(dict(gates=gates,all_required_passed=passed,decision=result['decision']),indent=2))


if __name__=='__main__':
    main()
