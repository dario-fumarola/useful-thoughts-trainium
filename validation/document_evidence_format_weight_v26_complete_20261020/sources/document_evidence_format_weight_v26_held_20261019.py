"""Two fixed answer paths for the coefficient-matched format-only control."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from train import document_qa_evaluate as engine
from train.document_evidence_plan_evaluate import prompt_text
from document_evidence_v23_likelihood_contexts import context_text
from weekend_thinking.artifacts import atomic_json,sha256_file,utc_now

R=Path('/home/ubuntu/dario');W=R/'local/worktrees/document-evidence-eval-v23'
C=R/'local/diagnostics/document-evidence-format-weight-v26-control-v1'
D=R/'local/nvme-cpt/data/document-evidence-plan-v23-TRAIN2048-held255-v2'
O=R/'local/nvme-cpt/evaluation/document-evidence-format-weight-v26-held255-v1'
OLD=R/'local/nvme-cpt/evaluation/document-evidence-weight-v25-held255-native-v1'
REPLAY=R/'local/nvme-cpt/evaluation/document-evidence-weight-v25-held255-answer-replay-v1'
SCHEMA='document-evidence-format-weight-v26-held'

def sources():
    names=('train/document_evidence_plan_evaluate.py','train/document_evidence_plan_packing.py','train/document_evidence_plan_answers.py')
    local=[Path(__file__).resolve(),R/'local/analysis/document_evidence_v23_likelihood_contexts.py']
    return {**engine.sources(),**{n:sha256_file(W/n) for n in names},**{str(p.relative_to(R)):sha256_file(p) for p in local}}

def prepare():
    import torch
    from train.evaluate_local import setup_model
    assert engine.ROOT==W
    review_file=C/'evaluation-release.json';review=engine.read(review_file)
    assert review['status']=='RELEASED_FORMAT_WEIGHT_V26_TERMINAL_FIXED_EVALUATION' and review['held_source_files']==sources()
    binding=C/'frozen-models/evidence_direct-format1-seed2026101923.json';model=engine.read(binding)
    assert model['status']=='AUDITED_EVIDENCE_FORMAT_WEIGHT_V26_TERMINAL256' and model['step']==256
    assert sha256_file(model['checkpoint'])==model['checkpoint_sha256']==review['model_checkpoint_sha256']
    for key in ('audit','protection'):assert sha256_file(model[key+'_file'])==model[key+'_sha256']
    manifest=engine.read(D/'manifest.json');assert sha256_file(D/'manifest.json')==review['data_manifest_sha256']
    info=manifest['files']['held_questions'];assert sha256_file(info['path'])==info['sha256']
    questions=engine.rows(info['path'])
    assert len(questions)==255 and all(set(q)=={'item_id','question','contexts'} for q in questions)
    p=engine.read(OLD/'protocol.json')
    assert sha256_file(OLD/'native-contexts.json')==p['files']['native-contexts.json']
    refs=engine.read(OLD/'native-contexts.json')['refs'];assert len(refs)==4
    torch.set_num_threads(16)
    backend,tok,loaded,signature,_=setup_model(SimpleNamespace(model_dir=Path(p['model_dir']),checkpoint=Path(model['checkpoint']),device='cpu'))
    assert all(loaded[k]==model[k] for k in ('step','run_id','spec_hash'))
    native=[]
    with torch.no_grad():
        for ref in refs:
            x=torch.full((1,1024),151643,dtype=torch.long);x[0,:len(ref['context_ids'])]=torch.tensor(ref['context_ids'])
            hidden=backend.hidden(x)
            native.append(backend.model.lm_head(hidden[:,len(ref['context_ids'])-1]).float().log_softmax(-1))
    native=torch.cat(native).cpu();del backend,hidden,x
    O.mkdir(parents=True,exist_ok=False);control=C/'HELD';control.mkdir(exist_ok=False)
    os.link(OLD/'native-contexts.json',O/'native-contexts.json');torch.save(native,O/'native-reference.pt')
    files={n:sha256_file(O/n) for n in ('native-contexts.json','native-reference.pt')};lengths={}
    for mode in ('direct','canonical_empty'):
        sizes=[];target=O/f'prompts-{mode}.jsonl'
        with target.open('x') as f:
            for q in questions:
                text=prompt_text(q,'direct') if mode=='direct' else context_text(q,'empty')
                parser='prior' if mode=='direct' else 'direct'
                if mode=='canonical_empty':assert text.endswith('Evidence:\nFinal answer:\n')
                ids=tok.encode(text,add_special_tokens=False)
                assert len(ids)<=3584 and not set(ids)&{151643,151645}
                sizes.append(len(ids));f.write(json.dumps(dict(item_id=q['item_id'],mode=mode,prompt_ids=ids,parser_mode=parser))+'\n')
        files[target.name]=sha256_file(target);lengths[mode]=max(sizes)
        # Byte-identical prompts to existing selector1 comparisons, except the mode label.
        original=OLD/'prompts-direct.jsonl' if mode=='direct' else REPLAY/'prompts-empty.jsonl'
        reference=engine.rows(original)
        assert [q['prompt_ids'] for q in engine.rows(target)]==[q['prompt_ids'] for q in reference]
        assert [q['item_id'] for q in reference]==[q['item_id'] for q in questions]
    release=dict(status='RELEASED_FIXED_FORMAT_WEIGHT_V26_HELD_GENERATION',utc=utc_now(),parent_review_sha256=sha256_file(review_file),source_files=sources(),model_modes={'evidence_direct':['direct','canonical_empty']},model_checkpoint_sha256=model['checkpoint_sha256'],interpretation='Inspected developmental255, no fresh confirmation. Canonical empty precloses evidence and prefills answer header.')
    atomic_json(control/'release.json',release)
    p.update(schema=SCHEMA,purpose='FIXED_FORMAT_WEIGHT_V26_DEVELOPMENTAL_HELD',created_utc=utc_now(),source_files=sources(),model=model,signature=signature,items=255,modes=['direct','canonical_empty'],shards=4,files=files,ordered_item_ids=[q['item_id'] for q in questions],maximum_prompt_tokens=lengths,question_file=info['path'],question_file_sha256=info['sha256'],release_file=str(control/'release.json'),release_sha256=sha256_file(control/'release.json'),limits=['All failures retained. Previously inspected held255; conditional developmental evidence only.','No evidence IDs or answer text supplied to the control. No prompt/parser adaptation.'])
    atomic_json(O/'protocol.json',p)
    print(json.dumps(dict(status='PREPARED_FORMAT_CONTROL_FIXED_HELD',items=255,modes=p['modes'],protocol_sha256=sha256_file(O/'protocol.json'))))

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('command',choices=('prepare','generate'));a.add_argument('--output',type=Path,default=O);a.add_argument('--shard',type=int);a.add_argument('--until');args=a.parse_args()
    if args.command=='prepare':prepare()
    else:
        assert args.shard is not None and args.until
        engine.generate(args.output,args.shard,args.until,schema=SCHEMA,expected_sources=sources())
