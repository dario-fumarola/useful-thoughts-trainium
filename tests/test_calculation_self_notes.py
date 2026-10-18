"""Question-only stochastic sampling, first-line transforms, and closure barriers."""
import json
from types import SimpleNamespace

import pytest
import torch

import train.calculation_self_notes as probe
from train.evaluate import GreedyDecoder
from train.neuron_backend import NeuronCausalBackend
from weekend_thinking.artifacts import canonical_hash, sha256_file
from weekend_thinking.core import policy_log_probs


class Tokenizer:
    pieces = {1: 'a', 2: 'a\nextra', 3: '\nremaining', 4: '#### 42', probe.EOS: '<EOS>'}
    def decode(self, ids, **kw):
        return ''.join(self.pieces.get(i, str(i)) for i in ids)
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [] if text == '' else [1] if text == 'a' else list(text.encode())
    def get_vocab(self):
        return {'a': 1, 'merged': 2, 'newline': 3}


@pytest.mark.parametrize('actions,stop,content,original', [
    ([2], 'newline', 'a', [2]),
    ([3], 'newline', '', [3]),
    ([1, probe.EOS], f'token:{probe.EOS}', 'a', [1]),
    ([1]*64, 'length_cap', 'a'*64, [1]*64),
    ([4, probe.EOS], f'token:{probe.EOS}', '#### 42', [4]),
])
def test_first_line_transform_retains_merged_bytes_eos_caps_and_violations(actions, stop, content, original):
    tok=Tokenizer();row=probe.formatted_actions(tok,actions)
    assert row['content_text']==content and row['content_ids']==tok.encode(content,add_special_tokens=False)
    assert row['raw_text']==tok.decode(actions) and row['original_content_ids']==original
    assert row['original_content_text']==tok.decode(original) and row['stop_type']==stop
    assert row['n_actions']==len(actions) and row['empty_first_line']==(content=='')
    assert row['final_answer_marker_in_raw']==('####' in row['raw_text'])


@pytest.mark.parametrize('actions', [[], [1], [2,1], [probe.EOS,1], [1]*65])
def test_paths_cannot_continue_after_the_first_stop_or_end_without_one(actions):
    with pytest.raises(ValueError):probe.formatted_actions(Tokenizer(),actions)


def questions():
    return [dict(item_id=f'gsm8k:train:{i+100:05d}',document_id=f'gsm8k:train:{i+100:05d}',
                 task='gsm8k',split='train',question=f'Question number {i}?') for i in range(32)]


@pytest.mark.parametrize('fault', ['answer', 'dev', 'duplicate', 'demonstration'])
def test_question_only_projection_rejects_answer_fields_wrong_split_and_collisions(tmp_path,fault):
    rows=questions()
    if fault=='answer':rows[0]['answer']='not permitted'
    elif fault=='dev':rows[0]['split']='dev'
    elif fault=='duplicate':rows[-1]=rows[0]
    else:rows[0].update(item_id='gsm8k:train:00000',document_id='gsm8k:train:00000')
    path=tmp_path/'questions.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError):probe.question_rows(path)


def test_only_pinned_train_question_column_enters_exact_plain_demonstrations(tmp_path,monkeypatch):
    import pyarrow.parquet as pq
    path=tmp_path/'canonical.parquet';path.write_bytes(b'pinned fixture')
    monkeypatch.setattr(probe,'TRAIN_SHA',sha256_file(path));calls=[]
    def read_table(source,*,columns):
        assert source==path and columns==['question'];calls.append(columns)
        values=[SimpleNamespace(as_py=lambda i=i:f'Canonical question {i}?') for i in range(3)]
        return SimpleNamespace(column=lambda key:values if key=='question' else pytest.fail('Answer column accessed'))
    monkeypatch.setattr(pq,'read_table',read_table)
    demos=probe.canonical_demonstrations(path);text=probe.writer_prompt('CURRENT QUESTION ONLY',demos)
    assert calls==[['question']] and [d['calculation'] for d in demos]==list(probe.DEMO_CALCULATIONS)
    assert text.startswith(probe.INSTRUCTION+'\n\nQuestion: Canonical question 0?\nCalculation: ')
    assert text.endswith('\n\nQuestion: CURRENT QUESTION ONLY\nCalculation:')
    assert text.count('Question: ')==4 and '<think>' not in text and '<|im_start|>' not in text


def test_sample_seeds_are_question_and_path_specific_independent_of_sharding():
    ids=[r['item_id'] for r in questions()]
    seeds={(key,n):probe.sample_seed(123,key,n) for key in ids for n in range(4)}
    assert len(set(seeds.values()))==128
    assert seeds=={(key,n):probe.sample_seed(123,key,n) for shard in range(4) for key in ids[shard::4] for n in range(4)}
    with pytest.raises(ValueError):probe.sample_seed(123,ids[0],True)


def test_actual_tiny_qwen_cached_sampling_replays_every_action_at_same_temperature_and_support():
    from transformers import Qwen3Config,Qwen3ForCausalLM
    torch.manual_seed(19)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=37,hidden_size=24,intermediate_size=40,
        num_hidden_layers=2,num_attention_heads=3,num_key_value_heads=1,head_dim=8,
        tie_word_embeddings=True,attn_implementation='eager')).eval()
    tok=Tokenizer();backend=NeuronCausalBackend(model,checkpoint_trunk=False,banned_ids=(0,36))
    decoder=GreedyDecoder(model,tok,banned_ids=backend.banned_ids)
    prompt=[10,11,12];seed=probe.sample_seed(19,'one-question',2)
    a=probe.sample_note(decoder,prompt,width=7,seed=seed,set_seed=torch.manual_seed,cap=8,eos_id=0)
    b=probe.sample_note(decoder,prompt,width=7,seed=seed,set_seed=torch.manual_seed,cap=8,eos_id=0)
    assert a['status']==b['status']=='ok' and a['actions']==b['actions'] and a['cached_logps']==b['cached_logps']
    assert not set(a['actions']) & {0,36}
    replay=probe.replay_logps(backend,prompt,a['actions'],7,cap=8,pad_id=0)
    with torch.no_grad():
        x=torch.tensor([prompt+a['actions'][:-1]])
        lp=policy_log_probs(model(x).logits[:,len(prompt)-1:],.7,(0,36))
        native=lp.gather(-1,torch.tensor(a['actions'])[None,:,None])[0,:,0]
    assert len(replay)==len(a['actions']) and torch.allclose(replay,native,atol=2e-6)
    assert torch.allclose(replay,torch.tensor(a['cached_logps']),atol=2e-6)
    padded=probe.padded_next(decoder,prompt,7,pad_id=0,cap=8)
    with torch.no_grad():native_next=model(torch.tensor([prompt])).logits[:,-1]
    assert torch.allclose(padded,native_next,atol=2e-6)


@pytest.fixture
def bundle(tmp_path,monkeypatch):
    rows=questions();ids=[r['item_id'] for r in rows];tok=Tokenizer()
    prompts=[dict(item_id=r['item_id'],document_id=r['document_id'],prompt_ids=[1],
                  question_sha256=canonical_hash(r['question'])) for r in rows]
    p=dict(item_ids=ids,seed=19,vocab_size=4)
    inputs=dict(rows=rows,prompts=prompts)
    (tmp_path/'protocol.json').write_text('{}');(tmp_path/'inputs.json').write_text('{}')
    refdir=tmp_path/'cpu-reference';refdir.mkdir();torch.save(torch.zeros(4,4),refdir/'next-logps.pt')
    (refdir/'reference.json').write_text('{}');ref=dict(shape=[4,4],signature={'base':'fixture'})
    monkeypatch.setattr(probe,'read_protocol',lambda output:(p,inputs))
    monkeypatch.setattr(probe,'reference_receipt',lambda output,protocol:ref)
    monkeypatch.setattr(probe,'tokenizer_for',lambda protocol:tok)
    monkeypatch.setattr(probe,'load_base',lambda *a,**kw:pytest.fail('Model accessed by complete loader'))
    for shard in range(4):
        d=tmp_path/'generation'/f'shard-{shard}';d.mkdir(parents=True)
        values=[]
        for i,key in enumerate(ids[shard::4]):
            for sample in range(4):
                actions=[3] if (i+sample)%3==0 else [2]
                row=dict(status='ok',item_id=f'{key}:sample:{sample}',question_id=key,document_id=key,
                    sample_index=sample,shard=shard,split='train',task='gsm8k',actions=actions,cached_logps=[-1.],
                    action_mask=[True]+[False]*63,seed=probe.sample_seed(19,key,sample),temperature=.7,
                    prompt_sha256=canonical_hash([1]),question_sha256=canonical_hash(rows[ids.index(key)]['question']),
                    current_gold_teacher_or_manual_notes_exposed=False,generation_wall_seconds=.01,
                    **probe.formatted_actions(tok,actions))
                if i==0 and sample==0:row['replayed_logps']=[-1.]
                values.append(row)
        (d/'notes.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in values))
        torch.save(torch.zeros(1,4),d/'forward-logps.pt')
        report=dict(status='complete',shard=shard,notes=32,expected_notes=32,item_ids=ids[shard::4],
            protocol_sha256=sha256_file(tmp_path/'protocol.json'),signature=ref['signature'],notes_sha256=sha256_file(d/'notes.jsonl'),
            forward_check=dict(max_error=0.,atol=.002,prompt_sha256=canonical_hash([1]),
                logps_sha256=sha256_file(d/'forward-logps.pt'),native_reference_sha256=sha256_file(refdir/'reference.json')),
            replay_check=dict(question_id=ids[shard],sample_index=0,actions=1,max_error=0.,atol=.001))
        (d/'run.json').write_text(json.dumps(report))
    return tmp_path,p,inputs


def alter_notes(output,change):
    path=output/'generation/shard-0/notes.jsonl';rows=[json.loads(s) for s in path.read_text().splitlines()]
    change(rows);path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report_path=path.parent/'run.json';r=json.loads(report_path.read_text());r['notes_sha256']=sha256_file(path)
    report_path.write_text(json.dumps(r))


def test_closed_loader_requires_all128_and_preserves_empty_paths_without_model_or_gold_access(bundle):
    output,p,inputs=bundle
    protocol,actual,notes,receipts=probe.load_complete(output)
    assert protocol==p and actual==inputs and len(notes)==128 and len(receipts)==16
    assert set(notes)=={(k,n) for k in p['item_ids'] for n in range(4)}
    assert any(r['content_ids']==[] for r in notes.values())


@pytest.mark.parametrize('fault', ['missing','duplicate','wrong_seed','wrong_content','failed','last_action_audit'])
def test_coherent_file_rehash_cannot_hide_missing_failed_transformed_or_unaudited_paths(bundle,fault):
    output,_,_=bundle
    def change(rows):
        if fault=='missing':rows.pop()
        elif fault=='duplicate':rows[-1]=rows[0]
        elif fault=='wrong_seed':rows[5]['seed']+=1
        elif fault=='wrong_content':rows[5]['content_text']='A nicer hand-edited note'
        elif fault=='failed':rows[5].update(status='error',error='preserved failure')
        else:rows[0]['replayed_logps']=[]
    alter_notes(output,change)
    with pytest.raises(ValueError):probe.load_complete(output)


@pytest.mark.parametrize('fault', ['forward','replay'])
def test_saved_numerical_evidence_is_recomputed_not_trusted_from_pass_labels(bundle,fault):
    output,_,_=bundle;d=output/'generation/shard-0'
    if fault=='forward':
        torch.save(torch.ones(1,4),d/'forward-logps.pt')
        r=json.loads((d/'run.json').read_text());r['forward_check']['logps_sha256']=sha256_file(d/'forward-logps.pt')
        (d/'run.json').write_text(json.dumps(r))
    else:alter_notes(output,lambda rows:rows[0].update(replayed_logps=[-.98]))
    with pytest.raises(ValueError):probe.load_complete(output)


@pytest.mark.parametrize('failure', ['exception','short','nonfinite','mismatch'])
def test_first_replay_failure_keeps_actual_sample_before_abort(tmp_path,monkeypatch,failure):
    rows=questions();prompts=[dict(item_id=r['item_id'],document_id=r['document_id'],prompt_ids=[1],
        question_sha256=canonical_hash(r['question'])) for r in rows]
    p=dict(model_dir='not-read',assets={},banned_ids=[],prefill_width=1,seed=19)
    (tmp_path/'protocol.json').write_text('{}');refdir=tmp_path/'cpu-reference';refdir.mkdir()
    (refdir/'reference.json').write_text('{}');torch.save(torch.zeros(4,4),refdir/'next-logps.pt')
    monkeypatch.setattr(probe,'read_protocol',lambda _:(p,dict(prompts=prompts)))
    monkeypatch.setattr(probe,'reference_receipt',lambda *a:dict(signature={'x':1}))
    monkeypatch.setattr(probe,'base_assets',lambda _: {})
    backend=SimpleNamespace(banned_ids=(),sync=lambda:None,model=None)
    tok=Tokenizer()
    monkeypatch.setattr(probe,'setup_model',lambda _: (backend,tok,None,{'x':1},lambda _:None))
    monkeypatch.setattr(probe,'GreedyDecoder',lambda *a,**kw:None)
    monkeypatch.setattr(probe,'padded_next',lambda *a,**kw:torch.zeros(1,4))
    monkeypatch.setattr(probe,'policy_log_probs',lambda *a,**kw:torch.zeros(1,4))
    sampled=dict(status='ok',actions=[2],cached_logps=[-1.],action_mask=[True]+[False]*63,
        generation_wall_seconds=.1,**probe.formatted_actions(tok,[2]))
    monkeypatch.setattr(probe,'sample_note',lambda *a,**kw:dict(sampled))
    def replay(*a,**kw):
        if failure=='exception':raise RuntimeError('Compiler/runtime audit failed')
        return torch.tensor([] if failure=='short' else [float('nan')] if failure=='nonfinite' else [-2.])
    monkeypatch.setattr(probe,'replay_logps',replay)
    with pytest.raises(RuntimeError,match='Failed note retained'):
        probe.generate(tmp_path,0,'2099-01-01T00:00:00+00:00')
    d=tmp_path/'generation/shard-0';actual=[json.loads(s) for s in (d/'notes.jsonl').read_text().splitlines()]
    assert len(actual)==1 and actual[0]['status']=='error'
    assert actual[0]['actions']==[2] and actual[0]['cached_logps']==[-1.] and actual[0]['raw_text']=='a\nextra'
    r=json.loads((d/'run.json').read_text());assert r['status']=='failed' and r['notes']==1 and r['replay_check']['error']
