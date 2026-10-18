"""CPU question-boundary, real-end/cache and complete-bundle semantics."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from train import calculation_teacher as teacher
from train.neuron_backend import NeuronCausalBackend
from weekend_thinking.artifacts import canonical_hash, sha256_file
from weekend_thinking.core import policy_log_probs


def questions():
    return [dict(item_id=f'item-{i}',document_id=f'doc-{i}',task='gsm8k',split='train',
                 question='Compute '+str(i)+' plus two.') for i in range(32)]


def write_rows(path,rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


@pytest.mark.parametrize('mutation',[{'answer':'secret'},{'verified_note':'secret'},
    {'gold':3},{'split':'dev'},{'question':''}])
def test_question_file_rejects_gold_unknown_fields_or_nontrain(tmp_path,mutation):
    rows=questions();rows[0].update(mutation);p=tmp_path/'questions.jsonl';write_rows(p,rows)
    with pytest.raises(ValueError,match='question-only'):teacher.question_rows(p)


def test_exact32_unique_order_is_preserved(tmp_path):
    rows=questions();p=tmp_path/'questions.jsonl';write_rows(p,rows)
    assert teacher.question_rows(p)==rows
    assert [[x['item_id'] for x in teacher.shard_questions(rows,s,4)] for s in range(4)]==[
        [f'item-{i}' for i in range(s,32,4)] for s in range(4)]
    write_rows(p,rows[:31])
    with pytest.raises(ValueError):teacher.question_rows(p)
    rows[-1]=rows[0];write_rows(p,rows)
    with pytest.raises(ValueError):teacher.question_rows(p)


class Tokenizer:
    chat_template='native'
    eos_token_id,pad_token_id=151645,151643

    def get_vocab(self):
        return {'newline':198,'eos':151645,'pad':151643}

    def apply_chat_template(self,messages,*,tokenize,add_generation_prompt,enable_thinking):
        assert add_generation_prompt and enable_thinking is False
        assert [m['role'] for m in messages]==['system','user']
        text='SYSTEM:'+messages[0]['content']+'\nUSER:'+messages[1]['content']+'\nASSISTANT:<think>\n\n</think>\n\n'
        return self.encode(text,add_special_tokens=False) if tokenize else text

    def encode(self,text,*,add_special_tokens):
        assert not add_special_tokens
        if text=='\n':return [198]
        return [ord(c) for c in text]


def test_native_prompt_contains_only_complete_question_and_fixed_instruction():
    a=teacher.native_prompt(Tokenizer(),'Two plus three?');b=teacher.native_prompt(Tokenizer(),'A longer question?')
    assert a['messages'][1]['content']=='Two plus three?'
    assert 'Omit the final answer' in a['messages'][0]['content']
    assert a['question_sha256']==canonical_hash('Two plus three?')
    assert len(a['prompt_ids'])!=len(b['prompt_ids'])
    assert '\n\n</think>\n\n' in a['rendered']


def test_native_prompt_accepts_actual_BatchEncoding_container_but_not_changed_IDs():
    from transformers.tokenization_utils_base import BatchEncoding
    class NativeContainer(Tokenizer):
        def apply_chat_template(self,messages,**kwargs):
            value=super().apply_chat_template(messages,**kwargs)
            return BatchEncoding(dict(input_ids=value,attention_mask=[1]*len(value))) if kwargs['tokenize'] else value
    expected=teacher.native_prompt(Tokenizer(),'Two plus three?')
    assert teacher.native_prompt(NativeContainer(),'Two plus three?')==expected
    class WrongIDs(NativeContainer):
        def apply_chat_template(self,messages,**kwargs):
            value=super().apply_chat_template(messages,**kwargs)
            if kwargs['tokenize']:value['input_ids']=value['input_ids']+[42]
            return value
    with pytest.raises(ValueError,match='tokenization differs'):
        teacher.native_prompt(WrongIDs(),'Two plus three?')


def test_prepare_binds_actual_prompt_document_metadata_and_greedy_recipe(tmp_path,monkeypatch):
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**k:Tokenizer())
    monkeypatch.setattr(teacher,'verify_teacher_assets',lambda _:None)
    monkeypatch.setattr(teacher,'check_deadline',lambda _:None)
    source=tmp_path/'questions.jsonl';values=questions();write_rows(source,values)
    for name in ('teacher','base'):
        (tmp_path/name).mkdir();(tmp_path/name/'ASSET_MANIFEST.json').write_text('{}')
    output=tmp_path/'prepared'
    p=teacher.prepare(output,source,tmp_path/'teacher',tmp_path/'base','bounded')
    inputs=json.loads((output/'inputs.json').read_text())
    assert inputs['rows']==values and inputs['question_only'] is True
    assert [(x['item_id'],x['document_id']) for x in inputs['prompts']]==[
        (x['item_id'],x['document_id']) for x in values]
    assert p['prefill_width']==max(len(x['prompt_ids']) for x in inputs['prompts'])
    assert p['temperature']==1. and p['cap']==256 and p['stop_ids']==[151645,151643]
    assert p['item_ids']==[x['item_id'] for x in values] and p['native_vocab']['equal']
    loaded_p,loaded_inputs=teacher.read_protocol(output)
    assert loaded_p==p and loaded_inputs==inputs


def tiny_model():
    from transformers import Qwen3Config,Qwen3ForCausalLM
    torch.manual_seed(31)
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=37,hidden_size=24,intermediate_size=40,
        num_hidden_layers=2,num_attention_heads=3,num_key_value_heads=1,head_dim=8,
        tie_word_embeddings=True,attn_implementation='eager')).eval()


def test_actual_last_position_and_cached_teacher_forced_greedy_parity(monkeypatch):
    monkeypatch.setattr(teacher,'CAP',8)
    model=tiny_model();backend=NeuronCausalBackend(model,checkpoint_trunk=False,banned_ids=(36,))
    tok=SimpleNamespace(decode=lambda ids,**kwargs:' '.join(map(str,ids)))
    decoder=teacher.RecordingDecoder(model,tok,banned_ids=(36,))
    prompt=[3,4,7]
    with torch.no_grad():
        padded=teacher.padded_next(decoder,prompt,7,0)
        hidden=model.model(input_ids=torch.tensor([prompt]),use_cache=False).last_hidden_state
        native=model.lm_head(hidden[:,-1]).float()
        assert torch.allclose(padded,native,atol=2e-6)
        decoder.record=True;decoder.cached_logps=[]
        decoded=decoder.decode(prompt,max_new_tokens=6,prefill_width=7,pad_id=0,
            stop_ids=(151645,151643),answer_line_stop=False)
        assert len(decoded.actions)==6 and 36 not in decoded.actions
        cached=torch.stack(decoder.cached_logps)
        replay=teacher.replay_logps(backend,prompt,decoded.actions,7,0)
        assert torch.allclose(cached,replay,atol=2e-6)


def test_cpu_reference_uses_unpadded_real_end_and_last_hidden_head(tmp_path,monkeypatch):
    import transformers
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.embedding=torch.nn.Embedding(8,3);self.lm_head=torch.nn.Linear(3,8,bias=False)
            self.config=SimpleNamespace(vocab_size=8);self.lengths=[];self.head_shapes=[]
            self.lm_head.register_forward_pre_hook(lambda _,args:self.head_shapes.append(list(args[0].shape)))
        @property
        def model(self):return self
        def tie_weights(self):self.lm_head.weight=self.embedding.weight
        def get_input_embeddings(self):return self.embedding
        def get_output_embeddings(self):return self.lm_head
        def forward(self,*,input_ids,use_cache):
            assert use_cache is False and not torch.is_grad_enabled()
            self.lengths.append(input_ids.shape[1]);return SimpleNamespace(last_hidden_state=self.embedding(input_ids))
    model=Model();info={k:set() for k in ('missing_keys','unexpected_keys','mismatched_keys')};info['error_msgs']=[]
    monkeypatch.setattr(transformers.AutoModelForCausalLM,'from_pretrained',lambda *a,**k:(model,info))
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**k:SimpleNamespace(get_vocab=lambda:{str(i):i for i in range(7)}))
    p=dict(teacher_dir='mock',item_ids=[f'item-{i}' for i in range(32)],
           native_vocab={'mapping_sha256':canonical_hash({str(i):i for i in range(7)})})
    inputs=dict(prompts=[dict(item_id=f'item-{i}',prompt_ids=[i%7]*(i+2)) for i in range(4)])
    monkeypatch.setattr(teacher,'read_protocol',lambda _: (p,inputs))
    monkeypatch.setattr(teacher,'verify_teacher_assets',lambda _:None)
    monkeypatch.setattr(teacher,'check_deadline',lambda _:None)
    (tmp_path/'protocol.json').write_text('{}')
    teacher.cpu_reference(tmp_path,1,'bounded')
    r=json.loads((tmp_path/'cpu-reference/reference.json').read_text())
    assert model.lengths==[2,3,4,5] and model.head_shapes==[[1,3]]*4
    assert r['shape']==[4,8] and r['tied'] and r['loading_info_types']['missing_keys']=='set'
    assert torch.load(tmp_path/'cpu-reference/next-logps.pt',weights_only=True).shape==(4,8)
    with pytest.raises(ValueError,match='16 threads'):teacher.cpu_reference(tmp_path,17,'bounded')


def complete_bundle(tmp_path,monkeypatch):
    ids=[f'item-{i}' for i in range(32)];p=dict(item_ids=ids,forward_atol=.002,replay_atol=.001)
    prompts=[dict(item_id=i,document_id=f'doc-{n}',prompt_ids=[7],question_sha256=canonical_hash(i)) for n,i in enumerate(ids)]
    inputs=dict(rows=questions(),prompts=prompts);signature={'parameters':12}
    monkeypatch.setattr(teacher,'read_protocol',lambda _:(p,inputs))
    monkeypatch.setattr(teacher,'reference_receipt',lambda *_:dict(signature=signature))
    (tmp_path/'protocol.json').write_text('{}')
    for shard in range(4):
        d=tmp_path/'generation'/f'shard-{shard}';d.mkdir(parents=True)
        rows=[]
        for item in prompts[shard::4]:
            row=dict(item_id=item['item_id'],document_id=item['document_id'],status='ok',split='train',task='gsm8k',shard=shard,
                actions=[151645],content_ids=[],action_mask=[True]+[False]*255,text='',raw_text='',
                stop_type='token:151645',n_actions=1,n_content_tokens=0,greedy=True,temperature=1.,
                gold_or_verified_notes_exposed=False,prompt_sha256=canonical_hash([7]),
                question_sha256=item['question_sha256'],generation_wall_seconds=.1)
            rows.append(row)
        rows[0].update(cached_logps=[-.25],replayed_logps=[-.25])
        write_rows(d/'notes.jsonl',rows)
        run=dict(status='complete',shard=shard,notes=8,expected_notes=8,item_ids=ids[shard::4],
            protocol_sha256=sha256_file(tmp_path/'protocol.json'),notes_sha256=sha256_file(d/'notes.jsonl'),
            signature=signature,forward_check=dict(max_error=0.),
            replay_check=dict(max_error=0.,item_id=ids[shard],actions=1))
        (d/'run.json').write_text(json.dumps(run))
    return p,inputs


def test_complete_loader_returns_all32_notes_and_pin_paths_without_model_load(tmp_path,monkeypatch):
    p,inputs=complete_bundle(tmp_path,monkeypatch)
    actual,actual_inputs,notes,receipts=teacher.load_complete(tmp_path)
    assert actual is p and actual_inputs is inputs and set(notes)==set(p['item_ids'])
    assert len(receipts)==13 and all(isinstance(path,Path) for path in receipts)


@pytest.mark.parametrize('change',[{'status':'failed'},{'notes':7},{'signature':{'parameters':13}},
    {'forward_check':{'max_error':.003}},{'replay_check':{'max_error':.002}}])
def test_complete_loader_rejects_partial_changed_model_or_failed_parity(tmp_path,monkeypatch,change):
    complete_bundle(tmp_path,monkeypatch);path=tmp_path/'generation/shard-1/run.json'
    run=json.loads(path.read_text());run.update(change);path.write_text(json.dumps(run))
    with pytest.raises(ValueError,match='complete'):teacher.load_complete(tmp_path)


def test_complete_loader_checks_stop_and_first_real_replay_even_with_new_file_hash(tmp_path,monkeypatch):
    complete_bundle(tmp_path,monkeypatch);d=tmp_path/'generation/shard-0'
    rows=[json.loads(x) for x in (d/'notes.jsonl').read_text().splitlines()]
    rows[0]['replayed_logps']=[-.5];write_rows(d/'notes.jsonl',rows)
    run=json.loads((d/'run.json').read_text());run['notes_sha256']=sha256_file(d/'notes.jsonl');(d/'run.json').write_text(json.dumps(run))
    with pytest.raises(ValueError,match='cached/replay'):teacher.load_complete(tmp_path)
