import torch
import json
from transformers import Qwen3Config, Qwen3ForCausalLM

from train.neuron_backend import NeuronCausalBackend
from train.partial_state_answer_driver import cache_prefill_check
from train.partial_state_answers import NumberLineDecoder
from weekend_thinking.artifacts import sha256_file


def test_padded_cached_prefill_matches_ordinary_causal_forward_on_real_cpu_model():
    # CPU architecture fixture only, never a released-weight or Trainium pass.
    torch.manual_seed(17)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=151650,hidden_size=16,
        intermediate_size=24,num_hidden_layers=1,num_attention_heads=2,
        num_key_value_heads=1,head_dim=8,tie_word_embeddings=True,
        attn_implementation='eager')).eval()
    backend=NeuronCausalBackend(model,checkpoint_trunk=False)
    decoder=NumberLineDecoder(model,None)
    result,vectors=cache_prefill_check(backend,decoder,[2,3,4,5],width=8,cap=6)
    assert result['max_logp_error']<1e-3 and result['argmax_equal']
    assert vectors['ordinary'].shape==vectors['cached'].shape==(1,151650)


def test_complete_DEV_denominator_and_requested_subset_do_not_assume_TRAIN128(tmp_path,monkeypatch):
    from train import partial_state_answer_driver as driver
    questions=[dict(item_id=str(i)) for i in range(512)]
    gold=tmp_path/'gold.jsonl'
    gold.write_text(''.join(json.dumps(dict(item_id=str(i),answer=i))+'\n' for i in range(512)))
    p=dict(items=512,panel='DEV_structure',modes=['cot'],files={'gold.jsonl':sha256_file(gold)},reader={'id':'fixture'},writer=None,writer_execution_device=None,limits=[])
    monkeypatch.setattr(driver,'load',lambda _: (p,questions,[]))
    (tmp_path/'protocol.json').write_text('{}')
    for shard in range(4):
        d=tmp_path/'answers'/f'shard-{shard}';d.mkdir(parents=True)
        values=[dict(item_id=str(i),mode='cot',text=f'1 + 1 = 2\nAnswer:\n{i}\n',status='ok',stop_type='number_line',n_actions=6,n_writer_actions=0,wall_seconds=1,source_note=None) for i in range(shard,512,4)]
        file=d/'answers.jsonl';file.write_text(''.join(json.dumps(v)+'\n' for v in values))
        (d/'run.json').write_text(json.dumps(dict(status='complete_all_attempts',answers_sha256=sha256_file(file))))
    result=driver.score(tmp_path)
    assert result['cot']['items']==result['cot']['correct']==512
    assert result['cot']['unsupported']==0
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert summary['panel']=='DEV_structure' and summary['contrasts']=={}
