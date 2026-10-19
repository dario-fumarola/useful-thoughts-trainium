"""Count retired replay work and check one causal-padding invariant on tiny CPU Qwen.

No released-model inference, accelerator profiling, learning run or speed claim.
The deliberately nonzero test advantage checks gradient equality even for an
empty note; actual all-empty leave-one-out groups would have zero advantage.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from train.neuron_backend import NeuronCausalBackend, pack_reader
from weekend_thinking.backend_contract import Rollouts
from weekend_thinking.core import Format, action_mask

R = Path('/home/ubuntu/dario')
torch.set_num_threads(2)
torch.manual_seed(19)
m = Qwen3ForCausalLM(Qwen3Config(vocab_size=37, hidden_size=24,
    intermediate_size=40, num_hidden_layers=2, num_attention_heads=3,
    num_key_value_heads=1, head_dim=8, tie_word_embeddings=True,
    attn_implementation='eager')).eval()
backend = NeuronCausalBackend(m, head_chunk=32, checkpoint_trunk=True)
fmt = Format((2,), (3,), 5, 0, header_ids=(4,))
prefix = torch.tensor([[10, 11, 12, 13]])
future = torch.tensor([[14]])
answers = {}
grads = {}
for width in (1024, 8):
    m.zero_grad(set_to_none=True)
    actions = torch.zeros((1, 4, width), dtype=torch.long)
    actions[:, :, 0] = 5
    mask = action_mask(actions, 5)
    rollout = Rollouts(actions, mask, torch.zeros_like(actions, dtype=torch.float32), 0, 16)
    packed, lengths, positions = pack_reader(prefix, actions, fmt, future)
    ll = backend.score(prefix, future, rollout, fmt=fmt, horizon=1)
    calls = []
    hook = m.lm_head.register_forward_hook(
        lambda module, inputs, output: calls.append(dict(rows=inputs[0].numel()//24,
                                                        grad_enabled=torch.is_grad_enabled())))
    loss = backend.replay_one_loss(prefix, actions[:, 0], mask[:, 0], torch.tensor(.25),
                                   fmt=fmt, temperature=.7, divisor=4)
    forward_calls = list(calls)
    loss.backward()
    hook.remove()
    grads[width] = {name: param.grad.clone() for name, param in m.named_parameters()}
    answers[str(width)] = dict(active_actions_per_note=1, reader_tensor_width=packed.shape[-1],
                              actual_reader_lengths=lengths.tolist(), target_positions=positions.tolist(),
                              replay_head_forward_calls=len(forward_calls),
                              replay_head_forward_rows=sum(v['rows'] for v in forward_calls),
                              replay_head_rows_including_checkpoint_recompute=sum(v['rows'] for v in calls),
                              loss=float(loss.detach()), reader_logprobs=ll.tolist())
assert torch.allclose(torch.tensor(answers['1024']['reader_logprobs']),
                      torch.tensor(answers['8']['reader_logprobs']), atol=1e-5, rtol=1e-5)
assert abs(answers['1024']['loss']-answers['8']['loss']) < 1e-6
for name in grads[1024]:
    assert torch.allclose(grads[1024][name], grads[8][name], atol=1e-6, rtol=1e-5), name
assert answers['1024']['replay_head_forward_rows'] == 1024
assert answers['8']['replay_head_forward_rows'] == 8
report = dict(status='CPU_EMPTY_NOTE_PADDING_WORK_CONFIRMED_AND_SINGLE_CASE_LOSS_GRADIENT_INVARIANT',
              utc=datetime.now(timezone.utc).isoformat(), cases=answers,
              maximum_parameter_gradient_difference=max(float((grads[1024][k]-grads[8][k]).abs().max()) for k in grads[1024]),
              inspected_source_sha256=hashlib.sha256((R/'train/neuron_backend.py').read_bytes()).hexdigest(),
              fixture='Random tiny two-layer Qwen3 CPU; vocabulary37,hidden24. No released weights or scientific data.',
              hardware_profile_status='NOT_RUN', production_change='NONE',
              limits=['Head work counts establish dense padded replay despite one active action; no timing or Trainium speedup inferred.',
                      'Reader and replay are causal, so removing post-stop padding can preserve this objective. One empty fixture is not full bucket-boundary/mixed-cap qualification.',
                      'No RL restart, new reward, length-normalization change or actual optimization deployed.',
                      'A future implementation must preserve all active actions, stops, behavior logprobs and original decode-work accounting; qualify mixed/capped cases and actual hardware.'])
out = R/'local/analysis/retired-R-empty-padding-v13-20261019.json'
with out.open('x') as stream:
    json.dump(report, stream, indent=2)
print(json.dumps(report, indent=2), flush=True)
