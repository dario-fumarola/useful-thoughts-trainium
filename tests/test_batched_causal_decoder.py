from dataclasses import asdict

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from train.batched_causal_decoder import BatchedGreedyDecoder, BatchedProgramDecoder, BatchGenerationFailure
from train.neuron_backend import CausalCache
from train.permutation_program_decoder import ProgramDecoder


class Tok:
    def decode(self, ids, **kwargs):
        return ''.join(chr(i) for i in ids)


def tiny(vocab=256):
    torch.manual_seed(432)
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=vocab, hidden_size=16, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        eos_token_id=0, pad_token_id=0, attn_implementation='eager')).eval()


@pytest.mark.parametrize('width', [4, 9])
def test_real_causal_batch_matches_singletons_and_uncached_reference(width):
    model = tiny(31)
    prefixes = [[2, 3, 4], [4, 5, 6], [2, 2, 7], [9, 8, 3]]
    batch = BatchedProgramDecoder(model, Tok()).decode_batch(prefixes, mode='trace',
        max_new_tokens=6, prefill_width=width, pad_id=0, stop_ids=(999,))
    for prefix, actual in zip(prefixes, batch):
        singleton = ProgramDecoder(model, Tok()).decode(prefix, mode='trace',
            max_new_tokens=6, prefill_width=width, pad_id=0, stop_ids=(999,))
        assert asdict(actual) == asdict(singleton)
        sequence, expected = list(prefix), []
        for _ in range(6):
            with torch.no_grad():
                value = int(model(torch.tensor([sequence])).logits[0, -1].argmax())
            sequence.append(value)
            expected.append(value)
        assert actual.actions == expected and actual.stop_type == 'length_cap'


def test_every_batched_cached_distribution_matches_ordinary_causal_reference():
    model = tiny(31)
    decoder = BatchedGreedyDecoder(model, Tok())
    sequences = torch.tensor([[2, 3, 4], [4, 5, 6], [2, 2, 7], [9, 8, 3]])
    padded = torch.zeros((4, 9), dtype=torch.long)
    padded[:, :3] = sequences
    cache = CausalCache(2, 15)
    with torch.no_grad():
        cached = decoder._next(padded, torch.arange(9), cache, torch.tensor([2]))
        for position in range(3, 9):
            ordinary = model(sequences).logits[:, -1]
            assert torch.allclose(cached.log_softmax(-1), ordinary.log_softmax(-1), atol=1e-6, rtol=0)
            token = ordinary.argmax(-1)[:, None]
            sequences = torch.cat((sequences, token), dim=1)
            cached = decoder._next(token, torch.tensor([position]), cache)


def scripted(sequences, *, banned=(), fail_at=None):
    class Scripted(BatchedProgramDecoder):
        def __init__(self):
            super().__init__(tiny(), Tok(), banned_ids=banned)
            self.step = 0

        def _next(self, ids, *args, **kwargs):
            if fail_at == self.step:
                raise RuntimeError('injected shared execution failure')
            logits = torch.full((len(sequences), 256), -100.)
            for row, sequence in enumerate(sequences):
                logits[row, sequence[min(self.step, len(sequence)-1)]] = 100.
            self.step += 1
            return logits
    return Scripted()


def test_uneven_eos_answer_and_cap_preserve_each_rows_exact_native_actions():
    decoder = scripted([[0, 81, 82], list(b'B\nignored'), list(b'xxxxxx')])
    values = decoder.decode_batch([[1, 2]] * 3, mode='direct', max_new_tokens=5,
                                  prefill_width=8, pad_id=0, stop_ids=(0,))
    assert values[0].actions == [0] and values[0].content_ids == []
    assert values[0].stop_type == 'token:0'
    assert values[1].actions == list(b'B\n') and values[1].text == 'B\n'
    assert values[1].stop_type == 'qa_answer_line'
    assert values[2].actions == list(b'xxxxx') and values[2].stop_type == 'length_cap'


def test_trace_stops_on_complete_answer_field_not_an_intermediate_node():
    text = b'r: A -> B\nFinal answer:\nB\n'
    values = scripted([list(text + b'ignored'), list(b'Final answer:\nC\nignored')]).decode_batch(
        [[1, 2]] * 2, mode='trace', max_new_tokens=100, prefill_width=8, pad_id=0, stop_ids=(0,))
    assert values[0].actions == list(text)
    assert values[1].actions == list(b'Final answer:\nC\n')


def test_shared_failure_keeps_completed_and_partial_rows_in_original_order():
    decoder = scripted([[0], list(b'abcd')], fail_at=2)
    with pytest.raises(BatchGenerationFailure) as caught:
        decoder.decode_batch([[1, 2]] * 2, mode='trace', max_new_tokens=8,
                             prefill_width=8, pad_id=0, stop_ids=(0,))
    values = caught.value.partials
    assert values[0].actions == [0] and values[0].stop_type == 'token:0'
    assert values[1].actions == list(b'ab') and values[1].stop_type == 'error'


def test_banned_tokens_remain_masked_in_every_batch_row():
    decoder = scripted([[65], [65]], banned=(65,))
    values = decoder.decode_batch([[1, 2]] * 2, mode='trace', max_new_tokens=2,
                                  prefill_width=8, pad_id=0, stop_ids=(0,))
    assert all(value.actions == [0] for value in values)


@pytest.mark.parametrize('prefixes,width,cap', [([], 8, 4), ([[]], 8, 4),
    ([[1], [1, 2]], 8, 4), ([[1, 2]], 1, 4), ([[1, 2]], 8, 0)])
def test_invalid_or_ragged_batches_rejected_before_any_model_call(prefixes, width, cap):
    decoder = BatchedGreedyDecoder(tiny(), Tok())
    with pytest.raises(ValueError):
        decoder.decode_batch(prefixes, prefill_width=width, max_new_tokens=cap)
