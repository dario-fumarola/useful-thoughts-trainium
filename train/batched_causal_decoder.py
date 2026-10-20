"""Equal-length batched greedy decoding using the existing causal cache.

This candidate is separate from the frozen V28 evaluator. Batches must have
equal real prefix lengths so the existing shared cache positions remain valid.
Finished rows stay in the batch but their later tokens are discarded. No gold,
future answer, or content-based selection enters the decoding interface.
"""
from __future__ import annotations

import torch

from train.evaluate import Decoded, GreedyDecoder
from train.neuron_backend import CausalCache, causal_mask
from train.permutation_program_answers import parse_answer
from train.permutation_program_evaluation import MODES
from weekend_thinking.core import policy_log_probs


class BatchGenerationFailure(RuntimeError):
    """Retain all finished and partial rows if a shared execution fails."""
    def __init__(self, error, partials):
        super().__init__(f'{type(error).__name__}: {error}')
        self.partials = partials


class BatchedGreedyDecoder(GreedyDecoder):
    def _next(self, ids, positions, cache, selected_position=None):
        cache.set_positions(positions)
        with self.amp():
            hidden = self.model.model(input_ids=ids, position_ids=positions[None],
                attention_mask={'full_attention': causal_mask(positions, cache.holder['width'])},
                past_key_values=cache, use_cache=True).last_hidden_state
            if selected_position is None:
                hidden = hidden[:, -1]
            else:
                index = selected_position.reshape(1, 1, 1).expand(ids.shape[0], 1, hidden.shape[-1])
                hidden = hidden.gather(1, index)[:, 0]
        with self.head_amp():
            return self.model.lm_head(hidden).float()

    @torch.no_grad()
    def decode_batch(self, prompt_ids, *, max_new_tokens, prefill_width,
                     pad_id=151643, stop_ids=(151643, 151645), answer_parser=None):
        lengths = {len(ids) for ids in prompt_ids}
        if (not prompt_ids or len(lengths) != 1 or min(lengths) < 1
                or max(lengths) > prefill_width or max_new_tokens < 1):
            raise ValueError('A nonempty equal-real-length batch and valid decoding bounds are required')
        size, length = len(prompt_ids), next(iter(lengths))
        actions = [[] for _ in prompt_ids]
        content = [[] for _ in prompt_ids]
        raw = ['' for _ in prompt_ids]
        done = [None for _ in prompt_ids]

        def snapshot(row, stop, text=None):
            return Decoded(list(actions[row]), list(content[row]), raw[row],
                           raw[row] if text is None else text, stop)

        try:
            # Host construction keeps XLA input shapes independent of real length.
            prompt = torch.full((size, prefill_width), pad_id, dtype=torch.long)
            prompt[:, :length] = torch.tensor(prompt_ids, dtype=torch.long)
            cache = CausalCache(self.model.config.num_hidden_layers, prefill_width + max_new_tokens)
            position = torch.tensor([length], device=self.device, dtype=torch.long)
            logits = self._next(prompt.to(self.device), torch.arange(prefill_width, device=self.device),
                                cache, position - 1)
            for step in range(max_new_tokens):
                token = policy_log_probs(logits, 1., self.banned_ids).argmax(-1)
                self.sync()
                values = token.cpu().tolist()
                for row, value in enumerate(values):
                    if done[row] is not None:
                        continue
                    actions[row].append(value)
                    if value in stop_ids:
                        done[row] = snapshot(row, f'token:{value}')
                        continue
                    content[row].append(value)
                    raw[row] = self.tokenizer.decode(content[row], skip_special_tokens=False,
                                                     clean_up_tokenization_spaces=False)
                    if answer_parser is not None:
                        parsed = answer_parser(raw[row])
                        if parsed.status == 'complete':
                            done[row] = snapshot(row, 'qa_answer_line', raw[row][:parsed.end])
                if all(value is not None for value in done):
                    return done
                if step + 1 < max_new_tokens:
                    # Already finished rows remain allocated. Transformer batches
                    # never attend across rows, so they cannot affect live rows.
                    logits = self._next(token[:, None], position, cache)
                    position = position + 1
            return [value if value is not None else snapshot(row, 'length_cap')
                    for row, value in enumerate(done)]
        except Exception as error:
            partials = [value if value is not None else snapshot(row, 'error')
                        for row, value in enumerate(done)]
            raise BatchGenerationFailure(error, partials) from error


class BatchedProgramDecoder(BatchedGreedyDecoder):
    def decode_batch(self, prompt_ids, *, mode, max_new_tokens=128,
                     prefill_width=256, pad_id=151643, stop_ids=(151643, 151645)):
        if mode not in MODES:
            raise ValueError('Declared program evaluation mode required')
        return super().decode_batch(prompt_ids, max_new_tokens=max_new_tokens,
            prefill_width=prefill_width, pad_id=pad_id, stop_ids=stop_ids,
            answer_parser=lambda text: parse_answer(text, mode))
