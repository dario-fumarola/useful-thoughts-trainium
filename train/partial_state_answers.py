"""Prospective complete-number-line parser and cached decoding for arithmetic.

Accept finite signed numbers, grouped commas, decimals and scientific notation.
Plain answers may repeat an Answer label; CoT requires its first Answer label.
Never search for a later correct-looking number. Errors and caps stay failures.
"""
from decimal import Decimal, InvalidOperation
import re

import torch

from train.evaluate import Decoded, GenerationFailure, GreedyDecoder
from train.neuron_backend import CausalCache
from weekend_thinking.core import policy_log_probs

NUMBER = r'[-+−]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?'
LINE = re.compile(r'[ \t]*(' + NUMBER + r')[ \t]*\.?[ \t]*\r?\n')
FINAL = re.compile(r'(' + NUMBER + r')[ \t]*\.?')
LABEL = re.compile(r'(?im)^[ \t]*Answer:[ \t]*(?:\r?\n)?')
PLAIN_LABEL = re.compile(r'(?i)Answer:[ \t]*(?:\r?\n)?')


def answer_start(text, cot=False):
    if cot:
        label = LABEL.search(text)
        if label is None:
            return None
        start = label.end()
    else:
        start = len(text) - len(text.lstrip())
        label = PLAIN_LABEL.match(text, start)
        if label is not None:
            start = label.end()
    return start + len(text[start:]) - len(text[start:].lstrip())


def first_number_line_end(text, cot=False):
    start = answer_start(text, cot)
    match = None if start is None else LINE.match(text, start)
    return None if match is None else match.end()


def parsed_number(text, *, cot=False, eos=False):
    start = answer_start(text, cot)
    if start is None:
        return None
    match = LINE.match(text, start)
    if match is None and eos:
        match = FINAL.fullmatch(text[start:].strip())
    if match is None:
        return None
    try:
        value = Decimal(match.group(1).replace(',', '').replace('−', '-'))
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def score_answer(prediction, gold, *, cot=False):
    stop = prediction['stop_type']
    existing = prediction.get('status', 'ok')
    if stop not in ('number_line', 'token:151643', 'length_cap', 'error') or existing not in ('ok', 'error', 'truncated', 'format_error'):
        raise ValueError('Unknown generation outcome; never infer a successful answer')
    value = parsed_number(prediction['text'], cot=cot, eos=stop == 'token:151643')
    expected = Decimal(str(gold))
    if not expected.is_finite():
        raise ValueError('Finite arithmetic gold required')
    status = (existing if existing != 'ok' else
              'error' if stop == 'error' else
              'truncated' if stop == 'length_cap' else
              'format_error' if value is None else 'ok')
    return dict(score=int(status == 'ok' and value == expected), status=status,
                parsed_number=None if value is None else str(value))


class NumberLineDecoder(GreedyDecoder):
    """Same verified cache/_next/math; a task-specific host stopping predicate.

    Writer notes use inherited decode(answer_line_stop=False). Only answer/CoT
    decoding below stops on a complete numeric line. Native actions/raw text are
    retained even when the final BPE token extends past the accepted boundary.
    """

    @torch.no_grad()
    def decode_number(self, prompt_ids, *, max_new_tokens, prefill_width,
                      pad_id=151643, stop_ids=(151643,), cot=False):
        if not prompt_ids or len(prompt_ids) > prefill_width or max_new_tokens < 1:
            raise ValueError('Invalid prompt/prefill/decode bounds')
        actions, content, raw_text = [], [], ''
        try:
            prompt = torch.full((1, prefill_width), pad_id, dtype=torch.long)
            prompt[0, :len(prompt_ids)] = torch.tensor(prompt_ids, dtype=torch.long)
            prompt = prompt.to(self.device)
            cache = CausalCache(self.model.config.num_hidden_layers, prefill_width + max_new_tokens)
            position = torch.tensor([len(prompt_ids)], device=self.device, dtype=torch.long)
            logits = self._next(prompt, torch.arange(prefill_width, device=self.device), cache, position-1)
            for step in range(max_new_tokens):
                token = policy_log_probs(logits, 1., self.banned_ids).argmax(-1)
                self.sync(); value = int(token.cpu()[0]); actions.append(value)
                if value in stop_ids:
                    return Decoded(actions, content, raw_text, raw_text, f'token:{value}')
                content.append(value)
                raw_text = self.tokenizer.decode(content, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                end = first_number_line_end(raw_text, cot)
                if end is not None:
                    return Decoded(actions, content, raw_text, raw_text[:end], 'number_line')
                if step+1 < max_new_tokens:
                    logits = self._next(token[:, None], position, cache); position = position+1
            return Decoded(actions, content, raw_text, raw_text, 'length_cap')
        except Exception as error:
            raise GenerationFailure(error, Decoded(actions, content, raw_text, raw_text, 'error')) from error
