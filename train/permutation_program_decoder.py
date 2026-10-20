"""Existing causal-cache greedy decoder, with a symbol-field stopping parser."""
from train.document_qa_decoder import QADecoder
from train.permutation_program_answers import parse_answer
from train.permutation_program_evaluation import MODES


class ProgramDecoder(QADecoder):
    def decode(self, prompt_ids, *, mode, max_new_tokens=128, prefill_width=256,
               pad_id=151643, stop_ids=(151643,151645)):
        if mode not in MODES:
            raise ValueError('Declared program evaluation mode required')
        return super().decode(prompt_ids, mode='prior' if mode == 'trace' else 'direct',
            max_new_tokens=max_new_tokens, prefill_width=prefill_width,
            pad_id=pad_id, stop_ids=stop_ids,
            answer_parser=lambda text, _: parse_answer(text, mode))
