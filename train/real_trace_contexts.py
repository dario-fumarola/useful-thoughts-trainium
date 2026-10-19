"""Prefix-only real-document calculation prompts and native reader contexts."""
from train.real_trace_packing import OPEN, ANSWER, encode

EOS = 151643


def question_projection(row):
    """Read exactly the identifiers and native prefix, never future/label fields."""
    q = {key: row[key] for key in ('item_id','document_id','prefix_ids')}
    q['prefix_ids'] = list(q['prefix_ids'])
    if (not all(isinstance(q[k],str) and q[k] for k in ('item_id','document_id'))
            or not 128 <= len(q['prefix_ids']) <= 512
            or any(type(t) is not int or t < 0 or t == EOS for t in q['prefix_ids'])):
        raise ValueError('Require exact native prefix and document identifiers')
    return q


def markers(tokenizer):
    opening, closing = encode(tokenizer,OPEN), encode(tokenizer,ANSWER)
    if (not opening or not closing or EOS in opening+closing
            or tokenizer.decode(closing,skip_special_tokens=False,
                clean_up_tokenization_spaces=False) != ANSWER):
        raise ValueError('Exact native calculation and answer boundaries required')
    return opening,closing


def writer_prompt(question,tokenizer):
    opening,_ = markers(tokenizer)
    return question['prefix_ids']+opening


def reader_context(question,tokenizer,note=None):
    """None is raw; an empty list is an actual empty-format control.

    A generated boundary's first token is excluded by GreedyDecoder. Append
    the same full fixed boundary for every conditioned context, including
    capped/EOS notes. This forced closure and all actions must be reported.
    """
    if note is None:
        return list(question['prefix_ids'])
    if any(type(t) is not int or t < 0 or t == EOS for t in note):
        raise ValueError('Note must contain intact native content, excluding sampled stop')
    opening,closing = markers(tokenizer)
    return question['prefix_ids']+opening+list(note)+closing


def boundary_stop_ids(tokenizer):
    """The declared first-native-boundary-token rule, not substring matching."""
    _,closing = markers(tokenizer)
    return EOS,closing[0]
