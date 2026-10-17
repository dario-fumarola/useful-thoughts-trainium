"""Native token placement, teacher exposure, stopping, and reference computation."""
import json
from types import SimpleNamespace

import pytest
import torch

from train import hindsight_teacher as teacher
from weekend_thinking.core import action_mask, content_from_actions, writer_inputs


class Tokenizer:
    eos_token_id, pad_token_id = 151645, 151643
    chat_template = "native-template"

    def get_vocab(self):
        return {"newline": 198, "eos": 151645, "pad": 151643}

    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        return [198] if text == "\n" else [ord(c) for c in text]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert [x["role"] for x in messages] == ["system", "user"]
        return f"SYSTEM:{messages[0]['content']}\nUSER:{messages[1]['content']}\nASSISTANT:<think>\n\n</think>\n\n"


def test_exact_vocab_and_stop_identity_are_required():
    tok = Tokenizer()
    assert teacher.native_vocab(tok, tok)["equal"] is True
    different = Tokenizer(); different.get_vocab = lambda: {"newline": 199, "eos": 151645, "pad": 151643}
    with pytest.raises(ValueError, match="vocab mappings differ"):
        teacher.native_vocab(different, tok)
    tok.pad_token_id = 0
    with pytest.raises(ValueError, match="EOS/pad"):
        teacher.native_vocab(tok, tok)


def test_segmented_chat_has_fixed_native_spans_and_none_has_only_blanks():
    saved = teacher.prompt_parts(Tokenizer()); parts = saved["token_segments"]
    prefix, future = list(range(512)), list(range(4000,4016))
    blank = teacher.build_prompt(prefix, None, parts)
    hindsight = teacher.build_prompt(prefix, future, parts)
    a = len(parts[0]); b = a+512+len(parts[1])
    assert len(blank) == len(hindsight)
    assert blank[a:a+512] == hindsight[a:a+512] == prefix
    assert blank[b:b+16] == [198]*16 and hindsight[b:b+16] == future
    assert blank[:b] == hindsight[:b] and blank[b+16:] == hindsight[b+16:]
    fmt = teacher.teacher_format(parts[-1][-1])
    assert writer_inputs([hindsight[:-1]], fmt) == [hindsight]
    assert fmt.header_ids == ()
    assert "continuation field may be blank" in saved["messages"][0]["content"]
    with pytest.raises(ValueError, match="prefix512"):
        teacher.build_prompt(prefix, future[:1], parts)


def test_both_eos_stop_but_endthink_is_content():
    for stop in (151645,151643):
        a = torch.tensor([[151668, 11, stop, 151643, 151643]])
        assert action_mask(a, 151645, (151643,)).tolist() == [[True,True,True,False,False]]
        assert content_from_actions(a[0].tolist(),151645,(151643,)) == [151668,11]
    assert teacher.teacher_format(198).additional_stop_ids == (151643,)


def test_matched_condition_batch_seed_is_distinct_across_batches():
    ids = [str(i) for i in range(16)]
    seeds = [teacher.batch_seed(ids[i:i+4],i//4) for i in range(0,16,4)]
    assert len(set(seeds)) == 4
    assert teacher.batch_seed(ids[:4],0) == seeds[0]


def test_defined_support_check_excludes_only_undefined_and_rejects_nonfinite():
    actual, reference = torch.zeros(4,8), torch.zeros(4,8)
    actual[:,7] = 100
    assert teacher.next_logp_error(actual,reference,list(range(7))) == 0
    actual[1,0] = .003
    assert teacher.next_logp_error(actual,reference,list(range(7))) > .002
    actual[1,0] = float("nan")
    with pytest.raises(RuntimeError,match="Nonfinite"):
        teacher.next_logp_error(actual,reference,list(range(7)))


def test_cpu_reference_uses_trunk_then_last_position_head_not_full_logits(tmp_path, monkeypatch):
    import transformers
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.embedding = torch.nn.Embedding(8,3)
            self.lm_head = torch.nn.Linear(3,8,bias=False)
            self.config = SimpleNamespace(vocab_size=8); self.head_shapes = []
            self.lm_head.register_forward_pre_hook(lambda _, args:self.head_shapes.append(list(args[0].shape)))
        def tie_weights(self): self.lm_head.weight = self.embedding.weight
        def get_input_embeddings(self): return self.embedding
        def get_output_embeddings(self): return self.lm_head
        @property
        def model(self): return self
        def forward(self, *, input_ids, use_cache):
            assert not use_cache and input_ids.shape == (4,6)
            assert not torch.is_grad_enabled()
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))
    model = Model()
    calls = []
    def load(*args,**kwargs):
        calls.append(kwargs); return model, dict(missing_keys=set(),unexpected_keys=set(),mismatched_keys=set(),error_msgs=[])
    monkeypatch.setattr(transformers.AutoModelForCausalLM,"from_pretrained",load)
    monkeypatch.setattr(transformers.AutoTokenizer,"from_pretrained",lambda *a,**k:SimpleNamespace(get_vocab=lambda:{str(i):i for i in range(7)}))
    p = dict(teacher_dir="local-teacher",native_vocab={"equal":True},selection={"item_ids":[str(i) for i in range(16)]})
    inputs = dict(prompts={c:[[i%7]*6 for i in range(16)] for c in teacher.CONDITIONS})
    monkeypatch.setattr(teacher,"read_protocol",lambda _: (p,inputs))
    (tmp_path/"protocol.json").write_text("{}")
    teacher.cpu_reference(tmp_path,1)
    report = json.loads((tmp_path/"cpu-reference/reference.json").read_text())
    assert report["status"] == "complete" and report["tied"]
    assert report["loading_info"]["missing_keys"] == []
    assert report["loading_info_types"]["missing_keys"] == "set"
    assert torch.load(tmp_path/"cpu-reference/loading-info.pt",weights_only=True)["missing_keys"] == set()
    assert report["shapes"] == {c:[4,8] for c in teacher.CONDITIONS}
    assert model.head_shapes == [[4,3],[4,3]]
    assert calls == [dict(local_files_only=True,trust_remote_code=False,dtype=torch.float32,
                          attn_implementation="eager",output_loading_info=True)]
    with pytest.raises(ValueError,match="sixteen threads"):
        teacher.cpu_reference(tmp_path,17)
