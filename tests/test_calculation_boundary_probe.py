"""CPU fixtures: source/span integrity, native gate, one-pass metrics and closure."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from train import calculation_boundaries as curator
from train import calculation_boundary_probe as probe
from train.evaluate_likelihood import ContinuationScore
from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file

DEADLINE = '2100-01-01T00:00:00+00:00'
SIGNATURE = {'parameters':1720574976, 'sampled_parameter_sha256':'CPU-fixture-not-hardware-evidence'}


class Tokenizer:
    eos_token_id = 151643
    def get_vocab(self):return {**{str(i):i for i in range(512)}, '<eos>':151643}
    def encode(self,text,**kwargs):
        result=[]
        while text:
            if text.startswith('\n\n'):result.append(271);text=text[2:]
            else:result.append(ord(text[0]));text=text[1:]
        return result
    def decode(self,ids,**kwargs):return ''.join('\n\n' if t==271 else chr(t) for t in ids)


class TinyTied(torch.nn.Module):
    """Real CPU tensors/head math, deliberately not actual Qwen weights."""
    def __init__(self):
        super().__init__();self.embedding=torch.nn.Embedding(151936,1)
        self.head=torch.nn.Linear(1,151936,bias=False);self.head.weight=self.embedding.weight
        torch.nn.init.zeros_(self.head.weight)
        self.model=lambda input_ids,**kw:SimpleNamespace(last_hidden_state=torch.zeros(*input_ids.shape,1))
        self.lm_head=self.head
        self.config=SimpleNamespace(num_hidden_layers=1,model_type='qwen3',layer_types=['full_attention'])
    def get_input_embeddings(self):return self.embedding
    def get_output_embeddings(self):return self.head


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    import transformers
    old_threads=torch.get_num_threads()
    tok=Tokenizer();directory=tmp_path/'dataset';directory.mkdir();source=tmp_path/'source.records.jsonl'
    source.write_text('Frozen synthetic source bytes, no corpus access.\n')
    source_hash=sha256_file(source);monkeypatch.setattr(curator,'SOURCE_SHA',source_hash)
    prefix=tok.encode('context '*20+'\n3*2+4 =');cut=len(prefix)
    full=prefix+tok.encode(' 10. source continuation.');full+=tok.encode(' ')*(528-len(full))
    good=tok.encode('Compute 3*2 = 6.');bad=tok.encode('Compute 3*2 = 7.');accepted=[]
    for split,n in probe.COUNTS.items():
        for i in range(n):
            key=f'{split}-{i}';pair=dict(correct_ids=good,corrupted_ids=bad,
                correct_text=tok.decode(good),corrupted_text=tok.decode(bad))
            rhs=dict(token_ids=full[cut:cut+3],token_count=3,serialized_text=' 10',value='10',numeric_token_mask=[False,True,True])
            accepted.append(dict(item_id=key,document_id='doc-'+key,split=split,source_file=str(source),
                source_sha256=source_hash,source_record_id='record-'+key,ordinary_source_ids=full,
                original_prefix_ids=full[:512],original_future_ids=full[512:],prefix_ids=prefix,
                writer_prefix_ids=prefix,writer_prefix_text=tok.decode(prefix),prefix_text=tok.decode(prefix),
                natural_continuation_ids=full[cut:],natural_future16_ids=full[cut:cut+16],
                numeric_target_ids=full[cut:cut+3],numeric_token_mask=[False,True,True],verified_numeric_rhs=rhs,
                note_ids=good,note_text=tok.decode(good),corrupted_note_ids=bad,corrupted_note_text=tok.decode(bad),
                boundary=dict(cut=cut,note=pair)))
    (directory/'accepted.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in accepted))
    (directory/'audit.jsonl').write_text('{"status":"fixture_only_no_model_selection"}\n')
    selection={}
    for split,n in probe.COUNTS.items():
        ordered=sorted((r for r in accepted if r['split']==split),key=lambda r:canonical_hash(dict(
            salt=curator.SALT,split=split,document_id=r['document_id'],item_id=r['item_id'])))
        selection[split]=dict(requested=n,available=n,selected_item_ids=[r['item_id'] for r in ordered],reserve_item_ids=[])
    root=Path(probe.__file__).resolve().parents[1]
    manifest=dict(schema=curator.SCHEMA,source_sha256=source_hash,source_file=str(source),selection_salt=curator.SALT,
        ready_for_feasibility=True,tokenizer_vocab_sha256=canonical_hash(tok.get_vocab()),selection=selection,
        source_files={'train/calculation_boundaries.py':sha256_file(root/'train/calculation_boundaries.py')},
        accepted_sha256=sha256_file(directory/'accepted.jsonl'),audit_sha256=sha256_file(directory/'audit.jsonl'),limits=['Fixture only.'])
    atomic_json(directory/'manifest.json',manifest)
    model_dir=tmp_path/'model';model_dir.mkdir();atomic_json(model_dir/'ASSET_MANIFEST.json',{'fixture':True})
    atomic_json(model_dir/'config.json',{'vocab_size':151936})
    assets=dict(model_id=probe.MODEL,revision=probe.REVISION,manifest_sha256=sha256_file(model_dir/'ASSET_MANIFEST.json'))
    def provenance(path):
        assert isinstance(path,Path)  # Catches the real helper's Path-only API contract.
        return assets
    monkeypatch.setattr(probe,'asset_provenance',provenance)
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**kw:tok)
    model=TinyTied();monkeypatch.setattr(probe,'load_base',lambda p:(model,tok))
    monkeypatch.setattr(probe,'sampled_fingerprint',lambda model:SIGNATURE)
    backend=SimpleNamespace(model=model,device=torch.device('cpu'),sync=lambda:None,
        banned_ids=tuple(sorted(set(range(151936))-set(tok.get_vocab().values()))),
        hidden=lambda ids:model.model(input_ids=ids).last_hidden_state)
    monkeypatch.setattr(probe,'setup_model',lambda args:(backend,tok,None,SIGNATURE,None))
    output=tmp_path/'probe';p=probe.prepare(output,directory,model_dir,DEADLINE)
    yield SimpleNamespace(output=output,directory=directory,p=p,tok=tok,model=model,accepted=accepted,manifest=manifest)
    torch.set_num_threads(old_threads)


def reseal_dataset(f,rows):
    path=f.directory/'accepted.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    m=probe.read(f.directory/'manifest.json');m['accepted_sha256']=sha256_file(path)
    atomic_json(f.directory/'manifest.json',m,overwrite=True)


def test_preparation_freezes_complete_inventory_and_both_native_context_roles(prepared):
    p,x=probe.load(prepared.output)
    assert len(x['rows'])==320 and p['expected_rows']==1920
    assert p['answers']==probe.answer_recipe(DEADLINE) and p['answers']['expected_rows']==384
    assert [(r['split'],r['mode']) for r in x['qualification_refs']]==[
        ('train','raw'),('train','correct'),('dev','raw'),('dev','correct')]
    for r in x['rows']:
        c=x['contexts'][r['item_id']];past=r['writer_prefix_ids']
        assert c['values']['raw']==past and c['values']['empty']==[271]+past
        assert c['values']['correct']==r['note_ids']+[271]+past
        donor=next(q for q in x['rows'] if q['item_id']==c['donor_item_id'])
        assert donor['split']==r['split'] and donor['item_id']!=r['item_id']
        assert len(c['values']['other'])==len(c['values']['correct'])
        assert sorted(c['values']['shuffled'][:-len(past)-1])==sorted(r['note_ids'])
    assert not (prepared.output/'cpu-reference').exists()


@pytest.mark.parametrize('damage',['short_split','duplicate_document','numeric_not_prefix','mask','causal_cut','missing_curator_hash','audit_bytes','source_bytes'])
def test_resealed_schema_or_source_damage_never_silently_shrinks_or_relabels(prepared,damage):
    values=copy.deepcopy(prepared.accepted);m=probe.read(prepared.directory/'manifest.json')
    if damage=='short_split':values.pop()
    elif damage=='duplicate_document':values[1]['document_id']=values[0]['document_id']
    elif damage=='numeric_not_prefix':values[0]['numeric_target_ids']=[49,48]
    elif damage=='mask':values[0]['numeric_token_mask']=[True,True,True]
    elif damage=='causal_cut':values[0]['writer_prefix_ids']=values[0]['writer_prefix_ids'][:-1]
    elif damage=='missing_curator_hash':
        m['source_files']={};atomic_json(prepared.directory/'manifest.json',m,overwrite=True)
    elif damage=='audit_bytes':(prepared.directory/'audit.jsonl').write_text('changed')
    else:Path(m['source_file']).write_text('changed source')
    if damage in ('short_split','duplicate_document','numeric_not_prefix','mask','causal_cut'):reseal_dataset(prepared,values)
    with pytest.raises(ValueError):probe.dataset(prepared.directory,prepared.tok)


def test_one_natural_vector_separates_whitespace_scalar_and_full_future():
    a=[-100.,-1.,-2.]+[-3.]*13;b=[-1000.,-1.,-2.]+[-3.]*13
    first=probe.derive_metrics(a,[False,True,True]);second=probe.derive_metrics(b,[False,True,True])
    assert first['numeric_nonwhitespace_ll']==second['numeric_nonwhitespace_ll']==-3
    assert first['scalar_joint_ll']==-103 and second['scalar_joint_ll']==-1003
    assert first['natural16_ll']==sum(a)
    with pytest.raises(ValueError):probe.derive_metrics(a,[False]*3)
    with pytest.raises(ValueError):probe.derive_metrics([float('nan')]+a[1:],[False,True,True])


def test_cpu_reference_executes_real_tiny_tied_head_and_publishes_receipt(prepared):
    probe.cpu_reference(prepared.output,1,DEADLINE)
    p,x=probe.load(prepared.output);r,t=probe.reference(prepared.output,p,x)
    assert r['tied'] and t.dtype==torch.float32 and t.shape==(4,151936) and torch.isfinite(t).all()
    assert r['refs']==x['qualification_refs'] and len(r['forward_wall_seconds'])==4
    assert torch.allclose(t,torch.full_like(t,-torch.log(torch.tensor(151936.)).item()))


def fake_score(backend,context,targets,**kwargs):
    assert len(targets)==1 and len(targets[0])==16 and kwargs['future_width']==16
    lp=[-100.,-1.,-2.]+[-3.]*13
    return [ContinuationScore(sum(lp),sum(lp)/16,lp,list(targets[0]),list(targets[0]))]


def test_actual_forward_gate_precedes_any_scoring_and_preserves_failure(prepared,monkeypatch):
    probe.cpu_reference(prepared.output,1,DEADLINE);d=prepared.output/'cpu-reference'
    t=torch.load(d/'next-logps.pt',weights_only=True);t[0,0]+=.01;torch.save(t,d/'next-logps.pt')
    r=probe.read(d/'reference.json');r['logps_sha256']=sha256_file(d/'next-logps.pt');atomic_json(d/'reference.json',r,overwrite=True)
    monkeypatch.setattr(probe,'score_continuations',lambda *a,**k:pytest.fail('Scored before failed qualification'))
    with pytest.raises(RuntimeError,match='gate failed'):probe.score(prepared.output,0,DEADLINE)
    run=prepared.output/'scores/shard-0'
    assert probe.read(run/'run.json')['status']=='failed' and (run/'forward-logps.pt').exists()
    assert not (run/'scores.jsonl').exists()


def test_single_natural_pass_per_cell_and_whitespace_mask_used(prepared,monkeypatch):
    probe.cpu_reference(prepared.output,1,DEADLINE);calls=[]
    def count(*args,**kwargs):calls.append(1);return fake_score(*args,**kwargs)
    monkeypatch.setattr(probe,'score_continuations',count);probe.score(prepared.output,0,DEADLINE)
    run=prepared.output/'scores/shard-0';values=probe.rows(run/'scores.jsonl')
    assert len(calls)==len(values)==480 and probe.read(run/'run.json')['status']=='complete'
    assert all(v['numeric_nonwhitespace_ll']==-3 and v['scalar_joint_ll']==-103 and v['natural16_ll']==-142 for v in values)


@pytest.fixture
def closed(prepared,monkeypatch):
    probe.cpu_reference(prepared.output,1,DEADLINE);monkeypatch.setattr(probe,'score_continuations',fake_score)
    for shard in range(4):probe.score(prepared.output,shard,DEADLINE)
    return prepared


def test_complete_summary_keeps_splits_metrics_absolute_means_and_document_cis(closed):
    result=probe.summarize(closed.output)
    assert result['rows']==1920 and len(result['score_rows'])==1920
    for split,n in probe.COUNTS.items():
        for metric in probe.METRICS:
            r=result['by_split'][split][metric]
            assert set(r['absolute_means'])==set(probe.MODES)
            assert all(v['documents']==v['items']==n and v['difference_b_minus_a']==0 for v in r['contrasts'].values())


@pytest.mark.parametrize('damage',['duplicate','nonfinite','numeric_metric','target','gate','running'])
def test_summary_refuses_resealed_cell_or_gate_damage(closed,damage):
    d=closed.output/'scores/shard-2';v=probe.rows(d/'scores.jsonl');r=probe.read(d/'run.json')
    if damage=='duplicate':v[-1]=v[0]
    elif damage=='nonfinite':v[0]['token_log_likelihoods'][0]=float('nan')
    elif damage=='numeric_metric':v[0]['numeric_nonwhitespace_ll']+=1
    elif damage=='target':v[0]['target_token_ids'][0]+=1
    elif damage=='gate':r['forward_check']['max_error']=.01
    else:r['status']='running'
    (d/'scores.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in v));r['scores_sha256']=sha256_file(d/'scores.jsonl')
    atomic_json(d/'run.json',r,overwrite=True)
    with pytest.raises(ValueError):probe.summarize(closed.output)
    assert not (closed.output/'summary.json').exists()


@pytest.mark.parametrize('text,stop,value',[
    ('  +10',f'token:{probe.PAD}','10'),('-0.5',f'token:{probe.PAD}','-1/2'),
    (' +2 / -4',f'token:{probe.PAD}','-1/2'),('.50. Thus more follows','length_cap','1/2'),
    ('10, with later unrelated continuation','length_cap','10'),('10\nThe next step follows','length_cap','10'),
    ('10.',f'token:{probe.PAD}','10'),('10   ',f'token:{probe.PAD}','10'),
    ('8.3/2',f'token:{probe.PAD}','83/20'),('4.15',f'token:{probe.PAD}','83/20')])
def test_initial_scalar_exact_equivalence_and_genuine_delimiters(text,stop,value):
    assert probe.parse_initial_scalar(text,stop)['value']==value


@pytest.mark.parametrize('text',[
    '10','10 ','10.','10,','+','-', '.',''])
def test_cap_cannot_certify_unterminated_scalar(text):
    assert probe.parse_initial_scalar(text,'length_cap')['status']=='incomplete'


@pytest.mark.parametrize('text',[
    '10+2','10 - 2','10*2','10 /','10/0','10/2/3','10^2','10e2','10E-2','10 1/2',
    '10 0.5','10,000','10, 000','10.2.3','10%','10 x','10 times 2','10 over 2','10 or 11',
    '10 modulo 3','10 squared','10 cubed','10 percent','10 percentage','10 raised to 2',
    '10 dividing 3','10 multiplied by 3','10 factorial','10 sqrt 3','10 powers of 2',
    '10 to 11','10×2','10 ÷ 2','10−2','10²','10_2','10\\times2','10\\frac{1}{2}',
    '10 : 2','10 = 2','10 < 11','10(2)','10. e+2','10. 2','１０','−10','The answer is 10',
    '__import__("os")', '1'*65,'1/0.0'])
def test_expression_prefix_unicode_or_unbounded_syntax_never_scores(text):
    assert probe.parse_initial_scalar(text,f'token:{probe.PAD}')['status']!='valid'
    assert probe.parse_initial_scalar(text,'length_cap')['status']!='valid'


class FakeGreedy:
    """Exercises the actual endpoint orchestration, not accelerator evidence."""
    calls=[]
    def __init__(self,model,tok,**kwargs):self.model=model;self.tok=tok
    def _next(self,ids,positions,cache,selected_position):
        assert ids.shape==(1,probe.WIDTH) and cache.holder['width']==probe.WIDTH+32
        assert selected_position.shape==(1,)
        return self.model.lm_head(torch.zeros(1,1))
    def decode(self,context,**kwargs):
        from train.evaluate import Decoded,GenerationFailure
        assert kwargs==dict(max_new_tokens=32,prefill_width=1024,pad_id=151643,stop_ids=(151643,),answer_line_stop=False)
        self.calls.append(context)
        # Raw is an equivalent fraction; correct is an unrelated continuation capped after a valid number.
        if context[0]!=271 and context[0]!=ord('C'):
            text='20/2';ids=self.tok.encode(text);return Decoded(ids+[probe.PAD],ids,text,text,f'token:{probe.PAD}')
        if context[0]==271:
            ids=self.tok.encode('10');raise GenerationFailure(RuntimeError('retained fixture failure'),Decoded(ids,ids,'10','10','error'))
        text='10. Further prose follows'.ljust(32);assert len(text)==32
        ids=self.tok.encode(text);return Decoded(ids,ids,text,text,'length_cap')


@pytest.fixture
def answered(prepared,monkeypatch):
    FakeGreedy.calls=[];probe.cpu_reference(prepared.output,1,DEADLINE)
    monkeypatch.setattr(probe,'GreedyDecoder',FakeGreedy)
    for shard in range(4):probe.answers(prepared.output,shard,DEADLINE)
    return prepared


def test_answer_endpoint_all384_exact_contexts_raw_actions_errors_and_document_cis(answered):
    result=probe.summarize_answers(answered.output);p,x=probe.load(answered.output)
    assert result['rows']==len(FakeGreedy.calls)==384 and result['absolute']['raw']['correct']==64
    assert result['absolute']['correct']['correct']==64 and result['absolute']['empty']['statuses']['error']==64
    assert all(v['denominator']==sum(v['statuses'].values())==64 for v in result['absolute'].values())
    expected=[x['contexts'][q['item_id']]['values'][m] for shard in range(4)
        for q in [r for r in x['rows'] if r['split']=='dev'][shard::4] for m in probe.MODES]
    assert FakeGreedy.calls==expected
    for q in result['answer_rows']:
        assert 'raw_action_text' in q and q['score'] in (0,1)
        if q['status']=='error':assert q['actions']==[49,48] and q['score']==0 and q['parsed']=={'status':'error'}
    assert all(v['documents']==v['items']==64 for v in result['contrasts'].values())
    assert not (answered.output/'summary.json').exists()  # Separate endpoint, no LL replacement.


@pytest.mark.parametrize('damage',['missing','duplicate','context','parser','stop','raw_text','support','gate','running'])
def test_free_value_summary_requires_all384_identity_and_actual_gates(answered,damage):
    d=answered.output/'answers/shard-0';v=probe.rows(d/'predictions.jsonl');r=probe.read(d/'run.json')
    if damage=='missing':v.pop();r['rows']-=1
    elif damage=='duplicate':v[-1]=v[0]
    elif damage=='context':v[0]['context_sha256']='changed'
    elif damage=='parser':v[0]['parsed']['value']='99'
    elif damage=='stop':v[0]['stop_type']='length_cap'
    elif damage=='raw_text':v[0]['raw_action_text']='changed'
    elif damage=='support':r['banned_ids_sha256']='changed'
    elif damage=='gate':r['forward_check']['max_error']=.01
    else:r['status']='running'
    (d/'predictions.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in v));r['predictions_sha256']=sha256_file(d/'predictions.jsonl')
    atomic_json(d/'run.json',r,overwrite=True)
    with pytest.raises(ValueError):probe.summarize_answers(answered.output)
    assert not (answered.output/'answers.summary.json').exists()


def test_cached_forward_gate_fails_before_any_generation_and_retains_vector(prepared,monkeypatch):
    probe.cpu_reference(prepared.output,1,DEADLINE);d=prepared.output/'cpu-reference'
    t=torch.load(d/'next-logps.pt',weights_only=True);t[3,0]+=.01;torch.save(t,d/'next-logps.pt')
    r=probe.read(d/'reference.json');r['logps_sha256']=sha256_file(d/'next-logps.pt');atomic_json(d/'reference.json',r,overwrite=True)
    FakeGreedy.calls=[];monkeypatch.setattr(probe,'GreedyDecoder',FakeGreedy)
    with pytest.raises(RuntimeError,match='gate failed'):probe.answers(prepared.output,3,DEADLINE)
    run=prepared.output/'answers/shard-3'
    assert not FakeGreedy.calls and not (run/'predictions.jsonl').exists()
    assert probe.read(run/'run.json')['status']=='failed' and (run/'forward-logps.pt').exists()


def test_frozen_answer_deadline_cannot_be_extended(prepared):
    probe.cpu_reference(prepared.output,1,DEADLINE)
    with pytest.raises(ValueError,match='Frozen answer deadline'):probe.answers(prepared.output,0,'2100-01-02T00:00:00+00:00')
    assert not (prepared.output/'answers').exists()
