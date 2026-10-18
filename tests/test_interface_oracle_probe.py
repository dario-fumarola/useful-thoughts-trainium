"""CPU contracts for the target-copy plumbing batch; no model evidence."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from train import interface_oracle_probe as probe
from train.evaluate_likelihood import ContinuationScore
from weekend_thinking.artifacts import atomic_json,canonical_hash,sha256_file

UNTIL='2100-01-01T00:00:00+00:00'
SIGNATURE=dict(parameters=1720574976,sampled_parameter_sha256='synthetic-not-model-evidence')

class Tokenizer:
    def get_vocab(self):return {**{chr(i):i for i in range(256)},'<think>':151667,'</think>':151668,'<eos>':151643}
    def encode(self,text,**kw):
        result=[]
        while text:
            if text.startswith('<think>'):result.append(151667);text=text[7:]
            elif text.startswith('</think>'):result.append(151668);text=text[8:]
            else:result.append(ord(text[0]));text=text[1:]
        return result
    def decode(self,ids,**kw):return ''.join({151667:'<think>',151668:'</think>',151643:'<eos>'}.get(i,chr(i)) for i in ids)


def general(i=0):
    return dict(cohort='general',split='dev',item_id=f'g{i}',document_id=f'gd{i}',prefix_ids=[80]*512,
        target_ids=[32]+[48+i%10]*15,copy_note_ids=[32]+[48+i%10]*15,reanchor_ids=[80]*24,primary_mask=[True]*16)


def arithmetic(i=0):
    t=Tokenizer();prefix='Source past: '+f'2+{i}=';target=f' {i+2}.';note=f' {i+2}';bad=f' {i+3}'
    return dict(cohort='arithmetic',split='dev',item_id=f'a{i}',document_id=f'ad{i}',prefix_text=prefix,
        prefix_ids=t.encode(prefix),target_ids=t.encode(target),copy_note_text=target,copy_note_ids=t.encode(target),correct_note_ids=t.encode(target),correct_note_text=target,
        corrupt_note_ids=t.encode(bad+'.'),corrupt_note_text=bad+'.',numeric_value=str(i+2),corrupt_numeric_value=str(i+3),
        closing_boundary_ids=t.encode('.'),numeric_token_mask=[c.isdigit() for c in target],closing_boundary_token_start=len(target)-1,
        reanchor_ids=t.encode(f'2+{i}='),reanchor_text=f'2+{i}=',boundary=dict(expression=f'2+{i}',prefix_ids=t.encode(prefix),reanchor_text=f'2+{i}='),primary_mask=[True]*len(target))


def format_files(tmp_path):
    tok=Tokenizer();fmt=tmp_path/'format.json';published=tmp_path/'published.json'
    atomic_json(fmt,dict(header_text='Old header\n',format=dict(header_ids=tok.encode('Old header\n'),
        open_ids=tok.encode('\n<think>'),close_ids=tok.encode('</think>\n'))))
    text='Think between </think> and </think>, after </think> continue.'
    atomic_json(published,dict(prompt_text=text,transcription='”'+text+'”'))
    return fmt,published


def test_six_exact_serializations_and_matched_empty(tmp_path):
    tok=Tokenizer();f,p=format_files(tmp_path);fmt=probe.read(f)
    parts=probe.serialization(tok,dict(fmt['format'],header_text=fmt['header_text']),probe.read(p))
    row=arithmetic();note=tok.encode(' 2');past=tok.decode(row['prefix_ids']);anchor=row['reanchor_text']
    expected={'I0':'Old header\n'+past+'\n<think> 2</think>\n',
        'I1':past+'\n<think> 2</think>\n','I2':past+' <think> 2</think> ',
        'I3':past+'\n<think> 2</think>\n'+anchor,
        'I4':probe.read(p)['prompt_text']+'\n\n'+past+'\n<think> 2</think>\n',
        'I5':past+'\n(scratch:  2)\n'+anchor}
    for interface in probe.INTERFACES:
        context=probe.reader_context(row,note,interface,parts)
        assert tok.decode(context)==expected[interface]
        # Removing the exact inserted note retains all wrapper/reanchor tokens.
        empty=tok.decode(probe.reader_context(row,[],interface,parts))
        assert empty==expected[interface].replace(' 2</think>','</think>').replace(' 2)\n',')\n')
    assert probe.reader_context(row,[999999],'raw',parts)==row['prefix_ids']
    assert probe.reader_context(row,note,'I2',parts)[-1]==ord(' ')
    with pytest.raises(ValueError):probe.reader_context(row,note,'not-an-interface',parts)


def test_publication_typo_and_decorative_quotes_are_explicit(tmp_path):
    f,p=format_files(tmp_path);tok=Tokenizer();fmt=probe.read(f);published=probe.read(p)
    for mutation in (dict(prompt_text='"'+published['prompt_text']),dict(prompt_text=published['prompt_text'].replace('</think>','<think>',1))):
        with pytest.raises(ValueError):probe.serialization(tok,dict(fmt['format'],header_text=fmt['header_text']),dict(published,**mutation))


def test_global_donors_raw_once_and_short_shuffle_truthful():
    selected=[general(i) for i in range(256)]+[arithmetic(i) for i in range(256)]
    controls=probe.make_controls(selected)
    for r in selected:
        c=controls[f'{r["cohort"]}:{r["item_id"]}'];v=c['values']
        assert c['other_donor_item_id']!=r['item_id']
        assert v['correct']==r['copy_note_ids'] and v['empty']==[]
        assert sorted(v['shuffled'])==sorted(v['correct']) and len(v['other'])==len(v['correct'])
        if r['cohort']=='general':
            assert c['corrupt_donor_item_id']!=c['other_donor_item_id']
            assert len(v['corrupt'])==16
    assert len(probe.conditions())==31 and probe.conditions().count(('raw','raw'))==1
    with pytest.raises(ValueError):probe.make_controls(selected[:-1])


class TinyTied(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embedding=torch.nn.Embedding(151936,1);self.head=torch.nn.Linear(1,151936,bias=False)
        self.head.weight=self.embedding.weight;torch.nn.init.zeros_(self.head.weight)
        self.lm_head=self.head;self.model=lambda input_ids,**kw:SimpleNamespace(last_hidden_state=torch.zeros(*input_ids.shape,1))
    def get_input_embeddings(self):return self.embedding
    def get_output_embeddings(self):return self.head


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    import transformers
    from train import computable_equations as curator
    monkeypatch.setattr(curator,'source_target',lambda selected,original,tok:dict(token_ids=original[selected['cut']:selected['cut']+3],value=str(int(tok.decode(original[selected['cut']:selected['cut']+3]).strip().rstrip('.'))),numeric_token_mask=[c.isdigit() for c in tok.decode(original[selected['cut']:selected['cut']+3])],closing_boundary_ids=tok.encode('.'),closing_boundary_token_start=2))
    monkeypatch.setattr(curator,'corrupt_surface',lambda target,tok:dict(ids=tok.encode(' '+str(int(target['value'])+1)+'.'),value=str(int(target['value'])+1)))
    monkeypatch.setattr(probe,'N',2);tok=Tokenizer();gen=tmp_path/'general';gen.mkdir();mathdir=tmp_path/'math';mathdir.mkdir()
    source=tmp_path/'source.jsonl';source.write_text('Synthetic bytes only\n')
    atomic_json(gen/'inputs.json',dict(rows=[general(i) for i in range(2)]))
    atomic_json(gen/'manifest.json',dict(count=2,inputs_sha256=sha256_file(gen/'inputs.json'),
        tokenizer_vocab_sha256=canonical_hash(tok.get_vocab()),input_hashes={str(source):sha256_file(source)}))
    ar=[arithmetic(i) for i in range(512)]
    for r in ar:
        r.update(original_source_ids=(r['prefix_ids']+r['target_ids']+[32]*528)[:528],natural_future16_ids=r['target_ids']+[32]*(16-len(r['target_ids'])),
            verified_numeric_rhs=dict(token_ids=r['target_ids'],value=r['numeric_value'],numeric_token_mask=r['numeric_token_mask'],closing_boundary_ids=r['closing_boundary_ids'],closing_boundary_token_start=r['closing_boundary_token_start']),
            source_file=str(source),source_sha256=curator.SOURCE_SHA)
    monkeypatch.setattr(curator,'select_pools',lambda accepted:dict(dev=dict(requested=512,available=len(accepted),selected_item_ids=[r['item_id'] for r in accepted[:512]],reserve_item_ids=[r['item_id'] for r in accepted[512:]])))
    (mathdir/'accepted.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ar))
    (mathdir/'audit.jsonl').write_text('Synthetic complete audit\n')
    atomic_json(mathdir/'manifest.json',dict(schema=curator.SCHEMA,selection_salt=curator.SALT,source_sha256=curator.SOURCE_SHA,source_file=str(source),source_files=curator.sources(),ready_for_requested_pools=True,phase1_dev_item_ids=[r['item_id'] for r in ar[:2]],held_post_sft_dev_item_ids=[r['item_id'] for r in ar[2:]],boundary_selection='Synthetic causal fixture policy',status='complete_CPU_preparation_not_model_or_training_qualification',tokenizer_vocab_sha256=canonical_hash(tok.get_vocab()),
        audit_sha256=sha256_file(mathdir/'audit.jsonl'),accepted_sha256=sha256_file(mathdir/'accepted.jsonl'),selection=curator.select_pools(ar)))
    modeldir=tmp_path/'model';modeldir.mkdir();atomic_json(modeldir/'config.json',dict(vocab_size=151936));atomic_json(modeldir/'ASSET_MANIFEST.json',{})
    assets=dict(model_id=probe.MODEL,revision=probe.REVISION,manifest_sha256=sha256_file(modeldir/'ASSET_MANIFEST.json'))
    monkeypatch.setattr(probe,'asset_provenance',lambda path:assets if isinstance(path,Path) else pytest.fail('Path API required'))
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**kw:tok)
    model=TinyTied();monkeypatch.setattr(probe,'load_base',lambda p:(model,tok));monkeypatch.setattr(probe,'sampled_fingerprint',lambda m:SIGNATURE)
    backend=SimpleNamespace(model=model,device=torch.device('cpu'),sync=lambda:None,hidden=lambda ids:model.model(input_ids=ids).last_hidden_state)
    monkeypatch.setattr(probe,'setup_model',lambda args:(backend,tok,None,SIGNATURE,None))
    f,p=format_files(tmp_path);excluded=tmp_path/'exclusions.json';atomic_json(excluded,dict(count=0,exclusions=[]))
    output=tmp_path/'output';protocol=probe.prepare(output,gen,mathdir,modeldir,f,p,UNTIL,excluded)
    yield SimpleNamespace(output=output,p=protocol,tok=tok,gen=gen,mathdir=mathdir,modeldir=modeldir)


def test_prepared_batch_refs_native_cpu_and_real_score_contract(prepared,monkeypatch):
    p,x=probe.load(prepared.output)
    assert p['expected_rows']==124 and len(x['context_hashes'])==4
    assert [(r['cohort'],r['interface']) for r in x['qualification_refs']]==[('general','raw'),('general','I4'),('arithmetic','raw'),('arithmetic','I3')]
    before=torch.get_num_threads()
    try:probe.cpu_reference(prepared.output,1,UNTIL)
    finally:torch.set_num_threads(before)
    ref,v=probe.reference(prepared.output,p,x);assert v.shape==(4,151936) and ref['tied']
    def score(backend,context,continuations,**kw):
        target=continuations[0];lp=[-float(i+1) for i in range(len(target))]
        assert kw['future_width']==16
        return [ContinuationScore(sum(lp),sum(lp)/len(lp),lp,target,target)]
    monkeypatch.setattr(probe,'score_continuations',score)
    probe.score(prepared.output,0,UNTIL)
    d=prepared.output/'scores/shard-0';actual=probe.rows(d/'scores.jsonl');report=probe.read(d/'run.json')
    assert report['status']=='complete' and len(actual)==31
    assert sum(r['interface']=='raw' for r in actual)==1 and all(r['full_span_ll']==-136 for r in actual)
    assert report['native_max_full_head_logp_error']==0


def test_preparation_and_reference_fail_closed(prepared):
    p,x=probe.load(prepared.output)
    with pytest.raises(ValueError):probe.cpu_reference(prepared.output,17,UNTIL)
    source=next(iter(probe.read(prepared.gen/'manifest.json')['input_hashes']))
    Path(source).write_text('Changed selected source')
    # General manifest retains its exact input pin; final prepare/load bind its bytes.
    (prepared.gen/'manifest.json').write_text('{}')
    with pytest.raises(ValueError):probe.load(prepared.output)


@pytest.mark.parametrize('mutation', ['same_value','missing_boundary','wrong_reanchor','undefined','shrink'])
def test_arithmetic_contract_rejects_invalid_publications(prepared,mutation):
    path=prepared.mathdir/'accepted.jsonl';actual=probe.rows(path)
    if mutation=='same_value':actual[0]['corrupt_numeric_value']='2/1'
    elif mutation=='missing_boundary':actual[0]['closing_boundary_ids']=[]
    elif mutation=='wrong_reanchor':actual[0]['reanchor_ids']=[1]
    elif mutation=='undefined':actual[0]['target_ids']=[999999]
    else:
        m=probe.read(prepared.mathdir/'manifest.json');m['selection']['dev']['selected_item_ids']=m['selection']['dev']['selected_item_ids'][:256]
        atomic_json(prepared.mathdir/'manifest.json',m,overwrite=True)
    path.write_text(''.join(json.dumps(r)+'\n' for r in actual))
    m=probe.read(prepared.mathdir/'manifest.json');m['accepted_sha256']=sha256_file(path);atomic_json(prepared.mathdir/'manifest.json',m,overwrite=True)
    with pytest.raises(ValueError):probe.load_arithmetic(prepared.mathdir,prepared.tok)


def test_summary_requires_every_cell_and_preserves_fullspan_primary(prepared,monkeypatch):
    p,x=probe.load(prepared.output);p['bootstrap']['repeats']=100
    actual=[]
    for r in x['rows']:
        for i,m in probe.conditions():
            base=-10 if i=='raw' else -11 if m=='empty' else -5 if m=='correct' else -8
            lp=[base/len(r['target_ids'])]*len(r['target_ids'])
            actual.append(dict(cohort=r['cohort'],item_id=r['item_id'],document_id=r['document_id'],interface=i,mode=m,
                status='ok',wall_seconds=.1,token_log_likelihoods=lp,target_token_ids=r['target_ids'],
                target_sha256=canonical_hash(r['target_ids']),context_sha256=x['context_hashes'][f'{r["cohort"]}:{r["item_id"]}'][f'{i}/{m}']['sha256'],
                full_span_ll=sum(lp),k1_ll=lp[0],numeric_only_ll=sum(a for a,on in zip(lp,r.get('numeric_token_mask',[])) if on)))
    result=probe.summarize_rows(p,x,actual)
    assert result['cohorts']['general']['chosen_interface'] is None  # n<256 cannot qualify
    assert result['cohorts']['general']['contrasts']['full_span_ll']['I0']['correct_minus_raw']['difference_b_minus_a']==5
    assert result['cohorts']['general']['absolute']['full_span_ll']['raw/raw']==-10
    assert result['cohorts']['arithmetic']['correct_corrupt_note_token_lengths']==dict(equal=2,unequal=0)
    assert result['cohorts']['general']['control_note_token_means']['empty']==0
    assert result['cohorts']['general']['control_note_token_means']['correct']==16
    for bad in (actual[:-1],actual+[actual[0]],[{**actual[0],'status':'error'}]+actual[1:],[{**actual[0],'full_span_ll':0}]+actual[1:]):
        with pytest.raises(ValueError):probe.summarize_rows(p,x,bad)


def test_whole_panel_inclusive_gate_and_deterministic_fullspan_choice():
    cc={i:{k:dict(difference_b_minus_a=v) for k,v in probe.THRESHOLDS.items()} for i in probe.INTERFACES}
    assert probe.choose_interface(cc,255)==([],None)
    assert probe.choose_interface(cc,256)==(list(probe.INTERFACES),'I0')
    cc['I3']['correct_minus_raw']['difference_b_minus_a']=3.
    assert probe.choose_interface(cc,256)[1]=='I3'
    cc['I3']['empty_minus_raw']['difference_b_minus_a']=-.50000001
    assert probe.choose_interface(cc,256)[1]=='I0'


def test_prepare_general_uses_first_ids_no_future_selection(tmp_path,monkeypatch):
    import transformers
    tok=Tokenizer();source=tmp_path/'records';source.write_text('Canonical source checksum fixture\n')
    selection_path=tmp_path/'selection';selection_path.write_text('Selection checksum fixture\n')
    ids=[f'g{i}' for i in range(512)]
    selection=dict(item_ids=ids,document_ids=[f'd{i}' for i in range(512)],count=512,split='dev')
    monkeypatch.setattr(probe,'read_selection',lambda path,**kw:selection)
    def selected(path,s,**kw):
        assert s['item_ids']==ids[:256] and s['document_ids']==selection['document_ids'][:256]
        assert kw['include_future'] is True
        return [dict(item_id=key,document_id=doc,split='dev',prefix=[80]*512,future=[32]+[49]*15)
            for key,doc in zip(s['item_ids'],s['document_ids'])]
    monkeypatch.setattr(probe,'load_selected',selected)
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**kw:tok)
    output=tmp_path/'general'
    probe.prepare_general(output,selection_path,source,tmp_path,UNTIL)
    x=probe.read(output/'inputs.json');m=probe.read(output/'manifest.json')
    assert [r['item_id'] for r in x['rows']]==ids[:256] and len(x['rows'])==256
    assert all(r['copy_note_ids']==r['target_ids'] and len(r['target_ids'])==16 for r in x['rows'])
    assert all(r['reanchor_ids']==r['prefix_ids'][-24:] for r in x['rows'])
    assert sha256_file(output/'prepare_source.py')==m['input_hashes'][str((output/'prepare_source.py').resolve())]


def test_failed_cell_retained_and_stops_without_retry(prepared,monkeypatch):
    old=torch.get_num_threads()
    try:probe.cpu_reference(prepared.output,1,UNTIL)
    finally:torch.set_num_threads(old)
    calls=[]
    def fail(*a,**kw):calls.append(1);raise RuntimeError('Synthetic scoring fault')
    monkeypatch.setattr(probe,'score_continuations',fail)
    with pytest.raises(RuntimeError,match='Synthetic scoring fault'):probe.score(prepared.output,0,UNTIL)
    d=prepared.output/'scores/shard-0'
    assert len(calls)==1 and probe.read(d/'run.json')['status']=='failed'
    saved=probe.rows(d/'scores.jsonl');assert len(saved)==1 and saved[0]['status']=='error'


def test_native_failure_stops_before_any_score(prepared,monkeypatch):
    old=torch.get_num_threads()
    try:probe.cpu_reference(prepared.output,1,UNTIL)
    finally:torch.set_num_threads(old)
    p,x=probe.load(prepared.output);ref,native=probe.reference(prepared.output,p,x)
    native+=1.;monkeypatch.setattr(probe,'reference',lambda *a:(ref,native))
    monkeypatch.setattr(probe,'score_continuations',lambda *a,**k:pytest.fail('No rows before qualification'))
    with pytest.raises(RuntimeError,match='qualification'):probe.score(prepared.output,0,UNTIL)
    d=prepared.output/'scores/shard-0';assert not (d/'scores.jsonl').exists()
    assert (d/'forward-logps.pt').is_file() and probe.read(d/'run.json')['status']=='failed'


def test_supervised_publication_is_explicitly_bound_not_mislabeled(prepared,monkeypatch):
    from train import computable_equations as base,computable_equations_supervised as supervised
    m=probe.read(prepared.mathdir/'manifest.json')
    m.update(schema=supervised.SCHEMA,selection_salt=supervised.SALT,source_files=supervised.sources(),
        status='complete_supervised_CPU_preparation_not_model_or_training_qualification',
        boundary_selection='FUTURE-AWARE SUPERVISED CURATION fixture')
    monkeypatch.setattr(supervised,'select_pools',base.select_pools)
    atomic_json(prepared.mathdir/'manifest.json',m,overwrite=True)
    actual,selected=probe.load_arithmetic(prepared.mathdir,prepared.tok)
    assert actual['boundary_selection'].startswith('FUTURE-AWARE') and len(selected)==2
    m['selection_salt']=base.SALT;atomic_json(prepared.mathdir/'manifest.json',m,overwrite=True)
    with pytest.raises(ValueError):probe.load_arithmetic(prepared.mathdir,prepared.tok)


def test_overlap_reporting_does_not_change_frozen_cohorts(prepared):
    p,x=probe.load(prepared.output)
    assert p['cohort_document_overlap']['general_with_phase1_arithmetic']==0
    assert p['cohort_document_overlap']['general_with_raw_next256_arithmetic']==0
    assert p['cohort_document_overlap']['raw_next256_not_claimed_unexposed'] is True
    assert [r['item_id'] for r in x['rows']]==['g0','g1','a0','a1']


def test_optional_supervised_pool_shortfall_does_not_shrink_phase1(prepared,monkeypatch):
    from train import computable_equations as base,computable_equations_supervised as supervised
    path=prepared.mathdir/'accepted.jsonl';actual=probe.rows(path)[:320]
    path.write_text(''.join(json.dumps(r)+'\n' for r in actual))
    monkeypatch.setattr(supervised,'select_pools',base.select_pools)
    m=probe.read(prepared.mathdir/'manifest.json')
    ids=[r['item_id'] for r in actual]
    m.update(schema=supervised.SCHEMA,selection_salt=supervised.SALT,source_files=supervised.sources(),
        status='complete_supervised_CPU_preparation_not_model_or_training_qualification',ready_for_requested_pools=False,
        accepted_sha256=sha256_file(path),selection=supervised.select_pools(actual),
        phase1_dev_item_ids=ids[:2],held_post_sft_dev_item_ids=ids[2:512])
    atomic_json(prepared.mathdir/'manifest.json',m,overwrite=True)
    publication,chosen=probe.load_arithmetic(prepared.mathdir,prepared.tok)
    assert not publication['ready_for_requested_pools']
    assert publication['selection']['dev']['available']==320
    assert [r['item_id'] for r in chosen]==ids[:2]
    # Too few for the ACTUAL scientific requirement still refuses: no shrink.
    monkeypatch.setattr(probe,'N',321)
    with pytest.raises(ValueError,match='complete fixed Phase1'):probe.load_arithmetic(prepared.mathdir,prepared.tok)
