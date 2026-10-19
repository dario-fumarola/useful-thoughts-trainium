"""One complete terminal pilot readout,after all frozen native and LL closures."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import csv,gzip,hashlib,json,math
from tokenizers import Tokenizer
from train.document_evidence_plan_readout import score_condition,paired_contrast,pilot_decisions
from train.document_qa_answers import parse_answer
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-evidence-plan-v23-pilot-v1'
read=lambda p:json.loads(Path(p).read_text())
rows=lambda p:[json.loads(line) for line in Path(p).read_text().splitlines()]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 # Validate complete closure/source identity before opening model outcomes.
 assert read(C/'ALL_GENERATION_CLOSED_NO_READOUT.json')['status']=='ALL1530_FROZEN_NATIVE_ATTEMPTS_CLOSED_NO_READOUT'
 assert read(C/'ALL_LIKELIHOOD_CLOSED_NO_READOUT.json')['status']=='ALL3062_ORIGINAL_POINTER_REPLAY_AND_ORDINARY_ROWS_CLOSED_NO_READOUT'
 assert read(C/'terminal-pipeline-completion.json')['status']=='BOTH_PILOT_TERMINALS_AUDITED_PROTECTED_BOUND_NO_HELD_READOUT'
 execution=read(C/'readout-execution-plan-v2.json')
 assert execution['source_files'][str(Path(__file__).relative_to(R))]==sha(__file__)
 assert all(sha(R/name)==value for name,value in execution['scoring_source_files'].items())
 release=read(C/'release.json');assert execution['pilot_release_sha256']==sha(C/'release.json')
 D=R/'local/nvme-cpt/data/document-evidence-plan-v23-TRAIN2048-held255-v2'
 manifest=read(D/'manifest.json');assert sha(D/'manifest.json')==release['data_manifest_sha256']
 for v in manifest['files'].values():assert sha(v['path'])==v['sha256']
 questions=rows(manifest['files']['held_questions']['path']);ids=[q['item_id'] for q in questions]
 gold={r['item_id']:r for r in rows(manifest['files']['held_gold']['path'])}
 assert len(ids)==len(set(ids))==255 and set(ids)==set(gold)
 P=R/'validation/document_evidence_plan_v23_pilot_complete_20261019';P.mkdir(exist_ok=False)
 tok=Tokenizer.from_file(str(R/'local/assets/qwen3-1.7b-base/tokenizer.json'))
 vocabulary=set(tok.get_vocab().values())
 payloads={}
 def payload(name,values):
  path=P/name;path.write_bytes(gzip.compress(''.join(json.dumps(v,allow_nan=False)+'\n' for v in values).encode(),mtime=0))
  payloads[str(path.relative_to(R))]=dict(rows=len(values),bytes=path.stat().st_size,sha256=sha(path))
 attempts={};observed={};tables={};native_total=0
 for group in read(C/'generation-registry.json')['groups']:
  O=Path(group['output']);p=read(O/'protocol.json');model=p['model']['id'];arm=p['model']['arm']
  assert sha(O/'protocol.json')==group['protocol_sha256'] and p['release_sha256']==sha(C/'release.json')
  assert p['ordered_item_ids']==ids and p['modes']==release['model_modes'][arm]
  assert read(Path(group['control'])/'closure.json')['assigned_physical_device_empty']
  assert all(sha(O/name)==value for name,value in p['files'].items())
  modes={mode:[] for mode in p['modes']}
  prompts={mode:{v['item_id']:v for v in rows(O/f'prompts-{mode}.jsonl')} for mode in modes}
  for shard in range(4):
   folder=O/'answers'/f'shard-{shard}';report=read(folder/'run.json')
   assert report['status']=='complete_all_attempts' and report['protocol_sha256']==sha(O/'protocol.json') and report['attempts_sha256']==sha(folder/'attempts.jsonl')
   assert read(folder/'native.json')['passed'] and read(folder/'cache-prefill.json')['max_logp_error']<=.001
   actual=rows(folder/'attempts.jsonl')
   assert len(actual)==report['rows']==len(ids[shard::4])*len(modes)
   for record in actual:
    assert record['item_id'] in ids[shard::4] and record['mode'] in modes
    assert record['n_actions']==len(record['actions'])<=128
    assert all(type(value) is int and value in vocabulary for value in record['actions'])
    assert record['prompt_tokens']==len(prompts[record['mode']][record['item_id']]['prompt_ids'])
    assert record['parser_mode']==prompts[record['mode']][record['item_id']]['parser_mode']=='prior'
    raw=tok.decode(record['content_ids'],skip_special_tokens=False);assert raw==record['raw_text']
    stop=record['stop_type'];terminated=stop.startswith('token:')
    if record['status']=='ok':
     assert stop in ['qa_answer_line','length_cap','token:151643','token:151645']
     if terminated:assert record['actions'] and record['actions'][-1] in [151643,151645] and stop==f"token:{record['actions'][-1]}"
     assert record['content_ids']==(record['actions'][:-1] if terminated else record['actions'])
     assert not any(value in [151643,151645] for value in record['content_ids'])
     if stop=='length_cap':assert len(record['actions'])==128
    if stop=='qa_answer_line':
     parsed_raw=parse_answer(raw,'prior');assert parsed_raw.status=='complete' and record['text']==raw[:parsed_raw.end]
    else:assert record['text']==raw
    assert math.isfinite(record['wall_seconds']) and record['wall_seconds']>=0
    modes[record['mode']].append(record)
  attempts[arm],observed[arm],tables[arm]={},{},{}
  for mode,records in modes.items():
   scored,summary=score_condition(records,questions,gold,mode=mode)
   attempts[arm][mode]={v['item_id']:v for v in records}
   observed[arm][mode]={v['item_id']:v for v in scored}
   summary['generation_wall_seconds_sum']=sum(v['wall_seconds'] for v in records)
   tables[arm][mode]=summary;native_total+=len(records)
   payload(f'{model}-{mode}-native-and-scored.jsonl.gz',[dict(native=attempts[arm][mode][key],score=observed[arm][mode][key]) for key in ids])
 assert set(tables)=={'evidence_direct','evidence_select'} and native_total==1530
 def comp(a,am,b,bm,metric='canonical_em'):
  return paired_contrast({k:v[metric] for k,v in observed[a][am].items()},{k:v[metric] for k,v in observed[b][bm].items()})
 pairs=dict(selected_minus_matched_direct=('evidence_direct','direct','evidence_select','selection'),
  selected_minus_same_direct=('evidence_select','direct','evidence_select','selection'),
  selected_minus_same_empty=('evidence_select','empty','evidence_select','selection'),
  selected_minus_control_selection=('evidence_direct','selection','evidence_select','selection'),
  trained_direct_minus_matched_direct=('evidence_direct','direct','evidence_select','direct'),
  annotated_oracle_minus_selected=('evidence_select','selection','evidence_select','oracle'),
  control_selection_minus_control_direct=('evidence_direct','direct','evidence_direct','selection'))
 contrasts={name:comp(*pair) for name,pair in pairs.items()}
 decisions=pilot_decisions(contrasts['selected_minus_matched_direct'],contrasts['selected_minus_same_direct'],contrasts['selected_minus_same_empty'],tables['evidence_select']['selection']['valid_nonempty_selections'],255)
 hop_strata={}
 for hop in sorted({v['hops'] for v in gold.values()}):
  keys=[key for key in ids if gold[key]['hops']==hop]
  hop_strata[str(hop)]=dict(items=len(keys),conditions={arm:{mode:dict(correct=sum(values[key]['canonical_em'] for key in keys),unsupported=sum(values[key]['parsed']['status']!='complete' for key in keys)) for mode,values in modes.items()} for arm,modes in observed.items()},interpretation='Descriptive only;four-hop cell hasfourquestions,no depthgeneralizationclaim.')
 likelihood={};absolute={};ordinary={};ll_total=0
 for group in read(C/'likelihood-registry-v2.json')['groups']:
  O=Path(group['output']);p=read(O/'protocol.json');arm=p['model']['arm']
  assert sha(O/'protocol.json')==group['protocol_sha256'] and p['release_sha256']==sha(C/'release.json')
  assert read(Path(group['control'])/'closure.json')['assigned_physical_device_empty']
  assert all(sha(O/name)==value for name,value in p['files'].items())
  expected={(v['panel'],v['mode'],v['item_id']):v for name in p['inventories'] for v in rows(O/name)}
  actual={}
  for shard in range(4):
   folder=O/'scores'/f'shard-{shard}';complete=read(folder/'completion.json')
   assert complete['rows']==(379 if shard==3 else 384) and complete['execution_errors']==0
   assert complete['synthetic_execution_rows']==(5 if shard==3 else 0)
   assert complete['scores_sha256']==sha(folder/'scores.jsonl') and complete['source_protocol_sha256']==sha(O/'protocol.json')
   assert all(v['passed'] for v in read(folder/'own-model-TRAIN-calibration.json').values())
   for v in rows(folder/'scores.jsonl'):
    key=(v['panel'],v['mode'],v['item_id']);assert key in expected and key not in actual and v['status']=='ok'
    e=expected[key];assert v['target_ids']==e['target_ids'] and v['tokens']==len(e['target_ids'])==len(v['token_logps'])
    assert v['context_tokens']==len(e['context_ids']) and not v['synthetic_padding_observation'] and v['K1_logp']==v['token_logps'][0]
    assert all(math.isfinite(x) and x<=1e-5 for x in v['token_logps']) and abs(sum(v['token_logps'])-v['full_logp'])<1e-10
    if v['panel']=='QA':assert v['target_ids'][-1]==151643 and v['EOS_logp']==v['token_logps'][-1] and abs(v['answer_content_logp']-sum(v['token_logps'][:-1]))<1e-10
    actual[key]=v
  assert set(actual)==set(expected) and len(actual)==1531;ll_total+=len(actual)
  likelihood[arm],absolute[arm]={},{}
  for mode in ['direct','empty','selected_id_replay','other_id_replay','oracle']:
   values={key[2]:v for key,v in actual.items() if key[:2]==('QA',mode)}
   assert set(values)==set(ids)
   likelihood[arm][mode]=values
   absolute[arm][mode]=dict(items=255,**{field:sum(v[field] for v in values.values())/255 for field in ['full_logp','answer_content_logp','EOS_logp']},fallbacks=sum(bool(v.get('selection_replay_status') and v['selection_replay_status']['fallback']) for v in values.values()))
  ordinary[arm]={key[2]:-v['full_logp']/32 for key,v in actual.items() if key[:2]==('ordinary','raw')}
  assert len(ordinary[arm])==256
  absolute[arm]['ordinary']=dict(items=256,targets=8192,mean_token_NLL=sum(ordinary[arm].values())/256)
  payload(group['model']+'-all-likelihood-rows.jsonl.gz',list(actual.values()))
 assert ll_total==3062
 llcontrasts={}
 for arm in likelihood:
  llcontrasts[arm]={}
  for part in ['full_logp','answer_content_logp','EOS_logp']:
   llcontrasts[arm][part]={name:paired_contrast({k:v[part] for k,v in likelihood[arm][a].items()},{k:v[part] for k,v in likelihood[arm][b].items()}) for name,a,b in [('selected_minus_empty','empty','selected_id_replay'),('selected_minus_other','other_id_replay','selected_id_replay'),('oracle_minus_direct','direct','oracle')]}
 # Reuse only the already-completed B0 ordinary256 reference,not old QA outcomes.
 old=Path(execution['B0_ordinary_reference']['output']);oldp=read(old/'protocol.json')
 assert oldp['model']['id']=='B0' and oldp['model']['checkpoint'] is None
 assert sha(old/'ordinary-raw.jsonl')==sha(Path(read(C/'likelihood-registry-v2.json')['groups'][0]['output'])/'ordinary-raw.jsonl')
 for name,value in execution['B0_ordinary_reference']['file_sha256'].items():assert sha(old/name)==value
 base_ordinary={}
 for shard in range(4):
  folder=old/'scores'/f'shard-{shard}';complete=read(folder/'completion.json')
  assert complete['execution_errors']==0 and complete['scores_sha256']==sha(folder/'scores.jsonl')
  assert all(v['passed'] for v in read(folder/'own-model-TRAIN-calibration.json').values())
  for value in rows(folder/'scores.jsonl'):
   if value['panel']!='ordinary':continue
   assert value['status']=='ok' and value['tokens']==32 and value['item_id'] not in base_ordinary
   base_ordinary[value['item_id']]=-value['full_logp']/32
 assert base_ordinary.keys()==ordinary['evidence_direct'].keys() and len(base_ordinary)==256
 absolute['B0_reused_ordinary_only']=dict(items=256,targets=8192,mean_token_NLL=sum(base_ordinary.values())/256,
  provenance='Already-completed V21 B0 ordinary256 only,identical data/kernel;not a new QA/inference observation or new accelerator work.')
 fixed=read(C/'fixed-example-selection.json')
 examples=[dict(question=next(q for q in questions if q['item_id']==key),gold=gold[key],conditions={arm:{mode:dict(native=attempts[arm][mode][key],score=observed[arm][mode][key]) for mode in attempts[arm]} for arm in attempts}) for key in fixed['item_ids']]
 (P/'fixed-hash-eight-examples.json').write_text(json.dumps(dict(selection=fixed,examples=examples),indent=2))
 payload('fixed-held-gold-annotations.jsonl.gz',[gold[key] for key in ids])
 with (P/'absolute-answers.csv').open('x') as stream:
  writer=csv.DictWriter(stream,fieldnames=['arm','mode','items','correct','canonical_em','canonical_f1','attributed_alias_em','execution_errors','capped','generated_actions']);writer.writeheader()
  for arm,modes in tables.items():
   for mode,value in modes.items():writer.writerow(dict(arm=arm,mode=mode,**{key:value[key] for key in writer.fieldnames if key not in ['arm','mode']}))
 training_audit=read(C/'terminal-independent-audit.json')
 assert training_audit['status']=='PASS_BOTH_TERMINAL256_EVIDENCE_TRAINING_STATES_NATIVE16_ORDER_COUNTERS_AND_LABELS'
 reserved={}
 for job in read(C/'plan.json')['jobs']:
  folder=C/job['key'];start=read(folder/'launch.json')['utc'];end=read(folder/'closure.json')['utc']
  reserved['train-'+job['key']]=2*(datetime.fromisoformat(end)-datetime.fromisoformat(start)).total_seconds()/3600
 for phase in ['generation','likelihood']:
  for group in read(C/(phase+'-registry-v2.json' if phase=='likelihood' else phase+'-registry.json'))['groups']:
   folder=Path(group['control']);start=read(folder/'plan.json')['utc'];end=read(folder/'closure.json')['utc']
   reserved[phase+'-'+group['model']]=(datetime.fromisoformat(end)-datetime.fromisoformat(start)).total_seconds()/3600
 failed_reserved={}
 for group in read(C/'likelihood-registry.json')['groups']:
  folder=Path(group['control']);start=read(folder/'plan.json')['utc']
  end=max(read(folder/f'shard-{i}/exit.json')['utc'] for i in range(4))
  failed_reserved[group['model']]=(datetime.fromisoformat(end)-datetime.fromisoformat(start)).total_seconds()/3600
 report=dict(status='COMPLETE_ONE_SEED_PASSAGE_SELECTION_PILOT_NOT_REPLICATION',utc=datetime.now(timezone.utc).isoformat(),
  release_sha256=sha(C/'release.json'),analyzer_sha256=sha(__file__),native_attempts=native_total,likelihood_original_rows=ll_total,
  execution_repair=dict(original_failure_receipt_sha256=sha(C/'likelihood-v1-failure-preserved.json'),repair_release_sha256=sha(C/'likelihood-io-repair-v2/release.json'),failed_calibration_reserved_chip_hours=failed_reserved,total_failed_calibration_reserved_chip_hours=sum(failed_reserved.values()),limits='OriginalIO-onlyfailurehadzeroheldrows;inputs/models/nativeanswers/gatesunchanged. Failedattemptcost is reportedseparately,not hidden.'),tables=tables,training_models=training_audit['models'],reserved_chip_hours=reserved,total_science_reserved_chip_hours=sum(reserved.values()),cost_scope='Controlreservationstart tophysicalclosure,including preparation/compilation/checks within each job. Qualification/oldB0reference excluded;concurrentwalltime isnot servinglatency.',canonical_EM_contrasts=contrasts,canonical_F1_secondary_contrasts={name:comp(*pair,metric='canonical_f1') for name,pair in pairs.items()},
  decisions=decisions,hop_strata_descriptive=hop_strata,absolute_likelihood=absolute,secondary_pointer_replay_likelihood_contrasts=llcontrasts,
  ordinary_NLL_selection_minus_direct=paired_contrast(ordinary['evidence_direct'],ordinary['evidence_select']),ordinary_NLL_minus_reused_B0={arm:paired_contrast(base_ordinary,values) for arm,values in ordinary.items()},
  payloads=payloads,limits=['One data-order seed;no replication,pretraining novelty,broadreasoning orordinaryCoT superiority established.',
   'Fresh255 mostlytwohop. Author support annotations are imperfect;pointer agreement isnot semanticcorrectness.',
   'CanonicalEMprimary,attributedaliases/F1secondary. Allnativeerrors/unsupportedanswers/caps remain in denominators.',
   'Likelihood iscanonicalpointerreplay withflaggedemptyfallback,not native-generation likelihood orcalibration.',
   'Concurrentwalltimes are observedexecutioncosts,not controlledservinglatency. No costdominancegate is tested.',
   'This changes supervision/loss/schedule fromV21 together;it isnot a causal rootcause ablation.'])
 target=R/'validation/document_evidence_plan_v23_pilot_complete_20261019.json'
 with target.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False)
 print(json.dumps(dict(status=report['status'],decisions=decisions),indent=2))

if __name__=='__main__':main()
