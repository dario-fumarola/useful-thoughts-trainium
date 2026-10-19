"""Independent pointer-label/order reconstruction, shared by recovery protection.

No packing functions are imported. Exact checkpoint payload auditing reuses
only the existing read-only model/master/moment inspection helper.
"""
from pathlib import Path
import json,math,random
from real_source_state_audit_v17_20261019 import read,sha,audit_checkpoint,equal
from weekend_thinking.artifacts import canonical_hash


def independent_counts(data,tok):
 counts=[]
 for row in data:
  q=row['question']
  text='Documents:\n'+'\n\n'.join(f'[{i}] {c["title"]}\n{c["text"]}' for i,c in enumerate(q['contexts'],1))+'\n\nQuestion: '+q['question']+'\n'
  instruction='Answer the question using the documents. Write Final answer: on its own line and the answer on the next line.\n'
  opening=text+instruction+'You may first list relevant document numbers in an Evidence: field, then give the final answer.\nResponse:\n'
  encode=lambda s:tok.encode(s,add_special_tokens=False)
  n=len(encode(opening))
  labels={}
  for mode,ids in [('empty',[]),('selected',row['support_indices'])]:
   prefix=encode(opening+'Evidence:'+''.join(f' [{i}]' for i in ids)+'\n')
   final=encode('Final answer:\n'+row['answer'])+[151643]
   joint=encode(opening+'Evidence:'+''.join(f' [{i}]' for i in ids)+'\nFinal answer:\n'+row['answer'])+[151643]
   assert joint[:len(prefix)]==prefix and joint[len(prefix):]==final
   labels[mode]=len(prefix)-n
  labels['answer']=len(final)
  counts.append(labels)
 return counts


def audit_metrics(job,upto,counts):
 p=read(job['protocol']);out=Path(job['output'])
 assert sha(job['protocol'])==job['protocol_sha256'] and p['items']==2048 and p['steps']==256
 data=[json.loads(line) for line in Path(p['data_file']).open()]
 assert len(data)==len(counts)==2048 and sha(p['data_file'])==p['data_sha256']
 order=[]
 for epoch in range(4):
  indices=list(range(2048));random.Random(p['seed']+epoch).shuffle(indices);order.extend(indices)
 run=read(out/'run.json')
 assert run['order_hash']==canonical_hash(order) and run['protocol_sha256']==job['protocol_sha256']
 metrics=[json.loads(line) for line in (out/'metrics.jsonl').open()][:upto]
 assert len(metrics)==upto
 labels=0
 for step,m in enumerate(metrics,1):
  chosen=order[(step-1)*32:step*32];epoch=(step-1)//64
  assert m['step']==step and m['ordered_item_ids']==[data[i]['item_id'] for i in chosen]
  answer=sum(counts[i]['answer'] for i in chosen)
  selection=sum(counts[i]['selected' if job['arm']=='evidence_select' else 'empty'] for i in chosen if (i+epoch)%2)
  labels+=8192+answer+selection
  assert m['counters']==dict(expressions=step*32,supervised_tokens=labels)
  c=m['loss_components']
  assert c['source_labels']==8192 and c['answer_labels']==answer and c['selection_labels']==selection
  assert c['source_weight']==c['answer_weight']==.5 and c['selection_weight']==1.
  assert all(math.isfinite(c[n]) for n in ['source_mean_loss','answer_mean_loss','selection_mean_loss'])
  assert abs(m['loss']-(.5*c['source_mean_loss']+.5*c['answer_mean_loss']+1.*c['selection_mean_loss']))<1e-5
  assert math.isfinite(m['gradient_norm']) and m['gradient_norm']>0
 return p,order,metrics
