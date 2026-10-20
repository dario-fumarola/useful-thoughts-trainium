"""Check every V28 held prompt/control on CPU, without model calls or gold files."""
import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer
from train.permutation_program_evaluation import MODES, available_item, controls, prompt_ids
from train.permutation_program_answers import parse_answer, execution_diagnostic


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data-dir','model-dir','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();manifest=a.data_dir/'manifest.json';m=json.loads(manifest.read_text())
    tok=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True,trust_remote_code=False)
    result={};total=0
    for panel in ['DEV_interpolation','DEV_composition','DEV_depth']:
        path=a.data_dir/(panel+'-questions.jsonl');assert sha(path)==m['files'][path.name]
        rows=[json.loads(l) for l in path.read_text().splitlines()];notes=controls(rows)
        lengths={mode:[] for mode in MODES};root_absent=0;empty_partial=0;donor_collision=0
        lookup={r['item_id']:r for r in rows}
        for row in rows:
            q=available_item(row);node=q.start;states=[node];lines=[]
            for op in q.program:
                nxt=getattr(q,op)[node];lines.append(f'{op}: {"ABCDEFGHIJKLMNOP"[node]} -> {"ABCDEFGHIJKLMNOP"[nxt]}\n');states.append(nxt);node=nxt
            final='ABCDEFGHIJKLMNOP'[node];trace=''.join(lines)+'Final answer:\n'+final
            assert parse_answer(trace,'trace',terminated=True).answer==final
            assert execution_diagnostic(trace,q)['complete_correct_execution']
            value=notes[row['item_id']]
            assert value['partial']==''.join(lines[:-1]) and len(value['partial'].splitlines())==len(q.program)-1
            assert final not in value['partial'];root_absent+=1;empty_partial+=int(not value['partial'])
            source=lookup[value['donor_item_id']]
            assert source['family_id']!=row['family_id'] and len(source['question']['program'])==len(q.program)
            assert value['donor']==notes[source['item_id']]['partial']
            donor_collision+=int(final in value['donor'])
            if len(q.program)>1:
                actual=value['partial'].splitlines();bad=value['corrupt'].splitlines()
                assert len(bad)==len(actual) and bad[:-1]==actual[:-1] and bad[-1][:-1]==actual[-1][:-1] and bad[-1][-1]!=actual[-1][-1]
            else:assert value['partial']==value['corrupt']==value['donor']==''
            for mode in MODES:
                ids=prompt_ids(row,tok,mode,value)
                assert ids and len(ids)<=256 and all(i not in (151643,151645) for i in ids)
                if mode!='trace':assert ids[-len(tok.encode('Final answer:\n',add_special_tokens=False)):]==tok.encode('Final answer:\n',add_special_tokens=False)
                if mode=='pause':assert ids.count(151662)==6*len(q.program)
                else:assert 151662 not in ids
                lengths[mode].append(len(ids));total+=1
        result[panel]=dict(questions=len(rows),families=len({v['family_id'] for v in rows}),maximum_prompt_tokens={k:max(v) for k,v in lengths.items()},correct_partial_excludes_root_symbol=root_absent,depth1_empty_partial_controls=empty_partial,unfiltered_donor_partial_mentions_recipient_root=donor_collision)
    report=dict(status='PASS_ALL_HELD_PREFIX_ONLY_NATIVE_CONTEXTS_AND_PARTIAL_CONTROLS_CPU_ONLY',native_contexts=total,manifest_sha256=sha(manifest),panels=result,model_forwards=0,checker_sha256=sha(__file__),limits=['CPU plumbing, not generated-model behavior or a learned capability result.','Exact partial controls omit root; donor/corrupt states can incidentally mention its symbol and are never utility-filtered.','Depth1 partial/corrupt/donor controls are all empty and must not be interpreted as independent oracle headroom.'])
    with a.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
