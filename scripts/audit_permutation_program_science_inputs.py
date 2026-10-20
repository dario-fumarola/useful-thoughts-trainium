"""Check actual final cohorts and fixed B0 demonstration without model calls."""
import argparse,json
from pathlib import Path
from transformers import AutoTokenizer
from train.permutation_program_science_evaluate import cohort,b0_demo,PANELS,sources
from train.permutation_program_evaluation import prompt_ids,controls
from weekend_thinking.artifacts import sha256_file


def main():
    a=argparse.ArgumentParser(description=__doc__)
    for name in ('data-dir','model-dir','output'):a.add_argument('--'+name,type=Path,required=True)
    p=a.parse_args();tok=AutoTokenizer.from_pretrained(p.model_dir,local_files_only=True,trust_remote_code=False)
    demo_id,demo=b0_demo(p.data_dir);maximum={};sizes={};total=0
    for panel in PANELS:
        original=cohort(p.data_dir,panel);base=cohort(p.data_dir,panel,base=True)
        sizes[panel]=dict(main_questions=len(original),main_families=len(original)//2,base_questions=len(base))
        for mode in ('direct','trace'):
            lengths=[]
            for item in base:
                text=tok.decode(prompt_ids(item,tok,mode),skip_special_tokens=False,clean_up_tokenization_spaces=False)
                ids=tok.encode(demo+text,add_special_tokens=False)
                assert len(ids)<=256 and not any(v in (151643,151645,151662) for v in ids)
                lengths.append(len(ids));total+=1
            maximum[panel+'-'+mode]=max(lengths)
    items=cohort(p.data_dir,'DEV_composition',diagnostic=True);notes=controls(items)
    for row in items:
        for mode in ('partial','corrupt','donor'):
            ids=prompt_ids(row,tok,mode,notes[row['item_id']]);assert len(ids)<=256;total+=1
    result=dict(status='PASS_ACTUAL_FIXED_COHORTS_DEMO_AND_DIAGNOSTIC_NATIVE_INPUTS_CPU_ONLY',model_forwards=0,native_inputs=total,
        cohorts=sizes,B0_fixed_demo_id=demo_id,B0_maximum_prompt_tokens=maximum,diagnostic_families=len(items)//2,
        evaluation_source_files=sources(),checker_sha256=sha256_file(__file__),
        limits=['CPU construction/guard proof,not held model outcomes.','One fixed depth1 demonstration is a task-format reference,not a primary matched-training baseline.','No held generation release exists before all six audited terminals.'])
    with p.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='evaluation_source_files'},indent=2))


if __name__=='__main__':main()
