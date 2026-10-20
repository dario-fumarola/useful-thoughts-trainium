"""Adapt read-only state auditor to the new4-worker all-view diagnostic."""
from pathlib import Path
import py_compile
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/permutation-program-v28-world4-qualification-v1'
s=(R/'local/diagnostics/document-evidence-format-weight-v26-world8-qualification-v1/independent-audit.py').read_text()
s=s.replace('actual long-context QA state','actual all-view program state').replace('frozen evidence-plan worktree','frozen program worktree')
s=s.replace('local/worktrees/document-evidence-format-weight-v26','local/worktrees/controlled-computation-v28').replace('document-evidence-format-weight-v26-world8-qualification-v1','permutation-program-v28-world4-qualification-v1').replace("local/nvme/diagnostics/permutation", "local/nvme-study/diagnostics/permutation")
s=s.replace("assert p['schema'] == 'document-evidence-plan-v23' and p['target_slots'] == 512 and p['accumulation'] == 4\nassert p['source_loss_weight'] == p['answer_loss_weight'] == .5 and p['selection_loss_weight'] == 1.","assert p['schema'] == 'permutation-program-sft-v28' and p['target_slots'] == 64 and p['accumulation'] == 1\nassert p['answer_loss_weight'] == p['work_loss_weight'] == 1. and p['world_size']==4 and p['width']==256")
s=s.replace('assigned_physical12_13_empty','assigned_physical12_empty').replace('actual8_on_physical12_13','actual4_on_physical12').replace('== 8\n    completion','== 4\n    completion')
s=s.replace('range(8)','range(4)').replace('All16 actual native/master witnesses','All8 actual native/master witnesses').replace("[('evidence_direct', O, 8, [2, 4, 6, 8])]", "[('program_qualification', O, 8, [4, 8])]")
a=s.index('        indices=order[i*32:(i+1)*32]');b=s.index("        assert value['counters']",a)
s=s[:a]+'''        indices=order[i*32:(i+1)*32]
        # Independent interpreter/label formula; no production packer or trace helper.
        expected_answer,expected_work=0,0
        for pos,index in enumerate(indices,start=i*32):
            row=data[index];q=row['question'];node=q['start'];trace_lines=[]
            for op in q['program']:
                nxt=q[op][node]
                trace_lines.append(f'{op}: {"ABCDEFGHIJKLMNOP"[node]} -> {"ABCDEFGHIJKLMNOP"[nxt]}\\n')
                node=nxt
            assert row['answer']=='ABCDEFGHIJKLMNOP'[node]
            text=''.join(trace_lines)
            assert row['full_trace']==text
            final=tok.encode('Final answer:\\n'+row['answer'],add_special_tokens=False)+[151643]
            assert len(final)==5
            expected_answer+=len(final)
            if pos%4==2:
                native=tok.encode(text,add_special_tokens=False)
                assert len(native)==6*len(q['program'])
                expected_work+=len(native)
        labels += expected_answer+expected_work
        component=value['loss_components']
        assert component['answer_labels']==expected_answer and component['work_labels']==expected_work
        assert component['answer_weight']==component['work_weight']==1.
        assert all(math.isfinite(component[k]) for k in ['answer_mean_loss','work_mean_loss'])
        expected_loss=component['answer_mean_loss']+component['work_mean_loss']
        assert abs(value['loss']-expected_loss)<1e-5
''' +s[b:]
s=s.replace('for i in range(8)','for i in range(4)').replace('(param.shape[0]+7)//8','(param.shape[0]+3)//4')
s=s.replace("arm == 'evidence_direct'", "arm == 'program_qualification'")
s=s.replace("growth=dict(format_fresh=(memory['evidence_direct']['4']-memory['evidence_direct']['2'])/2**30,\n    format_resumed=(memory['evidence_direct']['8']-memory['evidence_direct']['6'])/2**30)","growth=dict(resumed_minus_fresh_post_save_GiB=(memory['program_qualification']['8']-memory['program_qualification']['4'])/2**30)")
s=s.replace("[('format-control', O), ('control', C)]","[('program-diagnostic', O), ('control', C)]")
s=s.replace('==6\nresult','==5\nresult')
s=s.replace('PASS_ACTUAL_DOCUMENT_EVIDENCE_FORMAT_WEIGHT_FULL_RESTORE','PASS_ACTUAL_PROGRAM_ALL_VIEW_FULL_RESTORE')
s=s.replace("actual_updates=dict(evidence_direct=8),qualified_arms=['evidence_direct']", "actual_updates=dict(program_qualification=8),qualified_arms=['program_qualification']")
s=s.replace('all8_saved4_optimizer_digests_recomputed','all4_saved4_optimizer_digests_recomputed').replace('16native checks repeat4','8native checks repeat4').replace('Original four full states','Original two full states')
s=s.replace('the coefficient treatment must start fresh B0','each learning arm must start fresh B0').replace('Eight bounded real updates','Eight bounded mixed-view updates')
# Verify actual cast model parameters changed from the released B0, alongside master-update receipts.
needle="        assert active['lm_head.weight'].untyped_storage().data_ptr() == active['model.embed_tokens.weight'].untyped_storage().data_ptr()"
s=s.replace(needle,needle+"\n        assert any(not equal(active[name],param.detach().bfloat16()) for name,param in params)\n")
(C/'independent-audit.py').write_text(s)
py_compile.compile(str(C/'independent-audit.py'),doraise=True)
print('Prepared independent4-worker full4/full8 auditor; no hardware launched.')
