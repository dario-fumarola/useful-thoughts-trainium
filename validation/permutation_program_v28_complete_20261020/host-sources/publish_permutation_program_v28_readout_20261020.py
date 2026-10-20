"""Curate the complete V28 result; large synthetic evidence goes in release assets."""
from pathlib import Path
import csv,gzip,json,os,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from weekend_thinking.artifacts import atomic_json,sha256_file

R=Path('/home/ubuntu/dario')
C=R/'local/diagnostics/permutation-program-v28-complete-readout-io-v2'
E=R/'local/diagnostics/permutation-program-v28-fixed-evaluation-v1'
O=R/'local/nvme-cpt/evaluation/permutation-program-v28-ordinary-CPU-v1'
P=R/'validation/permutation_program_v28_complete_20261020'
read=lambda p:json.loads(Path(p).read_text())


def main():
    result=read(C/'readout.json');check=read(C/'independent-check.json')
    assert check['readout_sha256']==sha256_file(C/'readout.json') and check['independent_gates']==result['gates']
    assert read(C/'ordinary-independent-check.json')['status']=='PASS_ALL_ORDINARY_ROWS_MEANS_BOOTSTRAPS_AND28_INDEPENDENT_HF_FORWARDS'
    P.mkdir(exist_ok=False);files={}
    def copy(source,name):
        target=P/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        assert sha256_file(source)==sha256_file(target)
        with target.open('rb') as f:os.fsync(f.fileno())
        files[str(target.relative_to(R))]=sha256_file(target)
    for name in ('readout.json','independent-check.json','secondary-reference.json','fixed-examples.json',
                 'stored-trace-diagnosis.json','ordinary-independent-check.json','ordinary-independent-forward-plan.json',
                 'stage-history.json','COMPLETE.json'):
        copy(C/name,name)
    copy(O/'result.json','ordinary-result.json')
    copy(O/'protocol.json','ordinary-protocol.json')
    copy(E/'EXECUTION/completion.json','execution-completion.json')
    copy(E/'training-supervision-inventory.json','training-supervision-inventory.json')
    copy(E/'offline-analysis-IO-repair-v2.json','offline-analysis-IO-repair-v2.json')
    release=R/'local/releases/v28-controlled-computation-20261020'
    copy(release/'manifest.json','release-asset-manifest.json')
    copy(release/'SHA256SUMS','release-SHA256SUMS')
    for name in ('primary-contrasts.csv','absolute-answers-and-costs.csv'):
        assert sha256_file(C/'figures'/name)==sha256_file(C/'figures-publication-v3'/name)
    for path in (C/'figures-publication-v3').iterdir():
        if path.is_file():copy(path,path.name)
    for path in sorted((R/'local/analysis').glob('*permutation_program_v28*.py')):
        copy(path,'host-sources/'+path.name)
    # Preserve every ordinary score vector,without duplicating underlying real-page text/tokens.
    ordinary=P/'ordinary-page-token-scores.jsonl.gz'
    page_rows=0
    with gzip.open(ordinary,'xt') as stream:
        for source in sorted(O.glob('*-token-scores.jsonl')):
            for line in source.read_text().splitlines():
                value=json.loads(line)
                value.pop('context_ids');value.pop('target_ids')
                stream.write(json.dumps(value,allow_nan=False)+'\n');page_rows+=1
    assert page_rows==7*256
    files[str(ordinary.relative_to(R))]=sha256_file(ordinary)
    means=read(O/'result.json')['absolute_mean_token_NLL']
    keys=['B0',*[f'program-{arm}-seed{seed}-v1' for seed in (2026102028,2026102029) for arm in ('direct','pause','trace')]]
    labels=['B0\nuntouched','D\norder28','P\norder28','T\norder28','D\norder29','P\norder29','T\norder29']
    plt.rcParams.update({'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(9,4.6))
    bars=ax.bar(range(7),[means[k] for k in keys],color=['#555555','#65758b','#68409a','#167565','#65758b','#68409a','#167565'])
    ax.bar_label(bars,labels=[f'{means[k]:.3f}' for k in keys],padding=3)
    ax.axhline(means['B0'],ls='--',lw=1,color='#555555')
    ax.set_xticks(range(7),labels);ax.set_ylim(0,11)
    ax.set_ylabel('Mean ordinary-text NLL (nats/token; lower is better)')
    ax.set_title('Severe ordinary-text regression in the task-only recipe')
    fig.text(.5,.015,'Same CPU FP32 scorer and 256 existing pages; 32 target tokens/page. No ordinary LM loss in training.\nD: direct · P: provided pauses · T: full traces. Diagnostic pages are reused; this is not fresh confirmation.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.12,1,1))
    for suffix in ('svg','png'):
        path=P/('ordinary-NLL.'+suffix);fig.savefig(path,dpi=160,bbox_inches='tight')
        files[str(path.relative_to(R))]=sha256_file(path)
    plt.close(fig)
    absolute={(v['model'],v['panel'],v['mode']):v for v in result['absolute'] if not v['diagnostic']}
    table=[]
    for seed in (2026102028,2026102029):
        for arm,mode in (('direct','direct'),('direct','empty'),('pause','direct'),('pause','pause'),('trace','direct'),('trace','empty'),('trace','trace')):
            value=absolute[f'program-{arm}-seed{seed}-v1','DEV_composition',mode]
            table.append(dict(seed=seed,arm=arm,mode=mode,correct=value['correct'],items=value['items'],accuracy=value['accuracy']))
    report=dict(status='COMPLETE_FIXED_TWO_ORDER_CONTROLLED_COMPUTATION_REFERENCE',experiment_id='controlled-computation-v28',
        scientific_question='Do verified generated states improve held computation beyond direct training and provided pauses,and what remains without the written trace?',
        gates=result['gates'],primary_composition=table,primary_inference=result['primary_inference'],primary_training=result['primary_training'],
        training=dict(models=6,base='Qwen3-1.7B-Base',questions=32768,families=16384,passes=4,updates_per_model=4096,order_seeds=[2026102028,2026102029],same_initial_released_weights=True,ordinary_LM_training_term=False,
            nonzero_targets_direct_model=655360,nonzero_targets_trace_model=2031520,trace_to_direct_target_ratio=3.099853515625),
        evaluation=dict(groups=23,original_generation_attempts=378880,execution_errors=0,native_vectors=92,cache_vectors=92,primary_questions=18432,primary_counterfactual_families=9216,secondary_questions_per_panel=2048,paired_family_bootstrap_draws=5000),
        actual_training_reserved_chip_hours=16.5353719144,actual_evaluation_reserved_chip_hours=read(E/'EXECUTION/completion.json')['reserved_chip_hours'],
        ordinary_mean_token_NLL=means,
        longer_trace_correct={str(seed):absolute[f'program-trace-seed{seed}-v1','DEV_depth','trace']['correct'] for seed in (2026102028,2026102029)},
        principal_findings=['Inference gate passes in both orders on held compositions at trained depths4–6; full verified trace execution matches near-perfect answer accuracy.',
            'Direct-inference training gate fails; the strongest direct control scores11914/18432 versus1654/18432 for that trace model without traces.',
            'Length generalization fails: trace answers228/2048 and1/2048 at depths7–10. Second order emits exactly6 transitions on all2048;2030 are correct six-step prefixes.',
            'Ordinary prediction is severely degraded in every task-trained arm. Twenty-eight independent HF forwards,four pages/model,validate the stored score vectors.',
            'Provided correct partial states yield0/1024 in each trace model; final answers mostly copy their last supplied state. This untrained partial-completion interface omits the root computation.',
            'No evidence of autonomous discovery,internalization,general pretraining benefit or matched inference-FLOP efficiency follows.'],
        source_commits=dict(training='ce04236ecd4d7eaa9dae8471821e600561748069',evaluation='2eb3c1cf09d14604cc9b5dd0f97562656e95b6a5'),
        raw_artifact_release=dict(tag='v28-controlled-computation-20261020',url='https://github.com/dario-fumarola/useful-thoughts-trainium/releases/tag/v28-controlled-computation-20261020',local_archives_full_readback_fsync=True,remote_readback='PENDING'),
        files=files,
        limits=['Supervised finite-state laboratory,not ordinary continued pretraining. Two data-order seeds do not provide broad training-seed uncertainty.',
            'Trace recipes change auxiliary targets and some answer-training contexts. Equal source exposure/updates do not equal label information or measured FLOPs.',
            'Pause positions are supplied in parallel prefill; they do not match autoregressive compute. B0 has one fixed demonstration,not an optimized ceiling.',
            'All original failures/caps/unsupported answers remain. Composition,interpolation and length panels are separate; no pooling or checkpoint selection.',
            'Original model/optimizer states and all raw outputs remain on same-instance EBS. Public release assets contain synthetic evidence,not model weights.'])
    atomic_json(R/'validation/permutation_program_v28_complete_20261020.json',report)
    print(json.dumps(dict(status=report['status'],files=len(files),readout_sha256=sha256_file(P/'readout.json'))))


if __name__=='__main__':main()
