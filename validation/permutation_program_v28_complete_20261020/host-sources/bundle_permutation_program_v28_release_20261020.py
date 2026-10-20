"""Package complete synthetic evidence for a public release,not Git history."""
from pathlib import Path
import hashlib,io,json,os,tarfile
from weekend_thinking.artifacts import atomic_json,sha256_file

R=Path('/home/ubuntu/dario')
E=R/'local/diagnostics/permutation-program-v28-fixed-evaluation-v1'
C=R/'local/diagnostics/permutation-program-v28-complete-readout-io-v2'
B=R/'local/backups/evaluation/permutation-program-v28-fixed-v1'
O=R/'local/releases/v28-controlled-computation-20261020'
read=lambda p:json.loads(Path(p).read_text())


def bundle(name,entries,inline=None):
    path=O/name;expected={}
    with tarfile.open(path,'x:gz',compresslevel=6) as archive:
        for source,target,digest in entries:
            assert source.is_file() and not source.is_symlink()
            archive.add(source,arcname=target,recursive=False)
            expected[target]=digest
        for target,data in (inline or {}).items():
            info=tarfile.TarInfo(target);info.size=len(data);info.mode=0o644
            archive.addfile(info,io.BytesIO(data));expected[target]=hashlib.sha256(data).hexdigest()
    seen=set()
    with tarfile.open(path,'r:gz') as archive:
        for member in archive:
            assert member.isfile() and member.name in expected and member.name not in seen
            stream=archive.extractfile(member);h=hashlib.sha256()
            for chunk in iter(lambda:stream.read(8*2**20),b''):h.update(chunk)
            assert h.hexdigest()==expected[member.name];seen.add(member.name)
    assert seen==set(expected)
    with path.open('rb') as f:os.fsync(f.fileno())
    return dict(file=path.name,bytes=path.stat().st_size,sha256=sha256_file(path),files=len(expected),contents=expected)


def main():
    complete=read(E/'EXECUTION/completion.json')
    assert complete['status']=='COMPLETE_ALL_FIXED23_GROUPS_PHYSICALLY_CLOSED_NO_SCORES_REVIEWED'
    assert read(B/'COMPLETE.json')['status']=='PASS_ALL23_CLOSED_EVALUATIONS_PROTECTED'
    result=read(C/'readout.json');check=read(C/'independent-check.json')
    assert check['readout_sha256']==sha256_file(C/'readout.json') and check['independent_gates']==result['gates']
    assert read(C/'ordinary-independent-check.json')['status']=='PASS_ALL_ORDINARY_ROWS_MEANS_BOOTSTRAPS_AND28_INDEPENDENT_HF_FORWARDS'
    O.mkdir(parents=True,exist_ok=False)
    entries=[];index=[]
    for job in read(E/'registry.json')['jobs']:
        key=job['key'];source=B/key;receipt=read(B/'receipts'/(key+'.json'))
        assert receipt['original_protocol_sha256']==job['protocol_sha256']
        index.append('evaluations/'+key)
        for relative,info in receipt['files'].items():
            entries.append((source/relative,'v28-evaluation-bundle/evaluations/'+key+'/'+relative,info['sha256']))
    assert len(index)==len(set(index))==23 and len(entries)==704
    content=json.dumps(index,indent=2).encode()+b'\n'
    first=bundle('v28-original-evaluations.tar.gz',entries,{
        'v28-evaluation-bundle/index-portable.json':content,
        'v28-evaluation-bundle/expected-readout.sha256':(sha256_file(C/'readout.json')+'\n').encode()})
    print(json.dumps({k:v for k,v in first.items() if k!='contents'}),flush=True)
    data=R/'local/nvme-study/data/permutation-program-v28-study-v1'
    entries=[(data/n,'v28-training-data/'+n,sha256_file(data/n)) for n in ('TRAIN-questions.jsonl','TRAIN-labels.jsonl','manifest.json')]
    second=bundle('v28-training-data.tar.gz',entries)
    print(json.dumps({k:v for k,v in second.items() if k!='contents'}),flush=True)
    entries=[]
    for path in sorted(E.rglob('*')):
        if path.is_file():entries.append((path,'v28-controls/evaluation/'+str(path.relative_to(E)),sha256_file(path)))
    for path in sorted((R/'local/analysis').glob('*permutation_program_v28*.py')):
        entries.append((path,'v28-controls/host-sources/'+path.name,sha256_file(path)))
    third=bundle('v28-controls-and-host-sources.tar.gz',entries)
    print(json.dumps({k:v for k,v in third.items() if k!='contents'}),flush=True)
    manifest=dict(status='PASS_ALL_PUBLIC_RELEASE_ARCHIVES_FULL_READBACK_AND_FSYNC',
        experiment_id='controlled-computation-v28',readout_sha256=sha256_file(C/'readout.json'),
        source_sha256=sha256_file(__file__),archives=[first,second,third],
        numerical_training_commit='ce04236ecd4d7eaa9dae8471821e600561748069',
        numerical_evaluation_commit='2eb3c1cf09d14604cc9b5dd0f97562656e95b6a5',
        limits='Synthetic inputs,gold,all original attempts,native witnesses and control/source records only. No model/optimizer weights or real ordinary-page text is uploaded. No new model calls or altered scores.')
    atomic_json(O/'manifest.json',manifest)
    with (O/'SHA256SUMS').open('x') as f:
        for item in manifest['archives']:f.write(item['sha256']+'  '+item['file']+'\n')
    fd=os.open(O,os.O_RDONLY);os.fsync(fd);os.close(fd)
    print('ALL_RELEASE_ARCHIVES_VERIFIED',flush=True)


if __name__=='__main__':main()
