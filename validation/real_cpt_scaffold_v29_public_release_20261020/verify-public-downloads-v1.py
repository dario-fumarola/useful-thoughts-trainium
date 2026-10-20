"""Download every published V29 asset without authentication and verify bytes."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import urllib.request
from urllib.parse import urlparse

S=Path(__file__).resolve().parent
repo='dario-fumarola/useful-thoughts-trainium'
tag='v29-real-continuation-cpt-20261020'
commit='f3465e060f8d7c024d8b57d95c34cc991fce52ee'
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def api(endpoint):
    return json.loads(subprocess.run(['gh','api',endpoint],check=True,capture_output=True,text=True).stdout)
release=api('repos/'+repo+'/releases/tags/'+tag)
ref=api('repos/'+repo+'/git/ref/tags/'+tag)
assert ref['object']['type']=='commit' and ref['object']['sha']==commit
assert release['tag_name']==tag and not release['draft'] and not release['prerelease']
expected={}
for line in (S/'PUBLIC_SHA256SUMS').read_text().splitlines():
    digest,name=line.split('  ',1)
    assert sha(S/name)==digest
    expected[name]=digest
expected['PUBLIC_SHA256SUMS']=sha(S/'PUBLIC_SHA256SUMS')
assets={asset['name']:asset for asset in release['assets']}
assert len(expected)==len(assets)==12 and set(expected)==set(assets)
out=S/'public-download-readback'
out.mkdir(exist_ok=False)
records=[]
for name,digest in expected.items():
    asset=assets[name]
    assert asset['state']=='uploaded' and asset['size']==(S/name).stat().st_size
    if asset.get('digest'):assert asset['digest']=='sha256:'+digest
    request=urllib.request.Request(asset['browser_download_url'],headers={'User-Agent':'useful-thoughts-trainium-public-byte-audit'})
    target=out/name
    h=hashlib.sha256();size=0
    with urllib.request.urlopen(request,timeout=90) as response,target.open('xb') as stream:
        assert response.status==200
        host=urlparse(response.url).hostname
        for chunk in iter(lambda:response.read(8*2**20),b''):
            h.update(chunk);size+=len(chunk);stream.write(chunk)
        stream.flush();os.fsync(stream.fileno())
    assert size==asset['size'] and h.hexdigest()==digest and sha(target)==digest
    records.append(dict(name=name,bytes=size,sha256=digest,asset_id=asset['id'],
        public_url=asset['browser_download_url'],unauthenticated_http_status=200,
        final_public_host=host,full_download_and_local_readback=True,fsync=True))
    print('PUBLIC_FULL_DOWNLOAD_PASS',name,size,flush=True)
    with (out/'partial-receipt.json').open('w') as stream:json.dump(records,stream,indent=2)
result=dict(status='PASS_ALL12_PUBLIC_ASSETS_FULL_DOWNLOAD_SHA_SIZE_READBACK_AND_FSYNC',
    repository=repo,tag=tag,result_commit=commit,release_url=release['html_url'],
    release_id=release['id'],published_at=release['published_at'],
    assets=records,total_bytes=sum(v['bytes'] for v in records),
    verification_source_sha256=sha(__file__),
    standalone_replay=json.loads((S/'standalone-public-replay.json').read_text()),
    limits='Public numerical/generated-output/source/report evidence only. Original source windows,model weights and optimizer states remain local; this is not off-instance model protection.')
with (S/'public-release-receipt.json').open('x') as stream:
    json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())
fd=os.open(out,os.O_RDONLY);os.fsync(fd);os.close(fd)
print(result['status'],result['total_bytes'],flush=True)
