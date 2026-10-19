"""Normalize only generated CSV/SVG whitespace;assert content equivalence."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,io,json,xml.etree.ElementTree as ET
R=Path('/home/ubuntu/dario');F=R/'validation/document_QA_v21_complete_20261019.json';P=F.with_suffix('');sha=lambda b:hashlib.sha256(b).hexdigest();initial=sha(F.read_bytes());assert initial=='75e5caf4638ab4de27b53663707b99482ed8ac84f4ab95584549f4537f2da3a9';changes=[]
def tree(e):return (e.tag,tuple(sorted((k,' '.join(v.split())) for k,v in e.attrib.items())),' '.join((e.text or '').split()),' '.join((e.tail or '').split()),tuple(tree(c) for c in e))
for p in sorted(P.iterdir()):
 if p.suffix not in ('.csv','.svg'):continue
 old=p.read_bytes();new=('\n'.join(s.rstrip() for s in old.decode().splitlines())+'\n').encode()
 if old==new:continue
 if p.suffix=='.csv':assert list(csv.reader(io.StringIO(old.decode())))==list(csv.reader(io.StringIO(new.decode())))
 else:assert tree(ET.fromstring(old))==tree(ET.fromstring(new))
 p.write_bytes(new);changes.append(dict(path=str(p.relative_to(R)),original_sha256=sha(old),presentation_sha256=sha(new),content_equivalence='ParsedCSVcells identical' if p.suffix=='.csv' else 'XMLtree equal after insignificant whitespace normalization'))
assert sha(F.read_bytes())==initial
out=dict(status='PRESENTATION_WHITESPACE_ONLY_CONTENT_EQUIVALENCE_PASS',utc=datetime.now(timezone.utc).isoformat(),primary_report_sha256=initial,source_sha256=sha(Path(__file__).read_bytes()),changes=changes,limits='Originalexports remainin core/firstreviewEBSarchives. No rawoutputs,metrics,bootstrapintervals orscientificgates changed.')
(P/'presentation-normalization.json').write_text(json.dumps(out,indent=2));print('Normalized',len(changes),'CSV/SVGexports;all parsedcontent equivalencechecks pass;primarySHA unchanged')
