"""Finite go/no-go cost projection from completed actual qualification only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from weekend_thinking.artifacts import atomic_json,sha256_file


def read(path):return json.loads(Path(path).read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','notes','native','runtime','qualification','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();p=read(a.protocol);digest=sha256_file(a.protocol)
    qual=read(a.qualification);runtime=read(a.runtime)
    if (qual['status']!='PASS_ACTUAL_READER_FEATURE_QUALIFICATION' or qual['protocol_sha256']!=digest
            or runtime['status']!='ALL_QUALIFICATION_PROCESSES_CLOSED'
            or runtime['protocol_sha256']!=digest):
        raise ValueError('Completed actual qualification and process closure required')
    native_workers=[w for w in runtime['workers'] if w['execution_device']=='xla']
    if (len(native_workers)!=19 or any(w['returncode']!=0 or w['reserved_logical_cores']!=1
            or w['logical_cores_per_physical_chip']!=4 for w in native_workers)):
        raise ValueError('Exact four-generation/twelve-reader/three-causality qualification workers required')
    reservations=runtime['native_reservations']
    if (len(reservations)!=5 or any(r['reserved_physical_chips']!=1 or r['elapsed_wall_seconds']<=0
            for r in reservations)):
        raise ValueError('Whole physical-chip allocation for all five native stages required')
    # Include unused cores during each reserved-chip phase. Summing per-core
    # worker time would undercount stragglers and partial-chip causality work.
    spent=sum(r['elapsed_wall_seconds']/3600 for r in reservations)
    writer=[];strata=set()
    for part in range(4):
        directory=a.notes/f'part-{part}';meta=read(directory/'complete.json')
        path=directory/'notes.jsonl'
        if (meta['status']!='COMPLETE_FROZEN_GREEDY_NOTES' or meta['protocol_sha256']!=digest
                or sha256_file(path)!=meta['notes_sha256']):
            raise ValueError('Original complete qualification greedy attempts required')
        notes=[json.loads(line) for line in path.open()]
        if len(notes)!=8:raise ValueError('All qualification greedy sources required')
        for row in notes[2:]:
            if row['decode_iterations'] not in (32,64,96,128) or row['group_seconds']<=0:
                raise ValueError('Actual finite native allocation timing required')
            strata.add('calculation' if row['global_index']<16 else 'ordinary')
            writer.append(row['group_seconds']*128/row['decode_iterations'])
    if strata!={'calculation','ordinary'}:raise ValueError('Both source protocols must be profiled')
    reader=[]
    for role in ('B0_raw','T_raw','T_note'):
        for part in range(4):
            meta=read(a.native/role/f'part-{part}'/'complete.json')
            if (meta['status']!='COMPLETE_FROZEN_READER_FEATURES' or meta['protocol_sha256']!=digest
                    or not meta['execution_device'].startswith('xla') or meta['rows']!=8):
                raise ValueError('Actual complete native feature timings required')
            timings=meta['batch_timings']
            if len(timings)!=2 or any(v['documents']!=4 or v['science_path_seconds']<=0 for v in timings):
                raise ValueError('Exact production path B4 warm batches required')
            reader.append(timings[1]['science_path_seconds']/4)
    notes_count=sum(p['data'][name]['rows'] for name in ('TRAIN','DEV_calculation','DEV_ordinary'))
    contexts=notes_count*3+1024
    writer_hours=max(writer)*notes_count/4/3600
    reader_hours=2*max(reader)*contexts/4/3600
    projected=writer_hours+reader_hours+.75
    decision=spent<=.5 and projected<=8.
    atomic_json(a.output,dict(status='PASS_FINITE_ACTUAL_PROFILE' if decision else 'FAIL_FINITE_ACTUAL_PROFILE',
        protocol_sha256=digest,qualification_sha256=sha256_file(a.qualification),
        runtime_sha256=sha256_file(a.runtime),actual_qualification_reserved_chip_hours=spent,
        writer128_max_warm_seconds=max(writer),reader_max_warm_seconds_per_context=max(reader),
        projected_writer_reserved_chip_hours=writer_hours,projected_2x_reader_reserved_chip_hours=reader_hours,
        fixed_startup_and_qualification_allowance_reserved_chip_hours=.75,
        total_projected_reserved_chip_hours=projected,hard_ceiling_reserved_chip_hours=8.,
        counts=dict(greedy_notes=notes_count,reader_contexts=contexts),
        rule='Frozen before actual calls: skip first two notes/worker and first feature batch/worker; worst warm128-action scaled writer,2x worst warm reader, fixed.75 overhead. No profiling/policy sweep or outcome-driven budget increase.',
        limits='Projection, not a runtime guarantee. Whole-question release and actual supervisor hard cap still required. CPU reference/source/head time is separate; not billed instance hours or dollars.'))
    print('PROFILE_GO' if decision else 'PROFILE_NO_GO',projected,'qualification',spent,flush=True)


if __name__=='__main__':main()
