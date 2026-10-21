import json

import pytest

from train.causal_reader_gate_driver import owner_approval,qualification_identity,science_budget
from weekend_thinking.artifacts import sha256_file


def contract(tmp_path):
    original={k:{} for k in ('data','models','assets','hidden_size','vocabulary_size',
        'tolerances','partitions','secondary_interventions','design_sha256','data_audit_sha256')}
    parent=tmp_path/'original.json';parent.write_text(json.dumps(original))
    approval=tmp_path/'approval.json'
    approval.write_text(json.dumps(dict(status='OWNER_APPROVED_QUALIFICATION_AND_COST_DEVIATIONS',
        maximum_total_reserved_chip_hours=8.25,before_any_selector_fit_or_scientific_DEV_outcome=True,
        qualification_protocol_sha256=sha256_file(parent))))
    return dict(original,owner_approval_file=str(approval),owner_approval_sha256=sha256_file(approval),
                qualification_parent_protocol_file=str(parent))


def test_owner_exception_keeps_original_failure_explicit(tmp_path):
    p=contract(tmp_path)
    assert science_budget(p)==8.25
    q=dict(status='OWNER_ACCEPTED_QUALIFICATION_WITH_DECLARED_DEVIATIONS',
        approval_sha256=p['owner_approval_sha256'],original_requirements_pass=False,
        protocol_sha256=sha256_file(p['qualification_parent_protocol_file']))
    assert qualification_identity(p,tmp_path/'unused.json',q)==q['protocol_sha256']
    q['original_requirements_pass']=True
    with pytest.raises(ValueError,match='failed requirements'):
        qualification_identity(p,tmp_path/'unused.json',q)


def test_approval_does_not_allow_scientific_settings_to_change(tmp_path):
    p=contract(tmp_path);p['data']={'changed':True}
    with pytest.raises(ValueError,match='Scientific settings changed'):
        owner_approval(p)


def test_changed_approval_is_not_accepted(tmp_path):
    p=contract(tmp_path);open(p['owner_approval_file'],'a').write('\n')
    with pytest.raises(ValueError,match='record changed'):
        owner_approval(p)


def test_original_limit_without_owner_exception():
    assert science_budget({})==8.


def test_lower_prospective_cap_does_not_reopen_or_expand_the_parent_exception():
    assert science_budget({'maximum_total_reserved_chip_hours':4.5})==4.5
    for cap in (0,9,float('nan'),float('inf'),True):
        with pytest.raises(ValueError):science_budget({'maximum_total_reserved_chip_hours':cap})
