"""The reusable audit kernel: `audit_face` audits a raw contract face (plus
optional predicate conditions) so the SAME typed checks run at the composite
level (ProcessContract) and at the study level (study.yaml `contract.face`).
`Finding` carries optional graded fields (level/tier/score) so a leveled audit
(L0-L5) can ride the same record."""
from bigraph_schema.contract import ProcessContract, narrow_condition
from bigraph_schema.contract_audit import audit_face, audit_contract, Finding, AuditReport


def test_finding_carries_optional_grading_fields():
    # additive: the 4 positional args still work; level/tier/score default None.
    f = Finding('error', 'c', 'w', 'm')
    assert f.level is None and f.tier is None and f.score is None
    g = Finding('warning', 'c', 'w', 'm', level='L1', tier='hard', score=0.5)
    assert (g.level, g.tier, g.score) == ('L1', 'hard', 0.5)


def test_audit_face_on_raw_dict_flags_bad_range_and_units():
    face = {'inputs': {'t': {'_type': 'float', '_min': 10, '_max': 0},
                       'g': {'_type': 'float', '_units': 'not_a_unit_xyz'}},
            'outputs': {}}
    codes = {f.code for f in audit_face(object(), face)}
    assert 'bad_range' in codes and 'bad_units' in codes


def test_audit_face_condition_unknown_port_is_error():
    face = {'inputs': {'mass': 'float'}, 'outputs': {'mass': 'float'}}
    conditions = [{'kind': 'invariant', 'name': 'x', 'expr': 'outputs.flux == 0'}]
    findings = audit_face(object(), face, conditions=conditions)
    assert any(f.code == 'unknown_port' and f.severity == 'error' for f in findings)


def test_audit_face_clean_face_has_no_errors():
    face = {'inputs': {'mass': {'_type': 'float', '_min': 0, '_units': 'mg'}},
            'outputs': {'mass': {'_type': 'float', '_min': 0, '_units': 'mg'}}}
    assert not [f for f in audit_face(object(), face) if f.severity == 'error']


def test_audit_face_tolerates_missing_and_empty_face():
    assert audit_face(object(), None) == []
    assert audit_face(object(), {}) == []


def test_audit_contract_delegates_and_sets_grade():
    c = ProcessContract(face={'inputs': {'mass': 'float'}, 'outputs': {'mass': 'float'}})
    report = audit_contract(object(), c)
    assert isinstance(report, AuditReport)
    assert report.grade is not None and 0.0 <= report.grade <= 1.0
    # a lying predicate is still caught through the contract path (which routes
    # through audit_face)
    lying = narrow_condition(c, 'post', 'outputs.ghost >= 0')
    assert audit_contract(object(), lying).ok is False
