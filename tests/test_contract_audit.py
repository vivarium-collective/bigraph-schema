from bigraph_schema.contract import ProcessContract, narrow_condition
from bigraph_schema.contract_audit import completeness, Finding, AuditReport, audit_contract

def test_bare_float_ports_score_low():
    c = ProcessContract(face={'inputs': {'mass': 'float'}, 'outputs': {'mass': 'float'}})
    assert completeness(c) < 0.5

def test_constrained_ports_and_a_predicate_score_higher():
    c = narrow_condition(
        ProcessContract(face={'inputs': {'mass': 'positive_float[mM]'}, 'outputs': {'mass': 'positive_float[mM]'}}),
        'invariant', 'abs(sum(outputs.mass) - sum(inputs.mass)) <= tol', tol=1e-9)
    assert completeness(c) > completeness(ProcessContract(face={'inputs': {'mass': 'float'}, 'outputs': {'mass': 'float'}}))

def test_report_ok_iff_no_error():
    assert AuditReport(ok=True, findings=[Finding('info', 'x', 'y', 'z')]).ok is True
    assert AuditReport(ok=False, findings=[Finding('error', 'x', 'y', 'z')]).ok is False


class _Core:  # a stand-in; audit_contract must not require a real core in Phase 1
    pass

def test_predicate_referencing_unknown_port_is_an_error():
    c = narrow_condition(ProcessContract(face={'inputs': {'mass': 'float'}, 'outputs': {'mass': 'float'}}),
                         'invariant', 'outputs.flux == 0')  # no 'flux' port
    report = audit_contract(_Core(), c)
    assert report.ok is False
    assert any(f.severity == 'error' and 'flux' in f.message for f in report.findings)

def test_predicate_referencing_declared_ports_is_clean():
    c = narrow_condition(ProcessContract(face={'inputs': {'mass': 'float'}, 'outputs': {'mass': 'float'}}),
                         'invariant', 'abs(outputs.mass - inputs.mass) <= tol', tol=1e-9)
    report = audit_contract(_Core(), c)
    assert not [f for f in report.findings if f.severity == 'error']
