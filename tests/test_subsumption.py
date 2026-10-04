from bigraph_schema.subsumption import port_bounds, range_subsumes, units_compatible

def test_port_bounds_parses_both_spellings():
    assert port_bounds({'_type': 'float', '_min': 0, '_max': 5, '_units': 'fg'}) == (0, 5, 'fg')
    assert port_bounds('float[fg]') == (None, None, 'fg')
    assert port_bounds('float') == (None, None, None)
    assert port_bounds({'_type': 'float'}) == (None, None, None)

def test_range_subsumes_candidate_within_required():
    # candidate [0,3] ⊆ required [0,5]  → ok
    ok, _ = range_subsumes({'_min': 0, '_max': 5}, {'_min': 0, '_max': 3})
    assert ok is True

def test_range_subsumes_candidate_exceeds_required():
    # candidate [-1,5] not ⊆ required [0,5] → fail
    ok, reason = range_subsumes({'_min': 0, '_max': 5}, {'_min': -1, '_max': 5})
    assert ok is False and 'min' in reason

def test_range_subsumes_required_unbounded_accepts_anything():
    ok, _ = range_subsumes({'_type': 'float'}, {'_min': -10, '_max': 10})
    assert ok is True


def test_units_compatible_same_dimension():
    ok, _ = units_compatible({'_units': 'mg'}, {'_units': 'g'})
    assert ok is True   # mass ↔ mass

def test_units_incompatible_dimensions():
    ok, reason = units_compatible({'_units': 'mg'}, {'_units': 'second'})
    assert ok is False and 'convert' in reason.lower()

def test_units_absent_is_permissive():
    assert units_compatible({'_type': 'float'}, {'_type': 'float'})[0] is True
    assert units_compatible({'_units': 'mg'}, {'_type': 'float'})[0] is True  # candidate unitless

def test_units_compatible_without_pint_is_permissive(monkeypatch):
    import bigraph_schema.subsumption as sub
    monkeypatch.setattr(sub, '_unit_registry', None)
    ok, _ = units_compatible({'_units': 'mg'}, {'_units': 'second'})
    assert ok is True   # pint absent → cannot check → permit (matches audit_contract)


# Face subsumption tests (Task 3)
from bigraph_schema.core import allocate_core
from bigraph_schema.subsumption import face_subsumes

def _core():
    return allocate_core()

def test_face_subsumes_exact():
    core = _core()
    face = {'inputs': {'m': {'_type': 'float', '_min': 0, '_max': 5, '_units': 'mg'}}, 'outputs': {}}
    ok, fails, over = face_subsumes(core, face, {'inputs': {'m': {'_type': 'float', '_min': 0, '_max': 3, '_units': 'g'}}, 'outputs': {}})
    assert ok is True and fails == []

def test_face_missing_required_port_fails():
    core = _core()
    face = {'inputs': {'m': 'float'}, 'outputs': {}}
    ok, fails, _ = face_subsumes(core, face, {'inputs': {}, 'outputs': {}})
    assert ok is False and any('m' in f['reason'] for f in fails)

def test_face_out_of_range_fails():
    core = _core()
    face = {'inputs': {'m': {'_type': 'float', '_min': 0, '_max': 5}}, 'outputs': {}}
    ok, fails, _ = face_subsumes(core, face, {'inputs': {'m': {'_type': 'float', '_min': -1, '_max': 5}}, 'outputs': {}})
    assert ok is False and any('min' in f['reason'] for f in fails)

def test_over_provision_is_allowed_and_recorded():
    core = _core()
    face = {'inputs': {'m': 'float'}, 'outputs': {}}
    ok, fails, over = face_subsumes(core, face, {'inputs': {'m': 'float', 'extra': 'float'}, 'outputs': {}})
    assert ok is True and fails == [] and 'inputs.extra' in over


# Condition coverage tests (Task 4)
from bigraph_schema.contract import ProcessContract, narrow_condition
from bigraph_schema.subsumption import conditions_cover


def _with(kind, expr, name):
    return narrow_condition(ProcessContract(face={'inputs': {}, 'outputs': {}}), kind, expr, name=name)


def test_candidate_declares_required_guarantee():
    hole = _with('post', 'outputs.mass >= 0', 'nonneg')
    cand = _with('post', 'outputs.mass >= 0', 'nonneg')
    ok, missing = conditions_cover(hole, cand)
    assert ok is True and missing == []


def test_candidate_missing_required_guarantee():
    hole = _with('post', 'outputs.mass >= 0', 'nonneg')
    cand = ProcessContract(face={'inputs': {}, 'outputs': {}})  # declares nothing
    ok, missing = conditions_cover(hole, cand)
    assert ok is False and any('nonneg' in m['reason'] or 'post' in m['reason'] for m in missing)


def test_candidate_matches_by_normalized_expr_not_name():
    hole = _with('invariant', 'outputs.x - inputs.x <= tol', 'conservation')
    cand = _with('invariant', 'outputs.x - inputs.x <= tol', 'different_name')
    ok, missing = conditions_cover(hole, cand)
    assert ok is True   # same kind + same parsed expr ⇒ covered


def test_bare_hole_is_covered_by_anything():
    hole = ProcessContract(face={'inputs': {}, 'outputs': {}})
    cand = ProcessContract(face={'inputs': {}, 'outputs': {}})
    assert conditions_cover(hole, cand) == (True, [])


def test_face_mirrors_face_conforms_resolvability():
    # Pins inherited behavior: a primitive TYPE mismatch is NOT caught at the
    # face level (core.resolve does not reject it), matching the existing
    # face_conforms / admit path. Phase 2a adds bounds/units/conditions strictness.
    core = _core()
    face = {'inputs': {'m': 'float'}, 'outputs': {}}
    ok, fails, _ = face_subsumes(core, face, {'inputs': {'m': 'string'}, 'outputs': {}})
    assert ok is True and fails == []


# Contract subsumption tests (Task 5)
from bigraph_schema.subsumption import contract_subsumes, SubsumptionResult

def test_bare_hole_matches_on_face_alone():
    core = _core()
    hole = ProcessContract(face={'inputs': {'m': 'float'}, 'outputs': {}})
    cand = ProcessContract(face={'inputs': {'m': 'float', 'extra': 'float'}, 'outputs': {}})
    result = contract_subsumes(core, hole, cand)
    assert isinstance(result, SubsumptionResult)
    assert result.ok is True and result.fails == [] and 'inputs.extra' in result.over_provides

def test_near_miss_face_ok_but_missing_guarantee():
    core = _core()
    hole = narrow_condition(ProcessContract(face={'inputs': {'m': 'float'}, 'outputs': {'m': 'float'}}),
                            'post', 'outputs.m >= 0', name='nonneg')
    cand = ProcessContract(face={'inputs': {'m': 'float'}, 'outputs': {'m': 'float'}})
    result = contract_subsumes(core, hole, cand)
    assert result.ok is False
    assert any('nonneg' in f['reason'] for f in result.fails)

def test_full_fail_on_missing_port():
    core = _core()
    hole = ProcessContract(face={'inputs': {'m': 'float'}, 'outputs': {}})
    cand = ProcessContract(face={'inputs': {}, 'outputs': {}})
    result = contract_subsumes(core, hole, cand)
    assert result.ok is False and any('face' in f['condition'] for f in result.fails)
