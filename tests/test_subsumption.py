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
