from bigraph_schema.subsumption import port_bounds, range_subsumes

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
