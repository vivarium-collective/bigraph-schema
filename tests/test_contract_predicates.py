import pytest
from bigraph_schema.contract import ProcessContract, Amendment, narrow_condition


def _base():
    return ProcessContract(face={'inputs': {'mass': 'float'}, 'outputs': {'mass': 'float'}})


def test_add_invariant_is_monotone_and_readable_by_kind():
    c0 = _base()
    c1 = narrow_condition(c0, 'invariant', 'abs(sum(outputs.mass) - sum(inputs.mass)) <= tol', name='mass', tol=1e-9)
    assert c0.conditions() == []
    inv = c1.conditions(kind='invariant')
    assert len(inv) == 1 and inv[0]['name'] == 'mass' and inv[0]['tol'] == 1e-9
    assert c1.conditions(kind='pre') == []
    assert c1.predicates() == []  # admit-path callables untouched


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        narrow_condition(_base(), 'whatever', 'inputs.mass > 0')


def test_unparseable_expr_rejected_at_declaration():
    with pytest.raises(ValueError):
        narrow_condition(_base(), 'pre', 'inputs.mass >')


def test_condition_survives_to_dict_round_trip():
    c = narrow_condition(ProcessContract(face={'inputs': {}, 'outputs': {}}), 'post', 'all(outputs.mass >= 0)', name='nonneg')
    data = c.to_dict()
    conds = [(a.get('detail') or {}).get('condition') for a in data['amendments']]
    assert {'kind': 'post', 'name': 'nonneg', 'expr': 'all(outputs.mass >= 0)', 'tol': 0.0} in conds
    rebuilt = ProcessContract(face=data['face'], amendments=[Amendment(**a) for a in data['amendments']])
    assert len(rebuilt.conditions(kind='post')) == 1
