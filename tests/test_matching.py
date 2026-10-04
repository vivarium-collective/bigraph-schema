from bigraph_schema.core import allocate_core
from bigraph_schema.contract import ProcessContract, narrow_condition
from bigraph_schema.matching import find_candidates, CandidateMatch
from bigraph_schema.schema import Site


def _register_contract_hole(core, sort, hole_contract):
    core.contract_registry[sort] = hole_contract


def test_full_and_near_miss_are_both_returned():
    core = allocate_core()
    hole = narrow_condition(
        ProcessContract(face={'inputs': {'m': 'float'}, 'outputs': {'m': 'float'}}),
        'post', 'outputs.m >= 0', name='nonneg')
    _register_contract_hole(core, 'hole_sort', hole)

    from bigraph_schema.edge import Edge as Process

    class GoodProc(Process):
        contract = narrow_condition(
            ProcessContract(face={'inputs': {'m': 'float'}, 'outputs': {'m': 'float'}}),
            'post', 'outputs.m >= 0', name='nonneg')
        def inputs(self): return {'m': 'float'}
        def outputs(self): return {'m': 'float'}
        def update(self, state, interval): return {}

    class NearProc(Process):   # face fits, but does not declare the guarantee
        def inputs(self): return {'m': 'float'}
        def outputs(self): return {'m': 'float'}
        def update(self, state, interval): return {}

    core.register_link('good_proc', GoodProc)
    core.register_link('near_proc', NearProc)

    results = find_candidates(core, Site(_sort='hole_sort'))
    by_addr = {r.address: r for r in results}
    assert by_addr['good_proc'].match == 'full'
    assert by_addr['near_proc'].match == 'partial'
    assert any('nonneg' in f['reason'] for f in by_addr['near_proc'].fails)
    assert [r.match for r in results].index('full') < [r.match for r in results].index('partial')


def test_uninstantiable_candidate_is_skipped_not_crash():
    core = allocate_core()
    hole = ProcessContract(face={'inputs': {'m': 'float'}, 'outputs': {}})
    _register_contract_hole(core, 'hole2', hole)
    from bigraph_schema.edge import Edge as Process

    class Explodes(Process):
        def __init__(self, config=None, core=None):
            raise RuntimeError('needs real config')
        def inputs(self): return {'m': 'float'}
        def outputs(self): return {}
        def update(self, state, interval): return {}

    core.register_link('explodes', Explodes)
    results = find_candidates(core, Site(_sort='hole2'))
    assert all(r.address != 'explodes' for r in results)


def test_end_to_end_template_hole_surfaces_fitting_and_near_miss():
    """A hole requiring a mass-conservation guarantee: a conserving process is
    a full match; a non-conserving one of the same shape is a near-miss naming
    the exact failing guarantee (the §4 'copasi fits except…' example)."""
    core = allocate_core()
    hole = narrow_condition(
        ProcessContract(face={'inputs': {'mass': {'_type': 'float', '_min': 0, '_units': 'mg'}},
                              'outputs': {'mass': {'_type': 'float', '_min': 0, '_units': 'mg'}}}),
        'invariant', 'outputs.mass - inputs.mass <= tol', name='conservation', tol=1e-9)
    _register_contract_hole(core, 'conserver', hole)

    from bigraph_schema.edge import Edge as Process

    class Conserving(Process):
        contract = narrow_condition(
            ProcessContract(face={'inputs': {'mass': {'_type': 'float', '_min': 0, '_units': 'g'}},
                                  'outputs': {'mass': {'_type': 'float', '_min': 0, '_units': 'g'}}}),
            'invariant', 'outputs.mass - inputs.mass <= tol', name='conservation', tol=1e-9)
        def inputs(self): return {'mass': {'_type': 'float', '_min': 0, '_units': 'g'}}
        def outputs(self): return {'mass': {'_type': 'float', '_min': 0, '_units': 'g'}}
        def update(self, state, interval): return {}

    class Leaking(Process):   # same shape + units, no conservation guarantee
        def inputs(self): return {'mass': {'_type': 'float', '_min': 0, '_units': 'g'}}
        def outputs(self): return {'mass': {'_type': 'float', '_min': 0, '_units': 'g'}}
        def update(self, state, interval): return {}

    core.register_link('conserving', Conserving)
    core.register_link('leaking', Leaking)

    results = {r.address: r for r in find_candidates(core, Site(_sort='conserver'))}
    assert results['conserving'].match == 'full'
    assert results['leaking'].match == 'partial'
    assert any('conservation' in f['reason'] for f in results['leaking'].fails)
