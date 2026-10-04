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
