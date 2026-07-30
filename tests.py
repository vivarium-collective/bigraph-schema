import sys

import pytest

import numpy as np
import pandas as pd

from bigraph_schema import Edge, allocate_core, BASE_TYPES
from bigraph_schema.schema import (
    Float, String, Map, Tree, Link, Array, Overwrite, Node, Empty,
    Site, InnerName, OuterName, Interface)
from bigraph_schema.methods import check, render, serialize, apply, reconcile
from bigraph_schema.package.discover import (
    recursive_dynamic_import, _should_skip_submodule)


@pytest.fixture
def core():
    return allocate_core()


def test_allocate_core_isolation():
    """Two allocate_core() instances must not share registry mutations.

    Registering a type/link/method on one core must not be visible on a
    separately-allocated core, while both must still carry the base types
    discovered when the cached base was built.
    """
    a = allocate_core()
    b = allocate_core()

    # Mutable containers and caches are distinct objects per instance.
    assert a.registry is not b.registry
    assert a.link_registry is not b.link_registry
    assert a.method_registry is not b.method_registry
    assert a._access_cache is not b._access_cache
    assert a._resolve_cache is not b._resolve_cache
    assert a._promote_cache is not b._promote_cache

    # Both copies carry the shared base types.
    assert 'float' in a.registry
    assert 'float' in b.registry

    # Mutating one must not leak into the other.
    a.register_type('zzz_isolation_probe', 'float')
    assert 'zzz_isolation_probe' in a.registry
    assert 'zzz_isolation_probe' not in b.registry

    a.register_link('zzz_isolation_link', Edge)
    assert 'zzz_isolation_link' in a.link_registry
    assert 'zzz_isolation_link' not in b.link_registry

    a.register_method('zzz_isolation_method', lambda core, *args, **kw: None)
    assert 'zzz_isolation_method' in a.method_registry
    assert 'zzz_isolation_method' not in b.method_registry



# test data ----------------------------

default_a = 11.111

node_schema = {
    'a': {
        '_type': 'float',
        '_default': default_a},
    'b': {
        '_type': 'string',
        '_default': 'hello world!'},
    'c': 'array[(3|4),U36]'}

map_schema = {
        '_type': 'map',
        '_key': 'string',
        '_value': 'float'}

link_schema = {
    '_type': 'link',
    '_inputs': {
        'mass': 'float',
        'concentrations': map_schema},
    '_outputs': {
        'mass': 'delta',
        'concentrations': {
            '_type': 'map',
            '_key': 'string',
            '_value': 'delta'}}}

link_a = {
    'address': 'local:edge',
    'inputs': {
        'mass': ['cell', 'mass'],
        'concentrations': ['cell', 'internal']},
    'outputs': {
        'mass': ['cell', 'mass'],
        'concentrations': ['cell', 'internal']}}

uni_schema = 'outer:tuple[tuple[boolean],' \
        'enum[a,b,c],' \
        'tuple[integer,delta,nonnegative],' \
        'list[maybe[tree[path]]],' \
        'wrap[maybe[overwrite[integer]]],' \
        'path,' \
        'wires,' \
        'integer{11},' \
        'union[link[x:integer,y:string],float,string],' \
        'tree[link[x:(y:float|z:boolean)|y:integer,oo:maybe[string]]],' \
        'a:string|b:float,' \
        'map[a:string|c:float]]|' \
        'outest:string|' \
        'list_array:list[maybe[tree[array[(6|7),float]]]]'


list_array_schema = 'a:float|list_array:list[maybe[tree[array[(3|4),float]]]]'


def test_list_array_schema(core):
    schema = core.access(list_array_schema)


# tests --------------------------------------

def do_round_trip(core, schema):
    # generate a schema object from string expression
    type_ = core.access(schema)
    # generate a json object representing schema
    reified = core.render(type_, defaults=True)
    # finally, create another schema object
    round_trip = core.access(reified)
    final = core.render(round_trip, defaults=True)

    return type_, reified, round_trip, final

def test_problem_schema_1(core):
    # this round trip is broken, shape 3 vs. (3,)
    problem_schema = 'array[3,float]'
    problem_type, reified, round_trip, final = \
            do_round_trip(core, problem_schema)
    assert isinstance(round_trip._data, np.dtype)
    assert round_trip == problem_type

def test_problem_schema_2(core):
    # turns (3, int) into ('', '<i8')
    problem_schema = 'array[3,integer]'
    problem_type, reified, round_trip, final = do_round_trip(core, problem_schema)
    assert not isinstance(problem_type, str)
    assert round_trip == problem_type

def test_array(core):
    basic = 'array[(5|6),float]'
    basic_default = core.default(basic)

    basic_link = {
        '_type': 'link',
        '_inputs': {
            'x': 'float',
            'y': 'array[(6),float]'},
        '_outputs': {
            'z': 'float',
            'w': 'array[(5),float]'},
        'inputs': {
            'x': ['array', 4, 3],
            'y': ['array', 2]},
        'outputs': {
            'z': ['array', 1, 5],
            'w': ['array', '*', 3]}}

    basic_initial = {
        'array': np.array([
            x + (7.1 * y)
            for x in range(5)
            for y in range(6)]).reshape((5,6)),
        'link': basic_link}

    basic_schema, basic_state, _ = core.realize(
        {'array': basic},
        basic_initial)

    view = core.view(
        basic_schema,
        basic_state,
        ('link',))

    output_view = core.view(
        basic_schema,
        basic_state,
        ('link',),
        ports_key='outputs')

    project_schema, project_state = core.project(
        basic_schema,
        basic_state,
        ('link',),
        {'z': 5555.5, 'w': np.array([1., 2., 3., 4., 5.])})

    applied_state, applied_merges = core.apply(
        project_schema,
        basic_state,
        project_state)

    complex_spec = [('name', np.str_, 16),
                    ('grades', np.float64, (2,))]
    complex_dtype = np.dtype(complex_spec)
    array = np.zeros((3,4), dtype=complex_dtype)
    array_schema = core.infer(array)
    rendered = core.render(array_schema)


def test_structured_array(core):
    """Test structured array type expressions with named typed fields."""

    # Basic structured array: two fields
    schema = core.access('array[id:string|count:integer]')
    assert schema._data == np.dtype([('id', '<U'), ('count', '<i4')])

    # Structured array with explicit shape
    schema = core.access('array[5,id:string|count:integer]')
    assert schema._shape == (5,)
    assert schema._data == np.dtype([('id', '<U'), ('count', '<i4')])

    # Structured array with sub-array field
    schema = core.access('array[id:string|count:integer|mass:array[9,float]]')
    expected_dtype = np.dtype([('id', '<U'), ('count', '<i4'), ('mass', '<f8', (9,))])
    assert schema._data == expected_dtype

    # Create numpy array from structured dtype and verify field access
    arr = np.zeros(3, dtype=schema._data)
    assert arr.shape == (3,)
    assert arr['mass'].shape == (3, 9)
    assert arr['count'].dtype == np.int32

    # Mixed field types with boolean
    schema = core.access('array[name:string|values:array[3,integer]|flag:boolean]')
    expected_dtype = np.dtype([('name', '<U'), ('values', '<i4', (3,)), ('flag', '?')])
    assert schema._data == expected_dtype

    # Shaped structured array
    schema = core.access('array[10,x:float|y:float|z:float]')
    assert schema._shape == (10,)
    arr = np.zeros(10, dtype=schema._data)
    assert arr['x'].shape == (10,)

    # Single field (degenerate case)
    schema = core.access('array[value:float]')
    assert 'value' in schema._data.names


def test_apply_structured_array(core):
    """Test that apply on structured arrays adds numeric fields
    and preserves non-numeric fields."""
    dt = np.dtype([('id', '<U50'), ('count', '<i8'), ('mass', '<f8')])
    schema = Array(_shape=(3,), _data=dt)

    state = np.zeros(3, dtype=dt)
    state['id'] = ['a', 'b', 'c']
    state['count'] = [10, 20, 30]
    state['mass'] = [1.0, 2.0, 3.0]

    update = np.zeros(3, dtype=dt)
    update['count'] = [5, -3, 7]
    update['mass'] = [0.1, 0.2, 0.3]

    result, merges = apply(schema, state, update, ())

    assert list(result['count']) == [15, 17, 37], f"Expected additive counts, got {result['count']}"
    assert abs(result['mass'][0] - 1.1) < 1e-10, f"Expected additive mass, got {result['mass']}"
    assert list(result['id']) == ['a', 'b', 'c'], f"Expected preserved ids, got {result['id']}"


def test_apply_structured_array_dict_update(core):
    """Test that apply on structured arrays with dict updates
    handles both set and additive semantics."""
    dt = np.dtype([('id', '<U50'), ('count', '<i8')])
    schema = Array(_shape=(3,), _data=dt)

    state = np.zeros(3, dtype=dt)
    state['id'] = ['a', 'b', 'c']
    state['count'] = [10, 20, 30]

    # Additive dict update
    result, _ = apply(schema, state.copy(), {'count': np.array([1, 2, 3])}, ())
    assert list(result['count']) == [11, 22, 33]

    # Set dict update
    result, _ = apply(schema, state.copy(), {'set': {'count': np.array([100, 200, 300])}}, ())
    assert list(result['count']) == [100, 200, 300]


def test_reconcile_float(core):
    """Float reconciliation sums deltas."""
    result = reconcile(Float(), [1.0, 2.5, -0.5])
    assert result == 3.0


def test_reconcile_float_all_none(core):
    result = reconcile(Float(), [None, None])
    assert result is None


def test_reconcile_overwrite(core):
    """Overwrite reconciliation: last non-None wins."""
    from bigraph_schema.schema import Overwrite, Node
    result = reconcile(Overwrite(_value=Node()), ['first', 'second', None])
    assert result == 'second'


def test_reconcile_array_sparse(core):
    """Array with sparse updates: concatenate sparse entry lists."""
    schema = Array(_shape=(10,), _data=np.dtype('float64'))
    u1 = [(np.array([0, 1]), np.array([1.0, 2.0]))]
    u2 = [(np.array([2]), np.array([3.0]))]
    result = reconcile(schema, [u1, u2])
    # Two sparse entries: one from each update
    assert len(result) == 2
    assert list(result[0][0]) == [0, 1]
    assert list(result[1][0]) == [2]


def test_reconcile_array_dense(core):
    """Array with dense updates: element-wise sum."""
    schema = Array(_shape=(3,), _data=np.dtype('float64'))
    result = reconcile(schema, [np.array([1, 0, 0]), np.array([0, 2, 0])])
    assert list(result) == [1, 2, 0]


def test_reconcile_map(core):
    """Map reconciliation merges keys."""
    result = reconcile(Map(), [{'a': 1}, {'b': 2}, {'a': 3}])
    assert result['a'] == 3
    assert result['b'] == 2


def test_reconcile_dict_schema(core):
    """Dict schema reconciles per-key with sub-schema dispatch."""
    schema = {'x': Float(), 'y': Float()}
    result = reconcile(schema, [
        {'x': 1.0, 'y': 2.0},
        {'x': 3.0},
    ])
    assert result['x'] == 4.0
    assert result['y'] == 2.0


def test_reconcile_nested(core):
    """Nested dict schema reconciles recursively."""
    schema = {'inner': {'a': Float(), 'b': Float()}}
    result = reconcile(schema, [
        {'inner': {'a': 1.0}},
        {'inner': {'b': 2.0}},
        {'inner': {'a': 5.0}},
    ])
    assert result['inner']['a'] == 6.0
    assert result['inner']['b'] == 2.0


def test_reconcile_preserves_divide_sentinel_dict(core):
    """_divide, _add, _remove, _type are apply-layer directives —
    reconcile must pass them through even though ``is_schema_field``
    treats leading-underscore keys on dicts as metadata."""
    schema = {'0': {'mass': Float()}}
    update = {
        '_divide': {
            'mother': '0',
            'daughters': [{'key': '00'}, {'key': '01'}],
        }
    }
    result = reconcile(schema, [update])
    assert result is not None
    assert '_divide' in result
    assert result['_divide']['mother'] == '0'


def test_reconcile_preserves_divide_sentinel_map(core):
    """Same for Map-typed stores — _divide must survive reconcile
    alongside _add/_remove."""
    schema = Map(_value=Float())
    update = {
        '_divide': {
            'mother': '0',
            'daughters': [{'key': '00'}, {'key': '01'}],
        },
        '_add': {'2': 5.0},
    }
    result = reconcile(schema, [update])
    assert result is not None
    assert result['_divide']['mother'] == '0'
    assert result['_add'] == {'2': 5.0}


def test_reconcile_dict_schema_unions_add(core):
    """Dict-schema reconcile must union _add contributions across
    updates instead of last-non-None-wins (matching List/Map). Two
    processes each contributing a new entry should both survive."""
    schema = {'0': {'mass': Float()}}
    result = reconcile(schema, [
        {'_add': {'a': {'mass': 1.0}}},
        {'_add': {'b': {'mass': 2.0}}},
    ])
    assert result is not None
    assert result['_add'] == {'a': {'mass': 1.0}, 'b': {'mass': 2.0}}


def test_reconcile_dict_schema_unions_remove(core):
    """Dict-schema reconcile must union _remove contributions, deduped."""
    schema = {'0': {'mass': Float()}}
    result = reconcile(schema, [
        {'_remove': ['a']},
        {'_remove': ['b', 'a']},
    ])
    assert result is not None
    assert result['_remove'] == ['a', 'b']


def test_reconcile_dict_schema_divide_last_wins(core):
    """_divide remains last-non-None-wins; concurrent _divide updates
    on the same node within a tick are collapsed deterministically."""
    schema = {'0': {'mass': Float()}}
    result = reconcile(schema, [
        {'_divide': {'mother': '0', 'daughters': [{'key': '00'}, {'key': '01'}]}},
        {'_divide': {'mother': '0', 'daughters': [{'key': 'aa'}, {'key': 'bb'}]}},
    ])
    assert result is not None
    assert [d['key'] for d in result['_divide']['daughters']] == ['aa', 'bb']


def test_infer(core):
    default_schema, default_state = core.default(node_schema)
    node_inferred = core.infer(default_state)
    assert check(node_inferred, default_state)

def test_render(core):
    # render is the inverse of access
    node_type = core.access(node_schema)
    node_render = core.render(node_schema, defaults=True)
    assert node_render == render(node_type, defaults=True)

    link_type = core.access(link_schema)
    link_render = core.render(link_type, defaults=True)

    # can't do the same assertion as above, because two different renderings
    # exist
    assert core.access(link_render) == link_type
    assert link_render == core.render(core.access(link_render), defaults=True)

    map_type = core.access(map_schema)
    map_render = core.render(map_type, defaults=True)
    assert core.access(map_render) == core.access(map_schema)
    # fixed point is found
    assert map_render == core.render(core.access(map_render), defaults=True)

def test_uni_schema(core):
    uni_type = core.access(uni_schema)
    assert not isinstance(uni_type, str)

    uni_render = core.render(uni_type, defaults=True)
    round_trip = core.access(uni_render)

    def idx(a, b, n):
        return a['outer']._values[n], b['outer']._values[n]

    assert uni_render == core.render(core.access(uni_type), defaults=True)

def test_default(core):
    node_type = core.access(node_schema)
    default_schema, default_state = core.default(node_schema)
    assert 'a' in default_state
    assert isinstance(default_state['a'], float)
    assert default_state['a'] == default_a
    assert 'b' in default_state
    assert isinstance(default_state['b'], str)
    assert core.check(node_schema, default_state)

    value = 11.11
    assert core.default(core.infer(value))[1] == value

def test_resolve(core):
    float_number = core.resolve('float', 'number')
    assert render(float_number) == 'float'
    assert type(float_number) == BASE_TYPES['float']

    node_resolve = core.resolve(
        {'a': 'delta', 'b': 'node'},
        node_schema)
    rendered_a = render(node_resolve, defaults=True)['a']
    assert rendered_a['_type'] == 'delta'
    assert core.access(rendered_a)._default == node_schema['a']['_default']

    mutual = core.resolve(
        {'a': 'float', 'b': 'string'},
        {'b': 'wrap[string]', 'c': 'boolean'})
    assert 'a' in mutual
    assert 'b' in mutual
    assert 'c' in mutual

    failed = False
    try:
        core.resolve(
            {'a': 'map[string]', 'b': 'node'},
            node_schema)
    except Exception as e:
        failed = True
    assert failed


def test_promote(core):
    # Library schema: a Composite-like dict with several typed branches.
    library = {
        'global_time': 'float',
        'fields': 'map[array[float]]',
        'spatial_dFBA': {
            'dFBA[0,0]': 'float',
            'dFBA[1,0]': 'float',
        },
        'emitter': 'string',
    }

    # Sparse update schema: only touches `fields/glucose`. Inner shape
    # is a wire-level projection ``{0: {0: count}}`` — what
    # project_ports_fast emits for a per-cell process update.
    from bigraph_schema.schema import Float
    sparse = {
        'fields': {
            'glucose': {0: {0: 'float'}},
        },
    }

    promoted = core.promote(library, sparse)

    # Top result has only the keys sparse touched.
    assert set(promoted.keys()) == {'fields'}, (
        'promote should not pull in keys sparse never touched')

    # The typed Map at fields wins — the dict-of-glucose subtree from
    # sparse is replaced by the library's typed Map node so apply
    # dispatches correctly.
    fields_schema = promoted['fields']
    from bigraph_schema.schema import Map
    assert isinstance(fields_schema, Map), (
        f'expected Map at fields, got {type(fields_schema).__name__}')

    # Sanity: the spatial_dFBA branch is not in the result (sparse
    # never touched it, so promote must not walk into it).
    assert 'spatial_dFBA' not in promoted


def test_promote_keeps_sparse_when_library_missing(core):
    # If library has no entry at a key, promote keeps sparse's subtree
    # so the projection schema isn't silently dropped.
    library = {'a': 'float'}
    sparse = {'b': 'float'}
    promoted = core.promote(library, sparse)
    assert 'b' in promoted

def test_check(core):
    tree_a = {
        'a': {
            'b': 5.5},
        'c': 3.3}

    tree_b = {
        'a': {
            'b': 0.111,
            'x': 444.444},
        'd': 11.11}

    tree_schema = {
        '_type': 'tree',
        '_leaf': 'float'}

    tree_parse = 'tree[float]'
    tree_type = core.access(
        tree_parse)

    assert core.check(tree_schema, tree_a)
    assert core.check(tree_parse, tree_b)
    assert not core.check(tree_schema,'not a tree')

    link_a = {
        'address': 'local:edge',
        'config': {},
        'inputs': {
            'mass': ['cell', 'mass'],
            'concentrations': ['cell', 'internal']},
        'outputs': {
            'mass': ['cell', 'mass'],
            'concentrations': ['cell', 'internal']}}

    link_b = {
        'inputs': 5.0,
        'outputs': {
            'mass': ['cell', 'mass'],
            'concentrations': ['cell', 'internal']}}

    link_c = {
        'outputs': {
            'mass': ['cell', 'mass'],
            'concentrations': ['cell', 'internal']}}

    link_d = {
        'inputs': {
            'mass': ['cell', 11.111],
            'concentrations': ['cell', 'internal']},
        'outputs': {
            'mass': ['cell', 'mass'],
            'concentrations': ['cell', 'internal']}}

    assert not core.check(link_schema, link_a)
    assert not core.check(link_schema, link_b)
    assert not core.check(link_schema, link_c)
    assert not core.check(link_schema, link_d)
    assert not core.check(link_schema, 44.44444)

    _, a_instance, _ = core.realize(link_schema, link_a)
    _, b_instance, _ = core.realize(link_schema, link_b)
    _, c_instance, _ = core.realize(link_schema, link_c)

    assert core.check(link_schema, a_instance)
    assert core.check(link_schema, b_instance)
    assert core.check(link_schema, c_instance)


def test_serialize(core):
    link_type = core.access(link_schema)
    encoded_a = serialize(link_type, link_a)

    assert encoded_a['address'] == 'local:edge'
    assert encoded_a['_inputs'] == 'mass:float|concentrations:map[float]'

    encoded_b = core.serialize(
        {'a': 'float'},
        {'a': 55.55555})

    assert encoded_b['a'] == 55.55555

def test_realize(core):
    encoded_link = {
        'inputs': {
            'mass': ['cell','mass'],
            'concentrations': '["cell","internal"]'},
        'outputs': '{\
            "mass":["cell","mass"],\
            "concentrations":["cell","internal"]}'}

    decoded_schema, decoded_state, _ = core.realize(link_schema, encoded_link)

    assert isinstance(decoded_state['instance'], Edge)

    schema = {
        'a': 'integer',
        'b': 'tuple[float,string,map[integer]]'}
    code = {
        'a': '5555',
        'b': ('1111.1', "okay", '{"x": 5, "y": "11"}')}

    decoded_schema, decoded_state, _ = core.realize(schema, code)
    assert decoded_state['a'] == 5555
    assert decoded_state['b'][2]['y'] == 11

def todo_test_infer_link(core):
    # TODO: make input/output schema depend only on
    #   edge config

    link_state = {
        'link': {
            '_type': 'link',
            '_inputs': {
                'n': 'float',
                'x': {
                    'y': 'string'}},
            '_outputs': {
                'z': 'string'},
            'inputs': {
                'n': ['A'],
                'x': ['E']},
            'outputs': {
                'z': ['F', 'f', 'ff']}}}

    link_schema = core.infer(link_state)

    assert 'A' in link_schema and isinstance(link_schema['A'], Float)
    assert 'E' in link_schema and isinstance(link_schema['E']['y'], String)


def test_traverse(core):
    tree_a = {
        'a': {
            'b': 5.5,
            'y': 555.55,
            'x': {'further': {'down': 111111.111}}},
        'c': 3.3}
    further_schema, further_state = core.traverse(
        'tree[float]',
        tree_a,
        ['a', 'x', 'further'])
    assert isinstance(further_schema, Tree)
    assert further_state == {'down': 111111.111}

    down_schema, down_state = core.traverse(
        'tree[float]',
        tree_a,
        ['a', 'x', 'further', 'down'])
    assert isinstance(down_schema, Float)
    assert down_state == 111111.111

    star_schema, star_state = core.traverse(
        {'_type': 'map', '_value': {'a': 'float', 'b': 'string'}},
        {'X': {'a': 5.5, 'b': 'green'},
         'Y': {'a': 11.11, 'b': 'another green'},
         'Z': {'a': 22.2222, 'b': 'yet another green'}},
        ['*', 'a'])
    assert isinstance(star_schema, Map)
    assert isinstance(star_schema._value, Float)
    assert star_state['Y'] == 11.11
    assert 'Z' in star_state

    puts = {
        'mass': 'float',
        'concentrations': 'map[float]'}

    link_interface = {
        '_type': 'link',
        '_inputs': puts,
        '_outputs': puts}

    link_schema = core.access(
        link_interface)

    link_state = {
        'inputs': {
            'mass': ['cell', 'mass'],
            'concentrations': ['cell', 'internal']},
        'outputs': {
            'mass': ['cell', 'mass'],
            'concentrations': ['cell', 'internal']}}

    default_schema, default_link = core.default(link_schema)
    assert default_link['inputs']['mass'] == ['mass']

    simple_interface = {
        'cell': {
            'mass': 'float',
            'internal': 'map[float]'},
        'link': link_interface}

    initial_mass = 11.1111

    simple_graph = {
        'cell': {
            'mass': initial_mass,
            'internal': {
                'A': 3.333,
                'B': 44.44444,
                'C': 5555.555}},
        'link': link_state}

    simple_schema = core.access(
        simple_interface)

    down_schema, down_state = core.jump(
        simple_interface,
        simple_graph,
        'link')
    assert isinstance(down_schema, Link)
    assert 'inputs' in down_state

    mass_schema, mass_state = core.traverse(
        simple_interface,
        simple_graph,
        ['link', 'inputs', 'mass'])
    assert isinstance(mass_schema, Float)
    assert mass_state == initial_mass

    concentration_schema, concentration_state = core.traverse(
        simple_interface,
        simple_graph,
        ['link', 'outputs', 'concentrations', 'A'])
    assert isinstance(concentration_schema, Float)
    assert concentration_state == simple_graph['cell']['internal']['A']


def test_generate(core):
    schema = {
        'A': 'float',
        'B': 'enum[one,two,three]',
        'D': 'string{hello}',
        'units': 'map[number]'}

    state = {
        'C': {
            '_type': 'enum[x,y,z]',
            '_default': 'y'},

        'concentrations': {
            'glucose': 0.5353533},

        'link': {
            '_type': 'link',
            '_inputs': {
                'n': 'float{5.5}',
                'x': 'string{what}'},
            '_outputs': {
                'z': 'string{world}'},
            'inputs': {
                'n': ['A'],
                'x': ['E']},
            'outputs': {
                'z': ['F', 'f', 'ff']}},

        'units': {
            'meters': 11.1111,
            'seconds': 22.833333}}

    generated_schema, generated_state, _ = core.realize(schema, state)

    assert generated_state['A'] == 5.5
    assert generated_state['B'] == 'one'
    assert generated_state['C'] == 'y'
    assert generated_state['units']['seconds'] == 22.833333
    assert not hasattr(generated_schema['units'], 'meters')

    view = core.view(generated_schema, generated_state, ['link'])
    assert view['n'] == 5.5

    rendered = core.render(generated_schema, defaults=True)


def test_resolve_conflict(core):
    state = {
        'A': {
            '_type': 'link',
            '_inputs': {
                'number': 'float{3.333}'},
            '_outputs': {
                'place': 'string{world}'},
            'inputs': {'number': ['number']},
            'outputs': {'place': ['place']}},
        'B': {
            '_type': 'link',
            '_inputs': {
                'place': 'map[string]'},
            '_outputs': {
                'number': 'float'},
            'inputs': {'place': ['number']},
            'outputs': {'number': ['other place']}}}

    conflict = False
    try:
        schema, realized, _ = core.realize({}, state)
    except Exception as e:
        conflict = True

    assert conflict


def test_discover_skips_defaults(core):
    """``core.discover`` is the first phase of realize — it walks
    state + schema, coerces existing values, and collects port_merges,
    but does NOT fill missing schema keys with their defaults. That
    way, a later pass can supply port-enhanced defaults instead.
    """
    schema = {
        'present': 'float',
        'missing': 'float',
    }
    state = {'present': 1.5}

    decode_schema, decode_state, merges = core.discover(schema, state)

    # `present` stays (coerced through Float)
    assert decode_state['present'] == 1.5
    # `missing` is NOT filled by discover (would be 0.0 from Float default)
    assert 'missing' not in decode_state
    # schema still records the missing key so later phases can find it
    assert 'missing' in decode_schema


def test_discover_collects_port_merges(core):
    """A ``Link`` declared in state contributes port_merges during
    discover even when the target store is absent (so port defaults
    can be applied on the completion pass)."""
    schema = {'A': 'float'}
    state = {
        'edge': {
            '_type': 'link',
            '_inputs': {'n': 'float{5.5}'},
            'inputs': {'n': ['A']},
        },
    }
    _, _, merges = core.discover(schema, state)
    # port_merges should include a contribution for the `A` path
    target_paths = [m[0] for m in merges]
    assert any('A' in str(p) for p in target_paths), (
        f'expected port_merge targeting `A`, got {target_paths}')


def test_realize_port_default_overrides_schema_default(core):
    """Port-level defaults from a wire's input schema override the
    bare schema defaults. This is the reason realize is split into
    discover + completion — if defaults were pre-filled in one pass,
    port_merges would have nothing to override."""
    schema = {'A': 'float'}  # bare default would be 0.0
    state = {
        'link': {
            '_type': 'link',
            '_inputs': {'n': 'float{5.5}'},
            'inputs': {'n': ['A']},
        },
    }
    _, realized, _ = core.realize(schema, state)
    # Port-level default 5.5 must win over the bare Float default 0.0
    assert realized['A'] == 5.5


def test_realize_coerces_list_to_ndarray_via_port(core):
    """When a port declares an ``Array`` schema for a target store that
    was seeded with a Python list, realize's second pass should coerce
    the list to an ndarray of the declared dtype."""
    import numpy as np
    schema = {'arr': 'list[float]'}  # loose initial schema
    state = {
        'arr': [1.0, 2.0, 3.0],
        'edge': {
            '_type': 'link',
            '_outputs': {'a': 'array[float[64]]'},
            'outputs': {'a': ['arr']},
        },
    }
    _, realized, _ = core.realize(schema, state)
    assert isinstance(realized['arr'], np.ndarray), (
        f'expected ndarray, got {type(realized["arr"]).__name__}')
    assert realized['arr'].dtype == np.dtype('float64')
    np.testing.assert_array_equal(realized['arr'], [1.0, 2.0, 3.0])


def test_resolve_union_accepts_option_type(core):
    """A Union schema resolved against one of its option types should
    keep the Union — the concrete type is still a valid instance of
    the union."""
    union_schema = core.access('union[boolean,string,float]')
    bool_schema = core.access('boolean')
    # Forward and reverse both work
    r1 = core.resolve(union_schema, bool_schema)
    r2 = core.resolve(bool_schema, union_schema)
    assert r1.__class__.__name__ == 'Union'
    assert r2.__class__.__name__ == 'Union'


def test_resolve_union_rejects_type_outside_options(core):
    """A Union with options [boolean, string] resolved against a Float
    should fail — the float isn't one of the declared options."""
    union_schema = core.access('union[boolean,string]')
    float_schema = core.access('float')
    raised = False
    try:
        core.resolve(union_schema, float_schema)
    except Exception:
        raised = True
    assert raised, 'expected resolve to reject type outside union options'


def test_resolve_union_with_union(core):
    """Resolving two Unions yields a Union whose options are the
    deduplicated concatenation."""
    u1 = core.access('union[boolean,string]')
    u2 = core.access('union[string,float]')
    merged = core.resolve(u1, u2)
    option_types = {type(o).__name__ for o in merged._options}
    assert 'Boolean' in option_types
    assert 'String' in option_types
    assert 'Float' in option_types


# Union type — comprehensive coverage ------------------------------------
#
# Ordering contract: ``union[a,b,c]`` tries options left-to-right and
# returns the first option whose ``realize()`` accepts the value.
# Primitive realize functions (Boolean/Integer/Float/String) are strict
# about which Python types they accept, so the "accepts" test is based
# on actual type compatibility, not arbitrary coercion.

def test_union_realize_bool_matches_boolean(core):
    """In ``union[boolean,string,float]``, a bool value realizes to the
    Boolean option (not String, not Float)."""
    _, state, _ = core.realize('union[boolean,string,float]', True)
    assert state is True


def test_union_realize_str_matches_string(core):
    _, state, _ = core.realize('union[boolean,string,float]', 'hello')
    assert state == 'hello'


def test_union_realize_float_matches_float(core):
    _, state, _ = core.realize('union[boolean,string,float]', 1.5)
    assert state == 1.5


def test_union_realize_int_matches_float_not_boolean(core):
    """A plain int in ``union[boolean,float]`` should realize as Float
    (not Boolean — Boolean now rejects non-bool values)."""
    _, state, _ = core.realize('union[boolean,float]', 7)
    assert state == 7.0
    assert isinstance(state, float)


def test_union_realize_ordering_first_match_wins(core):
    """If two options both accept a value, the first option in the
    declared order wins. Here ``union[float,integer]`` with value 5
    realizes as Float (first match)."""
    _, state, _ = core.realize('union[float,integer]', 5)
    assert isinstance(state, float)
    assert state == 5.0


def test_union_realize_ordering_reverse(core):
    """Swap the order — now Integer wins for int values."""
    _, state, _ = core.realize('union[integer,float]', 5)
    assert isinstance(state, int)
    assert state == 5


def test_union_realize_bool_rejected_by_float(core):
    """Float option rejects bool (so unions can distinguish them)."""
    schema = core.access('float')
    from bigraph_schema.methods.realize import realize as realize_fn
    _, result, _ = realize_fn(core, schema, True)
    assert result is None


def test_union_realize_non_string_rejected_by_string(core):
    """String option rejects non-str values."""
    schema = core.access('string')
    from bigraph_schema.methods.realize import realize as realize_fn
    _, result, _ = realize_fn(core, schema, 42)
    assert result is None


def test_union_render_round_trip(core):
    """A union schema renders to a string form that re-parses to the
    same type graph (option types preserved in declared order)."""
    original = core.access('union[boolean,string,float]')
    rendered = core.render(original, defaults=True)
    reparsed = core.access(rendered)
    assert [type(o).__name__ for o in reparsed._options] == [
        'Boolean', 'String', 'Float']


def test_union_realize_dispatch_no_match_returns_none_state(core):
    """If no option accepts the value, the Union returns state=None so
    the caller can fall back to a default."""
    from bigraph_schema.methods.realize import realize as realize_fn
    schema = core.access('union[boolean,string]')
    _, state, _ = realize_fn(core, schema, 3.14)
    assert state is None


def test_union_realize_in_nested_dict(core):
    """Union embedded in a dict schema realizes the right option for
    each state value."""
    schema = {'val': 'union[boolean,string,float]'}
    _, a, _ = core.realize(schema, {'val': True})
    _, b, _ = core.realize(schema, {'val': 'mass_distribution'})
    _, c, _ = core.realize(schema, {'val': 123.456})
    assert a['val'] is True
    assert b['val'] == 'mass_distribution'
    assert c['val'] == 123.456


def test_union_inside_overwrite(core):
    """``overwrite[union[...]]`` is a common v2 pattern — parsing it
    must not error on the internal resolve between Overwrite's bare
    Node placeholder and the inner Union."""
    schema = core.access('overwrite[union[boolean,string,float]]')
    # Should round-trip without exception
    rendered = core.render(schema, defaults=True)
    re = core.access(rendered)
    assert re is not None


def test_union_resolve_against_bare_node(core):
    """A bare Node (empty placeholder) resolved with a Union should
    yield the Union — the bare Node carries no type constraint."""
    from bigraph_schema.schema import Node
    union_schema = core.access('union[boolean,string,float]')
    bare = Node()
    # Forward: bare Node + Union → Union
    r1 = core.resolve(bare, union_schema)
    assert r1.__class__.__name__ == 'Union'
    # Reverse: Union + bare Node → Union
    r2 = core.resolve(union_schema, bare)
    assert r2.__class__.__name__ == 'Union'


def test_union_render_survives_merge_embedding(core):
    """A Union used inside a merge-typed Tree (e.g.,
    ``a:union[b,c]|d:string``) must round-trip. The older tilde form
    (``a:b~c|d:string``) breaks the parser because merge and union
    are sibling alternatives at the same precedence level — nesting
    one inside the other without parentheses leaves a stray ``|``."""
    schema_expr = 'a:union[boolean,string,float]|d:string'
    schema = core.access(schema_expr)
    rendered = core.render(schema)
    # Must re-parse without the original error
    core.access(rendered)


def test_unify(core):
    default_hello = 'string{hello}'

    default_hello = {
        '_type': 'string',
        '_default': 'hello'}

    schema = {
        'A': 'float',
        'B': 'enum[one,two,three]',
        'D': 'string{hello}',
        'units': 'map[number]',
        'inner': {
            'G': 'boolean{true}',
            'link': {
                '_type': 'link',
                '_inputs': {
                    'n': 'float{3.333}',
                    'v': 'overwrite[string]',
                    'x': {
                        'xx': 'string{what}',
                        'xy': 'xor'}},
                '_outputs': {
                    'z': 'string{world}'}}}}

    state = {
        'C': {
            '_type': 'enum[x,y,z]',
            '_default': 'y'},

        'concentrations': {
            'glucose': 0.5353533},

        'inner': {
            'link': {
                'inputs': {
                    'n': ['..', 'A'],
                    'v': ['..', 'D'],
                    'x': {
                        'xx': ['W', 'w'],
                        'xy': ['G']}},
                'outputs': {
                    'z': ['F', 'f', 'ffff']}}},

        'units': {
            'meters': 11.1111,
            'seconds': 22.833333}}

    generated_schema, generated_state, _ = core.realize(
        schema,
        state)

    assert generated_state['A'] == 3.333
    assert generated_state['B'] == 'one'
    assert generated_state['C'] == 'y'
    assert generated_state['units']['seconds'] == 22.833333

    assert not hasattr(generated_schema['units'], 'meters')

    rendered = core.render(generated_schema, defaults=True)

    serialized = core.serialize(generated_schema, generated_state)
    realized = core.realize(generated_schema, serialized)

    link_view = core.view(
        generated_schema,
        generated_state,
        ['inner', 'link'])

    project_schema, project_state = core.project(
        generated_schema,
        generated_state,
        ['inner', 'link'],
        link_view,
        ports_key='inputs')

    assert project_state['A'] == generated_state['A']

    project_state['D'] = 'OVER'
    applied_state, merges = core.apply(project_schema, generated_state, project_state)

    assert applied_state['inner']['G'] == False
    assert applied_state['D'] == 'OVER'

    assert 'link' in applied_state['inner']


def test_generate_coverage(core):
    schema = {
        'A': 'link[x:integer,y:nonnegative]'}

    state = {
        'B': {
            '_type': 'boolean',
            '_default': True},
        'C': {
            '_type': 'tuple[number,number]',
            '_default': (0,0)}}

    generated_schema, generated_state, _ = core.realize(schema, state)

    deschema, destate, _ = core.realize(
        generated_schema,
        core.serialize(generated_schema, generated_state))

    assert generated_schema == deschema


def test_generate_tuple_default(core):
    schema = {
        'A': 'link[x:integer,y:nonnegative]'}

    state = {
        'B': {
            '_type': 'boolean',
            '_default': True},
        'C': {
            '_type': 'tuple[number,number]',
            '_default': (0,0)}}

    generated_schema, generated_state, _ = core.realize(schema, state)
    assert generated_state['C'] == (0,0)
    assert generated_state['B'] == True


def test_generate_promote_to_struct(core):
    """
    a map schema should update to a struct schema when merged with
    a struct containing incompatible fields
    """
    # TODO - test the doppleganger dict/Map vs. Map/dict
    # TODO - this should also happen to trees
    schema = {
        'A': 'link[x:integer,y:nonnegative]'}
    state = {
        'B': {
            '_type': 'boolean',
            '_default': True}}

    generated_schema, generated_state, _ = core.realize(schema, state)
    serialized = core.serialize(generated_schema, generated_state)

    deschema, destate, _ = core.realize(
        generated_schema,
        serialized)
    assert deschema == generated_schema

def test_bind(core):
    core

def test_merge(core):
    tree_a = {
        'a': {
            'b': 5.5,
            'y': 555.55,
            'x': {'further': {'down': 111111.111}}},
        'c': 3.3}

    tree_b = {
        'a': {
            'b': 0.111,
            'z': 999999.4444,
            'x': 444.444},
        'd': 11.11}

    tree_merge = core.merge('tree[float]', tree_b, tree_a)
    assert(tree_merge['a']['x']['further']['down'])

    key_merge = core.merge(
        {'a': 'float', 'b': 'string'},
        {'a': 333.333, 'c': 4444},
        {'a': 55555.555, 'd': '111111'})

    def inputs(self):
        return {
            'mass': 'wrap[float]'} 

    assert(key_merge == {
        'a': 55555.555,
        'c': 4444,
        'd': '111111'})

def test_frame(core):
    _dict = {
        "a": [1.0, 5.0, 6.0, 6.0],
        "b": [1, 5, 6, 6],
        "c": [True, False, False, False],
    }
    df = pd.DataFrame(_dict)
    schema = core.infer(df)

    from_string = core.access('dataframe[a:float|b:integer[64]|c:boolean]')

    assert schema == from_string

    default = core.default(schema)
    frame_schema, frame_state, _ = core.realize(schema, _dict)
    encoded = core.serialize(frame_schema, frame_state)
    realized_schema, realized_state, _ = core.realize(frame_schema, encoded)

    assert realized_state.equals(df)


def test_infer_star(core):
    core.infer({
        "global_time": {"_default": 0.0, "_updater": "accumulate"},
        "next_update_time": {"*": {}},
    })


def test_access_tuple(core):
    found = core.access({'0': {('what',): 'float'}})
    assert isinstance(found['0'][('what',)], Float)


def test_serialize_realize_shape(core):
    array_schema = core.access('array[38383,float]')
    assert array_schema._shape[0] == 38383
    assert len(array_schema._shape) == 1


def test_apply(core):
    core


# Milner bigraph structural types (Site, InnerName, OuterName, Interface)
# ================================================================
# These extend bigraph-schema from Milner's "ground bigraph" shape
# (g : ε → I) toward his full formalism where a bigraph arrow
# F : I → J can be composed. See Milner, *Space and Motion of
# Communicating Agents* (2008), §2.1 Defs. 2.1–2.3, and the plan in
# .claude/plans/milner-formalism.md.
#
# All four types inherit from Empty because they carry no state on
# their own — they are schema-level markers that become state-bearing
# only through composition. The only per-type override is `render`,
# which has to emit the specific type name rather than 'empty'.


def test_bigraph_structural_types_in_registry(core):
    """The four structural types are registered under their canonical
    names and access returns instances of the expected classes."""
    assert isinstance(core.access('site'), Site)
    assert isinstance(core.access('inner_name'), InnerName)
    assert isinstance(core.access('outer_name'), OuterName)
    assert isinstance(core.access('face'), Interface)


def test_bigraph_structural_types_are_empty(core):
    """Inheriting from Empty means these types have no realizable
    state — default is None and only None checks through."""
    for type_name in ('site', 'inner_name', 'outer_name', 'face'):
        schema = core.access(type_name)
        assert isinstance(schema, Empty), (
            f'{type_name} should inherit from Empty (got {type(schema)})')
        _, state = core.default(schema)
        assert state is None, (
            f'default({type_name}) should be None, got {state!r}')
        assert core.check(schema, None) is True
        assert core.check(schema, 'anything else') is False


def test_bigraph_structural_bare_round_trip(core):
    """A bare (unsorted, empty) structural schema renders as just its
    type name and round-trips cleanly."""
    for type_name in ('site', 'inner_name', 'outer_name', 'face'):
        schema = core.access(type_name)
        rendered = core.render(schema)
        assert rendered == type_name, (
            f'bare render({type_name}) should be {type_name!r}, got {rendered!r}')
        back = core.access(rendered)
        assert type(back) is type(schema)


def test_bigraph_structural_sorted_round_trip(core):
    """Site/InnerName/OuterName carry a `_sort` label — render must
    preserve it, and the dict form must round-trip."""
    for type_name, cls in (
            ('site', Site),
            ('inner_name', InnerName),
            ('outer_name', OuterName)):
        schema = core.access({'_type': type_name, '_sort': 'cell'})
        assert isinstance(schema, cls)
        assert schema._sort == 'cell'
        rendered = core.render(schema)
        assert rendered == {'_type': type_name, '_sort': 'cell'}
        back = core.access(rendered)
        assert isinstance(back, cls)
        assert back._sort == 'cell'


def test_interface_direct_construction(core):
    """Interface's inner structure (`_places` tuple, `_names` dict) is
    built directly in Python — the dict-access path for a populated
    interface is deferred until the composition machinery lands.
    Render must still emit the internal structure so it is visible."""
    iface = Interface(
        _places=(Site(), Site(_sort='cell')),
        _names={'x': '', 'y': 'cell'})
    rendered = core.render(iface)
    assert rendered == {
        '_type': 'face',
        '_places': ['site', {'_type': 'site', '_sort': 'cell'}],
        '_names': {'x': '', 'y': 'cell'}}
    # Empty state still the right answer — this Interface is a
    # schema-level descriptor, not a state container.
    _, state = core.default(iface)
    assert state is None


def test_bigraph_structural_types_present_in_base_types(core):
    """BASE_TYPES registry wires the four type names to their classes."""
    assert BASE_TYPES['site'] is Site
    assert BASE_TYPES['inner_name'] is InnerName
    assert BASE_TYPES['outer_name'] is OuterName
    assert BASE_TYPES['face'] is Interface


def test_interface_dict_form_access(core):
    """Populated Interface schemas built via the dict-access path:
    ``{'_type': 'face', '_places': [...], '_names': {...}}``.
    Requires the resolve(tuple, list) dispatcher to merge the incoming
    list of sub-schemas with the default empty ``_places`` tuple."""
    iface = core.access({
        '_type': 'face',
        '_places': [core.access('site'), core.access('site')],
        '_names': {'x': '', 'y': 'cell'}})
    assert isinstance(iface, Interface)
    assert len(iface._places) == 2
    for place in iface._places:
        assert isinstance(place, Site)
    # Empty-string sort labels pass through access() and become None
    # (``access('')`` → None), so ``{'x': ''}`` normalizes to
    # ``{'x': None}``. Both mean "unsorted" in Milner's terms.
    assert iface._names == {'x': None, 'y': 'cell'}


def test_number_object_function_defaults(core):
    """Default dispatchers for types whose schema fields include
    primitive metadata — Number (_bits/_units), Object (_class/_schema),
    Function (module/instance/attribute). Without their custom default
    handlers, the generic Node walker would recurse into those
    primitive fields and crash with ``default('')``."""
    _, number_state = core.default('number')
    assert number_state == 0

    _, object_state = core.default('object')
    assert object_state is None

    _, function_state = core.default('function')
    assert function_state is None


def test_assembly_identity_laws():
    """Identity laws on composition (Milner Def. 2.8 C3): for any
    composable arrow ``f``, ``id ∘ f = f`` and ``f ∘ id = f``. M1.4
    only implements the trivial ``id_ε`` case; full composition comes
    in M2 with the elementary bigraphs."""
    from bigraph_schema.assembly import EPSILON, identity, compose

    # EPSILON is the origin interface ⟨0, ∅⟩.
    assert EPSILON == Interface()
    assert EPSILON._places == ()
    assert EPSILON._names == {}

    # identity(ε) is ε itself at this milestone (no separate Bigraph
    # schema yet) — what matters is that the composition laws hold.
    id_eps = identity(EPSILON)
    assert id_eps == EPSILON

    # Right identity: f ∘ id_ε = f (on the origin case, f is ε too).
    assert compose(EPSILON, id_eps) == EPSILON

    # Left identity: id_ε ∘ f = f.
    g = Interface(_names={'x': None})
    assert compose(identity(EPSILON), g) == g

    # Right identity on a non-ε outer: g ∘ id_ε = g (because id_ε has
    # no sites/names, plugging nothing into g leaves g unchanged).
    assert compose(g, identity(EPSILON)) == g

    # Non-trivial Interface-only composition is not supported — compose
    # operates on dict schemas.
    import pytest
    with pytest.raises(NotImplementedError):
        compose(Interface(_names={'x': None}), Interface(_names={'y': None}))


# ── M2: interface derivation, elementary bigraphs, composition ──────

def test_interfaces_ground_schema(core):
    """A plain schema with no Sites and all ports wired is ground."""
    from bigraph_schema.assembly import interfaces, is_ground
    schema = {'cell': {'mass': core.access('float')}}
    inner, outer = interfaces(schema)
    assert inner._places == ()
    assert inner._names == {}
    assert 'cell' in outer._places
    assert is_ground(schema)


def test_interfaces_with_sites(core):
    """Sites in the tree show up as inner face places."""
    from bigraph_schema.assembly import interfaces, is_ground
    schema = {'container': {'data': core.access('float'), 'hole': Site()}}
    inner, outer = interfaces(schema)
    assert len(inner._places) == 1
    path, site = inner._places[0]
    assert path == ('container', 'hole')
    assert isinstance(site, Site)
    assert not is_ground(schema)


def test_interfaces_unwired_ports(core):
    """Unwired Link ports are inner/outer names in the interface."""
    from bigraph_schema.assembly import interfaces
    schema = core.access({
        'proc': {
            '_type': 'link',
            '_inputs': {'x': 'float', 'y': 'float'},
            '_outputs': {'z': 'delta'},
            'inputs': {'x': ['store', 'x']}}})
    inner, outer = interfaces(schema)
    assert 'y' in inner._names    # unwired input → inner name
    assert 'x' not in inner._names  # wired → not open
    assert 'z' in outer._names    # unwired output → outer name


def test_elementary_barren(core):
    """barren() is 1 : ε → ⟨1, ∅⟩. One empty root, ground."""
    from bigraph_schema.assembly import barren, interfaces, is_ground
    b = barren()
    inner, outer = interfaces(b)
    assert inner._places == () and inner._names == {}
    assert len(outer._places) == 1
    assert is_ground(b)


def test_elementary_merge(core):
    """merge(n) has n sites under one root."""
    from bigraph_schema.assembly import merge, interfaces
    m = merge(3)
    inner, outer = interfaces(m)
    assert len(inner._places) == 3
    assert len(outer._places) == 1


def test_elementary_ion(core):
    """ion has 1 site and outer names from its ports."""
    from bigraph_schema.assembly import ion, interfaces
    i = ion(core, 'K', ('x', 'y'))
    inner, outer = interfaces(i)
    assert len(inner._places) == 1
    assert set(outer._names.keys()) == {'x', 'y'}


def test_elementary_substitution_closure(core):
    """substitution has inner names X and outer name y.
    closure has inner name x and no outer names."""
    from bigraph_schema.assembly import substitution, closure, interfaces
    s = substitution(core, 'y', ('a', 'b'))
    inner_s, outer_s = interfaces(s)
    assert set(inner_s._names.keys()) == {'a', 'b'}
    assert set(outer_s._names.keys()) == {'y'}

    c = closure(core, 'x')
    inner_c, outer_c = interfaces(c)
    assert 'x' in inner_c._names
    assert outer_c._names == {}


def test_compose_fills_sites(core):
    """Composing merge(2) with two barren roots fills both sites."""
    from bigraph_schema.assembly import merge, barren, tensor, compose, interfaces, is_ground
    outer = merge(2)
    inner = tensor(barren('a'), barren('b'))
    result = compose(outer, inner)
    assert is_ground(result)
    # The two sites were replaced by the two barren roots' content
    region = result['region0']
    assert 'site0' in region
    assert 'site1' in region


def _resolve_wire(link_path, wire):
    """Resolve a wire the way ``realize.port_merges`` does.

    A wire is relative to the link's **parent** store, so the store it
    designates is ``link_path[:-1] + wire``.
    """
    return tuple(link_path[:-1]) + tuple(wire)


def _assert_wire_lands_on_a_store(schema, store_path):
    """Every proper ancestor of ``store_path`` must exist in ``schema`` and
    none of them may be a ``Link`` — ports read from stores, not from links."""
    node = schema
    for step in store_path[:-1]:
        assert isinstance(node, dict) and step in node, (
            f'wire target {store_path} escapes the composed tree at {step!r}')
        node = node[step]
        assert not isinstance(node, Link), (
            f'wire target {store_path} passes through the Link at {step!r}')


def test_compose_wires_link_into_wired_document(core):
    """Task 0: compose a Link-bearing subtree into a site inside a wired
    document and assert the resulting wires resolve.

    The other compose tests only ever substitute *empty* regions into empty
    holes, so the link-composition branch was unexercised. Joining an outer
    name to an inner name has to satisfy three things at once: the wire must
    be expressed in the **composed** tree's coordinates (not the filler's),
    it must land on a **store** rather than inside a ``Link`` node, and the
    filler's matching port must be wired to the **same** store — a join has
    two ends.
    """
    from bigraph_schema.assembly import compose

    outer = core.access({'world': {
        'hole': {'_type': 'site'},
        'consumer': {
            '_type': 'link',
            '_inputs': {'growth_rate': 'float'},
            '_outputs': {'biomass': 'float'},
            'outputs': {'biomass': ['biomass']}}}})

    inner = core.access({'producer': {
        'proc': {
            '_type': 'link',
            '_inputs': {'nutrient': 'float'},
            '_outputs': {'growth_rate': 'float'},
            'inputs': {'nutrient': ['nutrient']}}}})

    result = compose(outer, inner)

    consumer = result['world']['consumer']
    producer = result['world']['hole']['proc']

    # Both ends of the join are wired.
    assert 'growth_rate' in consumer.inputs, 'outer input port left dangling'
    assert 'growth_rate' in producer.outputs, 'filler output port left dangling'

    # Both ends designate the same store.
    consumer_store = _resolve_wire(
        ('world', 'consumer'), consumer.inputs['growth_rate'])
    producer_store = _resolve_wire(
        ('world', 'hole', 'proc'), producer.outputs['growth_rate'])
    assert consumer_store == producer_store, (
        f'join ends disagree: {consumer_store} vs {producer_store}')

    # And that store is reachable in the composed tree, through stores only.
    _assert_wire_lands_on_a_store(result, consumer_store)

    # The filler's already-wired port is untouched.
    assert producer.inputs['nutrient'] == ['nutrient']


def test_compose_wires_survive_realization(core):
    """The wires ``compose`` writes must materialize a real shared store.

    This is the end-to-end form of Task 0: realization is what turns wires
    into stores (``realize.port_merges``), so a wire that resolves on paper
    but not through ``realize`` is still dangling.
    """
    from bigraph_schema.assembly import compose

    outer = core.access({'world': {
        'hole': {'_type': 'site'},
        'consumer': {
            '_type': 'link',
            'address': 'local:edge',
            '_inputs': {'growth_rate': 'float'},
            '_outputs': {'biomass': 'float'},
            'outputs': {'biomass': ['biomass']}}}})

    inner = core.access({'producer': {
        'proc': {
            '_type': 'link',
            'address': 'local:edge',
            '_inputs': {'nutrient': 'float'},
            '_outputs': {'growth_rate': 'float'}}}})

    result = compose(outer, inner)
    schema, _state, _merges = core.realize({}, core.render(result))

    consumer_store = _resolve_wire(
        ('world', 'consumer'),
        result['world']['consumer'].inputs['growth_rate'])

    node = schema
    for step in consumer_store:
        assert isinstance(node, dict) and step in node, (
            f'realization did not materialize {consumer_store} '
            f'(missing {step!r})')
        node = node[step]

    # The join materialized as an actual typed store, reached from both ends.
    assert isinstance(node, Float)


def test_compose_raises_when_the_join_is_inexpressible(core):
    """Wires are relative to a link's parent and cannot ascend, so a join is
    only expressible when the outer link's parent contains the filled site.
    When it does not, ``compose`` must say so rather than emit a wire that
    silently resolves to nothing."""
    from bigraph_schema.assembly import compose

    # The consumer sits in a sibling subtree of the site, so no relative
    # wire from ``left/`` can reach a store under ``right/hole/``.
    outer = core.access({
        'left': {
            'consumer': {
                '_type': 'link',
                '_inputs': {'growth_rate': 'float'},
                '_outputs': {}}},
        'right': {
            'hole': {'_type': 'site'}}})

    inner = core.access({'producer': {
        'proc': {
            '_type': 'link',
            '_inputs': {},
            '_outputs': {'growth_rate': 'float'}}}})

    with pytest.raises(ValueError, match='growth_rate'):
        compose(outer, inner)


# ── fill + admits: one primitive, one filling discipline ────────────

MODEL_FACE = {
    '_type': 'link',
    '_inputs': {'glucose': 'float'},
    '_outputs': {'growth_rate': 'float'}}


def _sorted_body(core, sort, **extra):
    """A one-site document whose site carries ``sort``."""
    return core.access({'study': {
        'model': dict({'_type': 'site', '_sort': sort}, **extra),
        'note': 'string'}})


def _process(core, inputs, outputs):
    return core.access({
        '_type': 'link',
        '_inputs': inputs,
        '_outputs': outputs})


def test_fill_admits_a_conforming_filler(core):
    """A face-sorted site accepts a filler whose outer face conforms."""
    body = _sorted_body(core, MODEL_FACE)
    filler = _process(core, {'glucose': 'float'}, {'growth_rate': 'float'})

    filled = core.fill_sites(body, {'model': filler})

    from bigraph_schema.assembly import interfaces
    assert isinstance(filled['study']['model'], Link)
    assert not isinstance(filled['study']['model'], Site)
    assert interfaces(filled)[0]._places == ()   # the hole is closed


def test_fill_admits_an_over_providing_filler(core):
    """Conformance is structural subtyping: over-providing ports is fine."""
    body = _sorted_body(core, MODEL_FACE)
    generous = _process(
        core,
        {'glucose': 'float', 'oxygen': 'float'},
        {'growth_rate': 'float', 'waste': 'float'})

    filled = core.fill_sites(body, {'model': generous})
    assert 'oxygen' in filled['study']['model']._inputs


def test_fill_rejects_an_under_providing_filler(core):
    """Under-providing is a fill error that names the site and the port."""
    body = _sorted_body(core, MODEL_FACE)
    stingy = _process(core, {'glucose': 'float'}, {'biomass': 'float'})

    with pytest.raises(ValueError, match="site 'model'") as raised:
        core.fill_sites(body, {'model': stingy})
    assert 'growth_rate' in str(raised.value)


def test_fill_rejects_a_shape_mismatched_port(core):
    """A shared port whose type will not ``resolve`` is not admissible."""
    body = _sorted_body(core, MODEL_FACE)
    mistyped = _process(
        core, {'glucose': 'map[float]'}, {'growth_rate': 'float'})

    with pytest.raises(ValueError, match="site 'model'"):
        core.fill_sites(body, {'model': mistyped})


def test_fill_value_site_checks_and_rejects(core):
    """A value-sorted site is decided by ``check``.

    A value is *state*, so filling puts it on the site's sort as that
    sort's default — the filled tree stays a schema, and ``core.fill``
    materializes the value later.
    """
    body = _sorted_body(core, 'float')

    filled = core.fill_sites(body, {'model': 3.5})
    assert isinstance(filled['study']['model'], Float)
    assert filled['study']['model']._default == 3.5
    assert core.fill(filled, {})['study']['model'] == 3.5

    with pytest.raises(ValueError, match="site 'model'"):
        core.fill_sites(body, {'model': 'not a float'})


def test_fill_optional_site_falls_back_to_its_default(core):
    """``optional``/``default`` live on the site, not in a header."""
    body = _sorted_body(core, 'float', _default=1.25)

    filled = core.fill_sites(body, {})
    assert filled['study']['model']._default == 1.25

    # An explicit binding still wins over the default.
    assert core.fill_sites(
        body, {'model': 9.0})['study']['model']._default == 9.0


def test_fill_unsorted_site_admits_anything(core):
    """An unsorted site is a pure Milner hole — no filling constraint."""
    body = core.access({'world': {'hole': {'_type': 'site'}}})
    assert core.fill_sites(body, {'hole': 42})['world']['hole'] == 42


def test_fill_rejects_an_unknown_site_name(core):
    body = _sorted_body(core, 'float')
    with pytest.raises(ValueError, match='no such site'):
        core.fill_sites(body, {'nonexistent': 1.0})


def test_fill_addresses_a_shared_site_name_by_path(core):
    """A key shared by two sites is not a usable address; the error offers
    the path form, which addresses each site unambiguously."""
    body = core.access({
        'left': {'hole': {'_type': 'site'}},
        'right': {'hole': {'_type': 'site'}}})

    with pytest.raises(ValueError, match='more than one site') as raised:
        core.fill_sites(body, {'hole': 1.0})
    assert 'left/hole' in str(raised.value)
    assert 'right/hole' in str(raised.value)

    filled = core.fill_sites(body, {'left/hole': 1.0, 'right/hole': 2.0})
    assert filled['left']['hole'] == 1.0
    assert filled['right']['hole'] == 2.0


def test_fill_rejects_binding_one_site_under_two_addresses(core):
    """The bare key and the path name the same hole — binding both is a
    conflict, not a silent last-wins."""
    body = core.access({'study': {'model': {'_type': 'site'}}})

    assert core.fill_sites(body, {'study/model': 1.0})['study']['model'] == 1.0

    with pytest.raises(ValueError, match='bound twice'):
        core.fill_sites(body, {'model': 1.0, 'study/model': 2.0})


def test_register_sort_overrides_the_default_discipline(core):
    """A sort may decide admissibility for itself."""
    core.register_sort(
        'even', lambda core, site, filler: filler % 2 == 0)
    body = _sorted_body(core, 'even')

    assert core.fill_sites(body, {'model': 4})['study']['model'] == 4
    with pytest.raises(ValueError, match="site 'model'"):
        core.fill_sites(body, {'model': 3})


def test_admits_is_not_formation(core):
    """``admits`` (filling) and ``formation`` (nesting) are two relations.

    ``admits`` is consulted at the site, *before* substitution, while the
    site's ``_sort`` still exists; ``validate_sorting`` walks parent/child
    pairs *after*. The distinction is load-bearing: once a site is filled
    there is no site — and no ``_sort`` — left for ``formation`` to see.
    """
    from bigraph_schema.assembly import admits, interfaces, validate_sorting, Sorting

    body = _sorted_body(core, MODEL_FACE)
    (_path, site), = interfaces(body)[0]._places
    conforming = _process(core, {'glucose': 'float'}, {'growth_rate': 'float'})

    assert admits(core, site, conforming)
    assert not admits(core, site, _process(core, {}, {}))

    filled = core.fill_sites(body, {'model': conforming})
    assert interfaces(filled)[0]._places == ()   # the sort is gone with the site

    # Nesting is still policed separately, and says nothing about filling.
    unconstrained = Sorting(sorts=set(), controls={}, formation=None)
    assert validate_sorting(filled, unconstrained) == []


# ── the composition law ─────────────────────────────────────────────


def test_fill_independent_sites_commutes(core):
    """Filling independent sites is order-independent (a monoid action)."""
    body = core.access({'study': {
        'model': {'_type': 'site'},
        'reference': {'_type': 'site'}}})
    a = _process(core, {'glucose': 'float'}, {'growth_rate': 'float'})
    b = _process(core, {'oxygen': 'float'}, {'biomass': 'float'})

    both = core.fill_sites(body, {'model': a, 'reference': b})
    one_then_other = core.fill_sites(
        core.fill_sites(body, {'model': a}), {'reference': b})
    other_then_one = core.fill_sites(
        core.fill_sites(body, {'reference': b}), {'model': a})

    assert both == one_then_other == other_then_one


def test_partially_filled_document_is_still_fillable(core):
    """A document with sites left open is still a document (still a template)."""
    from bigraph_schema.assembly import is_ground

    body = core.access({'study': {
        'model': {'_type': 'site'},
        'reference': {'_type': 'site'}}})

    partial = core.fill_sites(body, {'model': 1.0})
    assert not is_ground(partial)          # still open at 'reference'
    assert isinstance(partial['study']['reference'], Site)

    ground = core.fill_sites(partial, {'reference': 2.0})
    assert is_ground(ground)


def test_compose_is_fill_with_positional_bindings(core):
    """``compose`` degrades to the same substitution ``fill_sites`` performs.

    Milner's sites are anonymous, so composition pairs them with roots by
    index while ``fill_sites`` pairs them by name; on a single-site document
    the two must agree.
    """
    from bigraph_schema.assembly import compose, merge, barren

    outer = merge(1)
    inner = barren('a')

    assert compose(outer, inner) == core.fill_sites(outer, {'site0': inner['a']})


def test_fill_sites_does_not_shadow_core_bind(core):
    """``Core.bind`` binds a logical key to a target and predates this work;
    the fill primitive must not take its name."""
    import inspect

    assert core.bind is not core.fill_sites
    assert list(inspect.signature(core.bind).parameters) == [
        'schema', 'state', 'raw_key', 'target']
    assert list(inspect.signature(core.fill_sites).parameters) == [
        'body', 'bindings']


def test_non_ground_document_survives_round_trip(core):
    """A template — a document that is not ground — must render and re-access."""
    from bigraph_schema.assembly import is_ground

    body = _sorted_body(core, 'float')
    assert not is_ground(body)

    # ``_sort`` is a schema field, so access resolves it to a schema node;
    # the round-trip claim is that render → access preserves the open site
    # *and* its sort.
    rendered = core.render(body)
    assert rendered['study']['model'] == {'_type': 'site', '_sort': 'float'}

    round_tripped = core.access(rendered)
    assert not is_ground(round_tripped)
    assert isinstance(round_tripped['study']['model'], Site)
    assert core.render(round_tripped) == rendered


# ── cardinality: replication is a reaction ──────────────────────────


def _colony(core, **region):
    """A document with one region marked for replication."""
    return core.access({'colony': {
        'cell': dict({'_control': 'replicate'}, **region),
        'medium': 'float'}})


def test_replicate_expands_a_marked_region(core):
    """n=3 yields three keyed copies of the region's contents."""
    body = _colony(core, mass='float', genome='string')

    replicated = core.replicate(body, {'cell': 3})

    assert sorted(replicated['colony']) == [
        'cell_0', 'cell_1', 'cell_2', 'medium']
    for index in range(3):
        copy_ = replicated['colony'][f'cell_{index}']
        assert sorted(copy_) == ['genome', 'mass']
        assert isinstance(copy_['mass'], Float)
    # The sibling that was never marked is untouched.
    assert isinstance(replicated['colony']['medium'], Float)


def test_replicate_is_deterministic(core):
    """Same input and same count → identical structure, every time."""
    body = _colony(core, mass='float', genome='string')
    assert core.replicate(body, {'cell': 3}) == core.replicate(body, {'cell': 3})


def test_replicate_n_of_one(core):
    """n=1 yields exactly one copy — still renamed, so the shape of a
    document does not depend on its count."""
    replicated = core.replicate(_colony(core, mass='float'), {'cell': 1})
    assert sorted(replicated['colony']) == ['cell_0', 'medium']


def test_replicate_count_defaults_to_the_region(core):
    """A region may carry its own ``_count``; an override wins."""
    body = core.access({'colony': {
        'cell': {'_control': 'replicate', '_count': 2, 'mass': 'float'}}})

    assert sorted(core.replicate(body)['colony']) == ['cell_0', 'cell_1']
    assert sorted(core.replicate(body, {'cell': 3})['colony']) == [
        'cell_0', 'cell_1', 'cell_2']


def test_replicate_drops_the_mark_so_it_reaches_quiescence(core):
    """Copies are unmarked, so replication cannot fire on its own output."""
    from bigraph_schema.assembly import collect_regions

    replicated = core.replicate(_colony(core, mass='float'), {'cell': 2})

    assert collect_regions(replicated) == {}
    assert core.replicate(replicated, {'cell': 5}) == replicated


def test_replicate_materializes_a_per_instance_site(core):
    """A site inside the region is materialized n times and is fillable
    per instance — addressed by path, since the copies share its key."""
    from bigraph_schema.assembly import collect_sites

    body = core.access({'colony': {'cell': {
        '_control': 'replicate',
        'seed': {'_type': 'site', '_sort': 'integer'}}}})

    replicated = core.replicate(body, {'cell': 3})
    addresses = sorted(a for a in collect_sites(replicated) if '/' in a)
    assert addresses == [
        'colony/cell_0/seed', 'colony/cell_1/seed', 'colony/cell_2/seed']

    filled = core.fill_sites(replicated, {
        'colony/cell_0/seed': 1,
        'colony/cell_1/seed': 2,
        'colony/cell_2/seed': 3})
    assert [core.fill(filled, {})['colony'][f'cell_{i}']['seed']
            for i in range(3)] == [1, 2, 3]


def test_replicate_expands_only_the_named_region(core):
    """Two marked siblings expand independently, each by its own count."""
    body = core.access({'colony': {
        'cell': {'_control': 'replicate', 'mass': 'float'},
        'vessel': {'_control': 'replicate', 'volume': 'float'}}})

    replicated = core.replicate(body, {'cell': 2, 'vessel': 3})

    assert sorted(replicated['colony']) == [
        'cell_0', 'cell_1', 'vessel_0', 'vessel_1', 'vessel_2']
    assert 'mass' in replicated['colony']['cell_0']
    assert 'volume' in replicated['colony']['vessel_0']


def test_replicate_rejects_a_bad_count(core):
    body = _colony(core, mass='float')
    for bad in (-1, 2.5, 'three'):
        with pytest.raises(ValueError, match='non-negative int'):
            core.replicate(body, {'cell': bad})

    from bigraph_schema.assembly import MAX_REPLICAS
    with pytest.raises(ValueError, match='MAX_REPLICAS'):
        core.replicate(body, {'cell': MAX_REPLICAS + 1})


def test_replicate_rule_is_a_shared_parameter_reaction(core):
    """The mechanism is Milner's parametric rule: every reactum site
    instantiates from the *same* redex site (§8.1), which is what makes
    the copies copies."""
    from bigraph_schema.assembly import replicate_rule

    rule = replicate_rule('cell', 3)
    assert set(rule.instantiation.values()) == {'contents'}
    assert sorted(rule.reactum) == ['cell_0', 'cell_1', 'cell_2']


# ── build: the template convenience ─────────────────────────────────


def test_build_is_the_litmus_test(core):
    """A template with a face-sorted model site, a value site, and a
    replicated region: drop in a conforming process and get a ground,
    runnable document — no code."""
    from bigraph_schema.assembly import interfaces

    template = core.access({'study': {
        'model': {'_type': 'site', '_sort': MODEL_FACE},
        'timestep': {'_type': 'site', '_sort': 'float', '_default': 1.0},
        'seeds': {
            '_control': 'replicate',
            'seed': {'_type': 'site', '_sort': 'integer'}}}})

    model = _process(core, {'glucose': 'float'}, {'growth_rate': 'float'})

    schema, state = core.build(template, {
        'model': model,
        'seeds': 2,
        'study/seeds_0/seed': 7,
        'study/seeds_1/seed': 8})

    assert sorted(schema['study']) == [
        'model', 'seeds_0', 'seeds_1', 'timestep']
    assert interfaces(schema)[0]._places == ()      # ground: no open sites
    assert state['study']['timestep'] == 1.0        # the site's default
    assert state['study']['seeds_0']['seed'] == 7
    assert state['study']['seeds_1']['seed'] == 8


def test_build_reports_an_unfilled_required_site(core):
    template = core.access({'study': {
        'model': {'_type': 'site', '_sort': MODEL_FACE},
        'timestep': {'_type': 'site', '_sort': 'float'}}})

    with pytest.raises(ValueError, match='not ground') as raised:
        core.build(template, {})
    assert 'study/model' in str(raised.value)
    assert 'study/timestep' in str(raised.value)


def test_build_reports_a_non_conforming_filler(core):
    template = core.access({'study': {
        'model': {'_type': 'site', '_sort': MODEL_FACE}}})
    wrong = _process(core, {'glucose': 'float'}, {'biomass': 'float'})

    with pytest.raises(ValueError, match="site 'model'"):
        core.build(template, {'model': wrong})


def test_build_of_a_ground_document_is_the_identity_on_shape(core):
    """A template with no sites and no marks is already a document."""
    template = core.access({'cell': {'mass': 'float'}})
    schema, state = core.build(template)
    assert schema == template
    assert state == {'cell': {'mass': 0.0}}


def test_build_with_an_addressed_filler(core):
    """The real case: a filler that names an address — every registered
    process does — builds to a runnable document with a live instance."""
    template = core.access({'study': {
        'model': {'_type': 'site', '_sort': MODEL_FACE}}})
    addressed = core.access({
        '_type': 'link',
        'address': 'local:edge',
        '_inputs': {'glucose': 'float'},
        '_outputs': {'growth_rate': 'float'}})

    schema, state = core.build(template, {'model': addressed})

    assert schema['study']['model'].address._default == {
        'protocol': 'local', 'data': 'edge'}
    assert state['study']['model']['address'] == {
        'protocol': 'local', 'data': 'edge'}
    assert isinstance(state['study']['model']['instance'], Edge)


# ── addresses are values, not type expressions ──────────────────────


@pytest.mark.parametrize('address', [
    'local:edge', 'edge', {'protocol': 'local', 'data': 'edge'}])
def test_access_compiles_an_address_to_a_protocol(core, address):
    """Every spelling of an address compiles to the same ``Protocol``.

    ``'local:edge'`` must not be routed through the type parser — it parses
    under the named-parameter grammar and yields ``{'local': 'edge'}``,
    which no consumer can read.
    """
    from bigraph_schema.schema import Protocol

    link = core.access({
        '_type': 'link',
        'address': address,
        '_inputs': {'x': 'float'},
        '_outputs': {'y': 'float'}})

    assert isinstance(link.address, Protocol)
    assert link.address._default == {'protocol': 'local', 'data': 'edge'}


def test_access_and_realize_agree_on_an_address(core):
    """``access`` and ``realize`` must compile the same link identically —
    they are two doors onto one document."""
    declaration = {
        '_type': 'link', 'address': 'local:edge',
        '_inputs': {'x': 'float'}, '_outputs': {'y': 'float'}}

    accessed = core.access(declaration)
    realized_schema, _state, _merges = core.realize({}, {'p': declaration})

    assert accessed.address._default == realized_schema['p'].address._default


@pytest.mark.parametrize('declaration', [
    {'_type': 'link', 'address': 'local:edge',
     '_inputs': {'x': 'float'}, '_outputs': {'y': 'float'}},
    {'_type': 'link',
     '_inputs': {'x': 'float'}, '_outputs': {'y': 'float'}},
    {'_type': 'link', 'address': 'local:edge', 'inputs': {'x': ['store']},
     '_inputs': {'x': 'float'}, '_outputs': {'y': 'float'}},
])
def test_link_round_trips_through_render(core, declaration):
    """``render`` is the inverse of ``access``: an address survives, and a
    link that never declared one does not acquire a stand-in."""
    accessed = core.access(declaration)
    rendered = core.render(accessed)

    assert core.access(rendered) == accessed
    if 'address' in declaration:
        assert rendered['address'] == 'local:edge'
    else:
        assert 'address' not in (rendered if isinstance(rendered, dict) else {})


class LevelProcess(Edge):
    """A registered edge that declares its ports on the class, the way every
    real process does — the declaration names an address, not an interface."""

    def inputs(self):
        return {'level': 'float'}

    def outputs(self):
        return {'level': 'float'}


def test_admits_resolves_a_filler_face_from_its_address(core):
    """A real process declaration carries an address and a config, not its
    ports. ``admits`` must resolve the face from the registered class, or a
    site could only ever be filled by a declaration restating its own
    interface — which no registered process does."""
    core.register_link('LevelProcess', LevelProcess)

    template = core.access({'study': {'model': {
        '_type': 'site',
        '_sort': {'_type': 'link',
                  '_inputs': {'level': 'float'},
                  '_outputs': {'level': 'float'}}}}})

    # No _inputs/_outputs declared anywhere — only the address.
    filler = core.access({'_type': 'link', 'address': 'local:LevelProcess'})
    assert filler._inputs == {}

    filled = core.fill_sites(template, {'model': filler})
    assert filled['study']['model'] is not None

    # An address naming a class with the wrong face is still rejected.
    core.register_link('EmptyProcess', Edge)
    wrong = core.access({'_type': 'link', 'address': 'local:EmptyProcess'})
    with pytest.raises(ValueError, match="site 'model'"):
        core.fill_sites(template, {'model': wrong})


# ── place semantics: a site takes one root, at the site's position ──


def test_fill_places_a_filler_at_the_site_position(core):
    """One site, one filler, at the site's own path.

    Evidence (recorded here because it settles the open question): handed a
    template whose ``model`` site is filled by a real registered process,
    ``process_bigraph.Composite`` reports
    ``process_paths == [('study', 'model')]`` — it expects the process node
    to sit exactly where the site was.
    """
    core.register_link('LevelProcess', LevelProcess)
    template = core.access({'study': {
        'model': {'_type': 'site'},
        'level': 'float'}})
    filler = core.access({'_type': 'link', 'address': 'local:LevelProcess'})

    filled = core.fill_sites(template, {'model': filler})

    assert isinstance(filled['study']['model'], Link)
    assert sorted(filled['study']) == ['level', 'model']


def test_fill_does_not_splice_a_multi_root_filler(core):
    """A multi-root filler nests **under** the site; it is not spliced into
    the site's parent.

    Splicing would (a) drop the site's key, so the filled region loses the
    name the template gave it, and (b) merge the filler's roots into the
    parent's namespace — here the filler's own ``level`` store would collide
    with the template's ``study/level`` and one of the two would be silently
    lost. Milner is the same answer from the other side: composition plugs
    **one root into one site**, so a filler with n roots fills n sites, not
    one — ``compose`` already enforces that arity.
    """
    template = core.access({'study': {
        'model': {'_type': 'site'},
        'level': 'float'}})
    multi = core.access({
        'A': {'_type': 'link', 'address': 'local:edge'},
        'B': {'_type': 'link', 'address': 'local:edge'},
        'level': 'float'})

    filled = core.fill_sites(template, {'model': multi})

    assert sorted(filled['study']) == ['level', 'model']
    assert sorted(filled['study']['model']) == ['A', 'B', 'level']
    # The filler's store and the template's store stay distinct.
    assert filled['study']['level'] is not filled['study']['model']['level']


def test_compose_requires_one_root_per_site(core):
    """The arity law that makes splicing wrong: sites and roots match 1:1."""
    from bigraph_schema.assembly import compose, merge, barren, tensor

    with pytest.raises(ValueError, match='faces must match'):
        compose(merge(1), tensor(barren('a'), barren('b')))


def test_rendered_link_realizes(core):
    """A rendered link must survive realization — the round trip is only
    useful if the far end still loads. An address-less link previously
    rendered a bare ``protocol`` schema that ``load_protocol`` rejected."""
    accessed = core.access({'p': {
        '_type': 'link',
        '_inputs': {'x': 'float'},
        '_outputs': {'y': 'float'},
        'inputs': {'x': ['x']}}})

    schema, state, _merges = core.realize({}, core.render(accessed))

    assert isinstance(state['p']['instance'], Edge)
    assert 'x' in schema


def test_compose_atom(core):
    """ion ∘ barren produces a K-atom (ion with site filled)."""
    from bigraph_schema.assembly import ion, barren, compose, interfaces, is_ground
    k = ion(core, 'K', ('x', 'y'))
    filler = barren('root')
    atom = compose(k, filler)
    inner, outer = interfaces(atom)
    assert inner._places == ()  # no more sites
    assert set(outer._names.keys()) == {'x', 'y'}  # names preserved


def test_tensor_disjoint(core):
    """Tensor product merges two schemas with disjoint keys."""
    from bigraph_schema.assembly import barren, tensor, interfaces
    t = tensor(barren('a'), barren('b'))
    _, outer = interfaces(t)
    assert set(outer._places) == {'a', 'b'}


def test_tensor_overlap_raises(core):
    """Tensor rejects schemas with overlapping keys."""
    from bigraph_schema.assembly import barren, tensor
    import pytest
    with pytest.raises(ValueError, match='disjoint'):
        tensor(barren('a'), barren('a'))


def test_compose_identity_on_schemas(core):
    """Category law C3: g ∘ id = g for dict schemas.
    A ground schema composed with EPSILON returns itself."""
    from bigraph_schema.assembly import compose, EPSILON
    g = {'cell': {'mass': core.access('float')}}
    assert compose(g, EPSILON) is g


def test_category_laws(core):
    """Verify the categorical axioms from Milner Defs. 2.8–2.11.

    These are the algebraic laws that make bigraph schemas a symmetric
    partial monoidal (spm) category with composition (∘) and tensor
    product (⊗). All laws are tested on elementary bigraphs built
    from the constructors in assembly.py.
    """
    from bigraph_schema.assembly import (
        interfaces, barren, merge, ion, tensor, compose, EPSILON)

    # ── C2: associativity of composition ──
    # h ∘ (g ∘ f) = (h ∘ g) ∘ f
    #
    # f: ε → ⟨2⟩   (two barren roots, ground)
    # g: ⟨2⟩ → ⟨1⟩ (merge two sites into one root)
    # h: ⟨1⟩ → ⟨1⟩ (one site wrapped in structure)
    f = tensor(barren('a'), barren('b'))
    g = merge(2)
    h = {'wrapper': {'content': Site(), 'extra': core.access('string')}}

    assert compose(h, compose(g, f)) == compose(compose(h, g), f), \
        'C2 (associativity): h∘(g∘f) != (h∘g)∘f'

    # ── C3: identity laws ──
    # id ∘ f = f  and  f ∘ id = f
    # For dict schemas, EPSILON (empty Interface) is the identity.
    schema = {'cell': {'mass': core.access('float')}}
    assert compose(schema, EPSILON) is schema, \
        'C3 (right identity): f∘id != f'
    assert compose(EPSILON, schema) is schema, \
        'C3 (left identity): id∘f != f'

    # ── M1: associativity of tensor ──
    # f ⊗ (g ⊗ h) = (f ⊗ g) ⊗ h
    a, b, c = barren('x'), barren('y'), barren('z')
    assert tensor(a, tensor(b, c)) == tensor(tensor(a, b), c), \
        'M1 (tensor associativity)'

    # ── M2: tensor unit ──
    # {} ⊗ f = f  and  f ⊗ {} = f
    assert tensor({}, barren('a')) == barren('a'), \
        'M2 (left unit): {}⊗f != f'
    assert tensor(barren('a'), {}) == barren('a'), \
        'M2 (right unit): f⊗{} != f'

    # ── M3: interchange (tensor commutes with composition) ──
    # (f₁ ⊗ g₁) ∘ (f₀ ⊗ g₀) = (f₁ ∘ f₀) ⊗ (g₁ ∘ g₀)
    #
    # f₀ = barren('a'), g₀ = barren('b')  (ground)
    # f₁ = {fa: {hole: Site()}}, g₁ = {gb: {hole: Site()}}
    f0 = barren('a')
    g0 = barren('b')
    f1 = {'fa': {'hole': Site()}}
    g1 = {'gb': {'hole': Site()}}

    lhs = compose(tensor(f1, g1), tensor(f0, g0))
    rhs = tensor(compose(f1, f0), compose(g1, g0))
    assert lhs == rhs, \
        'M3 (interchange): (f₁⊗g₁)∘(f₀⊗g₀) != (f₁∘f₀)⊗(g₁∘g₀)'

    # ── S1–S4: symmetry ──
    # In our dict-based model, tensor is dict merge and dict keys are
    # unordered (semantically). The symmetry γ_{I,J} is the identity —
    # swapping the order of tensor factors doesn't change the dict.
    # So S1–S4 are trivially satisfied:
    assert tensor(barren('a'), barren('b')) == tensor(barren('b'), barren('a')), \
        'S (symmetry): a⊗b should equal b⊗a for dicts'


def test_control_status_activity(core):
    """Dynamic control status (Milner Def. 8.2): reactions can only fire
    at locations where every ancestor has an active control. Controls
    are matched by type name or by dict key name."""
    from bigraph_schema.assembly import is_active, ACTIVE, PASSIVE, ATOMIC

    building = {
        'building': {
            'room': {
                'agent': core.access('float'),
                'computer': core.access('float')}}}

    # All active by default
    assert is_active(building, ('building', 'room', 'agent'))

    # Room is passive — agent inside is blocked
    status = {'room': PASSIVE}
    assert not is_active(building, ('building', 'room', 'agent'), status)

    # Building itself is still reachable
    assert is_active(building, ('building',), status)

    # Atomic control — Link ports typed 'float' are atomic leaves
    assert is_active(building, ('building', 'room'), {'float': ATOMIC})
    # The float node itself is reachable but is atomic — nothing below
    # it can host reactions (there's nothing below it anyway).


def test_reaction_rule_construction(core):
    """ReactionRule bundles a redex, reactum, and instantiation map.
    Default instantiation matches sites by key name."""
    from bigraph_schema.assembly import ReactionRule, interfaces

    # Simple rule: a site is replaced by a different structure
    redex = {'slot': Site()}
    reactum = {'slot': Site()}
    rule = ReactionRule(redex=redex, reactum=reactum, label='identity')
    assert rule.label == 'identity'
    assert rule.rate is None
    assert 'slot' in rule.instantiation

    # Rule with explicit instantiation
    redex2 = {'a': Site(), 'b': Site()}
    reactum2 = {'x': Site(), 'y': Site()}
    rule2 = ReactionRule(
        redex=redex2, reactum=reactum2,
        instantiation={'x': 'a', 'y': 'b'},
        rate=1.5,
        label='swap')
    assert rule2.instantiation == {'x': 'a', 'y': 'b'}
    assert rule2.rate == 1.5


def test_reaction_rule_built_environment(core):
    """Milner Ch. 1 (p. 8): rules B1–B3 for the built environment.

    B1: agent leaves a conference call (unlinking).
    B2: agent connects to a computer (linking).
    B3: agent enters a room (place change).

    We encode B3 here since it's the most interesting — it changes
    the place graph by moving the agent inside the room.
    """
    from bigraph_schema.assembly import ReactionRule, interfaces, compose

    # B3 redex: agent and room are siblings in the same building.
    # The room has a site for its existing contents.
    redex = {'agent': Site(), 'room': {'content': Site()}}
    # B3 reactum: agent is now inside the room.
    reactum = {'room': {'content': Site(), 'agent': Site()}}

    b3 = ReactionRule(redex=redex, reactum=reactum, label='B3')

    # The instantiation preserves both parameters: the agent subtree
    # fills 'agent' in the reactum, and existing room content fills
    # 'content' in the reactum.
    assert b3.instantiation.get('agent') == 'agent'
    assert b3.instantiation.get('content') == 'content'

    # Redex has 2 sites (agent + room content), reactum has 2 sites
    ri, _ = interfaces(b3.redex)
    qi, _ = interfaces(b3.reactum)
    assert len(ri._places) == 2
    assert len(qi._places) == 2


def test_find_matches_typed(core):
    """Matching uses schema types as controls: Float() in the redex
    matches Float() schema nodes and runtime float values."""
    from bigraph_schema.assembly import find_matches
    from bigraph_schema.schema import Float, Integer

    # Schema-against-schema
    schema = {'x': Float(), 'y': Integer(), 'sub': {'z': Float()}}
    matches = find_matches(schema, {'f': Float()})
    paths = [m.path for m in matches]
    assert () in paths       # x is Float at top level
    assert ('sub',) in paths  # z is Float inside sub

    # Schema-against-runtime-value
    state = {'temp': 22.5, 'count': 3, 'name': 'alice'}
    matches2 = find_matches(state, {'val': Float()})
    assert len(matches2) == 1
    assert matches2[0].key_map['val'] == 'temp'


def test_find_matches_controls(core):
    """_control annotations discriminate matching so that only nodes
    with the right control are matched."""
    from bigraph_schema.assembly import find_matches

    state = {
        'bldg': {
            '_control': 'building',
            'ag': {'_control': 'agent', 'mass': 70.0},
            'rm': {'_control': 'room', 'temp': 22.0}}}

    # Only match agent+room pairs (not building+agent, etc.)
    redex = {
        'a': {'_control': 'agent', 'data': Site()},
        'r': {'_control': 'room', 'stuff': Site()}}
    matches = find_matches(state, redex)
    assert len(matches) == 1
    assert matches[0].path == ('bldg',)
    assert matches[0].key_map['a'] == 'ag'
    assert matches[0].key_map['r'] == 'rm'


def test_fire_rule_b3(core):
    """End-to-end B3: agent enters room. The agent moves from being
    a sibling of the room to being inside it."""
    from bigraph_schema.assembly import ReactionRule, fire_rule

    state = {
        'bldg': {
            '_control': 'building',
            'alice': {'_control': 'agent', 'mass': 70.0},
            'lab': {
                '_control': 'room',
                'bob': {'_control': 'agent', 'mass': 80.0},
                'pc': {'_control': 'computer', 'cpu': 3.0}}}}

    redex = {
        'a': {'_control': 'agent', 'props': Site()},
        'r': {'_control': 'room', 'contents': Site()}}
    reactum = {
        'r': {'_control': 'room',
              'contents': Site(),
              'a': {'_control': 'agent', 'props': Site()}}}
    b3 = ReactionRule(
        redex=redex, reactum=reactum,
        instantiation={'props': 'props', 'contents': 'contents'},
        label='B3')

    new_state, match = fire_rule(state, b3)
    assert match is not None
    # Alice should have moved inside the room. Reactum keys are
    # remapped to original state keys: 'r' → 'lab', 'a' → 'alice'.
    bldg = new_state['bldg']
    assert 'alice' not in bldg  # no longer a sibling
    room = bldg['lab']          # room keeps its original key
    assert 'alice' in room      # agent inside room, original key
    # Per Milner's bigraph semantics, a Site is a hole in the place
    # graph that gets filled with a region; the region's children
    # become siblings at the slot, with their original identifiers
    # preserved by the merged key map. So:
    #   - alice's `mass` field stays as `mass` on alice (not nested
    #     under the redex Site name `props`).
    assert room['alice']['mass'] == 70.0
    assert 'props' not in room['alice']
    #   - The room's other contents (bob, pc) become direct siblings
    #     of alice, again with their original keys (not nested under
    #     the redex Site name `contents`).
    assert 'bob' in room and room['bob']['_control'] == 'agent'
    assert 'pc' in room and room['pc']['_control'] == 'computer'
    assert 'contents' not in room


def test_fire_rule_no_match(core):
    """fire_rule returns (state, None) when the redex doesn't match."""
    from bigraph_schema.assembly import ReactionRule, fire_rule

    state = {'x': 1.0}
    rule = ReactionRule(
        redex={'a': {'_control': 'agent', 'd': Site()}},
        reactum={'a': {'_control': 'agent', 'd': Site()}},
        label='noop')
    new_state, match = fire_rule(state, rule)
    assert match is None
    assert new_state is state


def test_linkvar_match_equality(core):
    """Two LinkVars with the same name in a redex must bind to the
    same wire path. Encodes the link-graph constraint
    "panel.auth and person.badge share an edge".
    """
    from bigraph_schema.assembly import (
        ReactionRule, LinkVar, find_matches)

    # bob's badge and the panel's auth wire to the same anchor —
    # this is what "linked" looks like at the runtime level.
    state = {
        'office': {
            '_control': 'Room',
            'panel': {
                '_control': 'CtrlPanel',
                'outputs': {'auth': ['..', '_edges', 'e1']}},
            'bob': {
                '_control': 'PersonSecured',
                'name': 'bob',
                'outputs': {'badge': ['..', '_edges', 'e1']}}}}

    redex = {
        'panel': {
            '_control': 'CtrlPanel',
            'outputs': {'auth': LinkVar('e')}},
        'p': {
            '_control': 'PersonSecured',
            'name': Site(),
            'outputs': {'badge': LinkVar('e')}}}

    matches = find_matches(state, redex)
    assert len(matches) == 1
    edges = matches[0].bindings.get('__edges__', {})
    assert edges['e'] == ['..', '_edges', 'e1']

    # Now break the link: bob points to a different anchor.
    state2 = {
        'office': {
            '_control': 'Room',
            'panel': {
                '_control': 'CtrlPanel',
                'outputs': {'auth': ['..', '_edges', 'e1']}},
            'bob': {
                '_control': 'PersonSecured',
                'name': 'bob',
                'outputs': {'badge': ['..', '_edges', 'OTHER']}}}}
    matches2 = find_matches(state2, redex)
    assert matches2 == [], 'mismatched wires must not match'


def test_linkvar_match_unconstrained_when_absent(core):
    """A redex that doesn't mention `outputs` doesn't constrain the
    wires on the state node — links are only checked where the redex
    asks for them.
    """
    from bigraph_schema.assembly import ReactionRule, find_matches

    state = {
        'office': {
            '_control': 'Room',
            'bob': {
                '_control': 'Person',
                'name': 'bob',
                # Bob has wires, but redex doesn't ask about them
                'outputs': {'badge': ['..', '_edges', 'e1']}}}}
    redex = {
        'p': {'_control': 'Person', 'name': Site()}}
    matches = find_matches(state, redex)
    assert len(matches) == 1


def test_linkvar_substitutes_in_reactum(core):
    """When the reactum uses a bound LinkVar, the captured wire
    path flows through to the new node — modelling "the moved
    Person keeps its CtrlPanel link" or "both panel and Person
    end up wired to the same edge".
    """
    from bigraph_schema.assembly import (
        ReactionRule, LinkVar, fire_rule)

    state = {
        'office': {
            '_control': 'Room',
            'panel': {
                '_control': 'CtrlPanel',
                'outputs': {'auth': ['..', '_edges', 'e1']}},
            'bob': {
                '_control': 'PersonSecured',
                'name': 'bob',
                'outputs': {'badge': ['..', '_edges', 'e1']}}}}

    # Redex captures the link between panel.auth and bob.badge.
    # Reactum keeps both endpoints on that same edge — i.e. an
    # identity-like rewrite that asserts the link survives.
    rule = ReactionRule(
        redex={
            'panel': {
                '_control': 'CtrlPanel',
                'outputs': {'auth': LinkVar('e')}},
            'p': {
                '_control': 'PersonSecured',
                'name': Site(),
                'outputs': {'badge': LinkVar('e')}}},
        reactum={
            'panel': {
                '_control': 'CtrlPanel',
                'outputs': {'auth': LinkVar('e')}},
            'p': {
                '_control': 'PersonSecured',
                'name': Site(),
                'outputs': {'badge': LinkVar('e')}}},
        instantiation={'name': 'name'},
        label='link_identity')

    new_state, match = fire_rule(state, rule)
    assert match is not None
    office = new_state['office']
    assert office['bob']['outputs']['badge'] == ['..', '_edges', 'e1']
    assert office['panel']['outputs']['auth'] == ['..', '_edges', 'e1']


def test_linkvar_fresh_edge_when_unbound(core):
    """A LinkVar that appears only in the reactum (not the redex)
    is interpreted as a *new* edge introduced by the rule. The
    instantiator mints a fresh anchor path — and reuses it across
    every occurrence of the same fresh variable, so both new
    endpoints end up on the same edge.
    """
    from bigraph_schema.assembly import (
        ReactionRule, LinkVar, fire_rule)

    state = {
        'office': {
            '_control': 'Room',
            'door': {'_control': 'Door'},
            'panel': {
                '_control': 'CtrlPanel',
                'outputs': {'auth': ['..', '_edges', 'e1']}},
            'alice': {'_control': 'Person', 'name': 'alice'}}}

    # enter_secure-style rule: a free Person in a Room with a
    # CtrlPanel becomes PersonSecured AND gets a new wire to a
    # fresh edge that the panel ALSO joins.
    rule = ReactionRule(
        redex={
            'r': {
                '_control': 'Room',
                'panel': {
                    '_control': 'CtrlPanel',
                    'outputs': Site()},
                'p': {'_control': 'Person', 'name': Site()},
                'rest': Site()}},
        reactum={
            'r': {
                '_control': 'Room',
                'panel': {
                    '_control': 'CtrlPanel',
                    'outputs': {'auth': LinkVar('fresh')}},
                'p': {
                    '_control': 'PersonSecured',
                    'name': Site(),
                    'outputs': {'badge': LinkVar('fresh')}},
                'rest': Site()}},
        instantiation={'rest': 'rest', 'name': 'name'},
        label='enter_secure')

    new_state, match = fire_rule(state, rule)
    assert match is not None
    office = new_state['office']
    panel_wire = office['panel']['outputs']['auth']
    alice_wire = office['alice']['outputs']['badge']
    # Both endpoints landed on the *same* fresh edge:
    assert panel_wire == alice_wire
    # And the fresh path is anchored under ``_edges`` with a
    # gensym-style id (starts with ``~e_``):
    assert panel_wire[0] == '_edges'
    assert panel_wire[1].startswith('~e_')


def test_absent_redex_rejects_present_key(core):
    """An ``Absent()`` marker on a redex key is a negative
    application condition: the rule must NOT match a state node
    that has that key bound to anything non-empty.
    """
    from bigraph_schema.assembly import (
        ReactionRule, Absent, find_matches)

    rule = ReactionRule(
        redex={
            'panel': {
                '_control': 'CtrlPanel',
                'outputs': Absent()}},
        reactum={
            'panel': {'_control': 'CtrlPanel'}},
        label='only_unbound')

    free_panel = {
        'panel': {'_control': 'CtrlPanel'}}
    bound_panel = {
        'panel': {
            '_control': 'CtrlPanel',
            'outputs': {'auth': ['_edges', 'e1']}}}

    # Free panel: matches.
    assert len(find_matches(free_panel, rule.redex)) == 1
    # Bound panel: ``Absent`` rules it out.
    assert find_matches(bound_panel, rule.redex) == []


def test_absent_treats_empty_dict_as_absent(core):
    """For wire-shaped fields like ``outputs``, an empty dict is
    semantically equivalent to "no port wired" — so ``Absent()``
    should accept it as well as a missing key.
    """
    from bigraph_schema.assembly import (
        ReactionRule, Absent, find_matches)

    rule = ReactionRule(
        redex={'panel': {
            '_control': 'CtrlPanel',
            'outputs': Absent()}},
        reactum={'panel': {'_control': 'CtrlPanel'}},
        label='only_unbound')

    empty_outputs = {
        'panel': {'_control': 'CtrlPanel', 'outputs': {}}}
    assert len(find_matches(empty_outputs, rule.redex)) == 1


def test_bond_create_destroy_cycle(core):
    """Full kinase-substrate-style cycle:

    1. Both panel and person start unbound.
    2. ``enter_secure`` (with ``Absent`` preimage on both ports
       and a fresh ``LinkVar('e')`` reactum) binds them to a new
       edge.
    3. ``leave_secure`` (with shared ``LinkVar('e')`` redex and
       no ``outputs`` in the reactum) consumes the bond and
       returns both nodes to the unbound state.

    The structural property under test: after firing both rules
    in sequence, the state is back to having no port wired to the
    minted edge — the bond is fully destroyed.
    """
    from bigraph_schema.assembly import (
        ReactionRule, LinkVar, Absent, fire_rule)

    state = {
        'office': {
            '_control': 'Room',
            'door': {'_control': 'Door'},
            'panel': {'_control': 'CtrlPanel'},
            'alice': {'_control': 'Person', 'name': 'alice'}}}

    enter = ReactionRule(
        redex={
            'r': {
                '_control': 'Room',
                'panel': {
                    '_control': 'CtrlPanel',
                    'outputs': Absent()},
                'p': {
                    '_control': 'Person',
                    'name': Site(),
                    'outputs': Absent()},
                'rest': Site()}},
        reactum={
            'r': {
                '_control': 'Room',
                'panel': {
                    '_control': 'CtrlPanel',
                    'outputs': {'auth': LinkVar('e')}},
                'p': {
                    '_control': 'PersonSecured',
                    'name': Site(),
                    'outputs': {'badge': LinkVar('e')}},
                'rest': Site()}},
        instantiation={'rest': 'rest', 'name': 'name'},
        label='enter_secure')

    leave = ReactionRule(
        redex={
            'r': {
                '_control': 'Room',
                'panel': {
                    '_control': 'CtrlPanel',
                    'outputs': {'auth': LinkVar('e')}},
                'p': {
                    '_control': 'PersonSecured',
                    'name': Site(),
                    'outputs': {'badge': LinkVar('e')}},
                'rest': Site()}},
        reactum={
            'r': {
                '_control': 'Room',
                'panel': {'_control': 'CtrlPanel'},
                'p': {'_control': 'Person', 'name': Site()},
                'rest': Site()}},
        instantiation={'rest': 'rest', 'name': 'name'},
        label='leave_secure')

    bound, match = fire_rule(state, enter)
    assert match is not None
    panel_wire = bound['office']['panel']['outputs']['auth']
    alice_wire = bound['office']['alice']['outputs']['badge']
    assert panel_wire == alice_wire  # bond formed

    free, match = fire_rule(bound, leave)
    assert match is not None
    panel_after = free['office']['panel']
    alice_after = free['office']['alice']
    # Both ports gone: the bond was consumed.
    assert 'outputs' not in panel_after
    assert 'outputs' not in alice_after
    # And the person reverted to the unbound control.
    assert alice_after['_control'] == 'Person'


def test_run_reactions_deterministic(core):
    """run_reactions in deterministic mode fires rules in order until
    no more matches are found."""
    from bigraph_schema.assembly import ReactionRule, run_reactions
    import random

    state = {
        'bldg': {
            '_control': 'building',
            'a1': {'_control': 'agent', 'mass': 70.0},
            'a2': {'_control': 'agent', 'mass': 80.0},
            'r1': {'_control': 'room', 'temp': 20.0}}}

    b3 = ReactionRule(
        redex={
            'a': {'_control': 'agent', 'props': Site()},
            'r': {'_control': 'room', 'contents': Site()}},
        reactum={
            'r': {'_control': 'room',
                  'contents': Site(),
                  'a': {'_control': 'agent', 'props': Site()}}},
        instantiation={'props': 'props', 'contents': 'contents'},
        label='B3')

    final, events = run_reactions(state, [b3], mode='deterministic')
    # Both agents should have entered the room
    assert len(events) == 2
    bldg = final['bldg']
    assert 'a1' not in bldg
    assert 'a2' not in bldg
    room = bldg['r1']
    # At least one agent is directly inside the room
    agents_inside = [k for k in room if isinstance(room[k], dict)
                     and room[k].get('_control') == 'agent']
    assert len(agents_inside) >= 1


def test_run_reactions_stochastic(core):
    """run_reactions in stochastic mode picks among candidates
    weighted by rate. Seeded RNG for reproducibility."""
    from bigraph_schema.assembly import ReactionRule, run_reactions
    import random

    state = {
        'bldg': {
            '_control': 'building',
            'agent': {'_control': 'agent', 'mass': 70.0},
            'room': {'_control': 'room', 'temp': 20.0}}}

    b3 = ReactionRule(
        redex={
            'a': {'_control': 'agent', 'props': Site()},
            'r': {'_control': 'room', 'contents': Site()}},
        reactum={
            'r': {'_control': 'room',
                  'contents': Site(),
                  'a': {'_control': 'agent', 'props': Site()}}},
        instantiation={'props': 'props', 'contents': 'contents'},
        rate=2.5,
        label='B3')

    rng = random.Random(42)
    final, events = run_reactions(
        state, [b3], mode='stochastic', rng=rng, max_steps=10)
    assert len(events) == 1
    assert events[0].rule_label == 'B3'
    # Agent is now inside the room
    assert 'agent' not in final['bldg']


def test_built_environment_scenario(core):
    """Full built-environment scenario from Milner Ch. 1 (pp. 7-9).

    Two buildings, multiple rooms, agents and computers. B3 fires
    repeatedly until all agents are inside rooms. Invariant: there
    are always exactly the same number of agents (conservation).
    """
    from bigraph_schema.assembly import ReactionRule, run_reactions

    state = {
        'bldg_a': {
            '_control': 'building',
            'alice': {'_control': 'agent', 'mass': 70.0},
            'bob':   {'_control': 'agent', 'mass': 80.0},
            'lab': {
                '_control': 'room',
                'pc1': {'_control': 'computer', 'cpu': 3.0}},
            'office': {
                '_control': 'room',
                'pc2': {'_control': 'computer', 'cpu': 2.5}}},
        'bldg_b': {
            '_control': 'building',
            'carol': {'_control': 'agent', 'mass': 55.0},
            'lounge': {
                '_control': 'room',
                'tv': {'_control': 'computer', 'cpu': 1.0}}}}

    b3 = ReactionRule(
        redex={
            'a': {'_control': 'agent', 'props': Site()},
            'r': {'_control': 'room', 'contents': Site()}},
        reactum={
            'r': {'_control': 'room',
                  'contents': Site(),
                  'a': {'_control': 'agent', 'props': Site()}}},
        instantiation={'props': 'props', 'contents': 'contents'},
        label='B3')

    final, events = run_reactions(state, [b3], max_steps=20)
    assert len(events) == 3, f'expected 3 agents to enter rooms, got {len(events)}'

    # Invariant: count all agents in the final state
    def count_control(d, control):
        n = 0
        if isinstance(d, dict):
            if d.get('_control') == control:
                n += 1
            for v in d.values():
                n += count_control(v, control)
        return n

    assert count_control(final, 'agent') == 3, 'agent conservation violated'
    assert count_control(final, 'room') == 3, 'room conservation violated'
    assert count_control(final, 'computer') == 3, 'computer conservation violated'


def test_sorting_validation(core):
    """Stratified place sorting rejects ill-sorted nesting."""
    from bigraph_schema.assembly import (
        stratified_sorting, validate_sorting, ACTIVE, PASSIVE)

    # CCS-style: p and a alternate
    sorting = stratified_sorting(
        sorts={'p', 'a'},
        phi={'p': 'a', 'a': 'p'},
        controls={
            'alt': {'sort': 'a', 'status': PASSIVE},
            'send': {'sort': 'p', 'status': PASSIVE},
            'get': {'sort': 'p', 'status': PASSIVE}})

    # Well-sorted: alt contains send (a → p via φ)
    good = {'alt1': {'_control': 'alt', 'send1': {'_control': 'send'}}}
    assert validate_sorting(good, sorting) == []

    # Ill-sorted: alt directly contains alt (a → a, but φ(a) = p)
    bad = {'alt1': {'_control': 'alt', 'alt2': {'_control': 'alt'}}}
    violations = validate_sorting(bad, sorting)
    assert len(violations) > 0


def test_binding_locality(core):
    """Links with wires that escape their subtree are unbound."""
    from bigraph_schema.assembly import is_bound, find_unbound_links

    schema = core.access({
        'cell': {
            'proc': {
                '_type': 'link',
                '_inputs': {'x': 'float'},
                '_outputs': {'y': 'delta'},
                'inputs': {'x': ['cell', 'mass']},
                'outputs': {'y': ['other', 'sink']}}}})

    assert not is_bound(schema, ('cell', 'proc'))

    escapes = find_unbound_links(schema)
    assert len(escapes) == 1
    assert escapes[0][1] == 'y'

    # Fully bound link
    schema2 = core.access({
        'cell': {
            'proc': {
                '_type': 'link',
                '_inputs': {'x': 'float'},
                '_outputs': {'y': 'delta'},
                'inputs': {'x': ['cell', 'mass']},
                'outputs': {'y': ['cell', 'out']}}}})
    assert is_bound(schema2, ('cell', 'proc'))
    assert find_unbound_links(schema2) == []


def test_ccs_brs(core):
    """CCS: matching send/get on a channel synchronises."""
    from bigraph_schema.calculi import ccs_brs
    from bigraph_schema.assembly import run_reactions

    sorting, rules, state = ccs_brs(channels=('x',))
    final, events = run_reactions(state, rules)
    assert len(events) == 1
    assert events[0].rule_label == 'CCS sync on x'
    # Both alternations consumed
    assert 'proc' not in final
    assert 'listener' not in final


def test_ambient_brs(core):
    """Mobile Ambients: A1 rule moves amb_x inside amb_y."""
    from bigraph_schema.calculi import ambient_brs
    from bigraph_schema.assembly import run_reactions

    sorting, rules, state = ambient_brs(names=('x', 'y'))
    final, events = run_reactions(state, rules)
    assert len(events) == 1
    assert events[0].rule_label == 'A1: in_y'
    # amb_x is now inside amb_y
    assert 'amb_x' not in final
    assert 'amb_x' in final['amb_y']


def test_petri_brs(core):
    """Petri Net: firing flips M→U and U→M."""
    from bigraph_schema.calculi import petri_brs
    from bigraph_schema.assembly import run_reactions

    sorting, rules, state = petri_brs(events=[('e1', 2, 1)])
    final, events = run_reactions(state, rules)
    assert len(events) == 1
    # Pre-conditions: M → U
    for key in ('c1', 'c2'):
        # After remapping, keys may have changed
        found = False
        for k, v in final.items():
            if isinstance(v, dict) and v.get('_control') == 'U':
                found = True
        assert found or True  # relaxed — check counts instead

    # Count controls
    m_count = sum(1 for v in final.values()
                  if isinstance(v, dict) and v.get('_control') == 'M')
    u_count = sum(1 for v in final.values()
                  if isinstance(v, dict) and v.get('_control') == 'U')
    assert m_count == 1, f'expected 1 M after firing, got {m_count}'
    assert u_count == 2, f'expected 2 U after firing, got {u_count}'


def test_pi_brs(core):
    """π-calculus: synchronisation on channel x passes name y from
    sender to receiver. The sent name appears as 'received_name'
    in the result."""
    from bigraph_schema.calculi import pi_brs
    from bigraph_schema.assembly import run_reactions

    sorting, rules, state = pi_brs(channels=('x',))
    final, events = run_reactions(state, rules)
    assert len(events) == 1
    assert events[0].rule_label == 'π sync on x'
    # The sent name y should appear in the result. The redex captures
    # the receiver's continuation under a Site that is then filled at
    # the top level — its children (the_name, data) splice into the
    # surrounding region as siblings rather than nesting under the
    # site name (Milner: a site is a hole; its filler region's roots
    # become children at the slot position).
    assert final.get('the_name') == 'y'
    assert final.get('data') == 42
    # Both continuations are nil
    assert final.get('send_cont', {}).get('_control') == 'nil'
    assert final.get('recv_cont', {}).get('_control') == 'nil'


def test_interfaces_container_traversal(core):
    """The interfaces() walker descends into container value schemas
    (Map._value, List._element, Tree._leaf, Wrap._value) so that
    Sites and Links inside containers are found."""
    from bigraph_schema.assembly import interfaces

    # Map whose values are Sites → every entry is a hole
    schema = {'pool': core.access({'_type': 'map', '_value': 'site'})}
    inner, _ = interfaces(schema)
    assert len(inner._places) == 1
    path, site = inner._places[0]
    assert path == ('pool', '*')

    # List of Links → ports found with wildcard path
    schema2 = core.access({'procs': {
        '_type': 'list',
        '_element': {
            '_type': 'link',
            '_inputs': {'x': 'float'},
            '_outputs': {'y': 'delta'}}}})
    inner2, outer2 = interfaces(schema2)
    assert 'x' in inner2._names
    assert inner2._names['x'] == ('procs', '*')

    # Maybe wrapping a Site → found at container path (no wildcard)
    schema3 = {'slot': core.access('maybe[site]')}
    inner3, _ = interfaces(schema3)
    assert len(inner3._places) == 1

    # Tree whose leaves are Sites
    schema4 = {'hier': core.access({'_type': 'tree', '_leaf': 'site'})}
    inner4, _ = interfaces(schema4)
    assert len(inner4._places) == 1
    assert inner4._places[0][0] == ('hier', '*')


def _build_fake_discovery_package(root):
    """Materialize a fake installable package tree on disk for the
    package-discovery walk to traverse.

    Layout (under ``root``)::

        fake_disco_pkg/__init__.py        # registers a sentinel marker
        fake_disco_pkg/boom.py            # raises RuntimeError at import
        fake_disco_pkg/good.py            # imports cleanly
        fake_disco_pkg/test_scaffold.py   # must NOT be imported (test_*)
        fake_disco_pkg/testing/__init__.py# must NOT be imported (testing)

    The ``boom`` module mimics the observed failure mode (starlette
    raising ``RuntimeError`` rather than ``ImportError`` when an optional
    test dependency is missing). The ``test_scaffold``/``testing`` modules
    raise on import too, so if the walk *did* import them the test would
    fail loudly rather than silently.
    """
    pkg = root / "fake_disco_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "boom.py").write_text(
        "raise RuntimeError('simulated optional-dep failure at import')\n")
    (pkg / "good.py").write_text("VALUE = 1\n")
    # If the discovery walk ever imports these, they explode — proving a
    # regression in the test-scaffolding skip rule.
    (pkg / "test_scaffold.py").write_text(
        "raise AssertionError('test_* module must not be imported by walk')\n")
    testing = pkg / "testing"
    testing.mkdir()
    (testing / "__init__.py").write_text(
        "raise AssertionError('testing pkg must not be imported by walk')\n")


def test_recursive_import_skips_runtime_error_module(tmp_path, capsys):
    """A submodule raising a non-ImportError (RuntimeError) at import time
    must be caught and skipped so the package walk completes — this is the
    httpx2/starlette crash class that aborted ``build_core()`` downstream."""
    _build_fake_discovery_package(tmp_path)
    sys.path.insert(0, str(tmp_path))
    try:
        core = allocate_core()
        # Must NOT raise even though `boom` raises RuntimeError at import.
        core, edges, types, visited = recursive_dynamic_import(
            core, "fake_disco_pkg")
    finally:
        sys.path.remove(str(tmp_path))
        for name in list(sys.modules):
            if name == "fake_disco_pkg" or name.startswith("fake_disco_pkg."):
                del sys.modules[name]

    out = capsys.readouterr().out
    # (b) a clear warning was emitted naming the module and exception type.
    assert "fake_disco_pkg.boom" in out
    assert "RuntimeError" in out


def test_recursive_import_skips_test_modules(tmp_path):
    """The walk must never import test scaffolding (test_*, tests, testing).
    Those fake modules raise on import, so reaching them would surface as an
    error here; instead they should be silently skipped."""
    _build_fake_discovery_package(tmp_path)
    sys.path.insert(0, str(tmp_path))
    try:
        core = allocate_core()
        recursive_dynamic_import(core, "fake_disco_pkg")
        # The good (non-test) submodule got imported; the test ones did not.
        assert "fake_disco_pkg.good" in sys.modules
        assert "fake_disco_pkg.testing" not in sys.modules
        assert "fake_disco_pkg.test_scaffold" not in sys.modules
    finally:
        sys.path.remove(str(tmp_path))
        for name in list(sys.modules):
            if name == "fake_disco_pkg" or name.startswith("fake_disco_pkg."):
                del sys.modules[name]


def test_should_skip_submodule_rules():
    """Unit-level guard for the test-scaffolding skip predicate."""
    assert _should_skip_submodule("tests")
    assert _should_skip_submodule("testing")
    assert _should_skip_submodule("test_e2e")
    assert not _should_skip_submodule("processes")
    assert not _should_skip_submodule("contest")  # not a prefix match


if __name__ == '__main__':
    core = allocate_core()

    test_infer(core)
    test_render(core)
    test_default(core)
    test_resolve(core)
    test_check(core)
    test_serialize(core)
    test_realize(core)
    test_merge(core)
    test_traverse(core)
    # test_infer_link(core)
    test_generate(core)
    test_generate_coverage(core)
    test_generate_promote_to_struct(core)
    test_uni_schema(core)
    test_list_array_schema(core)
    test_bind(core)

    test_problem_schema_1(core)
    test_problem_schema_2(core)

    test_frame(core)
    test_apply(core)
    test_unify(core)

    test_generate_tuple_default(core)
    test_array(core)
    test_infer_star(core)

    test_access_tuple(core)
    test_serialize_realize_shape(core)

    test_bigraph_structural_types_in_registry(core)
    test_bigraph_structural_types_are_empty(core)
    test_bigraph_structural_bare_round_trip(core)
    test_bigraph_structural_sorted_round_trip(core)
    test_interface_direct_construction(core)
    test_bigraph_structural_types_present_in_base_types(core)
    test_interface_dict_form_access(core)
    test_number_object_function_defaults(core)
    test_assembly_identity_laws()
    test_interfaces_ground_schema(core)
    test_interfaces_with_sites(core)
    test_interfaces_unwired_ports(core)
    test_elementary_barren(core)
    test_elementary_merge(core)
    test_elementary_ion(core)
    test_elementary_substitution_closure(core)
    test_compose_fills_sites(core)
    test_compose_atom(core)
    test_tensor_disjoint(core)
    test_tensor_overlap_raises(core)
    test_compose_identity_on_schemas(core)
    test_interfaces_container_traversal(core)
    test_category_laws(core)
    test_control_status_activity(core)
    test_reaction_rule_construction(core)
    test_reaction_rule_built_environment(core)
    test_find_matches_typed(core)
    test_find_matches_controls(core)
    test_fire_rule_b3(core)
    test_fire_rule_no_match(core)
    test_run_reactions_deterministic(core)
    test_run_reactions_stochastic(core)
    test_built_environment_scenario(core)
    test_sorting_validation(core)
    test_binding_locality(core)
    test_ccs_brs(core)
    test_ambient_brs(core)
    test_petri_brs(core)
    test_pi_brs(core)

    test_resolve_conflict(core)
