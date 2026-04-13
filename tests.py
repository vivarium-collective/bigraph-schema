import pytest

import numpy as np
import pandas as pd

from bigraph_schema import Edge, allocate_core, BASE_TYPES
from bigraph_schema.schema import Float, String, Map, Tree, Link, Array, Overwrite, Node
from bigraph_schema.methods import check, render, serialize, apply, reconcile


@pytest.fixture
def core():
    return allocate_core()



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

    basic_schema, basic_state = core.realize(
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

    _, a_instance = core.realize(link_schema, link_a)
    _, b_instance = core.realize(link_schema, link_b)
    _, c_instance = core.realize(link_schema, link_c)

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

    decoded_schema, decoded_state = core.realize(link_schema, encoded_link)

    assert isinstance(decoded_state['instance'], Edge)

    schema = {
        'a': 'integer',
        'b': 'tuple[float,string,map[integer]]'}
    code = {
        'a': '5555',
        'b': ('1111.1', "okay", '{"x": 5, "y": "11"}')}

    decoded_schema, decoded_state = core.realize(schema, code)
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

    generated_schema, generated_state = core.realize(schema, state)

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
        schema, realized = core.realize({}, state)
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
    _, realized = core.realize(schema, state)
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
    _, realized = core.realize(schema, state)
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
    _, state = core.realize('union[boolean,string,float]', True)
    assert state is True


def test_union_realize_str_matches_string(core):
    _, state = core.realize('union[boolean,string,float]', 'hello')
    assert state == 'hello'


def test_union_realize_float_matches_float(core):
    _, state = core.realize('union[boolean,string,float]', 1.5)
    assert state == 1.5


def test_union_realize_int_matches_float_not_boolean(core):
    """A plain int in ``union[boolean,float]`` should realize as Float
    (not Boolean — Boolean now rejects non-bool values)."""
    _, state = core.realize('union[boolean,float]', 7)
    assert state == 7.0
    assert isinstance(state, float)


def test_union_realize_ordering_first_match_wins(core):
    """If two options both accept a value, the first option in the
    declared order wins. Here ``union[float,integer]`` with value 5
    realizes as Float (first match)."""
    _, state = core.realize('union[float,integer]', 5)
    assert isinstance(state, float)
    assert state == 5.0


def test_union_realize_ordering_reverse(core):
    """Swap the order — now Integer wins for int values."""
    _, state = core.realize('union[integer,float]', 5)
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
    _, a = core.realize(schema, {'val': True})
    _, b = core.realize(schema, {'val': 'mass_distribution'})
    _, c = core.realize(schema, {'val': 123.456})
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

    generated_schema, generated_state = core.realize(
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

    generated_schema, generated_state = core.realize(schema, state)

    deschema, destate = core.realize(
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

    generated_schema, generated_state = core.realize(schema, state)
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

    generated_schema, generated_state = core.realize(schema, state)
    serialized = core.serialize(generated_schema, generated_state)

    deschema, destate = core.realize(
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
    frame_schema, frame_state = core.realize(schema, _dict)
    encoded = core.serialize(frame_schema, frame_state)
    realized_schema, realized_state = core.realize(frame_schema, encoded)

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

    test_resolve_conflict(core)
