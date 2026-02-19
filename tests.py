import pytest

import numpy as np
import pandas as pd

from bigraph_schema import Edge, allocate_core, BASE_TYPES
from bigraph_schema.schema import Float, String, Map, Tree, Link
from bigraph_schema.methods import check, render, serialize


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

    schema, realized = core.realize({}, state)


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
        'b': '',
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

    from_string = core.access('dataframe[a:float|b:integer|c:boolean]')

    assert schema == from_string

    default = core.default(schema)
    frame_schema, frame_state = core.realize(schema, _dict)
    encoded = core.serialize(frame_schema, frame_state)
    realized_schema, realized_state = core.realize(frame_schema, encoded)

    assert realized_state.equals(df)


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

    # test_resolve_conflict(core)
