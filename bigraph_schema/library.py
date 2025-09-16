import typing
import numpy as np
from numpy import dtype
import pytest
import logging

from plum import dispatch
from parsimonious.nodes import NodeVisitor
from dataclasses import dataclass, is_dataclass, replace

from bigraph_schema.schema import (
    BASE_TYPES,
    resolve_path,
    convert_jump,
    Node,
    Union,
    Tuple,
    Boolean,
    Number,
    Integer,
    Float,
    Delta,
    Nonnegative,
    String,
    Enum,
    Wrap,
    Maybe,
    Overwrite,
    List,
    Map,
    Tree,
    Dtype,
    Array,
    Key,
    Path,
    Wires,
    Schema,
    Edge,
    Jump,
    Star,
    Index,
)

from bigraph_schema.parse import visit_expression
from bigraph_schema.methods import (
    infer,
    render,
    default,
    resolve,
    check,
    serialize,
    deserialize,
    merge,
    jump,
    traverse,
    bind,

    apply)

# view
# project



def schema_keys(schema):
    keys = []
    for key in schema.__dataclass_fields__:
        if key.startswith('_'):
            keys.append(key)

    return keys


@dispatch
def handle_parameters(schema: Tuple, parameters):
    schema._values = parameters
    return schema

@dispatch
def handle_parameters(schema: Enum, parameters):
    schema._values = parameters
    return schema

@dispatch
def handle_parameters(schema: Union, parameters):
    schema._options = parameters
    return schema

@dispatch
def handle_parameters(schema: Map, parameters):
    if len(parameters) == 1:
        schema._value = parameters[0]
    else:
        schema._key, schema._value = parameters
    return schema

@dispatch
def handle_parameters(schema: Array, parameters):
    schema._shape = tuple([
        int(value)
        for value in parameters[0][0]._values])
    schema._data = dtype(parameters[1])

    return schema

@dispatch
def handle_parameters(schema: Edge, parameters):
    schema._inputs = parameters[0]
    schema._outputs = parameters[1]

    return schema

@dispatch
def handle_parameters(schema: Node, parameters):
    keys = schema_keys(schema)[1:]
    for key, parameter in zip(keys, parameters):
        setattr(schema, key, parameter)
    return schema
        
@dispatch
def handle_parameters(schema, parameters):
    return schema


class LibraryVisitor(NodeVisitor):
    """Visitor that walks a parsed tree and builds structured type expressions."""

    def __init__(self, library):
        self.library = library

    def visit_expression(self, node, visit):
        return visit[0]

    def visit_union(self, node, visit):
        head = [visit[0]]
        tail = [tree['visit'][1] for tree in visit[1]['visit']]
        return Union(_options=head + tail)

    def visit_merge(self, node, visit):
        head = [visit[0]]
        tail = [tree['visit'][1] for tree in visit[1]['visit']]
        nodes = head + tail

        if all(isinstance(tree, dict) for tree in nodes):
            merged = {}
            for tree in nodes:
                merged.update(tree)
            return merged
        else:
            return Tuple(_values=nodes)

    def visit_tree(self, node, visit):
        return visit[0]

    def visit_bigraph(self, node, visit):
        return visit[0]

    def visit_group(self, node, visit):
        group_value = visit[1]
        return group_value if isinstance(group_value, (list, tuple, dict)) else (group_value,)

    def visit_nest(self, node, visit):
        return {visit[0]: visit[2]}

    def visit_type_name(self, node, visit):
        schema = visit[0]

        type_parameters = [
            parameter
            for parameter in visit[1]['visit']]

        if type_parameters:
            schema = handle_parameters(
                schema,
                type_parameters[0])
            
        default_visit = visit[2]['visit']
        if default_visit:
            default = default_visit[0]
            if isinstance(schema, Node):
                schema._default = default
            elif isinstance(schema, dict):
                schema['_default'] = default

        return schema

    def visit_parameter_list(self, node, visit):
        first = [visit[1]]
        rest = [inner['visit'][1] for inner in visit[2]['visit']]
        full = first + rest

        return full

    def visit_default_block(self, node, visit):
        return visit[1]

    def visit_default(self, node, visit):
        return node.text

    def visit_symbol(self, node, visit):
        return self.library.access(node.text)

    def visit_nothing(self, node, visit):
        return None

    def generic_visit(self, node, visit):
        return {'node': node, 'visit': visit}


class Library():
    def __init__(self, types):
        self.registry = {}
        self.register_types(types)
        self.parse_visitor = LibraryVisitor(self)

    def register_type(self, key, data):
        if key in self.registry:
            self.update_type(key, data)
        else:
            self.registry[key] = data

    def register_types(self, types):
        for key, data in types.items():
            self.register_type(
                key,
                data)

    def update_type(self, key, data):
        self.registry[key] = deep_merge(
            self.registry[key],
            data)

    def select_fields(self, base, schema):
        select = {}
        for key in base.__dataclass_fields__.keys():
            schema_key = schema.get(key)
            if schema_key:
                if key == '_default':
                    down = schema_key
                else:
                    down = self.access(
                        schema_key)
                select[key] = down

        return select

    def make_instance(self, base, state):
        fields = self.select_fields(base, state)
        instance = base(**fields)

        return instance

    def access(self, key):
        if is_dataclass(key):
            return key

        elif isinstance(key, str):
            if key not in self.registry:
                try:
                    parsed = visit_expression(key, self.parse_visitor)
                    return parsed
                except Exception as e:
                    return key
            else:
                return self.registry[key]()

        elif isinstance(key, dict):
            if '_type' in key:
                type_key = key['_type']
                if not isinstance(type_key, Node):
                    type_key = self.access(type_key)

                fields = {
                    field: self.access(value)
                    for field, value in key.items()
                    if field not in ('_type', '_default')}

                base = replace(type_key, **fields)
                if key.get('_default') is not None:
                    base._default = key['_default']
                return base

            else:
                result = {}
                for subkey in key:
                    if subkey.startswith('_'):
                        result[subkey] = key[subkey]
                    else:
                        result[subkey] = self.access(
                            key[subkey])
                return result

        elif isinstance(key, list):
            return [self.access(element) for element in key]
        else:
            return key

    def blank_context(self, schema, state, path):
        return {
            'schema': schema,
            'state': state,
            'path': (),
            'subpath': path}

    def convert_jump(self, key):
        return convert_jump(key)

    def convert_path(self, path):
        resolved = resolve_path(path)
        return tuple([
            self.convert_jump(key)
            for key in resolved])

    def infer(self, state):
        return infer(self, state)

    def render(self, schema):
        found = self.access(schema)
        return render(found)

    def default(self, schema):
        found = self.access(schema)
        value = default(found)
        return deserialize(found, value)

    def resolve(self, current_schema, update_schema):
        current = self.access(current_schema)
        update = self.access(update_schema)
        return resolve(current, update)

    def check(self, schema, state):
        found = self.access(schema)
        return check(found, state)

    def serialize(self, schema, state):
        found = self.access(schema)
        return serialize(found, state)

    def deserialize(self, schema, state):
        found = self.access(schema)
        return deserialize(found, state)

    def generate(self, schema, state):
        found = self.access(schema)
        inferred = self.infer(state)
        resolved = self.resolve(inferred, found)
        merged = self.default(resolved)

        return resolved, merged

    def jump(self, schema, state, raw_key):
        found = self.access(schema)
        key = self.convert_jump(raw_key)
        context = self.blank_context(found, state, ())

        return jump(found, state, key, context)

    def traverse(self, schema, state, raw_path):
        found = self.access(schema)
        path = self.convert_path(raw_path)
        context = self.blank_context(found, state, path)

        return traverse(found, state, path, context)

    def bind(self, schema, state, raw_key, target):
        found = self.access(schema)
        key = self.convert_jump(raw_key)

        return bind(found, state, key, target)

    def merge(self, schema, state, merge_state):
        found = self.access(schema)
        return merge(found, state, merge_state)

    def apply(self, schema, state, update):
        found = self.access(schema)
        context = {
            schema: found,
            state: state,
            path: ()}
        return apply(schema, state, update, context)



# test data ----------------------------

@pytest.fixture
def core():
    return Library(
        BASE_TYPES)

default_a = 11.111
# represents a hash where keys a and b are required with types specified
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

edge_schema = {
    '_type': 'edge',
    '_inputs': {
        'mass': 'float',
        'concentrations': map_schema},
    '_outputs': {
        'mass': 'delta',
        'concentrations': {
            '_type': 'map',
            '_key': 'string',
            '_value': 'delta'}}}

edge_a = {
    'inputs': {
        'mass': ['cell', 'mass'],
        'concentrations': ['cell', 'internal']},
    'outputs': {
        'mass': ['cell', 'mass'],
        'concentrations': ['cell', 'internal']}}


to_implement = (
    # Node,
    # Union,
    # Tuple,
    # Boolean,
    # Number,
    # Integer,
    # Float,
    # Delta,
    # Nonnegative,
    # String,
    # Enum,
    # Wrap,
    # Maybe,
    # Overwrite,
    # List,
    Map,
    # Tree,
    # Dtype,
    # Array,
    Key,
    # Path,
    # Wires,
    # Schema,
    # Edge,
    Jump,
    Star,
    Index,
)
uni_schema = 'outer:tuple[tuple[boolean],' \
        'enum[a,b,c],' \
        'tuple[integer,delta,nonnegative],' \
        'list[maybe[tree[path]]],' \
        'wrap[maybe[overwrite[integer]]],' \
        'union[edge[x:integer,y:string],float,string],' \
        'path,' \
        'tree[edge[x:(y:float|z:boolean)|y:integer,oo:maybe[string]]],' \
        'wires,' \
        'integer{11},' \
        'a:string|b:float,' \
        'map[a:string|c:float]]|' \
        'outest:string'
        # 'list[maybe[tree[array[(3|4),float]]]],' \
        # 'dtype[a],' \

# tests --------------------------------------

def test_infer(core):
    default_node = core.default(node_schema)
    node_inferred = core.infer(default_node)

    assert check(node_inferred, default_node)

    # print(f"inferred {node_inferred}\nfrom {default_node}")
    # print(f'rendered schema:\n{render(node_inferred)}')

    # assert render(node_inferred)['a'] == node_schema['a']['_type']
    # assert render(node_inferred)['b'] == node_schema['b']['_type']

# render is the inverse of access
def test_render(core):
    node_type = core.access(node_schema)
    node_render = core.render(node_schema)
    assert node_render == render(node_type)

    edge_type = core.access(edge_schema)
    edge_render = core.render(edge_type)

    # can't do the same assertion as above, because two different renderings
    # exist
    assert core.access(edge_render) == edge_type
    assert edge_render == core.render(core.access(edge_render))

    map_type = core.access(map_schema)
    map_render = core.render(map_type)

    assert core.access(map_render) == core.access(map_schema)
    # fixed point is found
    assert map_render == core.render(core.access(map_render))

    uni_type = core.access(uni_schema)
    uni_render = core.render(uni_type)
    round_trip = core.access(uni_render)

    def idx(a, b, n):
        return a['outer']._values[n], b['outer']._values[n]

    assert round_trip == uni_type
    assert uni_render == core.render(core.access(uni_type))


def test_default(core):
    node_type = core.access(node_schema)
    default_node = core.default(node_schema)

    assert 'a' in default_node
    assert isinstance(default_node['a'], float)
    assert default_node['a'] == default_a
    assert 'b' in default_node
    assert isinstance(default_node['b'], str)

    assert core.check(node_schema, default_node)

    value = 11.11
    assert core.default(core.infer(value)) == value


def test_resolve(core):
    float_number = core.resolve('float', 'number')
    assert render(float_number) == 'float'
    assert type(float_number) == BASE_TYPES['float']

    node_resolve = core.resolve(
        {'a': 'delta', 'b': 'node'},
        node_schema)

    # assert render(node_resolve)['a']['_type'] == 'delta'
    # assert render(node_resolve)['a']['_default'] == node_schema['a']['_default']

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
        # print(e)
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

    assert core.check(
        tree_schema,
        tree_a)

    assert core.check(
        tree_parse,
        tree_b)

    assert not core.check(
        tree_schema,
        'not a tree')

    edge_a = {
        'inputs': {
            'mass': ['cell', 'mass'],
            'concentrations': ['cell', 'internal']},
        'outputs': {
            'mass': ['cell', 'mass'],
            'concentrations': ['cell', 'internal']}}

    edge_b = {
        'inputs': 5.0,
        'outputs': {
            'mass': ['cell', 'mass'],
            'concentrations': ['cell', 'internal']}}

    edge_c = {
        'outputs': {
            'mass': ['cell', 'mass'],
            'concentrations': ['cell', 'internal']}}

    edge_d = {
        'inputs': {
            'mass': ['cell', 11.111],
            'concentrations': ['cell', 'internal']},
        'outputs': {
            'mass': ['cell', 'mass'],
            'concentrations': ['cell', 'internal']}}

    assert core.check(edge_schema, edge_a)
    assert not core.check(edge_schema, edge_b)
    assert not core.check(edge_schema, edge_c)
    assert not core.check(edge_schema, edge_d)
    assert not core.check(edge_schema, 44.44444)


def test_serialize(core):
    edge_type = core.access(edge_schema)
    encoded_a = serialize(edge_type, edge_a)

    assert encoded_a == edge_a

    encoded_b = core.serialize(
        {'a': 'float'},
        {'a': 55.55555})

    assert encoded_b['a'] == '55.55555'


def test_deserialize(core):
    encoded_edge = {
        'inputs': {
            'mass': ['cell','mass'],
            'concentrations': '["cell","internal"]'},
        'outputs': '{\
            "mass":["cell","mass"],\
            "concentrations":["cell","internal"]}'}

    decoded = core.deserialize(
        edge_schema,
        encoded_edge)

    assert decoded == edge_a

    schema = {
        'a': 'integer',
        'b': 'tuple[float,string,map[integer]]'}

    code = {
        'a': '5555',
        'b': ('1111.1', "okay", '{"x": 5, "y": "11"}')}

    decode = core.deserialize(
        schema,
        code)

    assert decode['a'] == 5555
    assert decode['b'][2]['y'] == 11


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

    edge_interface = {
        '_type': 'edge',
        '_inputs': puts,
        '_outputs': puts}

    edge_schema = core.access(
        edge_interface)

    edge_state = {
        'inputs': {
            'mass': ['cell', 'mass'],
            'concentrations': ['cell', 'internal']},
        'outputs': {
            'mass': ['cell', 'mass'],
            'concentrations': ['cell', 'internal']}}

    assert core.check(edge_interface, edge_state)

    default_edge = core.default(edge_schema)

    assert default_edge['inputs']['mass'] == ['mass']

    simple_interface = {
        'cell': {
            'mass': 'float',
            'internal': 'map[float]'},
        'edge': edge_interface}

    initial_mass = 11.1111
    simple_graph = {
        'cell': {
            'mass': initial_mass,
            'internal': {
                'A': 3.333,
                'B': 44.44444,
                'C': 5555.555}},
        'edge': edge_state}

    simple_schema = core.access(
        simple_interface)

    down_schema, down_state = core.jump(
        simple_interface,
        simple_graph,
        'edge')

    assert isinstance(down_schema, Edge)
    assert 'inputs' in down_state

    mass_schema, mass_state = core.traverse(
        simple_interface,
        simple_graph,
        ['edge', 'inputs', 'mass'])

    assert isinstance(mass_schema, Float)
    assert mass_state == initial_mass

    concentration_schema, concentration_state = core.traverse(
        simple_interface,
        simple_graph,
        ['edge', 'outputs', 'concentrations', 'A'])

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
        'units': {
            'meters': 11.1111,
            'seconds': 22.833333}}

    generated_schema, generated_state = core.generate(
        schema,
        state)

    assert generated_state['A'] == 0.0
    assert generated_state['B'] == 'one'
    assert generated_state['C'] == 'y'
    assert generated_state['units']['seconds'] == 22.833333

    assert not hasattr(generated_schema['units'], 'meters')

    rendered = core.render(generated_schema)

    assert generated_state == core.deserialize(generated_schema, core.serialize(generated_schema, generated_state))


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

    tree_merge = core.merge(
        'tree[float]',
        tree_b,
        tree_a)

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


def test_apply(core):
    core


if __name__ == '__main__':
    core = Library(
        BASE_TYPES)

    test_infer(core)
    test_render(core)
    test_default(core)
    test_resolve(core)
    test_check(core)
    test_serialize(core)
    test_deserialize(core)
    test_merge(core)
    test_traverse(core)

    test_generate(core)
    test_bind(core)
    test_apply(core)
