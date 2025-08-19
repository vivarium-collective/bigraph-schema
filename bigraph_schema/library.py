import typing
import numpy as np

from plum import dispatch
from parsimonious.nodes import NodeVisitor
from dataclasses import dataclass, is_dataclass

from bigraph_schema.schema import (
    BASE_TYPES,
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

    generate,
    slice,
    bind,
    apply)


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
            
        return schema

    def visit_parameter_list(self, node, visit):
        first = [visit[1]]
        rest = [inner['visit'][1] for inner in visit[2]['visit']]
        full = first + rest
            
        return full

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
                    parse = visit_expression(key, self.parse_visitor)
                    return parse
                except Exception as e:
                    print(f'could not parse:\n{key}\n{e}')
                    return key
            else:
                return self.registry[key]()

        elif isinstance(key, dict):
            if '_type' in key:
                type_key = key.get('_type', 'node')
                base = self.registry[type_key]
                return self.make_instance(base, key)
            else:
                result = {}
                for subkey in key:
                    if subkey.startswith('_'):
                        result[subkey] = key[subkey]
                    else:
                        result[subkey] = self.access(
                            key[subkey])
                return result
        else:
            return key

    def infer(self, state):
        return infer(state)

    def render(self, schema):
        found = self.access(schema)
        return render(found)

    def default(self, schema):
        found = self.access(schema)
        return default(found)

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

    # def generate_nomethod(self, schema, state):
    #     given_schema = self.access(schema)
    #     decode_state = deserialize(given_schema, state)
    #     default_state = default(given_schema)
    #     merged_state = merge(given_schema, default_state, decode_state)

    #     inferred_schema = infer(merged_state)
    #     resolved_schema = resolve(inferred_schema, given_schema)

    #     final_schema, final_state = merge(
    #         resolved_schema,
    #         state,
    #         decoded_state) # ?

    def generate(self, schema, state):
        found = self.access(schema)
        context = {
            schema: found,
            state: state,
            path: ()}
        return generate(found, state, context)

    def traverse(self, schema, state, path):
        found = self.access(schema)
        context = {
            schema: found,
            state: state,
            path: ()}
        return traverse(schema, state, path, context)

    def bind(self, schema, state, key, target):
        pass

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

default_a = 11.111
node_schema = {
    'a': {
        '_type': 'float',
        '_default': default_a},
    'b': {
        '_type': 'string',
        '_default': 'hello world!'}}

edge_schema = {
    '_type': 'edge',
    '_inputs': {
        'mass': 'float',
        'concentrations': {
            '_type': 'map',
            '_key': 'string',
            '_value': 'float'}},
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


# tests --------------------------------------

def test_infer(core):
    default_node = core.default(node_schema)
    node_inferred = core.infer(default_node)

    print(f"inferred {node_inferred}\nfrom {default_node}")
    print(f'rendered schema:\n{render(node_inferred)}')

    assert render(node_inferred)['a'] == node_schema['a']['_type']
    assert render(node_inferred)['b'] == node_schema['b']['_type']


def test_render(core):
    node_type = core.access(node_schema)
    node_render = core.render(node_schema)
    assert node_render == node_schema == render(node_type)


def test_default(core):
    node_type = core.access(node_schema)

    default_node_a = default(node_type)
    default_node_b = core.default(node_schema)

    assert default_node_a == default_node_b

    assert 'a' in default_node_a
    assert isinstance(default_node_a['a'], float)
    assert default_node_a['a'] == default_a
    assert 'b' in default_node_a
    assert isinstance(default_node_a['b'], str)


def test_resolve(core):
    float_number = core.resolve('float', 'number')
    assert render(float_number) == 'float'
    assert type(float_number) == BASE_TYPES['float']

    node_resolve = core.resolve(
        {'a': 'delta', 'b': 'node'},
        node_schema)

    assert render(node_resolve)['a']['_type'] == 'delta'
    assert render(node_resolve)['a']['_default'] == node_schema['a']['_default']

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
        print(e)
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


def test_generate(core):
    core

def test_slice(core):
    core

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

    test_generate(core)
    test_slice(core)
    test_bind(core)
    test_apply(core)
