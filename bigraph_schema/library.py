import typing
import numpy as np

from parsimonious.nodes import NodeVisitor
from dataclasses import dataclass, is_dataclass

from bigraph_schema.schema import BASE_TYPES
from bigraph_schema.parse import visit_expression
from bigraph_schema.methods import (
    default,
    check,
    serialize,
    deserialize,
    generate,
    infer,
    slice,
    bind,
    merge,
    resolve,
    render)


def schema_keys(schema):
    keys = []
    for key in schema.__dataclass_fields__:
        if key.startswith('_'):
            keys.append(key)

    return keys


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
        type_parameters = visit[1]['visit']
        if type_parameters:
            parameters = []
            keys = schema_keys(schema)[1:]
            for key, parameter in zip(keys, type_parameters):
                setattr(schema, key, parameter[0])
            
        return schema

    def visit_parameter_list(self, node, visit):
        first = [visit[1]]
        rest = [inner['visit'][1] for inner in visit[2]['visit']]
        return first + rest

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

    def check(self, schema, state):
        found = self.access(schema)
        return check(found, state)

    def default(self, schema):
        found = self.access(schema)
        return default(found)

    def serialize(self, schema, state):
        found = self.access(schema)
        return serialize(found, state)

    def render(self, schema):
        found = self.access(schema)
        return render(found)

    def resolve(self, current_schema, update_schema):
        current = self.access(current_schema)
        update = self.access(update_schema)
        return resolve(current, update)


def test_library():
    library = Library(
        BASE_TYPES)

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
    tree_type = library.access(
        tree_parse)

    assert library.check(
        tree_schema,
        tree_a)

    assert library.check(
        tree_parse,
        tree_b)

    assert not library.check(
        tree_schema,
        'not a tree')

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

    default_a = 11.111
    node_schema = {
        'a': {
            '_type': 'float',
            '_default': default_a},
        'b': {
            '_type': 'string',
            '_default': 'hello world!'}}

    node_type = library.access(node_schema)
    node_render = render(node_type)
    assert node_render == node_schema

    default_node_a = default(node_type)
    default_node_b = library.default(node_schema)

    assert default_node_a == default_node_b

    assert 'a' in default_node_a
    assert isinstance(default_node_a['a'], float)
    assert default_node_a['a'] == default_a
    assert 'b' in default_node_a
    assert isinstance(default_node_a['b'], str)

    edge_type = library.access(edge_schema)

    assert library.check(edge_schema, edge_a)
    assert not library.check(edge_schema, edge_b)
    assert not library.check(edge_schema, edge_c)
    assert not library.check(edge_schema, edge_d)
    assert not library.check(edge_schema, 44.44444)

    encoded_a = serialize(edge_type, edge_a)

    assert encoded_a == edge_a

    encoded_b = library.serialize(
        {'a': 'float'},
        {'a': 55.55555})

    assert encoded_b['a'] == '55.55555'

    float_number = library.resolve('float', 'number')
    assert render(float_number) == 'float'
    assert type(float_number) == BASE_TYPES['float']

    node_resolve = library.resolve(
        {'a': 'delta', 'b': 'node'},
        node_schema)

    assert render(node_resolve)['a']['_type'] == 'delta'
    assert render(node_resolve)['a']['_default'] == node_schema['a']['_default']

    failed = False

    try:
        library.resolve(
            {'a': 'map[string]', 'b': 'node'},
            node_schema)

    except Exception as e:
        print(e)
        failed = True

    assert failed
    
    node_inferred = library.infer(default_node_a)
    print(f"inferred {node_inferred}\nfrom {default_node_a}")
    assert render(node_inferred)['a'] == node_schema['a']['_type']
    assert render(node_inferred)['b'] == node_schema['b']['_type']


if __name__ == '__main__':
    test_library()
