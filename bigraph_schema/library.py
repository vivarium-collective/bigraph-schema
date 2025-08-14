import typing
import numpy as np

from dataclasses import dataclass, is_dataclass

from bigraph_schema.schema import BASE_TYPES
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
    resolve)


class Library():
    def __init__(self, types):
        self.registry = {}
        self.register_types(types)

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
                # parse
                pass
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

    def check(self, schema, state):
        found = self.access(schema)
        return check(found, state)

    def default(self, schema):
        found = self.access(schema)
        return default(found)

    def serialize(self, schema, state):
        found = self.access(schema)
        return serialize(found, state)


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

    assert library.check(
        tree_schema,
        tree_a)

    assert library.check(
        tree_schema,
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


if __name__ == '__main__':
    test_library()
