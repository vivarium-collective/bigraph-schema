import typing
import numpy as np

from dataclasses import dataclass, is_dataclass

from bigraph_schema.schema import BASE_TYPES
from bigraph_schema.methods import default, check


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

    def check(self, schema, state):
        found = self.access(schema)
        return check(found, state)



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

    node_type = library.access({
        'a': 'float',
        'b': 'string'})

    default_node = default(node_type)

    assert 'a' in default_node
    assert isinstance(default_node['a'], float)
    assert 'b' in default_node
    assert isinstance(default_node['b'], str)

    edge_type = library.access(edge_schema)

    # import ipdb; ipdb.set_trace()

    assert library.check(
        edge_schema,
        edge_a)

    assert not library.check(
        edge_schema,
        edge_b)

    assert not library.check(
        edge_schema,
        edge_c)

    assert not library.check(
        edge_schema,
        edge_d)

    assert not library.check(
        edge_schema,
        44.44444)


if __name__ == '__main__':
    test_library()
