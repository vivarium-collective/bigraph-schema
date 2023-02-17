import copy
import random
from typing import Any

import numpy as np


class Registry(object):
    def __init__(self):
        """A Registry holds a collection of functions or objects."""
        self.registry = {}
        self.main_keys = []

    def register(self, key, item, alternate_keys=tuple()):
        """Add an item to the registry.

        Args:
            key: Item key.
            item: The item to add.
            alternate_keys: Additional keys under which to register the
                item. These keys will not be included in the list
                returned by ``Registry.list()``.

                This may be useful if you want to be able to look up an
                item in the registry under multiple keys.
        """
        keys = [key]
        keys.extend(alternate_keys)
        for registry_key in keys:
            if registry_key in self.registry:
                if item != self.registry[registry_key]:
                    raise Exception('registry already contains an entry for {}: {} --> {}'.format(
                        registry_key, self.registry[key], item))
            else:
                self.registry[registry_key] = item
        self.main_keys.append(key)

    def access(self, key):
        """Get an item by key from the registry."""
        return self.registry.get(key)

    def list(self):
        return list(self.main_keys)

    def validate(self, item):
        return True


type_schema_keys = [
    '_default', '_apply', '_serialize', '_deserialize', '_description']

def validate_schema(schema):
    

class TypeRegistry(Registry):
    def validate(self, schema):
        return validate_schema(item)


def accumulate(current, update):
    return current + update

def divide_float(value):
    half = value/2.0
    return (half, half)

def divide_int(value):
    half = value // 2
    other_half = half
    if value % 2 == 1:
        other_half += 1
    return half, other_half

type_registry = Registry()

types = {
    'float': {
        '_default': 0.0,
        '_apply': accumulate,
        '_serialize': str,
        '_deserialize': float,
        '_divide': divide_float,
        '_description': '64-bit floating point precision number'
    }, 

    'int': {
        '_default': 0,
        '_apply': accumulate,
        '_serialize': str,
        '_deserialize': int,
        '_divide': divide_int,
        '_description': '64-bit integer'
    }
}

supported_units = {
    'm/s': }

for units in supported_units:
    type_registry.register('m/s', {
        '_default': 0.0,
        '_apply': accumulate,
        '_serialize': str,
        '_deserialize': int
    })

for key, schema in types.items():
    type_registry.register(key, schema)


def test_schema_validate():
    int_schema = {
        'int': {
            '_default': 0,
            '_apply': accumulate,
            '_serialize': str,
            '_deserialize': int,
            '_divide': divide_int,
            '_description': '64-bit integer'
        }
    }

    dimension_schema = {
        'dimension': {
            '_description': 'a two-dimensional value'
            '_divide': custom_divide,
            'width': {'_type': 'int'},
            'height': {'_type': 'int'}
        }
    }

    mitochondria_schema = {
        'mitochondria': {
            'volume': {'_type': 'float'},
            'membrane': {
                'surface_proteins': {'_type': 'branch[protein]'},
                'potential': {'_type': 'microvolts'}},
            'mass': {'_type': 'membrane?'},
        }
    }

    cytoplasm_schema = {
        'cytoplasm': {
            'mitochondria': {'_type': 'branch[mitochondria]'}
            'proteins': {'_type': 'branch[mitochondria]'}
            'nucleus': {'_type': 'branch[mitochondria]'}
            'transcripts': {'_type': 'branch[mitochondria]'}
        }
    }

    cell_schema = {
        'cell': {
            'shape': {'_type': 'mesh'}
            'volume': {'_type': 'mL'}
            'temperature': {'_type': 'K'}
        }
    }

    cell_composite = {
        'environment': {
            'outer_shape': {
                '_type': 'mesh', '_value': []},
            'cellA': {
                'cytoplasm': {
                    'external_ions': {'_type': 'ions'},
                    'internal_ions': {'_type': 'ions'},
                    'electron_transport': {
                        '_type': 'process',
                        '_value': 'ElectronTransport',
                        '_ports': {
                            'external_ions': 'ions',
                            'internal_ions': 'ions'},
                        '_wires': {
                            'external_ions': ['..', 'external_ions'],
                            'internal_ions': ['..', 'internal_ions']}
                        }
                    },
                'inner_shape': {'_type': 'mesh', '_value': []},
                '_ports': {
                    'shape': 'mesh',
                    'volume': 'mL',
                    'temperature': 'K'
                },
                '_channel': {
                    'shape': ['inner_shape'],
                },
                '_wires': {
                    'shape': ['..', 'outer_shape']
                }
            }
        }
    }

    compose({
        'cell': {
            'membrane': cell_schema,
            'cytoplasm': cytoplasm_schema
        }
    }, {
        
    })
