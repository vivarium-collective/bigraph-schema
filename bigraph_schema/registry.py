import copy
import random
from typing import Any

import numpy as np

from bigraph_schema.parse import parse_type_parameters

required_schema_keys = [
    '_default',
    '_apply',
    '_serialize',
    '_deserialize',
]

optional_schema_keys = [
    '_value',
    '_divide',
    '_description',
    '_ports',
    '_parameters',
    '_super',
]

type_schema_keys = [
    '_default',
    '_value',
    '_apply',
    '_serialize',
    '_deserialize',
    '_divide',
    '_description',
    '_ports',
    '_parameters',
    '_super',
]

# class Function(object):
#     def __init__(self, domain, codomain, fn):
#         self.domain = domain
#         self.codomain = codomain
#         self.fn = fn

#     def __apply__(self, arguments):
#         return None

# divide_longest = Function(Dimension, )


def deep_merge(dct, merge_dct):
    """ Recursive dict merge
    This mutates dct - the contents of merge_dct are added to dct (which is also returned).
    If you want to keep dct you could call it like deep_merge(copy.deepcopy(dct), merge_dct)
    """
    if dct is None:
        dct = {}
    if merge_dct is None:
        merge_dct = {}
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.abc.Mapping)):
            deep_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]
    return dct


def type_merge(dct, merge_dct, path=tuple()):
    """Recursively merge type definitions, never overwrite.
    Args:
        dct: The dictionary to merge into. This dictionary is mutated
            and ends up being the merged dictionary.  If you want to
            keep dct you could call it like
            ``deep_merge_check(copy.deepcopy(dct), merge_dct)``.
        merge_dct: The dictionary to merge into ``dct``.
        path: If the ``dct`` is nested within a larger dictionary, the
            path to ``dct``. This is normally an empty tuple (the
            default) for the end user but is used for recursive calls.
    Returns:
        ``dct``
    """
    for k in merge_dct:
        if not k in dct:
            dct[k] = merge_dct[k]
        elif isinstance(dct[k], dict) and isinstance(
            merge_dct[k], collections.abc.Mapping):

            type_merge(dct[k], merge_dct[k], path + (k,))
        elif not k in type_schema_keys:
            raise ValueError(
                f'cannot merge types at path {path + (k,)}: '
                f'{dct} overwrites {k} from {merge_dct}'
            )
            
    return dct


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
                    raise Exception(
                        'registry already contains an entry for {}: {} --> {}'.format(
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


class TypeRegistry(Registry):
    def __init__(self):
        super().__init__()

        self.supers = {}
        self.register('any', {})

    def register(self, key, item, alternate_keys=tuple()):
        if isinstance(item, dict):
            supers = item.get('_super', ['any']) # list of immediate supers
            if isinstance(supers, str):
                supers = [supers]
            for su in supers:
                assert isinstance(
                    su, str), f"super for {key} must be a string, not {su}"
            self.supers[key] = supers
            for su in supers:
                su_type = self.registry.get(su, {})
                item = type_merge(item, su_type)

        super().register(key, item, alternate_keys)

    def resolve_parameters(self, qualified_type):
        type_name, parameter_types = qualified_type
        outer_type = self.registry.get(type_name)

        if outer_type is None:
            raise ValueError(f'type {qualified_type} is looking for type {type_name} but that is not in the registry')

        parameters = {}
        if '_parameters' in outer_type:
            parameter_names = outer_type['_parameters']
            resolved = [
                self.resolve_parameters(parameter_type)
                for parameter_type in parameter_types
            ]
            parameters = dict(zip(parameter_names, resolved))

        result = {
            '_type': type_name,
        }

        if parameters:
            result['_parameters'] = parameters

        return result


    def access(self, key):
        """Get an item by key from the registry."""
        typ = self.registry.get(key)

        if typ is None:
            parse = parse_type_parameters(key)
            if parse[0] in self.registry:
                typ = self.resolve_parameters(parse)
                print(f'parsed {key} to get {typ}')

        return typ

    def lookup(type_key, attribute):
        return self.access(type_key).get(attribute)

    def is_descendent(self, key, ancestor):
        for sup in self.supers.get(key, []):
            if sup == ancestor:
                return True
            else:
                found = self.is_descendent(sup, ancestor)
                if found:
                    return True
        return False


class RegistryRegistry(Registry):
    def type_attribute(self, type_key, attribute):
        type_registry = self.access('_type')
        type_value = type_registry.access(type_key)
        attribute_key = type_value.get(attribute)
        if attribute_key is not None:
            attribute_registry = self.access(attribute)
            return attribute_registry.access(attribute_key)


apply_registry = Registry()
serialize_registry = Registry()
deserialize_registry = Registry()
divide_registry = Registry()
type_registry = TypeRegistry()

registry_registry = RegistryRegistry()
registry_registry.register('_type', type_registry)
registry_registry.register('_apply', apply_registry)
registry_registry.register('_divide', divide_registry)
registry_registry.register('_serialize', serialize_registry)
registry_registry.register('_deserialize', deserialize_registry)



# class TypeRegistry(Registry):
#     def validate(self, schema):
#         return validate_schema(item)


def accumulate(current, update):
    return current + update

def concatenate(current, update):
    return current + update

def divide_float(value, _):
    half = value / 2.0
    return (half, half)

def divide_int(value, _):
    half = value // 2
    other_half = half
    if value % 2 == 1:
        other_half += 1
    return half, other_half

# def divide_longest(dimensions: Dimension) -> Tuple[Dimension, Dimension]:
def divide_longest(dimensions, _):
    # any way to declare the required keys for this function in the registry?
    # find a way to ask a function what type its domain and codomain are

    width = dimensions['width']
    height = dimensions['height']
    
    if width > height:
        a, b = divide_int(width)
        return [{'width': a, 'height': height}, {'width': b, 'height': height}]
    else:
        x, y = divide_int(height)
        return [{'width': width, 'height': x}, {'width': width, 'height': y}]


def divide_list(l, parameters):
    result = [[], []]
    divide_type = parameters[0]
    divide = registry_registry.type_attribute(divide_type, '_divide')

    for item in l:
        if isinstance(item, list):
            divisions = divide_list(item)
        else:
            divisions = divide(item)

        result[0].append(divisions[0])
        result[1].append(divisions[1])

    return result


def divide_dict(d, parameters):
    result = [{}, {}]
    # get the type of the values for this dict
    divide_type = parameters[0]
    divide = registry_registry.type_attribute(divide_type, '_divide')

    for key, value in d:
        if isinstance(value, dict):
            divisions = divide_dict(value)
        else:
            divisions = divide(value)

        result[0][key], result[1][key] = divisions

    return result


def replace(old_value, new_value):
    return new_value


apply_registry.register('accumulate', accumulate)
apply_registry.register('concatenate', concatenate)
apply_registry.register('replace', replace)
apply_registry.register('merge', deep_merge)
divide_registry.register('divide_float', divide_float)
divide_registry.register('divide_int', divide_int)
divide_registry.register('divide_longest', divide_longest)
divide_registry.register('divide_list', divide_list)
divide_registry.register('divide_dict', divide_dict)
serialize_registry.register('str', str)
deserialize_registry.register('float', float)
deserialize_registry.register('int', int)
deserialize_registry.register('str', str)
deserialize_registry.register('eval', eval)

# if super type is re-registered, propagate changes to subtypes

type_library = {
    'number': {
        # '_default': '0',
        '_apply': 'accumulate',
        '_serialize': 'str',
        # if you don't provide all the keys, it is abstract? 
        # '_deserialize': 'int',
        # '_divide': 'divide_int',
        '_description': 'abstract base type for numbers',
    },

    'int': {
        '_default': 0,
        # '_apply': 'accumulate',
        # '_serialize': 'str',
        '_deserialize': 'int',
        '_divide': 'divide_int',
        '_description': '64-bit integer',
        '_super': 'number',
    },

    'float': {
        '_default': '0.0',
        # '_apply': 'accumulate',
        # '_serialize': 'str',
        '_deserialize': 'float',
        '_divide': 'divide_float',
        '_description': '64-bit floating point precision number',
        '_super': 'number',
    }, 

    'string': {
        '_default': '',
        '_apply': 'replace',
        '_serialize': 'str',
        '_deserialize': 'str',
        '_divide': 'divide_int',
        '_description': '64-bit integer'
    },

    # 'float binomial division': {
    #     '_divide': 'divide_binomial',
    #     '_description': '64-bit integer',
    #     '_super': 'float', 
    # },

    'shape': {},

    'rectangle': {
        'width': {'_type': 'int'},
        'height': {'_type': 'int'},
        '_divide': 'divide_longest',
        '_description': 'a two-dimensional value',
        '_super': 'shape',
    },

    # if we override an existing non-_ key, throw an error?
    # cannot override existing keys unless it is of a subtype
    'cube': {
        'depth': {'_type': 'int'},
        '_super': 'rectangle',
    },

    'list': {
        '_default': '[]',
        '_apply': 'concatenate',
        '_serialize': 'str',
        '_deserialize': 'eval',
        '_divide': 'divide_list',
        '_parameters': ['A'],
        '_description': 'general list type (or sublists)'
    },

    'mapping': {
        '_default': '{}',
        '_apply': 'merge',
        '_serialize': 'str',
        '_deserialize': 'eval',
        '_divide': 'divide_dict',
        '_parameters': ['A'],
        '_description': 'mapping from str to some type (or nested dicts)'
    },

    'edge': {
        'wires': {'_type': 'mapping[list[string]]'},
    },

    # 'process': {
    #     'process': {'_type': 'process instance'},
    #     'config': {'_type': 'dict'},
    #     '_super': 'edge',
    # },
}


for key, schema in type_library.items():
    type_registry.register(key, schema)


supported_units = {
    'm/s': {
        '_default': 0.0,
        '_apply': 'accumulate',
        '_serialize': 'str',
        '_deserialize': 'float',
        '_divide': 'divide_float',
        '_description': 'meters per second'
    }
}

for key, units in supported_units.items():
    type_registry.register(key, units)


# class DimensionProcess(Process):
#     def ports_schema(self):
#         return {'_type': 'dimension'}

#         return {
#             '_description': 'a two-dimensional value',
#             '_divide': custom_divide,
#             'width': {'_type': 'float'},
#             'height': {'_type': 'float'},
#         }

#         return {
#             '_description': 'a two-dimensional value',
#             '_divide': custom_divide,
#             'width': {'_type': 'int'},
#             'height': {
#                 '_default': 0,
#                 '_apply': accumulate,
#                 '_serialize': str,
#                 '_deserialize': int,
#                 '_divide': divide_int,
#                 '_description': '64-bit integer'
#             }
#         }

def schema_zoo():
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
            'mitochondria': {'_type': 'branch[mitochondria]'},
            'proteins': {'_type': 'branch[mitochondria]'},
            'nucleus': {'_type': 'branch[mitochondria]'},
            'transcripts': {'_type': 'branch[mitochondria]'},
        }
    }

    cell_schema = {
        'cell': {
            'shape': {'_type': 'mesh'},
            'volume': {'_type': 'mL'},
            'temperature': {'_type': 'K'},
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
                    'other_ions': {'_type': {
                        '_default': 0.0,
                        '_apply': accumulate,
                        '_serialize': str,
                        '_deserialize': float,
                        '_divide': divide_float,
                        '_description': '64-bit floating point precision number'
                    }},
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


