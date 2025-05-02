"""
Utility functions for working with bigraph schemas
"""

import collections
import numpy as np


NONE_SYMBOL = '!nil'

DTYPE_MAP = {
    'float': 'float64',
    'integer': 'int64',
    'string': 'str'}

overridable_schema_keys = {'_type', '_default', '_check', '_apply', '_serialize', '_deserialize', '_fold', '_divide',
                           '_slice', '_bind', '_merge', '_type_parameters', '_value', '_description', '_inherit'}

# nonoverridable_schema_keys = type_schema_keys - overridable_schema_keys

merge_schema_keys = (
    '_ports',
    '_type_parameters',
)


def type_merge(dct, merge_dct, path=tuple(), merge_supers=False):
    """
    Recursively merge type definitions, never overwrite.

    Args:
    - dct: The dictionary to merge into. This dictionary is mutated and ends up being the merged dictionary.  If you
        want to keep dct you could call it like ``deep_merge_check(copy.deepcopy(dct), merge_dct)``.
    - merge_dct: The dictionary to merge into ``dct``.
    - path: If the ``dct`` is nested within a larger dictionary, the path to ``dct``. This is normally an empty tuple
        (the default) for the end user but is used for recursive calls.
    Returns:
    - dct
    """
    for k in merge_dct:
        if not k in dct or k in overridable_schema_keys:
            dct[k] = merge_dct[k]
        elif k in merge_schema_keys or isinstance(
                dct[k], dict
        ) and isinstance(
            merge_dct[k], collections.abc.Mapping
        ):
            type_merge(
                dct[k],
                merge_dct[k],
                path + (k,),
                merge_supers)

        else:
            raise ValueError(
                f'cannot merge types at path {path + (k,)}:\n'
                f'{dct}\noverwrites \'{k}\' from\n{merge_dct}')

    return dct


def visit_method(schema, state, method, values, core):
    """
    Visit a method for a schema and state and apply it, returning the result
    """
    schema = core.access(schema)
    method_key = f'_{method}'

    # TODO: we should probably cache all this
    if isinstance(state, dict) and method_key in state:
        visit = core.find_method(
            {method_key: state[method_key]},
            method_key)
    elif method_key in schema:
        visit = core.find_method(
            schema,
            method_key)
    else:
        visit = core.find_method(
            'any',
            method_key)

    result = visit(
        schema,
        state,
        values,
        core)

    return result


def is_empty(value):
    if isinstance(value, np.ndarray):
        return False
    elif value is None or value == {}:
        return True
    else:
        return False


def union_keys(schema, state):
    keys = {}
    for key in schema:
        keys[key] = True
    for key in state:
        keys[key] = True

    return keys
    # return set(schema.keys()).union(state.keys())


def tuple_from_type(tuple_type):
    if isinstance(tuple_type, tuple):
        return tuple_type

    elif isinstance(tuple_type, list):
        return tuple(tuple_type)

    elif isinstance(tuple_type, dict):
        tuple_list = [
            tuple_type[f'_{parameter}']
            for parameter in tuple_type['_type_parameters']]

        return tuple(tuple_list)
    else:
        raise Exception(f'do not recognize this type as a tuple: {tuple_type}')


def array_shape(core, schema):
    if '_type_parameters' not in schema:
        schema = core.access(schema)
    parameters = schema.get('_type_parameters', [])

    return tuple([
        int(schema[f'_{parameter}'])
        for parameter in schema['_type_parameters']])


def lookup_dtype(data_name):
    data_name = data_name or 'string'
    dtype_name = DTYPE_MAP.get(data_name)
    if dtype_name is None:
        raise Exception(f'unknown data type for array: {data_name}')

    return np.dtype(dtype_name)


def read_datatype(data_schema):
    return lookup_dtype(
        data_schema['_type'])


def read_shape(shape):
    return tuple([
        int(x)
        for x in tuple_from_type(
            shape)])


def compare_dicts(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        result = {}
        for key in union_keys(a, b):
            if key in a:
                if key in b:
                    inner = compare_dicts(a[key], b[key])
                    if inner:
                        result[key] = inner
                else:
                    result[key] = f'A: {a[key]}\nB: (missing)'
            else:
                result[key] = f'A: (missing)\nB: {b[key]}'
        if result:
            return result
    else:
        if a != b:
            return f'A: {a}\nB: {b}'


def get_path(tree, path):
    """
    Given a tree and a path, find the subtree at that path

    Args:
    - tree: the tree we are looking in (a nested dict)
    - path: a list/tuple of keys we follow down the tree to find the subtree we are looking for

    Returns:
    - subtree: the subtree found by following the list of keys down the tree
    """

    if len(path) == 0:
        return tree
    else:
        head = path[0]
        if not tree or head not in tree:
            return None
        else:
            return get_path(tree[head], path[1:])


def remove_path(tree, path):
    """
    Removes whatever subtree lives at the given path
    """

    if path is None or len(path) == 0:
        return None

    upon = get_path(tree, path[:-1])
    if upon is not None:
        del upon[path[-1]]
    return tree


def type_parameters_for(schema):
    parameters = []
    for key in schema['_type_parameters']:
        subschema = schema.get(f'_{key}', 'any')
        parameters.append(subschema)

    return parameters
