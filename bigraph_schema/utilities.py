import numpy as np


DTYPE_MAP = {
    'float': 'float64',
    'integer': 'int64',
    'string': 'str'}


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
