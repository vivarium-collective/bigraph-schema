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
