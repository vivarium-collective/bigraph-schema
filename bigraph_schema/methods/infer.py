from plum import dispatch
import numpy as np

from types import NoneType

MISSING_TYPES = {}

@dispatch
def infer(value: (int | np.int32 | np.int64 |
                  np.dtypes.Int32DType | np.dtypes.Int64DType),
          path: tuple):
    return 'integer'

@dispatch
def infer(value: bool, path: tuple):
    return 'boolean'

@dispatch
def infer(value: (float | np.float32 | np.float64 |
                  np.dtypes.Float32DType | np.dtypes.Float64DType),
          path: tuple):
    return 'float'

@dispatch
def infer(value: str, path: tuple):
    return 'string'

@dispatch
def infer(value: np.ndarray, path: tuple):
    import ipdb; ipdb.set_trace()

    shape = '|'.join([str(dimension) for dimension in value.shape])
    data = infer(
        dtype_schema(value.dtype),
        path+('_data',))

    return f'array[({shape}),{data}]'

@dispatch
def infer(value: list, path: tuple):
    element = 'any'
    if len(value) > 0:
        element = infer(
            value[0],
            path+('_element',))

    return f'list[{element}]'

def dict_schema(schema):
    parts = []
    for key, subschema in schema.items():
        if isinstance(subschema, dict):
            part = f'({dict_schema(subschema)})'
        else:
            part = subschema
        entry = f'{key}:{part}'
        parts.append(
            entry)

    return '|'.join(
        parts)

@dispatch
def infer(value: tuple, path: tuple):
    result = []
    for index, item in enumerate(value):
        key = f'_{index}'
        schema = infer(
            item,
            path+(key,))
        if isinstance(schema, dict):
            schema = dict_schema(schema)
        result.append(schema)

    inner = '|'.join(result)
    return f'({inner})'

@dispatch
def infer(value: NoneType, path: tuple):
    return 'maybe[any]'

@dispatch
def infer(value: set, path: tuple):
    return infer(
        list(value),
        path)

@dispatch
def infer(value: dict, path: tuple):
    subvalues = {}
    distinct_subvalues = []
    for key, subvalue in value.items():
        subvalues[key] = infer(
            subvalue,
            path+(key,))

        if subvalues[key] not in distinct_subvalues:
            distinct_subvalues.append(
                subvalues[key])

    if len(distinct_subvalues) == 1:
        map_value = distinct_subvalues[0]
        if isinstance(map_value, dict):
            map_value = dict_schema(
                map_value)
        if not map_value:
            map_value = 'any'

        return f'map[{map_value}]'

    else:
        return subvalues

@dispatch
def infer(value: object, path: object):
    type_name = str(type(value))

    value_keys = value.__dict__.keys()
    value_schema = {}

    for key in value_keys:
        if not key.startswith('_'):
            try:
                value_schema[key] = infer(
                    getattr(value, key),
                    path + (key,))
            except Exception as e:
                traceback.print_exc()
                print(e)

                if type_name not in MISSING_TYPES:
                    MISSING_TYPES[type_name] = set([])

                MISSING_TYPES[type_name].add(
                    path)

                value_schema[key] = 'any'

    return value_schema

def infer_schema(config, path=()) -> dict:
    '''Translate default values into corresponding bigraph-schema type declarations.'''
    ports = {}

    for key, value in config.items():
        ports[key] = infer(
            value,
            path+(key,))

    return ports


