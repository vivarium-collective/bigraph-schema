from plum import dispatch
import numpy as np
import traceback

from types import NoneType
from dataclasses import replace

from bigraph_schema.schema import (
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


from bigraph_schema.methods.serialize import serialize

MISSING_TYPES = {}


def set_default(schema, value):
    if value is not None:
        serialized = serialize(schema, value)
        if isinstance(schema, Node):
            schema = replace(schema, _default=serialized)
        elif isinstance(schema, dict):
            schema['_default'] = serialized

    return schema

@dispatch
def infer(core,
          value: (int | np.int32 | np.int64 |
                  np.dtypes.Int32DType | np.dtypes.Int64DType),
          path: tuple = ()):
    schema = Integer()
    return set_default(schema, value)

@dispatch
def infer(core, value: bool, path: tuple = ()):
    schema = Boolean()
    return set_default(schema, value)

@dispatch
def infer(core,
          value: (float | np.float32 | np.float64 |
                  np.dtypes.Float32DType | np.dtypes.Float64DType),
          path: tuple = ()):
    schema = Float()
    return set_default(schema, value)

@dispatch
def infer(core, value: str, path: tuple = ()):
    schema = String()
    return set_default(schema, value)

@dispatch
def infer(core, value: np.ndarray, path: tuple = ()):
    schema = Array(
        _shape=value.shape,
        _data=value.dtype) # Dtype(_fields=value.dtype))

    return set_default(schema, value)

@dispatch
def infer(core, value: list, path: tuple = ()):
    if len(value) > 0:
        element = infer(
            core,
            value[0],
            path+('_element',))
    else:
        element = Node()
    
    schema = List(_element=element)
    return set_default(schema, value)

@dispatch
def infer(core, value: tuple, path: tuple = ()):
    result = []
    for index, item in enumerate(value):
        if isinstance(item, np.str_):
            result.append(item)
        else:
            inner = infer(core, item, path+(index,))
            result.append(inner)

    schema = Tuple(_values=result)
    return set_default(schema, value)    

@dispatch
def infer(core, value: NoneType, path: tuple = ()):
    schema = Maybe(_value=Node())
    return set_default(schema, value)    

@dispatch
def infer(core, value: set, path: tuple = ()):
    return infer(
        core,
        list(value),
        path)

@dispatch
def infer(core, value: dict, path: tuple = ()):
    if '_type' in value:
        schema = core.access(value)
        default_value = None
        if '_default' in value:
            default_value = value['_default']
        elif isinstance(schema, Node) and schema._default is not None:
            default_value = schema._default
        elif isinstance(schema, dict) and '_default' in schema:
            default_value = schema['_default']
        return set_default(schema, default_value)

    elif '_default' in value:
        return infer(core, value['_default'])

    else:
        subvalues = {}
        distinct_subvalues = []
        for key, subvalue in value.items():
            subvalues[key] = infer(
                core,
                subvalue,
                path+(key,))

            if subvalues[key] not in distinct_subvalues:
                distinct_subvalues.append(
                    subvalues[key])

        if len(distinct_subvalues) == 1:
            map_value = distinct_subvalues[0]
            schema = Map(_value=map_value)
            return set_default(schema, value)
        else:
            return subvalues

@dispatch
def infer(core, value: object, path: tuple = ()):
    type_name = str(type(value))

    value_keys = value.__dict__.keys()
    value_schema = {}

    import ipdb; ipdb.set_trace()

    for key in value_keys:
        if not key.startswith('_'):
            try:
                value_schema[key] = infer(
                    core,
                    getattr(value, key),
                    path + (key,))
            except Exception as e:
                traceback.print_exc()
                print(e)

                if type_name not in MISSING_TYPES:
                    MISSING_TYPES[type_name] = set([])

                MISSING_TYPES[type_name].add(
                    path)

                value_schema[key] = Node()

    return value_schema

