from plum import dispatch
import numpy as np

from types import NoneType

from bigraph_schema.schema import (
    Node,
    Maybe,
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


MISSING_TYPES = {}


@dispatch
def infer(value: (int | np.int32 | np.int64 |
                  np.dtypes.Int32DType | np.dtypes.Int64DType),
          path: tuple = ()):
    return Integer()

@dispatch
def infer(value: bool, path: tuple = ()):
    return Boolean()

@dispatch
def infer(value: (float | np.float32 | np.float64 |
                  np.dtypes.Float32DType | np.dtypes.Float64DType),
          path: tuple = ()):
    return Float()

@dispatch
def infer(value: str, path: tuple = ()):
    return String()

@dispatch
def infer(value: np.ndarray, path: tuple = ()):
    return Array(_shape=value.shape, _data=value.dtype)

@dispatch
def infer(value: list, path: tuple = ()):
    if len(value) > 0:
        element = infer(
            value[0],
            path+('_element',))
    else:
        element = Node()
    
    return List(_element=element)

@dispatch
def infer(value: tuple, path: tuple = ()):
    result = [
        infer(
            item,
            path+(index,))
        for index, item in enumerate(value)]
    return Tuple(_values=result)

@dispatch
def infer(value: NoneType, path: tuple = ()):
    return Maybe(_value=Node())

@dispatch
def infer(value: set, path: tuple = ()):
    return infer(
        list(value),
        path)

@dispatch
def infer(value: dict, path: tuple = ()):
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
        return Map(_value=map_value)
    else:
        return subvalues

@dispatch
def infer(value: object, path: tuple = ()):
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

                value_schema[key] = Node()

    return value_schema

