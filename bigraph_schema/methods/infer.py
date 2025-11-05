from plum import dispatch
import numpy as np
from numpy.random.mtrand import RandomState
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
    NPRandom,
    String,
    Enum,
    Wrap,
    Maybe,
    Overwrite,
    List,
    Map,
    Tree,
    Array,
    Key,
    Path,
    Wires,
    Schema,
    Edge,
)


from bigraph_schema.methods.serialize import serialize
from bigraph_schema.methods.unify import unify

MISSING_TYPES = {}


def set_default(schema, value):
    if value is not None:
        serialized = serialize(schema, value)
        if isinstance(serialized, dict) and '_default' in serialized:
            serialized = serialized['_default']

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
    return set_default(schema, value), []

@dispatch
def infer(core, value: bool, path: tuple = ()):
    schema = Boolean()
    return set_default(schema, value), []

@dispatch
def infer(core,
          value: (float | np.float32 | np.float64 |
                  np.dtypes.Float32DType | np.dtypes.Float64DType),
          path: tuple = ()):
    schema = Float()
    return set_default(schema, value), []

@dispatch
def infer(core, value: str, path: tuple = ()):
    schema = String()
    return set_default(schema, value), []

@dispatch
def infer(core, value: np.ndarray, path: tuple = ()):
    schema = Array(
        _shape=value.shape,
        _data=value.dtype) # Dtype(_fields=value.dtype))

    return set_default(schema, value), []

@dispatch
def infer(core, value: RandomState, path: tuple = ()):
    state = value.get_state()
    data, merges = infer(core, state)
    schema = NPRandom(state=data)

    return set_default(schema, value), merges

@dispatch
def infer(core, value: list, path: tuple = ()):
    merges = []
    if len(value) > 0:
        element, merges = infer(
            core,
            value[0],
            path+('_element',))
    else:
        element = Node()
    
    schema = List(_element=element)
    return set_default(schema, value), merges

@dispatch
def infer(core, value: tuple, path: tuple = ()):
    result = []
    merges = []
    for index, item in enumerate(value):
        if isinstance(item, np.str_):
            result.append(item)
        else:
            inner, submerges = infer(core, item, path+(index,))
            merges += submerges
            result.append(inner)

    schema = Tuple(_values=result)
    return set_default(schema, value), merges

@dispatch
def infer(core, value: NoneType, path: tuple = ()):
    schema = Maybe(_value=Node())
    return set_default(schema, value), []

@dispatch
def infer(core, value: set, path: tuple = ()):
    return infer(
        core,
        list(value),
        path)

@dispatch
def infer(core, value: dict, path: tuple = ()):
    if '_type' in value:
        schema = core.access_type(value)
        # clean_value = {
        #     key: subvalue
        #     for key, subvalue in value.items()
        #     if not key.startswith('_')} or core.default(schema)
        # schema, state, merges = unify(core, schema, clean_value, path)

        schema, state, merges = unify(core, schema, value, path)
        return set_default(schema, state), merges

    elif '_default' in value:
        return infer(core, value['_default'])

    else:
        subvalues = {}
        distinct_subvalues = []
        merges = []
        for key, subvalue in value.items():
            subvalues[key], submerges = infer(
                core,
                subvalue,
                path+(key,))
            merges += submerges

            if len(distinct_subvalues) < 2 and subvalues[key] not in distinct_subvalues:
                distinct_subvalues.append(
                    subvalues[key])

        if len(distinct_subvalues) == 1:
            map_value = distinct_subvalues[0]
            schema = Map(_value=map_value)
            return set_default(schema, value), merges
        else:
            # return Place(_default=value, _subnodes=subvalues)
            return subvalues, merges

@dispatch
def infer(core, value: object, path: tuple = ()):
    type_name = str(type(value))

    value_keys = value.__dict__.keys()
    value_schema = {}

    merges = []

    for key in value_keys:
        if not key.startswith('_'):
            try:
                value_schema[key], submerges = infer(
                    core,
                    getattr(value, key),
                    path + (key,))
                merges += submerges

            except Exception as e:
                traceback.print_exc()
                print(e)

                if type_name not in MISSING_TYPES:
                    MISSING_TYPES[type_name] = set([])

                MISSING_TYPES[type_name].add(
                    path)

                value_schema[key] = Node()

    return value_schema, merges

