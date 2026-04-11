from plum import dispatch
import numpy as np
import pandas as pd
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
    Complex,
    Delta,
    Nonnegative,
    NPRandom,
    String,
    Enum,
    Wrap,
    Maybe,
    Overwrite,
    List,
    Set,
    Map,
    Tree,
    Array,
    Frame,
    Key,
    Path,
    Wires,
    Schema,
    Link,
    Quote,
    dtype_schema,
    get_frame_schema,
)
from bigraph_schema.edge import Edge


from bigraph_schema.methods.serialize import serialize
from bigraph_schema.methods.realize import realize

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
          value: (int | np.int8 | np.int16 | np.int32 | np.int64),
          path: tuple = ()):
    bits = 0
    if isinstance(value, np.integer):
        bits = value.dtype.itemsize * 8
    schema = Integer(_bits=bits)
    return set_default(schema, value), []

@dispatch
def infer(core, value: bool, path: tuple = ()):
    schema = Boolean()
    return set_default(schema, value), []

@dispatch
def infer(core,
          value: (float | np.float16 | np.float32 | np.float64),
          path: tuple = ()):
    bits = 0
    if isinstance(value, np.floating):
        bits = value.dtype.itemsize * 8
    schema = Float(_bits=bits)
    return set_default(schema, value), []

@dispatch
def infer(core, value: complex, path: tuple = ()):
    bits = 0
    if isinstance(value, np.complexfloating):
        bits = value.dtype.itemsize * 8
    schema = Complex(_bits=bits)
    return set_default(schema, value), []

@dispatch
def infer(core, value: str, path: tuple = ()):
    schema = String()
    return set_default(schema, value), []

@dispatch
def infer(core, value: np.dtype, path: tuple = ()):
    """numpy dtype — infer as the 'dtype' type."""
    from bigraph_schema.schema import Dtype
    schema = Dtype()
    return set_default(schema, value), []

@dispatch
def infer(core, value: np.ndarray, path: tuple = ()):
    schema = Array(
        _shape=value.shape,
        _data=value.dtype)

    return set_default(schema, value), []

@infer.dispatch
def infer(core, value: pd.DataFrame, path=()):
    columns = get_frame_schema(value)
    return Frame(_columns=columns), []

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
    if not value:
        return Set(), []
    # Infer element type from first element
    sample = next(iter(value))
    element_schema, merges = infer(core, sample, path)
    if isinstance(element_schema, Node):
        from dataclasses import replace as _replace
        element_schema = _replace(element_schema, **{'_default': None})
    return set_default(Set(_element=element_schema), value), merges


def separate_keys(d):
    schema = {}
    state = {}
    for key, value in d.items():
        if key.startswith('_'):
            schema[key] = value
        else:
            state[key] = value

    return schema, state

@dispatch
def infer(core, value: dict, path: tuple = ()):
    if not value:
        return Node(), []

    elif '_type' in value:
        schema_keys, state = separate_keys(value)
        schema = core.access_type(schema_keys)
        merges = []

        return set_default(schema, state), merges

    elif '_default' in value:
        return infer(core, value['_default'])

    else:
        subvalues = {}
        distinct_types = []
        merges = []

        for key, subvalue in value.items():
            subvalues[key], submerges = infer(
                core,
                subvalue,
                path+(key,))
            merges += submerges

            # Compare types ignoring defaults — Float(_default=1.5)
            # and Float(_default=2.0) are the same type for Map purposes
            inferred = subvalues[key]
            if isinstance(inferred, Node):
                type_sig = type(inferred)
            else:
                type_sig = type(inferred)

            if len(distinct_types) < 2 and type_sig not in distinct_types:
                distinct_types.append(type_sig)

        if len(distinct_types) == 1 and len(subvalues) >= 1:
            # All values are the same type — use Map
            first_schema = next(iter(subvalues.values()))
            # Strip the per-value default to get the clean type
            if isinstance(first_schema, Node):
                from dataclasses import replace as _replace
                map_value = _replace(first_schema, **{'_default': None})
            else:
                map_value = first_schema
            schema = Map(_value=map_value)
            return set_default(schema, value), merges
        else:
            # return Place(_default=value, _subnodes=subvalues)
            return subvalues, merges

@dispatch
def infer(core, value: object, path: tuple = ()):
    from bigraph_schema.schema import Object
    from bigraph_schema.methods.serialize import render

    # If the object looks like a process/step instance (has ports_schema,
    # next_update, update, etc.), treat it as opaque to avoid walking
    # into expensive simData internals.
    if hasattr(value, 'ports_schema') or hasattr(value, 'next_update'):
        schema = Quote(_value=Node())
        return set_default(schema, value), []

    type_name = str(type(value))

    if not hasattr(value, '__dict__'):
        schema = Quote(_value=Node())
        return set_default(schema, value), []

    cls = type(value)
    class_path = f'{cls.__module__}.{cls.__name__}'
    value_keys = value.__dict__.keys()
    field_schemas = {}

    merges = []

    for key in value_keys:
        try:
            field_schemas[key], submerges = infer(
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

            field_schemas[key] = Node()

    # Render each field schema to a string for the _schema dict
    rendered_schemas = {}
    for key, fs in field_schemas.items():
        rendered_schemas[key] = render(fs)

    schema = Object(_class=class_path, _schema=rendered_schemas)
    return set_default(schema, value), merges

