from plum import dispatch
import numpy as np

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

# from bigraph_schema.methods.serialize import serialize


def schema_keys(schema):
    keys = []
    for key in schema.__dataclass_fields__:
        if key.startswith('_'):
            keys.append(key)

    return keys

@dispatch
def handle_parameters(schema: Tuple, parameters):
    schema._values = parameters
    return schema

@dispatch
def handle_parameters(schema: Enum, parameters):
    schema._values = parameters
    return schema

@dispatch
def handle_parameters(schema: Union, parameters):
    schema._options = parameters
    return schema

@dispatch
def handle_parameters(schema: Map, parameters):
    if len(parameters) == 1:
        schema._value = parameters[0]
    else:
        schema._key, schema._value = parameters
    return schema

@dispatch
def handle_parameters(schema: Array, parameters):
    shape = parameters[0]
    if isinstance(shape, Tuple):
        shape = shape._values
    schema._shape = tuple([
        int(value)
        for value in shape])
    schema._data = np.dtype(parameters[1])

    return schema

@dispatch
def handle_parameters(schema: Edge, parameters):
    schema._inputs = parameters[0]
    schema._outputs = parameters[1]

    return schema

@dispatch
def handle_parameters(schema: Node, parameters):
    keys = schema_keys(schema)[1:]
    for key, parameter in zip(keys, parameters):
        setattr(schema, key, parameter)
    return schema

@dispatch
def handle_parameters(schema, parameters):
    return schema


