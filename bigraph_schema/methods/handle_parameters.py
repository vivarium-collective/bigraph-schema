from plum import dispatch
import numpy as np
import numpy.lib.format as nf

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
def align_parameters(schema: Tuple, parameters):
    return {
        '_values': parameters}

@dispatch
def align_parameters(schema: Enum, parameters):
    return {
        '_values': parameters}

@dispatch
def align_parameters(schema: Union, parameters):
    return {
        '_options': parameters}

@dispatch
def align_parameters(schema: Map, parameters):
    align = {}

    if len(parameters) == 1:
        align['_value'] = parameters[0]
    elif len(parameters) == 2:
        align['_key'], align['_value'] = parameters

    return align

@dispatch
def align_parameters(schema: Array, parameters):
    return {
        '_shape': parameters[0],
        '_data': parameters[1]}

@dispatch
def align_parameters(schema: Edge, parameters):
    align = {
        '_inputs': parameters[0],
        '_outputs': parameters[1]}

    return align

@dispatch
def align_parameters(schema: Node, parameters):
    align = {}
    keys = schema_keys(schema)[1:]
    for key, parameter in zip(keys, parameters):
        align[key] = parameter
    return align

@dispatch
def align_parameters(schema, parameters):
    raise Exception(f'unknown parameters for schema {schema}: {parameters}')

@dispatch
def assign_parameters(core, schema: Enum, parameters):
    if '_values' in parameters:
        schema._values = parameters['_values']
    return schema

@dispatch
def assign_parameters(core, schema: Array, parameters):
    schema._shape = tuple([
        int(value)
        for value in parameters.get('_shape', (1,))])

    schema._data = nf.descr_to_dtype(
        parameters.get('_data', 'float64'))

    return schema

@dispatch
def assign_parameters(core, schema: Node, parameters):
    for key, parameter in parameters.items():
        field = getattr(schema, key)
        subkey = core.access(parameter)
        resolve = core.resolve(field, subkey)
        setattr(schema, key, resolve)

    return schema

@dispatch
def assign_parameters(core, schema, parameters):
    import ipdb; ipdb.set_trace()

def handle_parameters(core, schema, parameters):
    align = align_parameters(schema, parameters)
    return assign_parameters(core, schema, align)

# @dispatch
# def handle_parameters(schema: Tuple, parameters):
#     schema._values = parameters
#     return schema

# @dispatch
# def handle_parameters(schema: Enum, parameters):
#     schema._values = parameters
#     return schema

# @dispatch
# def handle_parameters(schema: Union, parameters):
#     schema._options = parameters
#     return schema

# @dispatch
# def handle_parameters(schema: Map, parameters):
#     if len(parameters) == 1:
#         schema._value = parameters[0]
#     else:
#         schema._key, schema._value = parameters
#     return schema

# @dispatch
# def handle_parameters(schema: Array, parameters):
#     shape = parameters[0]
#     if isinstance(shape, Tuple):
#         shape = shape._values
#     schema._shape = tuple([
#         int(value)
#         for value in shape])
#     schema._data = np.dtype(parameters[1])

#     return schema

# @dispatch
# def handle_parameters(schema: Edge, parameters):
#     schema._inputs = parameters[0]
#     schema._outputs = parameters[1]

#     return schema

# @dispatch
# def handle_parameters(schema: Node, parameters):
#     keys = schema_keys(schema)[1:]
#     for key, parameter in zip(keys, parameters):
#         setattr(schema, key, parameter)
#     return schema

# @dispatch
# def handle_parameters(schema, parameters):
#     raise Exception(f'unknown parameters for schema {schema}: {parameters}')


