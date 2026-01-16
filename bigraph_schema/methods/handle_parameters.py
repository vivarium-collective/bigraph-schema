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
    Map,
    Tree,
    Array,
    Frame,
    Key,
    Path,
    Wires,
    Schema,
    Link,
    schema_dtype,
)

# aligning parameters takes them from positioned arguments and gives them keys
# in a dict.

# reifying the schema takes a dict of representations and turns them into schemas
# according to the parameters

# handling parameters combines these operations to go from positioned arguments to schemas

# we need aligning when parsing, but only reifying when inferring from state
# hence the distinction here

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
def align_parameters(schema: Frame, parameters):
    return {
        '_columns': parameters[0]}

@dispatch
def align_parameters(schema: Link, parameters):
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
def reify_schema(core, schema: Enum, parameters):
    if '_values' in parameters:
        schema._values = parameters['_values']
    return schema

@dispatch
def reify_schema(core, schema: Array, parameters):
    if '|' in parameters.get('_shape', ''):
        import ipdb; ipdb.set_trace()

    schema._shape = tuple([
        int(value)
        for value in parameters.get('_shape', (1,))])

    data = parameters.get('_data', 'float')
    data_schema = core.access(data)
    # if isinstance(data, Node):
    #     data = core.render(data)
    # schema._data = nf.descr_to_dtype(data)
    dtype = schema_dtype(data)
    if isinstance(dtype, Array):
        schema = replace(schema, **{'_shape': schema._shape + dtype._shape})
    else:
        schema._data = dtype

    return schema

@dispatch
def reify_schema(core, schema: Frame, parameters):
    schema._columns = parameters['_columns']
    return schema

@dispatch
def reify_schema(core, schema: Union, parameters):
    return replace(schema, **parameters)


def reify_schema_link(core, schema, parameters):
    if 'address' in parameters:
        schema.address = core.access(parameters['address'])
    if 'config' in parameters:
        schema.config = core.access(parameters['config'])
    if 'inputs' in parameters:
        schema.inputs = core.access(parameters['inputs'])
    if 'outputs' in parameters:
        schema.outputs = core.access(parameters['outputs'])
    if '_inputs' in parameters:
        schema._inputs = core.access(parameters['_inputs'])
    if '_outputs' in parameters:
        schema._outputs = core.access(parameters['_outputs'])

    return schema

@dispatch
def reify_schema(core, schema: Link, parameters):
    return reify_schema_link(core, schema, parameters)

@dispatch
def reify_schema(core, schema: Node, parameters):
    for key, parameter in parameters.items():
        subkey = core.access(parameter)

        if hasattr(schema, key):
            field = getattr(schema, key)
            resolve = core.resolve(field, subkey)
        else:
            resolve = subkey

        setattr(schema, key, resolve)

    return schema

@dispatch
def reify_schema(core, schema, parameters):
    import ipdb; ipdb.set_trace()

def handle_parameters(core, schema, parameters):
    align = align_parameters(schema, parameters)
    return reify_schema(core, schema, align)
