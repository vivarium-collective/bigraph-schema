from plum import dispatch
import numpy as np
from bigraph_schema.methods.serialize import serialize, render_associated

from bigraph_schema.schema import (
    Node,
    Atom,
    Empty,
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


def wrap_default(schema, result):
    found = None
    if isinstance(schema, Node) and schema._default:
        found = schema._default
    elif isinstance(schema, dict) and '_default' in schema:
        found = schema['_default']

    if found:
        inner_default = serialize(schema, found)
        return result + '{' + inner_default + '}'
    else:
        return result

@dispatch
def render(schema: Empty):
    return 'empty'

@dispatch
def render(schema: Maybe):
    value = render(schema._value)
    result = f'maybe[{value}]'
    return wrap_default(schema, result)

@dispatch
def render(schema: Overwrite):
    value = render(schema._value)
    result = f'overwrite[{value}]'
    return wrap_default(schema, result)

@dispatch
def render(schema: Wrap):
    value = render(schema._value)
    result = f'wrap[{value}]'
    return wrap_default(schema, result)

@dispatch
def render(schema: Union):
    options = [
        render(option)
        for option in schema._options]
    if all([isinstance(option,str) for option in options]):
        result = '~'.join(options)
    else:
        result = {'_type': 'union', '_options': options}
    return wrap_default(schema, result)

@dispatch
def render(schema: Tuple):
    values = [
        render(value)
        for value in schema._values]
    if all([isinstance(value, str) for value in values]):
        join = ','.join(values)
        result = f'tuple[{join}]'
    else:
        result = {'_type': 'tuple', '_values': values}
    return wrap_default(schema, result)

@dispatch
def render(schema: Boolean):
    result = 'boolean'
    return wrap_default(schema, result)

@dispatch
def render(schema: Integer):
    result = 'integer'
    return wrap_default(schema, result)

@dispatch
def render(schema: Delta):
    result = 'delta'
    return wrap_default(schema, result)

@dispatch
def render(schema: Nonnegative):
    result = 'nonnegative'
    return wrap_default(schema, result)

@dispatch
def render(schema: Float):
    result = 'float'
    return wrap_default(schema, result)

@dispatch
def render(schema: Number):
    result = 'number'
    return wrap_default(schema, result)

@dispatch
def render(schema: String):
    result = 'string'
    return wrap_default(schema, result)

@dispatch
def render(schema: Enum):
    values = ','.join(schema._values)
    result = f'enum[{values}]'
    return wrap_default(schema, result)

@dispatch
def render(schema: List):
    element = render(schema._element)
    if isinstance(element,str):
        result = f'list[{element}]'
    else:
        result = {'_type': 'list', '_element': element}
    return wrap_default(schema, result)

@dispatch
def render(schema: Map):
    key = render(schema._key)
    value = render(schema._value)

    if isinstance(key,str) and isinstance(value,str):
        if key == 'string':
            result = f'map[{value}]'
        else:
            result = f'map[{key},{value}]'
    else:
        result = {'_type': 'map', '_key': key, '_value': value}

    return wrap_default(schema, result)

@dispatch
def render(schema: Tree):
    leaf = render(schema._leaf)
    if isinstance(leaf, str):
        result = f'tree[{leaf}]'
    else:
        result = {'_type': 'tree', '_leaf': leaf}
    return wrap_default(schema, result)

@dispatch
def render(schema: Dtype):
    result = schema._fields
    return wrap_default(schema, result)

@dispatch
def render(schema: Array):
    shape = '|'.join([str(value) for value in schema._shape])
    data = schema._data.str
    result = f'array[({shape}),{data}]'
    return wrap_default(schema, result)

@dispatch
def render(schema: Key):
    result = 'key'
    return wrap_default(schema, result)

@dispatch
def render(schema: Path):
    result = 'path'
    return wrap_default(schema, result)

@dispatch
def render(schema: Wires):
    result = 'wires'
    return wrap_default(schema, result)

@dispatch
def render(schema: Edge):
    intermediate = {
        '_type': 'edge',
        '_inputs': render(schema._inputs),
        '_outputs': render(schema._outputs),
        'inputs': render(schema.inputs),
        'outputs': render(schema.outputs)}

    result = f'edge[{intermediate["_inputs"]},{intermediate["_outputs"]}]'
    return wrap_default(schema, result)

@dispatch
def render(schema: dict):
    parts = {}
    for key, value in schema.items():
        subrender = render(value)
        if isinstance(value, dict):
            subrender = f'({subrender})'
        parts[key] = subrender

    result = render_associated(parts)
    return wrap_default(schema, result)

@dispatch
def render(schema: Node):
    subrender = {}

    for key in schema.__dataclass_fields__:
        value = getattr(schema,key)
        if key == '_default':
            subrender[key] = serialize(schema, value)
        else:
            subrender[key] = render(value)

    result = render_associated(subrender)
    return wrap_default(schema, result)
