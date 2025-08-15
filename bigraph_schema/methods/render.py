from plum import dispatch
import numpy as np

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


def wrap_default(schema, result):
    if schema._default:
        return {
            '_type': result,
            '_default': schema._default}
    else:
        return result


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
def render(schema: Union):
    options = [
        render(option)
        for option in schema._options]
    result = '~'.join(options)
    return wrap_default(schema, result)

@dispatch
def render(schema: Tuple):
    values = [
        render(value)
        for value in schema._values]
    join = ','.join(values)
    result = f'tuple[{join}]'
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
    result = f'list[{element}]'
    return wrap_default(schema, result)

@dispatch
def render(schema: Map):
    key = render(schema._key)
    value = render(schema._value)

    if key == 'string':
        result = f'map[{value}]'
    else:
        result = f'map[{key},{value}]'

    return wrap_default(schema, result)

@dispatch
def render(schema: Tree):
    leaf = render(schema._leaf)
    result = f'tree[{leaf}]'
    return wrap_default(schema, result)

@dispatch
def render(schema: Dtype):
    result = render(schema._fields)
    return wrap_default(schema, result)

@dispatch
def render(schema: Array):
    shape = '|'.join(schema._shape)
    data = render(schema._data)
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
    result = {
        '_type': 'edge',
        '_inputs': render(schema._inputs),
        '_outputs': render(schema._outputs),
        'inputs': render(schema.inputs),
        'outputs': render(schema.outputs)}
    return wrap_default(schema, result)

@dispatch
def render(schema: dict):
    result = {
        key: render(value)
        for key, value in schema.items()}
    return result

@dispatch
def render(schema: Node):
    result = {}
    for key, value in schema.__dataclass_fields__:
        result[key] = render(value)

    return wrap_default(schema, result)
