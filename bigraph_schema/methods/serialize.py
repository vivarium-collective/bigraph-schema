from plum import dispatch
import numpy as np
from numpy.random.mtrand import RandomState
import numpy.lib.format as nf

from bigraph_schema.utilities import NONE_SYMBOL

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
    Link,
)

from bigraph_schema.methods.check import check
from bigraph_schema.methods.resolve import resolve


def wrap_default(schema, result):
    found = None
    if isinstance(schema, Node) and schema._default:
        found = schema._default
    elif isinstance(schema, dict) and '_default' in schema:
        found = schema['_default']

    if found:
        inner_default = found
        if isinstance(result, str) and isinstance(inner_default, str):
            result = result + '{' + inner_default + '}'
        elif isinstance(result, dict) and '_type' in result:
            result['_default'] = inner_default
        else:
            result = {
                '_type': result,
                '_default': inner_default}

    return result

@dispatch
def render(schema: Empty):
    return 'empty'

@dispatch
def render(schema: Maybe):
    value = render(schema._value)
    if isinstance(value, str):
        result = f'maybe[{value}]'
    else:
        result = {
            '_type': 'maybe',
            '_value': value}
    return wrap_default(schema, result)

@dispatch
def render(schema: Overwrite):
    value = render(schema._value)
    if isinstance(value, str):
        result = f'overwrite[{value}]'
    else:
        result = {
            '_type': 'overwrite',
            '_value': value}
    return wrap_default(schema, result)

@dispatch
def render(schema: Wrap):
    value = render(schema._value)
    if isinstance(value, str):
        result = f'wrap[{value}]'
    else:
        result = {
            '_type': 'wrap',
            '_value': value}
    return wrap_default(schema, result)

@dispatch
def render(schema: Union):
    options = [
        render(option)
        for option in schema._options]
    if all([isinstance(option,str) for option in options]):
        result = '~'.join(options)
    else:
        result = {
            '_type': 'union',
            '_options': options}
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
        result = {
            '_type': 'tuple',
            '_values': values}
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
    if isinstance(element, str):
        result = f'list[{element}]'
    else:
        result = {
            '_type': 'list',
            '_element': element}
    return wrap_default(schema, result)

@dispatch
def render(schema: Map):
    key = render(schema._key)
    value = render(schema._value)

    if isinstance(key, str) and isinstance(value,str):
        if key == 'string':
            result = f'map[{value}]'
        else:
            result = f'map[{key},{value}]'
    else:
        result = {
            '_type': 'map',
            '_key': key,
            '_value': value}

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
def render(schema: NPRandom):
    result = {
        '_type': 'random_state',
        'state': render(schema.state)}
    return wrap_default(schema, result)

# @dispatch
# def render(schema: Dtype):
    # fields = render(schema._fields)
    # result = {'_type': 'dtype', '_fields': fields}
    # return wrap_default(schema, result)

@dispatch
def render(schema: Array):
    shape = '|'.join([str(value) for value in schema._shape])
    data = nf.dtype_to_descr(schema._data)
    result = {'_type': 'array', '_shape': shape, '_data': data}
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
def render(schema: Link):
    intermediate = {
        '_type': 'link',
        '_inputs': render(schema._inputs),
        '_outputs': render(schema._outputs),
        'inputs': render(schema.inputs),
        'outputs': render(schema.outputs)}

    if isinstance(intermediate['_inputs'], str) and isinstance(intermediate['_outputs'], str):
        result = f'link[{intermediate["_inputs"]},{intermediate["_outputs"]}]'
    else:
        result = intermediate

    return wrap_default(schema, result)

@dispatch
def render(schema: dict):
    if '_type' in schema:
        return schema
    else:
        parts = {}
        for key, value in schema.items():
            subrender = render(value)
            parts[key] = subrender

        return wrap_default(schema, parts)

@dispatch
def render(schema: np.str_):
    return str(schema)

@dispatch
def render(schema: Node):
    subrender = {}

    for key in schema.__dataclass_fields__:
        value = getattr(schema,key)
        if key == '_default':
            subrender[key] = serialize(schema, value)
        else:
            subrender[key] = render(value)

    return wrap_default(schema, subrender)

@dispatch
def render(schema: str):
    return schema

@dispatch
def render(schema):
    import ipdb; ipdb.set_trace()

    return schema
    # what is happening

def render_associated(assoc):
    if all([isinstance(value, str) for value in assoc.values()]):
        parts = [f'{key}:{value}' for key, value in assoc.items()]
        assoc = '|'.join(parts)
    return assoc


@dispatch
def serialize(schema: Empty, state):
    return NONE_SYMBOL

@dispatch
def serialize(schema: Maybe, state):
    if state is None:
        return NONE_SYMBOL
    else:
        return serialize(
            schema._value,
            state)

@dispatch
def serialize(schema: Wrap, state):
    return serialize(schema._value, state)

@dispatch
def serialize(schema: Union, state):
    match = None
    for option in schema._options:
        if check(option, state):
            match = serialize(option, state)

            break
    return match

@dispatch
def serialize(schema: Tuple, state):
    return [
        serialize(subschema, value)
        for subschema, value in zip(schema._values, state)]

@dispatch
def serialize(schema: Boolean, state):
    if state:
        return 'true'
    else:
        return 'false'

@dispatch
def serialize(schema: NPRandom, state):
    if isinstance(state, RandomState):
        return serialize(
            schema.state,
            state.get_state())
    elif isinstance(state, (list, tuple)):
        return state
    else:
        import ipdb; ipdb.set_trace()

@dispatch
def serialize(schema: String, state):
    return state

@dispatch
def serialize(schema: np.str_, state):
    return str(state)

@dispatch
def serialize(schema: List, state):
    return [
        serialize(schema._element, element)
        for element in state]

@dispatch
def serialize(schema: Map, state):
    return {
        key: serialize(schema._value, value)
        for key, value in state.items()}

@dispatch
def serialize(schema: Tree, state):
    if check(schema._leaf, state):
        return serialize(schema._leaf, state)
    else:
        try:
            return {
                key: serialize(schema, branch)
                for key, branch in state.items()}
        except Exception as e:
            import ipdb; ipdb.set_trace()

@dispatch
def serialize(schema: dict, state):
    if not isinstance(state, dict):
        return {}

    result = {}

    for key, subschema in schema.items():
        if not key.startswith('_'):
            result[key] = serialize(
                subschema,
                state.get(key))

    return result


@dispatch
def serialize(schema: Number, state):
    return state


@dispatch
def serialize(schema: Atom, state):
    return str(state)


@dispatch
def serialize(schema: Array, state: np.ndarray):
    return state.tolist()

@dispatch
def serialize(schema: Array, state: list):
    return state

@dispatch
def serialize(schema: Array, state: dict):
    return state

@dispatch
def serialize(schema: Array, state):
    raise Exception(f'serializing array:\n  {schema}\nbut state is not an array?\n  {state}')

@dispatch
def serialize(schema: Link, state):
    address = serialize(schema.address, state.get('address'))
    instance = state.get('instance')
    unconfig = state.get('config')

    if instance is None:
        config_schema = {}
    else:
        config_schema = instance.core.access(instance.config_schema)

    config = serialize(config_schema, unconfig)
    inputs = serialize(schema.inputs, state.get('inputs'))
    outputs = serialize(schema.outputs, state.get('outputs'))
    _inputs = resolve(schema._inputs, state.get('_inputs'))
    _outputs = resolve(schema._outputs, state.get('_outputs'))

    return {
        'address': address,
        'config': config,
        '_inputs': render(_inputs),
        '_outputs': render(_outputs),
        'inputs': inputs,
        'outputs': outputs}


@dispatch
def serialize(schema: Node, state):
    if isinstance(state, dict):
        result = {}

        for key in schema.__dataclass_fields__:
            if not key in ('_default',):
                if key in state:
                    result[key] = serialize(
                        getattr(schema, key),
                        state[key])

        return result
    else:
        return str(state)

