from plum import dispatch
import numpy as np
from numpy.random.mtrand import RandomState

from bigraph_schema.schema import (
    Node,
    Atom,
    Empty,
    NONE_SYMBOL,
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
    Frame,
    Key,
    Path,
    Wires,
    Schema,
    Link,
    dtype_schema,
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
def render(schema: Empty, defaults=False):
    return 'empty'

@dispatch
def render(schema: Maybe, defaults=False):
    value = render(schema._value, defaults=defaults)
    if isinstance(value, str):
        result = f'maybe[{value}]'
    else:
        result = {
            '_type': 'maybe',
            '_value': value}
    return wrap_default(schema, result) if defaults else result

@dispatch
def render(schema: Overwrite, defaults=False):
    value = render(schema._value, defaults=defaults)
    if isinstance(value, str):
        result = f'overwrite[{value}]'
    else:
        result = {
            '_type': 'overwrite',
            '_value': value}
    return wrap_default(schema, result) if defaults else result

@dispatch
def render(schema: Wrap, defaults=False):
    value = render(schema._value, defaults=defaults)
    if isinstance(value, str):
        result = f'wrap[{value}]'
    else:
        result = {
            '_type': 'wrap',
            '_value': value}
    return wrap_default(schema, result) if defaults else result

@dispatch
def render(schema: Union, defaults=False):
    options = [
        render(option, defaults=defaults)
        for option in schema._options]
    if all([isinstance(option,str) for option in options]):
        result = '~'.join(options)
    else:
        result = {
            '_type': 'union',
            '_options': options}
    return wrap_default(schema, result) if defaults else result

@dispatch
def render(schema: Tuple, defaults=False):
    values = [
        render(value, defaults=defaults)
        for value in schema._values]
    if all([isinstance(value, str) for value in values]):
        join = ','.join(values)
        result = f'tuple[{join}]'
    else:
        result = {
            '_type': 'tuple',
            '_values': values}
    return wrap_default(schema, result) if defaults else result

@dispatch
def render(schema: Boolean, defaults=False):
    result = 'boolean'
    return wrap_default(schema, result) if defaults else result

@dispatch
def render(schema: Integer, defaults=False):
    result = 'integer'
    return wrap_default(schema, result) if defaults else result

@dispatch
def render(schema: Delta, defaults=False):
    result = 'delta'
    return wrap_default(schema, result) if defaults else result

@dispatch
def render(schema: Nonnegative, defaults=False):
    result = 'nonnegative'
    return wrap_default(schema, result) if defaults else result

@dispatch
def render(schema: Float, defaults=False):
    result = 'float'
    return wrap_default(schema, result) if defaults else result

@dispatch
def render(schema: Number, defaults=False):
    result = 'number'
    return wrap_default(schema, result) if defaults else result

@dispatch
def render(schema: String, defaults=False):
    result = 'string'
    return wrap_default(schema, result) if defaults else result

@dispatch
def render(schema: Enum, defaults=False):
    values = ','.join(schema._values)
    result = f'enum[{values}]'
    return wrap_default(schema, result) if defaults else result

@dispatch
def render(schema: List, defaults=False):
    element = render(schema._element, defaults=defaults)
    if isinstance(element, str):
        result = f'list[{element}]'
    else:
        result = {
            '_type': 'list',
            '_element': element}

    return wrap_default(schema, result) if defaults else result

@dispatch
def render(schema: Map, defaults=False):
    key = render(schema._key, defaults=defaults)
    value = render(schema._value, defaults=defaults)

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

    return wrap_default(schema, result) if defaults else result

@dispatch
def render(schema: Tree, defaults=False):
    leaf = render(schema._leaf, defaults=defaults)
    if isinstance(leaf, str):
        result = f'tree[{leaf}]'
    else:
        result = {'_type': 'tree', '_leaf': leaf}

    return wrap_default(schema, result) if defaults else result

@dispatch
def render(schema: NPRandom, defaults=False):
    result = {
        '_type': 'random_state',
        'state': render(schema.state, defaults=defaults)}
    return wrap_default(schema, result) if defaults else result


@dispatch
def render(schema: np.dtype, defaults=False):
    dtype = dtype_schema(schema)
    return render(dtype, defaults=defaults)

@dispatch
def render(schema: Array, defaults=False):
    shape = '|'.join([str(value) for value in schema._shape])
    data_schema = dtype_schema(schema._data)
    data = render(
        data_schema,
        defaults=defaults)

    result = f'array[{shape},{data}]'
    return wrap_default(schema, result) if defaults else result

@dispatch
def render(schema: Frame, defaults=False):
    columns = '|'.join([
        f'{key}:{render(value, default=defaults)}'
        for key, value in schema._columns.items()])
    result = f'dataframe[{columns}]'
    return result

@dispatch
def render(schema: Key, defaults=False):
    result = 'key'
    return wrap_default(schema, result) if defaults else result

@dispatch
def render(schema: Path, defaults=False):
    result = 'path'
    return wrap_default(schema, result) if defaults else result

@dispatch
def render(schema: Wires, defaults=False):
    result = 'wires'
    return wrap_default(schema, result) if defaults else result

@dispatch
def render(schema: Link, defaults=False):
    intermediate = {
        '_type': 'link',
        '_inputs': render(schema._inputs, defaults=defaults),
        '_outputs': render(schema._outputs, defaults=defaults),
        'inputs': render(schema.inputs, defaults=defaults),
        'outputs': render(schema.outputs, defaults=defaults)}

    if isinstance(intermediate['_inputs'], str) and isinstance(intermediate['_outputs'], str):
        result = f'link[{intermediate["_inputs"]},{intermediate["_outputs"]}]'
    else:
        result = intermediate

    return wrap_default(schema, result) if defaults else result

@dispatch
def render(schema: dict, defaults=False):
    if '_type' in schema:
        return schema
    else:
        parts = {}
        for key, value in schema.items():
            subrender = render(value, defaults=defaults)
            parts[key] = subrender

        if not defaults:
            parts = render_associated(parts)

        return wrap_default(schema, parts) if defaults else parts

@dispatch
def render(schema: np.str_, defaults=False):
    return str(schema)

@dispatch
def render(schema: Node, defaults=False):
    subrender = {}

    for key in schema.__dataclass_fields__:
        value = getattr(schema,key)
        if key == '_default':
            subrender[key] = serialize(schema, value)
        else:
            subrender[key] = render(value, defaults=defaults)

        if not defaults:
            subrender = render_associated(subrender)

    return wrap_default(schema, subrender) if defaults else subrender

@dispatch
def render(schema: str, defaults=False):
    return schema

@dispatch
def render(schema, defaults=False):
    return schema

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

@serialize.dispatch
def serialize(schema: Frame, state):
    if state is None:
        return {}
    return state.to_dict(orient="list")

@dispatch
def serialize(schema: Schema, state):
    return render(state)

@dispatch
def serialize(schema: Link, state):
    address = serialize(schema.address, state.get('address'))
    instance = state.get('instance')
    unconfig = state.get('config')

    if instance is None:
        config_schema = {}
    else:
        config_schema = instance.core.access(instance.config_schema)

    # config = serialize(config_schema, unconfig)
    # inputs = serialize(schema.inputs, state.get('inputs'))
    # outputs = serialize(schema.outputs, state.get('outputs'))
    _inputs = resolve(schema._inputs, state.get('_inputs'))
    _outputs = resolve(schema._outputs, state.get('_outputs'))

    encode = {
        'address': address,
        'config': unconfig,
        '_inputs': render(_inputs),
        '_outputs': render(_outputs)}

    if state.get('inputs'):
        encode['inputs'] = state.get('inputs')
    if state.get('outputs'):
        encode['outputs'] = state.get('outputs')

    return encode


@dispatch
def serialize(schema: None, state: None):
    return None

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
    elif state is None:
        return None
    else:
        return str(state)

