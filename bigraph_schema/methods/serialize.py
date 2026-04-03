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
    Complex,
    Delta,
    Nonnegative,
    Range,
    NPRandom,
    String,
    Enum,
    Wrap,
    Maybe,
    Overwrite,
    Const,
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
def render(schema: Const, defaults=False):
    value = render(schema._value, defaults=defaults)
    if isinstance(value, str):
        result = f'const[{value}]'
    else:
        result = {
            '_type': 'const',
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
def render(schema: Complex, defaults=False):
    result = 'complex'
    return wrap_default(schema, result) if defaults else result

@dispatch
def render(schema: Number, defaults=False):
    result = 'number'
    return wrap_default(schema, result) if defaults else result

@dispatch
def render(schema: Range, defaults=False):
    result = f'range[{schema._min},{schema._max}]'
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
    elif not element:
        result = 'list[node]'
    else:
        result = {
            '_type': 'list',
            '_element': element}

    return wrap_default(schema, result) if defaults else result

@dispatch
def render(schema: Set, defaults=False):
    element = render(schema._element, defaults=defaults)
    if isinstance(element, str):
        result = f'set[{element}]'
    else:
        result = {
            '_type': 'set',
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
        f'{key}:{render(value, defaults=defaults)}'
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
            parts[str(key)] = subrender

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
        value = getattr(schema, key)
        if key == '_default':
            default_render = serialize(schema, value)
            if default_render:
                subrender['_default'] = default_render
        else:
            subrender[key] = render(value, defaults=defaults)

    return wrap_default(schema, subrender) if defaults else subrender

@dispatch
def render(schema: str, defaults=False):
    return schema

@dispatch
def render(schema, defaults=False):
    return schema

def render_associated(assoc):
    if all([isinstance(key, str) and isinstance(value, str) for key, value in assoc.items()]):
        parts = [f'{key}:{value}' for key, value in assoc.items()]
        assoc = '|'.join(parts)

    return assoc


from bigraph_schema.methods.walk import walk as _walk


def _serialize_leaf(schema, state, path):
    """Leaf serialization for atoms and special types."""
    if isinstance(schema, Empty):
        return NONE_SYMBOL
    if isinstance(schema, Boolean):
        return 'true' if state else 'false'
    if isinstance(schema, NPRandom):
        if isinstance(state, RandomState):
            return serialize(schema.state, state.get_state())
        elif isinstance(state, (list, tuple)):
            return state
        else:
            raise Exception(f'cannot serialize NPRandom state: {state}')
    if isinstance(schema, Complex):
        return str(state)
    if isinstance(schema, Number):
        return state
    if isinstance(schema, String):
        return str(state)
    if isinstance(schema, Atom):
        return str(state)
    if isinstance(schema, np.str_):
        return str(state)
    if isinstance(schema, Array):
        if isinstance(state, np.ndarray):
            return state.tolist()
        elif isinstance(state, list):
            return state
        elif isinstance(state, dict):
            return state.get('data', state)
        else:
            raise Exception(
                f'serializing array:\n  {schema}\n'
                f'but state is not an array?\n  {state}')
    if isinstance(schema, Frame):
        if state is None:
            return {}
        return state.to_dict(orient="list")
    if isinstance(schema, Schema):
        return render(state)
    if isinstance(schema, Link):
        return _serialize_link(schema, state)
    if isinstance(schema, Node):
        if isinstance(state, dict):
            result = {}
            for key in schema.__dataclass_fields__:
                if key != '_default' and key in state:
                    result[key] = serialize(getattr(schema, key), state[key])
            return result
        elif state is None:
            return None
        else:
            return str(state)
    if schema is None and state is None:
        return None
    return str(state) if state is not None else None


def _serialize_combine(schema, children, path):
    """Assemble serialized children into the parent structure."""
    if isinstance(schema, Tuple):
        return list(children)
    if isinstance(schema, Set):
        return sorted(children, key=str)
    if isinstance(children, list):
        return children
    return children


def _serialize_link(schema, state):
    """Special-case serialization for Link type."""
    address = serialize(schema.address, state.get('address'))
    instance = state.get('instance')
    unconfig = state.get('config')

    if instance is None:
        config_schema = {}
    else:
        config_schema = instance.core.access(instance.config_schema)

    config = serialize(config_schema, unconfig)
    _inputs = schema._inputs
    _outputs = schema._outputs

    encode = {
        'address': address,
        'config': config,
        '_inputs': render(_inputs),
        '_outputs': render(_outputs)}

    if state.get('inputs'):
        encode['inputs'] = state.get('inputs')
    if state.get('outputs'):
        encode['outputs'] = state.get('outputs')
    if state.get('interval'):
        encode['interval'] = state.get('interval')
    if state.get('priority'):
        encode['priority'] = state.get('priority')

    return encode


def serialize(schema, state):
    """Serialize state according to schema.

    Uses walk for container recursion, with type-specific
    leaf handling for atoms and special types (Link, Frame, etc).
    """
    # Handle Maybe None before walk
    if isinstance(schema, Maybe):
        if state is None:
            return NONE_SYMBOL
        return serialize(schema._value, state)

    # Map needs special key serialization
    if isinstance(schema, Map) and isinstance(state, dict):
        return {
            serialize(schema._key, key): serialize(schema._value, value)
            for key, value in state.items()}

    # Special types that walk shouldn't recurse into
    if isinstance(schema, (Link, Frame, Array, NPRandom, Schema)):
        return _serialize_leaf(schema, state, ())

    # dict schema with non-dict state
    if isinstance(schema, dict) and not isinstance(state, dict):
        return {}

    if schema is None:
        return _serialize_leaf(schema, state, ())

    return _walk(schema, state, _serialize_leaf, _serialize_combine)

