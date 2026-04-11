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
    Quote,
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
    Object,
    dtype_schema,
    is_schema_field,
)

from bigraph_schema.methods.check import check
from bigraph_schema.methods.resolve import resolve


def wrap_default(schema, result):
    found = None
    if isinstance(schema, Node) and schema._default is not None:
        found = schema._default
    elif isinstance(schema, dict) and '_default' in schema:
        found = schema['_default']

    if found is not None:
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
def render(schema: Quote, defaults=False):
    value = render(schema._value, defaults=defaults)
    if isinstance(value, str):
        result = f'quote[{value}]'
    else:
        result = {
            '_type': 'quote',
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

def _render_number(name, schema, defaults=False):
    """Render a numeric type with optional [bits] and [bits,units] parameters."""
    parts = []
    if schema._bits:
        parts.append(str(schema._bits))
    if schema._units:
        parts.append(schema._units)
    result = f'{name}[{",".join(parts)}]' if parts else name
    return wrap_default(schema, result) if defaults else result

@dispatch
def render(schema: Integer, defaults=False):
    return _render_number('integer', schema, defaults)

@dispatch
def render(schema: Delta, defaults=False):
    return _render_number('delta', schema, defaults)

@dispatch
def render(schema: Nonnegative, defaults=False):
    return _render_number('nonnegative', schema, defaults)

@dispatch
def render(schema: Float, defaults=False):
    return _render_number('float', schema, defaults)

@dispatch
def render(schema: Complex, defaults=False):
    return _render_number('complex', schema, defaults)

@dispatch
def render(schema: Number, defaults=False):
    return _render_number('number', schema, defaults)

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

    if shape:
        result = f'array[{shape},{data}]'
    else:
        result = f'array[{data}]'
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
    intermediate = {'_type': 'link'}

    for field_name in schema.__dataclass_fields__:
        if not is_schema_field(schema, field_name):
            continue
        value = getattr(schema, field_name)
        intermediate[field_name] = render(value, defaults=defaults)

    # Compact form for simple links with only core fields
    if (isinstance(intermediate.get('_inputs'), str)
            and isinstance(intermediate.get('_outputs'), str)
            and len(intermediate) <= 5):
        result = f'link[{intermediate["_inputs"]},{intermediate["_outputs"]}]'
    else:
        result = intermediate

    return wrap_default(schema, result) if defaults else result

@dispatch
def render(schema: dict, defaults=False):
    if not schema:
        # Empty dict {} is the open/unconstrained schema — render as
        # "node" so it round-trips through parse correctly.
        return 'node'
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

    # Emit _type for registered Node subclasses so roundtrip works
    type_name = _node_type_name(type(schema))
    if type_name:
        subrender['_type'] = type_name

    for key in schema.__dataclass_fields__:
        if not is_schema_field(schema, key):
            if key == '_default':
                value = getattr(schema, key)
                default_render = serialize(schema, value)
                if default_render:
                    subrender['_default'] = default_render
            continue
        value = getattr(schema, key)
        subrender[key] = render(value, defaults=defaults)

    return wrap_default(schema, subrender) if defaults else subrender


# Reverse lookup: Node subclass -> registered type name
_TYPE_NAME_CACHE = {}

def _node_type_name(cls):
    """Get the registered type name for a Node subclass, if any."""
    if not _TYPE_NAME_CACHE:
        # Build cache from BASE_TYPES
        from bigraph_schema.schema import BASE_TYPES
        for name, type_cls in BASE_TYPES.items():
            if isinstance(type_cls, type):
                _TYPE_NAME_CACHE[type_cls] = name
    return _TYPE_NAME_CACHE.get(cls)

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


# ── serialize dispatch ──
# Each type has its own dispatch method. Container types call
# serialize() recursively on their children — plum routes to
# the correct method automatically. Custom types (registered in
# downstream packages) add their own @dispatch methods.


@dispatch
def serialize(schema: Empty, state):
    return NONE_SYMBOL


@dispatch
def serialize(schema: Boolean, state):
    if state is None:
        return None
    return 'true' if state else 'false'


@dispatch
def serialize(schema: NPRandom, state):
    if state is None:
        return None
    if isinstance(state, RandomState):
        return serialize(schema.state, state.get_state())
    if isinstance(state, (list, tuple)):
        return state
    raise Exception(f'cannot serialize NPRandom state: {state}')


@dispatch
def serialize(schema: Complex, state):
    return str(state) if state is not None else None


@dispatch
def serialize(schema: Number, state):
    if state is None:
        return None
    return state


@dispatch
def serialize(schema: Integer, state: np.integer):
    """numpy integer scalar → Python int."""
    return int(state)


@dispatch
def serialize(schema: Float, state: np.floating):
    """numpy float scalar → Python float."""
    return float(state)


@dispatch
def serialize(schema: Number, state: np.integer):
    return int(state)


@dispatch
def serialize(schema: Number, state: np.floating):
    return float(state)


@dispatch
def serialize(schema: String, state: np.integer):
    """numpy int used as map key with String schema."""
    return str(int(state))


@dispatch
def serialize(schema: String, state: np.floating):
    """numpy float used as map key with String schema."""
    return str(float(state))


@dispatch
def serialize(schema: Atom, state: np.integer):
    """numpy integer with generic Atom schema."""
    return int(state)


@dispatch
def serialize(schema: Atom, state: np.floating):
    """numpy float with generic Atom schema."""
    return float(state)


@dispatch
def serialize(schema: String, state):
    if state is None:
        return None
    return str(state)


@dispatch
def serialize(schema: Atom, state):
    if state is None:
        return None
    return str(state)


@dispatch
def serialize(schema: Array, state):
    if state is None:
        return None
    if isinstance(state, np.ndarray):
        result = state.tolist()
        # tolist() on structured arrays with sub-array fields
        # (e.g. dtype containing (int32, (2,))) leaves sub-arrays
        # as ndarrays inside tuples. Recursively convert them.
        if (isinstance(result, list) and result
                and isinstance(result[0], tuple)):
            result = [
                tuple(
                    v.tolist() if isinstance(v, np.ndarray) else v
                    for v in row)
                for row in result]
        return result
    if isinstance(state, list):
        return state
    if isinstance(state, dict):
        return state.get('data', state)
    raise Exception(
        f'serializing array:\n  {schema}\n'
        f'but state is not an array?\n  {state}')


@dispatch
def serialize(schema: Frame, state):
    if state is None:
        return {}
    return state.to_dict(orient="list")


@dispatch
def serialize(schema: Schema, state):
    return render(state)


@dispatch
def serialize(schema: Link, state):
    """Serialize a Link (process/step declaration)."""
    if not isinstance(state, dict):
        return state

    address = serialize(schema.address, state.get('address'))
    instance = state.get('instance')
    unconfig = state.get('config')

    if instance is None:
        config_schema = {}
    else:
        config_schema = instance.core.access(instance.config_schema)

    config = serialize(config_schema, unconfig)

    encode = {
        'address': address,
        'config': config,
        '_inputs': render(schema._inputs),
        '_outputs': render(schema._outputs)}

    if state.get('inputs'):
        encode['inputs'] = state.get('inputs')
    if state.get('outputs'):
        encode['outputs'] = state.get('outputs')
    if state.get('interval'):
        encode['interval'] = state.get('interval')
    if state.get('priority'):
        encode['priority'] = state.get('priority')
    if state.get('_triggers'):
        encode['_triggers'] = state.get('_triggers')

    return encode


@dispatch
def serialize(schema: Maybe, state):
    if state is None:
        return NONE_SYMBOL
    return serialize(schema._value, state)


@dispatch
def serialize(schema: Overwrite, state):
    return serialize(schema._value, state)


@dispatch
def serialize(schema: Wrap, state):
    return serialize(schema._value, state)


@dispatch
def serialize(schema: Quote, state):
    return serialize(schema._value, state)


@dispatch
def serialize(schema: Const, state):
    return serialize(schema._value, state)


def _serialize_map_key(schema, key):
    """Serialize a map key to a JSON-compatible string.

    JSON only supports string keys, so all keys are serialized through
    the schema and then converted to ``str``.  On load, ``realize``
    reverses this — e.g. ``realize(Integer, '42')`` → ``42``.
    """
    serialized = serialize(schema, key)
    if isinstance(serialized, str):
        return serialized
    return str(serialized)


@dispatch
def serialize(schema: Map, state):
    if not isinstance(state, dict):
        return {}
    return {
        _serialize_map_key(schema._key, key): serialize(schema._value, value)
        for key, value in state.items()}


@dispatch
def serialize(schema: Tree, state):
    if isinstance(state, dict):
        if check(schema._leaf, state):
            return serialize(schema._leaf, state)
        return {
            k: serialize(schema, v)
            for k, v in state.items()}
    return serialize(schema._leaf, state)


@dispatch
def serialize(schema: Tuple, state):
    if state is None:
        return None
    if isinstance(state, (list, tuple)):
        # Fixed-length: per-element schemas in _values
        return [serialize(s, v) for s, v in zip(schema._values, state)]
    return state


@dispatch
def serialize(schema: List, state):
    if state is None:
        return None
    if isinstance(state, (list, tuple, set, frozenset)):
        return [serialize(schema._element, v) for v in state]
    return state


@dispatch
def serialize(schema: Set, state):
    if state is None:
        return None
    if isinstance(state, (set, frozenset, list)):
        return sorted([serialize(schema._element, v) for v in state], key=str)
    return state


@dispatch
def serialize(schema: Union, state):
    # Try each variant — return the first that doesn't error
    for variant in schema._variants:
        try:
            return serialize(variant, state)
        except Exception:
            continue
    return str(state) if state is not None else None


@dispatch
def serialize(schema: Node, state):
    """Serialize a Node by walking its typed dataclass fields."""
    if state is None:
        return None
    if isinstance(state, dict):
        result = {}
        for key in schema.__dataclass_fields__:
            if is_schema_field(schema, key) and key in state:
                result[key] = serialize(getattr(schema, key), state[key])
        return result
    return str(state)


@dispatch
def serialize(schema: dict, state):
    """Serialize a dict schema by walking matching keys."""
    if not isinstance(state, dict):
        return {}
    result = {}
    for k, v in schema.items():
        if isinstance(k, str) and k.startswith('_'):
            continue
        if k in state:
            result[k] = serialize(v, state[k])
    return result


@dispatch
def render(schema: Object, defaults=False):
    if schema._class:
        result = f'object[{schema._class}]'
    else:
        result = 'object'
    return wrap_default(schema, result) if defaults else result


@dispatch
def serialize(schema: Object, state):
    """Serialize a Python object by walking its ``__dict__``.

    Produces ``{"_class": "...", "_schema": {...}, "fields": {...}}``
    where ``_schema`` maps field names to their rendered type strings
    and ``fields`` contains the serialized values.
    """
    if state is None:
        return None
    if isinstance(state, dict):
        # Already serialized form
        return state

    from bigraph_schema.methods.infer import infer

    cls = type(state)
    class_path = f'{cls.__module__}.{cls.__name__}'
    obj_dict = state.__dict__

    field_schemas = {}
    field_values = {}
    for key, value in obj_dict.items():
        # Infer schema for each field
        inferred_schema, _ = infer(None, value)
        field_schemas[key] = render(inferred_schema)
        field_values[key] = serialize(inferred_schema, value)

    return {
        '_class': class_path,
        '_schema': field_schemas,
        'fields': field_values,
    }


@dispatch
def serialize(schema, state):
    """Fallback: try to serialize based on state type."""
    if state is None:
        return None
    if isinstance(state, (str, int, float, bool)):
        return state
    if hasattr(state, 'item'):
        return state.item()
    return str(state)

