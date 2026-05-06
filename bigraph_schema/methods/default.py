from plum import dispatch
import numpy as np
import pandas as pd

from bigraph_schema.schema import (
    Node,
    Empty,
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
    Protocol,
    LocalProtocol,
    Schema,
    Link,
    Object,
    Function,
    schema_dtype,
    is_schema_field,
)


@dispatch
def default(schema: None):
    return None

@dispatch
def default(schema: tuple):
    return schema

@dispatch
def default(schema: Empty):
    return None

@dispatch
def default(schema: Wrap):
    if schema._default is not None:
        return schema._default
    else:
        return default(schema._value)

@dispatch
def default(schema: Maybe):
    return None

@dispatch
def default(schema: Union):
    if schema._default is not None:
        return schema._default
    else:
        return default(schema._options[0])

@dispatch
def default(schema: Tuple):
    if schema._default is not None:
        return schema._default
    else:
        return [
            default(subschema)
            for subschema in schema._values]

@dispatch
def default(schema: Boolean):
    if schema._default is not None:
        return schema._default
    else:
        return False

@dispatch
def default(schema: Number):
    # The bare ``number`` type, a parent of Integer/Float/etc. Subclasses
    # override with specific zeros (0, 0.0, 0+0j); this is the fallback.
    if schema._default is not None:
        return schema._default
    else:
        return 0

@dispatch
def default(schema: Integer):
    if schema._default is not None:
        return schema._default
    else:
        return 0

@dispatch
def default(schema: Float):
    if schema._default is not None:
        return schema._default
    else:
        return 0.0

@dispatch
def default(schema: Complex):
    if schema._default is not None:
        return schema._default
    else:
        return 0+0j

@dispatch
def default(schema: Range):
    if schema._default is not None:
        return schema._default
    else:
        return max(schema._min, 0.0) if schema._min != float('-inf') else 0.0

@dispatch
def default(schema: String):
    if schema._default is not None:
        return schema._default
    else:
        return ''

@dispatch
def default(schema: Enum):
    if schema._default is not None:
        return schema._default
    else:
        return schema._values[0]

@dispatch
def default(schema: List):
    if schema._default is not None:
        return schema._default
    else:
        return []

@dispatch
def default(schema: Set):
    if schema._default is not None:
        return schema._default
    else:
        return set()

@dispatch
def default(schema: Map):
    if schema._default is not None:
        return schema._default
    else:
        return {}

@dispatch
def default(schema: Tree):
    if schema._default is not None:
        return schema._default
    else:
        return {}


@dispatch
def default(schema: Array):
    # An explicit ``_default`` wins, but only if it carries actual data.
    # v1 ``ports_schema()`` helpers commonly use ``[]`` as a placeholder
    # for "fill in later," and ``_enrich_defaults`` propagates that empty
    # list onto the v2 Array schema. If the schema declares a non-trivial
    # shape, we'd rather hand back a properly-shaped zero array than an
    # empty list that won't survive serialization.
    has_value = schema._default is not None
    is_empty_placeholder = isinstance(schema._default, (list, tuple)) and len(schema._default) == 0
    has_shape = schema._shape and any(s for s in schema._shape)
    if has_value and not (is_empty_placeholder and has_shape):
        return schema._default
    return np.zeros(
        schema._shape,
        dtype=schema._data)

@dispatch
def default(schema: Frame):
    if schema._default is not None:
        return schema._default
    else:
        columns = {
            key: schema_dtype(column)
            for key, column in schema._columns.items()}
        dataframe = pd.DataFrame(columns=columns.keys())
        return dataframe.astype(columns)


def default_wires(schema, path=None):
    path = path or []

    if isinstance(schema, dict):
        result = {}
        for key, subschema in schema.items():
            subpath = path+[key]
            result[key] = default_wires(
                subschema,
                subpath)
        return result

    elif isinstance(schema, Node):
        return path


@dispatch
def default(schema: Protocol):
    if schema._default is not None:
        return schema._default
    else:
        return {
            'protocol': 'local',
            'data': 'edge'}

def default_link(schema: Link):
    if schema._default: 
        return schema._default
    else:
        return {
            'address': default(schema.address) or 'local:edge',
            'config': default(schema.config) or {},
            '_inputs': schema._inputs,
            '_outputs': schema._outputs,
            'inputs': default(schema.inputs) or default_wires(schema._inputs),
            'outputs': default(schema.outputs) or default_wires(schema._outputs)}

@dispatch
def default(schema: Link):
    return default_link(schema)

@dispatch
def default(schema: Object):
    # An Object holds a serialized Python instance reconstructed from
    # ``_class`` + per-field values. Without those, no instance can be
    # produced — the default state is None, matching what
    # ``realize(Object, None)`` yields.
    if schema._default is not None:
        return schema._default
    else:
        return None

@dispatch
def default(schema: Function):
    # A Function identifies a callable by import path (module +
    # optional instance + attribute). Absent those, no callable can be
    # resolved, so the default is None.
    if schema._default is not None:
        return schema._default
    else:
        return None

@dispatch
def default(schema: dict):
    if '_default' in schema:
        return schema['_default']
    else:
        result = {}
        for key in schema:
            if is_schema_field(schema, key):
                inner = default(
                    schema[key])
                result[key] = inner

        return result

@dispatch
def default(schema: Node):
    if schema._default is not None:
        return schema._default
    else:
        result = {}
        for key in schema.__dataclass_fields__:
            if is_schema_field(schema, key):
                inner = default(
                    getattr(schema, key))
                result[key] = inner

        return result

