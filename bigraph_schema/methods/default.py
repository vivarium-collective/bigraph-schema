from plum import dispatch
import numpy as np

from bigraph_schema.schema import (
    Node,
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
    Array,
    Key,
    Path,
    Wires,
    Schema,
    Edge,
)


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
    if schema._default is not None:
        return schema._default
    else:
        return np.zeros(
            schema._shape,
            dtype=schema._data)

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
def default(schema: Edge):
    return {
        'inputs': default(schema.inputs) or default_wires(schema._inputs),
        'outputs': default(schema.outputs) or default_wires(schema._outputs)}

@dispatch
def default(schema: dict):
    if '_default' in schema: 
        return schema['_default']
    else:
        result = {}
        for key in schema:
            if not key.startswith('_'):
                if schema[key] is None:
                    import ipdb; ipdb.set_trace()
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
            if not key.startswith('_'):
                inner = default(
                    getattr(schema, key))
                result[key] = inner

        return result

