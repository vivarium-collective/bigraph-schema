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
    Dtype,
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
    if schema._default:
        return schema._default
    else:
        return default(schema._value)

@dispatch
def default(schema: Union):
    if schema._default:
        return schema._default
    else:
        return default(schema._options[0])

@dispatch
def default(schema: Tuple):
    if schema._default:
        return schema._default
    else:
        return [
            default(subschema)
            for subschema in schema._values]

@dispatch
def default(schema: Boolean):
    if schema._default:
        return schema._default
    else:
        return False

@dispatch
def default(schema: Integer):
    if schema._default:
        return schema._default
    else:
        return 0

@dispatch
def default(schema: Float):
    if schema._default:
        return schema._default
    else:
        return 0.0

@dispatch
def default(schema: String):
    if schema._default:
        return schema._default
    else:
        return ''

@dispatch
def default(schema: Enum):
    if schema._default:
        return schema._default
    else:
        return schema._values[0]

@dispatch
def default(schema: List):
    if schema._default:
        return schema._default
    else:
        return []

@dispatch
def default(schema: Map):
    if schema._default:
        return schema._default
    else:
        return {}

@dispatch
def default(schema: Tree):
    if schema._default:
        return schema._default
    else:
        return {}

@dispatch
def default(schema: Dtype):
    if schema._default:
        return schema._default
    else:
        return np.dtype(schema._fields)

@dispatch
def default(schema: Array):
    if schema._default:
        return schema._default
    else:
        return np.zeros(
            tuple(schema._shape),
            dtype=dtype(schema._data))

@dispatch
def default(schema: Key):
    if schema._default:
        return schema._default
    else:
        return 0

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
    if schema._default:
        return schema._default
    else:
        result = {}
        for key in schema.__dataclass_fields__:
            if not key.startswith('_'):
                inner = default(
                    schema.getattr(key))
                result[key] = inner

        return result

