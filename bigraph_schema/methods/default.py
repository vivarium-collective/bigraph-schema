from plum import dispatch
import numpy as np

from bigraph_schema.schema import (
    Node,
    Maybe,
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
    List,
    Map,
    Tree,
    Dtype,
    Array,
)


@dispatch
def default(schema: Maybe):
    return None

@dispatch
def default(schema: Union):
    return default(schema._options[0])

@dispatch
def default(schema: Tuple):
    return [
        default(subschema)
        for subschema in schema._values]

@dispatch
def default(schema: Boolean):
    return False

@dispatch
def default(schema: Integer):
    return 0

@dispatch
def default(schema: Float):
    return 0.0

@dispatch
def default(schema: String):
    return ''

@dispatch
def default(schema: Enum):
    return schema._values[0]

@dispatch
def default(schema: List):
    return []

@dispatch
def default(schema: Map):
    return {}

@dispatch
def default(schema: Tree):
    return {}

@dispatch
def default(schema: Dtype):
    return np.dtype(schema._fields)

@dispatch
def default(schema: Array):
    return np.zeros(
        tuple(schema._shape),
        dtype=dtype(schema._data))

@dispatch
def default(schema: Node):
    return {}

