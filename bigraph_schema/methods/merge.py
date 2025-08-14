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
    Key,
    Path,
    Wires,
    Schema,
    Edge,
)


@dispatch
def merge(schema: Node):
    pass
