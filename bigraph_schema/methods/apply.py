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
def apply(schema: Node, state, update, top_schema=None, top_state=None, path=()):
    if top_schema is None:
        top_schema = schema
    if top_state is None:
        top_state = state

