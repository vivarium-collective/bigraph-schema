"""
is_empty — type-aware emptiness check.

Determines whether a value is the "zero" or "identity" for its schema type.
Used by fill/merge to avoid overwriting meaningful state values with
type defaults.
"""

from plum import dispatch
import numpy as np

from bigraph_schema.schema import (
    Node,
    Empty,
    Boolean,
    Integer,
    Float,
    Complex,
    Delta,
    Nonnegative,
    Number,
    String,
    Array,
    List,
    Set,
    Map,
    Tree,
    Wrap,
    Quote,
    Overwrite,
    Maybe,
    Const,
    Tuple,
    NPRandom,
)


@dispatch
def is_empty(schema: Empty, value):
    return True


@dispatch
def is_empty(schema: Float, value):
    return value is None or value == 0.0


@dispatch
def is_empty(schema: Integer, value):
    return value is None or value == 0


@dispatch
def is_empty(schema: Number, value):
    return value is None or value == 0


@dispatch
def is_empty(schema: Complex, value):
    return value is None or value == 0j


@dispatch
def is_empty(schema: Delta, value):
    return value is None or value == 0


@dispatch
def is_empty(schema: Nonnegative, value):
    return value is None or value == 0


@dispatch
def is_empty(schema: Boolean, value):
    return value is None


@dispatch
def is_empty(schema: String, value):
    return value is None or value == ''


@dispatch
def is_empty(schema: Array, value):
    if value is None:
        return True
    if isinstance(value, np.ndarray):
        return value.size == 0
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    return False


@dispatch
def is_empty(schema: List, value):
    return value is None or (isinstance(value, list) and len(value) == 0)


@dispatch
def is_empty(schema: Set, value):
    return value is None or (isinstance(value, set) and len(value) == 0)


@dispatch
def is_empty(schema: Map, value):
    return value is None or (isinstance(value, dict) and len(value) == 0)


@dispatch
def is_empty(schema: Tree, value):
    return value is None or (isinstance(value, dict) and len(value) == 0)


@dispatch
def is_empty(schema: Tuple, value):
    return value is None or (isinstance(value, tuple) and len(value) == 0)


@dispatch
def is_empty(schema: Wrap, value):
    return is_empty(schema._value, value)


@dispatch
def is_empty(schema: Overwrite, value):
    return is_empty(schema._value, value)


@dispatch
def is_empty(schema: Quote, value):
    return value is None


@dispatch
def is_empty(schema: Maybe, value):
    if value is None:
        return True
    return is_empty(schema._value, value)


@dispatch
def is_empty(schema: NPRandom, value):
    return value is None


@dispatch
def is_empty(schema: Const, value):
    return value is None


@dispatch
def is_empty(schema: dict, value):
    return value is None or (isinstance(value, dict) and len(value) == 0)


@dispatch
def is_empty(schema: Node, value):
    return value is None


@dispatch
def is_empty(schema, value):
    return value is None
