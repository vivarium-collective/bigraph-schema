"""
Lenient type conversion — try to make state fit the schema without raising.

coerce(Float, '5') → 5.0
coerce(Integer, 5.0) → 5
coerce(String, 42) → '42'
coerce(Boolean, 'true') → True
"""

from plum import dispatch
import numpy as np

from bigraph_schema.schema import (
    Node,
    Atom,
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
)

from bigraph_schema.methods.check import check
from bigraph_schema.methods.default import default


@dispatch
def coerce(schema: Empty, state):
    return None


@dispatch
def coerce(schema: Boolean, state):
    if isinstance(state, bool):
        return state
    if isinstance(state, str):
        return state.lower() in ('true', '1', 'yes')
    if isinstance(state, (int, float)):
        return bool(state)
    return False


@dispatch
def coerce(schema: Integer, state):
    if isinstance(state, int) and not isinstance(state, bool):
        return state
    try:
        return int(state)
    except (ValueError, TypeError):
        return 0


@dispatch
def coerce(schema: Float, state):
    if isinstance(state, float):
        return state
    try:
        return float(state)
    except (ValueError, TypeError):
        return 0.0


@dispatch
def coerce(schema: Complex, state):
    if isinstance(state, complex):
        return state
    try:
        return complex(state)
    except (ValueError, TypeError):
        return 0+0j


@dispatch
def coerce(schema: Range, state):
    val = coerce(Float(), state)
    return max(schema._min, min(schema._max, val))


@dispatch
def coerce(schema: String, state):
    if isinstance(state, str):
        return state
    return str(state)


@dispatch
def coerce(schema: Enum, state):
    if isinstance(state, str) and state in schema._values:
        return state
    s = str(state)
    if s in schema._values:
        return s
    return schema._values[0] if schema._values else ''


@dispatch
def coerce(schema: Maybe, state):
    if state is None:
        return None
    return coerce(schema._value, state)


@dispatch
def coerce(schema: Wrap, state):
    return coerce(schema._value, state)


@dispatch
def coerce(schema: Tuple, state):
    if not isinstance(state, (list, tuple)):
        return default(schema)
    result = []
    for i, v in enumerate(schema._values):
        if i < len(state):
            result.append(coerce(v, state[i]))
        else:
            result.append(default(v))
    return tuple(result)


@dispatch
def coerce(schema: List, state):
    if isinstance(state, (list, tuple)):
        return [coerce(schema._element, e) for e in state]
    if isinstance(state, set):
        return [coerce(schema._element, e) for e in state]
    return []


@dispatch
def coerce(schema: Set, state):
    if isinstance(state, set):
        return {coerce(schema._element, e) for e in state}
    if isinstance(state, (list, tuple)):
        return {coerce(schema._element, e) for e in state}
    return set()


@dispatch
def coerce(schema: Map, state):
    if isinstance(state, dict):
        return {k: coerce(schema._value, v) for k, v in state.items()}
    return {}


@dispatch
def coerce(schema: Tree, state):
    if check(schema._leaf, state):
        return state
    if isinstance(state, dict):
        return {k: coerce(schema, v) for k, v in state.items()}
    return coerce(schema._leaf, state)


@dispatch
def coerce(schema: Array, state):
    if isinstance(state, np.ndarray):
        return state.astype(schema._data)
    try:
        return np.array(state, dtype=schema._data).reshape(schema._shape)
    except (ValueError, TypeError):
        return default(schema)


@dispatch
def coerce(schema: dict, state):
    if not isinstance(state, dict):
        return {k: default(v) for k, v in schema.items() if not k.startswith('_')}
    result = {}
    for k, v in schema.items():
        if isinstance(k, str) and k.startswith('_'):
            continue
        if k in state:
            result[k] = coerce(v, state[k])
        else:
            result[k] = default(v)
    return result


@dispatch
def coerce(schema: Node, state):
    if check(schema, state):
        return state
    return default(schema)


@dispatch
def coerce(schema, state):
    return state
