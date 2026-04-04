"""
Compute the minimal update that transforms state_a into state_b.

The relationship: apply(schema, a, diff(schema, a, b)) ≈ b
"""

from plum import dispatch

from bigraph_schema.schema import (
    Node,
    Atom,
    Empty,
    Union,
    Tuple,
    Boolean,
    Or,
    And,
    Xor,
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
    is_schema_field,
)

from bigraph_schema.methods.check import check


@dispatch
def diff(schema: Empty, state_a, state_b):
    return None


@dispatch
def diff(schema: Maybe, state_a, state_b):
    if state_a is None and state_b is None:
        return None
    if state_a is None:
        return state_b
    if state_b is None:
        return None
    return diff(schema._value, state_a, state_b)


@dispatch
def diff(schema: Wrap, state_a, state_b):
    return diff(schema._value, state_a, state_b)


@dispatch
def diff(schema: Overwrite, state_a, state_b):
    return state_b


@dispatch
def diff(schema: Const, state_a, state_b):
    return None


@dispatch
def diff(schema: Boolean, state_a, state_b):
    if state_a == state_b:
        return None
    return state_b


@dispatch
def diff(schema: String, state_a, state_b):
    if state_a == state_b:
        return None
    return state_b


@dispatch
def diff(schema: Number, state_a, state_b):
    if state_a == state_b:
        return None
    return state_b - state_a


@dispatch
def diff(schema: Tuple, state_a, state_b):
    if state_a == state_b:
        return None
    result = []
    for i, v in enumerate(schema._values):
        a = state_a[i] if i < len(state_a) else None
        b = state_b[i] if i < len(state_b) else None
        result.append(diff(v, a, b))
    if all(r is None for r in result):
        return None
    return tuple(result)


@dispatch
def diff(schema: List, state_a, state_b):
    if state_a == state_b:
        return None
    return state_b


@dispatch
def diff(schema: Set, state_a, state_b):
    if state_a == state_b:
        return None
    added = state_b - state_a
    removed = state_a - state_b
    result = {}
    if added:
        result['_add'] = added
    if removed:
        result['_remove'] = removed
    return result or None


@dispatch
def diff(schema: Map, state_a, state_b):
    if state_a == state_b:
        return None
    result = {}
    for key in state_b:
        if key in state_a:
            d = diff(schema._value, state_a[key], state_b[key])
            if d is not None:
                result[key] = d
        else:
            if '_add' not in result:
                result['_add'] = {}
            result['_add'][key] = state_b[key]
    removed = [k for k in state_a if k not in state_b]
    if removed:
        result['_remove'] = removed
    return result or None


@dispatch
def diff(schema: Tree, state_a, state_b):
    if check(schema._leaf, state_a) and check(schema._leaf, state_b):
        return diff(schema._leaf, state_a, state_b)
    if isinstance(state_a, dict) and isinstance(state_b, dict):
        result = {}
        for key in state_b:
            if key in state_a:
                d = diff(schema, state_a[key], state_b[key])
                if d is not None:
                    result[key] = d
            else:
                result[key] = state_b[key]
        return result or None
    return state_b


@dispatch
def diff(schema: dict, state_a, state_b):
    if not isinstance(state_a, dict) or not isinstance(state_b, dict):
        return state_b
    result = {}
    for key, subschema in schema.items():
        if not is_schema_field(schema, key):
            continue
        a = state_a.get(key)
        b = state_b.get(key)
        d = diff(subschema, a, b)
        if d is not None:
            result[key] = d
    return result or None


@dispatch
def diff(schema: Node, state_a, state_b):
    if state_a == state_b:
        return None
    return state_b


@dispatch
def diff(schema, state_a, state_b):
    return state_b
