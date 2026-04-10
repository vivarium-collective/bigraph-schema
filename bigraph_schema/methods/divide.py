"""Type-driven cell division: split a state value into two daughter values
according to its schema.

The contract is:
  divide(schema, state, context=None, path=(), rng=None)
      -> (daughter_a, daughter_b)

`context` is the parent state (typically the whole cell) so that contextual
dividers — e.g. unique molecules whose split depends on sibling arrays —
can resolve relative paths like `("..", "RNA")`. Most local dividers
ignore it. `path` is the absolute path of `state` within `context`.

Each numeric/array type knows how to split itself (binomial for integer
counts, halve for floats, etc.). Container types (Map, Tree) recurse into
their fields, threading `context` and growing `path`. Reference / opaque
types share the same underlying object between daughters.

Used by `apply(DivideMap, ...)` when it sees a `_divide` sentinel update.
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
    String,
    List,
    Set,
    Map,
    Tree,
    Array,
)


def _binomial_split(rng, total):
    """Sample (a, b) such that a + b == total and a ~ Binomial(total, 0.5).

    Works element-wise on numpy arrays of integers and on Python ints.
    """
    if hasattr(total, 'dtype') and np.issubdtype(total.dtype, np.integer):
        a = rng.binomial(total, 0.5).astype(total.dtype)
        return a, total - a
    if isinstance(total, (int, np.integer)):
        a = int(rng.binomial(int(total), 0.5))
        return a, int(total) - a
    # Floating-point: split in half (no binomial available)
    return total / 2.0, total / 2.0


@dispatch
def divide(schema: Node, state, context=None, path=(), rng=None):
    """Walk an arbitrary Node's typed fields when state is a dict.

    Many cell schemas are auto-resolved to a plain Node with per-field
    attributes set via setattr (e.g. cell.bulk = BulkArray, cell.unique
    = Tree, cell.listeners = Map). Recurse into each known field so
    field-specific divide() dispatchers fire. Unknown fields (not on
    the schema) are shared by reference — they're typically static
    metadata or instance handles.
    """
    if state is None:
        return None, None
    if not isinstance(state, dict):
        return state, state
    inner_context = context if context is not None else state
    a, b = {}, {}
    for k, v in state.items():
        if k.startswith('_'):
            # Schema metadata leaking into state — share
            a[k] = b[k] = v
            continue
        sub_schema = getattr(schema, k, None)
        if sub_schema is None:
            # No typed schema for this field — share by reference
            a[k] = b[k] = v
        else:
            a[k], b[k] = divide(sub_schema, v, inner_context, path + (k,), rng)
    return a, b


@dispatch
def divide(schema: dict, state, context=None, path=(), rng=None):
    """Auto-resolved schemas often appear as plain Python dicts mapping
    field name → Node. Walk the dict the same way `apply(dict, ...)` does:
    for each schema field, recurse into the corresponding state value;
    unknown state keys are shared by reference (they have no schema info).
    """
    if state is None:
        return None, None
    if not isinstance(state, dict):
        return state, state
    inner_context = context if context is not None else state
    a, b = {}, {}
    for k, v in state.items():
        if k.startswith('_'):
            a[k] = b[k] = v
            continue
        sub_schema = schema.get(k) if isinstance(schema, dict) else None
        if sub_schema is None:
            a[k] = b[k] = v
        else:
            a[k], b[k] = divide(sub_schema, v, inner_context, path + (k,), rng)
    return a, b


@dispatch
def divide(schema: Empty, state, context=None, path=(), rng=None):
    return state, state


@dispatch
def divide(schema: Atom, state, context=None, path=(), rng=None):
    return state, state


@dispatch
def divide(schema: Boolean, state, context=None, path=(), rng=None):
    return state, state


@dispatch
def divide(schema: Integer, state, context=None, path=(), rng=None):
    """Binomial split on integer counts."""
    if state is None:
        return None, None
    rng = rng or np.random.default_rng()
    return _binomial_split(rng, state)


@dispatch
def divide(schema: Float, state, context=None, path=(), rng=None):
    """Floating-point: split in half."""
    if state is None:
        return None, None
    return state / 2.0, state / 2.0


@dispatch
def divide(schema: Number, state, context=None, path=(), rng=None):
    if state is None:
        return None, None
    if hasattr(state, 'dtype') and np.issubdtype(state.dtype, np.integer):
        rng = rng or np.random.default_rng()
        return _binomial_split(rng, state)
    return state / 2.0, state / 2.0


@dispatch
def divide(schema: String, state, context=None, path=(), rng=None):
    return state, state


@dispatch
def divide(schema: Array, state, context=None, path=(), rng=None):
    """Numerical array: binomial split on integer dtype, halve on float."""
    if state is None:
        return None, None
    if hasattr(state, 'dtype') and np.issubdtype(state.dtype, np.integer):
        rng = rng or np.random.default_rng()
        a, b = _binomial_split(rng, state)
        return a, b
    return state / 2.0, state / 2.0


@dispatch
def divide(schema: List, state, context=None, path=(), rng=None):
    """Default: share the list by reference. Override per element type
    if random partition is wanted."""
    return state, state


@dispatch
def divide(schema: Map, state, context=None, path=(), rng=None):
    """Recurse into each value, splitting field-by-field. Threads context
    and grows path so contextual dividers can resolve relative wires."""
    if state is None:
        return {}, {}
    if not isinstance(state, dict):
        return state, state
    # If no parent context was passed, treat the dict itself as the context.
    inner_context = context if context is not None else state
    a, b = {}, {}
    for k, v in state.items():
        a[k], b[k] = divide(
            schema._value, v, inner_context, path + (k,), rng)
    return a, b


@dispatch
def divide(schema: Tree, state, context=None, path=(), rng=None):
    """Tree: recurse with the same schema (the tree's leaf type carries
    the per-leaf semantics)."""
    if state is None:
        return None, None
    if not isinstance(state, dict):
        # Leaf — delegate to leaf type
        return divide(schema._leaf, state, context, path, rng)
    inner_context = context if context is not None else state
    a, b = {}, {}
    for k, v in state.items():
        a[k], b[k] = divide(schema, v, inner_context, path + (k,), rng)
    return a, b
