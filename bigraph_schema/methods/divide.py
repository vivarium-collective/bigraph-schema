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
    Link,
)


def _binomial_split(rng, total):
    """Sample (a, b) such that a + b == total.

    For non-negative integers (numpy arrays or scalars), uses binomial(p=0.5).
    For negative integers, falls back to deterministic halve so we don't
    crash numpy.binomial (which requires n >= 0). For floats, halves.
    """
    # Numpy array path
    if isinstance(total, np.ndarray):
        if np.issubdtype(total.dtype, np.integer):
            if (total < 0).any():
                # Mixed sign — deterministic halve preserves the sum
                a = total // 2
                return a, total - a
            a = rng.binomial(total, 0.5).astype(total.dtype)
            return a, total - a
        return total / 2.0, total / 2.0
    # Numpy scalar (np.int64 etc.) — has dtype but is not ndarray
    if hasattr(total, 'dtype') and np.issubdtype(total.dtype, np.integer):
        if int(total) < 0:
            a = int(total) // 2
            return a, int(total) - a
        a = int(rng.binomial(int(total), 0.5))
        return a, int(total) - a
    # Plain Python int
    if isinstance(total, (int, np.integer)):
        if int(total) < 0:
            a = int(total) // 2
            return a, int(total) - a
        a = int(rng.binomial(int(total), 0.5))
        return a, int(total) - a
    # Float / unknown — halve
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
    """Numerical array: binomial split for non-negative integer arrays
    (interpretable as counts), halve for floats, share by reference
    for anything else.

    A generic Array could carry counts, indices, deltas, timestamps,
    listeners, or sparse update buffers, and the framework has no way
    to know which. Splitting only makes sense for non-negative
    integer counts (binomial) and floats (halve). Negative-integer
    arrays (deltas, signed indices) crash numpy.binomial — share
    them instead. Subclasses that know their semantics (e.g.
    BulkArray, UniqueArray in vEcoli) override this dispatcher with
    a real molecule-aware split.
    """
    if state is None:
        return None, None
    if isinstance(state, np.ndarray):
        if np.issubdtype(state.dtype, np.integer):
            if (state < 0).any():
                # Mixed signs — share by reference (safest)
                return state, state
            rng = rng or np.random.default_rng()
            return _binomial_split(rng, state)
        if np.issubdtype(state.dtype, np.floating):
            return state / 2.0, state / 2.0
        # Unknown dtype — share
        return state, state
    return state, state


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


@dispatch
def divide(schema: Link, state, context=None, path=(), rng=None):
    """Clone a Link (process/step instance) into two fresh instances.

    A Link's state is a dict carrying:
        - 'instance': the Python object running the process/step logic
        - 'config' (or 'parameters'): the constructor args
        - 'address', '_inputs', '_outputs', 'inputs', 'outputs': wiring

    To produce two viable daughters, we instantiate the same class with
    the same config TWICE. Each daughter gets its own fresh `instance`
    so per-cell state (caches, RandomState, accumulators) is no longer
    shared between daughters.

    The non-instance fields ('address', wires, schemas) are reused by
    reference — they describe the link, not its mutable state.

    This mirrors v1 vivarium's `composer.generate()` which builds a
    fresh process tree per daughter cell.
    """
    if not isinstance(state, dict):
        return state, state
    instance = state.get('instance')
    if instance is None:
        # No instance — share by reference (e.g. partially-built link)
        return state, state

    cls = type(instance)
    # Prefer the explicit config; fall back to instance.parameters which
    # most vivarium-style processes carry.
    config = state.get('config')
    if config is None:
        config = getattr(instance, 'parameters', None) or getattr(instance, 'config', None)
    if config is None:
        return state, state

    try:
        d1_instance = cls(config)
        d2_instance = cls(config)
    except Exception:
        # Class construction failed — fall back to share-by-reference.
        # This keeps division non-fatal even when a link can't be cloned.
        return state, state

    # Vivarium-style processes do per-port cache initialization inside
    # ports_schema() as a side effect (e.g. Requester sets
    # self.cached_bulk_ports there). Call it once on each fresh instance
    # so the daughters' instances are fully wired up the same way the
    # mother's was at composer-build time.
    for inst in (d1_instance, d2_instance):
        if hasattr(inst, 'ports_schema'):
            try:
                inst.ports_schema()
            except Exception:
                pass

    # Build daughter link states by shallow-copying the original dict
    # and replacing only the instance. The rest (wires, schemas, config)
    # is shared by reference — it describes the link, not its state.
    d1 = dict(state)
    d2 = dict(state)
    d1['instance'] = d1_instance
    d2['instance'] = d2_instance
    return d1, d2
