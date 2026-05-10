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
    Delta,
    String,
    List,
    Set,
    Map,
    Tree,
    Array,
    Link,
    Quantity,
    Wrap,
    DivideReset,
    DivideShare,
    LineageSeed,
    NPRandom,
)


# Key used by ``divide(NPRandom)`` to publish the divide rng on
# context for sibling dispatchers (``divide(Array)`` for binomial
# bulk splits, etc.) to consume.
DIVIDE_RNG_CONTEXT_KEY = '__divide_rng__'
from bigraph_schema.methods.default import default


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


def _split_keys_by_rng_first(schema, state):
    """Partition state keys into (NPRandom-typed first, then rest).

    The framework dispatches by schema type, but dict iteration order
    is insertion order — there's no guarantee that an NPRandom field
    is visited before its sibling Array fields. This helper enforces
    "rng-publishers run first" so ``divide(NPRandom)`` populates
    ``context[DIVIDE_RNG_CONTEXT_KEY]`` before ``divide(Array)`` calls
    that need it consume it.
    """
    is_dict = isinstance(schema, dict)
    def _sub(name):
        return schema.get(name) if is_dict else getattr(schema, name, None)
    rng_keys, other_keys = [], []
    for k in state:
        if k.startswith('_'):
            other_keys.append(k)
            continue
        if isinstance(_sub(k), NPRandom):
            rng_keys.append(k)
        else:
            other_keys.append(k)
    return rng_keys + other_keys


@dispatch
def divide(schema: Node, state, context=None, path=(), rng=None):
    """Walk an arbitrary Node's typed fields when state is a dict.

    Visits NPRandom-typed fields first so their dispatcher can
    publish the rng on inner_context, making it available to sibling
    fields' dispatchers (e.g. integer-array binomial splits). All
    other fields are visited in their original order.
    """
    if state is None:
        return None, None
    if not isinstance(state, dict):
        return state, state
    inner_context = context if context is not None else state
    a, b = {}, {}
    for k in _split_keys_by_rng_first(schema, state):
        v = state[k]
        if k.startswith('_'):
            a[k] = b[k] = v
            continue
        sub_schema = getattr(schema, k, None)
        if sub_schema is None:
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

    Same NPRandom-first ordering as the Node dispatcher above.
    """
    if state is None:
        return None, None
    if not isinstance(state, dict):
        return state, state
    inner_context = context if context is not None else state
    a, b = {}, {}
    for k in _split_keys_by_rng_first(schema, state):
        v = state[k]
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
def divide(schema: NPRandom, state, context=None, path=(), rng=None):
    """RNG state shares by reference between daughters AND publishes
    itself on ``context[DIVIDE_RNG_CONTEXT_KEY]`` so sibling Array
    dispatchers can use it for deterministic binomial splits.

    Without this, ``divide(Array)`` on integer counts falls back to
    a fresh ``np.random.default_rng()`` (system-entropy seed) every
    division — splits become non-deterministic across runs even with
    the same seeds and same state.
    """
    if isinstance(state, (np.random.RandomState, np.random.Generator)):
        if isinstance(context, dict):
            # Don't clobber an explicit publisher upstream (e.g. nested
            # cells with their own rng).
            context.setdefault(DIVIDE_RNG_CONTEXT_KEY, state)
    return state, state


@dispatch
def divide(schema: Wrap, state, context=None, path=(), rng=None):
    """Wrap subclasses delegate divide to their inner type. This lets
    nested wrappers like ``overwrite[divide_reset[boolean]]`` dispatch
    to the innermost type-specific divide (e.g. DivideReset).

    If this Wrap declares ``_default`` but the inner Wrap has none,
    propagate ours down so the inner can produce a correctly-shaped
    default. Without this, ``Overwrite[DivideReset[Array[Float]]]``
    declared with ``_default=[0]*21`` (on Overwrite) would divide to
    the bare ``Array`` default (length 1) instead of the configured
    21-element zero array.
    """
    inner = schema._value
    if (schema._default is not None
            and isinstance(inner, Wrap)
            and inner._default is None):
        from dataclasses import replace
        inner = replace(inner, _default=schema._default)
    return divide(inner, state, context, path, rng)


@dispatch
def divide(schema: DivideReset, state, context=None, path=(), rng=None):
    """Reset the inner value to its default on division.

    Mirrors v1's ``_divider: {set_value: <default>}``. Both daughters
    receive the inner type's default rather than the mother's value.
    """
    reset = default(schema)
    return reset, reset


@dispatch
def divide(schema: DivideShare, state, context=None, path=(), rng=None):
    """Share the inner value across both daughters.

    Mirrors v1's ``_divider: "set"``. Both daughters receive the same
    reference to mother's value — no copy, no halving, no reset.
    Use this to override a child type's default divide behavior, e.g.
    when ``array[float]`` would otherwise halve a rate-typed array.
    """
    return state, state


@dispatch
def divide(schema: LineageSeed, state, context=None, path=(), rng=None):
    """Share the base seed across both daughters.

    Per-daughter seed derivation happens at each daughter's next
    realize() — base values flow through divide unchanged so the
    ``LineageSeed`` realize dispatcher can recombine them with whatever
    ``DerivationContext`` is active when daughter processes are
    instantiated. Without this dispatch, ``Wrap``'s default would
    delegate to the inner Integer's binomial split, which would corrupt
    the seed.
    """
    return state, state


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
    """Default: share (each daughter gets the same value).

    Halving only makes sense for additive/extensive quantities
    (masses, concentrations × volume). Plain floats often represent
    intensive quantities (time, rates, ratios) where halving corrupts
    the value. Explicit ``Delta`` fields opt into halving; other
    extensive quantities should declare a custom ``divide`` dispatch
    (like ``BulkArray``'s binomial split)."""
    return state, state


@dispatch
def divide(schema: Delta, state, context=None, path=(), rng=None):
    """Additive-change type: halve so the two daughters sum to the
    mother's total (mass, cumulative growth, etc.)."""
    if state is None:
        return None, None
    return state / 2.0, state / 2.0


@dispatch
def divide(schema: Number, state, context=None, path=(), rng=None):
    """Numeric default: share. Integer counts should flow through
    ``BulkArray``/``Integer`` dispatches; this catch-all preserves
    values (e.g. bit-widths of scalars) rather than mutating them."""
    return state, state


@dispatch
def divide(schema: String, state, context=None, path=(), rng=None):
    return state, state


@dispatch
def divide(schema: Array, state, context=None, path=(), rng=None):
    """Numerical array: binomial split for non-negative integer arrays
    (interpretable as counts), halve for floats, share by reference
    for anything else.

    The binomial rng comes from (in order): explicit ``rng=`` arg →
    ``context[DIVIDE_RNG_CONTEXT_KEY]`` (published by an
    ``NPRandom`` dispatcher visited earlier in the same divide walk)
    → ``np.random.default_rng()`` fallback. The fallback is
    non-deterministic; it should only fire when no rng exists in the
    schema being divided, which is rarely what callers want.

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
            if rng is None and isinstance(context, dict):
                rng = context.get(DIVIDE_RNG_CONTEXT_KEY)
            if rng is None:
                rng = np.random.default_rng()
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
    """Produce two daughter link DECLARATIONS from a mother's link state.

    Instead of cloning the mother's instance, we strip the 'instance'
    key and keep only the declaration fields (address, config, wires).
    The framework's realize() step will instantiate fresh instances for
    each daughter from the address + config — exactly like at initial
    composite creation.

    This ensures daughters get completely fresh process state (new
    RandomState, empty caches, etc.) rather than inheriting stale
    mother state.
    """
    if not isinstance(state, dict):
        return state, state

    # Strip the instance — keep everything else (address, config, wires,
    # _inputs, _outputs, _type, priority/interval). realize() will
    # re-instantiate from address + config.
    d1 = {k: v for k, v in state.items() if k != 'instance'}
    d2 = {k: v for k, v in state.items() if k != 'instance'}
    return d1, d2


@dispatch
def divide(schema: Quantity, state, context=None, path=(), rng=None):
    """Unit-aware split: extensive (mass/amount/volume) halves, intensive
    (concentration, rate, time, dimensionless) shares.

    Computed from ``state.dimensionality``: scale factor =
    ``mass_exp + substance_exp + length_exp / 3``. Positive →
    extensive → halve; otherwise intensive → share."""
    import pint
    if state is None:
        return None, None
    if not isinstance(state, pint.Quantity):
        return state, state
    dim = dict(state.dimensionality)
    mass = float(dim.get('[mass]', 0))
    substance = float(dim.get('[substance]', 0))
    length = float(dim.get('[length]', 0))
    scale = mass + substance + length / 3.0
    if scale > 0:
        half = state / 2
        return half, half
    return state, state
