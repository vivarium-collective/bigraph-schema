"""
Reconcile multiple updates to the same state path into a single update.

When multiple steps produce updates targeting the same state path within
one timestep, reconcile combines them according to the schema type's
semantics before apply is called.

The relationship:
    apply(schema, state, reconcile(schema, [u1, u2, u3]))
    ==
    apply(schema, apply(schema, apply(schema, state, u1), u2), u3)

...for commutative types. For non-commutative types (like UniqueArray),
reconcile provides correct batching that sequential apply cannot.

Reconcile also populates an optional ``ReconcileSummary`` sink as it
walks: leaf paths and a structural-sentinel flag. Callers that need
that info (notably ``Composite.apply_updates``) install the sink before
calling and read it after — eliminating a redundant ``_walk_update``
pass over the same tree.
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
    Frame,
    Link,
)
from bigraph_schema.methods.events import get_reconcile_sink


@dispatch
def reconcile(schema: Node, updates: list):
    """Default: merge dict updates recursively, last value wins for leaves."""
    # If all updates are dicts, merge them key-by-key
    dict_updates = [u for u in updates if isinstance(u, dict)]
    if dict_updates and len(dict_updates) == len([u for u in updates if u is not None]):
        return reconcile({}, dict_updates)

    # Otherwise, last non-None wins
    for update in reversed(updates):
        if update is not None:
            return update
    return None


@dispatch
def reconcile(schema: Empty, updates: list):
    return None


@dispatch
def reconcile(schema: Const, updates: list):
    """Immutable — ignore all updates."""
    return None


@dispatch
def reconcile(schema: Overwrite, updates: list):
    """Last non-None update wins."""
    for update in reversed(updates):
        if update is not None:
            return update
    return None


@dispatch
def reconcile(schema: Float, updates: list):
    """Sum all numeric deltas."""
    total = 0.0
    for update in updates:
        if update is not None:
            total += update
    return total if total != 0.0 else None


@dispatch
def reconcile(schema: Integer, updates: list):
    """Sum all integer deltas."""
    total = 0
    for update in updates:
        if update is not None:
            total += update
    return total if total != 0 else None


@dispatch
def reconcile(schema: Boolean, updates: list):
    """Last non-None update wins."""
    for update in reversed(updates):
        if update is not None:
            return update
    return None


@dispatch
def reconcile(schema: String, updates: list):
    """Last non-None update wins."""
    for update in reversed(updates):
        if update is not None:
            return update
    return None


def _apply_sparse_dict_into(arr, sparse):
    """Add a sparse-dict update ``{k: {k2: val}}`` into ndarray ``arr`` in place."""
    for k, v in sparse.items():
        if isinstance(v, dict):
            _apply_sparse_dict_into(arr[k], v)
        else:
            arr[k] = arr[k] + v


def _apply_sparse_list_into(arr, sparse_list):
    """Add a sparse-list update ``[(idx, delta), ...]`` into ndarray ``arr`` in place.

    Uses numpy's batched ``np.add.at`` to avoid per-element Python overhead
    when the sparse list is non-trivial.
    """
    n = len(sparse_list)
    if n == 0:
        return
    if n == 1:
        idx, val = sparse_list[0]
        arr[idx] = arr[idx] + val
        return
    indices = np.fromiter((s[0] for s in sparse_list), dtype=np.intp, count=n)
    deltas = np.fromiter((s[1] for s in sparse_list), dtype=arr.dtype, count=n)
    np.add.at(arr, indices, deltas)


def _merge_array_deltas(a, b):
    """Combine two Array deltas. Handles:
      - dict + dict: sparse-coordinate union with recursive merge, summing numeric leaves.
      - ndarray + ndarray: element-wise sum.
      - ndarray + (dict | list): copy the ndarray and apply the sparse update in place
        (one full-array allocation instead of two: a scratch zeros + the sum).
    """
    if isinstance(a, dict) and isinstance(b, dict):
        merged = dict(a)
        for k, v in b.items():
            if k in merged:
                merged[k] = _merge_array_deltas(merged[k], v)
            else:
                merged[k] = v
        return merged
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a + b
    arr_operand, other = (a, b) if isinstance(a, np.ndarray) else (
        (b, a) if isinstance(b, np.ndarray) else (None, None))
    if arr_operand is not None:
        if isinstance(other, dict):
            result = arr_operand.copy()
            _apply_sparse_dict_into(result, other)
            return result
        if isinstance(other, list):
            result = arr_operand.copy()
            _apply_sparse_list_into(result, other)
            return result
    return a + b


@dispatch
def reconcile(schema: Array, updates: list):
    """Element-wise sum of array deltas. Supports homogeneous and mixed
    update forms (ndarray, sparse list ``[(idx, delta), ...]``, sparse dict
    ``{j: {i: val}}``, and ``{'set': ...}`` overwrite)."""
    result = None
    for update in updates:
        if update is None:
            continue
        if isinstance(update, dict) and 'set' in update:
            # Set overwrites — last one wins.
            result = update
            continue
        if result is None:
            if isinstance(update, list):
                result = list(update)
            else:
                result = update
            continue
        if isinstance(update, list) and isinstance(result, list):
            result.extend(update)
            continue
        result = _merge_array_deltas(result, update)
    return result


@dispatch
def reconcile(schema: List, updates: list):
    """Merge list operations: concatenate adds, union removes."""
    adds = []
    removes = set()
    plain = []

    for update in updates:
        if update is None:
            continue
        if isinstance(update, dict):
            if '_add' in update:
                adds.extend(update['_add'])
            if '_remove' in update:
                rm = update['_remove']
                if rm == 'all':
                    removes = 'all'
                elif removes != 'all':
                    removes.update(rm if isinstance(rm, (set, frozenset)) else rm)
        elif isinstance(update, list):
            plain.extend(update)

    if plain and not adds and removes != 'all':
        # Plain list concatenation
        return plain

    result = {}
    if adds:
        result['_add'] = adds
    if removes:
        result['_remove'] = list(removes) if isinstance(removes, set) else removes
    return result if result else None


@dispatch
def reconcile(schema: Map, updates: list):
    """Merge map operations: union adds, union removes, recursive merge values.

    When multiple updates target the same key, they are recursively
    reconciled using the map's value schema instead of last-write-wins.
    This is essential when multiple processes write to different fields
    of the same map entry within a single timestep.

    Structural sentinels (``_add``/``_remove``/``_divide``) are carved
    out because ``apply(Map, ...)`` handles them holistically — they
    are not per-key value updates.
    """
    sink = get_reconcile_sink()
    adds = {}
    removes = []
    divide = None  # Last non-None _divide wins.
    # Group regular key updates by key so multiple updates to the same
    # key can be recursively reconciled.
    grouped_value_updates = {}

    for update in updates:
        if update is None:
            continue
        if isinstance(update, dict):
            if '_add' in update:
                add = update['_add']
                if sink is not None:
                    sink.has_structural = True
                if isinstance(add, dict):
                    adds.update(add)
                elif isinstance(add, list):
                    for k, v in add:
                        adds[k] = v
            if '_remove' in update:
                if sink is not None:
                    sink.has_structural = True
                removes.extend(update['_remove'])
            if '_divide' in update and update['_divide'] is not None:
                if sink is not None:
                    sink.has_structural = True
                divide = update['_divide']
            # Regular key updates: collect ALL updates per key, not just the last
            for key, value in update.items():
                if key not in ('_add', '_remove', '_divide'):
                    grouped_value_updates.setdefault(key, []).append(value)

    # Recursively reconcile multiple updates targeting the same key
    value_schema = schema._value
    value_updates = {}
    if sink is not None:
        # When a sink is installed, always recurse so deeper paths get
        # tracked. The recursion on a single update is cheap (returns
        # the update unchanged for atomic types, walks for dict/Map).
        parent_path = sink.path_stack
        for key, sub_updates in grouped_value_updates.items():
            sink.paths.append(parent_path + (key,))
            sink.path_stack = parent_path + (key,)
            try:
                reconciled = reconcile(value_schema, sub_updates)
                if reconciled is not None:
                    value_updates[key] = reconciled
                elif len(sub_updates) == 1:
                    # Atomic-typed reconcile may return None to signal
                    # "no-op delta" (e.g. zero Float). Single-update
                    # path keeps the original update intact.
                    value_updates[key] = sub_updates[0]
            finally:
                sink.path_stack = parent_path
    else:
        for key, sub_updates in grouped_value_updates.items():
            if len(sub_updates) == 1:
                value_updates[key] = sub_updates[0]
            else:
                reconciled = reconcile(value_schema, sub_updates)
                if reconciled is not None:
                    value_updates[key] = reconciled

    result = {}
    if adds:
        result['_add'] = adds
    if removes:
        result['_remove'] = removes
    if divide is not None:
        result['_divide'] = divide
    result.update(value_updates)
    return result if result else None


@dispatch
def reconcile(schema: Tree, updates: list):
    """Recursive reconcile on matching keys."""
    return reconcile(schema._leaf, updates)


@dispatch
def reconcile(schema: Wrap, updates: list):
    """Delegate to inner value type."""
    return reconcile(schema._value, updates)


_STRUCTURAL_SENTINELS = frozenset({'_add', '_remove', '_divide', '_type'})


@dispatch
def reconcile(schema: dict, updates: list):
    """Reconcile dict schema: group updates by key, reconcile each.

    Structural sentinels (``_add``/``_remove``/``_divide``/``_type``)
    are preserved as-is — they are apply-layer directives, not schema
    fields. Without this carve-out, ``is_schema_field`` filters them
    out and the structural change (e.g., cell division) never reaches
    apply."""
    from bigraph_schema.schema import is_schema_field

    sink = get_reconcile_sink()

    # Collect all keys across updates
    all_keys = set()
    for update in updates:
        if isinstance(update, dict):
            all_keys.update(update.keys())

    result = {}
    parent_path = sink.path_stack if sink is not None else ()
    for key in all_keys:
        if key in _STRUCTURAL_SENTINELS:
            if sink is not None:
                sink.has_structural = True
            # Last non-None wins for structural directives — they
            # don't need to be deep-merged because apply() handles
            # them holistically.
            for update in reversed(updates):
                if isinstance(update, dict) and key in update and update[key] is not None:
                    result[key] = update[key]
                    break
            continue
        if not is_schema_field(schema, key):
            continue
        key_updates = []
        for update in updates:
            if isinstance(update, dict) and key in update:
                key_updates.append(update[key])
        if key_updates:
            subschema = schema.get(key, Node())
            if sink is not None:
                sink.paths.append(parent_path + (key,))
                sink.path_stack = parent_path + (key,)
                try:
                    reconciled = reconcile(subschema, key_updates)
                finally:
                    sink.path_stack = parent_path
            else:
                reconciled = reconcile(subschema, key_updates)
            if reconciled is not None:
                result[key] = reconciled

    return result if result else None
