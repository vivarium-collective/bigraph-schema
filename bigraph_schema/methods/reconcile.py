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


def _add_sparse_dict_into(out, sparse, prefix=()):
    """Additively apply a sparse-dict update ``{k: {k2: val}}`` into ndarray ``out``."""
    for k, v in sparse.items():
        idx = prefix + (k,)
        if isinstance(v, dict):
            _add_sparse_dict_into(out, v, idx)
        else:
            out[idx] = out[idx] + v


def _add_sparse_list_into(out, sparse_list):
    """Additively apply a sparse-list update ``[(idx, delta), ...]`` into ndarray ``out``."""
    for idx, val in sparse_list:
        out[idx] = out[idx] + val


def _merge_array_deltas(a, b):
    """Combine two Array deltas. Handles:
      - dict + dict: sparse-coordinate union with recursive merge, summing numeric leaves.
      - ndarray + ndarray: element-wise sum.
      - ndarray + (dict | list): normalize sparse form to ndarray of additions, then sum.
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
        sparse_arr = np.zeros(arr_operand.shape, dtype=arr_operand.dtype)
        if isinstance(other, dict):
            _add_sparse_dict_into(sparse_arr, other)
            return arr_operand + sparse_arr
        if isinstance(other, list):
            _add_sparse_list_into(sparse_arr, other)
            return arr_operand + sparse_arr
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
                if isinstance(add, dict):
                    adds.update(add)
                elif isinstance(add, list):
                    for k, v in add:
                        adds[k] = v
            if '_remove' in update:
                removes.extend(update['_remove'])
            if '_divide' in update and update['_divide'] is not None:
                divide = update['_divide']
            # Regular key updates: collect ALL updates per key, not just the last
            for key, value in update.items():
                if key not in ('_add', '_remove', '_divide'):
                    grouped_value_updates.setdefault(key, []).append(value)

    # Recursively reconcile multiple updates targeting the same key
    value_schema = schema._value
    value_updates = {}
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

    # Collect all keys across updates
    all_keys = set()
    for update in updates:
        if isinstance(update, dict):
            all_keys.update(update.keys())

    result = {}
    for key in all_keys:
        if key in _STRUCTURAL_SENTINELS:
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
            reconciled = reconcile(subschema, key_updates)
            if reconciled is not None:
                result[key] = reconciled

    return result if result else None
