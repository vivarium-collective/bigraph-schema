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
    update forms (ndarray, sparse list ``[(idx, delta), ...]``, sparse
    dict ``{j: {i: val}}``).

    ``{'set': {field: values}}`` updates overwrite specified fields on
    structured arrays. Apply treats ``set`` as absorbing — its branch
    is ``if 'set' in update: ... else: <additive>``, so a single update
    is either a set or additive, never both. Reconcile mirrors that:
    when any update in the batch is a ``set`` form, the last set wins
    and concurrent deltas in the same batch are dropped (they would
    target state being overwritten anyway).
    """
    last_set = None
    result = None

    for update in updates:
        if update is None:
            continue
        if isinstance(update, dict) and 'set' in update:
            last_set = update
            continue
        if last_set is not None:
            # Set is absorbing — drop deltas after / before that
            # appear once we've seen a set in the batch.
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

    if last_set is not None:
        return last_set
    return result


@dispatch
def reconcile(schema: List, updates: list):
    """Merge list operations into a single update.

    Rule: apply removes first, then all adds — irrespective of the
    order in which updates were received. ``{'_remove': 'all'}``
    therefore clears only pre-existing state; items contributed by
    sibling updates in the same batch survive.

    Plain-list updates (``[x, y]``) are equivalent to ``{'_add': [x, y]}``
    and are folded into the combined ``_add`` whenever any structural
    sentinel appears in the batch. When no sentinels appear, the
    plain-list fast-path returns a concatenation rather than a dict.
    """
    adds = []
    remove_indexes = []
    seen_indexes = set()
    remove_all = False
    has_structural = False

    for update in updates:
        if update is None:
            continue
        if isinstance(update, dict):
            if '_add' in update:
                has_structural = True
                adds.extend(update['_add'])
            if '_remove' in update:
                has_structural = True
                rm = update['_remove']
                if rm == 'all':
                    remove_all = True
                elif not remove_all:
                    for idx in rm:
                        if idx not in seen_indexes:
                            seen_indexes.add(idx)
                            remove_indexes.append(idx)
        elif isinstance(update, list):
            adds.extend(update)

    if not has_structural:
        return adds if adds else None

    result = {}
    if adds:
        result['_add'] = adds
    if remove_all:
        result['_remove'] = 'all'
    elif remove_indexes:
        result['_remove'] = remove_indexes
    return result if result else None


@dispatch
def reconcile(schema: Set, updates: list):
    """Merge set operations into a single update.

    Symmetric with List: apply removes first, then all adds —
    irrespective of receipt order. ``{'_remove': 'all'}`` clears
    pre-existing state only; sibling additions in the same batch
    survive.

    Plain set updates (a bare ``set``) are equivalent to
    ``{'_add': set}`` and fold into the combined ``_add`` whenever any
    structural sentinel appears in the batch. When no sentinels appear,
    plain set updates are returned as a unioned set rather than a dict.
    """
    adds = []
    seen_adds = set()
    removes = []
    seen_removes = set()
    remove_all = False
    has_structural = False

    def _add_one(item):
        if item not in seen_adds:
            seen_adds.add(item)
            adds.append(item)

    for update in updates:
        if update is None:
            continue
        if isinstance(update, dict):
            if '_add' in update:
                has_structural = True
                for item in update['_add']:
                    _add_one(item)
            if '_remove' in update:
                has_structural = True
                rm = update['_remove']
                if rm == 'all':
                    remove_all = True
                elif not remove_all:
                    rm_iter = rm if isinstance(rm, (set, frozenset, list, tuple)) else [rm]
                    for item in rm_iter:
                        if item not in seen_removes:
                            seen_removes.add(item)
                            removes.append(item)
        elif isinstance(update, (set, frozenset)):
            for item in update:
                _add_one(item)

    if not has_structural:
        return set(adds) if adds else None

    result = {}
    if adds:
        result['_add'] = set(adds)
    if remove_all:
        result['_remove'] = 'all'
    elif removes:
        result['_remove'] = set(removes)
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
    seen_removes = set()
    remove_all = False
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
                rm = update['_remove']
                if rm == 'all':
                    remove_all = True
                elif not remove_all:
                    for k in rm:
                        if k not in seen_removes:
                            seen_removes.add(k)
                            removes.append(k)
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
    if remove_all:
        result['_remove'] = 'all'
    elif removes:
        result['_remove'] = removes
    if divide is not None:
        result['_divide'] = divide
    result.update(value_updates)
    return result if result else None


@dispatch
def reconcile(schema: Tuple, updates: list):
    """Reconcile per-position Tuple updates.

    Tuple updates are positional sequences (lists or tuples). For each
    position ``i`` over ``schema._values``, collect contributions from
    every update where ``i < len(update)`` and ``update[i] is not
    None``, then recurse with the component schema at position ``i``.

    Missing positions and explicit Nones are treated as "no update at
    that position" — symmetric with the apply path, where
    ``apply(component, state[i], None, ...)`` returns ``state[i]``
    unchanged.
    """
    non_none = [u for u in updates if u is not None]
    if not non_none:
        return None
    if len(non_none) == 1:
        return non_none[0]

    n = len(schema._values)
    result = [None] * n
    has_any = False

    for i in range(n):
        contributions = [
            u[i] for u in non_none
            if i < len(u) and u[i] is not None]
        if not contributions:
            continue
        if len(contributions) == 1:
            result[i] = contributions[0]
        else:
            result[i] = reconcile(schema._values[i], contributions)
        has_any = True

    return result if has_any else None


@dispatch
def reconcile(schema: Tree, updates: list):
    """Reconcile Tree updates.

    Tree is recursively ``dict-of-Tree-or-leaf``. ``apply(Tree, ...)``
    decides per-call whether to treat state as a leaf (dispatch to
    ``apply(schema._leaf, ...)``) or as a tree-node (handle structural
    sentinels + recurse per-key with ``apply(schema, ...)``).

    Reconcile has no state to inspect, so mode is inferred from the
    update shapes:

    - All non-None updates are non-dict → leaf mode: delegate to
      ``reconcile(schema._leaf, updates)``.
    - Any update is a dict → tree-node mode: handle ``_add`` /
      ``_remove`` structurally at this node and recurse per-key with
      ``reconcile(schema, ...)``.
    - Mixed batch (both dict and non-dict updates): the non-dict
      updates correspond to whole-node overwrites in ``apply``; resolve
      to last-non-None-wins among the overwrites and drop earlier
      dict-form contributions, matching the apply-level outcome.

    Limitation: when ``schema._leaf`` itself accepts dict-shaped updates
    (rare — e.g. ``Tree[Map[Float]]``), the disambiguation can't be
    decided without state. Tree is intended for atomic leaf types.
    """
    non_none = [u for u in updates if u is not None]
    if not non_none:
        return None

    any_dict = any(isinstance(u, dict) for u in non_none)
    any_non_dict = any(not isinstance(u, dict) for u in non_none)

    if not any_dict:
        return reconcile(schema._leaf, updates)

    if any_non_dict:
        # Whole-node overwrite wins; sequential apply would land on
        # whichever non-dict update came last.
        for u in reversed(updates):
            if u is not None and not isinstance(u, dict):
                return u
        # All "non-dict" turned out to be None — fall through.

    sink = get_reconcile_sink()
    adds = {}
    removes = []
    grouped_value_updates = {}

    for update in updates:
        if not isinstance(update, dict):
            continue
        if '_add' in update:
            if sink is not None:
                sink.has_structural = True
            add = update['_add']
            if isinstance(add, dict):
                adds.update(add)
            elif isinstance(add, list):
                for k, v in add:
                    adds[k] = v
        if '_remove' in update:
            if sink is not None:
                sink.has_structural = True
            removes.extend(update['_remove'])
        for key, value in update.items():
            if key not in ('_add', '_remove'):
                grouped_value_updates.setdefault(key, []).append(value)

    parent_path = sink.path_stack if sink is not None else ()
    value_updates = {}
    for key, sub_updates in grouped_value_updates.items():
        if sink is not None:
            sink.paths.append(parent_path + (key,))
            sink.path_stack = parent_path + (key,)
            try:
                reconciled = reconcile(schema, sub_updates)
            finally:
                sink.path_stack = parent_path
        else:
            if len(sub_updates) == 1:
                reconciled = sub_updates[0]
            else:
                reconciled = reconcile(schema, sub_updates)
        if reconciled is not None:
            value_updates[key] = reconciled

    result = {}
    if adds:
        result['_add'] = adds
    if removes:
        result['_remove'] = removes
    result.update(value_updates)
    return result if result else None


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
    parent_path = sink.path_stack if sink is not None else ()

    # Single-update fast path: most apply_updates calls in a layered
    # composite end up reconciling exactly one update at the dict
    # level. Skip the per-key gather and just walk the one update's
    # keys directly.
    if len(updates) == 1:
        update = updates[0]
        if not isinstance(update, dict):
            return None
        result = {}
        for key, value in update.items():
            if isinstance(key, str) and key.startswith('_'):
                if key in _STRUCTURAL_SENTINELS:
                    if sink is not None:
                        sink.has_structural = True
                    if value is not None:
                        result[key] = value
                    continue
                if isinstance(schema, dict) and not is_schema_field(schema, key):
                    continue
            # Match the general path's filter (is_schema_field, not strict
            # dict-membership). is_schema_field is map-aware: dynamic keys
            # of map schemas (e.g. agent IDs in a `pool: map[...]`) are
            # valid even though they don't appear in the static schema dict.
            # Strict membership rejected them, breaking dynamic-structure
            # tests that grow a pool of agents at runtime.
            if isinstance(schema, dict) and not is_schema_field(schema, key):
                continue
            subschema = schema.get(key, Node()) if isinstance(schema, dict) else Node()
            if sink is not None:
                sink.paths.append(parent_path + (key,))
                sink.path_stack = parent_path + (key,)
                try:
                    reconciled = reconcile(subschema, [value])
                finally:
                    sink.path_stack = parent_path
            else:
                reconciled = reconcile(subschema, [value])
            if reconciled is not None:
                result[key] = reconciled
        return result if result else None

    # General path: walk updates ONCE, collecting per-key value lists
    # in the same pass that builds the key set. Saves a second loop
    # over updates per key.
    keys_with_updates: dict = {}
    for update in updates:
        if not isinstance(update, dict):
            continue
        for k, v in update.items():
            keys_with_updates.setdefault(k, []).append(v)

    result = {}
    for key, key_updates in keys_with_updates.items():
        if isinstance(key, str) and key.startswith('_'):
            if key in _STRUCTURAL_SENTINELS:
                if sink is not None:
                    sink.has_structural = True
                if key == '_add':
                    # Union all _add contributions, mirroring Map's
                    # semantics so concurrent processes can each
                    # contribute new keys without overwriting.
                    merged = {}
                    for v in key_updates:
                        if v is None:
                            continue
                        if isinstance(v, dict):
                            merged.update(v)
                        elif isinstance(v, (list, tuple)):
                            for entry in v:
                                if isinstance(entry, (list, tuple)) and len(entry) == 2:
                                    merged[entry[0]] = entry[1]
                    if merged:
                        result[key] = merged
                elif key == '_remove':
                    # Union all _remove contributions.
                    merged_remove = []
                    seen = set()
                    for v in key_updates:
                        if v is None:
                            continue
                        for item in v:
                            if item not in seen:
                                seen.add(item)
                                merged_remove.append(item)
                    if merged_remove:
                        result[key] = merged_remove
                else:
                    # _divide / _type — last non-None wins. These are
                    # singleton directives; a second contribution within
                    # one tick is at best redundant, at worst a
                    # contradiction (worth flagging in a future hardening
                    # pass).
                    for v in reversed(key_updates):
                        if v is not None:
                            result[key] = v
                            break
                continue
            # Underscore key that isn't a structural sentinel — only
            # keep it if the schema declares it.
            if not is_schema_field(schema, key):
                continue
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
