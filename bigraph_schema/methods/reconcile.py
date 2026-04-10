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


@dispatch
def reconcile(schema: Array, updates: list):
    """Element-wise sum of array deltas."""
    result = None
    for update in updates:
        if update is None:
            continue
        if isinstance(update, list):
            # Sparse updates: [(idx, delta), ...]
            if result is None:
                result = []
            result.extend(update)
        elif isinstance(update, np.ndarray):
            if result is None:
                result = np.zeros_like(update)
            result = result + update
        elif isinstance(update, dict):
            # Field-level updates: {'field': values} or {'set': {...}}
            if result is None:
                result = {}
            if 'set' in update:
                # Set overwrites — last one wins
                result = update
            else:
                for field, values in update.items():
                    if field in result:
                        result[field] = result[field] + values
                    else:
                        result[field] = values
        else:
            result = update
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
    """
    adds = {}
    removes = []
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
            # Regular key updates: collect ALL updates per key, not just the last
            for key, value in update.items():
                if key not in ('_add', '_remove'):
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


@dispatch
def reconcile(schema: dict, updates: list):
    """Reconcile dict schema: group updates by key, reconcile each."""
    from bigraph_schema.schema import is_schema_field

    # Collect all keys across updates
    all_keys = set()
    for update in updates:
        if isinstance(update, dict):
            all_keys.update(update.keys())

    result = {}
    for key in all_keys:
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
