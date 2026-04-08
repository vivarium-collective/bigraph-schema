from plum import dispatch
import numpy as np

from bigraph_schema.methods.check import check
from bigraph_schema.methods.default import default

from bigraph_schema.schema import (
    Node,
    Atom,
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
    Quote,
    Maybe,
    Overwrite,
    Const,
    List,
    Set,
    Map,
    Tree,
    Array,
    Frame,
    Key,
    Path,
    Wires,
    Schema,
    Link,
    is_schema_field,
)


@dispatch
def apply(schema: Maybe, state, update, path):
    if state is None:
        if update is not None:
            return update, []
    elif update is None:
        return state, []
    else:
        return apply(schema._value, state, update, path)


@dispatch
def apply(schema: Wrap, state, update, path):
    return apply(schema._value, state, update, path)


@dispatch
def apply(schema: Overwrite, state, update, path):
    if update is None:
        return state, []
    # When both are dicts, delegate to the inner value schema
    # to preserve keys not in the update (e.g. listener keys).
    # Pure Overwrite (replace) only applies to leaf values.
    if isinstance(state, dict) and isinstance(update, dict):
        return apply(schema._value, state, update, path)
    return update, []


@dispatch
def apply(schema: Quote, state, update, path):
    # Opaque — if there's an update, replace; otherwise keep state
    if update is not None:
        return update, []
    return state, []


@dispatch
def apply(schema: Const, state, update, path):
    return state, []


@dispatch
def apply(schema: Union, state, update, path):
    found = None
    for option in schema._options:
        if check(option, state) and check(option, update):
            found = option
            break

    if found is not None:
        return apply(found, state, update, path)
    else:
        return update, []


@dispatch
def apply(schema: Tuple, state, update, path):
    merges = []
    result = []
    for index, value in enumerate(schema._values):
        if index < len(state):
            if update and index < len(update):
                substate, submerges = apply(
                    value,
                    state[index],
                    update[index],
                    path+(index,))
                result.append(substate)
                merges += submerges
            else:
                result.append(state[index])
        elif index < len(update):
            result.append(update[index])
        else:
            result.append(default(value))

    return tuple(result), merges


@dispatch
def apply(schema: List, state, update, path):
    # Coerce state to list if it arrived as wrong type
    if not isinstance(state, list):
        state = list(state) if hasattr(state, '__iter__') and not isinstance(state, (str, dict)) else []

    if update is None:
        return state, []

    result = []
    merges = []

    if isinstance(update, dict):
        if '_remove' in update:
            indexes = update['_remove']
            if indexes == 'all':
                result = []
            else:
                result = [
                    item
                    for index, item in enumerate(state)
                    if index not in indexes]

        if '_add' in update:
            result += update['_add']

    elif isinstance(update, np.ndarray):
        result = update

    elif isinstance(state, np.ndarray):
        result = state

    else:
        result = state + update

    return result, merges


@dispatch
def apply(schema: Set, state, update, path):
    result = set(state) if state else set()
    if isinstance(update, dict):
        if '_add' in update:
            result |= set(update['_add'])
        if '_remove' in update:
            result -= set(update['_remove'])
    elif isinstance(update, set):
        result |= update
    return result, []


@dispatch
def apply(schema: Map, state, update, path):
    if isinstance(state, list):
        state = {}
    result = state.copy() if isinstance(state, dict) else {}
    merges = []

    if update is None:
        return state, merges

    if '_add' in update:
        add_update = update['_add']
        if isinstance(add_update, list):
            for add_key, add_value in update['_add']:
                result[add_key] = add_value
        elif isinstance(add_update, dict):
            for add_key, add_value in update['_add'].items():
                result[add_key] = add_value

    for key, value in result.items():
        if key in update:
            result[key], submerges = apply(
                schema._value,
                value,
                update[key],
                path+(key,))
            merges += submerges

    if '_remove' in update:
        for remove_key in update['_remove']:
            if remove_key in result:
                del result[remove_key]

    return result, merges


@dispatch
def apply(schema: Tree, state, update, path):
    if check(schema._leaf, state):
        return apply(schema._leaf, state, update, path)

    result = state.copy()
    if '_remove' in update:
        for remove_key in update['_remove']:
            del result[remove_key]

    if '_add' in update:
        for add_key, add_value in update['_add']:
            result[add_key] = add_value

    for key, value in result:
        if key in update:
            result[key], submerges = apply(
                schema,
                value,
                update[key],
                path+(key,))
            merges += submerges

    return result, merges


@dispatch
def apply(schema: Atom, state, update, path):
    if update is None:
        return state, []
    if state is None:
        return update, []

    return (state + update), []


@dispatch
def apply(schema: String, state, update, path):
    return update, []

@dispatch
def apply(schema: Boolean, state, update, path):
    return update, []


@dispatch
def apply(schema: Or, state, update, path):
    return state or update, []


@dispatch
def apply(schema: And, state, update, path):
    return state and update, []


@dispatch
def apply(schema: Xor, state, update, path):
    return (state or update) and not (state and update), []


@apply.dispatch
def apply(schema: Frame, state, update, path):
    return update, []


@dispatch
def apply(schema: Array, state, update, path):
    # Ensure state is an ndarray — it may arrive as a list from
    # realize or merge when type information was lost.
    if isinstance(state, list):
        state = np.array(state, dtype=schema._data)

    if isinstance(update, list):
        # Sparse index updates: [(index, delta), ...]
        for idx, delta in update:
            state[idx] += delta
        return state, []

    elif isinstance(update, dict):
        # Field-level updates on structured arrays: {'field': values}
        # or nested update operations: {'set': {'field': values}}
        if isinstance(state, np.ndarray) and state.dtype.names:
            if 'set' in update:
                # Set semantics: replace field values
                for field, values in update['set'].items():
                    if field in state.dtype.names:
                        state[field] = values
            else:
                for field, values in update.items():
                    if field in state.dtype.names:
                        state[field] += values
            return state, []

    if hasattr(update, 'shape'):
        if state.size == 0:
            # State was initialized empty — replace with update
            state = update
        elif isinstance(state, np.ndarray) and state.dtype.names:
            # Structured array: add field-by-field for numeric fields
            index = tuple([
                slice(0, dimension)
                for dimension in update.shape])
            for field in state.dtype.names:
                if np.issubdtype(state.dtype[field], np.number):
                    state[field][index] += update[field][index]
        else:
            if state.ndim != update.ndim or state.shape != update.shape:
                # Shape mismatch — replace state with update
                state = update
            else:
                # Additive update for matching shapes
                state += update

    return state, []

@dispatch
def apply(schema: dict, state: np.ndarray, update, path):
    if update is None:
        return state, []
    merges = []
    for key, subschema in schema.items():
        if key in update:
            substate = update[key]
            state[key], submerges = apply(subschema, state[key], substate, path+(key,))
            merges += submerges
    return state, merges
                

@dispatch
def apply(schema: dict, state, update, path):
    if update is None:
        return state, []
    if state is None:
        return update, []

    merges = []
    result = {}

    for key, subschema in schema.items():
        if not is_schema_field(schema, key):
            continue

        if key not in state and key not in update:
            continue

        result[key], submerges = apply(
            subschema,
            state.get(key),
            update.get(key),
            path+(key,))
        merges += submerges

    for key in state.keys():
        if not key in result and not key in schema:
            result[key] = state[key]

    return result, merges


@dispatch
def apply(schema: Float, state, update, path):
    """Additive: float updates are deltas."""
    if update is None:
        return state, []
    if state is None:
        return update, []
    return state + update, []


@dispatch
def apply(schema: Integer, state, update, path):
    """Additive: integer updates are deltas."""
    if update is None:
        return state, []
    if state is None:
        return update, []
    return state + update, []


@dispatch
def apply(schema: Node, state, update, path):
    merges = []

    if isinstance(state, dict) and isinstance(update, dict):
        result = {}
        for key in schema.__dataclass_fields__:
            if not is_schema_field(schema, key):
                continue
            subschema = getattr(schema, key)
            result[key], submerges = apply(
                subschema,
                state.get(key),
                update.get(key),
                path+(key,))
            merges += submerges

        # Preserve state keys not covered by schema fields.
        # For keys in both state and update, recursively apply
        # using Node() as schema to preserve nested state.
        all_keys = set(state.keys()) | set(update.keys())
        for key in all_keys:
            if key not in result:
                if key in state and key in update:
                    result[key], submerges = apply(
                        Node(), state[key], update[key], path+(key,))
                    merges += submerges
                elif key in update:
                    result[key] = update[key]
                else:
                    result[key] = state[key]

    else:
        result = update

    return result, merges
