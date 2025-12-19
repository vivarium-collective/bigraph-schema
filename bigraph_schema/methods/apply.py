from plum import dispatch
import numpy as np

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
    Delta,
    Nonnegative,
    String,
    Enum,
    Wrap,
    Maybe,
    Overwrite,
    List,
    Map,
    Tree,
    Array,
    Key,
    Path,
    Wires,
    Schema,
    Link,
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
    return update, []


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
    else:
        result = state + update

    return result, merges


@dispatch
def apply(schema: Map, state, update, path):
    result = state.copy()
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


@dispatch
def apply(schema: Array, state, update, path):
    return state + update, []

@dispatch
def apply(schema: dict, state: np.ndarray, update, path):
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
        if key in ('_inherit',):
            continue

        if key not in state:
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
def apply(schema: Node, state, update, path):
    merges = []
    if isinstance(state, dict) and isinstance(update, dict):
        result = {}
        for key in schema.__dataclass_fields__:
            subschema = getattr(schema, key)
            result[key], submerges = apply(
                subschema,
                state.get(key),
                update.get(key),
                path+(key,))
            merges += submerges

    else:
        result = update

    return result, merges
