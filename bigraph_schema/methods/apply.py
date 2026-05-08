from plum import dispatch
import numpy as np

from bigraph_schema.methods.check import check
from bigraph_schema.methods.default import default

from bigraph_schema.methods.divide import divide as _divide_state
from bigraph_schema.methods.events import emit, NodeAdded, NodeRemoved, Divided

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
    """Overwrite means replace. Always.

    Sibling-preservation for structured types is the job of the
    structured dict / Node schema itself — declare the dict layout
    explicitly with per-leaf overwrite[T] rather than wrapping a
    whole subtree in overwrite[node].
    """
    if update is None:
        return state, []
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
def apply(schema: Path, state, update, path):
    """Path is a list whose update semantics are replacement, not concat.

    Wire paths (the leaves of a Wires tree) are atomic addresses, so an
    update of `['a', 'b']` should *replace* the wire, not append to it.
    """
    if update is None:
        return state, []
    if isinstance(update, list):
        return list(update), []
    return update, []


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
        if '_remove' in update:
            rm = update['_remove']
            if rm == 'all':
                result = set()
            else:
                result -= set(rm)
        if '_add' in update:
            result |= set(update['_add'])
    elif isinstance(update, set):
        result |= update
    return result, []


def _deep_merge_into(base, overlay):
    """Recursively merge `overlay` into `base`, mutating `base`.

    For matching dict keys, recurse. For non-dict values (or where
    only one side has a key), the overlay wins. Used by `_divide` to
    layer caller-provided per-daughter overrides on top of the type-
    driven divide walk result.
    """
    if not isinstance(overlay, dict):
        return overlay
    if not isinstance(base, dict):
        return overlay
    for key, overlay_value in overlay.items():
        if key in base and isinstance(base[key], dict) and isinstance(overlay_value, dict):
            base[key] = _deep_merge_into(base[key], overlay_value)
        else:
            base[key] = overlay_value
    return base


def _handle_divide_sentinel(value_schema, state, update, path):
    """Process a `_divide` sentinel from a Map / dict update.

    Shape:
        {'_divide': {
            'mother': <key>,
            'daughters': {
                <daughter_key_1>: <override_1>,  # partial state
                <daughter_key_2>: <override_2>,
            },
        }}

    Or the simpler form (no overrides — pure type-driven split):
        {'_divide': {
            'mother': <key>,
            'daughters': [<daughter_key_1>, <daughter_key_2>],
        }}

    The handler is a TWO-PHASE merge:

    1. **Type-driven divide walk**: call `divide(value_schema, mother_state)`
       which walks the value's schema field-by-field and produces two
       baseline daughter states (bulk binomial-split, unique molecules
       split via the type-specific dividers, listeners shared, etc.).

    2. **Caller overrides layered on top**: each daughter's override
       dict is deep-merged onto the baseline. The caller typically
       provides fresh link declarations (address, config, wires, instance)
       so the daughters don't share the mother's process instances. Any
       fields the caller doesn't specify keep the type-driven default.

    The framework's realize() pass (triggered because `_divide` is
    treated as a structural change) then instantiates any new links
    in the merged daughter states and projects their wired state into
    the tree.

    Returns the updated state dict (mutated in place).
    """
    spec = update['_divide']
    mother = spec['mother']
    daughters = spec['daughters']

    if not isinstance(state, dict):
        raise ValueError(
            f'_divide at {path}: state is not a dict, got {type(state).__name__}')
    if mother not in state:
        raise ValueError(
            f'_divide at {path}: mother key {mother!r} not in state '
            f'(have keys {sorted(state.keys())})')

    # Normalize the daughters spec into a list of (key, override) tuples.
    # Accepted forms:
    #   - dict:           {daughter_key: override_dict}
    #   - list of strs:   ['00', '01']                         (no overrides)
    #   - list of pairs:  [('00', override), ('01', override)]
    #   - list of dicts:  [{'key': '00', ...}, {'key': '01', ...}]
    #     (v1 vivarium-style: 'key' identifies the daughter; any
    #     additional fields like 'processes'/'steps'/'flow'/'topology'/
    #     'initial_state' are vivarium engine concepts that don't apply to
    #     the composite engine and are ignored. The 'initial_state' field,
    #     if present, is treated as the daughter override so callers can
    #     still seed daughter-specific state.)
    if isinstance(daughters, dict):
        daughter_items = [(k, v) for k, v in daughters.items()]
    elif isinstance(daughters, list):
        daughter_items = []
        for item in daughters:
            if isinstance(item, str):
                daughter_items.append((item, {}))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                daughter_items.append(tuple(item))
            elif isinstance(item, dict) and 'key' in item:
                # v1 vivarium daughter dict — extract just the key. The
                # other v1 fields ('processes'/'steps'/'flow'/'topology'/
                # 'initial_state') are vivarium-engine concepts and don't
                # apply to the composite engine. The type-driven divide
                # walk + divide(Link) handles fresh instance construction.
                daughter_items.append((item['key'], {}))
            else:
                raise ValueError(
                    f'_divide at {path}: bad daughter entry {item!r}')
    else:
        raise ValueError(
            f'_divide at {path}: daughters must be dict or list, '
            f'got {type(daughters).__name__}')

    if len(daughter_items) != 2:
        raise ValueError(
            f'_divide at {path}: expected exactly 2 daughters, got '
            f'{len(daughter_items)}')

    # Phase 1: type-driven divide walk produces two baseline daughters.
    mother_state = state[mother]
    a, b = _divide_state(value_schema, mother_state,
                         context=mother_state, path=())

    # Phase 2: deep-merge each daughter's overrides onto the baseline.
    (key_a, override_a), (key_b, override_b) = daughter_items
    daughter_a = _deep_merge_into(a, override_a) if override_a else a
    daughter_b = _deep_merge_into(b, override_b) if override_b else b

    # Remove the mother and install daughters.
    del state[mother]
    state[key_a] = daughter_a
    state[key_b] = daughter_b

    # Emit Divided event so composite-layer consumers can update their
    # process_paths/step_paths indexes incrementally (no full rescan).
    emit(Divided(
        path=path,
        mother_key=mother,
        daughter_keys=[key_a, key_b],
        daughters_state={key_a: daughter_a, key_b: daughter_b},
        daughters_schema=value_schema))
    return state


@dispatch
def apply(schema: Map, state, update, path):
    if isinstance(state, list):
        state = {}
    if not isinstance(state, dict):
        state = {}
    merges = []

    if update is None:
        return state, merges

    # _divide sentinel: remove the mother key, install caller-provided
    # daughter states. The framework's realize() pass (triggered by
    # `_divide` being a structural sentinel) instantiates any new
    # links in the daughter states. Symmetric with _add/_remove.
    if isinstance(update, dict) and '_divide' in update:
        state = _handle_divide_sentinel(schema._value, state, update, path)
        rest = {k: v for k, v in update.items() if k != '_divide'}
        if not rest:
            return state, merges
        update = rest

    value_schema = schema._value

    # _remove: 'all' clears pre-existing state first, so adds and
    # value updates from sibling updates in the same batch survive.
    # Targeted _remove: [keys] still runs at the end of the apply
    # pass (after adds and value updates), preserving the existing
    # "remove wins over add for the same key" semantics.
    remove_directive = update.get('_remove')
    if remove_directive == 'all':
        for remove_key in list(state.keys()):
            del state[remove_key]
            emit(NodeRemoved(path=path, key=remove_key))

    if '_add' in update:
        add_update = update['_add']
        if isinstance(add_update, list):
            for add_key, add_value in update['_add']:
                state[add_key] = add_value
                emit(NodeAdded(path=path, key=add_key,
                               state=add_value, schema=value_schema))
        elif isinstance(add_update, dict):
            for add_key, add_value in update['_add'].items():
                state[add_key] = add_value
                emit(NodeAdded(path=path, key=add_key,
                               state=add_value, schema=value_schema))

    # Update-driven walk: iterate update keys (matching state) instead
    # of state keys, mutating in place. Skip _add/_remove sentinels
    # which were handled above.
    for key, update_value in update.items():
        if key in ('_add', '_remove'):
            continue
        if key not in state:
            continue
        new_value, submerges = apply(
            value_schema,
            state[key],
            update_value,
            path+(key,))
        if new_value is not state[key]:
            state[key] = new_value
        if submerges:
            merges += submerges

    if remove_directive is not None and remove_directive != 'all':
        for remove_key in remove_directive:
            if remove_key in state:
                del state[remove_key]
                emit(NodeRemoved(path=path, key=remove_key))

    return state, merges


@dispatch
def apply(schema: Tree, state, update, path):
    if update is None:
        return state, []
    if check(schema._leaf, state):
        return apply(schema._leaf, state, update, path)
    if not isinstance(state, dict):
        return update, []
    if not isinstance(update, dict):
        return update, []

    # Lazy copy: keep ``result`` aliased to ``state`` until we know we
    # need to mutate, then allocate. Tree apply runs many times per
    # tick and most calls produce no change; the eager ``dict(state)``
    # was the dominant fixed cost.
    merges = []
    result = state
    state_dict = state  # alias used while result still points at the original

    add_update = update.get('_add')
    if add_update:
        result = dict(state_dict)
        if isinstance(add_update, list):
            for add_key, add_value in add_update:
                result[add_key] = add_value
                emit(NodeAdded(path=path, key=add_key,
                               state=add_value, schema=schema))
        elif isinstance(add_update, dict):
            result.update(add_update)
            for add_key, add_value in add_update.items():
                emit(NodeAdded(path=path, key=add_key,
                               state=add_value, schema=schema))

    for key, update_value in update.items():
        if key == '_add' or key == '_remove':
            continue
        if key in state_dict:
            child = state_dict[key]
            new_value, submerges = apply(
                schema, child, update_value, path + (key,))
            if new_value is not child:
                if result is state_dict:
                    result = dict(state_dict)
                result[key] = new_value
            if submerges:
                merges += submerges
        else:
            if result is state_dict:
                result = dict(state_dict)
            result[key] = update_value

    remove_update = update.get('_remove')
    if remove_update:
        if result is state_dict:
            result = dict(state_dict)
        for remove_key in remove_update:
            if remove_key in result:
                del result[remove_key]
                emit(NodeRemoved(path=path, key=remove_key))

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
    if update is None:
        return state, []
    if state is None:
        return update, []

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
        # Sparse cell-projection update: {i: {j: delta, ...}, ...}.
        # Wires that target a single cell of an Array (e.g. per-cell
        # spatial processes wired to ``['fields','glucose',i,j]``)
        # produce projection updates of this shape; each integer key
        # is an axis index, leaves are additive scalars.
        if isinstance(state, np.ndarray) and all(
                isinstance(k, (int, np.integer)) for k in update.keys()):
            for idx, sub in update.items():
                if isinstance(sub, dict):
                    apply(schema, state[idx], sub, path + (idx,))
                elif sub is not None:
                    state[idx] += sub
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

    # _divide sentinel: type-driven walk + caller overrides. See
    # _handle_divide_sentinel docstring for the contract.
    if isinstance(update, dict) and '_divide' in update:
        spec = update['_divide']
        mother = spec['mother']
        if mother in schema:
            # Mutate the schema dict in place: pop the mother key,
            # reuse its schema for each daughter (daughters are
            # structurally homogeneous instances of the same type).
            value_schema = schema.pop(mother)
            state = _handle_divide_sentinel(value_schema, state, update, path)
            # Extract daughter keys to register in the schema dict
            daughters = spec['daughters']
            if isinstance(daughters, dict):
                daughter_keys = list(daughters.keys())
            else:
                daughter_keys = []
                for item in daughters:
                    if isinstance(item, str):
                        daughter_keys.append(item)
                    elif isinstance(item, (list, tuple)):
                        daughter_keys.append(item[0])
                    elif isinstance(item, dict) and 'key' in item:
                        daughter_keys.append(item['key'])
            for d in daughter_keys:
                schema[d] = value_schema
        update = {k: v for k, v in update.items() if k != '_divide'}
        if not update:
            return state, []

    merges = []

    # Update-driven walk: iterate update keys (typically 5-20) instead
    # of schema keys (typically 50-100). State is mutated in place — we
    # only write back when the apply produces a NEW value object. This
    # mirrors v1 vivarium's Store.apply_update which descends the
    # state tree directly via update keys.
    for key, update_value in update.items():
        if not is_schema_field(schema, key):
            continue
        if key not in schema:
            continue
        if update_value is None:
            continue

        subschema = schema[key]
        sub_state = state.get(key)
        new_value, submerges = apply(
            subschema,
            sub_state,
            update_value,
            path+(key,))
        if new_value is not sub_state:
            state[key] = new_value
        if submerges:
            merges += submerges

    return state, merges


@dispatch
def apply(schema: Float, state, update, path):
    """Additive: float updates are deltas."""
    if update is None or update == 0:
        return state, []
    if state is None:
        return update, []
    return state + update, []


@dispatch
def apply(schema: Integer, state, update, path):
    """Additive: integer updates are deltas. Zero-delta updates fast-path
    to ``return state``; on the vEcoli hot path 80%+ of integer apply
    calls are zero-deltas (per-process listener fields with no change),
    so this avoids both the addition and the wasteful new-int allocation
    that would force the parent dict-apply to write back an identical
    value."""
    if update is None or update == 0:
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
