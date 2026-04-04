from plum import dispatch
import numpy as np

from dataclasses import replace, dataclass

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
    Key,
    Path,
    Wires,
    Schema,
    Link,
    is_schema_field,
)

from bigraph_schema.methods.check import check
from bigraph_schema.methods.default import default


@dispatch
def merge(schema: Empty, current, update, path=()):
    return None


@dispatch
def merge(schema: Maybe, current, update, path=()):
    if update is None:
        return current
    elif current is None:
        return update
    else:
        return merge(
            schema._value,
            current,
            update,
            path=path)


@dispatch
def merge(schema: Wrap, current, update, path=()):
    return merge(schema._value, current, update, path=path)

@dispatch
def merge(schema: Overwrite, current, update, path=()):
    if update is None:
        return current
    return update

@dispatch
def merge(schema: Const, current, update, path=()):
    return current

@dispatch
def merge(schema: Union, current, update, path=()):
    current_option = None
    update_option = None
    for option in schema._options:
        if not current_option and check(option, current):
            current_option = option
        if not update_option and check(option, update):
            update_option = option
        if current_option and update_option:
            break

    if current_option == update_option:
        return merge(current_option, current, update, path=path)
    else:
        return update


@dispatch
def merge(schema: Tuple, current, update, path=()):
    if path:
        head = path[0]
        if head >= len(schema._values):
            raise Exception(f'trying to merge at path {path} but index {head} is out of range:\n\n{schema}\n\n{current}\n\n{update}\n\n')

        subschema = schema._values[head]
        field = list(current)
        field[head] = merge(subschema, current[head], update, path[1:])
        return tuple(field)

    else:
        if update is None:
            return current
        if current is None:
            return update

        return tuple([
            merge(schema_value, current_value, update_value)
            for schema_value, current_value, update_value, index in zip(schema._values, current, update, range(len(schema._values)))])


@dispatch
def merge(schema: List, current, update, path=()):
    if path:
        head = path[0]
        subschema = schema._element
        current[head] = merge(subschema, current[head], update, path[1:])
        return current

    elif current is None or current == []:
        return update

    elif update is None or update == []:
        return current

    elif isinstance(update, np.ndarray):
        return update

    elif isinstance(current, np.ndarray):
        return current

    else:
        return current + update


@dispatch
def merge(schema: Set, current, update, path=()):
    if current is None:
        return update
    if update is None:
        return current
    return current | update

@dispatch
def merge(schema: Map, current, update, path=()):
    if path:
        head = path[0]

        if head == '*':
            current = current or {}
            for key, value in update.items():
                current[key] = merge(
                    schema._value,
                    current.get(key),
                    value,
                    path[1:])

            return current

        else:
            subschema = schema._value
            current[head] = merge(subschema, current[head], update, path[1:])
            return current

    result = {}
    if current is None:
        return update

    elif update is None:
        return current

    else:
        for key in current.keys() | update.keys():
            if key in update:
                if key in current:
                    if not isinstance(key, str) or not key.startswith('_'):
                        result[key] = merge(
                            schema._value,
                            current[key],
                            update[key])
                else:
                    result[key] = update[key]
            else:
                result[key] = current[key]

        return result


@dispatch
def merge(schema: Tree, current, update, path=()):
    if path:
        head = path[0]
        current[head] = merge(schema, current[head], update, path[1:])
        return current

    current_leaf = check(schema._leaf, current)
    update_leaf = check(schema._leaf, update)

    if current_leaf and update_leaf:
        return merge(schema._leaf, current, update)
    elif not current_leaf and not update_leaf:
        result = {}
        for key in current.keys() | update.keys():
            if key in update:
                if key in current:
                    result[key] = merge(
                        schema,
                        current[key],
                        update[key])
                else:
                    result[key] = update[key]
            else:
                result[key] = current[key]
        return result
    else:
        return update


@dispatch
def merge(schema: Array, current, update, path=()):
    if path:
        if current is None:
            current = default(schema)

        head = path[0]
        if head == '*':
            for index, row, subupdate in zip(range(current.shape[0]), current, update):
                subschema = replace(schema, **{'_shape': schema._shape[1:]})
                current[index] = merge(subschema, row, subupdate, path=path[1:])

        else:
            row = current[head]
            subschema = replace(schema, **{'_shape': schema._shape[1:]})
            subvalue = merge(subschema, row, update, path=path[1:])
            current[head] = subvalue

        return current

    # TODO: more sophisticated merge for arrays?
    if update is None:
        if current is None:
            return default(schema)
        else:
            return current
    else:
        return update

@dispatch
def merge(schema: Frame, current, update, path=()):
    if update.empty:
        return current
    else:
        return update

@dispatch
def merge(schema: Atom, current, update, path=()):
    from bigraph_schema.methods.is_empty import is_empty as _is_empty
    if not _is_empty(schema, update):
        return update
    elif not _is_empty(schema, current):
        return current
    elif update is not None:
        return update
    elif current is not None:
        return current


@dispatch
def merge(schema: Wires, current, update, path=()):
    return update


@dispatch
def merge(schema: Link, current, update, path=()):
    if not current:
        return update
    if not update:
        return current

    result = {}
    for key in schema.__dataclass_fields__:
        if not key.startswith('_'):
            down = getattr(schema, key)
            result[key] = merge(
                down,
                current.get(key),
                update.get(key))

    return result


@dispatch
def merge(schema: Node, current, update, path=()):
    if path:
        head = path[0]
        subschema = None
        if hasattr(schema, head):
            subschema = getattr(schema, head)

        if isinstance(current, dict):
            substate = current.get(head)
            current[head] = merge(subschema, substate, update, path[1:])
            return current
        else:
            substate = None
            if hasattr(current, head):
                substate = getattr(current, head)
            submerge = merge(subschema, substate, update, path[1:])
            setattr(current, head, submerge)
            return current

    elif isinstance(current, dict) and isinstance(update, dict):
        down = {}
        for key in schema.__dataclass_fields__:
            if is_schema_field(schema, key):
                down[key] = getattr(schema, key)
        return merge(down, current, update)

    else:
        # Non-empty wins over empty. If both non-empty, update wins.
        from bigraph_schema.methods.is_empty import is_empty as _is_empty

        update_empty = _is_empty(schema, update)
        current_empty = _is_empty(schema, current)

        if not update_empty:
            return update
        elif not current_empty:
            return current
        else:
            return None


@dispatch
def merge(schema, current, update, path=()):
    if path:
        head = path[0]
        current = current or {}
        if isinstance(current, dict):
            current[head] = merge(schema, current.get(head), update, path[1:])
            return current
    return update


def _value_is_empty(value):
    """Simple emptiness check for dict merge (no schema context)."""
    if isinstance(value, np.ndarray):
        return False
    else:
        return not value


def tuplify_dict(d):
    if isinstance(d, dict):
        tulip = []
        for key, value in d.items():
            tulip.append((
                key,
                tuplify_dict(value)))
    else:
        return d


@dispatch
def merge(schema: dict, current, update, path=()):
    if path:
        head = path[0]
        if head == '*':
            raise Exception('star path not supported in dict merge')
        else:
            current = current or {}
            subschema = schema.get(head)
            substate = current.get(head)
            current[head] = merge(subschema, substate, update, path[1:])
            return current

    result = {}
    if _value_is_empty(current):
        return update
    if _value_is_empty(update):
        return current

    if isinstance(update, np.ndarray):
        return update
    elif not isinstance(update, dict):
        return current

    for key in schema.keys() | current.keys() | update.keys():
        if not is_schema_field(schema, key):
            continue
        if key in schema:
            result[key] = merge(
                schema[key],
                current.get(key),
                update.get(key))
        elif key in current and key in update:
            # Both have this key but no schema — recursively merge
            result[key] = merge(
                Node(),
                current[key],
                update[key])
        elif key in update:
            result[key] = update[key]
        elif key in current:
            result[key] = current[key]
        else:
            raise Exception('logically impossible')

        if result[key] is None:
            del result[key]

    return result

def merge_update(schema, current, update, path=()):
    if path:
        raise Exception('merge_update with path not implemented')

    # Pick the more informative default — prefer update over current.
    # Avoid merging defaults via merge() which can corrupt schemas
    # by walking state data (bare lists) through schema methods.
    current_default = current._default if isinstance(current, Node) else current.get('_default')
    update_default = update._default if isinstance(update, Node) else update.get('_default') if isinstance(update, dict) else None

    state = update_default if update_default is not None else current_default

    if isinstance(schema, Node):
        schema = replace(schema, _default=state)
    elif isinstance(schema, dict):
        schema['_default'] = state
    else:
        raise Exception(f'do not recognize schema: {schema}')
    return schema


