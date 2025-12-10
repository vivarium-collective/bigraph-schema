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

from bigraph_schema.methods.check import check
from bigraph_schema.methods.default import default


@dispatch
def merge(schema: Empty, current, update):
    return None


@dispatch
def merge(schema: Maybe, current, update):
    if update is None:
        return current
    elif current is None:
        return update
    else:
        return merge(schema._value, current, update)


@dispatch
def merge(schema: Wrap, current, update):
    return merge(schema._value, current, update)


@dispatch
def merge(schema: Union, current, update):
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
        return merge(current_option, current, update)
    else:
        return update


@dispatch
def merge(schema: Tuple, current, update):
    return tuple([
        merge(schema_value, current_value, update_value)
        for schema_value, current_value, update_value in zip(schema._values, current, update)])


@dispatch
def merge(schema: List, current, update):
    if current is None:
        return update
    if update is None:
        return current
    else:
        return current + update


@dispatch
def merge(schema: Map, current, update):
    result = {}
    if current is None:
        return update
    elif update is None:
        return current
    else:
        for key in current.keys() | update.keys():
            if key in update:
                if key in current:
                    if not key.startswith('_'):
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
def merge(schema: Tree, current, update):
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
def merge(schema: Atom, current, update):
    result = None
    if update and update is not None:
        result = update
    elif current and current is not None:
        result = current
    else:
        result = default(schema)

    return result


@dispatch
def merge(schema: Wires, current, update):
    return update


@dispatch
def merge(schema: Link, current, update):
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
def merge(schema: Node, current, update):
    if isinstance(current, dict) and isinstance(update, dict):
        down = {}
        for key in schema.__dataclass_fields__:
            down[key] = getattr(schema, key)
        return merge(down, current, update)
    else:
        result = merge(
            Atom(),
            current,
            update)

        if result is None:
            result = default(schema)

        return result


@dispatch
def merge(schema, current, update):
    return update


def is_empty(value):
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
def merge(schema: dict, current, update):
    result = {}
    if is_empty(current):
        return update
    if is_empty(update):
        return current

    if not isinstance(update, dict):
        return current

    if isinstance(current, np.ndarray):
        import ipdb; ipdb.set_trace()

    for key in schema.keys() | current.keys() | update.keys():
        if key in schema:
            result[key] = merge(
                schema[key],
                current.get(key),
                update.get(key))
        elif key in update:
            result[key] = update[key]
        else:
            result[key] = current[key]

        if key in schema and schema[key] and result[key] is None:
            if key.startswith('_'):
                result[key] = schema[key]
            else:
                result[key] = default(
                    schema[key])

        if result[key] is None:
            del result[key]

    return result

def merge_update(schema, current, update):
    current_state = default(current)
    update_state = default(update)
    state = current_state

    if update_state:
        if current_state:
            state = merge(schema, current_state, update_state)
        else:
            state = update_state

    if isinstance(schema, Node):
        schema = replace(schema, _default=state)
    elif isinstance(schema, dict):
        schema['_default'] = state
    else:
        raise Exception(f'do not recognize schema: {schema}')
    return schema


