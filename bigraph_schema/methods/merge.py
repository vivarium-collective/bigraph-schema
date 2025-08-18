from plum import dispatch
import numpy as np

from bigraph_schema.schema import (
    Node,
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
    Dtype,
    Array,
    Key,
    Path,
    Wires,
    Schema,
    Edge,
)

from bigraph_schema.methods import check


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
    return current + update


@dispatch
def merge(schema: Map, current, update):
    result = {}
    for key in current.keys() | update.keys():
        if key in update:
            if key in current:
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
def merge(schema: Node, current, update):
    return update
