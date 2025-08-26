from plum import dispatch
import numpy as np

from bigraph_schema.schema import (
    Node,
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
    Dtype,
    Array,
    Key,
    Path,
    Wires,
    Schema,
    Edge,
)


def resolve_subclass(subclass, superclass):
    result = {}
    for key in subclass.__dataclass_fields__:    
        if key == '_default':
            result[key] = subclass._default or superclass._default
        else:
            subattr = getattr(subclass, key)
            if hasattr(superclass, key):
                superattr = getattr(superclass, key)
                try:
                    outcome = resolve(subattr, superattr)
                except Exception as e:
                    raise Exception(f'\ncannot resolve subtypes for attribute \'{key}\':\n{subattr}\n{superattr}\n\n  due to\n{e}')
                result[key] = outcome
            else:
                result[key] = subattr
    resolved = type(subclass)(**result)
    return resolved


@dispatch
def resolve(current: Wrap, update: Wrap):
    if type(current) == type(update):
        value = resolve(current._value, update._value)
        return type(current)(_value=value)


@dispatch
def resolve(current: Wrap, update: Node):
    value = resolve(current._value, update)
    return type(current)(_value=value)


@dispatch
def resolve(current: Node, update: Wrap):
    value = resolve(current, update._value)
    return type(update)(_value=value)


@dispatch
def resolve(current: Node, update: Node):
    current_type = type(current)
    update_type = type(update)
    if current_type == update_type or issubclass(current_type, update_type):
        return resolve_subclass(current, update)
    elif issubclass(update_type, current_type):
        return resolve_subclass(update, current)
    else:
        # raise Exception('cannot resolve types', {
        #     'current': current,
        #     'update': update})
        raise Exception(f'\ncannot resolve types:\n{current}\n{update}\n')


@dispatch
def resolve(current: dict, update: dict):
    result = {}
    all_keys = set(current.keys()).union(set(update.keys()))
    for key in all_keys:
        try:
            value = resolve(
                current.get(key),
                update.get(key))
        except Exception as e:
            raise Exception(f'\ncannot resolve subtypes for key \'{key}\':\n{current}\n{update}\n\n  due to\n{e}')

        result[key] = value
    return result


@dispatch
def resolve(current, update):
    if current is None:
        return update
    elif update is None:
        return current
    else:
        # raise Exception('cannot resolve types, not schemas', {
        #     'current': current,
        #     'update': update})
        raise Exception(f'\ncannot resolve types, not schemas:\n{current}\n{update}\n')
