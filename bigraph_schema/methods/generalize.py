import copy
from plum import dispatch
import numpy as np

from dataclasses import replace, dataclass

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
    Array,
    Key,
    Path,
    Wires,
    Schema,
    Edge,
)


from bigraph_schema.methods.default import default
from bigraph_schema.methods.merge import merge, merge_update


def generalize_subclass(subclass, superclass):
    result = {}
    for key in superclass.__dataclass_fields__:    
        if key == '_default':
            result[key] = superclass._default or subclass._default
        else:
            subattr = getattr(subclass, key)
            if not key.startswith('_'):
                superattr = getattr(superclass, key)
                try:
                    outcome = generalize(subattr, superattr)
                except Exception as e:
                    raise Exception(f'\ncannot generalize subtypes for attribute \'{key}\':\n{subattr}\n{superattr}\n\n  due to\n{e}')
                result[key] = outcome
            else:
                result[key] = subattr
    generalized = type(superclass)(**result)
    return generalized


@dispatch
def generalize(current: Empty, update: Node):
    return update

@dispatch
def generalize(current: Node, update: Empty):
    return current

@dispatch
def generalize(current: Wrap, update: Wrap):
    value = generalize(current._value, update._value)
    return value
    # if type(current) == type(update):
    #     return type(current)(_value=value)
    # elif issubclass(current_type, update_type):
    #     return generalize_subclass(current, update)
    # elif issubclass(update_type, current_type):
    #     return generalize_subclass(update, current)
    # else:
    #     return update

@dispatch
def generalize(current: Wrap, update: Node):
    value = generalize(current._value, update)
    return value

@dispatch
def generalize(current: Node, update: Wrap):
    value = generalize(current, update._value)
    return value

@dispatch
def generalize(current: Node, update: Node):
    current_type = type(current)
    update_type = type(update)
    if current_type == update_type or issubclass(current_type, update_type):
        return generalize_subclass(current, update)
    elif issubclass(update_type, current_type):
        return generalize_subclass(update, current)
    else:
        raise Exception(f'\ncannot generalize types:\n{current}\n{update}\n')

@dispatch
def generalize(current: Map, update: dict):
    result = current._value
    try:
        for key, value in update.items():
            result = generalize(result, value)
        generalized = replace(current, _value=result)

    except:
        # upgrade from map to struct schema
        map_default = default(current)
        generalized = {
            key: current._value
            for key in map_default}
        generalized.update(update)

    schema = merge_update(generalized, current, update)
    return schema

@dispatch
def generalize(current: dict, update: Map):
    result = update._value
    for key, value in current.items():
        result = generalize(result, value)
    result = replace(result, _default=update._value._default)
    generalized = replace(update, _value=result)

    schema = merge_update(generalized, current, update)
    return schema

@dispatch
def generalize(current: Tree, update: Map):
    value = current._leaf
    leaf = update._value
    update_leaf = generalize(leaf, value)
    result = copy.copy(current)
    generalized = replace(result, _leaf=update_leaf)

    schema = merge_update(generalized, current, update)
    return schema

@dispatch
def generalize(current: Tree, update: Tree):
    current_leaf = current._leaf
    update_leaf = update._leaf
    generalized = generalize(current_leaf, update_leaf)
    result = replace(current, _leaf=generalized)

    schema = merge_update(result, current, update)
    return schema

@dispatch
def generalize(current: Tree, update: Node):
    leaf = current._leaf
    try:
        generalized = generalize(leaf, update)
    except:
        raise(f'update schema is neither a tree or a leaf:\n{current}\n{update}')

    replace(current, _leaf=generalized)
    return current

@dispatch
def generalize(current: Tree, update: dict):
    result = copy.copy(current)
    leaf = current._leaf
    for key, value in update.items():
        try:
            leaf = generalize(leaf, value)
        except:
            result = generalize(result, value)
    generalized = replace(result, _leaf=leaf)

    schema = merge_update(generalized, current, update)
    return schema

@dispatch
def generalize(current: dict, update: dict):
    result = {}
    all_keys = set(current.keys()).union(set(update.keys()))
    for key in all_keys:
        try:
            value = generalize(
                current.get(key),
                update.get(key))
        except Exception as e:
            raise Exception(f'\ncannot generalize subtypes for key \'{key}\':\n{current}\n{update}\n\n  due to\n{e}')

        result[key] = value
    return result

@dispatch
def generalize(current: Node, update: dict):
    fields = set(current.__dataclass_fields__)
    keys = set(update.keys())

    if len(keys.difference(fields)) > 0:
        return update
    else:
        return current

# @dispatch
# def generalize(current: dict, update: Node):
#     fields = set(update.__dataclass_fields__)
#     keys = set(current.keys())

#     for key in keys.intersect(fields):
#         getattr(update, key)
    
    

@dispatch
def generalize(current: list, update: list):
    return tuple(update)


@dispatch
def generalize(current, update):
    
    if current is None:
        return update
    elif update is None:
        return current
    else:
        import ipdb; ipdb.set_trace()
        raise Exception(f'\ncannot generalize types, not schemas:\n{current}\n{update}\n')


