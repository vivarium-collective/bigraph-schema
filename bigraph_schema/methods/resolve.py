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
    Link,
)


from bigraph_schema.methods.default import default
from bigraph_schema.methods.merge import merge, merge_update


def resolve_subclass(subclass, superclass):
    result = {}
    for key in subclass.__dataclass_fields__:    
        if key == '_default':
            result[key] = subclass._default or superclass._default
        else:
            subattr = getattr(subclass, key)
            if hasattr(superclass, key): # and not key.startswith('_'):
                superattr = getattr(superclass, key)
                if isinstance(superattr, (Node, dict)):
                    try:
                        outcome = resolve(subattr, superattr)
                    except Exception as e:
                        raise Exception(f'\ncannot resolve subtypes for attribute \'{key}\':\n{subattr}\n{superattr}\n\n  due to\n{e}')

                    result[key] = outcome
                else:
                    result[key] = subattr
            else:
                result[key] = subattr

    resolved = type(subclass)(**result)
    return resolved


@dispatch
def resolve(current: Empty, update: Node):
    return update

@dispatch
def resolve(current: Node, update: Empty):
    return current

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

    elif isinstance(update, String):
        default_value = update_type._default
        if default_value:
            return replace(
                current,
                **{'_default': default_value})
        else:
            return current

    else:
        raise Exception(f'\ncannot resolve types:\n{current}\n{update}\n')

@dispatch
def resolve(current: Map, update: dict):
    result = current._value
    try:
        for key, value in update.items():
            result = resolve(result, value)
        resolved = replace(current, _value=result)

    except:
        # upgrade from map to struct schema
        map_default = default(current)
        resolved = {
            key: current._value
            for key in map_default}
        resolved.update(update)

    schema = merge_update(resolved, current, update)
    return schema

@dispatch
def resolve(current: dict, update: Map):
    result = update._value

    try:
        for key, value in current.items():
            result = resolve(result, value)
        resolved = replace(update, _value=result)

    except:
        # upgrade from map to struct schema
        map_default = default(update)
        resolved = {
            key: update._value
            for key in map_default}
        current.update(resolved)

    schema = merge_update(resolved, current, update)
    return schema

@dispatch
def resolve(current: Tree, update: Map):
    value = current._leaf
    leaf = update._value
    update_leaf = resolve(leaf, value)
    result = copy.copy(current)
    resolved = replace(result, _leaf=update_leaf)

    schema = merge_update(resolved, current, update)
    return schema

@dispatch
def resolve(current: Tree, update: Tree):
    current_leaf = current._leaf
    update_leaf = update._leaf
    resolved = resolve(current_leaf, update_leaf)
    result = replace(current, _leaf=resolved)

    schema = merge_update(result, current, update)
    return schema

@dispatch
def resolve(current: Tree, update: Node):
    leaf = current._leaf
    try:
        resolved = resolve(leaf, update)
    except:
        raise Exception(f'update schema is neither a tree or a leaf:\n{current}\n{update}')

    replace(current, _leaf=resolved)
    return current

@dispatch
def resolve(current: Tree, update: dict):
    result = copy.copy(current)
    leaf = current._leaf
    for key, value in update.items():
        try:
            leaf = resolve(leaf, value)
        except:
            result = resolve(result, value)
    resolved = replace(result, _leaf=leaf)

    schema = merge_update(resolved, current, update)
    return schema

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
def resolve(current: Link, update: dict):
    schema = current
    for key in ['_inputs', '_outputs']:
        if key in update:
            subupdate = update[key]
            attr = getattr(schema, key)
            subresolve = resolve(attr, subupdate)
            schema = replace(schema, **{key: subresolve})
        # else:
        #     schema = replace(schema, **{key: subupdate})

    return schema

@dispatch
def resolve(current: dict, update: Link):
    schema = update
    for key in ['_inputs', '_outputs']:
        if key in current:
            subupdate = current[key]
            attr = getattr(schema, key)
            subresolve = resolve(attr, subupdate)
            schema = replace(schema, **{key: subresolve})
        # else:
        #     schema = replace(schema, **{key: subupdate})

    return schema

@dispatch
def resolve(current: Node, update: dict):
    fields = set(current.__dataclass_fields__)
    keys = set(update.keys())

    if len(keys.difference(fields)) > 0:
        return update
    else:
        return current

@dispatch
def resolve(current: dict, update: Node):
    fields = set(update.__dataclass_fields__)
    keys = set(current.keys())

    if len(keys.difference(fields)) > 0:
        return update
    else:
        return current

# @dispatch
# def resolve(current: String, update: Node):
#     if current._default:
#         update._default = current._default
#     return update

# @dispatch
# def resolve(current: String, update: Wrap):
#     return resolve(current, update._value)

# @dispatch
# def resolve(current: String, update: String):
#     if update._default or not current._default:
#         return update
#     else:
#         return current

# @dispatch
# def resolve(current: Node, update: String):
#     # import ipdb; ipdb.set_trace()
#     if update._default:
#         current = replace(current, **{'_default': update._default})
#     return current


# @dispatch
# def resolve(current: dict, update: Node):
#     fields = set(update.__dataclass_fields__)
#     keys = set(current.keys())

#     for key in keys.intersect(fields):
#         getattr(update, key)
    
    

@dispatch
def resolve(current: list, update: list):
    return tuple(update)


@dispatch
def resolve(current, update):
    
    if current is None:
        return update
    elif update is None:
        return current
    else:
        import ipdb; ipdb.set_trace()
        raise Exception(f'\ncannot resolve types, not schemas:\n{current}\n{update}\n')


