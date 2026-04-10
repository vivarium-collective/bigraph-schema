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
    dtype_schema,
    schema_dtype,
    is_schema_field,
)


from bigraph_schema.methods.default import default
from bigraph_schema.methods.merge import merge, merge_update
from bigraph_schema.methods.is_empty import is_empty as _is_empty


def read_link_path(schema):
    if hasattr(schema, '_link_path'):
        return schema._link_path
    elif isinstance(schema, dict):
        if "_link_path" in schema:
            return schema["_link_path"]
        return []
    else:
        return []


def resolve_subclass(subclass, superclass):
    result = {}
    for key in subclass.__dataclass_fields__:
        if not is_schema_field(subclass, key):
            if key == '_default':
                result[key] = subclass._default or superclass._default
            else:
                result[key] = getattr(subclass, key)
            continue

        subattr = getattr(subclass, key)
        if hasattr(superclass, key):
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


def resolve_empty(empty, update, path=None):
    if path:
        head = path[0]
        result = {}
        result[head] = resolve(empty, update, path[1:])
        return result
    else:
        return update

@dispatch
def resolve(current: Empty, update: Empty, path=None):
    return resolve_empty(current, update, path=path)

@dispatch
def resolve(current: Empty, update: Node, path=None):
    return resolve_empty(current, update, path=path)

@dispatch
def resolve(current: Node, update: Empty, path=None):
    return resolve_empty(update, current, path=path)

# Disambiguate (SubType, Empty) for all subtypes that have (SubType, Node) dispatches
@dispatch
def resolve(current: Wrap, update: Empty, path=None):
    return current

@dispatch
def resolve(current: Array, update: Empty, path=None):
    return current

@dispatch
def resolve(current: Array, update: Wrap, path=None):
    value = resolve(current, update._value, path=path)
    result = type(update)(_value=value)
    if result._default is None and current._default is not None:
        result._default = current._default
    return result

@dispatch
def resolve(current: Tree, update: Empty, path=None):
    return current

@dispatch
def resolve(current: Tree, update: Wrap, path=None):
    value = resolve(current, update._value, path=path)
    result = type(update)(_value=value)
    if result._default is None and current._default is not None:
        result._default = current._default
    return result

@dispatch
def resolve(current: String, update: Empty, path=None):
    return current

@dispatch
def resolve(current: dict, update: Empty, path=None):
    return current

@dispatch
def resolve(current: dict, update: Wrap, path=None):
    if path:
        head = path[0]
        down_resolve = resolve(
            current.get(head, {}),
            update,
            path[1:])
        current[head] = down_resolve
        return current
    # dict schemas don't have _default, just wrap the value
    value = resolve(current, update._value, path=path)
    return type(update)(_value=value)

@dispatch
def resolve(current: Wrap, update: Wrap, path=None):
    if type(current) == type(update):
        value = resolve(current._value, update._value, path=path)
        schema = type(current)(_value=value)

        if update._default is not None:
            schema._default = update._default
        else:
            schema._default = current._default

        return schema
    else:
        # TODO: resolve wrappings somehow?
        raise Exception(f'cannot resolve two different wrappings {current} {update}')


@dispatch
def resolve(current: Boolean, update: Boolean, path=None):
    if current._default:
        return current
    else:
        return update

@dispatch
def resolve(current: Wrap, update: Node, path=None):
    value = resolve(current._value, update, path=path)
    return type(current)(_value=value)

@dispatch
def resolve(current: Integer, update: Float, path=None):
    if _is_empty(update, update._default):
        if _is_empty(current, current._default):
            return update
        else:
            return replace(update, **{'_default': current._default})
    else:
        return update

@dispatch
def resolve(current: Float, update: Integer, path=None):
    if _is_empty(update, update._default):
        return current
    elif _is_empty(current, current._default):
        return replace(current, **{'_default': update._default})
    else:
        return current

@dispatch
def resolve(current: Node, update: Wrap, path=None):
    value = resolve(current, update._value, path=path)
    result = type(update)(_value=value)
    # Preserve the more informative default
    if result._default is None and hasattr(current, '_default') and current._default is not None:
        result._default = current._default
    return result

@dispatch
def resolve(current: Node, update: Node, path=None):
    if path:
        head = path[0]
        if current == Node():
            current = {
                head: resolve({}, update, path[1:])}
            return current
        else:
            down_current = None
            if hasattr(current, head):
                down_current = getattr(current, head)
            down_resolve = resolve(down_current, update, path[1:])
            setattr(current, head, down_resolve)
            return current

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
        raise Exception(f'\ncannot resolve types:\n{read_link_path(current)}: {current}\n{read_link_path(update)}: {update}\n')

def resolve_map(current: Map, update, path=None):
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
def resolve(current: Map, update: Map, path=None):
    if path:
        head = path[0]

        if head == '*':
            down_resolve = resolve(
                current._value,
                update._value,
                path[1:])

        else:
            down_resolve = resolve(
                current._value,
                update,
                path[1:])

        return replace(current, **{'_value': down_resolve})

    else:
        return resolve_map(current, update, path=path)

@dispatch
def resolve(current: Map, update: Empty, path=None):
    return current

@dispatch
def resolve(current: Map, update: Wrap, path=None):
    value = resolve(current, update._value, path=path)
    result = type(update)(_value=value)
    if result._default is None and current._default is not None:
        result._default = current._default
    return result

@dispatch
def resolve(current: Map, update: Node, path=None):
    if path:
        head = path[0]

        if head == '*':
            for key in update.__dataclass_fields__:
                value = current._value
                if not key.startswith('_'):
                    value = resolve(
                        value,
                        getattr(update, key),
                        path[1:])

            return replace(current, **{'_value': value})

        else:
            down_resolve = resolve(
                current._value,
                update,
                path[1:])

        return replace(current, **{'_value': down_resolve})

    else:
        return resolve_map(current, update, path=path)


@dispatch
def resolve(current: Map, update: dict, path=None):
    if path:
        head = path[0]
        down_resolve = resolve(current._value, update, path[1:])
        return replace(current, **{'_value': down_resolve})

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
        for key, value in update.items():
            if is_schema_field(update, key):
                resolved[key] = value

    schema = merge_update(resolved, current, update)
    return schema

@dispatch
def resolve(current: dict, update: Map, path=None):
    if path:
        head = path[0]
        if head == '*':
            if current:
                for key, subcurrent in current.items():
                    current[key] = resolve(
                        subcurrent,
                        update._value,
                        path[1:])
                return current

            else:
                subvalue = resolve(
                    current,
                    update._value,
                    path[1:])

                return replace(
                    update,
                    **{'_value': subvalue})

        else:
            down_resolve = resolve(
                current.get(head, {}),
                update,
                path[1:])
            current[head] = down_resolve
            return current

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
        for key, value in current.items():
            if is_schema_field(current, key):
                resolved[key] = value

    schema = merge_update(resolved, current, update)
    return schema


def tree_path(current, update, path):
    head = path[0]
    down_resolve = resolve(current, update, path[1:])
    if isinstance(down_resolve, Tree):
        return down_resolve
    else:
        return replace(current, **{'_leaf': down_resolve})    

@dispatch
def resolve(current: Tree, update: Map, path=None):
    if path:
        return tree_path(current, update, path)

    value = current._leaf
    leaf = update._value
    update_leaf = resolve(leaf, value)
    result = copy.copy(current)
    resolved = replace(result, _leaf=update_leaf)

    schema = merge_update(resolved, current, update)
    return schema

@dispatch
def resolve(current: Tree, update: Tree, path=None):
    if path:
        return tree_path(current, update, path)

    current_leaf = current._leaf
    update_leaf = update._leaf
    resolved = resolve(current_leaf, update_leaf)
    result = replace(current, _leaf=resolved)

    schema = merge_update(result, current, update)
    return schema

@dispatch
def resolve(current: Tree, update: Node, path=None):
    if path:
        return tree_path(current, update, path)

    leaf = current._leaf
    try:
        resolved = resolve(leaf, update)
    except:
        raise Exception(f'update schema is neither a tree or a leaf:\n{current}\n{update}')

    replace(current, _leaf=resolved)
    return current

@dispatch
def resolve(current: Tree, update: dict, path=None):
    if path:
        return tree_path(current, update, path)

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
def resolve(current: dict, update: dict, path=None):
    if path:
        head = path[0]
        down_resolve = resolve(
            current.get(head, {}),
            update,
            path[1:])
        current[head] = down_resolve
        return current

    result = {}

    all_keys = list(current.keys())
    for key in update.keys():
        if not key in current:
            all_keys.append(key)

    for key in all_keys:
        if not is_schema_field(current, key):
            continue

        try:
            value = resolve(
                current.get(key),
                update.get(key))

        except Exception as e:
            raise Exception(f'\ncannot resolve subtypes for key \'{key}\':\n{current}\n{update}\n\n  due to\n{e}')

        result[key] = value
    return result


def resolve_array_path(array: Array, update, path=None):
    if path:
        head = path[0]
        subshape = array._shape[1:]

        if subshape:
            down_schema = replace(array, **{
                '_shape': subshape})
            down_resolve = resolve(down_schema, update, path=path[1:])
            up_schema = replace(down_resolve, **{
                '_shape': (array._shape[0],) + tuple(down_resolve._shape)})
            return up_schema
        else:
            data_schema = dtype_schema(array._data)

            if isinstance(update, Array):
                if update._shape:
                    raise Exception(f'resolving arrays but they have different dimensions:\n\n{array}\n\n{update}')
                else:
                    subupdate = dtype_schema(update._data)
            elif isinstance(update, dict):
                subupdate = update.get(head)
            else:
                subupdate = update
                # raise Exception(f'resolving array with incompatible schema:\\n{array}\\n{update}')

            subschema = resolve(data_schema, subupdate, path=path[1:])
            dtype = schema_dtype(subschema)
            if isinstance(dtype, Array):
                up_schema = replace(array, **{
                    '_shape': array._shape + dtype._shape})
            else:
                up_schema = replace(array, **{
                    '_data': dtype})
            
            return up_schema
    else:
        return array


@dispatch
def resolve(current: Array, update: Array, path=None):
    if path:
        return resolve_array_path(current, update, path=path)

    new_shape = [
        max(current_shape, update_shape)
        for current_shape, update_shape in zip(current._shape, update._shape)]
    if len(current._shape) > len(update._shape):
        new_shape += current._shape[len(update._shape):]
    if len(update._shape) > len(current._shape):
        new_shape += update._shape[len(current._shape):]

    # Prefer the more specific subtype
    base = update if issubclass(type(update), type(current)) else current
    return replace(base, **{'_shape': new_shape})


@dispatch
def resolve(current: Array, update: Node, path=None):
    if path:
        return resolve_array_path(current, update, path=path)

    # TODO:
    #   finish array behavior

    return current

    # for key, subschema in update.items():
    #     if isinstance(key, int):

# @dispatch
# def resolve(current: Node, update: Array, path=None):
#     if path:
#         import ipdb; ipdb.set_trace()

#         return resolve_array_path(update, current, path=path)

#     return update

@dispatch
def resolve(current: Array, update: dict, path=None):
    if path:
        return resolve_array_path(current, update, path=path)
    else:
        return current

def resolve_dict_path(current, update, path=None):
    if path:
        head = path[0]
        if head == '*':
            if isinstance(update, Array):
                row_shape = update._shape[0]
                if not all([isinstance(key, int) for key in current.keys()]):
                    raise Exception(f'trying to resolve a dict and array but the keys are not all indexes:\n\n{current}\n\n{update}')

                if current:
                    row_shape = max(row_shape, max(current.keys()) + 1)

                subshape = update._shape[1:]
                if subshape:
                    subschema = replace(update, **{'_shape': subshape})
                else:
                    subschema = dtype_schema(update._data)

                if current:
                    resolve_schema = subschema
                    for key, subcurrent in current.items():
                        merge_schema = resolve(
                            subcurrent,
                            subschema,
                            path=path[1:])
                        resolve_schema = resolve(resolve_schema, merge_schema)

                else:
                    resolve_schema = resolve(
                        {},
                        subschema,
                        path=path[1:])

                    if isinstance(resolve_schema, dict):
                        inner_index = max(resolve_schema.keys())
                        inner = resolve_schema[inner_index]
                        if isinstance(inner, Array):
                            inner_shape = (inner_index+1,) + inner._shape
                        else:
                            inner_shape = (inner_index+1,)
                        resolve_schema = replace(update, **{'_shape': inner_shape})

                if isinstance(resolve_schema, Array):
                    resolve_shape = (row_shape,) + resolve_schema._shape
                    result_schema = replace(update, **{'_shape': resolve_shape})
                else:
                    dtype = schema_dtype(resolve_schema)

                    if isinstance(dtype, Array):
                        result_schema = replace(update, **{'_shape': update._shape + dtype._shape})
                    else:
                        result_schema = replace(update, **{'_data': dtype})

                return result_schema

        else:
            down_schema = current.get(head, {})
            down_resolve = resolve(down_schema, update, path=path[1:])
            current[head] = down_resolve
            return current

    else:
        return update

@dispatch
def resolve(current: dict, update: Array, path=None):
    if path:
        return resolve_dict_path(current, update, path=path)
    return update

def resolve_link(link: Link, update, path=None):
    if path:
        head = path[0]
        down_schema = {}
        if hasattr(link, head):
            down_schema = getattr(link, head)
        down_resolve = resolve(
            down_schema,
            update,
            path[1:])
        return replace(link, **{head: down_resolve})

    schema = link
    for key in ['_inputs', '_outputs']:
        if key in update:
            subupdate = update[key]
            attr = getattr(schema, key)
            subresolve = resolve(attr, subupdate)
            schema = replace(schema, **{key: subresolve})

    return schema

@dispatch
def resolve(current: Link, update: dict, path=None):
    return resolve_link(current, update, path=path)

@dispatch
def resolve(current: dict, update: Link, path=None):
    return resolve_link(update, current, path=path)

@dispatch
def resolve(current: Node, update: dict, path=None):
    if path:
        head = path[0]
        # If this Node has no such field (e.g. Overwrite, Maybe, etc.)
        # and the path segment is a dynamic key (like a daughter agent
        # id), we can't descend into it — return current unchanged.
        if not hasattr(current, head) or head not in current.__dataclass_fields__:
            return current
        down_schema = getattr(current, head)
        down_resolve = resolve(down_schema, update, path[1:])
        return replace(current, **{head: down_resolve})

    fields = set(current.__dataclass_fields__)
    keys = set(update.keys())

    if len(keys.difference(fields)) > 0:
        return update
    else:
        return current

@dispatch
def resolve(current: dict, update: Node, path=None):
    if path:
        return resolve_dict_path(current, update, path=path)

    if not current:
        return update

    fields = set(update.__dataclass_fields__)
    keys = set(current.keys())

    if len(keys.difference(fields)) > 0:
        return current
    else:
        return update

@dispatch
def resolve(current: String, update: Wrap, path=None):
    return replace(update, **{'_value':resolve(current, update._value, path=path)})

@dispatch
def resolve(current: String, update: Node, path=None):
    if current._default:
        update._default = current._default
    return update

# @dispatch
# def resolve(current: Node, update: String):
#     if update._default:
#         current._default = update._default
#     return current

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
def resolve(current: List, update: List, path=None):
    if current._default:
        if update._default:
            if len(current._default) > len(update._default):
                return current
            else:
                return update
        else:
            return current
    else:
        return update


@dispatch
def resolve(current: List, update: Array, path=None):
    """When the same path is declared as both List[T] and Array[T],
    favor Array — it's the more specific numerical form. The List is
    typically a generic fallback (often inferred from an empty default
    `[]`), while the Array carries shape and dtype information that
    matters for downstream operations.

    If the List has a non-empty default and the Array doesn't, lift the
    list as the Array's default so we don't lose the data.
    """
    if current._default and update._default is None:
        # Preserve the non-empty list as the array's default
        try:
            return replace(update, **{'_default': current._default})
        except Exception:
            return update
    return update


@dispatch
def resolve(current: Array, update: List, path=None):
    """Mirror of resolve(List, Array) — Array still wins."""
    if update._default and current._default is None:
        try:
            return replace(current, **{'_default': update._default})
        except Exception:
            return current
    return current

@dispatch
def resolve(current: List, update: Tuple, path=None):
    if not update._default and current._default:
        update._default = tuple(current._default)
    return update


@dispatch
def resolve(current: tuple, update: Tuple, path=None):
    result = []
    index = 0

    for subcurrent, subelement in zip(current, update._values):
        subresolve = resolve(subcurrent, subupdate, path=path)
        result.append(subresolve)
        index += 1

    if len(current) > len(update._values):
        result += list(current[index:])
    elif len(update._values) > len(current):
        result += list(update._values[index:])

    update._values = result
    return update


@dispatch
def resolve(current: tuple, update: tuple, path=None):
    result = []
    index = 0

    for subcurrent, subelement in zip(current, update):
        subresolve = resolve(subcurrent, subupdate, path=path)
        result.append(subresolve)
        index += 1

    if len(current) > len(update):
        result += list(current[index:])
    elif len(update) > len(current):
        result += list(update[index:])

    return tuple(result)


@dispatch
def resolve(current: list, update: list, path=None):
    ### ???
    return tuple(update)


@dispatch
def resolve(current, update, path=None):
    if current is None or current == {} or current == []:
        return update
    elif update is None or update == {} or update == []:
        return current
    else:
        raise Exception(f'\ncannot resolve types, not schemas:\n{current}\n{update}\n')


