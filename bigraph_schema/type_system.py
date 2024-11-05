"""
===========
Type System
===========
"""

import copy
import pprint
import pytest
import random
import inspect
import numbers
import functools
import numpy as np

from pint import Quantity
from pprint import pformat as pf

import typing
from typing import Any, Tuple, Mapping, Optional, Callable, NewType, Union
from dataclasses import field, make_dataclass, asdict

from bigraph_schema.units import units, render_units_type
from bigraph_schema.parse import parse_expression
from bigraph_schema.registry import (
    NONE_SYMBOL, Registry, default,
    type_schema_keys, 
    is_schema_key, non_schema_keys, strip_schema_keys, type_parameter_key,
    deep_merge, hierarchy_depth,
    establish_path, set_path, transform_path, remove_omitted)

import bigraph_schema.data as data
import bigraph_schema.data


TYPE_FUNCTION_KEYS = [
    '_apply',
    '_check',
    '_fold',
    '_divide',
    '_react',
    '_serialize',
    '_deserialize',
    '_slice',
    '_bind',
    '_merge']


overridable_schema_keys = set([
    '_type',
    '_default',
    '_check',
    '_apply',
    '_serialize',
    '_deserialize',
    '_fold',
    '_divide',
    '_slice',
    '_bind',
    '_merge',
    '_type_parameters',
    '_value',
    '_description',
    '_inherit',
])


SYMBOL_TYPES = ['enum']

nonoverridable_schema_keys = type_schema_keys - overridable_schema_keys


merge_schema_keys = (
    '_ports',
    '_type_parameters',
)


TYPE_SCHEMAS = {
    'float': 'float'}


def diff(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        result = {}
        for key in union_keys(a, b):
            if key in a:
                if key in b:
                    inner = diff(a[key], b[key])
                    if inner:
                        result[key] = inner
                else:
                    result[key] = f'A: {a[key]}\nB: (missing)'
            else:
                result[key] = f'A: (missing)\nB: {b[key]}'
        if result:
            return result
    else:
        if a != b:
            return f'A: {a}\nB: {b}'


def type_merge(dct, merge_dct, path=tuple(), merge_supers=False):
    """
    Recursively merge type definitions, never overwrite.

    Args:
    - dct: The dictionary to merge into. This dictionary is mutated and ends up being the merged dictionary.  If you 
        want to keep dct you could call it like ``deep_merge_check(copy.deepcopy(dct), merge_dct)``.
    - merge_dct: The dictionary to merge into ``dct``.
    - path: If the ``dct`` is nested within a larger dictionary, the path to ``dct``. This is normally an empty tuple 
        (the default) for the end user but is used for recursive calls.
    Returns:
    - dct
    """
    for k in merge_dct:
        if not k in dct or k in overridable_schema_keys:
            dct[k] = merge_dct[k]
        elif k in merge_schema_keys or isinstance(
            dct[k], dict
        ) and isinstance(
            merge_dct[k], collections.abc.Mapping
        ):
            type_merge(
                dct[k],
                merge_dct[k],
                path + (k,),
                merge_supers)

        else:
            raise ValueError(
                f'cannot merge types at path {path + (k,)}:\n'
                f'{dct}\noverwrites \'{k}\' from\n{merge_dct}')
            
    return dct


def get_path(tree, path):
    """
    Given a tree and a path, find the subtree at that path
    
    Args:
    - tree: the tree we are looking in (a nested dict)
    - path: a list/tuple of keys we follow down the tree to find the subtree we are looking for
    
    Returns:
    - subtree: the subtree found by following the list of keys down the tree
    """

    if len(path) == 0:
        return tree
    else:
        head = path[0]
        if not tree or head not in tree:
            return None
        else:
            return get_path(tree[head], path[1:])


def remove_path(tree, path):
    """
    Removes whatever subtree lives at the given path
    """

    if path is None or len(path) == 0:
        return None

    upon = get_path(tree, path[:-1])
    if upon is not None:
        del upon[path[-1]]
    return tree


def resolve_path(path):
    resolve = []

    for step in path:
        if step == '..':
            if len(resolve) == 0:
                raise Exception(f'cannot go above the top in path: "{path}"')
            else:
                resolve = resolve[:-1]
        else:
            resolve.append(step)

    return tuple(resolve)


def apply_schema(schema, current, update, core):
    outcome = core.resolve_schemas(current, update)
    return outcome


def visit_method(schema, state, method, values, core):
    schema = core.access(schema)
    method_key = f'_{method}'

    # TODO: we should probably cache all this
    if isinstance(state, dict) and method_key in state:
        visit = core.find_method(
            {method_key: state[method_key]},
            method_key)

    elif method_key in schema:
        visit = core.find_method(
            schema,
            method_key)

    else:
        visit = core.find_method(
            'any',
            method_key)

    result = visit(
        schema,
        state,
        values,
        core)

    return result


def fold_any(schema, state, method, values, core):
    if isinstance(state, dict):
        result = {}
        for key, value in state.items():
            if key.startswith('_'):
                result[key] = value
            else:
                if key in schema:
                    fold = core.fold_state(
                        schema[key],
                        value,
                        method,
                        values)
                    result[key] = fold

    else:
        result = state

    visit = visit_method(
        schema,
        result,
        method,
        values,
        core)

    return visit


def fold_tuple(schema, state, method, values, core):
    if not isinstance(state, (tuple, list)):
        return visit_method(
            schema,
            state,
            method,
            values,
            core)
    else:
        parameters = core.parameters_for(schema)
        result = []
        for parameter, element in zip(parameters, state):
            fold = core.fold(
                parameter,
                element,
                method,
                values)
            result.append(fold)

        result = tuple(result)

        return visit_method(
            schema,
            result,
            method,
            values,
            core)


def fold_union(schema, state, method, values, core):
    union_type = find_union_type(
        core,
        schema,
        state)

    result = core.fold(
        union_type,
        state,
        method,
        values)

    return result


def type_parameters_for(schema):
    parameters = []
    for key in schema['_type_parameters']:
        subschema = schema.get(f'_{key}', 'any')
        parameters.append(subschema)

    return parameters


def dataclass_union(schema, path, core):
    parameters = type_parameters_for(schema)
    subtypes = []
    for parameter in parameters:
        dataclass = core.dataclass(
            parameter,
            path)

        if isinstance(dataclass, str):
            subtypes.append(dataclass)
        elif isinstance(dataclass, type):
            subtypes.append(dataclass.__name__)
        else:
            subtypes.append(str(dataclass))

    parameter_block = ', '.join(subtypes)
    return eval(f'Union[{parameter_block}]')


def divide_any(schema, state, values, core):
    divisions = values.get('divisions', 2)

    if isinstance(state, dict):
        result = [
            {}
            for _ in range(divisions)]

        for key, value in state.items():
            for index in range(divisions):
                result[index][key] = value[index]

        return result

    else:
        # TODO: division operates on and returns dictionaries
#         return {
#             id: copy.deepcopy(state),
#             for generate_new_id(existing_id, division) in range(divisions)}
# ?????

        return [
            copy.deepcopy(state)
            for _ in range(divisions)]


def resolve_any(schema, update, core):
    schema = schema or {}
    outcome = schema.copy()

    for key, subschema in update.items():
        if key == '_type' and key in outcome:
            if outcome[key] != subschema:
                if core.inherits_from(outcome[key], subschema):
                    continue
                elif core.inherits_from(subschema, outcome[key]):
                    outcome[key] = subschema
                else:
                    raise Exception(f'cannot resolve types when updating\ncurrent type: {schema}\nupdate type: {update}')

        elif not key in outcome or type_parameter_key(update, key):
            if subschema:
                outcome[key] = subschema
        else:
            outcome[key] = core.resolve_schemas(
                outcome.get(key),
                subschema)

    return outcome


def dataclass_any(schema, path, core):
    parts = path
    if not parts:
        parts = ['top']
    dataclass_name = '_'.join(parts)

    if isinstance(schema, dict):
        type_name = schema.get('_type', 'any')

        branches = {}
        for key, subschema in schema.items():
            if not key.startswith('_'):
                branch = core.dataclass(
                    subschema,
                    path + [key])

                def default(subschema=subschema):
                    return core.default(subschema)

                branches[key] = (
                    key,
                    branch,
                    field(default_factory=default))

        dataclass = make_dataclass(
            dataclass_name,
            branches.values(),
            namespace={
                '__module__': 'bigraph_schema.data'})

        setattr(
            bigraph_schema.data,
            dataclass_name,
            dataclass)

    else:
        schema = core.access(schema)
        dataclass = core.dataclass(schema, path)

    return dataclass


def dataclass_tuple(schema, path, core):
    parameters = type_parameters_for(schema)
    subtypes = []

    for index, key in enumerate(schema['type_parameters']):
        subschema = schema.get(key, 'any')
        subtype = core.dataclass(
            subschema,
            path + [index])

        subtypes.append(subtype)

    parameter_block = ', '.join(subtypes)
    return eval(f'tuple[{parameter_block}]')


def divide_tuple(schema, state, values, core):
    divisions = values.get('divisions', 2)

    return [
        tuple([item[index] for item in state])
        for index in range(divisions)]


def apply_tree(schema, current, update, core):
    leaf_type = core.find_parameter(
        schema,
        'leaf')

    if current is None:
        current = core.default(leaf_type)

    if isinstance(current, dict) and isinstance(update, dict):
        for key, branch in update.items():
            if key == '_add':
                current.update(branch)
            elif key == '_remove':
                for removed_path in branch:
                    if isinstance(removed_path, str):
                        removed_path = [removed_path]
                    current = remove_path(current, removed_path)
            elif isinstance(branch, dict):
                subschema = schema
                if key in schema:
                    subschema = schema[key]

                current[key] = core.apply(
                    subschema,
                    current.get(key),
                    branch)

            elif core.check(leaf_type, branch):
                current[key] = core.apply(
                    leaf_type,
                    current.get(key),
                    branch)

            else:
                raise Exception(f'state does not seem to be of leaf type:\n  state: {state}\n  leaf type: {leaf_type}')

        return current

    elif core.check(leaf_type, current):
        return core.apply(
            leaf_type,
            current,
            update)

    else:
        raise Exception(f'trying to apply an update to a tree but the values are not trees or leaves of that tree\ncurrent:\n  {pf(current)}\nupdate:\n  {pf(update)}\nschema:\n  {pf(schema)}')


def apply_any(schema, current, update, core):
    if isinstance(current, dict):
        return apply_tree(
            current,
            update,
            'tree[any]',
            core)
    else:
        return update


def slice_any(schema, state, path, core):
    if not isinstance(path, (list, tuple)):
        if path is None:
            path = ()
        else:
            path = [path]

    if len(path) == 0:
        return schema, state

    elif len(path) > 0:
        head = path[0]
        tail = path[1:]
        step = None

        if isinstance(state, dict):
            if head not in state:
                state[head] = core.default(
                    schema.get(head))
            step = state[head]

        elif hasattr(state, head):
            step = getattr(state, head)

        if head in schema:
            return core.slice(
                schema[head],
                step,
                tail)
        else:
            return slice_any(
                {},
                step,
                tail,
                core)


def check_any(schema, state, core):
    if isinstance(schema, dict):
        for key, subschema in schema.items():
            if not key.startswith('_'):
                if isinstance(state, dict):
                    if key in state:
                        check = core.check_state(
                            subschema,
                            state[key])

                        if not check:
                            return False
                    else:
                        return False
                else:
                    return False

        return True
    else:
        return True


def serialize_any(schema, state, core):
    if isinstance(state, dict):
        tree = {}

        for key in non_schema_keys(schema):
            encoded = core.serialize(
                schema.get(key, schema),
                state.get(key))
            tree[key] = encoded

        return tree

    else:
        return str(state)


def deserialize_any(schema, state, core):
    if isinstance(state, dict):
        tree = {}

        for key, value in state.items():
            if is_schema_key(key):
                decoded = value
            else:
                decoded = core.deserialize(
                    schema.get(key, 'any'),
                    value)

            tree[key] = decoded

        for key in non_schema_keys(schema):
            if key not in tree:
                # if key not in state:
                #     decoded = core.default(
                #         schema[key])
                # else:
                if key in state:
                    decoded = core.deserialize(
                        schema[key],
                        state[key])

                    tree[key] = decoded

        return tree

    else:
        return state


def is_empty(value):
    if isinstance(value, np.ndarray):
        return False
    elif value is None or value == {}:
        return True
    else:
        return False


def merge_any(schema, current_state, new_state, core):
    # overwrites in place!
    # TODO: this operation could update the schema (by merging a key not
    #   already in the schema) but that is not represented currently....
    merge_state = None

    if is_empty(current_state):
        merge_state = new_state

    elif is_empty(new_state):
        merge_state = current_state
    
    elif isinstance(new_state, dict):
        if isinstance(current_state, dict):
            for key, value in new_state.items():
                if is_schema_key(key):
                    current_state[key] = value
                else:
                    current_state[key] = core.merge(
                        schema.get(key),
                        current_state.get(key),
                        value)
            merge_state = current_state
        else:
            merge_state = new_state
    else:
        merge_state = new_state

    return merge_state

    # return core.deserialize(
    #     schema,
    #     merge_state)


def bind_any(schema, state, key, subschema, substate, core):
    result_schema = core.resolve_schemas(
          schema,
          {key: subschema})

    state[key] = substate

    return result_schema, state


def apply_tuple(schema, current, update, core):
    parameters = core.parameters_for(schema)
    result = []

    for parameter, current_value, update_value in zip(parameters, current, update):
        element = core.apply(
            parameter,
            current_value,
            update_value)

        result.append(element)

    return tuple(result)


def check_tuple(schema, state, core):
    if not isinstance(state, (tuple, list)):
        return False

    parameters = core.parameters_for(schema)
    for parameter, element in zip(parameters, state):
        if not core.check(parameter, element):
            return False

    return True


def slice_tuple(schema, state, path, core):
    if len(path) > 0:
        head = path[0]
        tail = path[1:]

        if str(head) in schema['_type_parameters']:
            try:
                index = schema['_type_parameters'].index(str(head))
            except:
                raise Exception(f'step {head} in path {path} is not a type parameter of\n  schema: {pf(schema)}\n  state: {pf(state)}')
            index_key = f'_{index}'
            subschema = core.access(schema[index_key])

            return core.slice(subschema, state[head], tail)
        else:
            raise Exception(f'trying to index a tuple with a key that is not an index: {state} {head}')
    else:
        return schema, state


def serialize_tuple(schema, value, core):
    parameters = core.parameters_for(schema)
    result = []

    for parameter, element in zip(parameters, value):
        encoded = core.serialize(
            parameter,
            element)

        result.append(encoded)

    return tuple(result)


def deserialize_tuple(schema, state, core):
    parameters = core.parameters_for(schema)
    result = []

    if isinstance(state, str):
        if (state[0] == '(' and state[-1] == ')') or (state[0] == '[' and state[-1] == ']'):
            state = state[1:-1].split(',')
        else:
            return None

    for parameter, code in zip(parameters, state):
        element = core.deserialize(
            parameter,
            code)

        result.append(element)

    return tuple(result)


def bind_tuple(schema, state, key, subschema, substate, core):
    new_schema = schema.copy()
    new_schema[f'_{key}'] = subschema
    open = list(state)
    open[key] = substate

    return new_schema, tuple(open)


def find_union_type(core, schema, state):
    parameters = core.parameters_for(schema)

    for possible in parameters:
        if core.check(possible, state):
            return core.access(possible)

    return None


def apply_union(schema, current, update, core):
    current_type = find_union_type(
        core,
        schema,
        current)

    update_type = find_union_type(
        core,
        schema,
        update)

    if current_type is None:
        raise Exception(f'trying to apply update to union value but cannot find type of value in the union\n  value: {current}\n  update: {update}\n  union: {list(bindings.values())}')
    elif update_type is None:
        raise Exception(f'trying to apply update to union value but cannot find type of update in the union\n  value: {current}\n  update: {update}\n  union: {list(bindings.values())}')

    # TODO: throw an exception if current_type is incompatible with update_type

    return core.apply(
        update_type,
        current,
        update)


def check_union(schema, state, core):
    found = find_union_type(
        core,
        schema,
        state)

    return found is not None and len(found) > 0


def slice_union(schema, state, path, core):
    union_type = find_union_type(
        core,
        schema,
        state)

    return core.slice(
        union_type,
        state,
        path)


def bind_union(schema, state, key, subschema, substate, core):
    union_type = find_union_type(
        core,
        schema,
        state)

    return core.bind(
        union_type,
        state,
        key,
        subschema,
        substate)


def serialize_union(schema, value, core):
    union_type = find_union_type(
        core,
        schema,
        value)

    return core.serialize(
        union_type,
        value)


def deserialize_union(schema, encoded, core):
    if encoded == NONE_SYMBOL:
        return None
    else:
        parameters = core.parameters_for(schema)

        for parameter in parameters:
            value = core.deserialize(
                parameter,
                encoded)

            if value is not None:
                return value


def default_any(schema, core):
    default = {}

    for key, subschema in schema.items():
        if not is_schema_key(key):
            default[key] = core.default(
                subschema)

    return default


def default_tuple(schema, core):
    parts = []
    for parameter in schema['_type_parameters']:
        subschema = schema[f'_{parameter}']
        part = core.default(subschema, core)
        parts.append(part)

    return tuple(parts)


def default_union(schema, core):
    final_parameter = schema['_type_parameters'][-1]
    subschema = schema[f'_{final_parameter}']

    return core.default(subschema)


def union_keys(schema, state):
    return set(schema.keys()).union(state.keys())


def generate_any(core, schema, state, top_schema=None, top_state=None, path=None):
    schema = schema or {}
    state = state or core.default(schema)
    top_schema = top_schema or schema
    top_state = top_state or state
    path = path or []

    generated_schema = {}
    generated_state = {}

    if isinstance(state, dict):
        visited = set([])

        all_keys = union_keys(
            schema,
            state)

        non_schema_keys = [
            key
            for key in all_keys
            if not is_schema_key(key)]

        if non_schema_keys:
            base_schema = {
                key: subschema
                for key, subschema in schema.items()
                if is_schema_key(key)}
        else:
            base_schema = schema

        found_value = None
        for key in all_keys:
            if is_schema_key(key):
                if key == '_value':
                    found_value = state.get(
                        key,
                        schema.get(key))

                elif key in state:
                    schema[key] = state[key]

            else:
                subschema, substate, top_schema, top_state = core.generate_recur(
                    schema.get(key),
                    state.get(key),
                    top_schema=top_schema,
                    top_state=top_state,
                    path=path+[key])

                schema[key] = core.resolve_schemas(
                    schema.get(key, {}),
                    subschema)

                state[key] = substate

        if found_value:
            state = deep_merge(
                state,
                found_value)

    return schema, state, top_schema, top_state


def is_method_key(key, parameters):
    return key.startswith('_') and key not in type_schema_keys and key not in [
        f'_{parameter}' for parameter in parameters]


registry_types = {
    'any': {
        '_type': 'any',
        '_default': default_any,
        '_slice': slice_any,
        '_apply': apply_any,
        '_check': check_any,
        '_generate': generate_any,
        '_serialize': serialize_any,
        '_deserialize': deserialize_any,
        '_dataclass': dataclass_any,
        '_resolve': resolve_any,
        '_fold': fold_any,
        '_merge': merge_any,
        '_bind': bind_any,
        '_divide': divide_any},

    'tuple': {
        '_type': 'tuple',
        '_default': default_tuple,
        '_apply': apply_tuple,
        '_check': check_tuple,
        '_slice': slice_tuple,
        '_serialize': serialize_tuple,
        '_deserialize': deserialize_tuple,
        '_dataclass': dataclass_tuple,
        '_fold': fold_tuple,
        '_divide': divide_tuple,
        '_bind': bind_tuple,
        '_description': 'tuple of an ordered set of typed values'},

    'union': {
        '_type': 'union',
        '_default': default_union,
        '_apply': apply_union,
        '_check': check_union,
        '_slice': slice_union,
        '_serialize': serialize_union,
        '_deserialize': deserialize_union,
        '_dataclass': dataclass_union,
        '_fold': fold_union,
        '_description': 'union of a set of possible types'}}


class TypeSystem(Registry):
    """Handles type schemas and their operation"""

    def __init__(self):
        super().__init__()

        self.inherits = {}

        self.default_registry = Registry(function_keys=[
            'schema',
            'core'])

        self.check_registry = Registry(function_keys=[
            'state',
            'schema',
            'core'])

        self.apply_registry = Registry(function_keys=[
            'current',
            'update',
            'schema',
            'core'])

        self.serialize_registry = Registry(function_keys=[
            'value',
            'schema',
            'core'])

        self.deserialize_registry = Registry(function_keys=[
            'encoded',
            'schema',
            'core'])

        self.fold_registry = Registry(function_keys=[
             'method',
             'state',
             'schema',
             'core'])

        self.react_registry = Registry()
        self.method_registry = Registry()

        # register all the base methods and types
        self.apply_registry.register(
            'set',
            set_apply)

        self.register_types(registry_types)
        self.register_types(base_type_library)

        register_units(self, units)
        register_base_reactions(self)


    def register_types(core, type_library):
        for type_key, type_data in type_library.items():
            if not core.exists(type_key):
                core.register(
                    type_key,
                    type_data)

        return core
        

    def lookup(self, type_key, attribute):
        return self.access(type_key).get(attribute)


    def lookup_registry(self, underscore_key):
        """
        access the registry for the given key
        """

        if underscore_key == '_type':
            return self
        root = underscore_key.strip('_')
        registry_key = f'{root}_registry'
        if hasattr(self, registry_key):
            return getattr(self, registry_key)


    def find_registry(self, underscore_key):
        """
        access the registry for the given key
        and create if it doesn't exist
        """

        registry = self.lookup_registry(underscore_key)
        if registry is None:
            registry = Registry()
            setattr(
                self,
                f'{underscore_key[1:]}_registry',
                registry)

        return registry


    def register(self, key, schema, alternate_keys=tuple(), force=False):
        """
        register the schema under the given key in the registry
        """

        if isinstance(schema, str):
            schema = self.find(schema)
        schema = copy.deepcopy(schema)

        if '_type' not in schema:
            schema['_type'] = key

        if isinstance(schema, dict):
            inherits = schema.get('_inherit', [])  # list of immediate inherits
            if isinstance(inherits, str):
                inherits = [inherits]
                schema['_inherit'] = inherits

            self.inherits[key] = []
            for inherit in inherits:
                inherit_type = self.access(inherit)
                new_schema = copy.deepcopy(inherit_type)

                # schema = self.resolve(
                #     new_schema,
                #     schema)

                schema = self.merge_schemas(
                    new_schema,
                    schema)

                # schema = type_merge(
                #     new_schema,
                #     schema)

                self.inherits[key].append(
                    inherit_type)

            parameters = schema.get('_type_parameters', [])
            for subkey, subschema in schema.items():
                if subkey == '_default' or subkey in TYPE_FUNCTION_KEYS or is_method_key(subkey, parameters):
                    if callable(subschema):
                        registry = self.find_registry(subkey)
                        function_name, module_key = registry.register_function(subschema)

                        schema[subkey] = function_name
                    else:
                        schema[subkey] = subschema

                elif subkey not in type_schema_keys:
                    if schema['_type'] in SYMBOL_TYPES:
                        schema[subkey] = subschema
                    else:
                        lookup = self.find(subschema)
                        if lookup is None:
                            import ipdb; ipdb.set_trace()
                            raise Exception(
                                f'trying to register a new type ({key}), '
                                f'but it depends on a type ({subkey}) which is not in the registry')
                        else:
                            schema[subkey] = lookup
        else:
            raise Exception(
                f'all type definitions must be dicts '
                f'with the following keys: {type_schema_keys}\nnot: {schema}')

        super().register(
            key,
            schema,
            alternate_keys,
            force)


    def resolve_parameters(self, type_parameters, schema):
        """
        find the types associated with any type parameters in the schema
        """

        return {
            type_parameter: self.access(
                schema.get(f'_{type_parameter}'))
            for type_parameter in type_parameters}


    def register_reaction(self, reaction_key, reaction):
        self.react_registry.register(
            reaction_key,
            reaction)


    def types(self):
        return {
            type_key: type_data
            for type_key, type_data in self.registry.items()}


    def merge_schemas(self, current, update):
        if not isinstance(current, dict):
            return update
        if not isinstance(update, dict):
            return update
    
        merged = {}

        for key in union_keys(current, update):
            if key in current:
                if key in update:
                    merged[key] = self.merge_schemas(
                        current[key],
                        update[key])
                else:
                    merged[key] = current[key]
            else:
                merged[key] = update[key]

        return merged


    def merge_schema_keys(self, schema, state):
        if not isinstance(schema, dict):
            schema = self.find(schema)
        if not isinstance(state, dict):
            return schema, state
    
        merged_schema = {}
        merged_state = {}

        for key in union_keys(schema, state):
            if is_schema_key(key):
                if key in state:
                    merged_schema[key] = self.merge_schemas(
                        schema.get(key, {}),
                        state[key])
                else:
                    merged_schema[key] = schema[key]
            else:
                merged_schema[key], merged_state[key] = self.merge_schema_keys(
                    schema.get(key, {}),
                    state.get(key, None))

        return merged_schema, merged_state


    def exists(self, type_key):
        return type_key in self.registry


    def find(self, schema, strict=False):
        """
        expand the schema to its full type information from the type registry
        """

        found = None

        if schema is None:
            return self.access('any', strict=strict)

        elif isinstance(schema, dict):
            if '_description' in schema:
                return schema

            elif '_union' in schema:
                union_schema = {
                    '_type': 'union',
                    '_type_parameters': []}

                for index, element in enumerate(schema['_union']):
                    union_schema['_type_parameters'].append(str(index))
                    union_schema[f'_{index}'] = element

                return self.access(
                    union_schema,
                    strict=strict)

            elif '_type' in schema:
                registry_type = self.retrieve(
                    schema['_type'])

                # found = self.resolve(
                #     registry_type,
                #     schema)

                found = self.merge_schemas(
                    registry_type,
                    schema)

                # found = schema.copy()

                # for key, value in registry_type.items():
                #     if  key == '_type' or key not in found:
                #         found[key] = value

            else:
                found = {
                   key: self.access(
                       branch,
                       strict=strict)
                   for key, branch in schema.items()}

        elif isinstance(schema, int):
            return schema

        elif isinstance(schema, tuple):
            tuple_schema = {
                '_type': 'tuple',
                '_type_parameters': []}

            for index, element in enumerate(schema):
                tuple_schema['_type_parameters'].append(str(index))
                tuple_schema[f'_{index}'] = element

            return self.access(
                tuple_schema,
                strict=strict)

        elif isinstance(schema, list):
            if isinstance(schema[0], int):
                return schema

            bindings = []
            if len(schema) > 1:
                schema, bindings = schema
            else:
                schema = schema[0]
            found = self.access(
                schema,
                strict=strict)

            if len(bindings) > 0:
                found = found.copy()

                if '_type_parameters' not in found:
                    found['_type_parameters'] = []
                    for index, binding in enumerate(bindings):
                        found['_type_parameters'].append(str(index))
                        found[f'_{index}'] = binding
                else:
                    for parameter, binding in zip(found['_type_parameters'], bindings):
                        binding_type = self.access(
                            binding,
                            strict=strict) or binding

                        found[f'_{parameter}'] = binding_type

        elif isinstance(schema, str):
            found = self.registry.get(schema)

            if found is None and schema not in ('', '{}'):
                try:
                    parse = parse_expression(schema)
                    if parse != schema:
                        found = self.access(
                            parse,
                            strict=strict)
                    elif not strict:
                        found = {'_type': schema}

                except Exception:
                    print(f'type did not parse: {schema}')
                    traceback.print_exc()
                    
        return found


    def access(self, schema, strict=False):
        if isinstance(schema, str):
            return self.access_str(schema, strict=strict)
        else:
            return self.find(schema, strict=strict)


    @functools.lru_cache(maxsize=None)
    def access_str(self, schema, strict=False):
        return self.find(
            schema,
            strict)


    def retrieve(self, schema):
        """
        like access(schema) but raises an exception if nothing is found
        """

        found = self.find(
            schema,
            strict=True)

        if found is None:
            raise Exception(f'schema not found for type: {schema}')
        return found


    def find_parameter(self, schema, parameter):
        schema_key = f'_{parameter}'
        if schema_key not in schema:
            schema = self.access(schema)
        if schema_key not in schema:
            return 'any'
            # raise Exception(f'parameter {parameter} not found in schema:\n  {schema}')

        parameter_type = self.access(
            schema[schema_key])

        return parameter_type


    def parameters_for(self, initial_schema):
        '''
        find any type parameters for this schema if they are present
        '''

        if '_type_parameters' in initial_schema:
            schema = initial_schema
        else:
            schema = self.access(initial_schema)

        if '_type_parameters' not in schema:
            return []
        else:
            result = []
            for parameter in schema['_type_parameters']:
                parameter_key = f'_{parameter}'
                if parameter_key not in schema:
                    raise Exception(
                        f'looking for type parameter {parameter_key} but it is not in the schema:\n  {pf(schema)}')
                else:
                    parameter_type = schema[parameter_key]
                    result.append(
                        parameter_type)

            return result


    def validate_schema(self, schema, enforce_connections=False):
        # TODO:
        #   check() always returns true or false,
        #   validate() returns information about what doesn't match

        # add ports and wires
        # validate ports are wired to a matching type,
        #   either the same type or a subtype (more specific type)
        # declared ports are equivalent to the presence of a process
        #   where the ports need to be looked up

        report = {}

        if schema is None:
            report = 'schema cannot be None'

        elif isinstance(schema, str):
            typ = self.access(
                schema,
                strict=True)

            if typ is None:
                report = f'type: {schema} is not in the registry'

        elif isinstance(schema, dict):
            report = {}

            schema_keys = set([])
            branches = set([])

            for key, value in schema.items():
                if key == '_type':
                    typ = self.access(
                        value,
                        strict=True)
                    if typ is None:
                        report[key] = f'type: {value} is not in the registry'

                elif key in type_schema_keys:
                    schema_keys.add(key)
                    registry = self.lookup_registry(key)
                    if registry is None or key == '_default':
                        # deserialize and serialize back and check it is equal
                        pass
                    elif isinstance(value, str):
                        element = registry.access(
                            value,
                            strict=True)

                        if element is None:
                            report[key] = f'no entry in the {key} registry for: {value}'
                    elif not inspect.isfunction(value):
                        report[key] = f'unknown value for key {key}: {value}'
                else:
                    branches.add(key)
                    branch_report = self.validate_schema(value)
                    if len(branch_report) > 0:
                        report[key] = branch_report

        return report


    # TODO: if its an edge, ensure ports match wires
    # TODO: make this work again, return information about what is wrong
    #   with the schema
    def validate_state(self, original_schema, state):
        schema = self.access(original_schema)
        validation = {}

        if '_serialize' in schema:
            if '_deserialize' not in schema:
                validation = {
                    '_deserialize': f'serialize found in type without deserialize: {schema}'
                }
            else:
                serialize = self.serialize_registry.access(
                    schema['_serialize'])
                deserialize = self.deserialize_registry.access(
                    schema['_deserialize'])
                serial = serialize(state)
                pass_through = deserialize(serial)

                if state != pass_through:
                    validation = f'state and pass_through are not the same: {serial}'
        else:
            for key, subschema in schema.items():
                if key not in type_schema_keys:
                    if key not in state:
                        validation[key] = f'key present in schema but not in state: {key}\nschema: {schema}\nstate: {state}\n'
                    else:
                        subvalidation = self.validate_state(
                            subschema,
                            state[key])
                        if not (subvalidation is None or len(subvalidation) == 0):
                            validation[key] = subvalidation

        return validation


    def representation(self, schema, level=None):
        '''
        produce a string representation of the schema
        * intended to be the inverse of parse_expression()
        '''


        if isinstance(schema, str):
            return schema

        elif isinstance(schema, tuple):
            inner = [
                self.representation(element)
                for element in schema]

            pipes = '|'.join(inner)
            return f'({pipes})'
            
        elif isinstance(schema, dict):
            if '_type' in schema:
                type = schema['_type']

                inner = []
                block = ''
                if '_type_parameters' in schema:
                    for parameter_key in schema['_type_parameters']:
                        schema_key = f'_{parameter_key}'
                        if schema_key in schema:
                            parameter = self.representation(
                                schema[schema_key])
                            inner.append(parameter)
                        else:
                            inner.append('()')

                    commas = ','.join(inner)
                    block = f'[{commas}]'

                if type == 'tuple':
                    pipes = '|'.join(inner)
                    return f'({pipes})'
                else:
                    return f"{type}{block}"

            else:
                inner = {}
                for key in non_schema_keys(schema):
                    subschema = self.representation(
                        schema[key])

                    inner[key] = subschema

                colons = [
                    f'{key}:{value}'
                    for key, value in inner.items()]

                pipes = '|'.join(colons)
                return f'({pipes})'
        else:
            print(f'no representation for {schema}')
            return str(schema)


    def default(self, schema):
        '''
        produce the default value for the provided schema
        '''

        default = None
        found = self.retrieve(schema)

        if '_default' in found:
            default_value = found['_default']
            if isinstance(default_value, str):
                default_method = self.default_registry.access(default_value)
                if default_method and callable(default_method):
                    default = default_method(found, self)
                else:
                    default = self.deserialize(
                        found,
                        default_value)

            elif not '_deserialize' in found:
                raise Exception(
                    f'asking for default but no deserialize in {found}')

            else:
                default = self.deserialize(found, found['_default'])

        else:
            default = {}
            for key, subschema in found.items():
                if not is_schema_key(key):
                    default[key] = self.default(subschema)

        return default


    def choose_method(self, schema, state, method_name):
        '''
        find in the provided state, or schema if not there,
        a method for the given method_name
        '''

        method_key = f'_{method_name}'
        found = None

        if isinstance(state, dict) and method_key in state:
            found = state[method_key]

        elif isinstance(state, dict) and '_type' in state:
            method_type = self.find(
                state['_type'])

            if method_type:
                found = method_type.get(method_key)

        if not found and isinstance(schema, dict) and method_key in schema:
            found = schema[method_key]

        if found is None:
            any_type = self.access('any')
            found = any_type[method_key]

        registry = self.lookup_registry(method_key)
        method_function = registry.access(
            found)

        return method_function


    def slice(self, schema, state, path):
        '''
        find the subschema and substate at a node in the place graph
        given by the provided path
        '''

        if not isinstance(path, (list, tuple)):
            path = [path]

        schema = self.access(schema)
        if '..' in path:
            path = resolve_path(path)

        slice_function = self.choose_method(
            schema,
            state,
            'slice')

        return slice_function(
            schema,
            state,
            path,
            self)


    def match_node(self, schema, state, pattern):
        if isinstance(pattern, dict):
            if not isinstance(state, dict):
                return False

            if '_type' in pattern and not self.is_compatible(schema, pattern):
                return False

            for key, subpattern in pattern.items():
                if key.startswith('_'):
                    continue

                if key in schema:
                    subschema = schema[key]
                else:
                    subschema = schema

                if key in state:
                    matches = self.match_node(
                        subschema,
                        state[key],
                        pattern[key])
                    if not matches:
                        return False
                else:
                    return False

            return True

        else:
            return pattern == state


    def match_recur(self, schema, state, pattern, mode='first', path=()):
        matches = []

        match = self.match_node(
            schema,
            state,
            pattern)

        if match:
            if mode == 'first' or mode == 'immediate':
                return [path]
            else:
                matches.append(path)
        elif mode == 'immediate':
            return []

        if isinstance(state, dict):
            for key, substate in state.items():
                if key.startswith('_'):
                    continue

                if key in schema:
                    subschema = schema[key]
                else:
                    subschema = schema

                submatches = self.match_recur(
                    subschema,
                    state[key],
                    pattern,
                    mode=mode,
                    path=path + (key,))

                if mode == 'first' and len(submatches) > 0:
                    return submatches[0:1]
                else:
                    matches.extend(submatches)

        return matches


    def match(self, original_schema, state, pattern, mode='first', path=()):
        """
        find the path or paths to any instances of a
        given pattern in the tree.

        "mode" can be a few things:
        * immediate: only match top level
        * first: only return the first match
        * random: return a random match of all that matched
        * all (or any other value): return every match in the tree
        """

        schema = self.access(original_schema)

        matches = self.match_recur(
            schema,
            state,
            pattern,
            mode=mode,
            path=path)

        if mode == 'random':
            matches_count = len(matches)
            if matches_count > 0:
                choice = random.randint(
                    0,
                    matches_count-1)
                matches = [matches[choice]]

        return matches


    def react(self, schema, state, reaction, mode='random'):
        # TODO: explain all this
        # TODO: after the reaction, fill in the state with missing values
        #   from the schema

        # TODO: add schema to redex and reactum

        if 'redex' in reaction or 'reactum' in reaction or 'calls' in reaction:
            redex = reaction.get('redex', {})
            reactum = reaction.get('reactum', {})
            calls = reaction.get('calls', {})
        else:
            # single key with reaction name
            reaction_key = list(reaction.keys())[0]
            make_reaction = self.react_registry.access(
                reaction_key)
            react = make_reaction(
                schema,
                state,
                reaction.get(reaction_key, {}),
                self)

            redex = react.get('redex', {})
            reactum = react.get('reactum', {})
            calls = react.get('calls', {})

        paths = self.match(
            schema,
            state,
            redex,
            mode=mode)
        
        # for path in paths:
        #     path_schema, path_state = self.slice(
        #         schema,
        #         state,
        #         path)

        def merge_state(before):
            remaining = remove_omitted(
                redex,
                reactum,
                before)

            merged = deep_merge(
                remaining,
                reactum)

            return merged

        for path in paths:
            state = transform_path(
                state,
                path,
                merge_state)

        return state


    # TODO: maybe all fields are optional?
    def dataclass(self, schema, path=None):
        path = path or []

        dataclass_function = self.choose_method(
            schema,
            {},
            'dataclass')

        return dataclass_function(
            schema,
            path,
            self)
        

    def resolve(self, schema, update):
        if update is None:
            return schema
        else:
            resolve_function = self.choose_method(
                schema,
                update,
                'resolve')

            return resolve_function(
                schema,
                update,
                self)


    def resolve_schemas(self, initial_current, initial_update):
        current = self.access(initial_current)
        update = self.access(initial_update)

        if self.equivalent(current, update):
            outcome = current

        elif self.inherits_from(current, update):
            outcome = current

        elif self.inherits_from(update, current):
            outcome = update

        elif '_type' in current and '_type' in update and current['_type'] == update['_type']:
            outcome = current.copy()

            for key in update:
                if key == '_type_parameters' and '_type_parameters' in current:
                    for parameter in update['_type_parameters']:
                        parameter_key = f'_{parameter}'
                        if parameter in current['_type_parameters']:
                            outcome[parameter_key] = self.resolve_schemas(
                                current[parameter_key],
                                update[parameter_key])
                        else:
                            outcome[parameter_key] = update[parameter_key]
                elif key not in outcome or type_parameter_key(current, key):
                    key_update = update[key]
                    if key_update:
                        outcome[key] = key_update
                else:
                    outcome[key] = self.resolve_schemas(
                        outcome.get(key),
                        update[key])

        elif '_type' in update and '_type' not in current:
            outcome = self.resolve(update, current)

        else:
            outcome = self.resolve(current, update)

        # elif '_type' in current:
        #     outcome = self.resolve(current, update)

        # elif '_type' in update:
        #     outcome = self.resolve(update, current)

        # else:
        #     outcome = self.resolve(current, update)
        #     outcome = current.copy()

        #     for key in update:
        #         if not key in outcome or is_schema_key(update, key):
        #             key_update = update[key]
        #             if key_update:
        #                 outcome[key] = key_update
        #         else:
        #             outcome[key] = self.resolve_schemas(
        #                 outcome.get(key),
        #                 update[key])

        return outcome


    def check_state(self, schema, state):
        schema = self.access(schema)

        check_function = self.choose_method(
            schema,
            state,
            'check')

        return check_function(
            schema,
            state,
            self)


    def check(self, initial_schema, state):
        schema = self.retrieve(initial_schema)
        return self.check_state(schema, state)
    

    def fold_state(self, schema, state, method, values):
        schema = self.access(schema)

        fold_function = self.choose_method(
            schema,
            state,
            'fold')

        return fold_function(
            schema,
            state,
            method,
            values,
            self)


    def fold(self, initial_schema, state, method, values=None):
        schema = self.retrieve(initial_schema)
        return self.fold_state(
            schema,
            state,
            method,
            values)


    def validate(self, schema, state):
        # TODO:
        #   go through the state using the schema and
        #   return information about what doesn't match

        return {}


    def apply_update(self, schema, state, update):
        if isinstance(update, dict) and '_react' in update:
            new_state = self.react(
                schema,
                state,
                update['_react'])

            state = self.deserialize(schema, new_state)

        elif isinstance(update, dict) and '_fold' in update:
            fold = update['_fold']

            if isinstance(fold, dict):
                method = fold['method']
                values = {
                    key: value
                    for key, value in fold.items()
                    if key != 'method'}

            elif isinstance(fold, str):
                method = fold
                values = {}

            else:
                raise Exception(f'unknown fold: {pf(update)}')

            state = self.fold(
                schema,
                state,
                method,
                values)

        elif '_apply' in schema and schema['_apply'] != 'any':
            apply_function = self.apply_registry.access(schema['_apply'])
            
            state = apply_function(
                schema,
                state,
                update,
                self)

        elif isinstance(schema, str) or isinstance(schema, list):
            schema = self.access(schema)
            state = self.apply_update(schema, state, update)

        elif isinstance(update, dict):
            for key, branch in update.items():
                if key not in schema:
                    raise Exception(
                        f'trying to update a key that is not in the schema'
                        f'for state: {key}\n{state}\nwith schema:\n{schema}')
                else:
                    subupdate = self.apply_update(
                        self.access(schema[key]),
                        state[key],
                        branch)

                    state[key] = subupdate
        else:
            raise Exception(
                f'trying to apply update\n  {update}\nto state\n  {state}\n'
                f'with schema\n  {schema}\nbut the update is not a dict')

        return state


    def apply(self, original_schema, initial, update):
        schema = self.access(original_schema)
        state = copy.deepcopy(initial)
        return self.apply_update(schema, state, update)


    def apply_slice(self, schema, state, path, update):
        path = path or ()
        if len(path) == 0:
            result = self.apply(
                schema,
                state,
                update)

        else:
            subschema, substate = self.slice(
                schema,
                state,
                path[0])

            if len(path) == 1:
                subresult = self.apply(
                    subschema,
                    substate,
                    update)

                result = self.bind(
                    schema,
                    state,
                    path[1:],
                    subschema,
                    subresult)

            else:
                subresult = self.apply_slice(
                    subschema,
                    substate,
                    path[1:],
                    update)

                result = state

        return result


    def set_update(self, schema, state, update):
        if '_apply' in schema:
            apply_function = self.apply_registry.access('set')
            
            state = apply_function(
                schema,
                state,
                update,
                self)

        elif isinstance(schema, str) or isinstance(schema, list):
            schema = self.access(schema)
            state = self.set_update(schema, state, update)

        elif isinstance(update, dict):
            for key, branch in update.items():
                if key not in schema:
                    raise Exception(
                        f'trying to update a key that is not in the schema'
                        f'for state: {key}\n{state}\nwith schema:\n{schema}')
                else:
                    subupdate = self.set_update(
                        schema[key],
                        state[key],
                        branch)

                    state[key] = subupdate
        else:
            raise Exception(
                f'trying to apply update\n  {update}\nto state\n  {state}\n'
                f'with schema\n{schema}, but the update is not a dict')

        return state


    def set(self, original_schema, initial, update):
        schema = self.access(original_schema)
        state = copy.deepcopy(initial)

        return self.set_update(schema, state, update)


    def merge(self, schema, current_state, new_state):
        schema = self.access(schema)

        merge_function = self.choose_method(
            schema,
            new_state,
            'merge')

        return merge_function(
            schema,
            current_state,
            new_state,
            self)


    def bind(self, schema, state, key, target_schema, target_state):
        schema = self.retrieve(schema)

        bind_function = self.choose_method(
            schema,
            state,
            'bind')

        return bind_function(
            schema,
            state,
            key,
            target_schema,
            target_state,
            self)


    def set_slice(self, schema, state, path, target_schema, target_state, defer=False):
        if len(path) == 0:
            return schema, self.merge(
                schema,
                state,
                target_state)

        elif len(path) == 1:
            key = path[0]
            destination_schema, destination_state = self.slice(
                schema,
                state,
                key)

            final_schema = self.resolve_schemas(
                destination_schema,
                target_schema)

            if not defer:
                result_state = self.merge(
                    final_schema,
                    destination_state,
                    target_state)

            else:
                result_state = self.merge(
                    final_schema,
                    target_state,
                    destination_state)

            return self.bind(
                schema,
                state,
                key,
                final_schema,
                result_state)

        else:
            path = resolve_path(path)

            head = path[0]
            tail = path[1:]
            
            down_schema, down_state = self.slice(
                schema,
                state,
                head)

            result_schema, result_state = self.set_slice(
                down_schema,
                down_state,
                tail,
                target_schema,
                target_state,
                defer=defer)

            return self.bind(
                schema,
                state,
                head,
                result_schema,
                result_state)


    def serialize(self, schema, state):
        schema = self.retrieve(schema)

        serialize_function = self.choose_method(
            schema,
            state,
            'serialize')

        return serialize_function(
            schema,
            state,
            self)


    def deserialize(self, schema, state):
        schema = self.retrieve(schema)

        deserialize_function = self.choose_method(
            schema,
            state,
            'deserialize')

        return deserialize_function(
            schema,
            state,
            self)


    def fill_ports(self, interface, wires=None, state=None, top_schema=None, top_state=None, path=None):
        # deal with wires
        if wires is None:
            wires = {}
        if state is None:
            state = {}
        if top_schema is None:
            top_schema = schema
        if top_state is None:
            top_state = state
        if path is None:
            path = []

        if isinstance(interface, str):
            interface = {'_type': interface}

        for port_key, subwires in wires.items():
            if port_key in interface:
                port_schema = interface[port_key]
            else:
                port_schema, subwires = self.slice(
                    interface,
                    wires,
                    port_key)

            if isinstance(subwires, dict):
                if isinstance(state, dict):
                    state = self.fill_ports(
                        port_schema,
                        wires=subwires,
                        state=state,
                        top_schema=top_schema,
                        top_state=top_state,
                        path=path)

            else:
                if isinstance(subwires, str):
                    subwires = [subwires]

                subschema, substate = self.set_slice(
                    top_schema,
                    top_state,
                    path[:-1] + subwires,
                    port_schema,
                    self.default(port_schema),
                    defer=True)

        return state


    def fill_state(self, schema, state=None, top_schema=None, top_state=None, path=None, type_key=None, context=None):
        # if a port is disconnected, build a store
        # for it under the '_open' key in the current
        # node (?)

        # inform the user that they have disconnected
        # ports somehow

        if schema is None:
            return None
        if state is None:
            state = self.default(schema)
        if top_schema is None:
            top_schema = schema
        if top_state is None:
            top_state = state
        if path is None:
            path = []

        if '_inputs' in schema:
            inputs = state.get('inputs', {})
            state = self.fill_ports(
                schema['_inputs'],
                wires=inputs,
                state=state,
                top_schema=top_schema,
                top_state=top_state,
                path=path)

        if '_outputs' in schema:
            outputs = state.get('outputs', {})
            state = self.fill_ports(
                schema['_outputs'],
                wires=outputs,
                state=state,
                top_schema=top_schema,
                top_state=top_state,
                path=path)

        if isinstance(schema, str):
            schema = self.access(schema)

        branches = non_schema_keys(schema)

        if isinstance(state, dict):
            for branch in branches:
                subpath = path + [branch]
                state[branch] = self.fill_state(
                    schema[branch],
                    state=state.get(branch),
                    top_schema=top_schema,
                    top_state=top_state,
                    path=subpath)

            state_keys = non_schema_keys(state)
            for key in set(state_keys) - set(branches):
                subschema, substate = self.slice(
                    schema,
                    state,
                    key)

                subpath = path + [key]
                state[key] = self.fill_state(
                    subschema,
                    substate,
                    top_schema=top_schema,
                    top_state=top_state,
                    path=subpath)

        return state


    def fill(self, original_schema, state=None):
        schema = self.access(original_schema)

        return self.fill_state(
            schema,
            state=state)


    def ports_schema(self, schema, instance, edge_path, ports_key='inputs'):
        found = self.access(schema)

        edge_schema, edge_state = self.slice(
            schema,
            instance,
            edge_path)

        ports_schema = edge_state.get(f'_{ports_key}', 
                                      edge_schema.get(f'_{ports_key}'))
        ports = edge_state.get(ports_key)
        
        return ports_schema, ports


    def view(self, schema, wires, path, top_schema=None, top_state=None):
        result = {}

        if isinstance(wires, str):
            wires = [wires]

        if isinstance(wires, (list, tuple)):
            _, result = self.slice(
                top_schema,
                top_state,
                list(path) + list(wires))

        elif isinstance(wires, dict):
            result = {}
            for port_key, port_path in wires.items():
                subschema, _ = self.slice(
                    schema,
                    {},
                    port_key)

                inner_view = self.view(
                    subschema,
                    port_path,
                    path,
                    top_schema=top_schema,
                    top_state=top_state)

                if inner_view is not None:
                    result[port_key] = inner_view
        else:
            raise Exception(f'trying to project state with these ports:\n{schema}\nbut not sure what these wires are:\n{wires}')

        return result


    def view_edge(self, schema, state, edge_path=None, ports_key='inputs'):
        """
        project the current state into a form the edge expects, based on its ports.
        """

        if schema is None:
            return None
        if state is None:
            state = self.default(schema)
        if edge_path is None:
            edge_path = []

        ports_schema, ports = self.ports_schema(
            schema,
            state,
            edge_path=edge_path,
            ports_key=ports_key)

        if not ports_schema:
            return None
        if ports is None:
            return None

        return self.view(
            ports_schema,
            ports,
            edge_path[:-1],
            top_schema=schema,
            top_state=state)


    def project(self, ports, wires, path, states):
        result = {}

        if isinstance(wires, str):
            wires = [wires]

        if isinstance(wires, (list, tuple)):
            destination = resolve_path(list(path) + list(wires))
            result = set_path(
                result,
                destination,
                states)

        elif isinstance(wires, dict):
            if isinstance(states, list):
                result = [
                    self.project(ports, wires, path, state)
                    for state in states]
            else:
                branches = []
                for key in wires.keys():
                    subports, substates = self.slice(ports, states, key)
                    projection = self.project(
                        subports,
                        wires[key],
                        path,
                        substates)
                
                    if projection is not None:
                        branches.append(projection)

                branches = [
                    branch
                    for branch in branches
                    if branch is not None]

                result = {}
                for branch in branches:
                    deep_merge(result, branch)
        else:
            raise Exception(
                f'inverting state\n  {states}\naccording to ports schema\n  {ports}\nbut wires are not recognized\n  {wires}')

        return result


    def project_edge(self, schema, instance, edge_path, states, ports_key='outputs'):
        """
        Given states from the perspective of an edge (through its ports), produce states aligned to the tree
        the wires point to.
        (inverse of view)
        """

        if schema is None:
            return None
        if instance is None:
            instance = self.default(schema)

        ports_schema, ports = self.ports_schema(
            schema,
            instance,
            edge_path,
            ports_key)

        if ports_schema is None:
            return None
        if ports is None:
            return None

        return self.project(
            ports_schema,
            ports,
            edge_path[:-1],
            states)


    def equivalent(self, icurrent, iquestion):
        if icurrent == iquestion:
            return True

        current = self.access(icurrent)
        question = self.access(iquestion)

        if current is None:
            return question is None

        if current == {}:
            return question == {}

        elif '_type' in current:
            if '_type' in question:
                if current['_type'] == question['_type']:
                    if '_type_parameters' in current:
                        if '_type_parameters' in question:
                            for parameter in current['_type_parameters']:
                                parameter_key = f'_{parameter}'
                                if parameter_key in question:
                                    if not self.equivalent(current[parameter_key], question[parameter_key]):
                                        return False
                        else:
                            return False
                else:
                    return False
            else:
                return False
        else:
            if '_type' in question:
                return False

        for key, value in current.items():
            if not is_schema_key(key): # key not in type_schema_keys:
                if key not in question or not self.equivalent(current.get(key), question[key]):
                    return False

        for key in set(question.keys()) - set(current.keys()):
            if not is_schema_key(key): # key not in type_schema_keys:
                if key not in question or not self.equivalent(current.get(key), question[key]):
                    return False

        return True


    def inherits_from(self, descendant, ancestor):
        descendant = self.access(descendant)
        ancestor = self.access(ancestor)

        if descendant == {}:
            return ancestor == {}

        if descendant is None:
            return True

        if ancestor is None:
            return False

        if isinstance(ancestor, int):
            if isinstance(descendant, int):
                return ancestor == descendant
            else:
                return False

        elif isinstance(ancestor, list):
            if isinstance(descendant, list):
                if len(ancestor) == len(descendant):
                    for a, d in zip(ancestor, descendant):
                        if not self.inherits_from(d, a):
                            return False
                else:
                    return False

        elif '_type' in ancestor and ancestor['_type'] == 'any':
            return True

        elif '_type' in descendant:
            if '_inherit' in descendant:
                for inherit in descendant['_inherit']:

                    if self.equivalent(inherit, ancestor) or self.inherits_from(inherit, ancestor):
                        return True

                return False

            elif '_type_parameters' in descendant:
                for type_parameter in descendant['_type_parameters']:
                    parameter_key = f'_{type_parameter}'
                    if parameter_key in ancestor and parameter_key in descendant:
                        if not self.inherits_from(descendant[parameter_key], ancestor[parameter_key]):
                            return False

            if '_type' not in ancestor or descendant['_type'] != ancestor['_type']:
                return False

        else:
            for key, value in ancestor.items():
                if key not in type_schema_keys:
                # if not key.startswith('_'):
                    if key in descendant:
                        if not self.inherits_from(descendant[key], value):
                            return False
                    else:
                        return False

        return True


    # def infer_wires(self, ports, state, wires, top_schema=None, top_state=None, path=None, internal_path=None):
    def infer_wires(self, ports, wires, top_schema=None, top_state=None, path=None, internal_path=None):
        top_schema = top_schema or {}
        top_state = top_state or state
        path = path or ()
        internal_path = internal_path or ()

        if isinstance(ports, str):
            ports = self.access(ports)

        if isinstance(wires, (list, tuple)):
            if len(wires) == 0:
                destination_schema, destination_state = top_schema, top_state

            else:
                destination_schema, destination_state = self.slice(
                    top_schema,
                    top_state,
                    path[:-1] + wires)

            merged_schema = apply_schema(
                'schema',
                destination_schema,
                ports,
                self)

            merged_state = self.complete(
                merged_schema,
                destination_state)

        else:
            for port_key, port_wires in wires.items():
                subschema, substate = self.slice(
                    ports,
                    {},
                    port_key)

                if isinstance(port_wires, dict):
                    top_schema, top_state = self.infer_wires(
                        subschema,
                        # substate,
                        port_wires,
                        top_schema=top_schema,
                        top_state=top_state,
                        path=path,
                        internal_path=internal_path+(port_key,))

                # port_wires must be a list
                elif len(port_wires) == 0:
                    raise Exception(f'no wires at port "{port_key}" in ports {ports} with state {state}')

                else:
                    compound_path = resolve_path(
                        path[:-1] + tuple(port_wires))

                    compound_schema, compound_state = self.set_slice(
                        {}, {},
                        compound_path,
                        subschema or 'any',
                        self.default(subschema))

                    top_schema = self.resolve(
                        top_schema,
                        compound_schema)

                    top_state = self.merge(
                        top_schema,
                        compound_state,
                        top_state)

        return top_schema, top_state


    def infer_edge(self, schema, state, top_schema=None, top_state=None, path=None):
        '''
        given the schema and state for this edge, and its path relative to
        the top_schema and top_state, make all the necessary completions to
        both the schema and the state according to the input and output schemas
        of this edge in '_inputs' and '_outputs', along the wires in its state
        under 'inputs' and 'outputs'.

        returns the top_schema and top_state, even if the edge is deeply embedded,
        as the particular wires could have implications anywhere in the tree.
        '''

        schema = schema or {}
        top_schema = top_schema or schema
        top_state = top_state or state
        path = path or ()

        if self.check('edge', state):
            for port_key in ['inputs', 'outputs']:
                ports = state.get(port_key)
                schema_key = f'_{port_key}'
                port_schema = schema.get(schema_key, {})
                state_schema = state.get(schema_key, {})

                schema[schema_key] = self.resolve(
                    port_schema,
                    self.access(
                        state_schema))

                if ports:
                    top_schema, top_state = self.infer_wires(
                        schema[schema_key],
                        # state,
                        ports,
                        top_schema=top_schema,
                        top_state=top_state,
                        path=path)

        return top_schema, top_state


    def infer_schema(self, schema, state, top_schema=None, top_state=None, path=None):
        """
        Given a schema fragment and an existing state with _type keys,
        return the full schema required to describe that state,
        and whatever state was hydrated (edges) during this process

        """

        # during recursive call, schema is kept at the top level and the 
        # path is used to access it (!)

        schema = schema or {}
        top_schema = top_schema or schema
        top_state = top_state or state
        path = path or ()

        if isinstance(state, dict):
            state_schema = None
            if '_type' in state:
                state_type = {
                    key: value
                    for key, value in state.items()
                    if is_schema_key(key)}

                schema = self.resolve(
                    schema,
                    state_type)

            if '_type' in schema:
                hydrated_state = self.deserialize(
                    schema,
                    state)

                top_schema, top_state = self.set_slice(
                    top_schema,
                    top_state,
                    path,
                    schema,
                    hydrated_state)

                top_schema, top_state = self.infer_edge(
                    schema,
                    hydrated_state,
                    top_schema,
                    top_state,
                    path)

            else:
                for key in state:
                    inner_path = path + (key,)
                    inner_schema, inner_state = self.slice(
                        schema,
                        state,
                        key)

                    top_schema, top_state = self.infer_schema(
                        inner_schema,
                        inner_state,
                        top_schema=top_schema,
                        top_state=top_state,
                        path=inner_path)

        elif isinstance(state, str):
            pass

        else:
            type_schema = TYPE_SCHEMAS.get(
                type(state).__name__,
                'any')

            top_schema, top_state = self.set_slice(
                top_schema,
                top_state,
                path,
                type_schema,
                state)

        return top_schema, top_state
        

    def hydrate(self, schema, state):
        hydrated = self.deserialize(schema, state)
        return self.fill(schema, hydrated)


    def complete(self, initial_schema, initial_state):
        full_schema = self.access(
            initial_schema)

        state = self.deserialize(
            full_schema,
            initial_state)

        # fill in the parts of the composition schema
        # determined by the state
        schema, state = self.infer_schema(
            full_schema,
            state)

        final_state = self.fill(schema, state)

        # TODO: add flag to types.access(copy=True)
        return self.access(schema), final_state
        

    def generate_recur(self, schema, state, top_schema=None, top_state=None, path=None):
        merged_schema, merged_state = self.merge_schema_keys(
            schema,
            state)

        found = self.retrieve(
            merged_schema)

        generate_function = self.choose_method(
            found,
            merged_state,
            'generate')

        return generate_function(
            self,
            found,
            merged_state,
            top_schema=top_schema,
            top_state=top_state,
            path=path)


    def generate(self, schema, state):
        _, _, top_schema, top_state = self.generate_recur(
            schema,
            state)

        return top_schema, top_state


    def find_method(self, schema, method_key):
        if not isinstance(schema, dict) or method_key not in schema:
            schema = self.access(schema)

        if method_key in schema:
            registry = self.lookup_registry(
                method_key)

            if registry is not None:
                method_name = schema[method_key]
                method = registry.access(method_name)

                return method


    def import_types(self, package, strict=False):
        for type_key, type_data in package.items():
            if not (strict and self.exists(type_key)):
                self.register(
                    type_key,
                    type_data)


    def define(self, method_name, methods):
        method_key = f'_{method_name}'
        for type_key, method in methods.items():
            self.register(
                type_key,
                {method_key: method})


    def link_place(self, place, link):
        pass


    def compose(self, a, b):
        pass


    def query(self, schema, instance, redex):
        subschema = {}
        return subschema


def check_number(schema, state, core=None):
    return isinstance(state, numbers.Number)

def check_boolean(schema, state, core=None):
    return isinstance(state, bool)

def check_integer(schema, state, core=None):
    return isinstance(state, int) and not isinstance(state, bool)

def check_float(schema, state, core=None):
    return isinstance(state, float)

def check_string(schema, state, core=None):
    return isinstance(state, str)


class Edge:
    def __init__(self):
        pass


    def inputs(self):
        return {}


    def outputs(self):
        return {}


    def interface(self):
        """Returns the schema for this type"""
        return {
            'inputs': self.inputs(),
            'outputs': self.outputs()}


#################
# Apply methods #
#################

def apply_boolean(schema, current: bool, update: bool, core=None) -> bool:
    """Performs a bit flip if `current` does not match `update`, returning update. Returns current if they match."""
    if current != update:
        return update
    else:
        return current


def serialize_boolean(schema, value: bool, core) -> str:
    return str(value)


def deserialize_boolean(schema, encoded, core) -> bool:
    if encoded == 'true':
        return True
    elif encoded == 'false':
        return False
    elif encoded == True or encoded == False:
        return encoded


def accumulate(schema, current, update, core):
    if current is None:
        return update
    if update is None:
        return current
    else:
        return current + update


def set_apply(schema, current, update, core):
    if isinstance(current, dict) and isinstance(update, dict):
        for key, value in update.items():
            # TODO: replace this with type specific functions (??)
            if key in schema:
                subschema = schema[key]
            elif '_leaf' in schema:
                if core.check(schema['_leaf'], value):
                    subschema = schema['_leaf']
                else:
                    subschema = schema
            elif '_value' in schema:
                subschema = schema['_value']

            current[key] = set_apply(
                subschema,
                current.get(key),
                value,
                core)

        return current

    else:
        return update        


def concatenate(schema, current, update, core=None):
    return current + update


def dataclass_float(schema, path, core):
    return float


def dataclass_integer(schema, path, core):
    return int


def divide_float(schema, state, values, core):
    divisions = values.get('divisions', 2)
    portion = float(state) / divisions
    return [
        portion
        for _ in range(divisions)]


# ##################
# # Divide methods #
# ##################
# # support dividing by ratios?
# # ---> divide_float({...}, [0.1, 0.3, 0.6])

# def divide_float(schema, value, ratios, core=None):
#     half = value / 2.0
#     return (half, half)


# support function core for registrys?
def divide_integer(schema, value, values, core):
    half = value // 2
    other_half = half
    if value % 2 == 1:
        other_half += 1
    return [half, other_half]


def divide_longest(schema, dimensions, values, core):
    # any way to declare the required keys for this function in the registry?
    # find a way to ask a function what type its domain and codomain are

    width = dimensions['width']
    height = dimensions['height']

    if width > height:
        a, b = divide_integer(width)
        return [{'width': a, 'height': height}, {'width': b, 'height': height}]
    else:
        x, y = divide_integer(height)
        return [{'width': width, 'height': x}, {'width': width, 'height': y}]


def replace(schema, current, update, core=None):
    return update


def serialize_string(schema, value, core=None):
    return value


def deserialize_string(schema, encoded, core=None):
    if isinstance(encoded, str):
        return encoded


def to_string(schema, value, core=None):
    return str(value)


def deserialize_integer(schema, encoded, core=None):
    value = None
    try:
        value = int(encoded)
    except:
        pass

    return value


def deserialize_float(schema, encoded, core=None):
    value = None
    try:
        value = float(encoded)
    except:
        pass

    return value


def evaluate(schema, encoded, core=None):
    return eval(encoded)


def apply_list(schema, current, update, core):
    element_type = core.find_parameter(
        schema,
        'element')

    if current is None:
        current = []

    if core.check(element_type, update):
        result = current + [update]
        return result

    elif isinstance(update, list):
        result = current + update
        # for current_element, update_element in zip(current, update):
        #     applied = core.apply(
        #         element_type,
        #         current_element,
        #         update_element)

        #     result.append(applied)

        return result
    else:
        raise Exception(f'trying to apply an update to an existing list, but the update is not a list or of element type:\n  update: {update}\n  element type: {pf(element_type)}')


def check_list(schema, state, core):
    element_type = core.find_parameter(
        schema,
        'element')

    if isinstance(state, list):
        for element in state:
            check = core.check(
                element_type,
                element)

            if not check:
                return False

        return True
    else:
        return False


def dataclass_list(schema, path, core):
    element_type = core.find_parameter(
        schema,
        'element')

    dataclass = core.dataclass(
        element_type,
        path + ['element'])

    return list[dataclass]


def slice_list(schema, state, path, core):
    element_type = core.find_parameter(
        schema,
        'element')

    if len(path) > 0:
        head = path[0]
        tail = path[1:]

        if not isinstance(head, int) or head >= len(state):
            raise Exception(f'bad index for list: {path} for {state}')

        step = state[head]
        return core.slice(element_type, step, tail)
    else:
        return schema, state


def serialize_list(schema, value, core=None):
    element_type = core.find_parameter(
        schema,
        'element')

    return [
        core.serialize(
            element_type,
            element)
        for element in value]


def deserialize_list(schema, encoded, core=None):
    if isinstance(encoded, list):
        element_type = core.find_parameter(
            schema,
            'element')

        return [
            core.deserialize(
                element_type,
                element)
            for element in encoded]


def check_tree(schema, state, core):
    leaf_type = core.find_parameter(
        schema,
        'leaf')

    if isinstance(state, dict):
        for key, value in state.items():
            check = core.check({
                '_type': 'tree',
                '_leaf': leaf_type},
                value)

            if not check:
                return core.check(
                    leaf_type,
                    value)

        return True
    else:
        return core.check(leaf_type, state)


def dataclass_tree(schema, path, core):
    leaf_type = core.find_parameter(
        schema,
        'leaf')

    leaf_dataclass = core.dataclass(
        leaf_type,
        path + ['leaf'])

    dataclass_name = '_'.join(path)
    # block = f"{dataclass_name} = NewType('{dataclass_name}', Union[{leaf_dataclass}, Mapping[str, '{dataclass_name}']])"
    block = f"NewType('{dataclass_name}', Union[{leaf_dataclass}, Mapping[str, '{dataclass_name}']])"

    dataclass = eval(block)
    setattr(data, dataclass_name, dataclass)

    return dataclass


def slice_tree(schema, state, path, core):
    leaf_type = core.find_parameter(
        schema,
        'leaf')

    if len(path) > 0:
        head = path[0]
        tail = path[1:]

        if not head in state:
            state[head] = {}

        step = state[head]
        if core.check(leaf_type, step):
            return core.slice(leaf_type, step, tail)
        else:
            return core.slice(schema, step, tail)
    else:
        return schema, state


def serialize_tree(schema, value, core):
    if isinstance(value, dict):
        encoded = {}
        for key, subvalue in value.items():
            encoded[key] = serialize_tree(
                schema,
                subvalue,
                core)

    else:
        leaf_type = core.find_parameter(
            schema,
            'leaf')

        if core.check(leaf_type, value):
            encoded = core.serialize(
                leaf_type,
                value)
        else:
            raise Exception(f'trying to serialize a tree but unfamiliar with this form of tree: {value} - current schema:\n {pf(schema)}')

    return encoded


def deserialize_tree(schema, encoded, core):
    if isinstance(encoded, dict):
        tree = {}
        for key, value in encoded.items():
            if key.startswith('_'):
                tree[key] = value
            else:
                tree[key] = deserialize_tree(schema, value, core)

        return tree

    else:
        leaf_type = core.find_parameter(
            schema,
            'leaf')

        if leaf_type:
            return core.deserialize(
                leaf_type,
                encoded)
        else:
            return encoded


def apply_map(schema, current, update, core=None):
    if not isinstance(current, dict):
        raise Exception(f'trying to apply an update to a value that is not a map:\n  value: {current}\n  update: {update}')
    if not isinstance(update, dict):
        raise Exception(f'trying to apply an update that is not a map:\n  value: {current}\n  update: {update}')

    value_type = core.find_parameter(
        schema,
        'value')

    result = current.copy()

    for key, update_value in update.items():
        if key == '_add':
            import ipdb; ipdb.set_trace()

            for addition_key, addition in update_value.items():
                filled = core.hydrate(
                    value_type,
                    addition)

                result[addition_key] = filled

        elif key == '_remove':
            for remove_key in update_value:
                if remove_key in result:
                    del result[remove_key]

        elif key not in current:
            # This supports adding without the '_add' key, if the key is not in the state
            filled = core.hydrate(
                value_type,
                update_value)
            result[key] = filled

            # raise Exception(f'trying to update a key that does not exist:\n  value: {current}\n  update: {update}')
        else:
            result[key] = core.apply(
                value_type,
                result[key],
                update_value)

    return result


def resolve_map(schema, update, core):
    if isinstance(update, dict):
        value_schema = update.get(
            '_value',
            schema.get('_value', {}))

        for key, subschema in update.items():
            if not is_schema_key(key):
                value_schema = core.resolve_schemas(
                    value_schema,
                    subschema)

        schema['_type'] = update.get(
            '_type',
            schema.get('_type', 'map'))
        schema['_value'] = value_schema

    return schema


def resolve_array(schema, update, core):
    if not '_shape' in schema:
        schema = core.access(schema)
    if not '_shape' in schema:
        raise Exception(f'array must have a "_shape" key, not {schema}')

    data_schema = schema.get('_data', {})

    if '_type' in update:
        data_schema = core.resolve_schemas(
            data_schema,
            update.get('_data', {}))

        if update['_type'] == 'array':
            if '_shape' in update:
                if update['_shape'] != schema['_shape']:
                    raise Exception(f'arrays must be of the same shape, not \n  {schema}\nand\n  {update}')

        elif core.inherits_from(update, schema):
            schema.update(update)

        elif not core.inherits_from(schema, update):
            raise Exception(f'cannot resolve incompatible array schemas:\n  {schema}\n  {update}')

    else:
        for key, subschema in update.items():
            if isinstance(key, int):
                key = (key,)

            if len(key) > len(schema['_shape']):
                raise Exception(f'key is longer than array dimension: {key}\n{schema}\n{update}')
            elif len(key) == len(schema['_shape']):
                data_schema = core.resolve_schemas(
                    data_schema,
                    subschema)
            else:
                shape = tuple_from_type(
                    schema['_shape'])

                subshape = shape[len(key):]
                inner_schema = schema.copy()
                inner_schema['_shape'] = subshape
                inner_schema = core.resolve_schemas(
                    inner_schema,
                    subschema)

                data_schema = inner_schema['_data']

    schema['_data'] = data_schema

    return schema


def tuple_from_type(tuple_type):
    if isinstance(tuple_type, tuple):
        return tuple_type

    elif isinstance(tuple_type, list):
        return tuple(tuple_type)

    elif isinstance(tuple_type, dict):
        tuple_list = [
            tuple_type[f'_{parameter}']
            for parameter in tuple_type['_type_parameters']]

        return tuple(tuple_list)

    else:
        raise Exception(f'do not recognize this type as a tuple: {tuple_type}')


# def resolve_tree(schema, update, core):
#     import ipdb; ipdb.set_trace()

#     if isinstance(update, dict):
#         leaf_schema = schema.get('_leaf', {})

#         if '_type' in update:
#             if update['_type'] == 'map':
#                 value_schema = update.get('_value', {})
#                 leaf_schema = core.resolve_schemas(
#                     leaf_schema,
#                     value_schema)

#             elif update['_type'] == 'tree':
#                 for key, subschema in update.items():
#                     if not key.startswith('_'):
#                         leaf_schema = core.resolve_schemas(
#                             leaf_schema,
#                             subschema)
#             else:
#                 leaf_schema = core.resolve_schemas(
#                     leaf_schema,
#                     update)

#             schema['_leaf'] = leaf_schema
#         else:
#             for key, subupdate in 

#     return schema



def dataclass_map(schema, path, core):
    value_type = core.find_parameter(
        schema,
        'value')

    dataclass = core.dataclass(
        value_type,
        path + ['value'])
    
    return Mapping[str, dataclass]


def check_map(schema, state, core=None):
    value_type = core.find_parameter(
        schema,
        'value')

    if not isinstance(state, dict):
        return False

    for key, substate in state.items():
        if not core.check(value_type, substate):
            return False

    return True


def slice_map(schema, state, path, core):
    value_type = core.find_parameter(
        schema,
        'value')

    if len(path) > 0:
        head = path[0]
        tail = path[1:]

        if not head in state:
            state[head] = core.default(
                value_type)

        step = state[head]
        return core.slice(
            value_type,
            step,
            tail)
    else:
        return schema, state


def serialize_map(schema, value, core=None):
    value_type = core.find_parameter(
        schema,
        'value')

    return {
        key: core.serialize(
            value_type,
            subvalue) if not is_schema_key(key) else subvalue
        for key, subvalue in value.items()}


def deserialize_map(schema, encoded, core=None):
    if isinstance(encoded, dict):
        value_type = core.find_parameter(
            schema,
            'value')

        return {
            key: core.deserialize(
                value_type,
                subvalue) if not is_schema_key(key) else subvalue
            for key, subvalue in encoded.items()}


def apply_maybe(schema, current, update, core):
    if current is None or update is None:
        return update
    else:
        value_type = core.find_parameter(
            schema,
            'value')

        return core.apply(
            value_type,
            current,
            update)


def dataclass_maybe(schema, path, core):
    value_type = core.find_parameter(
        schema,
        'value')

    dataclass = core.dataclass(
        value_type,
        path + ['value'])
    
    return Optional[dataclass]


def check_maybe(schema, state, core):
    if state is None:
        return True
    else:
        value_type = core.find_parameter(
            schema,
            'value')

        return core.check(value_type, state)


def slice_maybe(schema, state, path, core):
    if state is None:
        return schema, None

    else:
        value_type = core.find_parameter(
            schema,
            'value')

        return core.slice(
            value_type,
            state,
            path)


def serialize_maybe(schema, value, core):
    if value is None:
        return NONE_SYMBOL
    else:
        value_type = core.find_parameter(
            schema,
            'value')

        return core.serialize(
            value_type,
            value)


def deserialize_maybe(schema, encoded, core):
    if encoded == NONE_SYMBOL or encoded is None:
        return None
    else:
        value_type = core.find_parameter(
            schema,
            'value')

        return core.deserialize(value_type, encoded)


# TODO: deal with all the different unit core
def apply_units(schema, current, update, core):
    return current + update


def check_units(schema, state, core):
    # TODO: expand this to check the actual units for compatibility
    return isinstance(state, Quantity)


def serialize_units(schema, value, core):
    return str(value)


def deserialize_units(schema, encoded, core):
    if isinstance(encoded, Quantity):
        return encoded
    else:
        return units(encoded)


def apply_path(schema, current, update, core):
    # paths replace previous paths
    return update


def generate_tree(core, schema, state, top_schema=None, top_state=None, path=None):
    schema = schema or {}
    state = state or core.default(schema)
    top_schema = top_schema or schema
    top_state = top_state or state
    path = path or []

    leaf_type = core.find_parameter(
        schema,
        'leaf')

    if core.check(state, leaf_type):
        generate_schema, generate_state, top_schema, top_state = core.generate_recur(
            leaf_type,
            state,
            top_schema=top_schema,
            top_state=top_state,
            path=path)
    else:
        generate_schema = {}
        generate_state = {}

        all_keys = set(schema.keys()).union(state.keys())
        non_schema_keys = [
            key
            for key in all_keys
            if not is_schema_key(key)]

        if non_schema_keys:
            base_schema = {
                key: subschema
                for key, subschema in schema.items()
                if is_schema_key(key)}
        else:
            base_schema = schema

        for key in all_keys:
            if not is_schema_key(key):
                subschema = schema.get(key)
                substate = state.get(key)

                if not substate or core.check(leaf_type, substate):
                    base_schema = leaf_type

                subschema = core.merge_schemas(
                    base_schema,
                    subschema)

                subschema, generate_state[key], top_schema, top_state = core.generate_recur(
                    subschema,
                    substate,
                    top_schema=top_schema,
                    top_state=top_state,
                    path=path+[key])

            elif key in state:
                generate_schema[key] = state[key]
            elif key in schema:
                generate_schema[key] = schema[key]
            else:
                raise Exception(' the impossible has occurred now is the time for celebration')

    return generate_schema, generate_state, top_schema, top_state


def generate_ports(core, schema, wires, top_schema=None, top_state=None, path=None):
    schema = schema or {}
    wires = wires or {}
    top_schema = top_schema or schema
    top_state = top_state or {}
    path = path or []

    if isinstance(schema, str):
        schema = {'_type': schema}

    for port_key, subwires in wires.items():
        if port_key in schema:
            port_schema = schema[port_key]
        else:
            port_schema, subwires = core.slice(
                schema,
                wires,
                port_key)

        if isinstance(subwires, dict):
            top_schema, top_state = core.generate_ports(
                port_schema,
                subwires,
                top_schema=top_schema,
                top_state=top_state,
                path=path)

        else:
            if isinstance(subwires, str):
                subwires = [subwires]

            default_state = core.default(
                port_schema)

            top_schema, top_state = core.set_slice(
                top_schema,
                top_state,
                path[:-1] + subwires,
                port_schema,
                default_state,
                defer=True)
            
    return top_schema, top_state


def generate_edge(core, schema, state, top_schema=None, top_state=None, path=None):
    schema = schema or {}
    state = state or {}
    top_schema = top_schema or schema
    top_state = top_state or state
    path = path or []

    generate_schema, generate_state, top_schema, top_state = generate_any(
        core,
        schema,
        state,
        top_schema=top_schema,
        top_state=top_state,
        path=path)

    top_schema, top_state = core.set_slice(
        top_schema,
        top_state,
        path,
        generate_schema,
        generate_state)

    for port_key in ['inputs', 'outputs']:
        port_schema = generate_schema.get(
            f'_{port_key}', {})
        ports = generate_state.get(
            port_key, {})

        top_schema, top_state = generate_ports(
            core,
            port_schema,
            ports,
            top_schema=top_schema,
            top_state=top_state,
            path=path)

    return generate_schema, generate_state, top_schema, top_state


def apply_edge(schema, current, update, core):
    result = current.copy()
    result['inputs'] = core.apply(
        'wires',
        current.get('inputs'),
        update.get('inputs'))

    result['outputs'] = core.apply(
        'wires',
        current.get('outputs'),
        update.get('outputs'))

    return result


def dataclass_edge(schema, path, core):
    inputs = schema.get('_inputs', {})
    inputs_dataclass = core.dataclass(
        inputs,
        path + ['inputs'])

    outputs = schema.get('_outputs', {})
    outputs_dataclass = core.dataclass(
        outputs,
        path + ['outputs'])

    return Callable[[inputs_dataclass], outputs_dataclass]


def check_ports(state, core, key):
    return key in state and core.check(
        'wires',
        state[key])


def check_edge(schema, state, core):
    return isinstance(state, dict) and check_ports(state, core, 'inputs') and check_ports(state, core, 'outputs')


def serialize_edge(schema, value, core):
    return value


def deserialize_edge(schema, encoded, core):
    return encoded


def array_shape(core, schema):
    if '_type_parameters' not in schema:
        schema = core.access(schema)
    parameters = schema.get('_type_parameters', [])

    return tuple([
        int(schema[f'_{parameter}'])
        for parameter in schema['_type_parameters']])


def check_array(schema, state, core):
    shape_type = core.find_parameter(
        schema,
        'shape')

    return isinstance(state, np.ndarray) and state.shape == array_shape(core, shape_type) # and state.dtype == bindings['data'] # TODO align numpy data types so we can validate the types of the arrays



def dataclass_array(schema, path, core):
    return np.ndarray


def slice_array(schema, state, path, core):
    if len(path) > 0:
        head = path[0]
        tail = path[1:]
        step = state[head]

        if isinstance(step, np.ndarray):
            sliceschema = schema.copy()
            sliceschema['_shape'] = step.shape
            return core.slice(
                sliceschema,
                step,
                tail)
        else:
            data_type = core.find_parameter(
                schema,
                'data')

            return core.slice(
                data_type,
                step,
                tail)

    else:
        return schema, state


def apply_array(schema, current, update, core):
    if isinstance(update, dict):
        paths = hierarchy_depth(update)
        for path, inner_update in paths.items():
            if len(path) > len(schema['_shape']):
                raise Exception(f'index is too large for array update: {path}\n  {schema}')
            else:
                index = tuple(path)
                current[index] += inner_update

        return current

    else:
        return current + update


def serialize_array(schema, value, core):
    """ Serialize numpy array to list """

    if isinstance(value, dict):
        return value
    elif isinstance(value, str):
        import ipdb; ipdb.set_trace()
    else:
        array_data = 'string'
        dtype = value.dtype.name
        if dtype.startswith('int'):
            array_data = 'integer'
        elif dtype.startswith('float'):
            array_data = 'float'

        return {
            'list': value.tolist(),
            'data': array_data,
            'shape': list(value.shape)}


DTYPE_MAP = {
    'float': 'float64',
    'integer': 'int64',
    'string': 'str'}


def lookup_dtype(data_name):
    data_name = data_name or 'string'
    dtype_name = DTYPE_MAP.get(data_name)
    if dtype_name is None:
        raise Exception(f'unknown data type for array: {data_name}')

    return np.dtype(dtype_name)


def read_datatype(data_schema):
    return lookup_dtype(
        data_schema['_type'])


def read_shape(shape):
    return tuple([
        int(x)
        for x in tuple_from_type(
            shape)])


def deserialize_array(schema, encoded, core):
    if isinstance(encoded, np.ndarray):
        return encoded

    elif isinstance(encoded, dict):
        if 'value' in encoded:
            return encoded['value']
        else:
            found = core.retrieve(
                encoded.get(
                    'data',
                    schema['_data']))

            dtype = read_datatype(
                found)

            shape = read_shape(
                schema['_shape'])

            if 'list' in encoded:
                return np.array(
                    encoded['list'],
                    dtype=dtype).reshape(
                        shape)
            else:
                return np.zeros(
                    tuple(shape),
                    dtype=dtype)


def default_tree(schema, core):
    leaf_schema = core.find_parameter(
        schema,
        'leaf')

    default = {}

    non_schema_keys = [
        key
        for key in schema
        if not is_schema_key(key)]

    if non_schema_keys:
        base_schema = {
            key: subschema
            for key, subschema in schema.items()
            if is_schema_key(key)}

        for key in non_schema_keys:
            subschema = core.merge_schemas(
                base_schema,
                schema[key])

            subdefault = core.default(
                subschema)

            if subdefault:
                default[key] = subdefault

    return default


def default_array(schema, core):
    data_schema = core.find_parameter(
        schema,
        'data')

    dtype = read_datatype(
        data_schema)

    shape = read_shape(
        schema['_shape'])

    return np.zeros(
        shape,
        dtype=dtype)


def fold_list(schema, state, method, values, core):
    element_type = core.find_parameter(
        schema,
        'element')
    
    if core.check(element_type, state):
        result = core.fold(
            element_type,
            state,
            method,
            values)

    elif isinstance(state, list):
        subresult = [
            fold_list(
                schema,
                element,
                method,
                values, 
                core)
            for element in state]

        result = visit_method(
            schema,
            subresult,
            method,
            values, 
            core)

    else:
        raise Exception(f'state does not seem to be a list or an eelement:\n  state: {state}\n  schema: {schema}')

    return result


def fold_tree(schema, state, method, values, core):
    leaf_type = core.find_parameter(
        schema,
        'leaf')
    
    if core.check(leaf_type, state):
        result = core.fold(
            leaf_type,
            state,
            method,
            values)

    elif isinstance(state, dict):
        subresult = {}

        for key, branch in state.items():
            if key.startswith('_'):
                subresult[key] = branch
            else:
                subresult[key] = fold_tree(
                    schema[key] if key in schema else schema,
                    branch,
                    method,
                    values,
                    core)

        result = visit_method(
            schema,
            subresult,
            method,
            values,
            core)

    else:
        raise Exception(f'state does not seem to be a tree or a leaf:\n  state: {state}\n  schema: {schema}')

    return result


def fold_map(schema, state, method, values, core):
    value_type = core.find_parameter(
        schema,
        'value')
    
    subresult = {}

    for key, value in state.items():
        subresult[key] = core.fold(
            value_type,
            value,
            method,
            values)
    
    result = visit_method(
        schema,
        subresult,
        method,
        values,
        core)

    return result


def fold_maybe(schema, state, method, values, core):
    value_type = core.find_parameter(
        schema,
        'value')

    if state is None:
        result = core.fold(
            'any',
            state,
            method,
            values)

    else:
        result = core.fold(
            value_type,
            state,
            method,
            values)

    return result


def divide_list(schema, state, values, core):
    element_type = core.find_parameter(
        schema,
        'element')

    if core.check(element_type, state):
        return core.fold(
            element_type,
            state,
            'divide',
            values)

    elif isinstance(state, list):
        divisions = values.get('divisions', 2)
        result = [[] for _ in range(divisions)]

        for elements in state:
            for index in range(divisions):
                result[index].append(
                    elements[index])

        return result

    else:
        raise Exception(f'trying to divide list but state does not resemble a list or an element.\n  state: {pf(state)}\n  schema: {pf(schema)}')


def divide_tree(schema, state, values, core):
    leaf_type = core.find_parameter(
        schema,
        'leaf')
    
    if core.check(leaf_type, state):
        return core.fold(
            leaf_type,
            state,
            'divide',
            values)

    elif isinstance(state, dict):
        divisions = values.get('divisions', 2)
        division = [{} for _ in range(divisions)]

        for key, value in state.items():
            for index in range(divisions):
                division[index][key] = value[index]

        return division

    else:
        raise Exception(f'trying to divide tree but state does not resemble a leaf or a tree.\n  state: {pf(state)}\n  schema: {pf(schema)}')


def divide_map(schema, state, values, core):
    if isinstance(state, dict):
        divisions = values.get('divisions', 2)
        division = [{} for _ in range(divisions)]
        for key, value in state.items():
            for index in range(divisions):
                division[index][key] = value[index]

        return division
    else:
        raise Exception(f'trying to divide a map but state is not a dict.\n  state: {pf(state)}\n  schema: {pf(schema)}')        


def register_units(core, units):
    for unit_name in units._units:
        try:
            unit = getattr(units, unit_name)
        except:
            # print(f'no unit named {unit_name}')
            continue

        dimensionality = unit.dimensionality
        type_key = render_units_type(dimensionality)
        if not core.exists(type_key):
            core.register(type_key, {
                '_default': '',
                '_apply': apply_units,
                '_check': check_units,
                '_serialize': serialize_units,
                '_deserialize': deserialize_units,
                '_description': 'type to represent values with scientific units'})

    return core



def dataclass_boolean(schema, path, core):
    return bool


def dataclass_string(schema, path, core):
    return str


def apply_enum(schema, current, update, core):
    parameters = core.parameters_for(schema)
    if update in parameters:
        return update
    else:
        raise Exception(f'{update} is not in the enum, options are: {parameters}')


def check_enum(schema, state, core):
    if not isinstance(state, str):
        return False

    parameters = core.parameters_for(schema)
    return state in parameters


def slice_string(schema, state, path, core):
    raise Exception(f'cannot slice into an string: {path}\n{state}\n{schema}')


def serialize_enum(schema, value, core):
    return value


def deserialize_enum(schema, state, core):
    return value


def bind_enum(schema, state, key, subschema, substate, core):
    new_schema = schema.copy()
    new_schema[f'_{key}'] = subschema
    open = list(state)
    open[key] = substate

    return new_schema, tuple(open)


def fold_enum(schema, state, method, values, core):
    if not isinstance(state, (tuple, list)):
        return visit_method(
            schema,
            state,
            method,
            values,
            core)
    else:
        parameters = core.parameters_for(schema)
        result = []
        for parameter, element in zip(parameters, state):
            fold = core.fold(
                parameter,
                element,
                method,
                values)
            result.append(fold)

        result = tuple(result)

        return visit_method(
            schema,
            result,
            method,
            values,
            core)


def dataclass_enum(schema, path, core):
    parameters = type_parameters_for(schema)
    subtypes = []

    for index, key in enumerate(schema['type_parameters']):
        subschema = schema.get(key, 'any')
        subtype = core.dataclass(
            subschema,
            path + [index])

        subtypes.append(subtype)

    parameter_block = ', '.join(subtypes)
    return eval(f'tuple[{parameter_block}]')


def divide_enum(schema, state, values, core):
    divisions = values.get('divisions', 2)

    return [
        tuple([item[index] for item in state])
        for index in range(divisions)]


# def merge_edge(schema, current_state, new_state, core):
#     merge = deep_merge(
#         current_state,
#         new_state)

#     return core.deserialize(
#         schema,
#         merge)


def serialize_schema(schema, state, core):
    return state


def deserialize_schema(schema, state, core):
    return state


def default_enum(schema, core):
    parameter = schema['_type_parameters'][0]
    return schema[f'_{parameter}']


def default_edge(schema, core):
    return {
        # '_type': schema['_type'],
        'inputs': core.default(
            schema['inputs']),
        'outputs': core.default(
            schema['outputs'])}


base_type_library = {
    'boolean': {
        '_type': 'boolean',
        '_default': False,
        '_check': check_boolean,
        '_apply': apply_boolean,
        '_serialize': serialize_boolean,
        '_deserialize': deserialize_boolean,
        '_dataclass': dataclass_boolean},

    # abstract number type
    'number': {
        '_type': 'number',
        '_check': check_number,
        '_apply': accumulate,
        '_serialize': to_string,
        '_description': 'abstract base type for numbers'},

    'integer': {
        '_type': 'integer',
        '_default': 0,
        # inherit _apply and _serialize from number type
        '_check': check_integer,
        '_deserialize': deserialize_integer,
        '_dataclass': dataclass_integer,
        '_description': '64-bit integer',
        '_inherit': 'number'},

    'float': {
        '_type': 'float',
        '_default': 0.0,
        '_check': check_float,
        '_deserialize': deserialize_float,
        '_divide': divide_float,
        '_dataclass': dataclass_float,
        '_description': '64-bit floating point precision number',
        '_inherit': 'number'},

    'string': {
        '_type': 'string',
        '_default': '',
        '_check': check_string,
        '_apply': replace,
        '_serialize': serialize_string,
        '_deserialize': deserialize_string,
        '_dataclass': dataclass_string,
        '_description': '64-bit integer'},

    'enum': {
        '_type': 'enum',
        '_default': default_enum,
        '_apply': apply_enum,
        '_check': check_enum,
        '_serialize': serialize_string,
        '_deserialize': deserialize_string,
        '_dataclass': dataclass_string,
        '_description': 'enumeration type for a selection of key values'},

    'list': {
        '_type': 'list',
        '_default': [],
        '_check': check_list,
        '_slice': slice_list,
        '_apply': apply_list,
        '_serialize': serialize_list,
        '_deserialize': deserialize_list,
        '_dataclass': dataclass_list,
        '_fold': fold_list,
        '_divide': divide_list,
        '_type_parameters': ['element'],
        '_description': 'general list type (or sublists)'},

    'map': {
        '_type': 'map',
        '_default': {},
        '_apply': apply_map,
        '_serialize': serialize_map,
        '_deserialize': deserialize_map,
        '_resolve': resolve_map,
        '_dataclass': dataclass_map,
        '_check': check_map,
        '_slice': slice_map,
        '_fold': fold_map,
        '_divide': divide_map,
        '_type_parameters': ['value'],
        '_description': 'flat mapping from keys of strings to values of any type'},

    'tree': {
        '_type': 'tree',
        '_default': default_tree,
        '_generate': generate_tree,
        '_check': check_tree,
        '_slice': slice_tree,
        '_apply': apply_tree,
        '_serialize': serialize_tree,
        '_deserialize': deserialize_tree,
        '_dataclass': dataclass_tree,
        '_fold': fold_tree,
        '_divide': divide_tree,
        # '_resolve': resolve_tree,
        '_type_parameters': ['leaf'],
        '_description': 'mapping from str to some type in a potentially nested form'},

    'array': {
        '_type': 'array',
        '_default': default_array,
        '_check': check_array,
        '_slice': slice_array,
        '_apply': apply_array,
        '_serialize': serialize_array,
        '_deserialize': deserialize_array,
        '_dataclass': dataclass_array,
        '_resolve': resolve_array,
        '_type_parameters': [
            'shape',
            'data'],
        '_description': 'an array of arbitrary dimension'},

    'maybe': {
        '_type': 'maybe',
        '_default': NONE_SYMBOL,
        '_apply': apply_maybe,
        '_check': check_maybe,
        '_slice': slice_maybe,
        '_serialize': serialize_maybe,
        '_deserialize': deserialize_maybe,
        '_dataclass': dataclass_maybe,
        '_fold': fold_maybe,
        '_type_parameters': ['value'],
        '_description': 'type to represent values that could be empty'},

    'path': {
        '_type': 'path',
        '_inherit': 'list[string]',
        '_apply': apply_path},

    'wires': {
        '_type': 'wires',
        '_inherit': 'tree[path]'},

    'schema': {
        '_type': 'schema',
        '_inherit': 'tree[any]',
        '_apply': apply_schema,
        '_serialize': serialize_schema,
        '_deserialize': deserialize_schema},

    'edge': {
        '_type': 'edge',
        '_default': default_edge,
        '_generate': generate_edge,
        '_apply': apply_edge,
        '_serialize': serialize_edge,
        '_deserialize': deserialize_edge,
        '_dataclass': dataclass_edge,
        '_check': check_edge,
        # '_merge': merge_edge,
        '_type_parameters': ['inputs', 'outputs'],
        '_description': 'hyperedges in the bigraph, with inputs and outputs as type parameters',
        'inputs': 'wires',
        'outputs': 'wires'}}


def add_reaction(schema, state, reaction, core):
    path = reaction.get('path')

    redex = {}
    establish_path(
        redex,
        path)

    reactum = {}
    node = establish_path(
        reactum,
        path)

    deep_merge(
        node,
        reaction.get('add', {}))

    return {
        'redex': redex,
        'reactum': reactum}


def remove_reaction(schema, state, reaction, core):
    path = reaction.get('path', ())
    redex = {}
    node = establish_path(
        redex,
        path)

    for remove in reaction.get('remove', []):
        node[remove] = {}

    reactum = {}
    establish_path(
        reactum,
        path)

    return {
        'redex': redex,
        'reactum': reactum}


def replace_reaction(schema, state, reaction, core):
    path = reaction.get('path', ())

    redex = {}
    node = establish_path(
        redex,
        path)

    for before_key, before_state in reaction.get('before', {}).items():
        node[before_key] = before_state

    reactum = {}
    node = establish_path(
        reactum,
        path)

    for after_key, after_state in reaction.get('after', {}).items():
        node[after_key] = after_state

    return {
        'redex': redex,
        'reactum': reactum}


def divide_reaction(schema, state, reaction, core):
    mother = reaction['mother']
    daughters = reaction['daughters']

    mother_schema, mother_state = core.slice(
        schema,
        state,
        mother)

    division = core.fold(
        mother_schema,
        mother_state,
        'divide', {
            'divisions': len(daughters),
            'daughter_configs': [daughter[1] for daughter in daughters]})

    after = {
        daughter[0]: daughter_state
        for daughter, daughter_state in zip(daughters, division)}

    replace = {
        'before': {
            mother: {}},
        'after': after}

    return replace_reaction(
        schema,
        state,
        replace,
        core)


def register_base_reactions(core):
    core.register_reaction('add', add_reaction)
    core.register_reaction('remove', remove_reaction)
    core.register_reaction('replace', replace_reaction)
    core.register_reaction('divide', divide_reaction)


def register_cube(core):
    cube_schema = {
        'shape': {
            '_type': 'shape',
            '_description': 'abstract shape type'},
        
        'rectangle': {
            '_type': 'rectangle',
            '_divide': divide_longest,
            '_description': 'a two-dimensional value',
            '_inherit': 'shape',
            'width': {'_type': 'integer'},
            'height': {'_type': 'integer'},
        },
        
        # cannot override existing keys unless it is of a subtype
        'cube': {
            '_type': 'cube',
            '_inherit': 'rectangle',
            'depth': {'_type': 'integer'},
        },
    }

    for type_key, type_data in cube_schema.items():
        core.register(type_key, type_data)

    return core


@pytest.fixture
def core():
    core = TypeSystem()
    return register_test_types(core)


def register_test_types(core):
    register_cube(core)

    core.register('compartment', {
        'counts': 'tree[float]',
        'inner': 'tree[compartment]'})

    core.register('metaedge', {
        '_inherit': 'edge',
        '_inputs': {
            'before': 'metaedge'},
        '_outputs': {
            'after': 'metaedge'}})

    return core


def test_reregister_type(core):
    core.register('A', {'_default': 'a'})
    with pytest.raises(Exception) as e:
        core.register(
            'A', {'_default': 'b'},
            strict=True)

    core.register('A', {'_default': 'b'})

    assert core.access('A')['_default'] == 'b'


def test_generate_default(core):
    int_default = core.default(
        {'_type': 'integer'}
    )

    assert int_default == 0

    cube_default = core.default(
        {'_type': 'cube'})

    assert 'width' in cube_default
    assert 'height' in cube_default
    assert 'depth' in cube_default

    nested_default = core.default(
        {'a': 'integer',
         'b': {
             'c': 'float',
             'd': 'cube'},
         'e': 'string'})

    assert nested_default['b']['d']['width'] == 0


def test_apply_update(core):
    schema = {'_type': 'cube'}
    state = {
        'width': 11,
        'height': 13,
        'depth': 44,
    }

    update = {
        'depth': -5
    }

    new_state = core.apply(
        schema,
        state,
        update
    )

    assert new_state['width'] == 11
    assert new_state['depth'] == 39


def print_schema_validation(core, library, should_pass):
    for key, declaration in library.items():
        report = core.validate_schema(declaration)
        if len(report) == 0:
            message = f'valid schema: {key}'
            if should_pass:
                print(f'PASS: {message}')
                pprint.pprint(declaration)
            else:
                raise Exception(f'FAIL: {message}\n{declaration}\n{report}')
        else:
            message = f'invalid schema: {key}'
            if not should_pass:
                print(f'PASS: {message}')
                pprint.pprint(declaration)
            else:
                raise Exception(f'FAIL: {message}\n{declaration}\n{report}')


def test_validate_schema(core):
    # good schemas
    print_schema_validation(core, base_type_library, True)

    good = {
        'not quite int': {
            '_default': 0,
            '_apply': accumulate,
            '_serialize': to_string,
            '_deserialize': deserialize_integer,
            '_description': '64-bit integer'
        },
        'ports match': {
            'a': {
                '_type': 'integer',
                '_value': 2
            },
            'edge1': {
                '_type': 'edge[a:integer]',
                # '_type': 'edge',
                # '_ports': {
                #     '1': {'_type': 'integer'},
                # },
            }
        }
    }        

    # bad schemas
    bad = {
        'empty': None,
        'str?': 'not a schema',
        'branch is weird': {
            'left': {'_type': 'ogre'},
            'right': {'_default': 1, '_apply': accumulate},
        },
    }

    # test for ports and wires mismatch

    print_schema_validation(core, good, True)
    print_schema_validation(core, bad, False)


def test_fill_integer(core):
    test_schema = {
        '_type': 'integer'
    }

    full_state = core.fill(test_schema)
    direct_state = core.fill('integer')
    generated_schema, generated_state = core.generate(
        test_schema, None)

    assert generated_schema['_type'] == 'integer'
    assert full_state == direct_state == 0 == generated_state


def test_fill_cube(core):
    test_schema = {'_type': 'cube'}
    partial_state = {'height': 5}

    full_state = core.fill(
        test_schema,
        state=partial_state)

    assert 'width' in full_state
    assert 'height' in full_state
    assert 'depth' in full_state
    assert full_state['height'] == 5
    assert full_state['depth'] == 0


def test_fill_in_missing_nodes(core):
    test_schema = {
        'edge 1': {
            '_type': 'edge',
            '_inputs': {
                'I': 'float'},
            '_outputs': {
                'O': 'float'}}}

    test_state = {
        'edge 1': {
            'inputs': {
                'I': ['a']},
            'outputs': {
                'O': ['a']}}}

    filled = core.fill(
        test_schema,
        test_state)

    assert filled == {
        'a': 0.0,
        'edge 1': {
            'inputs': {
                'I': ['a']},
            'outputs': {
                'O': ['a']}}}


def test_overwrite_existing(core):
    test_schema = {
        'edge 1': {
            '_type': 'edge',
            '_inputs': {
                'I': 'float'},
            '_outputs': {
                'O': 'float'}}}

    test_state = {
        'a': 11.111,
        'edge 1': {
            'inputs': {
                'I': ['a']},
            'outputs': {
                'O': ['a']}}}

    filled = core.fill(
        test_schema,
        test_state)

    assert filled == {
        'a': 11.111,
        'edge 1': {
            'inputs': {
                'I': ['a']},
            'outputs': {
                'O': ['a']}}}


def test_fill_from_parse(core):
    test_schema = {
        'edge 1': 'edge[I:float,O:float]'}

    test_state = {
        'edge 1': {
            'inputs': {
                'I': ['a']},
            'outputs': {
                'O': ['a']}}}

    filled = core.fill(
        test_schema,
        test_state)

    assert filled == {
        'a': 0.0,
        'edge 1': {
            'inputs': {
                'I': ['a']},
            'outputs': {
                'O': ['a']}}}


# def test_fill_in_disconnected_port(core):
#     test_schema = {
#         'edge1': {
#             '_type': 'edge',
#             '_ports': {
#                 '1': {'_type': 'float'}}}}

#     test_state = {}


# def test_fill_type_mismatch(core):
#     test_schema = {
#         'a': {'_type': 'integer', '_value': 2},
#         'edge1': {
#             '_type': 'edge',
#             '_ports': {
#                 '1': {'_type': 'float'},
#                 '2': {'_type': 'float'}},
#             'wires': {
#                 '1': ['..', 'a'],
#                 '2': ['a']},
#             'a': 5}}


# def test_edge_type_mismatch(core):
#     test_schema = {
#         'edge1': {
#             '_type': 'edge',
#             '_ports': {
#                 '1': {'_type': 'float'}},
#             'wires': {
#                 '1': ['..', 'a']}},
#         'edge2': {
#             '_type': 'edge',
#             '_ports': {
#                 '1': {'_type': 'integer'}},
#             'wires': {
#                 '1': ['..', 'a']}}}


def test_establish_path(core):
    tree = {}
    destination = establish_path(
        tree,
        ('some',
         'where',
         'deep',
         'inside',
         'lives',
         'a',
         'tiny',
         'creature',
         'made',
         'of',
         'light'))

    assert tree['some']['where']['deep']['inside']['lives']['a']['tiny']['creature']['made']['of']['light'] == destination


def test_fill_ports(core):
    cell_state = {
        'cell1': {
            'nucleus': {
                'transcription': {
                    '_type': 'edge',
                    'inputs': {'DNA': ['chromosome']},
                    'outputs': {
                        'RNA': [ '..', 'cytoplasm']}}}}}

    schema, state = core.complete(
        {},
        cell_state)

    assert 'chromosome' in schema['cell1']['nucleus']


def test_expected_schema(core):
    # equivalent to previous schema:

    # expected = {
    #     'store1': {
    #         'store1.1': {
    #             '_value': 1.1,
    #             '_type': 'float',
    #         },
    #         'store1.2': {
    #             '_value': 2,
    #             '_type': 'integer',
    #         },
    #         'process1': {
    #             '_ports': {
    #                 'port1': {'_type': 'type'},
    #                 'port2': {'_type': 'type'},
    #             },
    #             '_wires': {
    #                 'port1': 'store1.1',
    #                 'port2': 'store1.2',
    #             }
    #         },
    #         'process2': {
    #             '_ports': {
    #                 'port1': {'_type': 'type'},
    #                 'port2': {'_type': 'type'},
    #             },
    #             '_wires': {
    #                 'port1': 'store1.1',
    #                 'port2': 'store1.2',
    #             }
    #         },
    #     },
    #     'process3': {
    #         '_wires': {
    #             'port1': 'store1',
    #         }
    #     }
    # }

    dual_process_schema = {
        'process1': 'edge[input1:float|input2:integer,output1:float|output2:integer]',
        'process2': {
            '_type': 'edge',
            '_inputs': {
                'input1': 'float',
                'input2': 'integer'},
            '_outputs': {
                'output1': 'float',
                'output2': 'integer'}}}

    core.register(
        'dual_process',
        dual_process_schema,
    )

    test_schema = {
        # 'store1': 'process1.edge[port1.float|port2.int]|process2[port1.float|port2.int]',
        'store1': 'dual_process',
        'process3': 'edge[input_process:dual_process,output_process:dual_process]'}

    test_state = {
        'store1': {
            'process1': {
                'inputs': {
                    'input1': ['store1.1'],
                    'input2': ['store1.2']},
                'outputs': {
                    'output1': ['store2.1'],
                    'output2': ['store2.2']}},
            'process2': {
                'inputs': {
                    'input1': ['store2.1'],
                    'input2': ['store2.2']},
                'outputs': {
                    'output1': ['store1.1'],
                    'output2': ['store1.2']}}},
        'process3': {
            'inputs': {
                'input_process': ['store1']},
            'outputs': {
                'output_process': ['store1']}}}
    
    outcome = core.fill(test_schema, test_state)

    assert outcome == {
        'process3': {
            'inputs': {
                'input_process': ['store1']},
            'outputs': {
                'output_process': ['store1']}},
        'store1': {
            'process1': {
                'inputs': {
                    'input1': ['store1.1'],
                    'input2': ['store1.2']},
                'outputs': {
                    'output1': ['store2.1'],
                    'output2': ['store2.2']}},
            'process2': {
                'inputs': {'input1': ['store2.1'],
                           'input2': ['store2.2']},
                'outputs': {'output1': ['store1.1'],
                            'output2': ['store1.2']}},
            'store1.1': 0.0,
            'store1.2': 0,
            'store2.1': 0.0,
            'store2.2': 0}}


def test_link_place(core):
    # TODO: this form is more fundamental than the compressed/inline dict form,
    #   and we should probably derive that from this form

    bigraph = {
        'nodes': {
            'v0': 'integer',
            'v1': 'integer',
            'v2': 'integer',
            'v3': 'integer',
            'v4': 'integer',
            'v5': 'integer',
            'e0': 'edge[e0-0:int|e0-1:int|e0-2:int]',
            'e1': {
                '_type': 'edge',
                '_ports': {
                    'e1-0': 'integer',
                    'e2-0': 'integer'}},
            'e2': {
                '_type': 'edge[e2-0:int|e2-1:int|e2-2:int]'}},

        'place': {
            'v0': None,
            'v1': 'v0',
            'v2': 'v0',
            'v3': 'v2',
            'v4': None,
            'v5': 'v4',
            'e0': None,
            'e1': None,
            'e2': None},

        'link': {
            'e0': {
                'e0-0': 'v0',
                'e0-1': 'v1',
                'e0-2': 'v4'},
            'e1': {
                'e1-0': 'v3',
                'e1-1': 'v1'},
            'e2': {
                'e2-0': 'v3',
                'e2-1': 'v4',
                'e2-2': 'v5'}},

        'state': {
            'v0': '1',
            'v1': '1',
            'v2': '2',
            'v3': '3',
            'v4': '5',
            'v5': '8',
            'e0': {
                'wires': {
                    'e0-0': 'v0',
                    'e0-1': 'v1',
                    'e0-2': 'v4'}},
            'e1': {
                'wires': {
                    'e1-0': 'v3',
                    'e1-1': 'v1'}},
            'e2': {
                'e2-0': 'v3',
                'e2-1': 'v4',
                'e2-2': 'v5'}}}

    placegraph = { # schema
        'v0': {
            'v1': int,
            'v2': {
                'v3': int}},
        'v4': {
            'v5': int},
        'e0': 'edge',
        'e1': 'edge',
        'e2': 'edge'}

    hypergraph = { # edges
        'e0': {
            'e0-0': 'v0',
            'e0-1': 'v1',
            'e0-2': 'v4'},
        'e1': {
            'e1-0': 'v3',
            'e1-1': 'v1'},
        'e2': {
            'e2-0': 'v3',
            'e2-1': 'v4',
            'e2-2': 'v5'}}

    merged = {
        'v0': {
            'v1': {},
            'v2': {
                'v3': {}}},
        'v4': {
            'v5': {}},
        'e0': {
            'wires': {
                'e0.0': ['v0'],
                'e0.1': ['v0', 'v1'],
                'e0.2': ['v4']}},
        'e1': {
            'wires': {
                'e0.0': ['v0', 'v2', 'v3'],
                'e0.1': ['v0', 'v1']}},
        'e2': {
            'wires': {
                'e0.0': ['v0', 'v2', 'v3'],
                'e0.1': ['v4'],
                'e0.2': ['v4', 'v5']}}}

    result = core.link_place(placegraph, hypergraph)
    # assert result == merged


def test_units(core):
    schema_length = {
        'distance': {'_type': 'length'}}

    state = {'distance': 11 * units.meter}
    update = {'distance': -5 * units.feet}

    new_state = core.apply(
        schema_length,
        state,
        update
    )

    assert new_state['distance'] == 9.476 * units.meter


def test_unit_conversion(core):
    # mass * length ^ 2 / second ^ 2

    units_schema = {
        'force': 'length^2*mass/time^2'}

    force_units = units.meter ** 2 * units.kg / units.second ** 2

    instance = {
        'force': 3.333 * force_units}


def test_serialize_deserialize(core):
    schema = {
        'edge1': {
            # '_type': 'edge[1:int|2:float|3:string|4:tree[int]]',
            '_type': 'edge',
            '_outputs': {
                '1': 'integer',
                '2': 'float',
                '3': 'string',
                '4': 'tree[integer]'}},
        'a0': {
            'a0.0': 'integer',
            'a0.1': 'float',
            'a0.2': {
                'a0.2.0': 'string'}},
        'a1': 'tree[integer]'}

    instance = {
        'edge1': {
            'outputs': {
                '1': ['a0', 'a0.0'],
                '2': ['a0', 'a0.1'],
                '3': ['a0', 'a0.2', 'a0.2.0'],
                '4': ['a1']}},
        'a1': {
            'branch1': {
                'branch2': 11,
                'branch3': 22},
            'branch4': 44}}
    
    instance = core.fill(schema, instance)

    encoded = core.serialize(schema, instance)
    decoded = core.deserialize(schema, encoded)

    assert instance == decoded


# is this a lens?
def test_project(core):
    schema = {
        'edge1': {
            # '_type': 'edge[1:int|2:float|3:string|4:tree[int]]',
            # '_type': 'edge',
            '_type': 'edge',
            '_inputs': {
                '1': 'integer',
                '2': 'float',
                '3': 'string',
                'inner': {
                    'chamber': 'tree[integer]'},
                '4': 'tree[integer]'},
            '_outputs': {
                '1': 'integer',
                '2': 'float',
                '3': 'string',
                'inner': {
                    'chamber': 'tree[integer]'},
                '4': 'tree[integer]'}},
        'a0': {
            'a0.0': 'integer',
            'a0.1': 'float',
            'a0.2': {
                'a0.2.0': 'string'}},
        'a1': {
            '_type': 'tree[integer]'}}

    path_format = {
        '1': 'a0>a0.0',
        '2': 'a0>a0.1',
        '3': 'a0>a0.2>a0.2.0'}

    # TODO: support separate schema/instance, and 
    #   instances with '_type' and type parameter keys
    # TODO: support overriding various type methods
    instance = {
        'a0': {
            'a0.0': 11},
        'edge1': {
            'inputs': {
                '1': ['a0', 'a0.0'],
                '2': ['a0', 'a0.1'],
                '3': ['a0', 'a0.2', 'a0.2.0'],
                'inner': {
                    'chamber': ['a1', 'a1.0']},
                '4': ['a1']},
            'outputs': {
                '1': ['a0', 'a0.0'],
                '2': ['a0', 'a0.1'],
                '3': ['a0', 'a0.2', 'a0.2.0'],
                'inner': {
                    'chamber': {
                        'X': ['a1', 'a1.0', 'Y']}},
                '4': ['a1']}},
        'a1': {
            'a1.0': {
                'X': 555},
            'branch1': {
                'branch2': 11,
                'branch3': 22},
            'branch4': 44}}

    instance = core.fill(schema, instance)

    states = core.view_edge(
        schema,
        instance,
        ['edge1'])

    update = core.project_edge(
        schema,
        instance,
        ['edge1'],
        states)

    assert update == {
        'a0': {
            'a0.0': 11,
            'a0.1': 0.0,
            'a0.2': {
                'a0.2.0': ''}},
        'a1': {
            'a1.0': {
                'X': 555,
                'Y': {}},
            'branch1': {
                'branch2': 11,
                'branch3': 22},
            'branch4': 44}}

    # TODO: make sure apply does not mutate instance
    updated_instance = core.apply(
        schema,
        instance,
        update)

    add_update = {
        '4': {
            'branch6': 111,
            'branch1': {
                '_add': {
                    'branch7': 4444,
                    'branch8': 555,
                },
                '_remove': ['branch2']},
            '_add': {
                'branch5': 55},
            '_remove': ['branch4']}}

    inverted_update = core.project_edge(
        schema,
        updated_instance,
        ['edge1'],
        add_update)

    modified_branch = core.apply(
        schema,
        updated_instance,
        inverted_update)

    assert modified_branch == {
        'a0': {
            'a0.0': 22,
            'a0.1': 0.0,
            'a0.2': {
                'a0.2.0': ''}},
        'edge1': {'inputs': {'1': ['a0', 'a0.0'],
                             '2': ['a0', 'a0.1'],
                             '3': ['a0', 'a0.2', 'a0.2.0'],
                             'inner': {
                                 'chamber': ['a1', 'a1.0']},
                             '4': ['a1']},
                  'outputs': {'1': ['a0', 'a0.0'],
                              '2': ['a0', 'a0.1'],
                              '3': ['a0', 'a0.2', 'a0.2.0'],
                              'inner': {
                                  'chamber': {
                                      'X': ['a1', 'a1.0', 'Y']}},
                              '4': ['a1']}},
        'a1': {
            'a1.0': {
                'X': 1110,
                'Y': {}},
            'branch1': {
                'branch3': 44,
                'branch7': 4444,
                'branch8': 555,},
            'branch6': 111,
            'branch5': 55}}


def test_check(core):
    assert core.check('float', 1.11)
    assert core.check({'b': 'float'}, {'b': 1.11})


def test_inherits_from(core):
    assert core.inherits_from(
        'float',
        'number')

    assert core.inherits_from(
        'tree[float]',
        'tree[number]')

    assert core.inherits_from(
        'tree[path]',
        'tree[list[string]]')

    assert not core.inherits_from(
        'tree[path]',
        'tree[list[number]]')

    assert not core.inherits_from(
        'tree[float]',
        'tree[string]')

    assert not core.inherits_from(
        'tree[float]',
        'list[float]')

    assert core.inherits_from({
        'a': 'float',
        'b': 'schema'}, {

        'a': 'number',
        'b': 'tree'})

    assert not core.inherits_from({
        'a': 'float',
        'b': 'schema'}, {

        'a': 'number',
        'b': 'number'})


def test_resolve_schemas(core):
    resolved = core.resolve_schemas({
        'a': 'float',
        'b': 'map[list[string]]'}, {
        'a': 'number',
        'b': 'map[path]',
        'c': 'string'})

    assert resolved['a']['_type'] == 'float'
    assert resolved['b']['_value']['_type'] == 'path'
    assert resolved['c']['_type'] == 'string'

    raises_on_incompatible_schemas = False
    try:
        core.resolve_schemas({
            'a': 'string',
            'b': 'map[list[string]]'}, {
                'a': 'number',
                'b': 'map[path]',
                'c': 'string'})
    except:
        raises_on_incompatible_schemas = True

    assert raises_on_incompatible_schemas


def test_apply_schema(core):
    current = {
        'a': 'number',
        'b': 'map[path]',
        'd': ('float', 'number', 'list[string]')}

    update = {
        'a': 'float',
        'b': 'map[list[string]]',
        'c': 'string',
        'd': ('number', 'float', 'path')}

    applied = apply_schema(
        'schema',
        current,
        update,
        core)

    assert applied['a']['_type'] == 'float'
    assert applied['b']['_value']['_type'] == 'path'
    assert applied['c']['_type'] == 'string'
    assert applied['d']['_0']['_type'] == 'float'
    assert applied['d']['_1']['_type'] == 'float'
    assert applied['d']['_2']['_type'] == 'path'


def apply_foursquare(schema, current, update, core):
    if isinstance(current, bool) or isinstance(update, bool):
        return update
    else:
        for key, value in update.items():
            current[key] = apply_foursquare(
                schema,
                current[key],
                value,
                core)

        return current
                

def test_foursquare(core):
    foursquare_schema = {
        '_apply': apply_foursquare,
        '00': 'boolean~foursquare',
        '01': 'boolean~foursquare',
        '10': 'boolean~foursquare',
        '11': 'boolean~foursquare'}

    core.register(
        'foursquare',
        foursquare_schema)

    example = {
        '00': True,
        '01': False,
        '10': False,
        '11': {
            '00': True,
            '01': False,
            '10': False,
            '11': {
                '00': True,
                '01': False,
                '10': False,
                '11': {
                    '00': True,
                    '01': False,
                    '10': False,
                    '11': {
                        '00': True,
                        '01': False,
                        '10': False,
                        '11': {
                            '00': True,
                            '01': False,
                            '10': False,
                            '11': False}}}}}}

    assert core.check(
        'foursquare',
        example)

    example['10'] = 5

    assert not core.check(
        'foursquare',
        example)

    update = {
        '01': True,
        '11': {
            '01': True,
            '11': {
                '11': True,
                '10': {
                    '10': {
                        '00': True,
                        '11': False}}}}}

    result = core.apply(
        'foursquare',
        example,
        update)

    assert result == {
        '00': True,
        '01': True,
        '10': 5,
        '11': {'00': True,
               '01': True,
               '10': False,
               '11': {'00': True,
                      '01': False,
                      '10': {
                          '10': {
                              '00': True,
                              '11': False}},
                      '11': True}}}


def test_add_reaction(core):
    single_node = {
        'environment': {
            '_type': 'compartment',
            'counts': {'A': 144},
            'inner': {
                '0': {
                    'counts': {'A': 13},
                    'inner': {}}}}}

    add_config = {
        'path': ['environment', 'inner'],
        'add': {
            '1': {
                'counts': {
                    'A': 8}}}}

    schema, state = core.infer_schema(
        {},
        single_node)

    assert '0' in state['environment']['inner']
    assert '1' not in state['environment']['inner']

    result = core.apply(
        schema,
        state, {
            '_react': {
                'add': add_config}})

            # '_react': {
            #     'reaction': 'add',
            #     'config': add_config}})

    assert '0' in result['environment']['inner']
    assert '1' in result['environment']['inner']


def test_remove_reaction(core):
    single_node = {
        'environment': {
            '_type': 'compartment',
            'counts': {'A': 144},
            'inner': {
                '0': {
                    'counts': {'A': 13},
                    'inner': {}},
                '1': {
                    'counts': {'A': 13},
                    'inner': {}}}}}

    remove_config = {
        'path': ['environment', 'inner'],
        'remove': ['0']}

    schema, state = core.infer_schema(
        {},
        single_node)

    assert '0' in state['environment']['inner']
    assert '1' in state['environment']['inner']

    result = core.apply(
        schema,
        state, {
            '_react': {
                'remove': remove_config}})

    assert '0' not in result['environment']['inner']
    assert '1' in state['environment']['inner']
    

def test_replace_reaction(core):
    single_node = {
        'environment': {
            '_type': 'compartment',
            'counts': {'A': 144},
            'inner': {
                '0': {
                    'counts': {'A': 13},
                    'inner': {}},
                '1': {
                    'counts': {'A': 13},
                    'inner': {}}}}}

    # replace_config = {
    #     'path': ['environment', 'inner'],
    #     'before': {'0': {'A': '?1'}},
    #     'after': {
    #         '2': {
    #             'counts': {
    #                 'A': {'function': 'divide', 'arguments': ['?1', 0.5], }}},
    #         '3': {
    #             'counts': {
    #                 'A': '@1'}}}}

    replace_config = {
        'path': ['environment', 'inner'],
        'before': {'0': {}},
        'after': {
            '2': {
                'counts': {
                    'A': 3}},
            '3': {
                'counts': {
                    'A': 88}}}}

    schema, state = core.infer_schema(
        {},
        single_node)

    assert '0' in state['environment']['inner']
    assert '1' in state['environment']['inner']

    result = core.apply(
        schema,
        state, {
            '_react': {
                'replace': replace_config}})

    assert '0' not in result['environment']['inner']
    assert '1' in result['environment']['inner']
    assert '2' in result['environment']['inner']
    assert '3' in result['environment']['inner']


def test_reaction(core):
    single_node = {
        'environment': {
            'counts': {},
            'inner': {
                '0': {
                    'counts': {}}}}}

    # TODO: compartment type ends up as 'any' at leafs?

    # TODO: come at divide reaction from the other side:
    #   ie make a call for it, then figure out what the
    #   reaction needs to be
    def divide_reaction(container, mother, divider):
        daughters = divider(mother)

        return {
            'redex': mother,
            'reactum': daughters}

    embedded_tree = {
        'environment': {
            '_type': 'compartment',
            'counts': {},
            'inner': {
                'agent1': {
                    '_type': 'compartment',
                    'counts': {},
                    'inner': {
                        'agent2': {
                            '_type': 'compartment',
                            'counts': {},
                            'inner': {},
                            'transport': {
                                'wires': {
                                    'outer': ['..', '..'],
                                    'inner': ['inner']}}}},
                    'transport': {
                        'wires': {
                            'outer': ['..', '..'],
                            'inner': ['inner']}}}}}}


    mother_tree = {
        'environment': {
            '_type': 'compartment',
            'counts': {
                'A': 15},
            'inner': {
                'mother': {
                    '_type': 'compartment',
                    'counts': {
                        'A': 5}}}}}

    divide_react = {
        '_react': {
            'redex': {
                'mother': {
                    'counts': '@counts'}},
            'reactum': {
                'daughter1': {
                    'counts': '@daughter1_counts'},
                'daughter2': {
                    'counts': '@daughter2_counts'}},
            'calls': [{
                'function': 'divide_counts',
                'arguments': ['@counts', [0.5, 0.5]],
                'bindings': ['@daughter1_counts', '@daughter2_counts']}]}}

    divide_update = {
        '_react': {
            'reaction': 'divide_counts',
            'config': {
                'id': 'mother',
                'state_key': 'counts',
                'daughters': [
                    {'id': 'daughter1', 'ratio': 0.3},
                    {'id': 'daughter2', 'ratio': 0.7}]}}}

    divide_update_concise = {
        '_react': {
            'divide_counts': {
                'id': 'mother',
                'state_key': 'counts',
                'daughters': [
                    {'id': 'daughter1', 'ratio': 0.3},
                    {'id': 'daughter2', 'ratio': 0.7}]}}}


def test_map_type(core):
    schema = 'map[integer]'

    state = {
        'a': 12,
        'b': 13,
        'c': 15,
        'd': 18}

    update = {
        'b': 44,
        'd': 111}

    assert core.check(schema, state)
    assert core.check(schema, update)
    assert not core.check(schema, 15)

    result = core.apply(
        schema,
        state,
        update)

    assert result['a'] == 12
    assert result['b'] == 57
    assert result['d'] == 129

    encode = core.serialize(schema, update)
    assert encode['d'] == '111'

    decode = core.deserialize(schema, encode)
    assert decode == update


def test_tree_type(core):
    schema = 'tree[maybe[integer]]'

    state = {
        'a': 12,
        'b': 13,
        'c': {
            'e': 5555,
            'f': 111},
        'd': None}

    update = {
        'a': None,
        'c': {
            'e': 88888,
            'f': 2222,
            'G': None},
        'd': 111}

    assert core.check(schema, state)
    assert core.check(schema, update)
    assert core.check(schema, 15)
    assert core.check(schema, None)
    assert core.check(schema, {'c': {'D': None, 'e': 11111}})
    assert not core.check(schema, 'yellow')
    assert not core.check(schema, {'a': 5, 'b': 'green'})
    assert not core.check(schema, {'c': {'D': False, 'e': 11111}})
    
    result = core.apply(
        schema,
        state,
        update)

    assert result['a'] == None
    assert result['b'] == 13
    assert result['c']['f'] == 2333
    assert result['d'] == 111

    encode = core.serialize(schema, update)
    assert encode['a'] == NONE_SYMBOL
    assert encode['d'] == '111'

    decode = core.deserialize(schema, encode)
    assert decode == update


def test_maybe_type(core):
    schema = 'map[maybe[integer]]'

    state = {
        'a': 12,
        'b': 13,
        'c': None,
        'd': 18}

    update = {
        'a': None,
        'c': 44,
        'd': 111}

    assert core.check(schema, state)
    assert core.check(schema, update)
    assert not core.check(schema, 15)
    
    result = core.apply(
        schema,
        state,
        update)

    assert result['a'] == None
    assert result['b'] == 13
    assert result['c'] == 44
    assert result['d'] == 129

    encode = core.serialize(schema, update)
    assert encode['a'] == NONE_SYMBOL
    assert encode['d'] == '111'

    decode = core.deserialize(schema, encode)
    assert decode == update


def test_tuple_type(core):
    schema = {
        '_type': 'tuple',
        '_type_parameters': ['0', '1', '2'],
        '_0': 'string',
        '_1': 'int',
        '_2': 'map[maybe[float]]'}

    schema = ('string', 'int', 'map[maybe[float]]')
    schema = 'tuple[string,int,map[maybe[float]]]'
    schema = 'string|integer|map[maybe[float]]'

    state = (
        'aaaaa',
        13, {
            'a': 1.1,
            'b': None})

    update = (
        'bbbbbb',
        10, {
            'a': 33.33,
            'b': 4.44444})

    assert core.check(schema, state)
    assert core.check(schema, update)
    assert not core.check(schema, 15)
    
    result = core.apply(
        schema,
        state,
        update)

    assert len(result) == 3
    assert result[0] == update[0]
    assert result[1] == 23
    assert result[2]['a'] == 34.43
    assert result[2]['b'] == update[2]['b']

    encode = core.serialize(schema, state)
    assert encode[2]['b'] == NONE_SYMBOL
    assert encode[1] == '13'

    decode = core.deserialize(schema, encode)
    assert decode == state

    tuple_type = core.access('(3|4|10)')
    assert '_2' in tuple_type
    assert tuple_type['_2'] == '10'

    tuple_type = core.access('tuple[9,float,7]')
    assert '_2' in tuple_type
    assert tuple_type['_2'] == '7'



def test_union_type(core):
    schema = {
        '_type': 'union',
        '_type_parameters': ['0', '1', '2'],
        '_0': 'string',
        '_1': 'integer',
        '_2': 'map[maybe[float]]'}

    schema = 'string~integer~map[maybe[float]]'

    state = {
        'a': 1.1,
        'b': None}

    update = {
        'a': 33.33,
        'b': 4.44444}

    assert core.check(schema, state)
    assert core.check(schema, update)
    assert core.check(schema, 15)
    
    wrong_state = {
        'a': 1.1,
        'b': None}

    wrong_update = 'a different type'

    assert core.check(schema, wrong_state)
    assert core.check(schema, wrong_update)
    
    # TODO: deal with union apply of different types

    result = core.apply(
        schema,
        state,
        update)

    assert result['a'] == 34.43
    assert result['b'] == update['b']

    encode = core.serialize(schema, state)
    assert encode['b'] == NONE_SYMBOL

    decode = core.deserialize(schema, encode)
    assert decode == state


def test_union_values(core):
    schema = 'map[integer~string~map[maybe[float]]]'

    state = {
        'a': 'bbbbb',
        'b': 15}

    update = {
        'a': 'aaaaa',
        'b': 22}

    assert core.check(schema, state)
    assert core.check(schema, update)
    assert not core.check(schema, 15)
    
    result = core.apply(
        schema,
        state,
        update)

    assert result['a'] == 'aaaaa'
    assert result['b'] == 37

    encode = core.serialize(schema, state)
    decode = core.deserialize(schema, encode)

    assert decode == state


def test_array_type(core):
    shape = (3, 4, 10)
    shape_representation = core.representation(shape)
    shape_commas = ','.join([
        str(x)
        for x in shape])

    schema = {
        '_type': 'map',
        '_value': {
            '_type': 'array',
            # '_shape': '(3|4|10)',
            '_shape': shape_representation,
            '_data': 'float'}}

    schema = f'map[array[tuple[{shape_commas}],float]]'
    schema = f'map[array[{shape_representation},float]]'

    state = {
        'a': np.zeros(shape),
        'b': np.ones(shape)}

    update = {
        'a': np.full(shape, 5.555),
        'b': np.full(shape, 9.999)}

    assert core.check(schema, state)
    assert core.check(schema, update)
    assert not core.check(schema, 15)
    
    result = core.apply(
        schema,
        state,
        update)

    assert result['a'][0, 0, 0] == 5.555
    assert result['b'][0, 0, 0] == 10.999

    encode = core.serialize(schema, state)
    assert encode['b']['shape'] == list(shape)
    assert encode['a']['data'] == 'float'

    decode = core.deserialize(schema, encode)

    for key in state:
        assert np.equal(
            decode[key],
            state[key]).all()

    found = core.find(
        schema)

    default = core.default(
        found['_value'])

    assert default.shape == shape


def test_infer_edge(core):
    initial_schema = {}
    initial_state = {
        'fade': {
            '_type': 'edge',
            '_inputs': {
                'yellow': 'array[(3|4|10),float]'},
            '_outputs': {
                'green': 'array[(11|5|8),float]'},
            'inputs': {
                'yellow': ['yellow']},
            'outputs': {
                'green': ['green']}}}

    update = {
        'yellow': np.ones((3, 4, 10)),
        'fade': {
            'inputs': {
                'yellow': ['red']},
            'outputs': {
                'green': ['green', 'green', 'green']}}}

    schema, state = core.complete(
        initial_schema,
        initial_state)

    assert core.check(schema, state)
    assert not core.check(schema, 15)

    result = core.apply(
        schema,
        state,
        update)

    assert result['yellow'][0, 0, 0] == 1.0
    assert result['fade']['inputs']['yellow'] == ['red']

    encode = core.serialize(schema, state)
    decode = core.deserialize(schema, encode)

    assert np.equal(
        decode['yellow'],
        state['yellow']).all()


def test_edge_type(core):
    schema = {
        'fade': {
            '_type': 'edge',
            '_inputs': {
                'yellow': {
                    '_type': 'array',
                    '_shape': 'tuple(3,4,10)',
                    '_data': 'float'}},
            '_outputs': {
                'green': {
                    '_type': 'array',
                    '_shape': 'tuple(11,5,8)',
                    '_data': 'float'}}}}

    initial_schema = {
        'fade': 'edge[yellow:array[(3|4|10),float],green:array[(11|5|8),float]]'}

    initial_state = {
        # 'yellow': np.zeros((3, 4, 10)),
        # 'green': np.ones((11, 5, 8)),
        'fade': {
            'inputs': {
                'yellow': ['yellow']},
            'outputs': {
                'green': ['green']}}}

    schema, state = core.complete(
        initial_schema,
        initial_state)

    update = {
        'yellow': np.ones((3, 4, 10)),
        'fade': {
            'inputs': {
                'yellow': ['red']},
            'outputs': {
                'green': ['green', 'green', 'green']}}}

    assert core.check(schema, state)
    assert not core.check(schema, 15)

    result = core.apply(
        schema,
        state,
        update)

    assert result['yellow'][0, 0, 0] == 1.0
    assert result['fade']['inputs']['yellow'] == ['red']

    encode = core.serialize(schema, state)
    decode = core.deserialize(schema, encode)

    assert np.equal(
        decode['yellow'],
        state['yellow']).all()


def test_edge_complete(core):
    edge_schema = {
        '_type': 'edge',
        '_inputs': {
            'concentration': 'float',
            'field': 'map[boolean]'},
        '_outputs': {
            'target': 'boolean',
            # 'inner': {
            #     'nested': 'boolean'},
            'total': 'integer',
            'delta': 'float'}}    

    edge_state = {
        'inputs': {
            'concentration': ['molecules', 'glucose'],
            'field': ['states']},
        'outputs': {
            'target': ['states', 'X'],
            # 'inner': {
            #     'nested': ['states', 'A']},
            'total': ['emitter', 'total molecules'],
            'delta': ['molecules', 'glucose']}}

    # edge_state = {
    #     'inputs': {
    #         'concentration': ['..', 'molecules', 'glucose'],
    #         'field': ['..', 'states']},
    #     'outputs': {
    #         'target': ['..', 'states', 'X'],
    #         'total': ['..', 'emitter', 'total molecules'],
    #         'delta': ['..', 'molecules', 'glucose']}}

    full_schema, full_state = core.complete(
        {'edge': edge_schema},
        {'edge': edge_state})

    assert full_schema['states']['_type'] == 'map'



def test_divide(core):
    schema = {
        'a': 'tree[maybe[float]]',
        'b': 'float~list[string]',
        'c': {
            'd': 'map[edge[GGG:float,OOO:float]]',
            'e': 'array[(3|4|10),float]'}}

    state = {
        'a': {
            'x': {
                'oooo': None,
                'y': 1.1,
                'z': 33.33},
            'w': 44.444},
        'b': ['1', '11', '111', '1111'],
        'c': {
            'd': {
                'A': {
                    'inputs': {
                        'GGG': ['..', '..', 'a', 'w']},
                    'outputs': {
                        'OOO': ['..', '..', 'a', 'x', 'y']}},
                'B': {
                    'inputs': {
                        'GGG': ['..', '..', 'a', 'x', 'y']},
                    'outputs': {
                        'OOO': ['..', '..', 'a', 'w']}}},
            'e': np.zeros((3, 4, 10))}}

    divisions = 3
    division = core.fold(
        schema,
        state,
        'divide',
        {'divisions': divisions})

    assert len(division) == divisions
    assert 'a' in division[0].keys()
    assert len(division[1]['b']) == len(state['b'])


def test_merge(core):
    current_schema = {
        'a': 'tree[maybe[float]]',
        'b': 'float~list[string]',
        'c': {
            'd': 'map[edge[GGG:float,OOO:float]]',
            'e': 'array[(3|4|10),float]'}}

    current_state = {
        'a': {
            'x': {
                'oooo': None,
                'y': 1.1,
                'z': 33.33},
            'w': 44.444},
        'b': ['1', '11', '111', '1111'],
        'c': {
            'd': {
                'A': {
                    'inputs': {
                        'GGG': ['..', '..', 'a', 'w']},
                    'outputs': {
                        'OOO': ['..', '..', 'a', 'x', 'y']}},
                'B': {
                    'inputs': {
                        'GGG': ['..', '..', 'a', 'x', 'y']},
                    'outputs': {
                        'OOO': ['..', '..', 'a', 'w']}}},
            'e': np.zeros((3, 4, 10))}}

    merge_state = {
        'z': 555.55,
        'b': ['333333333'],
        'a': {
            'x': {
                'x': {
                    'o': 99999.11}}}}

    result = core.merge(
        current_schema,
        current_state,
        merge_state)

    assert result['z'] == merge_state['z']
    assert result['b'] == merge_state['b']
    assert result['a']['x']['x']['o'] == merge_state['a']['x']['x']['o']


def test_bind(core):
    current_schema = {
        'a': 'tree[maybe[float]]',
        'b': 'float~list[string]',
        'c': {
            'd': 'map[edge[GGG:float,OOO:float]]',
            'e': 'array[(3|4|10),float]'}}

    current_state = {
        'a': {
            'x': {
                'oooo': None,
                'y': 1.1,
                'z': 33.33},
            'w': 44.444},
        'b': ['1', '11', '111', '1111'],
        'c': {
            'd': {
                'A': {
                    'inputs': {
                        'GGG': ['..', '..', 'a', 'w']},
                    'outputs': {
                        'OOO': ['..', '..', 'a', 'x', 'y']}},
                'B': {
                    'inputs': {
                        'GGG': ['..', '..', 'a', 'x', 'y']},
                    'outputs': {
                        'OOO': ['..', '..', 'a', 'w']}}},
            'e': np.zeros((3, 4, 10))}}

    result_schema, result_state = core.bind(
        current_schema,
        current_state,
        'z',
        'float',
        555.55)

    assert result_schema['z']['_type'] == 'float'
    assert result_state['z'] == 555.55


def test_slice(core):
    schema, state = core.slice(
        'map[float]',
        {'aaaa': 55.555},
        ['aaaa'])

    schema, state = core.complete({}, {
        'top': {
            '_type': 'tree[list[maybe[(float|integer)~string]]]',
            'AAAA': {
                'BBBB': {
                    'CCCC': [
                        (1.3, 5),
                        'okay',
                        (55.555, 1),
                        None,
                        'what',
                        'is']}},
            'DDDD': [
                (3333.1, 88),
                'in',
                'between',
                (66.8, -3),
                None,
                None,
                'later']}})

    float_schema, float_state = core.slice(
        schema,
        state,
        ['top', 'AAAA', 'BBBB', 'CCCC', 2, 0])

    assert float_schema['_type'] == 'float'
    assert float_state == 55.555

    assert core.slice(
        schema,
        state,
        ['top', 'AAAA', 'BBBB', 'CCCC', 3])[1] is None


def test_set_slice(core):
    float_schema, float_state = core.set_slice(
        'map[float]',
        {'aaaa': 55.555},
        ['bbbbb'],
        'float',
        888.88888)

    assert float_schema['_type'] == 'map'
    assert float_state['bbbbb'] == 888.88888

    schema, state = core.complete({}, {
        'top': {
            '_type': 'tree[list[maybe[(float|integer)~string]]]',
            'AAAA': {
                'BBBB': {
                    'CCCC': [
                        (1.3, 5),
                        'okay',
                        (55.555, 1),
                        None,
                        'what',
                        'is']}},
            'DDDD': [
                (3333.1, 88),
                'in',
                'between',
                (66.8, -3),
                None,
                None,
                'later']}})

    leaf_schema, leaf_state = core.set_slice(
        schema,
        state,
        ['top', 'AAAA', 'BBBB', 'CCCC', 2, 1],
        'integer',
        33)

    assert core.slice(
        leaf_schema,
        leaf_state, [
            'top',
            'AAAA',
            'BBBB',
            'CCCC',
            2,
            1])[1] == 33


def from_state(dataclass, state):
    if hasattr(dataclass, '__dataclass_fields__'):
        fields = dataclass.__dataclass_fields__
        state = state or {}

        init = {}
        for key, field in fields.items():
            substate = from_state(
                field.type,
                state.get(key))
            init[key] = substate
        instance = dataclass(**init)
    # elif get_origin(dataclass) in [typing.Union, typing.Mapping]:
    #     instance = state
    else:
        instance = state
        # instance = dataclass(state)

    return instance


def test_dataclass(core):
    simple_schema = {
        'a': 'float',
        'b': 'integer',
        'c': 'boolean',
        'x': 'string'}

    # TODO: accept just a string instead of only a path
    simple_dataclass = core.dataclass(
        simple_schema,
        ['simple'])

    simple_state = {
        'a': 88.888,
        'b': 11111,
        'c': False,
        'x': 'not a string'}

    simple_new = simple_dataclass(
        a=1.11,
        b=33,
        c=True,
        x='what')

    simple_from = from_state(
       simple_dataclass,
       simple_state)

    nested_schema = {
        'a': {
            'a': {
                'a': 'float',
                'b': 'float'},
            'x': 'float'}}

    nested_dataclass = core.dataclass(
        nested_schema,
        ['nested'])

    nested_state = {
        'a': {
            'a': {
                'a': 13.4444,
                'b': 888.88},
            'x': 111.11111}}

    nested_new = data.nested(
        data.nested_a(
            data.nested_a_a(
                a=222.22,
                b=3.3333),
            5555.55))

    nested_from = from_state(
        nested_dataclass,
        nested_state)

    complex_schema = {
        'a': 'tree[maybe[float]]',
        'b': 'float~list[string]',
        'c': {
            'd': 'map[edge[GGG:float,OOO:float]]',
            'e': 'array[(3|4|10),float]'}}

    complex_dataclass = core.dataclass(
        complex_schema,
        ['complex'])

    complex_state = {
        'a': {
            'x': {
                'oooo': None,
                'y': 1.1,
                'z': 33.33},
            'w': 44.444},
        'b': ['1', '11', '111', '1111'],
        'c': {
            'd': {
                'A': {
                    'inputs': {
                        'GGG': ['..', '..', 'a', 'w']},
                    'outputs': {
                        'OOO': ['..', '..', 'a', 'x', 'y']}},
                'B': {
                    'inputs': {
                        'GGG': ['..', '..', 'a', 'x', 'y']},
                    'outputs': {
                        'OOO': ['..', '..', 'a', 'w']}}},
            'e': np.zeros((3, 4, 10))}}

    complex_from = from_state(
        complex_dataclass,
        complex_state)

    complex_dict = asdict(complex_from)

    # assert complex_dict == complex_state ? 

    assert complex_from.a['x']['oooo'] is None
    assert len(complex_from.c.d['A']['inputs']['GGG'])
    assert isinstance(complex_from.c.e, np.ndarray)


def test_enum_type(core):
    core.register(
        'planet',
        'enum[mercury,venus,earth,mars,jupiter,saturn,neptune]')

    # core.register('planet', {
    #     '_type': 'enum',
    #     '_type_parameters': ['0', '1', '2', '3', '4', '5', '6'],
    #     '_0': 'mercury',
    #     '_1': 'venus',
    #     '_2': 'earth',
    #     '_3': 'mars',
    #     '_4': 'jupiter',
    #     '_5': 'saturn',
    #     '_6': 'neptune'})

    assert core.default('planet') == 'mercury'

    solar_system_schema = {
        'planets': 'map[planet]'}

    solar_system = {
        'planets': {
            '3': 'earth',
            '4': 'mars'}}

    jupiter_update = {
        'planets': {
            '5': 'jupiter'}}

    pluto_update = {
        'planets': { 
            '7': 'pluto'}}

    assert core.check(
        solar_system_schema,
        solar_system)

    assert core.check(
        solar_system_schema,
        jupiter_update)

    assert not core.check(
        solar_system_schema,
        pluto_update)

    with_jupiter = core.apply(
        solar_system_schema,
        solar_system,
        jupiter_update)

    try:
        core.apply(
            solar_system_schema,
            solar_system,
            pluto_update)

        assert False
    except Exception as e:
        print(e)
        assert True


def test_map_schema(core):
    schema = {
        'greetings': 'map[hello:string]',
        'edge': {
            '_type': 'edge',
            '_inputs': {
                'various': {
                    '_type': 'map',
                    '_value': {
                        'world': 'string'}}},
            '_outputs': {
                'referent': 'float'}}}

    state = {
        'edge': {
            'inputs': {
                'various': ['greetings']},
            'outputs': {
                'referent': ['where']}},

        'greetings': {
            'a': {
                'hello': 'yes'},
            'b': {
                'hello': 'again',
                'world': 'present'},
            'c': {
                'other': 'other'}}}

    complete_schema, complete_state = core.complete(
        schema,
        state)

    assert complete_schema['greetings']['_value']['hello']['_type'] == 'string'
    assert complete_schema['greetings']['_value']['world']['_type'] == 'string'

    assert 'world' in complete_state['greetings']['a']
    assert complete_schema['greetings']['_value']['world']['_type'] == 'string'


def test_representation(core):
    schema_examples = [
        'map[float]',
        '(string|float)',
        'tree[(a:float|b:map[string])]',
        'array[(5|11),maybe[integer]]',
        'edge[(x:float|y:tree[(z:float)]),(w:(float|float|float))]']

    for example in schema_examples:
        full_type = core.access(example)
        representation = core.representation(full_type)

        if example != representation:
            raise Exception(f'did not receive the same type after parsing and finding the representation:\n  {example}\n  {representation}')


def test_generate(core):
    schema = {
        'A': 'float',
        'B': 'enum[one,two,three]',
        'units': 'map[float]'}

    state = {
        'C': {
            '_type': 'enum[x,y,z]',
            '_value': 'y'},
        'units': {
            'x': 11.1111,
            'y': 22.833333}}

    import ipdb; ipdb.set_trace()

    generated_schema, generated_state = core.generate(
        schema,
        state)

    import ipdb; ipdb.set_trace()


def test_edge_cycle(core):
    empty_schema = {}
    empty_state = {}

    A_schema = {
        'A': {
            '_type': 'metaedge',
            '_inputs': {
                'before': {
                    'inputs': {'before': {'_default': ['B']}},
                    'outputs': {'after': {'_default': ['A']}}}},
            '_outputs': {
                'after': {
                    'inputs': {'before': {'_default': ['A']}},
                    'outputs': {'after': {'_default': ['C']}}}},
            'inputs': {'before': {'_default': ['C']}},
            'outputs': {'after': {'_default': ['B']}}}}

    A_state = {
        'A': {
            '_type': 'metaedge',
            '_inputs': {
                'before': {
                    'inputs': {'before': {'_default': ['B']}},
                    'outputs': {'after': {'_default': ['A']}}}},
            '_outputs': {
                'after': {
                    'inputs': {'before': {'_default': ['A']}},
                    'outputs': {'after': {'_default': ['C']}}}},
            'inputs': {'before': {'_default': ['C']}},
            'outputs': {'after': {'_default': ['B']}}}}

    schema_from_schema, state_from_schema = core.generate(
        A_schema,
        empty_state)

    schema_from_state, state_from_state = core.generate(
        empty_schema,
        A_state)

    # print(diff(schema_from_schema, schema_from_state))
    # print(diff(state_from_schema, state_from_state))

    assert schema_from_schema == schema_from_state
    assert state_from_schema == state_from_state


if __name__ == '__main__':
    core = TypeSystem()
    core = register_test_types(core)

    test_reregister_type(core)
    test_generate_default(core)
    test_apply_update(core)
    test_validate_schema(core)
    test_fill_integer(core)
    test_fill_cube(core)
    test_establish_path(core)
    test_overwrite_existing(core)
    test_fill_in_missing_nodes(core)
    test_fill_from_parse(core)
    test_fill_ports(core)
    test_expected_schema(core)
    test_units(core)
    test_serialize_deserialize(core)
    test_project(core)
    test_inherits_from(core)
    test_apply_schema(core)
    test_resolve_schemas(core)
    test_add_reaction(core)
    test_remove_reaction(core)
    test_replace_reaction(core)
    test_unit_conversion(core)
    test_map_type(core)
    test_tree_type(core)
    test_maybe_type(core)
    test_tuple_type(core)
    test_array_type(core)
    test_union_type(core)
    test_union_values(core)
    test_infer_edge(core)
    test_edge_type(core)
    test_edge_complete(core)
    test_foursquare(core)
    test_divide(core)
    test_merge(core)
    test_bind(core)
    test_slice(core)
    test_set_slice(core)
    test_dataclass(core)
    test_enum_type(core)
    test_map_schema(core)
    test_representation(core)
    test_generate(core)
    test_edge_cycle(core)
