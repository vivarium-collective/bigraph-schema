"""
========
Registry
========
"""

import inspect
import copy
import collections
import pytest
import traceback
import functools
import numpy as np

from pprint import pformat as pf

from bigraph_schema.protocols import local_lookup_module, function_module



def deep_merge_copy(dct, merge_dct):
    return deep_merge(copy.deepcopy(dct), merge_dct)


def deep_merge(dct, merge_dct):
    """Recursive dict merge
    
    This mutates dct - the contents of merge_dct are added to dct (which is also returned).
    If you want to keep dct you could call it like deep_merge(copy.deepcopy(dct), merge_dct)
    """
    if dct is None:
        dct = {}
    if merge_dct is None:
        merge_dct = {}
    if not isinstance(merge_dct, dict):
        return merge_dct

    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.abc.Mapping)):
            deep_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]
    return dct


def validate_merge(state, dct, merge_dct):
    """Recursive dict merge
    
    This mutates dct - the contents of merge_dct are added to dct (which is also returned).
    If you want to keep dct you could call it like deep_merge(copy.deepcopy(dct), merge_dct)
    """
    dct = dct or {}
    merge_dct = merge_dct or {}
    state = state or {}

    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.abc.Mapping)):
            if k not in state:
                state[k] = {}

            validate_merge(
                state[k],
                dct[k],
                merge_dct[k])
        else:
            if k in state:
                dct[k] = state[k]
            elif k in dct:
                if dct[k] != merge_dct[k]:
                    raise Exception(f'cannot merge dicts at key "{k}":\n{dct}\n{merge_dct}')
            else:
                dct[k] = merge_dct[k]
    return dct


def establish_path(tree, path, top=None, cursor=()):
    """
    Given a tree and a path in the tree that may or may not yet exist,
    add nodes along the path and return the final node which is now at the
    given path.
    
    Args:
    - tree: the tree we are establishing a path in
    - path: where the new subtree will be located in the tree
    - top: (None) a reference to the top of the tree
    - cursor: (()) the current location we are visiting in the tree
    
    Returns:
    - node: the new node of the tree that exists at the given path
    """

    if tree is None:
        tree = {}

    if top is None:
        top = tree
    if path is None or path == ():
        return tree
    elif len(path) == 0:
        return tree
    else:
        if isinstance(path, str):
            path = (path,)

        head = path[0]
        if head == '..':
            if len(cursor) == 0:
                raise Exception(
                    f'trying to travel above the top of the tree: {path}')
            else:
                return establish_path(
                    top,
                    cursor[:-1])
        else:
            if head not in tree:
                tree[head] = {}
            return establish_path(
                tree[head],
                path[1:],
                top=top,
                cursor=tuple(cursor) + (head,))


def set_path(tree, path, value, top=None, cursor=None):
    """
    Given a tree, a path, and a value, sets the location
    in the tree corresponding to the path to the given value
    
    Args:
    - tree: the tree we are setting a value in
    - path: where the new value will be located in the tree
    - value: the value to set at the given path in the tree
    - top: (None) a reference to the top of the tree
    - cursor: (()) the current location we are visiting in the tree
    
    Returns:
    - node: the new node of the tree that exists at the given path
    """

    if value is None:
        return None
    if len(path) == 0:
        return value

    final = path[-1]
    towards = path[:-1]
    destination = establish_path(tree, towards)
    destination[final] = value
    return tree


def set_star_path(tree, path, value, top=None, cursor=()):
    if tree is None:
        tree = {}
    if top is None:
        top = tree
    if path is None or len(path) == 0:
        return tree
    else:
        if isinstance(path, str):
            path = (path,)

        head = path[0]
        tail = path[1:]

        if head == '..':
            if len(cursor) == 0:
                raise Exception(
                    f'trying to travel above the top of the tree: {path}')
            else:
                return set_star_path(
                    top,
                    cursor[:-1],
                    value)
        elif head == '*':
            for key in value:
                tree[key] = set_star_path(
                    {},
                    tail,
                    value[key],
                    cursor=(key,))
            return top
        else:
            if len(tail) == 0:
                tree[head] = value
                return top
            else:
                if head not in tree:
                    tree[head] = {}
                return set_star_path(
                    tree[head],
                    tail,
                    value,
                    top=top,
                    cursor=tuple(cursor) + (head,))


def transform_path(tree, path, transform):
    """
    Given a tree, a path, and a transform (function), mutate the tree by replacing the subtree at the path by whatever 
    is returned from applying the transform to the existing value.
    
    Args:
    - tree: the tree we are setting a value in
    - path: where the new value will be located in the tree
    - transform: the function to apply to whatever currently lives at the given path in the tree
    
    Returns:
    - node: the node of the tree that exists at the given path
    """
    before = establish_path(tree, path)
    after = transform(before)

    return set_path(tree, path, after)


def hierarchy_depth(hierarchy, path=()):
    """
    Create a mapping of every path in the hierarchy to the node living at
    that path in the hierarchy.
    """

    base = {}

    for key, inner in hierarchy.items():
        down = tuple(path + (key,))
        if is_schema_key(key):
            base[path] = inner
        elif isinstance(inner, dict) and 'instance' not in inner:
            base.update(hierarchy_depth(inner, down))
        else:
            base[down] = inner

    return base


def remove_omitted(before, after, tree):
    """
    Removes anything in tree that was in before but not in after
    """

    if isinstance(before, dict):
        if not isinstance(tree, dict):
            raise Exception(
                f'trying to remove an entry from something that is not a dict: {tree}')

        if not isinstance(after, dict):
            return after

        for key, down in before.items():
            if not key.startswith('_'):
                if key not in after:
                    if key in tree:
                        del tree[key]
                else:
                    tree[key] = remove_omitted(
                        down,
                        after[key],
                        tree[key])

    return tree


def is_schema_key(key):
    return isinstance(key, str) and key.startswith('_')


def non_schema_keys(schema):
    """
    Filters out schema keys with the underscore prefix
    """
    return [
        element
        for element in schema.keys()
        if not is_schema_key(element)]


def strip_schema_keys(state):
    """remove schema keys from a state dictionary, including nested dictionaries"""
    if isinstance(state, dict):
        output = {}
        for key, value in state.items():
            if not is_schema_key(key):
                output[key] = strip_schema_keys(value)
    else:
        output = state
    return output


def type_parameter_key(schema, key):
    return key.strip('_') not in schema.get('_type_parameters', []) and key.startswith('_')


def default(type, default):
    return {
        '_type': type,
        '_default': default}


class Registry(object):
    """A Registry holds a collection of functions or objects"""

    def __init__(self, function_keys=None):
        function_keys = function_keys or []
        self.registry = {}
        self.main_keys = set([])
        self.function_keys = set(function_keys)

    def register(self, key, item, alternate_keys=tuple(), strict=False):
        """
        Add an item to the registry.

        Args:
        - key: Item key.
        - item: The item to add.
        - alternate_keys: Additional keys under which to register the item. These keys will not be included in the list
            returned by ``Registry.list()``. This may be useful if you want to be able to look up an item in the
            registry under multiple keys.
        - strict (bool): Disallow re-registration, overriding existing keys. False by default.
        """

        # check that registered function have the required function keys
        # TODO -- make this work to check the function keys
        if callable(item) and self.function_keys:
            sig = inspect.signature(item)
            sig_keys = set(sig.parameters.keys())
            # assert all(
            #     key in self.function_keys for key in sig_keys), f"Function '{item.__name__}' keys {sig_keys} are not all " \
            #                                                     f"in the function_keys {self.function_keys}"

        keys = [key]
        keys.extend(alternate_keys)
        for registry_key in keys:
            if registry_key in self.registry:
                if item != self.registry[registry_key]:
                    if strict:
                        raise Exception(
                            'registry already contains an entry for {}: {} --> {}'.format(
                                registry_key, self.registry[key], item))
                    elif isinstance(item, dict):
                        self.registry[registry_key] = deep_merge(
                            self.registry[registry_key],
                            item)
                    else:
                        self.registry[registry_key] = item

            else:
                self.registry[registry_key] = item
        self.main_keys.add(key)


    def register_function(self, function):
        if isinstance(function, str):
            module_key = function
            found = self.find(module_key)

            if found is None:
                found = local_lookup_module(
                    module_key)

                if found is None:
                    raise Exception(
                        f'function "{module_key}" not found for type data')

        elif inspect.isfunction(function):
            found = function
            module_key = function_module(found)
        
        function_name = module_key.split('.')[-1]
        self.register(function_name, found)
        self.register(module_key, found)

        return function_name, module_key


    def register_multiple(self, schemas, strict=False):
        for key, schema in schemas.items():
            self.register(key, schema, strict=strict)

    def find(self, key):
        return self.registry.get(key)
        

    def access(self, key):
        """
        get an item by key from the registry.
        """

        return self.find(key)

    def list(self):
        return list(self.main_keys)

    def validate(self, item):
        return True
