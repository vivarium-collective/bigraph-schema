"""
========
Registry
========

Utilities for managing registry entries, hierarchical state structures,
deep merging, and schema cleaning. Includes the `Registry` class for storing
type declarations or functions by key.
"""

import inspect
import copy
import collections
import traceback
import functools
import numpy as np
import pytest
from pprint import pformat as pf

from bigraph_schema.protocols import local_lookup_module, function_module

# --- Merge Utilities ---------------------------------------------------------

def deep_merge_copy(dct, merge_dct):
    """Return a deep copy of `dct` with `merge_dct` deeply merged into it."""
    return deep_merge(copy.deepcopy(dct), merge_dct)

def deep_merge(dct, merge_dct):
    """
    Deep merge `merge_dct` into `dct`, modifying `dct` in-place.

    Nested dictionaries are recursively merged.
    """
    if dct is None:
        dct = {}
    if merge_dct is None:
        merge_dct = {}
    if not isinstance(merge_dct, dict):
        return merge_dct

    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(v, collections.abc.Mapping)):
            deep_merge(dct[k], v)
        else:
            dct[k] = v
    return dct

def validate_merge(state, dct, merge_dct):
    """
    Like `deep_merge`, but raises an exception if values conflict and are not in `state`.
    """
    dct = dct or {}
    merge_dct = merge_dct or {}
    state = state or {}

    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(v, collections.abc.Mapping)):
            if k not in state:
                state[k] = {}
            validate_merge(state[k], dct[k], v)
        else:
            if k in state:
                dct[k] = state[k]
            elif k in dct and dct[k] != v:
                raise Exception(f'cannot merge dicts at key "{k}":\n{dct}\n{merge_dct}')
            else:
                dct[k] = v
    return dct

# --- Tree Path Utilities -----------------------------------------------------

def establish_path(tree, path, top=None, cursor=()):
    """
    Create or traverse a path in a nested dictionary (tree structure).
    """
    if tree is None:
        tree = {}
    if top is None:
        top = tree
    if path is None or len(path) == 0:
        return tree
    if isinstance(path, str):
        path = (path,)
    head = path[0]
    if head == '..':
        if len(cursor) == 0:
            raise Exception(f'trying to travel above the top of the tree: {path}')
        return establish_path(top, cursor[:-1])
    if head not in tree:
        tree[head] = {}
    return establish_path(tree[head], path[1:], top=top, cursor=cursor + (head,))

def set_path(tree, path, value, top=None, cursor=None):
    """
    Set `value` at the given `path` in a tree-like dictionary.
    """
    if value is None:
        return None
    if len(path) == 0:
        return value
    final = path[-1]
    destination = establish_path(tree, path[:-1])
    destination[final] = value
    return tree

def set_star_path(tree, path, value, top=None, cursor=()):
    """
    Set a value at a wildcard (`*`) path, filling in multiple keys.
    """
    if tree is None:
        tree = {}
    if top is None:
        top = tree
    if isinstance(path, str):
        path = (path,)
    if len(path) == 0:
        return tree

    head, tail = path[0], path[1:]
    if head == '..':
        if len(cursor) == 0:
            raise Exception(f'trying to travel above the top of the tree: {path}')
        return set_star_path(top, cursor[:-1], value)
    elif head == '*':
        for key in value:
            tree[key] = set_star_path({}, tail, value[key], cursor=(key,))
        return top
    else:
        if len(tail) == 0:
            tree[head] = value
        else:
            if head not in tree:
                tree[head] = {}
            set_star_path(tree[head], tail, value, top=top, cursor=cursor + (head,))
        return top

def transform_path(tree, path, transform):
    """
    Apply a transformation function to a value at a specific path in a tree.
    """
    before = establish_path(tree, path)
    after = transform(before)
    return set_path(tree, path, after)

def hierarchy_depth(hierarchy, path=()):
    """
    Recursively collect all node paths in a hierarchy.
    """
    base = {}
    for key, inner in hierarchy.items():
        down = path + (key,)
        if is_schema_key(key):
            base[path] = inner
        elif isinstance(inner, dict) and 'instance' not in inner:
            base.update(hierarchy_depth(inner, down))
        else:
            base[down] = inner
    return base

# --- Schema Tools ------------------------------------------------------------

def remove_omitted(before, after, tree):
    """
    Remove keys from `tree` that exist in `before` but not in `after`.
    """
    if isinstance(before, dict):
        if not isinstance(tree, dict):
            raise Exception(f'trying to remove an entry from non-dict: {tree}')
        if not isinstance(after, dict):
            return after
        for key, down in before.items():
            if not key.startswith('_'):
                if key not in after:
                    tree.pop(key, None)
                else:
                    tree[key] = remove_omitted(down, after[key], tree[key])
    return tree

def is_schema_key(key):
    """Check if a key is a schema key (starts with underscore)."""
    return isinstance(key, str) and key.startswith('_')

def non_schema_keys(schema):
    """Return list of non-schema keys from a dictionary."""
    return [k for k in schema if not is_schema_key(k)]

def strip_schema_keys(state):
    """
    Recursively remove all schema keys from a dictionary.
    """
    if isinstance(state, dict):
        return {k: strip_schema_keys(v) for k, v in state.items() if not is_schema_key(k)}
    return state

def type_parameter_key(schema, key):
    """Check if a key is a special parameter override."""
    return key.strip('_') not in schema.get('_type_parameters', []) and key.startswith('_')

def default(type, default):
    """Return a schema dictionary with a type and default."""
    return {'_type': type, '_default': default}

# --- Registry ----------------------------------------------------------------

class Registry:
    """
    A registry for managing keyed objects or functions (e.g., schema types).

    Supports registering items under multiple keys, deep merging for dicts,
    and loading functions by string-qualified names.
    """

    def __init__(self, function_keys=None):
        self.registry = {}
        self.main_keys = set()
        self.function_keys = set(function_keys or [])

    def register(self, key, item, alternate_keys=(), strict=False):
        """
        Register an item under a key (and optional alternate keys).
        Optionally enforce strict uniqueness.

        If the item is a dict and already registered, a deep merge is attempted.
        """
        keys = [key] + list(alternate_keys)

        for registry_key in keys:
            if registry_key in self.registry:
                if item != self.registry[registry_key]:
                    if strict:
                        raise Exception(
                            f'Registry conflict for {registry_key}: {self.registry[registry_key]} vs {item}')
                    elif isinstance(item, dict):
                        self.registry[registry_key] = deep_merge(self.registry[registry_key], item)
                    else:
                        self.registry[registry_key] = item
            else:
                self.registry[registry_key] = item

        self.main_keys.add(key)

    def register_function(self, function):
        """
        Register a function by name or function object.

        Returns:
            (function_name, module_key)
        """
        if isinstance(function, str):
            module_key = function
            found = self.find(module_key) or local_lookup_module(module_key)
            if found is None:
                raise Exception(f'Function "{module_key}" not found for type data')
        elif inspect.isfunction(function):
            found = function
            module_key = function_module(found)
        else:
            raise TypeError(f'Unsupported function type: {type(function)}')

        function_name = module_key.split('.')[-1]
        self.register(function_name, found)
        self.register(module_key, found)

        return function_name, module_key

    def register_multiple(self, schemas, strict=False):
        """Register multiple schemas from a dictionary of {key: item}."""
        for key, schema in schemas.items():
            self.register(key, schema, strict=strict)

    def find(self, key):
        """Retrieve an item from the registry by key (or None)."""
        return self.registry.get(key)

    def access(self, key):
        """Alias for `find()`."""
        return self.find(key)

    def list(self):
        """List all primary (non-alternate) keys in the registry."""
        return list(self.main_keys)

    def validate(self, item):
        """Stub for validation logic (currently always True)."""
        return True
