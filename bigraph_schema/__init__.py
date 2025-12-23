# from bigraph_schema.registry import (
#     deep_merge, validate_merge, default, Registry, hierarchy_depth, is_schema_key, establish_path,
#     strip_schema_keys, type_parameter_key, non_schema_keys, set_path, transform_path)

# from bigraph_schema.utilities import get_path, visit_method

# from bigraph_schema.type_system import TypeSystem
# from bigraph_schema.type_functions import FUNCTION_TYPE, METHOD_TYPE, type_schema_keys, resolve_path


from bigraph_schema.edge import Edge
from bigraph_schema.protocols import local_lookup_module

from bigraph_schema.schema import BASE_TYPES, resolve_path, deep_merge, make_default
from bigraph_schema.core import Core, allocate_core

import bigraph_schema.methods


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

def is_schema_key(key):
    """Check if a key is a schema key (starts with underscore)."""
    return isinstance(key, str) and key.startswith('_')

def strip_schema_keys(state):
    """
    Recursively remove all schema keys from a dictionary.
    """
    if isinstance(state, dict):
        return {
            k: strip_schema_keys(v)
            for k, v in state.items()
            if not is_schema_key(k)}

    return state
