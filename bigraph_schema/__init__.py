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

def _has_top_level_schema_keys(state):
    """Scan ONLY the top-level dict keys for any underscore-prefixed
    string key. Used as a cheap pre-check by strip_schema_keys.

    Note: this is intentionally non-recursive. Recursive scanning ate
    ~0.27s per 30s sim and grew with state size. Process input views
    don't carry nested schema keys in practice — if they do, we still
    fall through to the recursive strip below.
    """
    if not isinstance(state, dict):
        return False
    for k in state:
        if isinstance(k, str) and k.startswith('_'):
            return True
    return False


def strip_schema_keys(state):
    """
    Recursively remove all schema keys from a dictionary.

    Fast path: if the top-level dict has no schema keys, return it
    unchanged (avoids the deep copy AND skips the recursive walk).
    Most process input views are already clean — the strip walk is
    wasted work in the common case.
    """
    if not isinstance(state, dict):
        return state
    if not _has_top_level_schema_keys(state):
        return state
    return {
        k: strip_schema_keys(v)
        for k, v in state.items()
        if not is_schema_key(k)}


def class_address(cls):
    """Return the ``local:!module.Class`` address used by realize() to
    import a Python class by dotted path.
    """
    return f'local:!{cls.__module__}.{cls.__name__}'


def tuples_to_lists(value):
    """Recursively convert tuples to lists in a tree. Useful for
    converting vivarium-style topology tuples into process-bigraph
    wire lists (JSON-friendly)."""
    if isinstance(value, tuple):
        return [tuples_to_lists(v) for v in value]
    if isinstance(value, list):
        return [tuples_to_lists(v) for v in value]
    if isinstance(value, dict):
        return {k: tuples_to_lists(v) for k, v in value.items()}
    return value


def make_arrays_writeable(state):
    """Recursively replace any read-only numpy arrays in a state tree
    with writeable copies. Parquet-loaded arrays are read-only by
    default, which breaks processes that mutate them in place.
    """
    import numpy as _np
    if isinstance(state, dict):
        for key, value in state.items():
            if isinstance(value, _np.ndarray):
                if not value.flags.writeable:
                    state[key] = value.copy()
                    state[key].flags.writeable = True
            elif isinstance(value, dict):
                make_arrays_writeable(value)
    return state


def capture_object_state(instance, skip=None, debug=False):
    """Snapshot the JSON-safe portion of a Python object's ``__dict__``.

    Objects that round-trip through ``__new__`` + ``__dict__`` (e.g. the
    ``Object`` / ``SharedProcess`` schemas) lose lazy/derived instance
    attributes unless they're in the declared config. This walks the
    instance's ``__dict__`` and returns a dict of capturable attrs:
    Python scalars, strings, ``None``, numpy scalars/arrays.

    Args:
        instance: the object to snapshot.
        skip: iterable of attribute names to skip. Defaults to skipping
            ``parameters`` keys (usually already in config), plus
            ``core``, ``parameters``, ``random_state``, ``schema``.
            Underscore-prefixed attributes are always skipped.
        debug: if True, prints captured and skipped attribute lists.

    Returns:
        dict of ``{attr_name: JSON-safe value}`` ready to be stored
        alongside the config for later restoration via
        :func:`restore_object_value`.
    """
    import numpy as _np
    snap = {}
    skipped = []
    # Default skip set: declared config + framework-attached bookkeeping.
    params = getattr(instance, 'parameters', None)
    params_keys = set(params.keys()) if isinstance(params, dict) else set()
    framework = {'core', 'parameters', 'random_state', 'schema'}
    extra_skip = set(skip or ())
    for attr, value in instance.__dict__.items():
        if attr.startswith('_'):
            skipped.append((attr, 'underscore')); continue
        if attr in params_keys:
            skipped.append((attr, 'in params')); continue
        if attr in framework or attr in extra_skip:
            skipped.append((attr, 'framework')); continue
        # ndarray first — numpy scalar ints subclass Python int.
        if isinstance(value, _np.ndarray):
            if value.ndim <= 2 and value.dtype.kind in ('i', 'u', 'f', 'b'):
                snap[attr] = {
                    '__ndarray__': True,
                    'data': value.tolist(),
                    'dtype': str(value.dtype),
                    'shape': list(value.shape),
                }
            else:
                skipped.append((attr, f'ndarray dtype={value.dtype} ndim={value.ndim}'))
        elif isinstance(value, (bool, int, float, str)) or value is None:
            snap[attr] = value
        elif isinstance(value, _np.integer):
            snap[attr] = {
                '__npint__': True,
                'value': int(value),
                'dtype': str(value.dtype),
            }
        elif isinstance(value, _np.floating):
            snap[attr] = {
                '__npfloat__': True,
                'value': float(value),
                'dtype': str(value.dtype),
            }
        elif isinstance(value, _np.bool_):
            snap[attr] = bool(value)
        else:
            skipped.append((attr, type(value).__name__))
    if debug:
        cls = type(instance).__name__
        print(f'[capture_object_state] {cls}: captured={list(snap.keys())} '
              f'skipped={skipped}', flush=True)
    return snap


def restore_object_value(value):
    """Inverse of the per-attribute encoding produced by
    :func:`capture_object_state` — turns the JSON-safe wrapper back
    into the original numpy object (scalar or ndarray). Plain values
    pass through unchanged.
    """
    import numpy as _np
    if isinstance(value, dict):
        if value.get('__ndarray__'):
            return _np.asarray(value['data'], dtype=value['dtype']).reshape(value['shape'])
        if value.get('__npint__'):
            return _np.dtype(value['dtype']).type(value['value'])
        if value.get('__npfloat__'):
            return _np.dtype(value['dtype']).type(value['value'])
    return value
