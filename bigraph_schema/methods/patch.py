"""
JSON-patch-style operations on schema+state trees.

Supports add, remove, replace, and move operations.
Each operation is a dict with 'op', 'path', and optionally 'value'/'from'.
"""

from bigraph_schema.methods.default import default


def _get_at_path(state, path):
    """Get value at a nested path."""
    current = state
    for key in path:
        if isinstance(current, dict):
            current = current[key]
        elif isinstance(current, (list, tuple)):
            current = current[int(key)]
        else:
            raise Exception(f'cannot traverse {type(current)} with key {key}')
    return current


def _set_at_path(state, path, value):
    """Set value at a nested path, returning modified state."""
    if not path:
        return value

    key = path[0]
    rest = path[1:]

    if isinstance(state, dict):
        result = dict(state)
        if rest:
            result[key] = _set_at_path(result.get(key, {}), rest, value)
        else:
            result[key] = value
        return result

    elif isinstance(state, (list, tuple)):
        result = list(state)
        idx = int(key)
        if rest:
            result[idx] = _set_at_path(result[idx], rest, value)
        else:
            if idx == len(result):
                result.append(value)
            else:
                result[idx] = value
        return type(state)(result) if isinstance(state, tuple) else result

    raise Exception(f'cannot set at path {path} in {type(state)}')


def _delete_at_path(state, path):
    """Delete value at a nested path, returning modified state."""
    if not path:
        raise Exception('cannot delete at empty path')

    key = path[0]
    rest = path[1:]

    if isinstance(state, dict):
        result = dict(state)
        if rest:
            result[key] = _delete_at_path(result[key], rest)
        else:
            del result[key]
        return result

    elif isinstance(state, (list, tuple)):
        result = list(state)
        idx = int(key)
        if rest:
            result[idx] = _delete_at_path(result[idx], rest)
        else:
            del result[idx]
        return type(state)(result) if isinstance(state, tuple) else result

    raise Exception(f'cannot delete at path {path} in {type(state)}')


def patch(core, schema, state, operations):
    """
    Apply JSON-patch-style operations to state.

    Args:
        core: Core instance for schema access.
        schema: The schema of the state.
        state: The state to patch.
        operations: List of operation dicts:
            {'op': 'add', 'path': ['a', 'b'], 'value': 42}
            {'op': 'remove', 'path': ['a', 'b']}
            {'op': 'replace', 'path': ['a'], 'value': 'new'}
            {'op': 'move', 'from': ['a'], 'path': ['b']}

    Returns:
        The patched state.
    """
    result = state

    for op_spec in operations:
        op = op_spec['op']
        path = op_spec.get('path', [])

        if op == 'add':
            result = _set_at_path(result, path, op_spec['value'])

        elif op == 'remove':
            result = _delete_at_path(result, path)

        elif op == 'replace':
            result = _set_at_path(result, path, op_spec['value'])

        elif op == 'move':
            from_path = op_spec['from']
            value = _get_at_path(result, from_path)
            result = _delete_at_path(result, from_path)
            result = _set_at_path(result, path, value)

        else:
            raise Exception(f'unknown patch operation: {op}')

    return result
