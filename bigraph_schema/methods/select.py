"""
Extract a subset of fields from state with proper schema tracking.

select(core, schema, state, ['a', 'b']) → ({'a': Float, 'b': String}, {'a': 1.0, 'b': 'hi'})
select(core, schema, state, [['x', 'y'], ['z']]) → nested extraction
"""


def select(core, schema, state, paths):
    """
    Extract a subset of fields at given paths.

    Args:
        core: Core instance for traversal.
        schema: The full schema.
        state: The full state.
        paths: List of paths. Each path is either:
            - A string key (single level)
            - A list of keys (nested path)

    Returns:
        (result_schema, result_state) dict containing only the selected fields.
    """
    result_schema = {}
    result_state = {}

    for path in paths:
        if isinstance(path, str):
            path = [path]

        sub_schema, sub_state = core.traverse(schema, state, path)

        # Set at the leaf key
        key = path[-1]
        result_schema[key] = sub_schema
        if sub_state is not None:
            result_state[key] = sub_state

    return result_schema, result_state
