"""
Schema migration — convert state from one schema to another.

Handles field matching by name, type coercion for mismatches,
and defaults for missing fields.
"""

from bigraph_schema.schema import (
    Node,
    Atom,
)

from bigraph_schema.methods.default import default
from bigraph_schema.methods.coerce import coerce
from bigraph_schema.methods.check import check


def transform(core, source_schema, target_schema, state):
    """
    Convert state from source_schema to target_schema.

    Args:
        core: Core instance for schema access.
        source_schema: The schema the state currently conforms to.
        target_schema: The desired schema.
        state: The state to transform.

    Returns:
        (target_schema_compiled, transformed_state)
    """
    source = core.access(source_schema)
    target = core.access(target_schema)

    result = _transform(core, source, target, state)
    return target, result


def _transform(core, source, target, state):
    # Both are dicts/structs: match fields by name (always recurse to fill defaults)
    if isinstance(target, dict) and isinstance(state, dict):
        result = {}
        for key, target_sub in target.items():
            if isinstance(key, str) and key.startswith('_'):
                continue
            if key in state:
                source_sub = source.get(key) if isinstance(source, dict) else None
                result[key] = _transform(core, source_sub, target_sub, state[key])
            else:
                result[key] = default(target_sub)
        return result

    # Both are Nodes with fields
    if isinstance(target, Node) and not isinstance(target, Atom):
        target_fields = {
            k: getattr(target, k)
            for k in target.__dataclass_fields__
            if not k.startswith('_')}

        if target_fields and isinstance(state, dict):
            result = {}
            for key, target_sub in target_fields.items():
                source_sub = None
                if isinstance(source, Node) and hasattr(source, key):
                    source_sub = getattr(source, key)
                elif isinstance(source, dict):
                    source_sub = source.get(key)

                if key in state:
                    result[key] = _transform(core, source_sub, target_sub, state[key])
                else:
                    result[key] = default(target_sub)
            return result

    # If state already fits target, return as-is
    if check(target, state):
        return state

    # Leaf types: try coercion
    return coerce(target, state)
