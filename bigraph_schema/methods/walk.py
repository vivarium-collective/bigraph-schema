"""
Generic fold/catamorphism over schema+state trees.

walk handles the structural recursion for all container types,
letting callers focus on leaf behavior and result assembly.
"""

import numpy as np

from bigraph_schema.schema import (
    Node,
    Atom,
    Empty,
    Union,
    Tuple,
    Boolean,
    Number,
    Integer,
    Float,
    Complex,
    Delta,
    Nonnegative,
    Range,
    String,
    Enum,
    Wrap,
    Maybe,
    Overwrite,
    Const,
    List,
    Set,
    Map,
    Tree,
    Array,
    Frame,
    Key,
    Path,
    Wires,
    Schema,
    Link,
)

from bigraph_schema.methods.check import check


def walk(schema, state, leaf_fn, combine_fn=None, path=()):
    """
    Generic fold over schema+state trees.

    Args:
        schema: A compiled schema node or dict.
        state: The state value to walk alongside the schema.
        leaf_fn(schema, state, path) -> result:
            Called at leaf types (atoms, arrays, frames, or when no
            further recursion is possible).
        combine_fn(schema, children, path) -> result:
            Called to assemble child results into a parent result.
            children is a list (for Tuple/List/Set) or dict (for
            Map/Tree/dict/Node).  If None, the raw children structure
            is returned.
        path: Current path tuple, extended as the walk descends.

    Returns:
        The result of folding leaf_fn/combine_fn over the tree.
    """

    def _combine(schema, children, path):
        if combine_fn is not None:
            return combine_fn(schema, children, path)
        return children

    # ── Wrapper types: unwrap and recurse ──

    if isinstance(schema, Maybe):
        if state is None:
            return leaf_fn(schema, state, path)
        return walk(schema._value, state, leaf_fn, combine_fn, path)

    if isinstance(schema, (Wrap, Overwrite, Const)):
        return walk(schema._value, state, leaf_fn, combine_fn, path)

    if isinstance(schema, Union):
        for option in schema._options:
            if check(option, state):
                return walk(option, state, leaf_fn, combine_fn, path)
        return leaf_fn(schema, state, path)

    # ── Container types: recurse into children ──

    if isinstance(schema, Tuple) and isinstance(state, (list, tuple)):
        children = [
            walk(v, state[i] if i < len(state) else None,
                 leaf_fn, combine_fn, path + (i,))
            for i, v in enumerate(schema._values)]
        return _combine(schema, children, path)

    if isinstance(schema, Set) and isinstance(state, (set, list, tuple)):
        children = [
            walk(schema._element, e, leaf_fn, combine_fn, path + (i,))
            for i, e in enumerate(state)]
        return _combine(schema, children, path)

    if isinstance(schema, List) and isinstance(state, (list, tuple)):
        children = [
            walk(schema._element, e, leaf_fn, combine_fn, path + (i,))
            for i, e in enumerate(state)]
        return _combine(schema, children, path)

    if isinstance(schema, Map) and isinstance(state, dict):
        children = {
            k: walk(schema._value, v, leaf_fn, combine_fn, path + (k,))
            for k, v in state.items()
            if not (isinstance(k, str) and k.startswith('_'))}
        return _combine(schema, children, path)

    if isinstance(schema, Tree):
        if isinstance(state, dict):
            if check(schema._leaf, state):
                return walk(schema._leaf, state, leaf_fn, combine_fn, path)
            children = {
                k: walk(schema, v, leaf_fn, combine_fn, path + (k,))
                for k, v in state.items()}
            return _combine(schema, children, path)
        else:
            return walk(schema._leaf, state, leaf_fn, combine_fn, path)

    if isinstance(schema, dict) and isinstance(state, dict):
        children = {}
        for k, v in schema.items():
            if isinstance(k, str) and k.startswith('_'):
                continue
            children[k] = walk(v, state.get(k), leaf_fn, combine_fn, path + (k,))
        return _combine(schema, children, path)

    # Links are opaque — serialize/realize handle them as a unit
    # (config serialization needs access to the instance's config_schema)
    if isinstance(schema, Link):
        return leaf_fn(schema, state, path)

    # Structured Node with named fields
    if isinstance(schema, Node) and not isinstance(schema, Atom):
        fields = {
            k: getattr(schema, k)
            for k in schema.__dataclass_fields__
            if not k.startswith('_')}
        if fields and isinstance(state, dict):
            children = {
                k: walk(v, state.get(k), leaf_fn, combine_fn, path + (k,))
                for k, v in fields.items()}
            return _combine(schema, children, path)

    # ── Leaf types ──

    return leaf_fn(schema, state, path)
