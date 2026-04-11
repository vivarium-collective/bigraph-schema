"""
bundle.py — Dispatched method for serializing schema+state to a bundle.

``bundle(schema, state, context)`` works like ``serialize`` but produces
a mix of JSON-inline values and Parquet file references. Large arrays
are written directly to Parquet (skipping the intermediate Python-list
representation that ``serialize`` creates), while scalars and small
values stay inline.

The BundleContext object carries the output directory and a dedup map.
Types that don't override ``bundle`` fall through to ``serialize``.

Usage::

    from bigraph_schema.methods.bundle import bundle, BundleContext
    ctx = BundleContext(arrays_dir='out/arrays')
    document = bundle(schema, state, context=ctx)
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
from plum import dispatch

from bigraph_schema.schema import (
    Node,
    Atom,
    Empty,
    NONE_SYMBOL,
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
    NPRandom,
    String,
    Enum,
    Wrap,
    Quote,
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
    Object,
    is_schema_field,
)
from bigraph_schema.methods.serialize import serialize, render


# ---------------------------------------------------------------------------
# Bundle context
# ---------------------------------------------------------------------------

REF_KEY = '$bundle_ref'
MIN_ARRAY_BYTES = 10_000  # ~10 KB threshold for externalizing


@dataclass
class BundleContext:
    """Accumulates array files during a bundle operation.

    Attributes:
        arrays_dir: Filesystem directory where Parquet files are written.
        min_bytes: Minimum raw byte size before an array is externalized.
        refs: Content-hash → filename map for deduplication.
    """
    arrays_dir: str = 'arrays'
    min_bytes: int = MIN_ARRAY_BYTES
    refs: Dict[str, str] = field(default_factory=dict)

    def save_array(self, arr: np.ndarray, path_hint: str) -> Dict[str, Any]:
        """Save *arr* to a Parquet file and return a ``$bundle_ref`` marker.

        If an array with identical content has already been saved, the
        existing file is reused (content-addressed deduplication).
        """
        from process_bigraph.bundle import _save_array_parquet

        os.makedirs(self.arrays_dir, exist_ok=True)

        content_id = hashlib.sha256(
            arr.tobytes() + str(arr.dtype).encode()
        ).hexdigest()[:16]

        if content_id in self.refs:
            filename = self.refs[content_id]
        else:
            safe_hint = path_hint.replace('.', '_').replace('/', '_')
            if len(safe_hint) > 60:
                safe_hint = safe_hint[:60]
            filename = f'{safe_hint}_{content_id}.parquet'
            filepath = os.path.join(self.arrays_dir, filename)
            _save_array_parquet(arr, filepath)
            self.refs[content_id] = filename

        marker = {
            REF_KEY: f'arrays/{filename}',
            'shape': list(arr.shape),
            'dtype': str(arr.dtype),
        }
        if arr.dtype.names:
            marker['structured'] = True
        return marker


# ---------------------------------------------------------------------------
# Default: fall through to serialize
# ---------------------------------------------------------------------------

@dispatch
def bundle(schema, state, context: Optional[BundleContext] = None):
    """Fallback: delegate to serialize, but catch large arrays."""
    if isinstance(state, np.ndarray) and context is not None:
        if state.nbytes >= context.min_bytes:
            return context.save_array(state, 'array')
        return state.tolist()
    return serialize(schema, state)


@dispatch
def bundle(schema: Empty, state, context: Optional[BundleContext] = None):
    return NONE_SYMBOL


# ---------------------------------------------------------------------------
# Scalars — inline (same as serialize)
# ---------------------------------------------------------------------------

@dispatch
def bundle(schema: Boolean, state, context: Optional[BundleContext] = None):
    return serialize(schema, state)

@dispatch
def bundle(schema: Number, state, context: Optional[BundleContext] = None):
    return serialize(schema, state)

@dispatch
def bundle(schema: Integer, state: np.integer, context: Optional[BundleContext] = None):
    return int(state)

@dispatch
def bundle(schema: Float, state: np.floating, context: Optional[BundleContext] = None):
    return float(state)

@dispatch
def bundle(schema: Number, state: np.integer, context: Optional[BundleContext] = None):
    return int(state)

@dispatch
def bundle(schema: Number, state: np.floating, context: Optional[BundleContext] = None):
    return float(state)

@dispatch
def bundle(schema: String, state, context: Optional[BundleContext] = None):
    return serialize(schema, state)

@dispatch
def bundle(schema: Atom, state, context: Optional[BundleContext] = None):
    return serialize(schema, state)

@dispatch
def bundle(schema: Complex, state, context: Optional[BundleContext] = None):
    return serialize(schema, state)

@dispatch
def bundle(schema: NPRandom, state, context: Optional[BundleContext] = None):
    return serialize(schema, state)

@dispatch
def bundle(schema: Schema, state, context: Optional[BundleContext] = None):
    return serialize(schema, state)


# ---------------------------------------------------------------------------
# Array — the key override: go directly to Parquet
# ---------------------------------------------------------------------------

@dispatch
def bundle(schema: Array, state, context: Optional[BundleContext] = None):
    """Bundle an Array: write large arrays directly to Parquet.

    This skips the ``state.tolist()`` step that ``serialize`` does,
    avoiding creation of millions of Python objects for large arrays.
    """
    if state is None:
        return None
    if isinstance(state, np.ndarray):
        if context is not None and state.nbytes >= context.min_bytes:
            return context.save_array(state, 'array')
        # Small array — inline as list
        return state.tolist()
    if isinstance(state, list):
        return state
    if isinstance(state, dict):
        return state.get('data', state)
    return serialize(schema, state)


@dispatch
def bundle(schema: Frame, state, context: Optional[BundleContext] = None):
    return serialize(schema, state)


# ---------------------------------------------------------------------------
# Wrapper types — unwrap and recurse
# ---------------------------------------------------------------------------

@dispatch
def bundle(schema: Maybe, state, context: Optional[BundleContext] = None):
    if state is None:
        return None
    return bundle(schema._value, state, context)

@dispatch
def bundle(schema: Overwrite, state, context: Optional[BundleContext] = None):
    return bundle(schema._value, state, context)

@dispatch
def bundle(schema: Wrap, state, context: Optional[BundleContext] = None):
    return bundle(schema._value, state, context)

@dispatch
def bundle(schema: Quote, state, context: Optional[BundleContext] = None):
    return bundle(schema._value, state, context)

@dispatch
def bundle(schema: Const, state, context: Optional[BundleContext] = None):
    return bundle(schema._value, state, context)


# ---------------------------------------------------------------------------
# Container types — recurse into children
# ---------------------------------------------------------------------------

@dispatch
def bundle(schema: Map, state, context: Optional[BundleContext] = None):
    if not isinstance(state, dict):
        return {}
    from bigraph_schema.methods.serialize import _serialize_map_key
    return {
        _serialize_map_key(schema._key, k): bundle(schema._value, v, context)
        for k, v in state.items()}


@dispatch
def bundle(schema: Tree, state, context: Optional[BundleContext] = None):
    if not isinstance(state, dict):
        return {}
    result = {}
    for k, v in state.items():
        if isinstance(v, dict):
            result[k] = bundle(schema, v, context)  # Tree recurses with self
        else:
            result[k] = bundle(schema._leaf, v, context)
    return result


@dispatch
def bundle(schema: Tuple, state, context: Optional[BundleContext] = None):
    if state is None:
        return None
    if isinstance(state, (list, tuple)):
        return [
            bundle(schema._values[i] if i < len(schema._values) else schema._values[-1],
                   v, context)
            for i, v in enumerate(state)]
    return serialize(schema, state)


@dispatch
def bundle(schema: List, state, context: Optional[BundleContext] = None):
    if state is None:
        return None
    if isinstance(state, (list, tuple)):
        return [bundle(schema._element, v, context) for v in state]
    return serialize(schema, state)


@dispatch
def bundle(schema: Set, state, context: Optional[BundleContext] = None):
    if state is None:
        return None
    if isinstance(state, (set, frozenset, list)):
        return [bundle(schema._element, v, context) for v in state]
    return serialize(schema, state)


@dispatch
def bundle(schema: Union, state, context: Optional[BundleContext] = None):
    for variant in schema._options:
        try:
            return bundle(variant, state, context)
        except Exception:
            continue
    return serialize(schema, state)


# ---------------------------------------------------------------------------
# Link (process/step declarations) — recurse into config
# ---------------------------------------------------------------------------

@dispatch
def bundle(schema: Link, state, context: Optional[BundleContext] = None):
    """Bundle a Link: same structure as serialize but recurses with bundle."""
    if not isinstance(state, dict):
        return state

    address = serialize(schema.address, state.get('address'))
    instance = state.get('instance')
    unconfig = state.get('config')

    if instance is None:
        config_schema = {}
    else:
        config_schema = instance.core.access(instance.config_schema)

    config = bundle(config_schema, unconfig, context)

    encode = {
        'address': address,
        'config': config,
        '_inputs': render(schema._inputs),
        '_outputs': render(schema._outputs)}

    if state.get('inputs'):
        encode['inputs'] = state.get('inputs')
    if state.get('outputs'):
        encode['outputs'] = state.get('outputs')
    if state.get('interval'):
        encode['interval'] = state.get('interval')
    if state.get('priority'):
        encode['priority'] = state.get('priority')
    if state.get('_triggers'):
        encode['_triggers'] = state.get('_triggers')

    return encode


# ---------------------------------------------------------------------------
# Object — serialize Python objects via __dict__ with inferred schemas
# ---------------------------------------------------------------------------

@dispatch
def bundle(schema: Object, state, context: Optional[BundleContext] = None):
    """Bundle a Python object by walking its ``__dict__``.

    Infers a schema for each field, bundles the value (arrays go to
    parquet), and produces the serialized form with embedded schemas.
    """
    if state is None:
        return None
    if isinstance(state, dict):
        return state

    from bigraph_schema.methods.infer import infer
    from bigraph_schema.methods.serialize import render

    cls = type(state)
    class_path = f'{cls.__module__}.{cls.__name__}'
    obj_dict = state.__dict__

    field_schemas = {}
    field_values = {}
    for key, value in obj_dict.items():
        inferred_schema, _ = infer(None, value)
        field_schemas[key] = render(inferred_schema)
        field_values[key] = bundle(inferred_schema, value, context)

    return {
        '_class': class_path,
        '_schema': field_schemas,
        'fields': field_values,
    }


# ---------------------------------------------------------------------------
# Node — walk typed fields (same pattern as serialize)
# ---------------------------------------------------------------------------

def _bundle_key(key):
    """Convert a dict key to a JSON-compatible string.

    JSON only supports string keys, so non-string keys are serialized:
    - numpy integers → str(int(k))
    - tuples → JSON string of the list form
    """
    if isinstance(key, str):
        return key
    if isinstance(key, np.integer):
        return str(int(key))
    if isinstance(key, tuple):
        import json as _json
        return _json.dumps([_bundle_key(k) for k in key])
    if isinstance(key, (int, float, bool)):
        return str(key)
    return str(key)


def _bundle_value(value, context):
    """Bundle a state value that has no schema — infer the best approach."""
    if isinstance(value, np.ndarray):
        if context is not None and value.nbytes >= context.min_bytes:
            return context.save_array(value, 'array')
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {_bundle_key(k): _bundle_value(v, context) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_bundle_value(v, context) for v in value]
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    # Fallback — try serialize
    return serialize(Node(), value)


@dispatch
def bundle(schema: Node, state, context: Optional[BundleContext] = None):
    """Bundle a Node by walking its typed dataclass fields."""
    if state is None:
        return None
    if isinstance(state, np.ndarray):
        if context is not None and state.nbytes >= context.min_bytes:
            return context.save_array(state, 'array')
        return state.tolist()
    if isinstance(state, dict):
        result = {}
        schema_keys = set()
        for key in schema.__dataclass_fields__:
            if is_schema_field(schema, key) and key in state:
                result[key] = bundle(
                    getattr(schema, key), state[key], context)
                schema_keys.add(key)
        # Handle extra state keys not in schema (common in configs)
        for key in state:
            if key not in schema_keys and not (isinstance(key, str) and key.startswith('_')):
                result[key] = _bundle_value(state[key], context)
        return result
    if isinstance(state, np.integer):
        return int(state)
    if isinstance(state, np.floating):
        return float(state)
    return str(state)


@dispatch
def bundle(schema: dict, state, context: Optional[BundleContext] = None):
    """Bundle a dict schema by walking matching keys."""
    if not isinstance(state, dict):
        return {}
    result = {}
    schema_keys = set()
    for k, v in schema.items():
        if isinstance(k, str) and k.startswith('_'):
            continue
        if k in state:
            result[k] = bundle(v, state[k], context)
            schema_keys.add(k)
    # Handle extra state keys not in schema
    for k in state:
        if k not in schema_keys and not (isinstance(k, str) and k.startswith('_')):
            result[k] = _bundle_value(state[k], context)
    return result
