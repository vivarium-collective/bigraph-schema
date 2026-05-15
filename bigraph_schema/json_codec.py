"""JSON codec for bigraph-schema state documents.

The dispatch-based :func:`~bigraph_schema.methods.serialize.serialize` /
:func:`~bigraph_schema.methods.realize.realize` pair handles per-leaf
serialization when a paired schema is available. Many downstream consumers
(workspace dashboards, save-state caches, subprocess runners) need to JSON-
serialize a state document **without** carrying the matched schema, so a
schema-agnostic codec is required as well.

This module exposes that codec. The wire format uses tagged dicts so values
round-trip without schema info:

- ``pint.Quantity`` → ``{"__pint__": true, "magnitude": ..., "units": "..."}``
  (array-valued Quantities also carry ``"__magnitude_dtype__"``)
- ``numpy.ndarray`` → ``{"__numpy__": true, "dtype": "...", "shape": [...], "data": [...]}``
- ``numpy`` structured array →
  ``{"__numpy_structured__": true, "dtype": [...], "shape": [...], "data": [...]}``
- ``numpy`` scalars → native ``int``/``float``/``bool``
- ``set``/``frozenset`` → ``{"__set__": true, "data": [...]}``
- ``bytes`` → ``{"__bytes__": true, "data": "<hex>"}``

Tuples are NOT tagged — ``json.JSONEncoder`` bypasses ``default()`` for
tuples since they're natively encodable as JSON arrays. They round-trip
as lists.

This matches the format already in use in ``v2ecoli/cache.py:NumpyJSONEncoder``;
the canonical implementation now lives here and consumers should import from
this module rather than redefining their own. Saved state files using the
v2ecoli encoder remain readable.

The ``pint`` reconstruction uses
:func:`~bigraph_schema.units.get_quantity_registry`, so consumers that swap
the registry (e.g. v2ecoli setting pint's application registry to bigraph-
schema's shared one) get coherent ``Quantity`` instances.
"""
from __future__ import annotations

import json
from typing import Any


class BigraphJSONEncoder(json.JSONEncoder):
    """JSON encoder for bigraph-schema state trees.

    Use directly via ``json.dumps(obj, cls=BigraphJSONEncoder)`` or through
    the :func:`dumps` convenience wrapper. Pair with :func:`bigraph_json_hook`
    on the read side.
    """

    def default(self, obj: Any) -> Any:
        # pint.Quantity — emit tagged {magnitude, units} so we round-trip
        # without needing schema info at deserialize time. Array-valued
        # Quantities also carry their magnitude dtype so e.g. counts stay int.
        try:
            import pint
            if isinstance(obj, pint.Quantity):
                mag = obj.magnitude
                try:
                    import numpy as np
                    if isinstance(mag, np.ndarray):
                        return {
                            '__pint__': True,
                            'magnitude': mag.tolist(),
                            '__magnitude_dtype__': str(mag.dtype),
                            'units': str(obj.units),
                        }
                except ImportError:
                    pass
                if isinstance(mag, (int, float)):
                    return {'__pint__': True, 'magnitude': mag, 'units': str(obj.units)}
                return {'__pint__': True, 'magnitude': float(mag), 'units': str(obj.units)}
        except ImportError:
            pass

        # numpy
        try:
            import numpy as np
            if isinstance(obj, np.ndarray):
                if obj.dtype.names:
                    dtype_list = []
                    for name in obj.dtype.names:
                        field_dtype = obj.dtype[name]
                        if field_dtype.shape:
                            dtype_list.append((name, str(field_dtype.base), list(field_dtype.shape)))
                        else:
                            dtype_list.append((name, str(field_dtype)))
                    return {
                        '__numpy_structured__': True,
                        'dtype': dtype_list,
                        'shape': list(obj.shape),
                        'data': [list(row) for row in obj.tolist()],
                    }
                return {
                    '__numpy__': True,
                    'dtype': str(obj.dtype),
                    'shape': list(obj.shape),
                    'data': obj.tolist(),
                }
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
        except ImportError:
            pass

        # builtins json.dumps can't represent
        if isinstance(obj, (set, frozenset)):
            return {'__set__': True, 'data': sorted(obj, key=str)}
        if isinstance(obj, bytes):
            return {'__bytes__': True, 'data': obj.hex()}
        # Note: tuples are NOT tagged. json.JSONEncoder bypasses ``default()``
        # for tuples because they're natively encodable as JSON arrays, so a
        # tag here wouldn't fire. Tuples round-trip as lists. Callers that
        # need tuple-ness preserved should convert at the schema layer.

        return super().default(obj)


def bigraph_json_hook(obj: Any) -> Any:
    """JSON ``object_hook`` that reverses :class:`BigraphJSONEncoder`.

    Pass to ``json.load(f, object_hook=bigraph_json_hook)`` or through
    :func:`loads`. Recognises the tag keys (``__pint__`` etc.) and rebuilds
    the corresponding Python objects. Unrecognised dicts pass through
    unchanged, so plain JSON objects still load as ``dict``.
    """
    if not isinstance(obj, dict):
        return obj
    if obj.get('__pint__') or obj.get('__pint_array__'):
        # ``__pint_array__`` is v2ecoli's legacy variant tag for array-valued
        # Quantities; accept it as an alias of ``__pint__`` so saved state
        # files from before this module existed keep loading.
        from bigraph_schema.units import get_quantity_registry
        ureg = get_quantity_registry()
        mag = obj.get('magnitude', obj.get('value'))
        dtype = obj.get('__magnitude_dtype__')
        if obj.get('__pint_array__') or dtype is not None:
            import numpy as np
            mag = np.array(mag, dtype=dtype) if dtype else np.array(mag)
        return ureg.Quantity(mag, obj['units'])
    if obj.get('__numpy__'):
        import numpy as np
        return np.array(obj['data'], dtype=obj['dtype']).reshape(obj['shape'])
    if obj.get('__numpy_structured__'):
        import numpy as np
        dtype = np.dtype([tuple(field) for field in obj['dtype']])
        return np.array([tuple(row) for row in obj['data']], dtype=dtype).reshape(obj['shape'])
    if obj.get('__set__'):
        return set(obj['data'])
    if obj.get('__bytes__'):
        return bytes.fromhex(obj['data'])
    return obj


def dumps(obj: Any, **kwargs: Any) -> str:
    """Serialize ``obj`` to a JSON string via :class:`BigraphJSONEncoder`."""
    return json.dumps(obj, cls=BigraphJSONEncoder, **kwargs)


def loads(s: str, **kwargs: Any) -> Any:
    """Deserialize a JSON string via :func:`bigraph_json_hook`."""
    return json.loads(s, object_hook=bigraph_json_hook, **kwargs)


# ---------------------------------------------------------------------------
# Tests (inline, run via `python -m bigraph_schema.json_codec` or pytest).
# ---------------------------------------------------------------------------

def test_pint_scalar_round_trip():
    import pint
    from bigraph_schema.units import units as ureg
    q = ureg.Quantity(5.0, 'femtogram')
    s = dumps(q)
    q2 = loads(s)
    assert isinstance(q2, pint.Quantity)
    assert q2.magnitude == 5.0
    assert str(q2.units) == str(q.units)


def test_pint_array_round_trip():
    import numpy as np
    import pint
    from bigraph_schema.units import units as ureg
    q = ureg.Quantity(np.array([1, 2, 3], dtype=np.int64), 'count')
    s = dumps(q)
    q2 = loads(s)
    assert isinstance(q2, pint.Quantity)
    assert isinstance(q2.magnitude, np.ndarray)
    assert q2.magnitude.dtype == np.int64
    assert (q2.magnitude == np.array([1, 2, 3])).all()
    assert str(q2.units) == str(q.units)


def test_pint_int_magnitude_preserved():
    """Int magnitudes shouldn't silently widen to float on round-trip."""
    from bigraph_schema.units import units as ureg
    q = ureg.Quantity(42, 'count')
    s = dumps(q)
    q2 = loads(s)
    assert q2.magnitude == 42
    assert isinstance(q2.magnitude, int)


def test_numpy_ndarray_round_trip():
    import numpy as np
    a = np.array([[1.5, 2.5], [3.5, 4.5]])
    s = dumps(a)
    a2 = loads(s)
    assert isinstance(a2, np.ndarray)
    assert a2.dtype == a.dtype
    assert a2.shape == a.shape
    assert (a2 == a).all()


def test_numpy_scalar_round_trip():
    import numpy as np
    s = dumps({'i': np.int64(7), 'f': np.float32(1.5), 'b': np.bool_(True)})
    obj = loads(s)
    assert obj == {'i': 7, 'f': 1.5, 'b': True}
    assert isinstance(obj['i'], int)
    assert isinstance(obj['f'], float)
    assert isinstance(obj['b'], bool)


def test_set_round_trip():
    s = dumps({1, 2, 3})
    assert loads(s) == {1, 2, 3}


def test_bytes_round_trip():
    payload = b'\x00\x01\xff'
    assert loads(dumps(payload)) == payload


def test_tuple_becomes_list():
    """Tuples round-trip as lists — json.JSONEncoder bypasses ``default()``
    for tuples since they're natively JSON-encodable as arrays. Documenting
    this so callers know tuple-ness isn't preserved by this codec."""
    out = loads(dumps((1, 'two', 3.0)))
    assert out == [1, 'two', 3.0]
    assert isinstance(out, list)


def test_nested_state_tree():
    """The hook must fire at every level — top-level dict with nested Quantities."""
    import numpy as np
    from bigraph_schema.units import units as ureg
    tree = {
        'agent': {
            'mass': ureg.Quantity(123.4, 'fg'),
            'counts': ureg.Quantity(np.array([10, 20], dtype=np.int64), 'count'),
            'genome': b'ATCG',
            'tags': {'a', 'b'},
            'history': [
                {'t': 0, 'rate': ureg.Quantity(0.5, '1/second')},
                {'t': 1, 'rate': ureg.Quantity(0.6, '1/second')},
            ],
        },
    }
    out = loads(dumps(tree))
    assert out['agent']['mass'].magnitude == 123.4
    assert (out['agent']['counts'].magnitude == np.array([10, 20])).all()
    assert out['agent']['counts'].magnitude.dtype == np.int64
    assert out['agent']['genome'] == b'ATCG'
    assert out['agent']['tags'] == {'a', 'b'}
    assert out['agent']['history'][1]['rate'].magnitude == 0.6


def test_plain_dict_passes_through():
    """Dicts without tag keys must round-trip as plain dicts."""
    payload = {'just': 'a', 'plain': {'nested': [1, 2]}}
    assert loads(dumps(payload)) == payload


def test_v2ecoli_format_compat():
    """Saved files using v2ecoli's NumpyJSONEncoder must still load.

    v2ecoli emits ``__pint__`` for scalars and ``__pint_array__`` for array-
    valued Quantities. This codec accepts both so saved state files written
    before this module existed remain readable.
    """
    import numpy as np
    import pint
    legacy_blob = (
        '{"mass": {"__pint__": true, "magnitude": 1.5, "units": "femtogram"},'
        ' "rates": {"__pint_array__": true, "magnitude": [0.1, 0.2], "units": "1/second"}}'
    )
    out = loads(legacy_blob)
    assert isinstance(out['mass'], pint.Quantity)
    assert out['mass'].magnitude == 1.5
    assert isinstance(out['rates'], pint.Quantity)
    assert isinstance(out['rates'].magnitude, np.ndarray)
    assert (out['rates'].magnitude == np.array([0.1, 0.2])).all()


if __name__ == '__main__':
    test_pint_scalar_round_trip()
    test_pint_array_round_trip()
    test_pint_int_magnitude_preserved()
    test_numpy_ndarray_round_trip()
    test_numpy_scalar_round_trip()
    test_set_round_trip()
    test_bytes_round_trip()
    test_tuple_becomes_list()
    test_nested_state_tree()
    test_plain_dict_passes_through()
    test_v2ecoli_format_compat()
    print('json_codec: all tests passed')
