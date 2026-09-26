"""Regression tests for the schema-agnostic JSON codec.

Covers the bug where :class:`bigraph_schema.json_codec.BigraphJSONEncoder`
could not serialize bigraph-schema's own ``Node`` dataclasses (String,
Integer, Map, …). These surface in schema-agnostic state trees — e.g. a
process's ``_inputs`` schema captured by ``gather_emitter_results`` — and
``json.dumps(..., cls=BigraphJSONEncoder)`` raised
``TypeError: Object of type String is not JSON serializable``.
"""

import json

import pytest

from bigraph_schema.json_codec import (
    BigraphJSONEncoder,
    bigraph_json_hook,
    dumps,
    loads,
)
from bigraph_schema.schema import String, Integer, Map


def test_encoder_serializes_bare_node():
    """A lone Node no longer raises; it emits its tagged dict."""
    s = json.dumps(String(_default="cell_0"), cls=BigraphJSONEncoder)
    assert json.loads(s) == {"__bigraph_node__": "String", "_default": "cell_0"}


def test_node_leaked_into_state_serializes():
    """The real failure: a Node nested in an otherwise-plain state dict.

    Mirrors process ``_inputs`` schema leaking into emitter results.
    """
    state = {
        "cell": {"grow_divide": {"_inputs": {"agent_id": String(_default="cell_0")}}},
        "count": 7,
    }
    s = json.dumps(state, cls=BigraphJSONEncoder)  # must not raise
    assert "__bigraph_node__" in s


def test_node_round_trips():
    """Encode → decode rebuilds the same Node subclasses, incl. nested ones."""
    original = {
        "s": String(_default="x"),
        "i": Integer(_default=3),
        "m": Map(_default={}, _key=String(_default=""), _value=String(_default="v")),
        "plain": {"a": 1, "b": [1, 2, 3]},
    }
    back = loads(dumps(original))

    assert isinstance(back["s"], String) and back["s"]._default == "x"
    assert isinstance(back["i"], Integer) and back["i"]._default == 3
    assert isinstance(back["m"], Map)
    assert isinstance(back["m"]._value, String) and back["m"]._value._default == "v"
    assert back["s"] == original["s"]
    assert back["m"] == original["m"]
    # plain JSON is untouched
    assert back["plain"] == {"a": 1, "b": [1, 2, 3]}


def test_unknown_node_class_falls_back_to_dict():
    """A tag naming a class this build doesn't have degrades to a plain dict."""
    payload = json.dumps({"__bigraph_node__": "NotARealNodeType", "_default": 1})
    out = json.loads(payload, object_hook=bigraph_json_hook)
    assert out == {"__bigraph_node__": "NotARealNodeType", "_default": 1}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
