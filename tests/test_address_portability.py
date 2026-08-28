"""Composite-address portability validation (Level-1 lint).

A bare ``local:<name>`` edge address resolves via the process registry, so it
depends on the *resolving* core having ``<name>`` registered. A composite that
realizes in a full/discovering core can then fail in a thin/plain core (e.g. a
detached run subprocess's ``build_core``) with a deep, unlocated
``no link found at address: {'protocol': 'local', 'data': '<name>'}``.

The self-describing ``local:!module.path.Class`` form resolves by import and is
portable across cores. These helpers flag the non-portable form and check a
document against the core that will realize it. See
``bigraph_schema/protocols.py``.
"""

from bigraph_schema import (
    Core,
    registry_dependent_addresses,
    unresolvable_addresses,
    assert_portable_addresses,
    iter_link_addresses,
)
from bigraph_schema.edge import Edge
import pytest


class _Widget(Edge):
    """A trivial local edge, importable by module path for the `!` form."""

    def inputs(self):
        return {}

    def outputs(self):
        return {}

    def update(self, state, interval=None):
        return {}


# A composite document: one edge wired by a bare (registry) address, one by the
# self-describing module-path (`!`) address, and a non-local address.
_THIS = f"{__name__}._Widget"

DOCUMENT = {
    "bare_edge": {"_type": "step", "address": "local:_Widget",
                  "inputs": {}, "outputs": {}},
    "portable_edge": {"_type": "step", "address": f"local:!{_THIS}",
                      "inputs": {}, "outputs": {}},
    "nested": {
        "inner_bare": {"_type": "step", "address": "local:AnotherOne",
                       "inputs": {}, "outputs": {}},
    },
    "remote_edge": {"_type": "step", "address": "http://example.invalid/x"},
    "not_an_edge": {"some": "data", "value": 3},
}


def test_iter_link_addresses_finds_every_edge_including_nested():
    found = {"/".join(map(str, p)): a for p, a in iter_link_addresses(DOCUMENT)}
    assert found["bare_edge"] == "local:_Widget"
    assert found["portable_edge"] == f"local:!{_THIS}"
    assert found["nested/inner_bare"] == "local:AnotherOne"
    assert found["remote_edge"] == "http://example.invalid/x"
    # a plain data dict is not reported as an edge
    assert "not_an_edge" not in found


def test_registry_dependent_addresses_flags_only_bare_local():
    reported = {p: name for p, name in registry_dependent_addresses(DOCUMENT)}
    # Only the two bare local:<name> addresses — NOT the `!` form or the remote one.
    assert reported == {
        ("bare_edge",): "_Widget",
        ("nested", "inner_bare"): "AnotherOne",
    }


def test_assert_portable_addresses_raises_and_names_paths():
    with pytest.raises(ValueError) as excinfo:
        assert_portable_addresses(DOCUMENT)
    msg = str(excinfo.value)
    assert "bare_edge" in msg and "AnotherOne" in msg
    # suggests the portable form
    assert "local:!" in msg


def test_assert_portable_passes_when_all_self_resolving():
    ok_doc = {
        "e": {"_type": "step", "address": f"local:!{_THIS}"},
        "http": {"_type": "step", "address": "sozzle://host/thing"},
    }
    assert registry_dependent_addresses(ok_doc) == []
    assert_portable_addresses(ok_doc)  # does not raise


def test_unresolvable_addresses_checks_against_a_specific_core():
    core = Core({})  # a plain core: no _Widget / AnotherOne registered
    unresolved = {"/".join(map(str, p)): a
                  for p, a in unresolvable_addresses(DOCUMENT, core)}
    # The bare names don't resolve in this core...
    assert "bare_edge" in unresolved
    assert "nested/inner_bare" in unresolved
    # ...but the self-describing `!` form does (resolves by import).
    assert "portable_edge" not in unresolved

    # Registering the bare name makes it resolve — no longer reported.
    core.register_link("_Widget", _Widget)
    after = {"/".join(map(str, p)): a
             for p, a in unresolvable_addresses(DOCUMENT, core)}
    assert "bare_edge" not in after
    assert "nested/inner_bare" in after  # AnotherOne still missing
