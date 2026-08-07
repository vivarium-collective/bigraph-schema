"""Tests for the streamline-dispatch fixes.

Each test drives a code path that previously raised on execution
(NameError / TypeError / bare-string raise) or silently discarded a
computed result. Every test here executes the exact branch that was
broken-on-fire before this change.
"""

import pytest

from bigraph_schema import allocate_core
from bigraph_schema.schema import Node, Map, Tree


@pytest.fixture
def core():
    return allocate_core()


# --- CACHE BUG: register_type must invalidate the string parse cache ---

def test_register_type_invalidates_stale_string_cache(core):
    """access() must reflect a freshly registered type, not a stale parse.

    Before the fix, register_type popped the string key from the
    id-keyed ``_access_cache`` (a guaranteed no-op), leaving the stale
    parse in ``_access_string_cache`` to serve the old definition
    forever.
    """
    name = 'streamline_cache_probe'

    # Prime the string cache: an unregistered name parses to a bare
    # type-reference string.
    before = core.access(name)
    assert isinstance(before, str)

    # Register a real type under that name.
    core.register_type(name, {'_type': 'map', '_value': 'float'})

    # access() must now return the registered type, not the stale parse.
    after = core.access(name)
    assert isinstance(after, Map)


# --- BROKEN DISPATCH: check() on a Map with non-string keys ---

def test_check_map_non_string_keys(core):
    """The Map non-string-key branch of check() used an unimported
    ``realize``; it now checks each key against the key schema directly."""
    assert core.check('map[integer,float]', {1: 1.0, 2: 2.0}) is True
    # A key that does not match the key schema fails the check.
    assert core.check('map[integer,float]', {'a': 1.0}) is False


# --- BROKEN DISPATCH: validate() on a Map with non-string keys ---

def test_validate_map_non_string_keys(core):
    """validate()'s Map non-string-key branch used an unimported
    ``realize`` and misused its 3-tuple return; it now routes through
    ``_realize_map_key`` correctly."""
    # Valid integer keys: no exception raised.
    core.validate('map[integer,float]', {1: 1.0, 2: 2.0})


# --- BROKEN DISPATCH: jump() key/index coercion (kw_only dataclasses) ---

def test_jump_tuple_by_string_key(core):
    """Tuple jump by a string Key coerces to an Index — previously a
    ``TypeError`` from positional kw_only ``Index(...)`` construction."""
    schema, value = core.jump('tuple[float,float]', (1.5, 2.5), '0')
    assert value == 1.5


def test_jump_list_by_string_key(core):
    schema, value = core.jump('list[float]', [1.5, 2.5], '1')
    assert value == 2.5


def test_jump_map_by_integer_index(core):
    """Map jump by an Index coerces to a string Key — previously a
    ``TypeError`` from positional kw_only ``Key(...)`` construction."""
    schema, value = core.jump('map[float]', {'0': 1.5}, 0)
    assert value == 1.5


def test_jump_list_star_enumerate(core):
    """List Star jump iterated ``for index, value in state`` (unpacking a
    scalar element) — now ``enumerate(state)``."""
    schema, values = core.jump('list[float]', [1.5, 2.5], '*')
    assert values == [1.5, 2.5]


def test_jump_node_star_uses_field_keys(core):
    """Node Star jump iterated ``for key, value in schema.__dataclass_fields__``
    (unpacking dict keys) — a ``ValueError``. It now iterates keys and
    raises the intended, descriptive traverse error instead."""
    with pytest.raises(Exception) as excinfo:
        core.jump(Node(), {}, '*')
    assert not isinstance(excinfo.value, ValueError)
    assert 'no key' in str(excinfo.value)


# --- BROKEN DISPATCH: resolve() undefined ``subupdate`` in tuple loop ---

def test_resolve_tuple_pair(core):
    """resolve(tuple, Tuple) referenced an undefined ``subupdate`` in its
    loop — now uses the loop variable ``subelement``. Called via the free
    function so ``current`` stays a raw tuple (Core.access would coerce a
    tuple to a list and dispatch elsewhere)."""
    from bigraph_schema.methods.resolve import resolve as _resolve

    current = (core.access('float'), core.access('float'))
    update = core.access('tuple[float,float]')
    resolved = _resolve(current, update)  # previously raised NameError
    assert len(resolved._values) == 2


# --- DISCARDED RESULT: resolve(Tree, Node) dropped the resolved leaf ---

def test_resolve_tree_node_wires_resolved_leaf(core):
    """resolve(Tree, Node) computed ``replace(current, _leaf=resolved)``
    then returned the unmodified ``current``; the resolved leaf is now
    returned."""
    resolved = core.resolve('tree[float]', {'_type': 'float', '_default': 3.3})
    assert isinstance(resolved, Tree)
    assert resolved._leaf._default == 3.3


# --- BARE-STRING RAISE + DISCARDED RESULT: generalize(Tree, Node) ---

def test_generalize_tree_node_wires_resolved_leaf(core):
    """generalize(Tree, Node) dropped its ``replace(...)`` result (and its
    error path raised a bare string). The generalized leaf is now
    returned."""
    generalized = core.generalize('tree[float]', {'_type': 'float', '_default': 7.7})
    assert isinstance(generalized, Tree)
    assert generalized._leaf._default == 7.7
