"""The memoized front end for `apply` / `reconcile`.

These two are the whole per-tick dispatch load, so the cache is on the hot
path of every simulation. What matters is not that it is fast — it is that it
cannot answer differently from plum.
"""

import pytest
from plum import Function, dispatch

from bigraph_schema.fastdispatch import FastDispatch, fast_dispatch
from bigraph_schema.methods.apply import apply
from bigraph_schema.methods.reconcile import reconcile


def test_the_hot_functions_are_actually_wrapped():
    assert isinstance(apply, FastDispatch)
    assert isinstance(reconcile, FastDispatch)


def test_the_cache_is_used():
    """A second call with the same argument types must hit the memo."""
    @dispatch
    def widget(x: int, y: str):
        return ('int-str', x, y)

    fast = fast_dispatch(widget)
    assert fast(1, 'a') == ('int-str', 1, 'a')
    assert len(fast.dispatch_cache) == 1
    fast(2, 'b')
    assert len(fast.dispatch_cache) == 1, 'same types must not re-resolve'


def test_it_agrees_with_plum_on_every_dispatch():
    """Parity is the contract. Resolution stays plum's; this only memoizes."""
    @dispatch
    def shape(x: int):
        return 'int'

    @dispatch
    def shape(x: bool):                 # bool is a subclass of int
        return 'bool'

    @dispatch
    def shape(x: object):
        return 'object'

    fast = fast_dispatch(shape)
    for value in (1, True, False, 'text', 3.5, None, [1], {'a': 1}):
        assert fast(value) == shape(value), f'diverged on {value!r}'


def test_a_late_registration_invalidates_the_cache():
    """The case lazy type discovery makes real.

    A method registered *after* a type tuple is cached may be more specific
    for it. plum re-resolves on `_pending`; so must this, or a lazily
    imported schema type would dispatch to the wrong handler forever.
    """
    @dispatch
    def widen(x: object):
        return 'object'

    fast = fast_dispatch(widen)
    assert fast(7) == 'object'
    assert fast.dispatch_cache, 'precondition: the answer was cached'

    @dispatch
    def widen(x: int):                  # strictly more specific than object
        return 'int'

    assert fast(7) == 'int', 'stale cache served a superseded method'


def test_keyword_calls_fall_through_to_plum():
    """plum dispatches on positional arguments only, so a keyword call is
    plum's business — the memo keys on positional types and must not try."""
    @dispatch
    def named(x: int, *, scale: int = 1):
        return x * scale

    fast = fast_dispatch(named)
    assert fast(3, scale=2) == 6
    assert not fast.dispatch_cache, 'kwargs must not populate the memo'
    # ...and the same call without kwargs does cache
    assert fast(3) == 3
    assert len(fast.dispatch_cache) == 1


def test_a_declared_return_type_is_not_cached_or_converted_away():
    """The wrapper only caches methods plum reports as returning `Any`,
    for which conversion is a no-op. Anything else stays plum's problem."""
    @dispatch
    def converted(x: int) -> str:
        return 'value'

    fast = fast_dispatch(converted)
    assert fast(1) == 'value'
    assert not fast.dispatch_cache, 'a converting method must not be cached'


def test_plum_api_is_forwarded():
    """`@apply.dispatch` is how other packages register overloads — the
    wrapper has to be a drop-in for the name it replaces, not merely
    callable."""
    assert hasattr(apply, 'dispatch')
    assert hasattr(apply, 'methods')
    assert isinstance(apply.plum_function, Function)
    assert apply.__name__ == 'apply'


def test_an_unsupported_function_is_returned_unchanged():
    """A plum version that moves its internals makes this slower, not wrong."""
    def plain(x):
        return x

    assert fast_dispatch(plain) is plain
