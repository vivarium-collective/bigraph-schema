"""A memoized front end for the dispatch-hot generic functions.

``apply`` and ``reconcile`` are the two multiple-dispatch functions on the
per-tick hot loop, and together they account for **every** plum dispatch a
composite performs while running: ~300 per step, 100% of them these two.

plum already caches method resolution. What costs is the machinery around
that cache — ``Function.__call__`` calls ``_resolve_method_with_cache``,
which calls ``resolve_type_hint``/``_convert`` — so each dispatch pays three
Python frames to reach a dict lookup it was always going to hit.

:func:`fast_dispatch` keeps the dict lookup and drops the frames. It is
deliberately *not* a general-purpose dispatcher: it is correct only under the
conditions asserted below, and it checks them rather than assuming them.

**Why this stays correct**

- *Resolution is still plum's.* On a miss the wrapper asks plum, and caches
  what plum said. It never implements method resolution itself, so
  subclassing, unions and parametric types behave exactly as before.
- *Late registration is honoured.* A pending registration can change what an
  already-cached type tuple resolves to — and this codebase registers types
  lazily, so that is a live case, not a hypothetical. The wrapper checks
  ``_pending`` on **every** call, exactly as plum does, and drops its cache
  when plum has work to do. That check is one attribute lookup; it is not
  where the time was going.
- *Return conversion is not skipped, it is proven absent.* The wrapper caches
  a method only when plum reports its return type as ``Any``, for which
  ``_convert`` is a no-op. Anything else falls through to plum untouched, so
  annotating a return type later cannot silently lose the conversion.
- *Keyword calls fall through.* The cache keys on positional types only.

If plum's internals move, :func:`fast_dispatch` degrades to plain plum rather
than breaking: see the capability check at the bottom of this module.
"""

import os
from typing import Any

from plum import Function

DISABLE_ENV = 'BIGRAPH_SCHEMA_NO_FAST_DISPATCH'
"""Set to a truthy value to fall back to plain plum.

An escape hatch worth having on a change to the dispatch core: if a
dispatch-shaped bug ever shows up, this isolates the cache in one run
without a reinstall, and the two paths must agree.
"""


def _supported(function) -> bool:
    """Does this plum Function expose the internals the fast path needs?"""
    return (
        isinstance(function, Function)
        and hasattr(function, '_pending')
        and hasattr(function, '_resolve_pending_registrations')
        and hasattr(function, '_resolve_method_with_cache'))


class FastDispatch:
    """A memoizing stand-in for a plum ``Function``.

    Everything plum offers that this does not override — ``.dispatch`` (used
    downstream to register overloads from other packages), ``.invoke``,
    ``.methods``, ``.register`` — is forwarded to the wrapped Function, so
    this is a drop-in replacement for the name it rebinds rather than a
    narrower object that happens to be callable.
    """

    __slots__ = ('_function', '_resolve', '_cache', '__dict__')

    def __init__(self, function):
        self._function = function
        self._resolve = function._resolve_method_with_cache
        self._cache: dict = {}
        self.__doc__ = function.__doc__

    def __call__(self, *args, **kwargs):
        function = self._function
        cache = self._cache

        if function._pending:
            # plum has registrations it has not folded in yet. They may make
            # a *more specific* method available for a tuple already cached,
            # so the cache cannot be trusted across this boundary.
            function._resolve_pending_registrations()
            cache.clear()

        if kwargs:                       # not on the hot path; let plum have it
            return function(*args, **kwargs)

        types = tuple(map(type, args))
        method = cache.get(types)
        if method is None:
            method, return_type = self._resolve(args=args)
            if return_type is not Any:
                # A real return annotation means plum would convert. Don't
                # cache it and don't second-guess it.
                return function(*args)
            cache[types] = method
        return method(*args)

    # -- transparency ---------------------------------------------------
    def __getattr__(self, name):
        return getattr(self._function, name)

    @property
    def plum_function(self):
        """The wrapped plum Function — for tests and explicit escapes."""
        return self._function

    @property
    def dispatch_cache(self) -> dict:
        """The memo, so a test can assert it is actually being used."""
        return self._cache

    def __repr__(self):
        return f'<FastDispatch {getattr(self._function, "__name__", "?")}>'


def fast_dispatch(function):
    """Wrap a plum ``Function`` with an exact-type method cache.

    Returns the function unchanged when plum does not expose what the fast
    path needs — a version bump makes this slower, never wrong.
    """
    if os.environ.get(DISABLE_ENV):
        return function
    if not _supported(function):
        return function
    return FastDispatch(function)
