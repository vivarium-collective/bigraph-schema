"""Structural events emitted by ``apply`` for composite-layer consumption.

Apply traverses an update tree to mutate state. Structural sentinels
(``_add``/``_remove``/``_divide``) reshape the tree topology — adding,
removing, or replacing keys. Downstream consumers (the composite engine
in particular) need to react: rebuild process indexes, invalidate
compiled caches, refresh wire projections.

Rather than re-walking the update tree after apply (which the composite
historically did via ``_walk_update`` + ``find_instance_paths``), apply
emits these events as it observes them. Callers that opt in by passing
an ``events`` list to ``core.apply`` receive a stream of structural
observations they can use to update their indexes incrementally.

Sink semantics: events are pushed via ``emit()`` to whichever sink list
is currently installed via ``install_sink``. ``core.apply`` manages
install/uninstall around the dispatched apply call when its caller
provides an events accumulator. Compiled apply emits no events (value
ticks never produce structural changes).

Design notes:
- Single sink at a time. Nested apply calls (composite-of-composites)
  share the outer sink — events from inner applies bubble up. To
  isolate a nested apply, callers can swap sinks via the install/
  uninstall pair.
- Path is the absolute path in the *outer* state tree, threaded through
  apply via the ``path`` argument. The first sentinel-emitting handler
  uses the path it received; the composite's update wrapping in
  ``apply_updates`` ensures paths are absolute root-relative.
- Events are intentionally minimal: the consumer (composite layer)
  needs enough info to update its own indexes, no more.
"""

from dataclasses import dataclass
from typing import Any, List, Optional
import threading


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

@dataclass
class NodeAdded:
    """A new key was added to a container via ``_add``.

    Composite consumes this to:
    - scan the new subtree for Process/Step instances and add to
      ``process_paths`` / ``step_paths`` (vs the previous full-state
      rescan via ``find_instance_paths``)
    - invalidate any precompiled link caches under ``path`` (the parent
      container's wires may now reference the new key)
    """
    path: tuple
    key: Any
    state: Any
    schema: Any


@dataclass
class NodeRemoved:
    """A key was removed from a container via ``_remove``.

    Composite removes the path from process/step indexes and drops
    compiled-link cache entries under it.
    """
    path: tuple
    key: Any


@dataclass
class Divided:
    """``_divide`` sentinel consumed a mother key and produced daughter
    keys at the same container level. Distinct from
    ``NodeRemoved + NodeAddeds`` because the consumer often wants the
    mother→daughter relationship preserved (e.g. for lineage tracking).
    """
    path: tuple
    mother_key: Any
    daughter_keys: List[Any]
    daughters_state: dict
    daughters_schema: Any


# ---------------------------------------------------------------------------
# Sink machinery
# ---------------------------------------------------------------------------

_local = threading.local()


def install_sink(sink: Optional[List[Any]]) -> Optional[List[Any]]:
    """Install ``sink`` as the active events list. Returns the previous
    sink so the caller can restore it. Pass ``None`` to disable
    collection."""
    prev = getattr(_local, 'sink', None)
    _local.sink = sink
    return prev


def uninstall_sink(prev: Optional[List[Any]]) -> None:
    """Restore a previously-active sink (the value returned by
    ``install_sink``)."""
    _local.sink = prev


def emit(event: Any) -> None:
    """Push an event to the active sink, if one is installed.

    No-op when no sink is active — keeps the apply hot path free of
    overhead when callers don't care about events.
    """
    sink = getattr(_local, 'sink', None)
    if sink is not None:
        sink.append(event)


def current_sink() -> Optional[List[Any]]:
    """Return the active sink list, or None if events are being
    discarded. Used by tests / debugging."""
    return getattr(_local, 'sink', None)


# ---------------------------------------------------------------------------
# Reconcile summary
# ---------------------------------------------------------------------------
#
# Reconcile already walks the update tree to merge per-key updates. Callers
# that previously did a separate ``_walk_update`` pass to extract leaf paths
# and a structural-sentinel flag can instead install a ``ReconcileSummary``
# sink: reconcile populates ``paths`` and ``has_structural`` as it descends,
# at no extra walk cost.
#
# ``path_stack`` is the current absolute path under reconcile's top-level
# entry; dict/Map dispatches push/pop on recursion. Stored on the summary
# rather than thread-local because reconcile recursion never crosses
# threads.

@dataclass
class ReconcileSummary:
    """Sink for path/structural info collected during a reconcile walk.

    Pass an instance via ``install_reconcile_sink`` before calling
    ``core.reconcile``. After reconcile returns, ``paths`` holds every
    leaf path visited and ``has_structural`` is True if any
    ``_add``/``_remove``/``_divide``/``_type`` sentinel appeared.
    """
    paths: List[tuple]
    has_structural: bool = False
    # Internal: current absolute path during reconcile recursion.
    # dict/Map dispatches push/pop on entry/exit.
    path_stack: tuple = ()


def install_reconcile_sink(sink: Optional[ReconcileSummary]) -> Optional[ReconcileSummary]:
    """Install ``sink`` as the active reconcile summary. Returns previous."""
    prev = getattr(_local, 'reconcile_sink', None)
    _local.reconcile_sink = sink
    return prev


def uninstall_reconcile_sink(prev: Optional[ReconcileSummary]) -> None:
    """Restore a previously-active reconcile sink."""
    _local.reconcile_sink = prev


def get_reconcile_sink() -> Optional[ReconcileSummary]:
    """Return the active reconcile sink, or None."""
    return getattr(_local, 'reconcile_sink', None)


_STRUCTURAL_SENTINELS = frozenset(('_add', '_remove', '_type', '_divide'))


def emit_to_reconcile_sink(state: Any, sink: ReconcileSummary) -> None:
    """Walk ``state`` and populate the sink's paths + has_structural.

    Used by ``core.reconcile`` when a single update bypasses dispatch
    (the multi-update path's structural detection happens naturally in
    the dispatched reconcile). Cheap iterative walk — same cost as the
    old ``Composite._walk_update`` but emits to the sink directly.
    """
    if not isinstance(state, dict):
        sink.paths.append(sink.path_stack)
        return

    parent = sink.path_stack
    for key, value in state.items():
        if isinstance(key, str) and key.startswith('_'):
            if key in _STRUCTURAL_SENTINELS:
                sink.has_structural = True
            sink.paths.append(parent)
            continue
        new_path = parent + (key,)
        if isinstance(value, dict):
            sink.path_stack = new_path
            try:
                emit_to_reconcile_sink(value, sink)
            finally:
                sink.path_stack = parent
        else:
            sink.paths.append(new_path)
