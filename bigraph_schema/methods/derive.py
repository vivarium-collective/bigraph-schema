"""Realize-time derivation context: contextual values resolved at
process construction.

Some schema fields can't be stored as static literals because their
final value depends on context that isn't known until realize() runs —
notably per-generation seeds derived from the current lineage seed
(``v1: seed = (default + cli_seed) % RAND_MAX``).

Rather than threading context through every realize call signature, the
context is held in a thread-local stack. Callers install a
``DerivationContext`` around composite construction / load / divide;
schema-level realize dispatchers (currently ``LineageSeed``) read it
to compute their final value.

Sink semantics mirror ``events.py``:
- single context active at a time per thread
- nested install/uninstall lets callers temporarily override
- ``get_derivation_context`` returns ``None`` when nothing's installed,
  in which case derivers fall back to their stored base value (no-op
  derivation — preserves current behavior for callers that haven't
  opted in)
"""

from dataclasses import dataclass
import threading
from typing import Optional


@dataclass
class DerivationContext:
    """Context consulted by realize-time derivers.

    Fields:
      lineage_seed: Per-generation seed (CLI ``--seed`` in vEcoli).
        Combined into ``LineageSeed`` field values as
        ``(base + lineage_seed) % RAND_MAX``.
    """
    lineage_seed: int = 0


_local = threading.local()


def install_derivation_context(
    context: Optional[DerivationContext],
) -> Optional[DerivationContext]:
    """Install ``context`` as the active derivation context. Returns the
    previous context so the caller can restore it. Pass ``None`` to
    disable derivation (derivers fall back to stored base values)."""
    prev = getattr(_local, 'context', None)
    _local.context = context
    return prev


def uninstall_derivation_context(
    prev: Optional[DerivationContext],
) -> None:
    """Restore a previously-active context (the value returned by
    ``install_derivation_context``)."""
    _local.context = prev


def get_derivation_context() -> Optional[DerivationContext]:
    """Return the active derivation context, or ``None`` if none is
    installed. Schema dispatchers consult this to read context-derived
    values (e.g. lineage seeds)."""
    return getattr(_local, 'context', None)
