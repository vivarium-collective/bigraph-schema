"""Typed Translator kernel — the "refuse-to-skip" crossing primitive.

A ``Translator`` is a typed, *declared* crossing between two schema types.
Applying it (via ``Core.cross``) returns a typed value wrapped in
``Crossed`` — or a ``Refusal`` — **never** a silent ``None`` or a silently
substituted default. This is the whole point, and the contrast with
``bigraph_schema.methods.coerce.coerce`` (which silently returns a default
on mismatch, e.g. ``coerce(Integer, 'abc') -> 0``) is deliberate: a
Translator refuses exactly where ``coerce`` would quietly paper over the
mismatch.

This module is purely additive: it introduces new types and does not
modify any existing ``check``/``coerce``/``resolve``/``transform``
behavior.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class Refusal:
    """A value representing a translator's refusal to cross.

    Never ``None`` — a ``Refusal`` is a first-class outcome, not the
    absence of one, so callers cannot mistake "refused" for "nothing
    happened."
    """

    translator_id: str
    reason: str
    source: str
    target: str

    def __str__(self) -> str:
        return (
            f"Refusal[{self.translator_id}]: {self.reason} "
            f"({self.source} -> {self.target})"
        )


@dataclass(frozen=True)
class Crossed:
    """A value that successfully crossed from one type to another."""

    value: Any


@dataclass
class Translator:
    """A declared, typed crossing between ``source`` and ``target``.

    Attributes:
        id: Unique identifier used to register/look up this translator.
        source: The schema (or schema-encoding) the input must satisfy.
        target: The schema (or schema-encoding) the output must satisfy.
        mode: One of ``'total'``, ``'partial'``, ``'lossy'``, ``'widening'``.
            - ``total``: succeeds for every value of ``source``.
            - ``partial``: may refuse for some values of ``source``
              (``cross_fn`` may itself return a ``Refusal``).
            - ``lossy``: succeeds, but discards information (``loss`` is
              required and must describe what is lost).
            - ``widening``: succeeds, embedding into a strictly larger type.
        cross_fn: ``value -> value`` (or ``value -> Refusal`` for a
            ``partial`` crossing). Returning ``None`` is treated as an
            accidental omission, not a valid crossing, and is converted to
            a ``Refusal`` by ``Core.cross`` (the anti-silent-None law).
        loss: Required, non-empty description of what information is
            discarded when ``mode == 'lossy'``.
    """

    id: str
    source: Any
    target: Any
    mode: str
    cross_fn: Callable[[Any], Any]
    loss: Optional[str] = None


# Worst-to-best ordering used to compute the mode of a composed translator.
_MODE_SEVERITY = {
    'partial': 0,
    'lossy': 1,
    'widening': 2,
    'total': 3,
}


def worst_mode(mode_a: str, mode_b: str) -> str:
    """Return whichever of the two modes is 'worse' (less total)."""
    severity_a = _MODE_SEVERITY.get(mode_a, -1)
    severity_b = _MODE_SEVERITY.get(mode_b, -1)
    return mode_a if severity_a <= severity_b else mode_b
