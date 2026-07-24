"""
========
Contract
========

A ``ProcessContract`` is a structured, mathematically-legible *superset* of the
existing :attr:`bigraph_schema.edge.Edge.description` convention. ``description``
is a flat markdown/LaTeX string that already states a process's governing
equations; a ``ProcessContract`` adds explicit per-port / per-config / per-symbol
semantics around that same math.

The two are wired so the existing ``description`` is **reused, never rewritten**:
when a contract omits ``summary`` / ``math`` / ``description`` those fields are
derived by parsing the flat ``description`` string (see
:meth:`ProcessContract.from_description` and
:meth:`ProcessContract.merged_with_description`). Authored ``inputs`` /
``outputs`` / ``config`` / ``symbols`` are always left untouched.

Everything here is additive: a process that declares no contract resolves to a
description-derived (or docstring-derived) one, or to ``None`` — it never raises
and never changes runtime behavior.
"""

import inspect
import copy
import re
from dataclasses import dataclass, field, asdict


# Lines that read as governing equations rather than prose. Matched by any of
# the canonical relation / operator markers, or by a probability-distribution
# name (so a line like ``x ~ Multinomial(...)`` is captured as math).
_DIST_NAMES = (
    r"Multinomial|Binomial|Poisson|Normal|Gaussian|Bernoulli|Uniform|"
    r"Exponential|Gamma|Beta|Dirichlet|Categorical|Geometric|Lognormal|"
    r"LogNormal|Hypergeometric|NegativeBinomial|Weibull|Chi|StudentT"
)
_MATH_RE = re.compile(r"[=~≈←≥≤∑∏]|\b(?:" + _DIST_NAMES + r")\b")


@dataclass
class ProcessContract:
    """A structured superset of ``Edge.description``.

    All mutable fields use ``default_factory`` so instances never share state.
    """

    summary: str = ""
    description: str = ""
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    math: list = field(default_factory=list)
    symbols: dict = field(default_factory=dict)
    assumptions: list = field(default_factory=list)
    references: list = field(default_factory=list)

    def to_dict(self):
        """Return a JSON-safe plain-``dict`` representation."""
        return asdict(self)

    @classmethod
    def from_description(cls, text):
        """Parse a flat ``description`` string into a contract.

        - the first non-empty line becomes ``summary``
        - subsequent lines carrying an equation marker (see ``_MATH_RE``) become
          ``math`` entries
        - the remaining non-empty lines are joined into ``description``

        Returns ``None`` for empty / ``None`` input.
        """
        if text is None:
            return None
        text = str(text)
        if not text.strip():
            return None

        summary = ""
        math = []
        rest = []
        seen_summary = False
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if not seen_summary:
                summary = stripped
                seen_summary = True
                continue
            if _MATH_RE.search(stripped):
                math.append(stripped)
            else:
                rest.append(stripped)

        return cls(
            summary=summary,
            description="\n".join(rest),
            math=math)

    def merged_with_description(self, desc):
        """Return a copy with only EMPTY ``summary`` / ``math`` / ``description``
        filled from parsing ``desc``.

        Authored ``inputs`` / ``outputs`` / ``config`` / ``symbols`` /
        ``assumptions`` / ``references`` are preserved untouched.
        """
        merged = copy.deepcopy(self)
        parsed = ProcessContract.from_description(desc)
        if parsed is not None:
            if not merged.summary:
                merged.summary = parsed.summary
            if not merged.math:
                merged.math = list(parsed.math)
            if not merged.description:
                merged.description = parsed.description
        return merged


def _as_contract(declared):
    """Coerce a declared ``contract`` attribute into a ``ProcessContract``.

    Accepts a ``ProcessContract`` (returned as-is) or a plain ``dict`` (used as
    kwargs). Any other type yields ``None`` so resolution falls through.
    """
    if isinstance(declared, ProcessContract):
        return declared
    if isinstance(declared, dict):
        try:
            return ProcessContract(**declared)
        except TypeError:
            return None
    return None


def resolve_contract(instance):
    """Resolve the ``ProcessContract`` for a process/step ``instance``.

    Resolution order (first that yields a contract wins):

    1. a declared ``contract`` attribute (``ProcessContract`` or ``dict``),
       merged with the object's ``description`` (authored rows preserved);
    2. a description-derived contract from the object's ``description``;
    3. a docstring-derived contract from the class docstring.

    Never raises: odd input (``None``, a bare ``object()``, anything without a
    class docstring of its own) returns ``None``.
    """
    if instance is None:
        return None
    try:
        declared = getattr(instance, "contract", None)
        desc = getattr(instance, "description", None)

        contract = _as_contract(declared)
        if contract is not None:
            return contract.merged_with_description(desc or "")

        if desc:
            derived = ProcessContract.from_description(desc)
            if derived is not None:
                return derived

        # Docstring fallback — but ignore a docstring merely inherited from
        # ``object`` (so a bare ``object()`` resolves to None, while a class
        # with its own docstring resolves).
        doc = inspect.getdoc(type(instance))
        if doc and doc != inspect.getdoc(object):
            return ProcessContract.from_description(doc)
        return None
    except Exception:
        return None
