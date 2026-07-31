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


#: Documentary fields — ``annotate`` may touch these and nothing else.
ANNOTATABLE = (
    'summary', 'description', 'inputs', 'outputs', 'config',
    'math', 'symbols', 'assumptions', 'references')


class AmendmentError(Exception):
    """An amendment that would make a contract unsound or looser."""


@dataclass
class Amendment:
    """One ordered, provenance-carrying refinement of a contract.

    ``op`` is **``narrow``** (tighten the face, or add a predicate) or
    **``annotate``** (refine documentation). There is deliberately no
    ``extend``: a contract may get stricter and better-documented as it flows
    down through composition and filling, never looser and never gain ports
    it did not require. That is what keeps amendment *monotone*, and so keeps
    ``admits`` sound.
    """

    op: str
    target: str = ""
    detail: dict = field(default_factory=dict)
    by: str = ""
    when: str = ""
    why: str = ""

    def to_dict(self):
        detail = {
            key: (value if _is_jsonish(value) else repr(value))
            for key, value in (self.detail or {}).items()}
        return {'op': self.op, 'target': self.target, 'detail': detail,
                'by': self.by, 'when': self.when, 'why': self.why}


def _is_jsonish(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_jsonish(item) for item in value)
    if isinstance(value, dict):
        return all(_is_jsonish(item) for item in value.values())
    return False


@dataclass
class ProcessContract:
    """A structured superset of ``Edge.description``.

    A contract is the full interface spec of an **edge or a site**, and
    ``face`` is its machine-checkable **typed core** — the part ``admits``
    reads. The remaining fields are the documented meaning around that same
    interface. They are not two objects: the face is the typed projection of
    one contract.

    Note the two senses of "inputs" here, kept deliberately separate:
    ``face['inputs']`` maps a port to its **type** (checked), while
    ``inputs`` maps a port to its **prose semantics** (documentation only).

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
    face: dict = field(default_factory=lambda: {'inputs': {}, 'outputs': {}})
    amendments: list = field(default_factory=list)

    def face_ports(self, direction):
        """The typed core's ports for ``'inputs'`` or ``'outputs'``."""
        found = (self.face or {}).get(direction)
        return found if isinstance(found, dict) else {}

    def predicates(self):
        """Every predicate contributed by a ``narrow`` amendment."""
        found = []
        for amendment in self.amendments:
            predicate = (amendment.detail or {}).get('predicate')
            if callable(predicate):
                found.append(predicate)
        return found

    def to_dict(self):
        """Return a JSON-safe plain-``dict`` representation."""
        result = asdict(self)
        result['amendments'] = [
            amendment.to_dict() for amendment in self.amendments]
        return result

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


def _as_amendment(amendment):
    if isinstance(amendment, Amendment):
        return amendment
    if isinstance(amendment, dict):
        try:
            return Amendment(**amendment)
        except TypeError as error:
            raise AmendmentError(f'malformed amendment: {error}') from None
    raise AmendmentError(f'not an amendment: {amendment!r}')


def amend(contract, amendment):
    """Append one amendment to ``contract``, returning a **new** contract.

    Pure and append-only: the input is never mutated and no amendment is ever
    dropped, so a contract carries the full record of how its interface
    evolved — each entry with its ``by``/``when``/``why``.

    - ``narrow`` tightens the typed core: it may add ports the face did not
      require, and may add a ``predicate`` that a filler must also satisfy.
      It may **not** redefine a port the face already requires — replacing a
      port's type could widen what conforms, which would break monotonicity —
      and it may not remove one.
    - ``annotate`` refines documentation only, leaving admissibility exactly
      as it was.
    - anything else (notably ``extend``) is refused: a contract may become
      stricter and better-documented, never looser.

    **Monotonicity.** Because ``narrow`` only ever adds required ports and
    conjunctive predicates, every filler admissible under the amended
    contract was admissible under the original. ``admits`` therefore stays
    sound as a contract flows down through composition and filling.
    """
    amendment = _as_amendment(amendment)
    amended = copy.deepcopy(contract)
    detail = amendment.detail or {}

    if amendment.op == 'narrow':
        for direction in ('inputs', 'outputs'):
            added = detail.get(direction)
            if not isinstance(added, dict):
                continue
            existing = amended.face_ports(direction)
            for port, port_type in added.items():
                if port in existing and existing[port] != port_type:
                    raise AmendmentError(
                        f'narrow may not redefine the {direction[:-1]} port '
                        f'{port!r} ({existing[port]!r} -> {port_type!r}): '
                        f'replacing a port type can widen what conforms. '
                        f'Add a new port or a predicate instead.')
            amended.face = dict(amended.face or {})
            amended.face[direction] = {**existing, **added}

    elif amendment.op == 'annotate':
        for key, value in detail.items():
            if key not in ANNOTATABLE:
                raise AmendmentError(
                    f'annotate may only touch documentation '
                    f'{ANNOTATABLE}, not {key!r}')
            current = getattr(amended, key)
            if isinstance(current, dict) and isinstance(value, dict):
                setattr(amended, key, {**current, **value})
            elif isinstance(current, list) and isinstance(value, list):
                setattr(amended, key, [*current, *value])
            else:
                setattr(amended, key, value)

    else:
        raise AmendmentError(
            f'unsupported amendment op {amendment.op!r} — a contract may '
            f'only be narrowed or annotated, so that it can get stricter '
            f'and better documented but never looser')

    amended.amendments = [*amended.amendments, amendment]
    return amended


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
            resolved = contract.merged_with_description(desc or "")
        elif desc:
            resolved = ProcessContract.from_description(desc)
        else:
            # Docstring fallback — but ignore a docstring merely inherited
            # from ``object`` (so a bare ``object()`` resolves to None, while
            # a class with its own docstring resolves).
            doc = inspect.getdoc(type(instance))
            resolved = (
                ProcessContract.from_description(doc)
                if doc and doc != inspect.getdoc(object)
                else None)

        return _with_declared_face(instance, resolved)
    except Exception:
        return None


def _with_declared_face(instance, contract):
    """Fill an edge's typed core from its own ``interface()``.

    The face is the contract's machine-checkable projection, so an edge that
    reports ports carries them on its contract — an authored ``face`` wins.
    """
    interface = getattr(instance, 'interface', None)
    if not callable(interface):
        return contract

    try:
        face = interface() or {}
    except Exception:
        return contract

    declared = {
        'inputs': dict(face.get('inputs') or {}),
        'outputs': dict(face.get('outputs') or {})}
    if not declared['inputs'] and not declared['outputs']:
        return contract

    if contract is None:
        return ProcessContract(face=declared)
    if not contract.face_ports('inputs') and not contract.face_ports('outputs'):
        contract = copy.deepcopy(contract)
        contract.face = declared
    return contract
