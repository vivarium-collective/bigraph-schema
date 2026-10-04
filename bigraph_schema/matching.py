"""find_candidates — enumerate the process registry for a hole.

The reverse direction §4 of the design: given an open sorted site, return the
registered processes whose DECLARED contract subsumes the hole's, each with
why (full match or near-miss + failing condition). Enumerates core.link_registry
(the only process registry in this repo — there is no separate composite or
generator registry; a composite registered as a link is enumerated like any
other edge).
"""
from dataclasses import dataclass, field

from bigraph_schema.assembly import contract_of
from bigraph_schema.contract import resolve_contract
from bigraph_schema.subsumption import contract_subsumes


@dataclass
class CandidateMatch:
    address: str
    match: str                       # 'full' | 'partial'
    over_provides: list = field(default_factory=list)
    fails: list = field(default_factory=list)


def _candidate_contract(core, address):
    """Declared contract of a registered process, without a crash.

    Returns a ProcessContract or None. The cheapest declared-face source is
    resolve_contract(edge_class({}, core)) (the repo's own pattern); a
    candidate whose construction needs real config raises and is reported as
    None so the caller can skip it.
    """
    edge_class = core.link_registry.get(address)
    if edge_class is None:
        return None
    try:
        instance = edge_class({}, core)
        return resolve_contract(instance)
    except Exception:  # noqa: BLE001 - uninstantiable candidate: skip, don't crash
        return None


def find_candidates(core, site):
    """Processes whose declared contract satisfies the hole at `site`.

    Full matches first, then near-misses (face conforms but a bound/unit/
    condition fails). A candidate whose face does not conform at all, or that
    cannot be instantiated to read its contract, is omitted. Each result
    carries why.
    """
    hole = contract_of(core, site)
    if hole is None:
        return []

    full, partial = [], []
    for address in core.list_processes():
        candidate = _candidate_contract(core, address)
        if candidate is None:
            continue
        result = contract_subsumes(core, hole, candidate)
        if result.ok:
            full.append(CandidateMatch(address, 'full', result.over_provides, []))
        elif any(f['condition'].startswith('face.') for f in result.fails):
            continue
        else:
            partial.append(CandidateMatch(address, 'partial', result.over_provides, result.fails))

    return full + partial
