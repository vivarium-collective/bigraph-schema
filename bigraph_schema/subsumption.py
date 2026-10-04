"""Declaration-level contract subsumption — the reverse of contract_admits.

contract_admits (assembly.py) asks "does this filler INSTANCE admit into the
hole?". Subsumption asks "does this process's DECLARED contract admit into the
hole's declared contract?" — the relation find_candidates needs. Purely
additive; the instance-level admit path is untouched.
"""
import re
from dataclasses import dataclass, field

try:
    from bigraph_schema.units import units as _unit_registry
except Exception:  # noqa: BLE001 - pint is optional, mirror contract_audit
    _unit_registry = None

from bigraph_schema.contract_expr import parse as _parse_expr, names_in as _names_in

_BRACKET = re.compile(r'^[A-Za-z_][\w]*\[(?P<units>[^\]]+)\]$')


def port_bounds(port):
    """(_min, _max, _units) from a raw port declaration, dict or string.

    Reads the RAW declaration, never a resolved schema (core.resolve drops
    _min/_max for plain float). A 'float[fg]' string carries units only.
    """
    if isinstance(port, dict):
        return port.get('_min'), port.get('_max'), port.get('_units')
    if isinstance(port, str):
        match = _BRACKET.match(port.strip())
        if match:
            return None, None, match.group('units')
        return None, None, None
    return None, None, None


def range_subsumes(required_port, candidate_port):
    """Candidate's numeric range ⊆ required's. Returns (ok, reason).

    An absent required bound means "unconstrained on that side" and accepts
    any candidate bound. An absent candidate bound against a present required
    bound is a failure (candidate admits values the hole forbids).
    """
    req_lo, req_hi, req_units = port_bounds(required_port)
    cand_lo, cand_hi, cand_units = port_bounds(candidate_port)

    # Bounds/units is the exact region (spec §4): normalize candidate bounds
    # into the required units before comparing, so a scale mismatch (3 g vs a
    # 5 mg ceiling) is caught. Incompatible/unparseable units are left raw and
    # surfaced by units_compatible instead.
    if req_units and cand_units and _unit_registry is not None:
        try:
            factor = _unit_registry.Quantity(1, cand_units).to(req_units).magnitude
            if cand_lo is not None:
                cand_lo = cand_lo * factor
            if cand_hi is not None:
                cand_hi = cand_hi * factor
        except Exception:  # noqa: BLE001 - incompatible/unparseable units: leave raw
            pass

    if req_lo is not None:
        if cand_lo is None or cand_lo < req_lo:
            return False, f'candidate _min {cand_lo} is below required _min {req_lo}'
    if req_hi is not None:
        if cand_hi is None or cand_hi > req_hi:
            return False, f'candidate _max {cand_hi} is above required _max {req_hi}'
    return True, None


def units_compatible(required_port, candidate_port):
    """Candidate units convertible to required units. Returns (ok, reason).

    Absent units on either side, or pint unavailable, is permissive (cannot
    disprove compatibility → do not reject). Matches contract_audit's stance.
    """
    _, _, req_units = port_bounds(required_port)
    _, _, cand_units = port_bounds(candidate_port)
    if not req_units or not cand_units:
        return True, None
    if _unit_registry is None:
        return True, None
    try:
        req = _unit_registry.Unit(req_units)
        cand = _unit_registry.Unit(cand_units)
    except Exception as error:  # noqa: BLE001 - unparseable unit: cannot check
        return True, f'unit parse skipped: {error}'
    if req.is_compatible_with(cand):
        return True, None
    return False, f'cannot convert candidate {cand_units!r} to required {req_units!r}'


def _normalized(expr):
    """A kind-stable signature of an expression: the referenced-name set plus
    the stripped source. Structural, not semantic (per spec §11): two exprs
    that parse to the same names + text are treated as the same guarantee.
    """
    try:
        ast = _parse_expr(expr)
        return (frozenset(_names_in(ast)), (expr or '').replace(' ', ''))
    except Exception:  # noqa: BLE001 - unparseable: fall back to raw text
        return (frozenset(), (expr or '').replace(' ', ''))


def conditions_cover(hole_contract, candidate_contract):
    """Does the candidate DECLARE every condition the hole requires? Structural.

    For each hole condition, the candidate must declare one of the same kind
    whose expression normalizes identically. A sound filter: it will not pass
    a candidate that fails to even claim a required guarantee. Returns
    (ok, missing).
    """
    missing = []
    hole_conditions = hole_contract.conditions() if hasattr(hole_contract, 'conditions') else []
    cand_conditions = candidate_contract.conditions() if hasattr(candidate_contract, 'conditions') else []
    cand_index = {(c['kind'], _normalized(c.get('expr'))) for c in cand_conditions}
    for condition in hole_conditions:
        key = (condition['kind'], _normalized(condition.get('expr')))
        if key not in cand_index:
            missing.append({'condition': f"{condition['kind']}:{condition.get('name')}",
                            'reason': f"candidate does not declare {condition['kind']} "
                                      f"{condition.get('name')!r} ({condition.get('expr')!r})"})
    return (not missing), missing


def face_subsumes(core, required_face, candidate_face):
    """Structural + bound/unit subsumption between two declared faces.

    The candidate must provide every required port at a type core.resolve
    accepts (mirrors face_conforms), and for each required port its numeric
    range must be ⊆ and its units convertible. Over-provided ports are fine
    and recorded. Returns (ok, fails, over_provides).
    """
    fails = []
    over_provides = []
    required_face = required_face or {}
    candidate_face = candidate_face or {}

    # Note: the per-port type check below mirrors the existing face_conforms
    # (core.resolve), which is intentionally loose on primitive type identity
    # (e.g. it does not reject 'string' for a 'float' port). Phase 2a's new
    # strictness is bounds/units (here) and conditions (conditions_cover) — NOT
    # primitive-type identity, so that subsumption mirrors the admit path.
    for direction in ('inputs', 'outputs'):
        required = required_face.get(direction) or {}
        provided = candidate_face.get(direction) or {}
        if not isinstance(required, dict) or not isinstance(provided, dict):
            continue
        for port, req_schema in required.items():
            if port not in provided:
                fails.append({'condition': f'face.{direction}.{port}',
                              'reason': f'candidate does not provide {direction} port {port!r}'})
                continue
            cand_schema = provided[port]
            try:
                core.resolve(req_schema, cand_schema)
            except Exception as error:  # noqa: BLE001 - resolve failure = non-conforming type
                fails.append({'condition': f'face.{direction}.{port}',
                              'reason': f'{direction} port {port!r} type does not resolve: {error}'})
                continue
            ok, reason = range_subsumes(req_schema, cand_schema)
            if not ok:
                fails.append({'condition': f'bounds.{direction}.{port}', 'reason': reason})
            ok, reason = units_compatible(req_schema, cand_schema)
            if not ok:
                fails.append({'condition': f'units.{direction}.{port}', 'reason': reason})
        for port in provided:
            if port not in required:
                over_provides.append(f'{direction}.{port}')

    return (not fails), fails, over_provides


@dataclass
class SubsumptionResult:
    ok: bool
    over_provides: list = field(default_factory=list)
    fails: list = field(default_factory=list)   # [{'condition','reason'}]


def contract_subsumes(core, hole_contract, candidate_contract):
    """Does candidate's declared contract satisfy the hole's? The reverse,
    declaration-level counterpart of contract_admits. Face + bounds + units
    are exact; conditions are structural (spec §4/§11). Returns a
    SubsumptionResult; ok iff nothing failed.
    """
    hole_face = getattr(hole_contract, 'face', None) or {}
    cand_face = getattr(candidate_contract, 'face', None) or {}
    _, fails, over = face_subsumes(core, hole_face, cand_face)
    _, missing = conditions_cover(hole_contract, candidate_contract)
    all_fails = list(fails) + list(missing)
    return SubsumptionResult(ok=(not all_fails), over_provides=over, fails=all_fails)
