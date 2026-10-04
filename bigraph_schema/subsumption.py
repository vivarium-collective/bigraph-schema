"""Declaration-level contract subsumption — the reverse of contract_admits.

contract_admits (assembly.py) asks "does this filler INSTANCE admit into the
hole?". Subsumption asks "does this process's DECLARED contract admit into the
hole's declared contract?" — the relation find_candidates needs. Purely
additive; the instance-level admit path is untouched.
"""
import re

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
    req_lo, req_hi, _ = port_bounds(required_port)
    cand_lo, cand_hi, _ = port_bounds(candidate_port)

    if req_lo is not None:
        if cand_lo is None or cand_lo < req_lo:
            return False, f'candidate _min {cand_lo} is below required _min {req_lo}'
    if req_hi is not None:
        if cand_hi is None or cand_hi > req_hi:
            return False, f'candidate _max {cand_hi} is above required _max {req_hi}'
    return True, None
