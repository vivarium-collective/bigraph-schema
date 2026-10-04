"""Static audit of a ProcessContract: well-formedness, narrow soundness, and a
completeness grade. Pure analysis — never runs the process, never raises on a
well-formed-but-incomplete contract."""
from __future__ import annotations
from dataclasses import dataclass, field

_BARE = frozenset({'float', 'int', 'integer', 'number', 'string', 'boolean', 'bool', 'any'})

@dataclass
class Finding:
    severity: str   # 'error' | 'warning' | 'info'
    code: str
    where: str
    message: str

@dataclass
class AuditReport:
    ok: bool
    findings: list = field(default_factory=list)

def _port_constrained(port_type) -> bool:
    if isinstance(port_type, dict):
        keys = set(port_type)
        return bool(keys & {'_min', '_max', '_units', '_values'}) or port_type.get('_type') not in _BARE | {None}
    if isinstance(port_type, str):
        base = port_type.split('[', 1)[0].split('{', 1)[0].strip()
        return base not in _BARE or ('[' in port_type)
    return False

def completeness(contract) -> float:
    face = contract.face or {}
    ports = list((face.get('inputs') or {}).values()) + list((face.get('outputs') or {}).values())
    if not ports:
        port_score = 0.0
    else:
        port_score = sum(1 for p in ports if _port_constrained(p)) / len(ports)
    predicate_bonus = 0.25 if contract.conditions() else 0.0
    return min(1.0, 0.75 * port_score + predicate_bonus)
