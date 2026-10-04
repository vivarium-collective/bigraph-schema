"""Static audit of a ProcessContract: well-formedness, narrow soundness, and a
completeness grade. Pure analysis — never runs the process, never raises on a
well-formed-but-incomplete contract."""
from __future__ import annotations
from dataclasses import dataclass, field

from bigraph_schema.contract_expr import parse, names_in, ExprError

try:
    from bigraph_schema.units import units as _unit_registry  # the shared pint registry
except Exception:  # noqa: BLE001 — units optional; absence is not an audit error
    _unit_registry = None

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

def audit_contract(core, contract) -> AuditReport:
    findings = []
    face = contract.face or {}
    in_ports = set((face.get('inputs') or {}).keys())
    out_ports = set((face.get('outputs') or {}).keys())
    for predicate in contract.conditions():
        name = predicate.get('name', '?')
        try:
            ast = parse(predicate['expr'])
        except ExprError as error:
            findings.append(Finding('error', 'expr_parse', f'predicate:{name}', str(error)))
            continue
        for path in names_in(ast):
            root, port = path[0], (path[1] if len(path) > 1 else None)
            if root == 'inputs' and port not in in_ports:
                findings.append(Finding('error', 'unknown_port', f'predicate:{name}', f'inputs.{port} is not a declared input port'))
            elif root == 'outputs' and port not in out_ports:
                findings.append(Finding('error', 'unknown_port', f'predicate:{name}', f'outputs.{port} is not a declared output port'))
            elif root not in {'inputs', 'outputs', 'config', 'state'}:
                findings.append(Finding('error', 'unknown_root', f'predicate:{name}', f'{root} is not a valid reference root'))
    for direction in ('inputs', 'outputs'):
        for port, port_type in (face.get(direction) or {}).items():
            if not isinstance(port_type, dict):
                continue
            lo, hi = port_type.get('_min'), port_type.get('_max')
            if lo is not None and hi is not None:
                if not (isinstance(lo, (int, float)) and isinstance(hi, (int, float))):
                    findings.append(Finding('error', 'bad_range', f'{direction}.{port}', f'_min/_max must be numeric, got {lo!r}/{hi!r}'))
                elif lo > hi:
                    findings.append(Finding('error', 'bad_range', f'{direction}.{port}', f'_min {lo} > _max {hi}'))
            unit = port_type.get('_units')
            if unit and _unit_registry is not None:
                try:
                    _unit_registry.Unit(unit)
                except Exception:  # noqa: BLE001 — a bad unit string is the finding
                    findings.append(Finding('error', 'bad_units', f'{direction}.{port}', f'cannot parse units {unit!r}'))
    grade = completeness(contract)
    findings.append(Finding('info', 'completeness', 'contract', f'completeness grade {grade:.2f}'))
    return AuditReport(ok=not any(f.severity == 'error' for f in findings), findings=findings)
