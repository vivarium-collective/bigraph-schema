"""A tiny, safe expression language for contract predicates.

parse() -> an Expr tree (frozen); names_in() -> referenced port/config/state
paths; evaluate() -> value against a binding environment. NEVER uses eval/exec:
the string is parsed with the stdlib ``ast`` module and every node is checked
against a whitelist before a hand-written evaluator walks it.
"""
from __future__ import annotations
import ast as _ast
import operator
from dataclasses import dataclass

class ExprError(ValueError):
    """A contract expression is malformed or uses a disallowed construct."""

REDUCERS = frozenset({'sum', 'min', 'max', 'abs', 'all', 'any'})
_CMP = {_ast.Eq: '==', _ast.NotEq: '!=', _ast.Lt: '<', _ast.LtE: '<=', _ast.Gt: '>', _ast.GtE: '>='}
_BIN = {_ast.Add: '+', _ast.Sub: '-', _ast.Mult: '*', _ast.Div: '/'}

@dataclass(frozen=True)
class Expr:
    kind: str           # 'name' | 'lit' | 'cmp' | 'bin' | 'neg' | 'call'
    value: object = None
    op: str = ''
    args: tuple = ()

def parse(expr: str) -> Expr:
    if not isinstance(expr, str) or not expr.strip():
        raise ExprError('expression must be a non-empty string')
    try:
        tree = _ast.parse(expr, mode='eval').body
    except SyntaxError as error:
        raise ExprError(f'could not parse {expr!r}: {error.msg}') from None
    return _convert(tree, expr)

def _convert(node, expr):
    if isinstance(node, _ast.Constant):
        if isinstance(node.value, (int, float, bool, str)):
            return Expr('lit', value=node.value)
        raise ExprError(f'unsupported literal in {expr!r}')
    if isinstance(node, _ast.Name):
        if node.id == 'tol':
            return Expr('name', value=('tol',))
        raise ExprError(f'bare name {node.id!r} — use a path like inputs.x (only `tol` is bare)')
    if isinstance(node, _ast.Attribute):
        return Expr('name', value=_path(node, expr))
    if isinstance(node, _ast.UnaryOp) and isinstance(node.op, _ast.USub):
        return Expr('neg', args=(_convert(node.operand, expr),))
    if isinstance(node, _ast.BinOp) and type(node.op) in _BIN:
        return Expr('bin', op=_BIN[type(node.op)], args=(_convert(node.left, expr), _convert(node.right, expr)))
    if isinstance(node, _ast.Compare):
        if len(node.ops) != 1 or type(node.ops[0]) not in _CMP:
            raise ExprError(f'only a single simple comparison is allowed in {expr!r}')
        return Expr('cmp', op=_CMP[type(node.ops[0])], args=(_convert(node.left, expr), _convert(node.comparators[0], expr)))
    if isinstance(node, _ast.Call):
        if not isinstance(node.func, _ast.Name) or node.func.id not in REDUCERS:
            raise ExprError(f'only the reducers {sorted(REDUCERS)} may be called, not {_ast.dump(node.func)}')
        if node.keywords:
            raise ExprError('reducers take no keyword arguments')
        if len(node.args) != 1:
            raise ExprError(f'{node.func.id}() takes exactly 1 argument, got {len(node.args)}')
        return Expr('call', op=node.func.id, args=tuple(_convert(a, expr) for a in node.args))
    raise ExprError(f'disallowed construct {type(node).__name__} in {expr!r}')

def _path(node, expr):
    parts = []
    while isinstance(node, _ast.Attribute):
        if node.attr.startswith('__'):
            raise ExprError(f'dunder path segment {node.attr!r} not allowed in {expr!r}')
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, _ast.Name):
        raise ExprError(f'a name path must start at a bare root (inputs/outputs/config/state) in {expr!r}')
    parts.append(node.id)
    return tuple(reversed(parts))

def names_in(ast: Expr) -> set:
    found = set()
    def walk(node):
        if node.kind == 'name' and node.value != ('tol',):
            found.add(node.value)
        for arg in node.args:
            walk(arg)
    walk(ast)
    return found

_BIN_OP = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv}
_REDUCE = {'sum': sum, 'min': min, 'max': max, 'abs': abs, 'all': all, 'any': any}

def evaluate(ast: Expr, env: dict, *, tol: float = 0.0):
    def ev(node):
        if node.kind == 'lit':
            return node.value
        if node.kind == 'name':
            if node.value == ('tol',):
                return tol
            if node.value not in env:
                raise ExprError(f'no binding for {".".join(node.value)}')
            return env[node.value]
        if node.kind == 'neg':
            return -ev(node.args[0])
        if node.kind == 'bin':
            a, b = ev(node.args[0]), ev(node.args[1])
            return _BIN_OP[node.op](a, b)
        if node.kind == 'call':
            values = [ev(a) for a in node.args]
            return _REDUCE[node.op](*values) if node.op == 'abs' else _REDUCE[node.op](values[0])
        if node.kind == 'cmp':
            a, b = ev(node.args[0]), ev(node.args[1])
            if node.op == '==':
                return abs(a - b) <= tol if isinstance(a, (int, float)) and isinstance(b, (int, float)) else a == b
            if node.op == '!=':
                return not (abs(a - b) <= tol) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else a != b
            return {'<': a < b, '<=': a <= b, '>': a > b, '>=': a >= b}[node.op]
        raise ExprError(f'cannot evaluate node kind {node.kind!r}')
    return ev(ast)
