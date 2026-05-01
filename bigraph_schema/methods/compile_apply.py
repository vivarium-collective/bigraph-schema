"""
Compile a schema into a single inline Python function that applies an
update, replacing per-call plum dispatch with one straight-line call.

Design: ``compile_apply_emit`` is a plum-dispatched method, the same
pattern as ``apply``, ``default``, ``divide``, etc. Each schema type
registers an emitter that returns Python source for its type. The
emitters compose recursively at *compile* time (via dispatch); the
*resulting* function never goes through dispatch.

Usage::

    compiled = compile_apply(schema)        # exec'd Python function
    new_state, merges = compiled(state, update, ())

The compiled function is a flat Python body with no recursion through
the dispatch machinery. Custom/unsupported types fall back to a
captured reference to the dispatched ``apply`` for that subtree.

Cache invalidation: the caller (``Composite.find_instance_paths``)
clears the cache after structural changes, since schemas can be
mutated in place by ``apply(dict)``'s ``_divide`` branch.
"""

from plum import dispatch

from bigraph_schema.methods.apply import apply
from bigraph_schema.schema import (
    Node,
    Atom,
    Boolean,
    Or,
    And,
    Xor,
    Float,
    Integer,
    String,
    Wrap,
    Maybe,
    Overwrite,
    Const,
)


class CompileContext:
    """Accumulator for emitted Python source + captured closure values.

    The compile pass runs once per schema. ``emit`` appends a line to the
    function body (with current indent). ``fresh_var`` mints a unique
    local name. ``capture`` adds an object to the namespace that the
    final function will exec into, returning the local name to use in
    code (typically a schema reference or the dispatch fallback).
    """

    def __init__(self):
        self.lines = []
        self.indent = 1  # default body indent (one level inside def)
        self._counter = 0
        # Captured objects — passed as the exec namespace so the
        # compiled function can reference them by their assigned local
        # names without importing or dispatching.
        self.locals = {}

    def emit(self, line):
        if not line:
            self.lines.append('')
            return
        prefix = '    ' * self.indent
        for sub in line.split('\n'):
            self.lines.append(prefix + sub)

    def fresh(self, prefix='v'):
        self._counter += 1
        return f'_{prefix}{self._counter}'

    def capture(self, obj, name_hint='obj'):
        """Bind ``obj`` to a local in the compiled function's namespace.

        Returns the local name to use in emitted source.
        """
        self._counter += 1
        name = f'_{name_hint}{self._counter}'
        self.locals[name] = obj
        return name


@dispatch
def compile_apply_emit(schema: Node, ctx: CompileContext,
                        state_expr: str, update_expr: str) -> str:
    """Default fallback: emit a dispatched apply call.

    Used for types we haven't taught the compiler about (Map, Array,
    Tree, custom types like vEcoli's UniqueArray). Captures the schema
    and dispatched apply by reference so the runtime call doesn't go
    through plum lookup again — direct callable invocation.
    """
    schema_ref = ctx.capture(schema, 'schema')
    new = ctx.fresh('disp')
    ctx.emit(f'{new}, _sm = _dispatch_apply({schema_ref}, '
             f'{state_expr}, {update_expr}, ())')
    ctx.emit(f'if _sm: _merges.extend(_sm)')
    return new


@dispatch
def compile_apply_emit(schema: Const, ctx: CompileContext,
                        state_expr: str, update_expr: str) -> str:
    """Const: never updates."""
    return state_expr


@dispatch
def compile_apply_emit(schema: Overwrite, ctx: CompileContext,
                        state_expr: str, update_expr: str) -> str:
    """Overwrite: returns update directly (None-update already filtered
    by parent). Skipping the inner type's apply entirely matches
    ``apply(Overwrite)`` semantics."""
    return update_expr


@dispatch
def compile_apply_emit(schema: Maybe, ctx: CompileContext,
                        state_expr: str, update_expr: str) -> str:
    """Maybe: if state is None, take update; otherwise apply inner."""
    inner = schema._value
    new = ctx.fresh('m')
    ctx.emit(f'if {state_expr} is None:')
    ctx.indent += 1
    ctx.emit(f'{new} = {update_expr}')
    ctx.indent -= 1
    ctx.emit(f'else:')
    ctx.indent += 1
    inner_var = compile_apply_emit(inner, ctx, state_expr, update_expr)
    ctx.emit(f'{new} = {inner_var}')
    ctx.indent -= 1
    return new


@dispatch
def compile_apply_emit(schema: Wrap, ctx: CompileContext,
                        state_expr: str, update_expr: str) -> str:
    """Generic Wrap (DivideReset, Quote, etc.): delegate to inner.

    Matches ``apply(Wrap)``'s ``return apply(schema._value, ...)`` —
    the wrapper itself imposes no apply behavior, only the inner type
    does. (Overwrite, Maybe, Const have specific handlers above.)
    """
    return compile_apply_emit(schema._value, ctx, state_expr, update_expr)


@dispatch
def compile_apply_emit(schema: Float, ctx: CompileContext,
                        state_expr: str, update_expr: str) -> str:
    """Float: additive, with zero-update fast-path."""
    new = ctx.fresh('f')
    ctx.emit(f'if {update_expr} == 0 or {state_expr} is None:')
    ctx.indent += 1
    ctx.emit(f'{new} = {update_expr} if {state_expr} is None else {state_expr}')
    ctx.indent -= 1
    ctx.emit(f'else:')
    ctx.indent += 1
    ctx.emit(f'{new} = {state_expr} + {update_expr}')
    ctx.indent -= 1
    return new


@dispatch
def compile_apply_emit(schema: Integer, ctx: CompileContext,
                        state_expr: str, update_expr: str) -> str:
    """Integer: additive (same shape as Float)."""
    new = ctx.fresh('i')
    ctx.emit(f'if {update_expr} == 0 or {state_expr} is None:')
    ctx.indent += 1
    ctx.emit(f'{new} = {update_expr} if {state_expr} is None else {state_expr}')
    ctx.indent -= 1
    ctx.emit(f'else:')
    ctx.indent += 1
    ctx.emit(f'{new} = {state_expr} + {update_expr}')
    ctx.indent -= 1
    return new


@dispatch
def compile_apply_emit(schema: Boolean, ctx: CompileContext,
                        state_expr: str, update_expr: str) -> str:
    """Boolean: last-wins."""
    return update_expr


@dispatch
def compile_apply_emit(schema: Or, ctx: CompileContext,
                        state_expr: str, update_expr: str) -> str:
    """Or: ``state or update`` (matches ``apply(Or)``)."""
    new = ctx.fresh('or')
    ctx.emit(f'{new} = {state_expr} or {update_expr}')
    return new


@dispatch
def compile_apply_emit(schema: And, ctx: CompileContext,
                        state_expr: str, update_expr: str) -> str:
    """And: ``state and update``."""
    new = ctx.fresh('and')
    ctx.emit(f'{new} = {state_expr} and {update_expr}')
    return new


@dispatch
def compile_apply_emit(schema: Xor, ctx: CompileContext,
                        state_expr: str, update_expr: str) -> str:
    """Xor: ``(state or update) and not (state and update)``."""
    new = ctx.fresh('xor')
    ctx.emit(f'{new} = ({state_expr} or {update_expr}) and not '
             f'({state_expr} and {update_expr})')
    return new


@dispatch
def compile_apply_emit(schema: String, ctx: CompileContext,
                        state_expr: str, update_expr: str) -> str:
    """String: last-wins."""
    return update_expr


@dispatch
def compile_apply_emit(schema: Atom, ctx: CompileContext,
                        state_expr: str, update_expr: str) -> str:
    """Atom: ``state + update`` (matches ``apply(Atom)``)."""
    new = ctx.fresh('a')
    ctx.emit(f'if {state_expr} is None:')
    ctx.indent += 1
    ctx.emit(f'{new} = {update_expr}')
    ctx.indent -= 1
    ctx.emit(f'else:')
    ctx.indent += 1
    ctx.emit(f'{new} = {state_expr} + {update_expr}')
    ctx.indent -= 1
    return new


@dispatch
def compile_apply_emit(schema: dict, ctx: CompileContext,
                        state_expr: str, update_expr: str) -> str:
    """Dict (most common): for each schema field, apply if present in
    update.

    The caller has already checked ``update is not None`` (every emit
    is guarded by ``if u_var is not None`` from the parent), so we
    only handle the residual cases here: state-None replacement,
    non-dict update replacement, structural sentinel fallback, and
    the per-field walk for the normal case. Trimming the redundant
    None-check shaves bytecode that runs for every nested level
    (vEcoli schemas can be 5+ levels deep — listeners, agents, etc.).
    """
    new = ctx.fresh('d')
    # State-None replacement.
    ctx.emit(f'if {state_expr} is None:')
    ctx.indent += 1
    ctx.emit(f'{new} = {update_expr}')
    ctx.indent -= 1
    # Non-dict update: replace.
    ctx.emit(f'elif type({update_expr}) is not dict:')
    ctx.indent += 1
    ctx.emit(f'{new} = {update_expr}')
    ctx.indent -= 1
    # Structural sentinels: dispatch fallback (rare).
    ctx.emit(f'elif "_divide" in {update_expr} or "_add" in {update_expr} '
             f'or "_remove" in {update_expr} or "_type" in {update_expr}:')
    ctx.indent += 1
    schema_ref = ctx.capture(schema, 'dictschema')
    ctx.emit(f'{new}, _sm = _dispatch_apply({schema_ref}, '
             f'{state_expr}, {update_expr}, ())')
    ctx.emit(f'if _sm: _merges.extend(_sm)')
    ctx.indent -= 1
    ctx.emit(f'else:')
    ctx.indent += 1
    ctx.emit(f'{new} = {state_expr}')

    # Per-field branches.
    for key, subschema in schema.items():
        if isinstance(key, str) and key.startswith('_'):
            continue  # schema metadata
        u_var = ctx.fresh(f'u_{_sanitize(key)}')
        ctx.emit(f'{u_var} = {update_expr}.get({key!r})')
        ctx.emit(f'if {u_var} is not None:')
        ctx.indent += 1
        s_var = ctx.fresh(f's_{_sanitize(key)}')
        ctx.emit(f'{s_var} = {state_expr}.get({key!r})')
        # Recurse: dispatched compile_apply_emit emits inner code.
        inner_new = compile_apply_emit(subschema, ctx, s_var, u_var)
        # Write back if changed (matches dispatched apply(dict)).
        ctx.emit(f'if {inner_new} is not {s_var}:')
        ctx.indent += 1
        ctx.emit(f'{state_expr}[{key!r}] = {inner_new}')
        ctx.indent -= 1
        ctx.indent -= 1
    ctx.indent -= 1
    return new


def _sanitize(key):
    """Map an arbitrary schema key to a valid Python identifier suffix.

    Keeps ``a-zA-Z0-9_``; replaces anything else with ``_``. Used only
    for cosmetic clarity in generated locals; correctness doesn't
    depend on it.
    """
    if not isinstance(key, str):
        return 'k'
    out = []
    for ch in key:
        if ch.isalnum() or ch == '_':
            out.append(ch)
        else:
            out.append('_')
    return ''.join(out) or 'k'


def compile_apply(schema):
    """Return a Python function that applies an update under ``schema``.

    The returned function has the same signature/semantics as
    ``apply(schema, state, update, path)`` -> ``(new_state, merges)``
    but is a single straight-line function with no dispatch overhead
    on the hot path.
    """
    ctx = CompileContext()
    # Header
    header = ['def __compiled_apply(state, update, path=()):',
              '    if update is None:',
              '        return state, []',
              '    _merges = []']
    # Compile the body — top-level schema's emit handler decides shape.
    new_var = compile_apply_emit(schema, ctx, 'state', 'update')
    body = '\n'.join(ctx.lines)
    footer = f'    return {new_var}, _merges'
    source = '\n'.join(header) + '\n' + body + '\n' + footer

    # Bind the captured locals plus the dispatched apply fallback.
    namespace = dict(ctx.locals)
    namespace['_dispatch_apply'] = apply
    try:
        exec(source, namespace)
    except SyntaxError:
        # Defensive: if codegen ever produces invalid Python (bug or
        # exotic schema shape), surface it with the source so the
        # caller can pin the issue. Caller falls back to dispatch.
        raise SyntaxError(f'compile_apply produced invalid source:\n{source}')
    fn = namespace['__compiled_apply']
    # Stash the generated source on the function for debugging.
    fn.__source__ = source
    return fn
