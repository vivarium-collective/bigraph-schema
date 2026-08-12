"""Regression tests for four execution-confirmed bugs.

Each test drives the exact branch that was broken-on-fire before its fix
(or, for the ``Core.bind`` case, confirms the branch is already safe and
still has zero callers — see that test's docstring for why no code change
was needed there).
"""

import pathlib

import pytest

from bigraph_schema import allocate_core, class_address
from bigraph_schema.edge import Edge
from bigraph_schema.schema import Key, String


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def core():
    return allocate_core()


# --- BUG 1: Core.bind — investigated, does NOT currently reproduce -------
#
# The originally-reported defect was that ``Core.bind`` called an
# undefined free function ``bind(...)``, raising ``NameError`` on any
# invocation. That call was already rewritten (commit 4e31a88, "fix:
# streamline broken-on-fire dispatch...") to an inline, self-contained
# implementation before this branch was created — ``Core.bind`` no longer
# references anything undefined. Deleting a working, documented method
# would be an unrelated behavior change, so it is left in place. This
# test locks in the two facts the task cared about: it has zero callers,
# and calling it does not raise.

def test_core_bind_has_no_callers_and_does_not_raise(core):
    """``Core.bind`` is unused anywhere in the package or test suite, and
    (already, as of commit 4e31a88) does not raise when called."""
    package_root = REPO_ROOT / 'bigraph_schema'
    callers = []
    for path in package_root.rglob('*.py'):
        text = path.read_text()
        if '.bind(' in text and 'def bind' not in text:
            callers.append(path)
    assert callers == [], f'unexpected callers of .bind(: {callers}'

    # Exercise all three branches (Key/Star mapping-style, Index) —
    # none should raise.
    assert core.bind('any', {}, 'foo', 'bar') == {'foo': 'bar'}
    assert core.bind('any', [1, 2, 3], 1, 'X') == [1, 'X', 3]
    assert core.bind('any', [1, 2, 3], 10, 'X') == [1, 2, 3, 'X']


# --- BUG 2: validate(Key, state) — De Morgan inversion --------------------

def test_validate_key_accepts_valid_string_and_int(core):
    """``validate(Key, state)`` inverted its condition
    (``not isinstance(state, int) or isinstance(state, str)``), so a
    valid *string* key was reported invalid even though ``check(Key, ...)``
    correctly accepts it. Fixed to mirror ``check``'s
    ``isinstance(state, int) or isinstance(state, str)`` logic."""
    from bigraph_schema.methods.validate import validate as _validate
    from bigraph_schema.methods.check import check as _check

    key_schema = Key(_value='placeholder')

    # A valid string key: no validation error, matching check() == True.
    assert _validate(core, key_schema, 'some_key') is None
    assert _check(key_schema, 'some_key') is True

    # A valid int key: also no error.
    assert _validate(core, key_schema, 5) is None
    assert _check(key_schema, 5) is True

    # An invalid key (neither int nor str): both report failure.
    assert _validate(core, key_schema, [1, 2]) is not None
    assert _check(key_schema, [1, 2]) is False


# --- BUG 3: validate(String, state) — copy-pasted Float message ----------

def test_validate_string_message_mentions_string_not_float(core):
    """``validate(String, state)`` returned the copy-pasted
    ``"Float schema but state is not a float"`` message on failure.
    The message now correctly describes the String schema."""
    from bigraph_schema.methods.validate import validate as _validate

    message = _validate(core, String(), 123)
    assert message is not None
    assert 'string' in message.lower()
    assert 'float' not in message.lower()


# --- BUG 4: port_merges / append_link_path TypeError on undeclared port --

class _UndeclaredPortProc(Edge):
    """Minimal process declaring a single input port ``a``."""

    def inputs(self):
        return {'a': 'float'}

    def outputs(self):
        return {}


def test_realize_link_skips_undeclared_port_wire(core):
    """A wire referencing a port with no declared schema (here, a ``b``
    entry merged into ``_inputs`` as ``None``, which is how a wired-but-
    undeclared port shows up) used to blow up: ``port_merges`` passed the
    ``None`` schema straight into ``append_link_path``, which does
    ``'_link_path' in schema`` — a ``TypeError`` for ``None``. Confirmed
    via execution before the fix:

        TypeError: argument of type 'NoneType' is not iterable

    at ``bigraph_schema/core.py`` in ``append_link_path``, reached from
    ``Core.realize`` -> ``resolve_merges`` -> ``append_link_path``, fed by
    ``port_merges``. ``port_merges`` now skips (``continue``) whenever
    ``core.jump`` resolves a wired key to ``None``, mirroring the sibling
    ``view_ports``'s ``if subschema is None: continue`` guard.
    """
    encode = {
        'address': class_address(_UndeclaredPortProc),
        'config': {},
        # 'b' has no declared type (simulates an undeclared/unspecified
        # port reached via a wire) but IS wired -- this used to raise.
        '_inputs': {'b': None},
        'inputs': {'a': ['a'], 'b': ['b']},
        'outputs': {},
    }

    decode_schema, decode_state, merges = core.realize('link', encode)

    # No exception; the undeclared port's wire is preserved in state
    # (port_merges just skips contributing a *merge* for it).
    assert decode_state['inputs'] == {'a': ['a'], 'b': ['b']}
