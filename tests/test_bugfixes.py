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


# --- BUG 5: realize_link's 'instance' branch never rebinds a missing core -

class _MinimalEdge(Edge):
    """Minimal process with no ports — enough to exercise realize_link's
    ``'instance' in encode`` branch without any real domain logic."""

    def inputs(self):
        return {}

    def outputs(self):
        return {}


def test_realize_link_rebinds_core_on_a_supplied_instance_with_none_core(core):
    """A caller can hand realize_link an already-constructed ``'instance'``
    (``encode['instance']``) rather than an ``address``/``config`` pair —
    this is a real, supported, documented path (the branch's own comment:
    "Instance already exists — skip instantiation"), used e.g. by
    v2ecoli's ``EcoliStep``/``EcoliProcess`` adapter classes, whose custom
    ``__init__`` bypasses ``Edge.__init__``'s own "must provide a core"
    guard and can silently end up with ``self.core = None`` (a stale or
    unset ambient fallback).

    Before this fix, only the OTHER branch (fresh construction from
    address+config) defended against this — line ~673: ``if not
    hasattr(edge_instance, 'core') or edge_instance.core is None:
    edge_instance.core = core``. The 'instance' branch had no equivalent
    check, so a supplied instance with ``core=None`` built and ran without
    error and only failed much later, in ``Link.serialize``:

        config_schema = instance.core.access(instance.config_schema)
        AttributeError: 'NoneType' object has no attribute 'access'

    Confirmed live in production (sms-ecoli chain-dispatch, real
    CloudWatch logs, commit c2ae8eb) and reproduced locally against the
    real v2ecoli ``ecoli_baseline`` composite before this fix.
    """
    instance = _MinimalEdge({}, core=core)
    assert instance.core is core  # sanity: real construction binds it

    # Simulate the real defect: an instance handed to realize_link whose
    # own core got lost (e.g. via a custom __init__ that bypasses Edge's
    # guard, like v2ecoli's EcoliStep falling back to a stale/unset
    # ambient global). realize_link must not trust it blindly.
    instance.core = None

    encode = {
        'address': class_address(_MinimalEdge),
        'config': {},
        'instance': instance,
    }

    decode_schema, decode_state, merges = core.realize('link', encode)

    assert decode_state['instance'].core is core, (
        "realize_link's 'instance' branch did not rebind a missing core "
        "on a caller-supplied instance — Composite.serialize_state() will "
        "later crash on it with AttributeError: 'NoneType' object has no "
        "attribute 'access' (Link.serialize's instance.core.access(...))."
    )
