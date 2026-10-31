"""Regression for #188 — face-sorted sites reject ``local:!module.Class`` fillers.

A site sorted by a contract/face is filled by a bare edge *address*: its face is
the face of the class the address names (:func:`collect_face` ->
:func:`_face_from_address`). Resolving that class only through
``core.link_registry.get(canonical['data'])`` works for the registry form
(``local:<name>``) but NOT for the portable import form
(``local:!pkg.mod.Class``) — there ``canonical['data']`` is ``'!pkg.mod.Class'``,
never a registry key, so the lookup returned ``None``, ``collect_face`` reported
no ports, and ``admits`` rejected an otherwise-conforming filler.

The fix routes ``local`` addresses through
:func:`bigraph_schema.protocols.local_lookup`, which resolves BOTH forms. These
tests pin the import form (the bug) and keep the registry form (the control).
"""

from bigraph_schema.core import allocate_core
from bigraph_schema.contract import ProcessContract
from bigraph_schema.assembly import admits, admits_why, collect_face
from bigraph_schema.edge import Edge
from bigraph_schema.schema import Site


class _FaceWidget(Edge):
    """A trivial local edge with a real face, importable by module path so the
    ``local:!module.path.Class`` form can resolve it by import."""

    def inputs(self):
        return {'mass': 'float'}

    def outputs(self):
        return {'mass': 'float'}

    def update(self, state, interval=None):
        return {}


# Self-describing module-path (`!`) address for the class above — the portable
# form Jim's repro uses. ``__name__`` is the importable path of this test module.
_IMPORT_ADDRESS = f'local:!{__name__}._FaceWidget'
_REGISTRY_NAME = 'face_widget'
_REGISTRY_ADDRESS = f'local:{_REGISTRY_NAME}'

_HOLE_SORT = 'mass_hole'


def _core_with_hole():
    core = allocate_core()
    core.contract_registry[_HOLE_SORT] = ProcessContract(
        face={'inputs': {'mass': 'float'}, 'outputs': {'mass': 'float'}})
    return core


def test_import_form_filler_face_resolves():
    """``_FaceWidget``'s ports are found through the ``!`` import address — the
    unit at the heart of the bug (previously an empty face)."""
    core = _core_with_hole()
    inputs, outputs = collect_face(core, _IMPORT_ADDRESS)
    assert set(inputs) == {'mass'}
    assert set(outputs) == {'mass'}


def test_face_sorted_site_admits_import_form_filler():
    """THE FIX: a face-sorted site admits the ``local:!module.Class`` filler
    even though that class is NOT in the link registry."""
    core = _core_with_hole()
    # Guard: the class is genuinely absent from the registry, so admission can
    # only come from the import-form resolution path, not a stray registration.
    assert core.link_registry.get(f'{__name__}._FaceWidget') is None

    ok, reason = admits_why(core, Site(_sort=_HOLE_SORT), _IMPORT_ADDRESS)
    assert ok, f'import-form filler was rejected: {reason}'
    assert admits(core, Site(_sort=_HOLE_SORT), _IMPORT_ADDRESS)


def test_face_sorted_site_still_admits_registry_form_filler():
    """CONTROL: the bare ``local:<name>`` registry form admits exactly as before
    (the fix keeps the registry ``.get(...)`` path intact)."""
    core = _core_with_hole()
    core.register_link(_REGISTRY_NAME, _FaceWidget)

    ok, reason = admits_why(core, Site(_sort=_HOLE_SORT), _REGISTRY_ADDRESS)
    assert ok, f'registry-form filler was rejected: {reason}'
    assert admits(core, Site(_sort=_HOLE_SORT), _REGISTRY_ADDRESS)
