"""
=========
Protocols
=========

This module contains the protocols for retrieving processes from address.
"""

import sys
import inspect
import importlib


def function_module(function):
    """
    Retrieves the fully qualified name of a given function.
    """
    module = inspect.getmodule(function)

    return f'{module.__name__}.{function.__name__}'


def local_lookup_module(address):
    """Local Module Protocol

    Retrieves local module
    """
    if '.' in address:
        module_name, class_name = address.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    else:
        module = sys.modules[__name__]
        if hasattr(module, address):
            return getattr(sys.modules[__name__], address)


def local_lookup_registry(core, address):
    """Process Registry Protocol

    Retrieves from the process registry
    """
    return core.link_registry.get(address)


def local_lookup(core, address):
    """Local Lookup Protocol

    Retrieves local processes, from the process registry or from a local module
    """
    if address[0] == '!':
        instantiate = local_lookup_module(address[1:])
    else:
        instantiate = local_lookup_registry(core, address)
    return instantiate


# ---------------------------------------------------------------------------
# Address portability validation
# ---------------------------------------------------------------------------
# A composite document that realizes in one core (e.g. a full, discovering
# ``allocate_core``) can fail in another (e.g. a thin/plain core, or a detached
# run subprocess that builds its own core) when an edge is wired by a bare
# ``local:<name>`` address. That form resolves via ``local_lookup_registry`` —
# ``core.link_registry[<name>]`` — so it depends on the *resolving* core having
# ``<name>`` registered. The self-describing ``local:!module.path.Class`` form
# resolves by import (``local_lookup_module``) and is portable across cores.
#
# These helpers surface registry-dependent addresses so authors can prefer the
# portable form, and check a document against the specific core that will
# realize it — turning a deep, unlocated ``no link found at address`` realize
# error into an upfront, path-located report.

# Edge-internal keys that are NOT sub-state to recurse into.
_EDGE_INTERNAL_KEYS = frozenset(
    {'address', 'config', 'inputs', 'outputs', 'interval', '_type', 'instance'})


def iter_link_addresses(state, path=()):
    """Yield ``(path, address)`` for every edge-like node in a composite
    ``state`` — a dict carrying an ``'address'`` value — recursing into
    sub-states (so nested/composite processes are covered too)."""
    if not isinstance(state, dict):
        return
    address = state.get('address')
    if isinstance(address, (str, dict)):
        yield path, address
    for key, value in state.items():
        if key in _EDGE_INTERNAL_KEYS:
            continue
        yield from iter_link_addresses(value, path + (key,))


def _normalize(address):
    # Local import avoids a protocols<->schema import cycle at module load.
    from bigraph_schema.schema import normalize_address
    return normalize_address(address)


def registry_dependent_addresses(document):
    """Return ``[(path, name)]`` for every edge wired by a bare ``local:<name>``
    address — i.e. one that resolves via the process registry and so requires
    the resolving core to have ``<name>`` registered.

    The portable ``local:!module.path.Class`` form and non-``local`` protocols
    are NOT reported. Use as a portability lint: a document with an empty result
    realizes in ANY core; anything reported couples the document to a specific
    core's registrations (the ``build_core`` vs ``allocate_core`` divergence)."""
    reported = []
    for path, address in iter_link_addresses(document):
        norm = _normalize(address)
        if not isinstance(norm, dict) or norm.get('protocol') != 'local':
            continue
        data = norm.get('data')
        if isinstance(data, str) and not data.startswith('!'):
            reported.append((path, data))
    return reported


def unresolvable_addresses(document, core):
    """Return ``[(path, address)]`` for every ``local`` edge whose address does
    NOT resolve in ``core`` (via :func:`local_lookup`).

    Check a document against the exact core that will realize it — e.g. a run
    subprocess's ``build_core`` — before running, so a missing process is an
    upfront, located error instead of a deep ``no link found at address``."""
    reported = []
    for path, address in iter_link_addresses(document):
        norm = _normalize(address)
        if not isinstance(norm, dict) or norm.get('protocol') != 'local':
            continue
        data = norm.get('data')
        try:
            resolved = local_lookup(core, data) if isinstance(data, str) else None
        except Exception:
            resolved = None
        if resolved is None:
            reported.append((path, address))
    return reported


def assert_portable_addresses(document):
    """Raise ``ValueError`` if ``document`` contains any registry-dependent
    ``local:<name>`` address (see :func:`registry_dependent_addresses`).

    A drop-in gate for CI / a workspace's save-time validation: it keeps a
    composite realizable in any core by rejecting non-portable addresses, with a
    message that names each offending path and suggests the ``!`` form."""
    reported = registry_dependent_addresses(document)
    if reported:
        lines = '\n'.join(
            f"  {'.'.join(map(str, p)) or '<root>'}: 'local:{name}' "
            f"-> prefer 'local:!<module>.{name}'"
            for p, name in reported)
        raise ValueError(
            "composite has non-portable registry-dependent address(es); these "
            "resolve only where the process is pre-registered (e.g. a full "
            "allocate_core) and fail in a thin/plain core such as a run "
            f"subprocess's build_core:\n{lines}")


