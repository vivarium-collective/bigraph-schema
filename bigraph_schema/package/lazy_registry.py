"""
==================
Lazy link registry
==================

A ``dict``-compatible mapping that stands in for ``core.link_registry`` and
makes *process discovery* lazy.

Eager discovery (``recursive_dynamic_import``) imports EVERY submodule of EVERY
bigraph-schema-dependent distribution at ``allocate_core()`` time — including
heavy ones (torch, vEcoli, copasi, ...) even when a composite only instantiates
two processes. That import walk dominates cold start.

``LazyLinkRegistry`` splits the registry into two parts:

* ``_resolved`` — ``name -> class`` for links that are actually imported.
* ``_index``    — ``name -> module`` placeholders for links that are *known*
                  (their name and defining module were recorded by a prior real
                  discovery) but **not yet imported**.

Enumeration of NAMES (``keys``/``__iter__``/``__contains__``/``len``) consults
both halves and imports nothing, so callers that list "every available process"
(workbench Registry tab, ``vivarium-interface`` validation, ``/list-processes``)
keep working. Fetching a CLASS (``__getitem__``/``get``) imports just that one
module on first use; iterating VALUES (``items``/``values``/``dict(reg)``)
transparently materializes everything, so class-consuming callers stay correct
(they simply pay the import cost, which is exactly the "list them all" case).

The mapping is intentionally standalone (no ``core`` reference): materializing a
placeholder only needs to import the module and pick out its ``Edge`` classes,
so a ``LazyLinkRegistry`` can be cheaply ``.copy()``-ed per ``allocate_core()``
without materializing anything.
"""

import importlib
import inspect

from bigraph_schema import Edge


def _module_edge_classes(module_name):
    """Return ``{class_name: class}`` for Edge subclasses DEFINED in ``module_name``.

    Only classes whose ``__module__`` equals ``module_name`` are returned, so a
    process merely imported into the module (and owned by another module) is not
    re-registered here — it is handled when its own module materializes. This
    mirrors the fully-qualified-name keying used by eager discovery.
    """
    module = importlib.import_module(module_name)
    found = {}
    for _name, cls in inspect.getmembers(module, inspect.isclass):
        if not issubclass(cls, Edge) or cls is Edge:
            continue
        if cls.__module__ != module_name:
            continue
        found[cls.__name__] = cls
    return found


class LazyLinkRegistry:
    """A ``dict``-like link registry with lazy per-module import.

    Parameters
    ----------
    resolved : dict | None
        Already-imported ``name -> class`` entries (e.g. the base ``edge``
        link, or anything registered explicitly before discovery).
    index : dict | None
        Placeholder ``name -> module`` entries: a name that can be resolved by
        importing ``module`` and picking out the matching Edge class.
    """

    def __init__(self, resolved=None, index=None):
        self._resolved = dict(resolved or {})
        self._index = dict(index or {})
        # A placeholder that is superseded by an explicit set/registration must
        # never be resurrected from the index, so setitem always pops the index.

    # -- materialization ---------------------------------------------------

    def _materialize_module(self, module_name):
        """Import ``module_name`` and promote every placeholder it backs."""
        try:
            classes = _module_edge_classes(module_name)
        except Exception as exc:  # noqa: BLE001 - one bad module must not kill lookups
            # A placeholder whose module now fails to import: drop its
            # placeholders so we don't retry on every access, and surface
            # nothing (behaves like an absent link, same as eager's skip).
            print(
                f"lazy discovery: skipping `{module_name}` "
                f"({type(exc).__name__} raised at import: {exc})")
            for key in [k for k, v in self._index.items() if v == module_name]:
                del self._index[key]
            return
        for key in [k for k, v in self._index.items() if v == module_name]:
            short = key.rsplit(".", 1)[-1]
            cls = classes.get(short)
            if cls is not None:
                self._resolved[key] = cls
            del self._index[key]

    def _materialize_key(self, key):
        module_name = self._index.get(key)
        if module_name is not None:
            self._materialize_module(module_name)

    def materialize_all(self):
        """Import every placeholder module — used before enumerating VALUES."""
        for module_name in sorted(set(self._index.values())):
            self._materialize_module(module_name)

    # -- mapping protocol --------------------------------------------------

    def __contains__(self, key):
        return key in self._resolved or key in self._index

    def __getitem__(self, key):
        if key in self._resolved:
            return self._resolved[key]
        if key in self._index:
            self._materialize_key(key)
            if key in self._resolved:
                return self._resolved[key]
        raise KeyError(key)

    def get(self, key, default=None):
        if key in self._resolved:
            return self._resolved[key]
        if key in self._index:
            self._materialize_key(key)
        return self._resolved.get(key, default)

    def __setitem__(self, key, value):
        self._index.pop(key, None)
        self._resolved[key] = value

    def __delitem__(self, key):
        existed = False
        if key in self._resolved:
            del self._resolved[key]
            existed = True
        if key in self._index:
            del self._index[key]
            existed = True
        if not existed:
            raise KeyError(key)

    def __iter__(self):
        seen = set(self._resolved)
        yield from self._resolved
        for key in self._index:
            if key not in seen:
                yield key

    def __len__(self):
        return len(set(self._resolved) | set(self._index))

    def keys(self):
        return list(self)

    def names(self):
        """All known link names without importing anything (alias of keys)."""
        return list(self)

    def items(self):
        self.materialize_all()
        return self._resolved.items()

    def values(self):
        self.materialize_all()
        return self._resolved.values()

    def pop(self, key, *default):
        if key in self._resolved:
            self._index.pop(key, None)
            return self._resolved.pop(key)
        if key in self._index:
            del self._index[key]
            if key in self._resolved:
                return self._resolved.pop(key)
        if default:
            return default[0]
        raise KeyError(key)

    def update(self, other):
        for key, value in dict(other).items():
            self[key] = value

    def setdefault(self, key, default=None):
        if key in self:
            return self[key]
        self[key] = default
        return default

    def copy(self):
        """Shallow copy that preserves placeholders (does NOT materialize)."""
        return LazyLinkRegistry(resolved=self._resolved, index=self._index)

    # Convenience / introspection ----------------------------------------

    @property
    def resolved_count(self):
        return len(self._resolved)

    @property
    def pending_count(self):
        return len(self._index)

    def __repr__(self):
        return (
            f"LazyLinkRegistry(resolved={len(self._resolved)}, "
            f"pending={len(self._index)})")
