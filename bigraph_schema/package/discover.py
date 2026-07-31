import importlib
import importlib.metadata
import importlib.util
import pkgutil
import inspect
import os
import sys
import json
import hashlib
import tempfile
from typing import Dict, List, Tuple, Set, Type

from bigraph_schema import Edge
from bigraph_schema.schema import Node


# Submodule / subpackage names that core discovery must never import.
# Test scaffolding routinely pulls in heavy, test-only optional
# dependencies (e.g. fastapi/starlette ``TestClient`` needs httpx) and
# registers no Processes/Steps/Types worth discovering. Importing it can
# raise arbitrary exceptions and abort the whole walk, so skip it up front.
SKIP_SUBMODULES = frozenset({"tests", "testing"})


def _should_skip_submodule(subname: str) -> bool:
    """True for test scaffolding that must be excluded from the walk."""
    return subname in SKIP_SUBMODULES or subname.startswith("test_")


def find_edges(mapping, module_name=None):
    discovered = []
    for _, cls in mapping:
        # Only classes defined in this module (not imported into it)
        if not inspect.isclass(cls):
            continue

        if module_name and cls.__module__ != module_name:
            continue

        if not issubclass(cls, Edge) or cls is Edge:
            continue

        # Use the true module path for a stable registration key
        fq_name = f"{cls.__module__}.{cls.__name__}"
        discovered.append((fq_name, cls))

    return discovered


def find_types(mapping, module_name=None):
    """Discover Node subclasses defined in a module.

    Returns a list of (fully_qualified_name, class) tuples for classes
    that inherit from Node but are not part of the base schema module
    (i.e., user-defined types from domain packages).
    """
    discovered = []
    for _, cls in mapping:
        if not inspect.isclass(cls):
            continue

        if module_name and cls.__module__ != module_name:
            continue

        if not issubclass(cls, Node) or cls is Node:
            continue

        # Skip built-in schema types (defined in bigraph_schema itself)
        if cls.__module__.startswith('bigraph_schema.'):
            continue

        fq_name = f"{cls.__module__}.{cls.__name__}"
        discovered.append((fq_name, cls))

    return discovered


def recursive_dynamic_import(
    core,
    module,
    visited: Set[str] | None = None,
    is_package: bool = False,
) -> tuple[object, List[tuple[str, Type[Edge]]], List[tuple[str, type]]]:
    if visited is None:
        visited = set()

    edges = []
    types = []

    if inspect.ismodule(module):
        adjusted = module.__name__
        if adjusted in visited:
            return core, edges, types, visited

        visited.add(adjusted)

    if isinstance(module, str):
        # A dist name can map to MORE THAN ONE import package (a real package
        # plus a back-compat shim shipped in the same distribution). Walk EVERY
        # package for the dist — otherwise an empty shim shadows the real
        # package and its edges/types/visualizations are never discovered.
        #
        # `is_package` guards against the DIST-NAME == IMPORT-PACKAGE-NAME case
        # (e.g. a dist `v2ecoli` whose import package is also `v2ecoli`): without
        # it, recursing on that package name re-enters this dist branch and loops
        # forever (RecursionError). Recursions from the dist loop below (and from
        # the submodule walk) force `is_package=True` so the name is imported as a
        # package, never re-interpreted as a dist.
        if not is_package:
            dist_packages = core.distributions_packages.get(module)
            if dist_packages is not None:
                for package in dist_packages:
                    if package in visited:
                        continue
                    core, pkg_edges, pkg_types, visited = recursive_dynamic_import(
                        core, package, visited=visited, is_package=True)
                    edges.extend(pkg_edges)
                    types.extend(pkg_types)
                return core, edges, types, visited

        # An import-package/module name -> import + walk it.
        adjusted = module
        if adjusted in visited:
            return core, edges, types, visited
        visited.add(adjusted)

        try:
            module = importlib.import_module(adjusted)

        except (KeyboardInterrupt, SystemExit):
            # Never swallow interpreter-control signals: a Ctrl-C or an
            # intentional sys.exit() raised at import time must propagate
            # so discovery can't trap a deliberate interpreter exit.
            raise

        except ImportError as e:
            # Catch both ModuleNotFoundError (the target itself is missing)
            # and ImportError (the target exists but a dep inside it failed
            # to import — typical for optional-extras modules like
            # process_bigraph.protocols.ray when ray isn't installed).
            # Skip and continue scanning the rest of the package; absorbing
            # one missing optional dep should never break the discovery.
            missing = getattr(e, "name", None)
            if missing and missing != adjusted:
                print(f"skipping `{adjusted}` (missing optional dep `{missing}`)")
            else:
                print(f"module `{adjusted}` not found during dynamic import")
            return core, edges, types, visited

        except Exception as e:
            # A submodule can fail at import time with an exception that is
            # NOT an ImportError — e.g. starlette raising RuntimeError when
            # an optional test dependency (httpx) is absent, or an OSError
            # from a module that touches the filesystem at import. One broken
            # or optional submodule must never abort the entire package walk,
            # so log a clear warning and skip it. KeyboardInterrupt/SystemExit
            # are BaseException subclasses and are intentionally not caught
            # here (handled above), so Ctrl-C still works.
            print(
                f"skipping `{adjusted}` "
                f"({type(e).__name__} raised at import: {e})")
            return core, edges, types, visited

    # Allow module to register types into core
    if hasattr(module, "register_types"):
        core = module.register_types(core)

    mapping = inspect.getmembers(module, inspect.isclass)
    edges.extend(find_edges(mapping))
    types.extend(find_types(mapping))

    # Recurse into submodules if this is a package
    if hasattr(module, "__path__"):
        for _, subname, _ in pkgutil.iter_modules(module.__path__):
            # Never descend into test scaffolding (test_*, tests, testing).
            if _should_skip_submodule(subname):
                continue
            submod = f"{adjusted}.{subname}"
            core, sub_edges, sub_types, visited = recursive_dynamic_import(
                core, submod, visited=visited, is_package=True)
            edges.extend(sub_edges)
            types.extend(sub_types)

    return core, edges, types, visited


def is_process_library(dist: importlib.metadata.Distribution) -> bool:
    if dist.metadata["Name"] == "bigraph-schema":
        return True
    reqs = dist.requires or []
    return any("bigraph-schema" in r for r in reqs)


def load_local_modules(core, top=None) -> tuple[
        object,
        List[tuple[str, Type[Edge]]],
        List[tuple[str, type]]]:
    edges = []
    types = []
    visited = set([])

    for dist_name in core.distributions_packages:
        dist = importlib.metadata.distribution(dist_name)
        if not is_process_library(dist):
            continue

        core, found_edges, found_types, visited = recursive_dynamic_import(
            core,
            dist_name,
            visited=visited)

        edges.extend(found_edges)
        types.extend(found_types)

    if top:
        for key, value in top.items():
            if not inspect.isclass(value):
                if key == 'register_types':
                    core = value(core)
                continue

            if issubclass(value, Edge) and value is not Edge:
                fq_name = f"{value.__module__}.{value.__name__}"
                edges.append((fq_name, value))

            elif issubclass(value, Node) and value is not Node:
                if not value.__module__.startswith('bigraph_schema.'):
                    fq_name = f"{value.__module__}.{value.__name__}"
                    types.append((fq_name, value))

    return core, edges, types


def _eager_register(core, edges, types):
    """Register discovered edges/types into ``core`` (the classic eager path).

    Returns ``(edge_index, type_modules)`` describing exactly what was
    registered, so the lazy path can replay it later without importing:

    * ``edge_index`` — ``registry_key -> defining_module`` for every link key
      eager registration created (fully-qualified names, plus short-name
      aliases under the same first-wins rule).
    * ``type_modules`` — ordered, de-duplicated list of modules that must be
      imported to reproduce type state (modules that defined a discovered
      ``Node`` type, followed by modules exposing a ``register_types`` hook).
    """
    edge_index = {}
    # Seed "claimed" short names with whatever is already registered (e.g. the
    # base ``edge`` link) so the short-alias first-wins rule matches eager.
    claimed = set(core.link_registry.keys())

    for fq_name, edge_cls in edges:
        core.register_link(fq_name, edge_cls)
        module = edge_cls.__module__
        edge_index[fq_name] = module

        short = fq_name.split(".")[-1]
        if short not in claimed:
            core.register_link(short, edge_cls)
            claimed.add(short)
            edge_index[short] = module
        claimed.add(fq_name)

    type_modules = []
    seen_type_modules = set()

    for fq_name, type_cls in types:
        short = fq_name.split(".")[-1]
        core.register_type(short, type_cls)
        module = type_cls.__module__
        if module not in seen_type_modules:
            seen_type_modules.add(module)
            type_modules.append(module)

    # Modules exposing a ``register_types`` hook must be re-imported by the
    # lazy path so those (dict-based) types are registered too. Capture them in
    # import order (sys.modules is insertion-ordered), but ONLY for the
    # process-library import packages we actually walked — scanning every dist
    # would false-positive on unrelated ``register_types`` attributes (e.g.
    # torch's ``_OpNamespace``). We also require the hook to be a real function
    # defined in that module, not an arbitrary attribute of that name.
    prefixes = sorted({pkg for _dist, pkg in _process_lib_packages(core)})
    dotted = tuple(p + "." for p in prefixes)
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if mod_name not in prefixes and not mod_name.startswith(dotted):
            continue
        if mod_name in seen_type_modules:
            continue
        if _has_register_types_hook(mod, mod_name):
            seen_type_modules.add(mod_name)
            type_modules.append(mod_name)

    return edge_index, type_modules


def _has_register_types_hook(module, module_name):
    """True only for a genuine ``register_types`` hook defined in ``module``."""
    hook = getattr(module, "register_types", None)
    if not callable(hook):
        return False
    # Must be a plain function (not a class/namespace object) and, when it
    # exposes a ``__module__``, actually belong to this module — this filters
    # out re-exported or foreign callables of the same name.
    hook_module = getattr(hook, "__module__", None)
    return hook_module is None or hook_module == module_name


def _apply_type_modules(core, type_modules):
    """Re-import ``type_modules`` and replay their type registration.

    Reproduces eager ordering EXACTLY. Eager runs every ``register_types``
    hook during the package walk (before any ``Node`` types are registered),
    then registers all discovered ``Node`` subclasses afterwards. So we do the
    same in two passes over ``type_modules`` (which is ordered as the eager
    walk produced): pass 1 imports each module and runs its hook; pass 2
    registers each module's ``Node`` types.
    """
    imported = []
    for module_name in type_modules:
        try:
            module = importlib.import_module(module_name)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001
            print(
                f"lazy discovery: skipping type module `{module_name}` "
                f"({type(exc).__name__} raised at import: {exc})")
            continue
        imported.append((module_name, module))

    # Pass 1: register_types hooks, in order.
    for module_name, module in imported:
        if _has_register_types_hook(module, module_name):
            core = module.register_types(core)

    # Pass 2: Node subclasses, in order.
    for module_name, module in imported:
        for fq_name, type_cls in find_types(
                inspect.getmembers(module, inspect.isclass),
                module_name=module_name):
            core.register_type(fq_name.split(".")[-1], type_cls)
    return core


def _apply_lazy_index(core, index):
    """Install a :class:`LazyLinkRegistry` from a discovery ``index`` dict.

    ``index`` is ``{"edges": {key: module}, "type_modules": [...]}``. Types are
    registered eagerly (they cannot be triggered by address resolution), but
    only the lightweight type-bearing modules get imported — heavy edge-only
    distributions (torch, vEcoli, ...) are deferred until a process address is
    actually resolved.
    """
    from bigraph_schema.package.lazy_registry import LazyLinkRegistry

    core = _apply_type_modules(core, index.get("type_modules", []))
    core.link_registry = LazyLinkRegistry(
        resolved=dict(core.link_registry),
        index=index.get("edges", {}))
    return core


def discover_packages(core, top=None, eager=None):
    """Discover and register every bigraph-schema process package.

    By default (``eager=None``) discovery is LAZY: a cheap ``name -> module``
    index is built (or loaded from an on-disk cache keyed by installed-dist
    versions), process modules are imported only when their address is first
    resolved, and ``core.link_registry`` becomes a
    :class:`~bigraph_schema.package.lazy_registry.LazyLinkRegistry` that still
    enumerates every known name without importing.

    Pass ``eager=True`` (or set ``BIGRAPH_SCHEMA_LAZY_DISCOVERY=0``) to force
    the classic behaviour: import every submodule of every package up front and
    populate a plain-dict ``link_registry``. ``top`` (explicit class dicts) is
    always handled eagerly.
    """
    if eager is None:
        eager = os.environ.get("BIGRAPH_SCHEMA_LAZY_DISCOVERY", "1") == "0"

    # Explicit ``top`` classes and forced-eager both take the classic path.
    if eager or top is not None:
        core, edges, types = load_local_modules(core, top=top)
        _eager_register(core, edges, types)
        return core

    index = _load_or_build_index(core)
    if index is None:
        # Building failed for some reason — fall back to eager so discovery
        # never silently returns an empty registry.
        core, edges, types = load_local_modules(core, top=None)
        _eager_register(core, edges, types)
        return core

    return _apply_lazy_index(core, index)


# ---------------------------------------------------------------------------
# Discovery-index cache (keyed by installed-dist versions + source mtimes)
# ---------------------------------------------------------------------------
#
# The lazy path needs a correct name -> module index. Rather than guess it from
# a fragile static (AST) scan, we derive it from a REAL eager discovery the
# first time — which is exactly correct (names, short-alias winners, ordering,
# and edge-vs-type classification) — and memoize the result to disk keyed by
# the installed process-lib distributions. Subsequent cold starts (every
# subprocess-isolated run, every Ray worker) read the cache and skip importing
# heavy edge-only packages entirely.

_CACHE_VERSION = 1


def _process_lib_packages(core):
    """(dist_name, import_package) pairs for every process-library dist."""
    pairs = []
    for dist_name, packages in core.distributions_packages.items():
        try:
            dist = importlib.metadata.distribution(dist_name)
        except importlib.metadata.PackageNotFoundError:
            continue
        if not is_process_library(dist):
            continue
        for pkg in packages:
            pairs.append((dist_name, pkg))
    return sorted(set(pairs))


def _package_source_signature(pkg):
    """Cheap content signature of a top-level import package WITHOUT importing.

    Uses ``find_spec`` (which locates but does not execute a top-level package)
    and hashes the (relative path, mtime, size) of every ``*.py`` beneath it,
    so editable-install source edits invalidate the cache. Never imports the
    package's code, so no heavy dependency is pulled in here.
    """
    try:
        spec = importlib.util.find_spec(pkg)
    except (ImportError, AttributeError, ValueError):
        return f"{pkg}:nospec"
    if spec is None:
        return f"{pkg}:nospec"

    roots = list(getattr(spec, "submodule_search_locations", None) or [])
    if not roots and spec.origin and spec.origin != "namespace":
        roots = [os.path.dirname(spec.origin)]

    parts = []
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                fp = os.path.join(dirpath, fn)
                try:
                    st = os.stat(fp)
                except OSError:
                    continue
                rel = os.path.relpath(fp, root)
                parts.append(f"{rel}:{int(st.st_mtime_ns)}:{st.st_size}")
    parts.sort()
    return f"{pkg}:" + hashlib.sha1("|".join(parts).encode()).hexdigest()


def _installed_signature(core):
    """A hash string identifying the installed process-lib set + its source."""
    tokens = [f"cachev={_CACHE_VERSION}", f"py={sys.version_info[:2]}"]
    for dist_name, pkg in _process_lib_packages(core):
        try:
            version = importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            version = "?"
        tokens.append(f"{dist_name}=={version}")
        tokens.append(_package_source_signature(pkg))
    return hashlib.sha1("\n".join(tokens).encode()).hexdigest()


def _cache_dir():
    base = os.environ.get("BIGRAPH_SCHEMA_CACHE_DIR")
    if not base:
        base = os.path.join(
            os.path.expanduser("~"), ".cache", "bigraph-schema", "discovery")
    return base


def _cache_path(signature):
    return os.path.join(_cache_dir(), f"index-{signature}.json")


def _read_cache(signature):
    path = _cache_path(signature)
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("signature") != signature:
        return None
    if "edges" not in data or "type_modules" not in data:
        return None
    return data


def _write_cache(signature, index):
    path = _cache_path(signature)
    payload = dict(index)
    payload["signature"] = signature
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Atomic write so concurrent workers never read a half-written file.
        fd, tmp = tempfile.mkstemp(
            dir=os.path.dirname(path), prefix="index-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    except OSError:
        # A read-only or unwritable cache dir must never break discovery.
        pass


def _build_index(core):
    """Run a real eager discovery on a throwaway core and capture the index.

    The passed ``core`` is not mutated: discovery runs on a fresh ``Core`` so
    building the cache has no side effects on the caller's registries.
    """
    from bigraph_schema.core import Core, BASE_TYPES

    probe = Core(BASE_TYPES)
    probe, edges, types = load_local_modules(probe, top=None)
    edge_index, type_modules = _eager_register(probe, edges, types)
    return {"edges": edge_index, "type_modules": type_modules}


def _load_or_build_index(core):
    """Return the discovery index, from disk cache or by building (+caching)."""
    if os.environ.get("BIGRAPH_SCHEMA_DISCOVERY_CACHE", "1") == "0":
        # Cache disabled: build fresh every time (still lazy at resolution).
        try:
            return _build_index(core)
        except Exception:  # noqa: BLE001
            return None

    try:
        signature = _installed_signature(core)
    except Exception:  # noqa: BLE001
        signature = None

    if signature is not None:
        cached = _read_cache(signature)
        if cached is not None:
            return cached

    try:
        index = _build_index(core)
    except Exception:  # noqa: BLE001
        return None

    if signature is not None:
        _write_cache(signature, index)
    return index
