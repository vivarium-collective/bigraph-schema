import importlib
import importlib.metadata
import pkgutil
import inspect
from typing import Dict, List, Tuple, Set, Type

from bigraph_schema import Edge
from bigraph_schema.schema import Node


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
        adjusted = core.distributions_packages.get(module, module)
        if adjusted in visited:
            return core, edges, types, visited
        visited.add(adjusted)

        try:
            module = importlib.import_module(adjusted)

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

    # Allow module to register types into core
    if hasattr(module, "register_types"):
        core = module.register_types(core)

    mapping = inspect.getmembers(module, inspect.isclass)
    edges.extend(find_edges(mapping))
    types.extend(find_types(mapping))

    # Recurse into submodules if this is a package
    if hasattr(module, "__path__"):
        for _, subname, _ in pkgutil.iter_modules(module.__path__):
            submod = f"{adjusted}.{subname}"
            core, sub_edges, sub_types, visited = recursive_dynamic_import(
                core, submod, visited=visited)
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


def discover_packages(core, top=None):
    core, edges, types = load_local_modules(core, top=top)

    for fq_name, edge_cls in edges:
        core.register_link(fq_name, edge_cls)

        short = fq_name.split(".")[-1]
        if short not in core.link_registry:
            core.register_link(short, edge_cls)

    for fq_name, type_cls in types:
        short = fq_name.split(".")[-1]
        core.register_type(short, type_cls)

    return core
