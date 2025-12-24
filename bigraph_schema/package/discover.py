import importlib
import importlib.metadata
import pkgutil
import inspect
from typing import Dict, List, Tuple, Set, Type

from bigraph_schema import Edge


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


def recursive_dynamic_import(
    core,
    module,
    visited: Set[str] | None = None,
) -> tuple[object, List[tuple[str, Type[Edge]]]]:
    if visited is None:
        visited = set()

    if inspect.ismodule(module):
        adjusted = module.__name__
        if adjusted in visited:
            return core, [], visited

        visited.add(adjusted)

    if isinstance(module, str):
        adjusted = core.distributions_packages.get(module, module)
        if adjusted in visited:
            return core, [], visited
        visited.add(adjusted)

        try:
            module = importlib.import_module(adjusted)

        except ModuleNotFoundError as e:
            # e.name is the missing module name
            # If the missing name IS the module we tried to import, then it's truly not found.
            # Otherwise, it's a dependency import failure inside that module.
            print(f"module `{adjusted}` not found during dynamic import")

    # Allow module to register types into core
    if hasattr(module, "register_types"):
        core = module.register_types(core)

    mapping = inspect.getmembers(module, inspect.isclass)
    discovered = find_edges(mapping)

    # Recurse into submodules if this is a package
    if hasattr(module, "__path__"):
        for _, subname, _ in pkgutil.iter_modules(module.__path__):
            submod = f"{adjusted}.{subname}"
            core, sub_discovered, visited = recursive_dynamic_import(
                core, submod, visited=visited)
            discovered.extend(sub_discovered)

    return core, discovered, visited


def is_process_library(dist: importlib.metadata.Distribution) -> bool:
    if dist.metadata["Name"] == "bigraph-schema":
        return True
    reqs = dist.requires or []
    return any("bigraph-schema" in r for r in reqs)


def load_local_modules(core, top=None) -> tuple[object, List[tuple[str, Type[Edge]]]]:
    processes = []
    visited = set([])

    for dist_name in core.distributions_packages:
        dist = importlib.metadata.distribution(dist_name)
        if not is_process_library(dist):
            continue

        core, found, visited = recursive_dynamic_import(
            core,
            dist_name,
            visited=visited)

        processes.extend(found)

    if top:
        for key, value in top.items():
            if inspect.isclass(value) and issubclass(value, Edge):
                processes.append((key, value))

            if key == 'register_types':
                core = value(core)

    return core, processes


def discover_packages(core, top=None):
    core, discovered = load_local_modules(core, top=top)

    for fq_name, edge_cls in discovered:
        core.register_link(fq_name, edge_cls)

        short = fq_name.split(".")[-1]
        if short not in core.link_registry:
            core.register_link(short, edge_cls)

    return core
