import importlib
import importlib.metadata
import pkgutil
import inspect
from typing import Dict, List, Tuple, Set, Type

from bigraph_schema import Edge


def recursive_dynamic_import(
    core,
    package_name: str,
    *,
    visited: Set[str] | None = None,
) -> tuple[object, List[tuple[str, Type[Edge]]]]:
    if visited is None:
        visited = set()

    adjusted = package_name.replace("-", "_")

    if adjusted in visited:
        return core, []
    visited.add(adjusted)

    try:
        module = importlib.import_module(adjusted)
    except ModuleNotFoundError as e:
        # e.name is the missing module name
        # If the missing name IS the module we tried to import, then it's truly not found.
        # Otherwise, it's a dependency import failure inside that module.
        if getattr(e, "name", None) == adjusted:
            raise ModuleNotFoundError(
                f"Error: module `{adjusted}` not found when trying to dynamically import!"
            ) from e
        raise  # preserve the real traceback/cause

    # Allow module to register types into core
    if hasattr(module, "register_types"):
        core = module.register_types(core)

    discovered: List[tuple[str, Type[Edge]]] = []

    for _, cls in inspect.getmembers(module, inspect.isclass):
        # Only classes defined in this module (not imported into it)
        if cls.__module__ != module.__name__:
            continue
        if not issubclass(cls, Edge) or cls is Edge:
            continue

        # Use the true module path for a stable registration key
        fq_name = f"{cls.__module__}.{cls.__name__}"
        discovered.append((fq_name, cls))

    # Recurse into submodules if this is a package
    if hasattr(module, "__path__"):
        for _, subname, _ in pkgutil.iter_modules(module.__path__):
            submod = f"{adjusted}.{subname}"
            core, sub_discovered = recursive_dynamic_import(core, submod, visited=visited)
            discovered.extend(sub_discovered)

    return core, discovered


def is_process_library(dist: importlib.metadata.Distribution) -> bool:
    if dist.metadata["Name"] == "bigraph-schema":
        return True
    reqs = dist.requires or []
    return any("bigraph-schema" in r for r in reqs)


def load_local_modules(core) -> tuple[object, List[tuple[str, Type[Edge]]]]:
    processes: List[tuple[str, Type[Edge]]] = []
    for dist in importlib.metadata.distributions():
        if not is_process_library(dist):
            continue
        core, found = recursive_dynamic_import(core, dist.metadata["Name"])
        processes.extend(found)
    return core, processes


def discover_packages(core):
    core, discovered = load_local_modules(core)

    for fq_name, edge_cls in discovered:
        core.register_link(fq_name, edge_cls)

        short = fq_name.split(".")[-1]
        core.register_link(short, edge_cls)

    return core
