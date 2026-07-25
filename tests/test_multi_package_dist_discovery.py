"""A distribution can ship more than one import package (e.g. a real package
plus a back-compat shim). Discovery must AGGREGATE all packages for a dist —
not last-wins-collapse to a single one — else an empty shim can shadow the real
package and its edges/types/visualizations are never discovered.

Regression for the pbg-superpowers 0.16.0 case: the dist ships both
``viva_superpowers`` (real) and ``pbg_superpowers`` (shim); last-wins could
resolve the dist to the empty shim, so the real package's Demo* viz classes
weren't discovered.
"""
import importlib.metadata

from bigraph_schema import Core


def test_multi_package_dist_aggregates_not_last_wins(monkeypatch):
    # A dist ("multi-dist") that ships TWO import packages.
    fake = {
        "real_pkg": ["multi-dist"],
        "shim_pkg": ["multi-dist"],
        "bigraph_schema": ["bigraph-schema"],
    }
    monkeypatch.setattr(
        importlib.metadata, "packages_distributions", lambda: fake)

    core = Core({})

    packages = core.distributions_packages["multi-dist"]
    assert isinstance(packages, list)
    # BOTH packages are present — not just the last-wins one.
    assert set(packages) == {"real_pkg", "shim_pkg"}


def test_distributions_packages_values_are_lists():
    """Every dist maps to a list of its import packages."""
    core = Core({})
    assert all(isinstance(v, list) for v in core.distributions_packages.values())
    # bigraph-schema is always present and maps to the bigraph_schema package.
    assert "bigraph_schema" in core.distributions_packages.get("bigraph-schema", [])


def test_dist_name_equal_package_name_does_not_recurse(monkeypatch):
    """A dist whose NAME equals its import PACKAGE name (e.g. a dist ``v2ecoli``
    whose import package is also ``v2ecoli``) must not recurse forever.

    Regression: the multi-package walk recursed on each of a dist's packages;
    when a package name equalled the dist name it re-entered the dist branch and
    looped until ``RecursionError`` (~3000 frames). The ``is_package`` guard
    imports such a name as a package instead of re-interpreting it as a dist.
    """
    import importlib.metadata
    from bigraph_schema import Core
    from bigraph_schema.package.discover import recursive_dynamic_import

    fake = {"selfpkg": ["selfpkg"], "bigraph_schema": ["bigraph-schema"]}
    monkeypatch.setattr(
        importlib.metadata, "packages_distributions", lambda: fake)
    core = Core({})

    # Must terminate (the package import simply fails and is skipped),
    # NOT raise RecursionError.
    _core, _edges, _types, visited = recursive_dynamic_import(core, "selfpkg")
    assert "selfpkg" in visited
