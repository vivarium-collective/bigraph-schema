"""Tests for lazy process discovery.

Covers the LazyLinkRegistry mapping contract, the "resolve a process without
importing the whole ecosystem" guarantee, name enumeration without imports,
eager-mode parity, and the on-disk index cache. See
``bigraph_schema/package/lazy_registry.py`` and
``bigraph_schema/package/discover.py``.
"""

import os
import sys
import time
import textwrap

import pytest

import bigraph_schema.core as core_module
from bigraph_schema.core import allocate_core
from bigraph_schema.package.lazy_registry import LazyLinkRegistry
from bigraph_schema.package import discover as D
# Contract: these must remain importable for downstream eager consumers.
from bigraph_schema.package.discover import (  # noqa: F401
    recursive_dynamic_import,
    find_edges,
    find_types,
    discover_packages,
)


# ---------------------------------------------------------------------------
# A real, importable throwaway package whose import is observable
# ---------------------------------------------------------------------------

@pytest.fixture
def lazy_pkg(tmp_path, monkeypatch):
    """Create an importable module with an observable import side effect.

    Returns ``(module_name, marker_path)``. The module writes ``marker_path``
    when it is imported, so a test can assert import did / did not happen.
    """
    marker = tmp_path / "IMPORTED"
    name = "bgs_lazy_probe_pkg"
    src = tmp_path / f"{name}.py"
    src.write_text(textwrap.dedent(f"""
        import pathlib
        pathlib.Path(r"{marker}").write_text("x")

        from bigraph_schema import Edge

        class ProbeProcess(Edge):
            pass
    """))
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(name, None)
    yield name, marker
    sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# LazyLinkRegistry unit behaviour
# ---------------------------------------------------------------------------

class TestLazyLinkRegistry:

    def test_enumeration_does_not_import(self, lazy_pkg):
        name, marker = lazy_pkg
        reg = LazyLinkRegistry(
            resolved={"edge": object},
            index={"ProbeProcess": name, f"{name}.ProbeProcess": name})

        # Names, membership, len — all without importing.
        assert "ProbeProcess" in reg
        assert "edge" in reg
        assert "ProbeProcess" in reg.keys()
        assert set(reg.names()) == {"edge", "ProbeProcess", f"{name}.ProbeProcess"}
        assert len(reg) == 3
        assert not marker.exists(), "enumeration must not import the module"
        assert name not in sys.modules

    def test_get_imports_only_on_resolve(self, lazy_pkg):
        name, marker = lazy_pkg
        reg = LazyLinkRegistry(index={"ProbeProcess": name})
        assert not marker.exists()

        cls = reg.get("ProbeProcess")
        assert cls is not None and cls.__name__ == "ProbeProcess"
        assert marker.exists(), "get() must import the backing module"
        # Second lookup is served from the resolved cache.
        assert reg.get("ProbeProcess") is cls

    def test_get_missing_returns_default(self, lazy_pkg):
        name, _ = lazy_pkg
        reg = LazyLinkRegistry(index={"ProbeProcess": name})
        assert reg.get("does_not_exist") is None
        assert reg.get("does_not_exist", 42) == 42
        with pytest.raises(KeyError):
            reg["does_not_exist"]

    def test_items_materializes(self, lazy_pkg):
        name, marker = lazy_pkg
        reg = LazyLinkRegistry(index={"ProbeProcess": name})
        items = dict(reg.items())
        assert items["ProbeProcess"].__name__ == "ProbeProcess"
        assert marker.exists()

    def test_copy_preserves_placeholders(self, lazy_pkg):
        name, marker = lazy_pkg
        reg = LazyLinkRegistry(index={"ProbeProcess": name})
        clone = reg.copy()
        assert "ProbeProcess" in clone
        assert not marker.exists(), "copy() must not materialize"
        # Resolving on the clone doesn't require the original.
        assert clone.get("ProbeProcess").__name__ == "ProbeProcess"

    def test_setitem_supersedes_placeholder(self, lazy_pkg):
        name, marker = lazy_pkg
        reg = LazyLinkRegistry(index={"ProbeProcess": name})
        sentinel = object()
        reg["ProbeProcess"] = sentinel
        assert reg.get("ProbeProcess") is sentinel
        assert not marker.exists(), "explicit set must not import"


# ---------------------------------------------------------------------------
# allocate_core: lazy default, eager opt-in, parity
# ---------------------------------------------------------------------------

def _reset_cached_cores():
    core_module._cached_base_core = None
    core_module._cached_base_core_eager = None


class TestAllocateCore:

    def test_default_is_lazy_registry(self):
        _reset_cached_cores()
        core = allocate_core()
        assert isinstance(core.link_registry, LazyLinkRegistry)

    def test_eager_is_plain_dict(self):
        _reset_cached_cores()
        core = allocate_core(eager=True)
        assert type(core.link_registry) is dict

    def test_eager_lazy_name_parity(self):
        """Lazy and eager discovery must expose the same link + type names."""
        _reset_cached_cores()
        eager = allocate_core(eager=True)
        _reset_cached_cores()
        lazy = allocate_core()

        assert set(eager.link_registry.keys()) == set(lazy.link_registry.keys())
        assert set(eager.registry.keys()) == set(lazy.registry.keys())

    def test_base_edge_link_present(self):
        _reset_cached_cores()
        core = allocate_core()
        assert "edge" in core.link_registry

    def test_list_processes_enumerates_without_import(self, lazy_pkg):
        """Core.list_processes returns names for a lazy registry."""
        name, marker = lazy_pkg
        _reset_cached_cores()
        core = allocate_core()
        # Inject a placeholder and confirm it is enumerated but not imported.
        core.link_registry._index["ProbeProcess"] = name
        assert "ProbeProcess" in core.list_processes()
        assert not marker.exists()

    def test_isolated_copy_does_not_materialize(self, lazy_pkg):
        name, marker = lazy_pkg
        _reset_cached_cores()
        allocate_core()  # populates the cached base core
        # Inject a placeholder into the CACHED base; every subsequent
        # allocate_core() returns an isolated copy of it.
        core_module._cached_base_core.link_registry._index["ProbeProcess"] = name
        clone = allocate_core()
        assert "ProbeProcess" in clone.link_registry
        assert not marker.exists(), "copying a cached base must not import"


# ---------------------------------------------------------------------------
# Cold-start improvement (coarse, deterministic)
# ---------------------------------------------------------------------------

def test_lazy_cold_start_not_slower_than_eager():
    """Lazy allocate_core() must not be materially slower than eager.

    On any real install with heavy edge-only packages lazy is dramatically
    faster; on a minimal install (bigraph-schema alone) they are comparable.
    This asserts the guard direction with generous slack so it is not flaky.
    """
    _reset_cached_cores()
    t = time.perf_counter()
    allocate_core(eager=True)
    eager_dt = time.perf_counter() - t

    _reset_cached_cores()
    allocate_core()  # warm the disk cache
    _reset_cached_cores()
    t = time.perf_counter()
    allocate_core()
    lazy_dt = time.perf_counter() - t

    assert lazy_dt <= eager_dt + 0.5, (
        f"lazy cold start {lazy_dt:.3f}s slower than eager {eager_dt:.3f}s")


# ---------------------------------------------------------------------------
# Disk index cache
# ---------------------------------------------------------------------------

def test_index_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("BIGRAPH_SCHEMA_CACHE_DIR", str(tmp_path))
    _reset_cached_cores()
    core = allocate_core()  # writes cache
    files = list(tmp_path.glob("index-*.json"))
    assert files, "lazy discovery should have written an index cache file"

    # A second cold allocate should reuse it (still a lazy registry, same names).
    _reset_cached_cores()
    core2 = allocate_core()
    assert set(core.link_registry.keys()) == set(core2.link_registry.keys())
