# Address portability & process discovery across a multi-repo workspace

Status: **scoping** · Author: investigation from a workbench `no link found`
failure · Companion code: `bigraph_schema/protocols.py`
(`registry_dependent_addresses` / `unresolvable_addresses` /
`assert_portable_addresses`), `bigraph_schema/package/` (discovery).

## Problem

A composite document that realizes cleanly in one core can fail in another with

```
Exception: no link found at address: {'protocol': 'local', 'data': 'ShapeStep'}
```

The trigger is an edge wired by a **bare** `local:<name>` address. Resolution of
that form (`protocols.local_lookup` → `local_lookup_registry`) is
`core.link_registry.get(<name>)` — so it depends on the **resolving core** having
`<name>` registered. When the same document is realized by a *different* core
that lacks the registration, the edge cannot be found.

This bit a real flow: interactive **Apply** resolved a composite in a full core
and worked; a detached **Run** rebuilt it in a thin workspace `build_core` and
failed. The immediate fix (vivarium-collective/v2ecoli#622) rewrote the offending
addresses to the self-describing `local:!module.path.Class` form, which resolves
by import (`local_lookup_module`) and is therefore **core-independent**. That is
the right per-node robustness principle — but applied by hand.

The question this doc scopes: **how do we make process resolution robust and
repo-count-independent by construction**, so a document is portable across cores
and adding a repo to a workspace "just works"?

## What already exists (important)

bigraph-schema already ships a discovery substrate built for exactly the
multi-package case — the "scale to multiple repos" mechanism is largely present:

- **`bigraph_schema/package/discover.py`** — `discover_packages`, `find_edges`,
  `find_types`, `recursive_dynamic_import`, `is_process_library`. Scans installed
  **distributions** for process/type classes.
- **`bigraph_schema/package/lazy_registry.py`** — `LazyLinkRegistry`: resolve a
  process by name **without importing the whole ecosystem**, enumerate every
  known name without importing, plus an on-disk index cache.
- **`Core.distributions_packages`** — maps each distribution to *all* its import
  packages, **aggregating** multi-package dists (a real package + a back-compat
  shim) instead of last-wins-collapsing (regression-tested in
  `tests/test_multi_package_dist_discovery.py`).
- **`bigraph_schema.core.allocate_core(top=None, eager=None)`** — allocates a
  core *with all discovered packages*, **lazy by default** (process modules
  import only when an address is first resolved; toggle with
  `BIGRAPH_SCHEMA_LAZY_DISCOVERY=0` / `eager=True`).

So a core built via `bigraph_schema.allocate_core` can resolve processes across
every installed process-library distribution, lazily. That is the scaling story.

## The gap: adoption + two rough edges

The failure is **not** a missing mechanism — it is that the mechanism is not
uniformly ridden, plus two contract gaps:

1. **Consuming cores bypass discovery.** In the v2ecoli/workspace stack:
   - `v2ecoli.core.allocate_core` populates its registry with a hand-maintained
     list of explicit `register_link(...)` calls (dozens, guarded per optional
     dep) — an **imperative** list that every consuming core must reproduce.
   - the workspace `build_core` used by the **run subprocess** is thinner still
     and shares none of it, so a detached Run resolves in a core that is missing
     most processes.
   A plain `bigraph_schema.Core({})` has a plain-`dict` registry and resolves
   *nothing* by name; only a core built through `allocate_core`'s discovery does.
   The two cores therefore diverge, and documents that lean on registry names are
   not portable between them.

2. **Discovery keys on a process `name`; some processes don't declare one.** The
   node that triggered the failure (`v2ecoli.cell_shape.ShapeStep`) has no `name`
   attribute, so it is not discoverable under `"ShapeStep"` even by a discovering
   core — it was only ever reachable via an explicit `register_link("ShapeStep",
   …)` side-effect in the composite generator. A discovery contract (every
   process library edge declares a stable `name`) is a precondition for
   name-addressing to be reliable.

3. **`local_lookup` has no registry-miss fallback.** On a bare-name miss it
   returns `None` (→ `no link found`) with no attempt to discover. There is no
   bridge from "registry miss" to "ask discovery," and no collision policy if two
   dists export the same name.

## Proposal (three levels, increasing investment)

### Level 1 — portability lint (shipped alongside this doc)

`bigraph_schema/protocols.py` now provides:

- `registry_dependent_addresses(document)` → the bare `local:<name>` edges in a
  document (portability offenders; empty ⇒ realizes in any core).
- `unresolvable_addresses(document, core)` → local edges that do **not** resolve
  in a *given* core — check a document against the exact `build_core` that will
  realize it and get an upfront, path-located error instead of a deep realize
  failure.
- `assert_portable_addresses(document)` → raises, naming each offending path and
  suggesting the `!module.path` form. A drop-in CI / save-time gate.

Recommended adoption: workbench save-time validation and/or the run driver calls
`unresolvable_addresses(doc, build_core)` before spawning a Run; process-library
CIs call `assert_portable_addresses` on their composites. Documents stay
core-independent, and `!module.path` remains the deterministic form authors reach
for.

### Level 2 — discovery-on-miss in the `local` protocol

Extend `local_lookup`: on a `local_lookup_registry` miss, if the core carries a
`LazyLinkRegistry`, ask it to resolve/materialize `<name>` from the discovered
index — **with collision detection**: if two distributions export `<name>`, raise
an ambiguity error naming both and pointing at the `!module.path` disambiguator,
rather than silently last-wins. This keeps short names convenient while removing
the pre-registration dependency. It is bounded work (one resolution path, one
collision policy) but inherits the naming ambiguity that Level 3 also has to
solve.

### Level 3 — every core rides discovery (the structural fix)

Make the consuming core builders build **on** `bigraph_schema.allocate_core`'s
discovery instead of hand-maintained registration:

- `v2ecoli.core.allocate_core` keeps only genuinely dynamic registrations and
  otherwise inherits the discovered set; the workspace `build_core` (and the run
  subprocess) build from the **same** discovering base, so every core resolves
  the same processes. The `build_core` vs `allocate_core` divergence disappears
  at the root, and adding a repo to a workspace is just installing it.
- Pair with the **discovery contract** from gap (2): a process-library edge
  MUST declare a stable `name` to be name-addressable (enforced by a check in
  `find_edges`, or surfaced by `list_processes`), so discovery is complete and
  deterministic. `ShapeStep`-style nameless edges either gain a `name` or are
  only ever referenced by `!module.path`.

Because the lazy substrate (`LazyLinkRegistry`, on-disk index,
name-without-import enumeration) already exists, Level 3 is largely *wiring +
contract*, not new machinery, and it does not regress cold-start import cost
(discovery stays lazy).

## Recommendation

- **Land Level 1 now** (this change) and wire it into the workbench run driver /
  process-library CI — it converts a class of deep, unlocated realize failures
  into upfront, actionable ones and codifies the `!module.path` convention.
- **Target Level 3** for the multi-repo goal: it removes the divergence
  structurally and makes registration declarative-by-discovery, which is what
  actually scales to many repos in a workspace. Gate it on the process-`name`
  discovery contract.
- **Level 2 is optional** — a convenient interim for name-addressing, but it must
  solve the same collision policy Level 3 needs, so prefer investing that design
  effort directly in Level 3.

`!module.path` addresses remain the always-correct, zero-magic escape hatch under
every level.
