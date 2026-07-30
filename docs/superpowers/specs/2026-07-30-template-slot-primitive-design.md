# bigraph-schema `template` / `slot` primitive — design spec

**Status:** Design (Layer 1 of the framework-unification stack)
**Date:** 2026-07-30
**Repo:** bigraph-schema (keystone — every layer above depends on this)
**Umbrella:** `vivarium-workbench/docs/superpowers/specs/2026-07-29-framework-unification-design.md` (PR #676)
**Depends on:** nothing upstream. **Consumed by:** process-bigraph (Study/Investigation composites), vivarium-workbench (templates → study docs), v2ecoli.

---

## 1. Goal

Add a first-class **template**: a schema with **N named, typed slots** (holes).
**Binding** the slots — supplying a filler per slot — produces a **concrete
(ground) composite document**. A template must do **three** things:

1. **Constrain the shape of what fits a slot.** A slot is not "any composite" — it
   declares the **interface** (required inputs/outputs, and optionally an address
   protocol) that a filler process/composite must conform to. This is *structural
   typing over the bigraph*: only a process whose `interface()` matches may fill it.
2. **Parameterize scalar config.** Ordinary value slots (`float`, `string`,
   `map[...]`, …) — the generalization of `CompositeSpec.parameters`' flat `${name}`
   substitution.
3. **Parameterize structure (generative).** Some parameters change the *shape of
   the generated document*, not just leaf values — e.g. the **number of process
   instances** in a store, or the **number of store subtrees**. A cardinality slot
   drives multiplicity of a marked body region, expanding it into N instances on
   bind.

Litmus test (the umbrella's "template study"): a template whose composite slot —
constrained to processes with a given interface — binds a model-under-test into a
standard analysis-flush sub-network, with a cardinality slot for `n_seeds` /
`n_replicas`. Drop any conforming registered composite (`ecoli_baseline`,
`viva_munk.biofilm`, `pbg_copasi.steady-state`) into the slot → a runnable study
document, no code.

---

## 2. What already exists (grounding — do not rebuild)

Verified against the current tree (`bigraph_schema/`):

- **There is no `composite` / `process` / `step` type here.** The bigraph *is*
  the schema dict (dict nesting = place graph; `Link` nodes = link graph). A
  "composite document" is a nested-dict schema+state; **`Link`** (`schema.py:273`)
  is the edge/process/step primitive (`_inputs`/`_outputs` port-type maps +
  `inputs`/`outputs` wires). `Edge` (`edge.py:23`) is its Python side, declaring
  ports via `inputs()`/`outputs()`.
- **Typed holes already exist — the Milner `Site`.** `Site(Empty)`
  (`schema.py:603`) is "a hole in the place graph — an open inner-face position."
  `assembly.compose(outer, inner)` (`assembly.py:185`) **substitutes each `Site`
  in `outer` with a root from `inner`** (`_set_at_path`) and wires matching port
  names; `is_ground(schema)` (`assembly.py:163`) is True when **no Sites remain**.
  `interfaces(schema)` (`assembly.py:59`) enumerates the open sites/names. This is
  a *tested* "typed holes → fill → concrete document" pipeline
  (`tests.py:test_compose_fills_sites`, 1578).
- **Parameter machinery.** `align_parameters` (positional→named,
  `handle_parameters.py:63`) then `reify_schema` (`core.access` each param +
  `core.resolve` into the field, `handle_parameters.py:177`). Generic `Node`
  binding zips a type's `_schema_keys` fields with positional params
  (`handle_parameters.py:162`, `276`). Grammar supports `type[X,Y]`, named
  `type[a:X|b:Y]`, and `type{default}` (`parse.py:35`).
- **Type definition & registry.** A type is a `@dataclass` `Node` subclass whose
  **`_schema_keys` frozenset declares its parameter fields** (`schema.py`, e.g.
  `Map:227`), with `field(default_factory=...)` defaults; behavior is added via
  plum `@dispatch` methods under `methods/` (`default`, `apply`, `realize`, …).
  Register via `core.register_type(name, cls)` (`core.py:188`) or in `BASE_TYPES`
  (`schema.py:690`).
- **Fill / defaults.** `Core.fill(schema, state)` (`core.py:743`) = build defaults
  (`default.py`) then layer state; `Core.realize` (`core.py:599`) decodes/fills;
  `realize_link` (`realize.py:608`) instantiates edges and merges port schemas.
- **Cross-package registration/discovery.** `discover_packages` (`discover.py:228`)
  auto-scans every dist that depends on `bigraph-schema`; a downstream module
  exposing `register_types(core) -> core` is invoked on import
  (`discover.py:158`); `Node`/`Edge` subclasses are found by class scan.

**Consequence:** ~80% of "template" is present. The genuinely new pieces are (a) a
**named, type-constrained slot** (today's `Site` carries only a `_sort`, no name
or fill-type constraint), and (b) a **multi-slot binding operation** that is a thin
layer over `compose` + `reify`/`fill`.

---

## 3. The core design decision

**Binding is composition.** A template is a bigraph with an **inner face** made of
named, typed holes; filling a slot is **composing** the template with a filler
process-bigraph, following bigraph-schema's existing Milner algebra
(`assembly.compose`/`tensor`/`interfaces`/`is_ground`) — *not* a bespoke
`bind_template` side-mechanism. Where the current `compose` is too coarse we
**extend it in a standard-preserving way**, we do not fork it. Concretely:

- Milner `compose(a: I→J, b: J→K) = ab: I→K` (`assembly.compose`, `assembly.py:185`)
  already substitutes `Site`s and requires the shared face `J` to match. **The
  "shape of the process that fits a slot" is precisely that face** — a slot's
  declared interface *is* its local inner face; a filler composes iff its **outer
  face matches** (§4.4). Interface conformance is therefore a *law of composition*,
  not an extra check bolted on.
- The extensions this spec adds — all consistent with the algebra:
  1. **Named / partial composition.** Today's `compose` fills *all* sites
     positionally. Templates need to fill **one named slot at a time** (and leave
     others open), with a per-slot face. `compose_at(template, slot_name, filler)`
     is compose localized to a single named site; repeated application fills the
     rest and reduces to ordinary `compose` when there is one anonymous site.
  2. **Typed faces.** A slot's face carries a required interface
     (`link[inputs, outputs]`); the face-match becomes a `core.check`/`resolve`
     (structural subtyping) instead of name-only matching.
  3. **Parallel replication.** Cardinality reuses `tensor` (the standard parallel
     product, `assembly.py:263`) to place N copies — replication is composition,
     not a new operator.

With that principle fixed, the per-kind dispatch is:

Two candidate mechanisms exist; the umbrella conflated them. This spec picks a
**hybrid keyed on slot kind** — three kinds, matching the three goals in §1 — each
expressed through the composition algebra above:

| Slot kind | Filler | Type constraint | Mechanism |
|---|---|---|---|
| **process / composite** (structural, shape-typed) | a bigraph subtree / registered composite whose `interface()` conforms | an **interface**: required `_inputs`/`_outputs` (`link[in,out]`) + optional address protocol | **`Site` + `assembly.compose`** to substitute the subtree; **interface conformance check** before/at bind |
| **value** (scalar config) | a value or leaf schema | a value type (`float`, `int`, `string`, `map[...]`, `enum[...]`) | **`reify_schema` / `core.resolve`** into the slot position |
| **cardinality** (generative) | an `int` (or small value) that sets a **count** | `int` with optional `range[...]` | **generative expansion**: the count drives multiplicity of a **marked body region**, instantiated N times via `tensor` / `Map` sizing at bind |

- **Shape typing** (kind 1) is the "define the shape of the process that fits in"
  requirement: the slot's `_slot` *is* an interface, and `bind` rejects a filler
  whose `interface()` does not conform (§4.4). bigraph-schema already has the
  vocabulary — `Link._inputs/_outputs`, `Edge.interface()` (`edge.py:144`),
  `Interface` (`schema.py:664`), and `link[<inputs>,<outputs>]` typing.
- **Cardinality** (kind 3) is the "change number of processes / store trees"
  requirement: a value slot that additionally *generates structure* (§4.5).

**Rejected alternatives:** *pure `reify_schema`* can't substitute a whole subtree
into a place-graph position (that's `Site`/`compose`); *pure `Site`/`compose`* is
awkward for a scalar like `n_seeds` and gives no interface typing or generative
expansion. So: **one `Template` type; three slot kinds; `bind` dispatches per
kind.** The new code is the slot metadata + the binding driver; substitution,
resolution, interface checking, and tensor-expansion all reuse existing ops.

---

## 4. The primitives

### 4.1 A slot *is* a sorted `Site` — not a new type

bigraph-schema already models a typed hole: `Site._sort` (`schema.py:623`, a
Milner §6.2 sort label) + the sorting discipline in `assembly.py` (`Sorting`,
`stratified_sorting`, `validate_sorting`, `formation(parent_sort, child_sort)`).
**A slot is a `Site` whose `_sort` names the constraint on its filler** — we do
**not** subclass `Site`. Every existing traversal (`interfaces`/`compose`/
`is_ground`/`validate_sorting`) already handles it.

The constraint per kind is carried by the sort:

- **process slot** — `_sort` names a sort whose **formation is structural
  interface conformance**: a filler is admissible iff its outer face
  (`interface()` / collected `_inputs`,`_outputs`) conforms (§4.4). Sorts may be
  registered by name (`'model'`) or given inline as an interface literal that
  auto-registers an anonymous sort; either way the *mechanism* is the existing
  `Sorting.formation` — we supply a formation that calls `core.check`/`resolve`
  instead of nominal label-equality.
- **value slot** — `_sort` is a value type (`float`/`string`/`map[...]`);
  admissibility is `core.check(sort, value)`.
- **cardinality** — *not a site.* Replication is a region-level annotation in the
  Template header (§4.2), expanded via `tensor`; it is not a hole in the place
  graph, so it does not live on `Site`.

This keeps `Site` pure Milner (a sorted hole) and puts every ergonomic
concern — the slot's **name**, **optionality**, and **cardinality** — in the
Template header, where they belong.

*(Subclassing `Slot(Site)` remains a fallback if we later want the name/optional
flags physically on the hole; the recommendation is to avoid it — a slot is a
sorted site, nothing more.)*

### 4.2 `Template` — a schema with named slots + a body

A template is a schema **document** (an ordinary bigraph) whose place graph
contains sorted `Site`s, plus a small **header** that names them and declares the
non-hole metadata (Milner sites are positional/anonymous — the header is where
concrete names + optionality + cardinality live):

```python
@dataclass(kw_only=True)
class Template(Node):
    _schema_keys = Node._schema_keys | frozenset({'_slots', '_body'})
    _body: Node = field(default_factory=Node)    # the composite schema, with sorted Sites
    _slots: dict = field(default_factory=dict)   # slot_name -> {
                                                 #   'path': [...],          # the Site's position
                                                 #   'sort': <sort or interface literal>,
                                                 #   'optional': bool,
                                                 #   'cardinality': {'target':[...], 'range':[lo,hi]}?  }
```

- `_slots` maps a **name → the Site's path** (+ its sort/optionality, and a
  `cardinality` entry when the name drives a replicated region rather than fills a
  hole). Names/paths are how `bind` addresses a specific slot; the sort is the
  admissibility constraint validated by the `Sorting`.
- `_slots` is recoverable from `_body` (walk it, collect sorted `Site`s by path);
  the header is the authoritative *naming* layer + the home for cardinality (which
  has no Site to attach to). Single-composite study template = one process slot;
  comparison template (model-under-test vs. reference) = two.
- A `Template` registers like any type (`core.register_type('template', Template)`
  in `BASE_TYPES`); downstream templates register via the `register_types`
  discovery hook — so `study`, `investigation`, and concrete templates all become
  bigraph-schema types (the umbrella's "bigraph-schema as the registry").

### 4.3 `compose_at` (the primitive) and `bind` (the sugar)

The one genuinely new **primitive** is *named, typed, partial* composition — an
extension of `assembly.compose`. `bind` is convenience sugar that folds a whole
`{slot: filler}` map through it.

```python
def compose_at(core, template, slot_name, filler):
    """Compose one named slot. Extension of assembly.compose localized to a site.
    Precondition: outer_face(filler) matches inner_face(template @ slot_name)  (§4.4)
      process slot     → substitute filler subtree at the slot's path (compose)
                         + wire the faces (existing port-name wiring, assembly.py:185)
      value  slot      → core.access filler → core.resolve into the slot position
      cardinality slot → tensor N copies of the _target region (§4.5)
    Returns a template with that slot closed (others still open).
    """

def bind(core, template, bindings: dict) -> (schema, state):
    """Fold compose_at over bindings → a ground (site-free) composite document.
    Order: cardinality FIRST (replication grows the slot set), then process, then
    value; re-collect slots after each cardinality step. Missing binding → _default
    if _optional else raise. Then core.fill(result, {}); assert is_ground (modulo
    optional slots). Registered as core methods → core.compose_at / core.bind."""
```

- When a template has a single anonymous site and the filler matches the whole
  face, `compose_at` **is** `assembly.compose` — the extension degrades to the
  existing operation (no divergence from the standard).
- `compose_at` reuses the existing substitution (`_set_at_path`) and port-name
  wiring; the *only* additions are (a) selecting one **named** site and (b) the
  **typed** face-match (§4.4).
- Because each step is a composition, a partially-bound template is **still a valid
  template** (a bigraph with a smaller inner face) — templates compose with
  templates, and binding can be incremental / streamed.

### 4.4 Interface conformance = face matching (the composition law)

In Milner composition `a: I→J` ∘ `b: J→K` is defined **iff** the shared face `J`
matches. A slot's declared interface *is* the template's local **inner face** at
that site; the filler's **outer face** is `filler.interface()` (`edge.py:144`) or
the collected `_inputs`/`_outputs` of the filler subtree. `compose_at` is defined
iff:

- every port in the slot's face is present in the filler's outer face with a
  **compatible type** — `core.check` / `core.resolve` succeeds (structural
  subtyping: the filler may over-provide, never under-provide);
- if the slot pins an address protocol, the filler's `address` matches.

So "a process shaped `{glucose: float, biomass: map[float]} → {growth_rate: float}`"
is a *face*, and any registered process whose outer face conforms composes into it.

Mechanically this is the **existing sorting discipline** (`assembly.Sorting` /
`validate_sorting`, `assembly.py:401-491`) with a `formation` that decides
admissibility by **structural face-conformance** (`core.check`/`resolve`) rather
than nominal label-equality. We add no new type algebra — only a formation callback
that compares faces (not just port *names*). A non-conforming filler is a
**composition error** (equivalently, a sorting violation), reported as such.

### 4.5 Generative cardinality (number of processes / store trees)

A `cardinality` slot carries an int `n` and a `_target` path into the body marking
a **repeatable region** (a subtree — a process, or a store subtree). On bind, the
region is **instantiated `n` times** with indexed names:

- **Homogeneous, keyed by name → `Map`.** The natural encoding of "N processes in a
  store" is a `map[<region-schema>]` whose keys are the N instance ids; expansion
  sets the map's contents to `{f'{name}_{i}': region for i in range(n)}`. This is
  ordinary state, no new type.
- **Positional / parallel → `tensor`.** When the copies are structurally parallel
  (side-by-side subtrees, not a keyed store), `assembly.tensor` (`assembly.py:263`)
  composes `n` disjoint copies.
- **Numeric fan-out → `array` `_shape`.** A cardinality slot may instead set an
  `Array`'s `_shape` (e.g. an `[n_replicas]` axis), reusing `array[...]` sizing
  (`handle_parameters.py:89`).

Expansion happens **before** process/value binding so a per-instance `process` or
`value` slot inside the region is materialized `n` times and can be bound
per-instance (or bound once and broadcast). Cardinality is the one kind that
*generates* structure; keep its expansion a pure function of `(region, n)` so
binding stays deterministic and re-runnable.

---

## 5. Key contracts (must be nailed)

1. **Template document.** A registered schema with ≥1 `Slot`. Round-trips through
   `render`/`access`. Serializable to `*.template.{yaml,json}` (the on-disk form
   the workbench authors).
2. **Slot type constraint (per kind).** `process` → an interface the filler's
   `interface()` must **conform** to (§4.4); `value` → a value schema satisfying
   `core.check(_slot, value)`; `cardinality` → an int (in `range` if given) with a
   valid `_target` region path. `bind` validates each filler against its slot's
   constraint and raises with a clear message on mismatch (shape mismatch, mistyped
   value, out-of-range count).
3. **Interface conformance is structural subtyping.** A filler may over-provide
   ports/state; it must not under-provide any required port, and each shared port's
   type must `resolve`. No exact-match requirement — this is what makes slots
   reusable across processes of the same shape.
4. **Generative expansion is a pure function of `(region, n)`.** Deterministic and
   re-runnable; the same `n` always yields the same indexed structure. Cardinality
   binding commits before process/value binding.
5. **Binding result = ground composite.** `bind` returns `(schema, state)` with
   **no remaining required Slots** (`is_ground` modulo optional slots) — exactly
   what process-bigraph's `Composite` consumes (the Layer-2 handoff).
6. **Composition consistency.** `compose_at` obeys the algebra: (a) it degrades to
   `assembly.compose` for a single anonymous, whole-face site; (b) binding two
   independent named slots **commutes** (order-independent) — order only matters
   when one binding is inside a region a *cardinality* slot replicates, which is
   why cardinality binds first; (c) a partially-bound template is itself a valid
   template with a reduced inner face. These must hold as tests so the primitive
   stays a true extension of composition, not a fork.
4. **`CompositeSpec.parameters` subsumption.** A `CompositeSpec` with flat
   `${name}` params is the **all-scalar-slots, single-body** special case of a
   template. Layer 2 (process-bigraph) decides whether `CompositeSpec` becomes a
   thin adapter that lowers to a `Template`, or is refactored onto it — **this spec
   guarantees the template is a superset** (structural slots are the added power),
   so no `CompositeSpec` capability is lost.

---

## 6. Tests

Extend `tests.py` (the repo's single suite) alongside the existing golden cases:

- **Structural fill** — mirror `test_compose_fills_sites` (1578): a `Template` with
  one `process` slot; `bind` a conforming registered subtree; assert `is_ground`
  and that the filler's `Link` ports wired.
- **Shape conformance** — a `process` slot with interface
  `link[{glucose:float},{growth_rate:float}]`; bind a matching process (passes,
  incl. an over-providing one) and a non-matching one (raises with a shape-mismatch
  message).
- **Cardinality / generative** — a template with `n_replicas: cardinality` over a
  `_target` region; `bind n=3` yields three indexed instances (as a `map` and, in a
  second case, via `tensor`); `bind n=1` yields one; assert determinism (same `n` →
  identical structure).
- **Multi-slot** — a 2-slot comparison template (model-under-test + reference);
  bind both; assert ground and both subtrees present at their paths.
- **Value slot** — a `float`/`int` slot bound to a value; assert it lands and
  `core.check` rejects a mistyped filler; assert an `_optional` slot falls back to
  `_default`.
- **Round-trip** — extend `test_render`/`do_round_trip` (431, 123) so a template
  schema survives `access → render → access`.
- **Registration/discovery** — a downstream stub module exposing `register_types`
  that registers a template; assert `discover_packages` picks it up (extend
  `tests/test_multi_package_dist_discovery.py`).
- **Negative** — missing required binding raises; structural filler into a scalar
  slot (and vice-versa) raises with a clear message.

Golden references to stay consistent with: `test_uni_schema` (451) for the
parameter grammar, `test_compose_*` (1578-1599) for substitution semantics.

---

## 7. Migration / compatibility

- **Additive.** New `Slot`/`Template` types + a `bind` method; no existing type,
  method, or test changes behavior. `BASE_TYPES` gains `slot`, `template`,
  `composite` (alias).
- No on-disk migration in *this* repo — `*.template.{yaml,json}` authoring and the
  study/investigation migrator live in the workbench (Layer 4).
- Minimum-version note for downstreams: process-bigraph's Study spec (Layer 2) must
  pin the bigraph-schema version that introduces `bind`.

---

## 8. Risks & things to watch

- **Two-mechanism seam.** Keep the structural/scalar split *inside* `bind`; callers
  see one `bind(template, bindings)`. If a slot ever needs both (a subtree
  *parameterized* by scalars), bind the scalars into the subtree first, then compose
  — do not entangle the two paths.
- **`Site` semantics.** `Slot` must remain a faithful `Site` so `interfaces`/
  `compose`/`is_ground` keep working unmodified; add metadata, don't change hole
  behavior. Guard with the existing Milner tests.
- **`composite` is a marker, not a type.** Resist adding a real `composite` runtime
  type here (that belongs to process-bigraph). Over-typing would fork the
  "bigraph = the dict" invariant (`assembly.py:5`).
- **Generative expansion order & naming.** Cardinality binds first and must produce
  **stable, collision-free indexed names** (`{name}_{i}`); a per-instance slot
  inside an expanded region multiplies the slot set, so `bind` must re-collect
  slots after each cardinality expansion. Keep expansion a pure `(region, n)`
  function — no dependence on bind order among cardinality slots — or nested
  cardinalities (a store of N processes each with M sub-stores) become
  non-deterministic. Cap `n` (guard against a param that explodes the document).
- **Subsumption proof for `CompositeSpec`.** Layer 2 must show every current
  `CompositeSpec.parameters` use lowers to scalar slots; a golden corpus from
  process-bigraph should ride along when that layer lands.

---

## 9. Out of scope (YAGNI)

- A runtime `composite`/`process`/`step` type (structural; belongs downstream).
- Reaction-rule / rewrite semantics for templates (the `ReactionRule` engine
  exists at `assembly.py:736` if ever needed — not now).
- The study/investigation *documents* themselves and their migrator (Layers 2–4).
- Any workbench UI (the ProcessCard viewer is the presentation side, separate).

---

## 10. Decisions (resolved 2026-07-30)

Milner grounding: in bigraph theory the holes are **sites** (numbered, untyped);
composition plugs a filler's **roots into the context's sites** and is defined iff
the shared **face** matches; constraining what may fill a hole is a **sorting**.
bigraph-schema's `Site` is exactly Milner's site. Our `Slot` therefore = a
**concrete (named) site under a sorting discipline (a typed face)** — a recognized
extension, not a departure. This validates decisions 1 & 4 below.

1. **A slot *is* a sorted `Site` — no subclass** ✅ — `Site._sort` + the existing
   `Sorting`/`validate_sorting` discipline already model a typed hole. A slot is a
   `Site` whose sort's `formation` is structural interface conformance. The slot's
   **name, optionality, and cardinality** live in the **Template header** (Milner
   sites are anonymous/positional), keeping `Site` pure. (Subclassing `Slot(Site)`
   is a fallback only if we later want name/optional physically on the hole.)
2. **`composite` stays a marker here; the composite concept is hardened in
   process-bigraph** ✅ — bigraph-schema keeps no runtime `composite` type (no
   upstream dep). The first-class `Composite` / `generate_composite` lives and is
   hardened in **process-bigraph** (Layer 2), which references this template
   machinery — including migrating `viva_superpowers`'s composite-generator into
   process-bigraph.
3. **`bind` returns `(schema, state)`** ✅ — a directly runnable composite for
   Layer 2; matches `Core.default`/`realize`.
4. **Extend `assembly.compose` in place** ✅ — re-express `compose` as `compose_at`
   over the single anonymous site (one code path, old signature preserved); no
   parallel wrapper that could drift from the standard.
