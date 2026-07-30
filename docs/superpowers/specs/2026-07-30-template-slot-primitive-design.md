# bigraph-schema: sites, `fill`, and groundness — the template primitive

**Status:** Design (Layer 1 of the framework-unification stack) · revised after Fable architecture review
**Date:** 2026-07-30
**Repo:** bigraph-schema (keystone — every layer above depends on this)
**Umbrella:** `vivarium-workbench/docs/superpowers/specs/2026-07-29-framework-unification-design.md` (PR #676)
**Review:** `~/AI-Generated/2026-07-30-architecture-unification-review-fable.md`
**Depends on:** nothing upstream. **Consumed by:** process-bigraph, vivarium-workbench, v2ecoli.

---

## 1. Goal — one operation, one law

> There is one object: a **bigraph** — a typed document whose place graph is dict
> nesting, whose link graph is `Link` nodes with faces and wires, and whose holes
> are sorted **sites**. There is one operation: **`fill`** — substitute fillers
> into named open sites. There is one law: a document is runnable exactly when it
> is **ground** (no open sites, `is_ground`).

A **template is not a type**; it is simply a document that is **not yet ground**.
Filling its sites produces a ground document that process-bigraph's `Composite`
runs. This replaces the earlier framing (a new `Template`/`Slot` type + a bespoke
`bind`) — the machinery already exists (§2); we add one predicate-checked wrapper.

Templates still do three jobs, all as *sorts on sites*, all filled by the same
`fill`:

1. **Shape-constrain a filler** — a site's sort is a **face** (`link[in, out]`); a
   filler is admissible iff its outer face conforms (§4.4). "The shape of the
   process that fits" is Milner's face-matching, made typed.
2. **Value config** — a site whose sort is a value type (`float`/`string`/`map`).
   Generalizes `CompositeSpec`'s `${name}` — each `${name}` *is* a site (Layer 2a).
3. **Generative structure** — "number of processes / store subtrees" as a
   **reaction** that replicates a marked region *before* filling (§4.5), reusing
   the existing `ReactionRule`/`fire_rule`, not a new operator.

Litmus test: a template whose model-site is face-constrained; drop any conforming
registered composite (`ecoli_baseline`, `viva_munk.biofilm`,
`pbg_copasi.steady-state`) into it → a runnable study document, no code.

Vocabulary is the shared glossary (umbrella §1 / review §3.4): **site**, **sort**,
**admits**, **formation**, **face**, **wires**, **fill**, **compose**, **ground**,
**template**, **document**, **edge**, **reaction**. This doc says *site* not slot,
*fill* not bind/reify/substitute.

---

## 2. What already exists (verified — do not rebuild)

Verified against the tree; **four earlier claims were wrong and are corrected here.**

- **Sites are already named, sorted holes.** `Site(Empty)` with `_sort`
  (`schema.py:604,623`); `interfaces` / `is_ground` / `compose` / `tensor`
  (`assembly.py:59,163,185,263`). A site's **name is its key in the place graph** —
  `instantiate`/`Match` already address sites by dict key (below). So there is no
  need for a `_slots` header or a `Slot` subclass.
- **`fill` already exists as `instantiate`.** `assembly.instantiate(reactum,
  bindings, instantiation)` (`assembly.py:1044`) performs **named-site
  substitution**: sites addressed by key, multi-root fillers spliced as forests,
  single trees nested, everything deep-copied; it is tested through `fire_rule`.
  This is the substitution half of `fill`.
- **`is_ground`** (`assembly.py:163`) is the runnable predicate — no open sites.
- **Sorting is a *nesting* discipline, not a filling one (correction).**
  `Sorting.formation(parent_sort, child_sort)` (`assembly.py:376,444`,
  `validate_sorting`) asks whether a child sort may live *inside* a parent sort;
  it is **never consulted at a site**, and after `_set_at_path` the site's `_sort`
  is gone. So admissibility needs a **separate** relation (`admits`, §4.4), not
  `formation`.
- **`Core.bind` already exists** (`core.py:732`): `bind(self, schema, state,
  raw_key, target)` — a logical-key jump. Our operation must **not** shadow it;
  register as **`core.fill_sites`** (a.k.a. `core.instantiate`).
- **The `compose` link-branch is untested and probably wrong (correction).**
  `test_compose_fills_sites` (`tests.py:1578`) composes `merge(2)` with two
  `barren()` roots — **empty regions into empty holes**; no `Link`, port, or wire.
  The link-composition branch (`assembly.py:245-256`) is unexercised and wires the
  outer link to a path *inside the inner `Link` node* (`list(inner_link_path) +
  [port_name]`) rather than at a store — dangling for any real document.
  **Everything downstream sits on this branch → Task 0 (§6) must green it first.**
- **Faces & edges.** `Link` (`schema.py:273`) has `address`, `config`,
  `_inputs`/`_outputs` (faces = port *types*), `inputs`/`outputs` (wires).
  `Edge.interface()` (`edge.py:144`) is the face's type map; a `Composite`'s
  `bridge` is its outer face (Layer 2a states this once).
- **Registration/discovery** — `core.register_type` (`core.py:188`) and the
  `register_types(core)` discovery hook (`package/discover.py:158`). We need
  **almost none** of this: `template` is a property (`not is_ground`), not a type.

---

## 3. The design — collapse five names to one

**`fill` is the only substitution primitive.** Everything else is a call to it:

```
fill(core, body, bindings) -> body'        # instantiate (assembly.py:1044) + per-site admits check
compose(outer, inner)  =  fill(outer, positional_bindings(outer, inner))   # Milner ∘, a 2-line adapter
```

- **No `compose_at`.** "Fill one named site" is `fill(body, {name: filler})`;
  filling all is the same call. There is no localized variant to keep in sync.
- **No new `bind` name** (avoids the `Core.bind` collision) — register
  `core.fill_sites`. A high-level `template + overrides` convenience may wrap it
  (§4.3), but it is sugar, not a primitive.
- **One site kind.** A `${name}` scalar and a model subtree differ *only in the
  sort of the site they fill* — value vs. face. There are not three "slot kinds";
  there is one sorted site and a per-sort `admits`.
- **`compose` becomes a two-line adapter over `fill`** — the "no fork from the
  standard" goal, achieved by *deletion*, not by rewriting `compose`.

This is strictly less code than the previous draft and removes: `compose_at`, the
`Slot` type, the `composite` slot-type, `Template._slots`, and three "slot kinds".

---

## 4. The primitives

### 4.1 A site is a sorted `Site`; its name is its key

No new type, no subclass. A typed hole is a `Site` whose `_sort` constrains its
filler; its identity is its **key in the place graph** (how `instantiate` and
`Match.bindings` already address sites). `optional`/`default` live **on the site**
(`Site(_sort=…, _default=…)` — `Site` inherits `Empty`; a default is *state*, not
header ceremony). Walking `_body` recovers all sites by key — there is no header.

### 4.2 A template is a non-ground document

`template` is **not** a registered type and adds **nothing** to `BASE_TYPES`. It is
the *property* `not is_ground(document)`. A study is a ground document; an
investigation is a document with one site per dependent study (filled at runtime by
gate edges — umbrella §4). This is the most Milner-faithful and least-code form.

### 4.3 `fill` (primitive) and the template convenience

```python
def fill_sites(core, body, bindings):        # core.fill_sites / core.instantiate
    """Substitute fillers into named open sites (assembly.instantiate), checking
    admits(core, site, filler) per site BEFORE substitution. Returns body'."""

def build(core, template, overrides) -> (schema, state):   # convenience, NOT a primitive
    """fill_sites(template_body, overrides) → core.fill(defaults) → assert is_ground
    (modulo optional sites). Returns a runnable (schema, state)."""
```

- `build` returns **`(schema, state)`** — a directly runnable composite for Layer 2,
  matching `Core.default`/`realize`.
- A partially-filled template is **still a template** (a document with a smaller
  set of open sites) → templates fill into templates; filling is incremental.

### 4.4 `admits` — the filling discipline (≠ `formation`)

```python
def admits(core, site, filler) -> bool:
    """Is `filler` admissible for this sorted site? Checked BEFORE substitution,
    while the site (and its _sort) still exists. Default: face-conformance —
    filler's outer face (Edge.interface() / collected _inputs,_outputs) satisfies
    the site's face via core.resolve (structural subtyping: over-provide OK,
    under-provide not). A value-sorted site defaults to core.check(sort, value)."""
```

`admits` (filling) and `formation` (nesting) are **two relations that share an
arity**; do not conflate them. `validate_sorting` continues to police nesting
*after* substitution. Register `admits` per sort (`core.register_sort(name, fn)`);
a face literal auto-registers an anonymous sort whose `admits` is conformance.
"Shape of the process that fits" = the site's face; a non-conforming filler is a
**fill error** (equivalently a sorting violation), reported by name.

### 4.5 Cardinality — a reaction, not a slot (kept, simplified)

"Number of processes / store subtrees" is generative, so it is **not** a site fill
— it is a **`ReactionRule`** applied *before* filling:

- The template marks a repeatable region; a count `n` fires a rule whose **redex**
  is the marked region and **reactum** is *n* keyed copies, via the existing
  `fire_rule` (`assembly.py:1142`). Determinism and collision-free naming come from
  the existing `_remap_keys`/`_gensym_edge` — not restated as prose rules.
- **One mechanism only** — `Map` with generated keys (the natural "N processes in a
  store"); drop the earlier `tensor`/`array._shape` alternatives.
- Runs before `fill_sites` so a per-instance site inside the region is materialized
  `n` times, then filled per-instance (or filled once and broadcast).

*(This keeps the user's explicit requirement — configurable process/store counts —
while removing the seven ad-hoc rules the earlier draft invented.)*

---

### 4.6 Address injection — an abstract process (interface without implementation)

An edge has an `address` (its implementation/protocol), a `config`, a **face**
(`_inputs`/`_outputs` — port types), and **wires**. A template may leave the
**`address` as a site** while fixing the face (and config schema + wiring). That is
**a process definition without an implementation** — an *abstract process*: the
face is the contract, the address is the hole.

- **Fill = inject the address.** The filler is an `address` naming a registered
  process; `admits` requires that process's `interface()` to **conform to the
  site's face** (§4.4). Same `fill`, same `admits` — no new machinery; the site's
  filler is an address string rather than a subtree.
- Two flavors, both one operation: **(a) address-hole, else-fixed** — only the
  *implementation* varies (pure implementation-swap); **(b) whole-edge site** —
  inject a full process/composite.
- **Payoff:** flavor (a) is the solver/model-swap pattern — one solver face, inject
  `copasi` / `tellurium` / `simbio` — i.e. a cross-simulator comparison expressed as
  *one template with an address site*. In component-model terms it is a required
  interface satisfied by plugging in a conforming implementation.
- Depends on address resolution (§11 Task A): an injected address must resolve to a
  `Protocol` for `realize`. Address *injection* is address *resolution* applied to a
  filled address — the same fix enables both.

### 4.7 The contract = the face (typed core) + semantics + amendments

"The face is the contract" generalises: a **contract** is the full interface spec of
an **edge *or* a site**, and the **face** (`_inputs`/`_outputs`) is its
machine-checkable **typed core** — the part `admits` reads. This *unifies* the face
with the existing `ProcessContract` / `Edge.describe_contract()`
(`bigraph_schema/contract.py`), which already carries the documentary part (summary,
math, per-port semantics, assumptions, references). They are not two objects: the
face is the **typed projection** of one contract.

- **Universal.** `describe_contract()` today answers only for edges; extend it to
  **every edge and every site** — a **site's sort *is* a contract**. A template site
  says "a process satisfying *this contract*" (typed face + documented meaning), not
  merely "shaped like X". So the face-as-contract holds at exactly the granularity
  you fill.
- **`admits` = `face_conforms` + amendment predicates.** The face decides typed
  admissibility; amendments may add predicate-bearing constraints.

**Amendments** — an ordered, append-only, provenance-carrying refinement of a
contract:

```
amendment = { op, target, detail, by, when, why }
   op ∈  narrow    (tighten the face / add a port or constraint)
         annotate  (add or refine a description / per-port semantics)
```

Decision (per review): **`narrow` + `annotate` only** — a contract may get *stricter*
and *better-documented* as it flows down, never looser and never gain new ports
(`extend` is out until a use case demands it). This keeps amendment **monotone**, so
`admits` stays sound: a filler admissible for an amended (narrower) contract is
admissible for the original. Amendments give you the *description of a contract* as a
first-class, editable thing, and — because each carries `by/when/why` — the
**provenance** of how an interface evolved through composition and filling.

- `amend(contract, amendment)` is pure and append-only; a composite's contract is its
  constituents' contracts, amended.
- Lands downstream: Layer 2a exposes one `describe_contract` for composites; the
  viewer's contract region renders it (face + description + amendments), and a site
  renders its *required* contract.

## 5. Contracts

1. **`fill` is a monoid action on documents.** Filling **independent** sites
   commutes (order-independent); a partially filled document is still a document.
   This law lets binding be incremental, lets gating fill sites at runtime, and lets
   the viewer re-fill one site without rebuilding the world. Ordering matters only
   when a cardinality reaction must expand a region before its inner sites exist.
2. **`admits` gates every fill.** No substitution occurs unless the filler is
   admissible for the site's sort (face-conformance / value-check). A mismatch is a
   fill error naming the site.
3. **Ground = runnable.** `build` returns `(schema, state)` with no open required
   sites (`is_ground` modulo optional). Layer 2a enforces this at `Composite.__init__`.
4. **`compose` is `fill` with positional bindings** — degrades exactly to today's
   `assembly.compose` for anonymous whole-region sites; no divergence from Milner.
5. **Template subsumption.** Every `CompositeSpec` lowers to a template such that
   `build(to_template(spec), overrides) == to_document(spec, overrides)` for all
   existing specs (Layer 2a golden corpus).

---

## 6. Tests

- **Task 0 (blocking).** Compose a **`Link`-bearing subtree into a site inside a
  wired document** and assert the resulting **wires resolve** (ports read from
  stores, not from link nodes). This exercises the untested link-composition branch
  (`assembly.py:245-256`); fix its wire target if broken. **Nothing else is built
  until this is green.**
- **`fill` + `admits`.** Fill a face-sorted site with a conforming process (incl.
  an over-providing one) → passes; a non-conforming one → fill error naming the
  site. Fill a value site; reject a mistyped value; `_default` fallback for an
  optional site.
- **Address injection (abstract process, §4.6).** A template edge with a fixed face
  and an **`address` site**: inject a conforming registered process's address →
  `admits` passes and the filled edge `realize`s to a runnable process; inject a
  non-conforming address → fill error naming the face mismatch. Assert the two
  flavors (address-hole vs. whole-edge site) both go through the same `fill`.
- **Contract + amendments (§4.7).** `describe_contract` answers for a **site** (its
  required contract) as well as an edge; `amend` is append-only; a `narrow` amendment
  makes `admits` **stricter** (a filler rejected after narrowing was admissible
  before — monotonicity); `annotate` changes docs without changing admissibility;
  `extend` is refused.
- **Composition law.** Filling two independent sites commutes; a partially filled
  document is still fillable; `compose` degrades to `assembly.compose` on empties.
- **Cardinality.** A marked region + `n=3` yields three keyed instances
  deterministically (same `n` → identical); `n=1` yields one.
- **Round-trip.** A non-ground document survives `access → render → access`.
- **Naming/collision.** Assert `core.fill_sites` does not shadow `Core.bind`.

---

## 7. Migration / compatibility

Additive. New `core.fill_sites` + `admits`/`register_sort`; `compose` re-expressed
over `fill` (signature preserved). `BASE_TYPES` unchanged. No on-disk migration
here (that is Layer 2a/4). Downstreams pin the bigraph-schema version that adds
`fill_sites`.

---

## 8. Risks

- **The link-composition branch is load-bearing and unproven** — Task 0 is the gate
  (§6). Do not build on it blind.
- **`admits` vs `formation` confusion** — keep them two named relations; test that
  `admits` fires *before* substitution and `validate_sorting` *after*.
- **Cardinality determinism** — the reaction must produce stable keyed names; reuse
  `_remap_keys`/`_gensym_edge`; cap `n`.
- **Name hygiene** — `fill_sites`, not `bind` (collision); `site`, not `slot`.

---

## 9. Out of scope (YAGNI)

- A runtime `composite`/`process`/`step` type (structural; belongs downstream).
- The study/investigation documents + migrator (Layers 2–4).
- Any workbench UI (the ProcessCard viewer is the presentation side).
- General template rewrite semantics beyond the one cardinality reaction.

---

## 10. Decisions (resolved 2026-07-30)

Milner grounding: holes are **sites** (numbered/keyed, sorted); composition plugs a
filler's roots into sites, defined iff the shared **face** matches; constraining a
hole is a **sorting**. All four decisions land as *deletions*:

1. **A slot is a keyed, sorted `Site` — no subclass, no `Slot` type.** Name = key;
   sort = filling constraint; `optional`/`default` on the site.
2. **`composite` is not a slot type** — a "composite" filler is just an edge with a
   (possibly empty) face; the real composite concept is hardened in process-bigraph.
   `BASE_TYPES` gains nothing.
3. **`build` returns `(schema, state)`** — runnable for Layer 2.
4. **`fill` is the one primitive; `compose` is a two-line adapter over it; register
   `core.fill_sites`** (never shadow `Core.bind`). `assembly.instantiate` already
   implements the substitution.

---

## 11. Implementation status (2026-07-30 · PR #175)

Built by Fable; **1086 tests pass, 0 failures** (baseline 1067). Shipped:
`assembly.fill_sites` (the primitive), `admits`/`admits_why`/`face_conforms`/
`collect_face`/`collect_sites`, `compose` re-expressed over one shared substitution,
`core.fill_sites`/`register_sort`/`admits`. All deletions held (no `compose_at`,
`Slot`/`Template`, `_slots`; `BASE_TYPES` unchanged).

**Task 0 found the foundation was broken (3 bugs, now fixed + tested).** The
untested `compose` link-branch (a) never rebased the wire → realization crashed,
(b) pointed inside a `Link` node, (c) wired only one end. Fixed: a join wires **both**
ports to one shared store, rebased through the site; `compose` raises naming the port
when a join isn't expressible. Also fixed: `render` dropped a site's `_sort` (leaked a
Python object into the JSON document) — face-sorted templates now round-trip.

### Decision 5 — RESOLVED (place semantics): keep the wrapper / site-position form ✅

`fill_sites` keeps `compose`'s semantics — a filler goes **at the site's position**,
not forest-spliced at the parent — now documented and tested (not incidental).
`compose` is unchanged. Evidence from a live `Composite` against a real registered
process: forest-splicing a nested composite **drops the site's key** and **collides**
the filler's inner keys with the template's own (silently losing one), and violates
composition arity (one root → one site). `instantiate`'s forest-splice is correct for
reaction **matching** (a redex site captures sibling keys) — a different operation.

### Status (all green: 1113 tests, PR #175)

- **Address resolution — done.** Root cause: a link's `address` was compiled by the
  *type parser* (`core.access`), never becoming a `Protocol`, so `default`/`realize`/
  `render` walked it as a schema. Fix: one `schema.normalize_address` that `access`
  and `realize` share; `render(Protocol/Link)` corrected (also fixed a second
  pre-existing bug — an address-less link couldn't realize). Plus: `collect_face`
  now resolves a filler's face from `core.link_registry` (a real process declares
  address + config, not ports) — `admits` no longer rejects every genuine filler.
- **Done earlier:** cardinality reaction (§4.5), `build()` (§4.3).

### In flight (queued Fable batch)

- **Address-injection** explicit tests (§4.6).
- **Contract = face + amendments** (§4.7) — `describe_contract` to sites, `amend`
  (`narrow`/`annotate`), `admits` via the contract.

### Flagged, carried to Layer 2a (process-bigraph)

- `Composite.run()` **hangs on `IncreaseProcess`** even at `interval:1.0` — a real
  process-bigraph bug, unrelated to this layer.
- `is_bound`/`find_unbound_links` (`assembly.py:570`) read wires absolute-from-root,
  contradicting `port_merges`' relative-to-parent — untested; left alone.
