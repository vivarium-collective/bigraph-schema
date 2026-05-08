# Composite algebra

A categorical analysis of the bigraph-schema / process-bigraph stack, taken
from the perspective of the *Composite* as the object of study. Reconcile,
apply, schema, processes, and protocols all appear here as components of the
composite's algebraic and coalgebraic structure.

The aim is twofold:

1. Pin down the structure rigorously enough that we can name testable laws
   and write property-based tests against them.
2. Use the structure to *ask what's missing* — what the categorical view
   suggests should exist (uniformly, generically, named) but currently
   doesn't, or exists only in scattered, type-specific form.

## Cast of characters

A composite has several kinds of data, each with its own algebra. The full
structure is the *interaction* between these algebras.

| Layer | What it is | Key operations |
|---|---|---|
| **Schema** `S` | A tree of type declarations | `check`, `default`, `realize` |
| **State** `X(S)` | A value inhabiting a schema | constructed from defaults; mutated by `apply` |
| **Update** `U(S)` | A value representing a change | combined by `reconcile`; consumed by `apply` |
| **Process** `P` | A function reading state, emitting updates | invoked per-tick |
| **Composite** `C` | A state value containing processes that act on it | one tick = read → reconcile → apply |
| **Protocol** | A morphism between composites (local, RPC, distributed) | preserves apply/reconcile |
| **Bridge / Port** | A path-based slice of state into a process | get / set, lens-shaped |

We'll ascend through these in roughly that order.

## Layer 1 — The type triple `(X, U, apply)`

For each schema `S`, three sets and one operation:

- `X(S)` — the state set: values `x` with `check(S, x) = True`.
- `U(S)` — the update set: shapes of update the apply dispatch accepts. For
  most types this is `X(S)` itself plus optional structural sentinels
  (`_add`, `_remove`, `_divide`, `_type`, `set`); for some it's strictly
  smaller (e.g., `Const`'s `U(Const) = ∅`).
- `apply_S : X(S) × U(S) → X(S)` — the action of an update on state.

This is the atomic unit. Every schema constructor (`list[T]`, `map[T]`, etc.)
is a *functor* from "type triple of T" to "type triple of F(T)" — it lifts
the inner triple to a new one (Layer 3).

**Observation.** `U(S)` is currently anonymous and duck-typed (dicts with
reserved keys, raw values, lists). Making `U(S)` a named, schema-derived
type — a discriminated union of `Plain(value)`, `Add(items)`,
`Remove(indices)`, `RemoveAll()`, `Divide(spec)`, `Retype(schema)`, etc. —
would let us pattern-match exhaustively in `apply` and `reconcile`, catch
malformed updates at construction, and document each schema's update algebra
in one place. Currently the algebra is *implicit* in the apply/reconcile
dispatches and the only place it's enumerated is in this audit's reproducers.

**Open category-theoretic question.** Is the update set always functorially
derivable from the state set? The structural sentinels suggest no — they
encode operations on the *shape* of state, not just its value, so they
require additional structure beyond what the state set carries.

## Layer 2 — Updates as a monoid; apply as an action

For each schema `S`, `(U(S), reconcile, ε)` should form a **monoid**:

- **Identity**: `ε = None` (or `0` for numeric types), with
  `reconcile(s, [u]) = u` and `reconcile(s, []) = ε`.
- **Closure**: `reconcile(s, [u₁, u₂]) ∈ U(S)`.
- **Associativity**:
  `reconcile(s, [reconcile(s, [u₁, u₂]), u₃]) = reconcile(s, [u₁, reconcile(s, [u₂, u₃])])`.

For *commutative* types (numerics, sets, the symmetric add/remove cases of
List), the monoid is **commutative**. For non-commutative ones (`String`,
`Overwrite`, the asymmetric structural-sentinel cases) the monoid is
non-commutative — order matters and reconcile must define a canonical rule.

`apply_S : X × U → X` is a **monoid action** of `(U, reconcile, ε)` on `X`:

- `apply(x, ε) = x` — identity.
- `apply(x, reconcile(s, [u₁, u₂])) = apply(apply(x, u₁), u₂)` — compatibility.

This compatibility law is exactly what the reconcile docstring promises *for
commutative types*. For non-commutative types, the law's RHS depends on
order; reconcile is then a *batched* alternative semantics — capturing what
"these all happened concurrently in this tick" means, which `fold(apply)`
cannot.

**Testable laws** (machine-checkable via property tests):

```
L1. apply(x, reconcile(s, [])) == x
L2. apply(x, reconcile(s, [u])) == apply(x, u)
L3. reconcile(s, [reconcile(s, A), reconcile(s, B)]) == reconcile(s, A ++ B)        # associativity
L4. reconcile(s, A) == reconcile(s, permute(A))                                     # commutativity (commutative types only)
L5. apply(x, reconcile(s, A)) == fold(apply, x, A)                                  # action compatibility (commutative types only)
L6. apply(x, reconcile(s, A ++ [reset])) == apply(reset_apply(x, reset), filter_non_reset(A))   # reset-then-add (containers with reset sentinels)
```

L1–L4 should hold *uniformly across all schemas*. L5 holds for the
commutative subset. L6 is the *cross-cutting principle* the audit
identified — currently per-type and inconsistently honored.

**What's missing.** No property-based test suite exists. Adding one would
turn each schema's apply/reconcile pair into a small theorem to be checked
on every change. Hypothesis or similar can do this in a few hundred lines.

## Layer 3 — Schemas as functors

A schema constructor `F` (e.g. `list[·]`, `map[·]`, `tree[·]`, `tuple[·]`)
takes a triple `T = (X(T), U(T), apply_T)` and produces `F(T) = (X(F(T)),
U(F(T)), apply_{F(T)})`. The `@dispatch def apply(schema: List, ...)`,
`@dispatch def reconcile(schema: Map, ...)` machinery is the implementation
of these functors.

The functoriality has two layers:

1. **Inner layer (covariant)**: `F` lifts inner-element updates by recursing
   into the contained `T`. `apply(List[T], state, [u₁, u₂])` recurses into
   `apply(T, state[i], uᵢ)` for each position.
2. **Structural layer (extra-covariant)**: `F` introduces *new* operations
   on the F-shape that don't come from `T` — `_add`, `_remove`, `_divide`,
   `set`. These operations live in `U(F(T))` but are not images of any
   `U(T)` operation under `F`.

The inner layer is a clean covariant functor. The outer layer is *not* —
each container schema introduces its own structural algebra, and these
algebras are **not uniform**. List has `_add`/`_remove`/`_remove:'all'`. Map
has `_add`/`_remove`/`_divide`. Set has `_add`/`_remove` (but no reconcile
dispatch — see audit). Array has `set` (without underscore — different
naming). Tree has `_add`/`_remove` but its reconcile delegates to the leaf
type, dropping the structural layer entirely (audit: CRASH).

**What's missing — the *uniform structural algebra*.** The categorical view
says: these per-type structural sentinels are instances of one common
algebra. Specifically, for any container `F(T)`:

- `Add` — extend the structure with new positions / keys / elements.
- `Remove` — drop positions / keys / elements (with `RemoveAll` as the
  zero / absorbing element).
- `Divide` — split a position into two (only meaningful for some
  containers — `Map`, but not `List` or `Tuple`).
- `Set` — overwrite the entire structure (Array uses this; could be
  uniform across all containers).

Promoting these to a **named base mixin** (`StructuralUpdate[F, T]`) with
shared composition rules would:

- Make Set's missing reconcile / Tuple's missing reconcile / Tree's
  delegated-and-broken reconcile fall out of one base implementation.
- Eliminate the inconsistency between List/Map's "collect/union" handling
  of structural sentinels and dict-schema's "last-wins" handling.
- Bake the L6 reset-then-add law into the mixin, so any new container type
  inherits the right batch composition automatically.

This is the single most concrete unification the categorical view suggests.

**Reconcile sink as a natural transformation.** The `ReconcileSummary` sink
that tracks `(paths_walked, has_structural)` during recursive reconcile is
naturally typed as a **natural transformation** `reconcile ⇒ (reconcile ×
metadata)`. Making this explicit — having reconcile *return* a
`(combined_update, structural_metadata)` pair instead of side-effecting a
sink — would compose cleanly with other "what changed" analyses (audit
logging, cache invalidation, change notifications) without each having to
reinstall its own sink.

## Layer 4 — The composite as a coalgebra

A composite `C` carries:

- A schema `S_C`.
- A state `X_C` of type `S_C`.
- An index `process_paths` of where Processes live in `X_C`.

A **tick** is a function `tick : C → C`. It factors through:

```
              read           run            reconcile         apply
   C  ─────►  (C, P*)  ─────►  (C, U*)  ─────►  (C, U)  ─────►  C
```

where `P*` is the multiset of resident processes and `U*` is their emitted
update list.

This factorization is **coalgebraic**: each tick is one step of a structure
`c : C → F(C)` where `F` is the functor "expand a composite into a tick's
worth of per-process updates plus the carrier state." The trajectory of a
composite is the **anamorphism** — the unfold of `c` from an initial state.

Two consequences:

1. **Bisimulation** is a meaningful notion: two composites are *equivalent*
   when their tick-coalgebras have the same observable behavior (perhaps
   modulo a homomorphism — see Layer 7). This gives a formal handle on
   "model equivalence," which the codebase currently treats only
   case-by-case.
2. **Streaming / corecursive structures** are the natural shape of
   trajectories. Currently `Composite.run(t)` is a Python loop; expressing
   it as an unfold over the coalgebra would expose the streaming structure
   and make e.g. checkpointing, replay, and bisimulation testing cheap.

**Coalgebraic structural sentinels.** When a process emits `_add`, the
*coalgebra itself changes* — the next tick's `F` has a different `P*` set.
This is a coalgebra whose carrier evolves with the state: a higher-order
coalgebraic / dependent picture. The cleanest formalism here is probably
the **dialectica** category, where both forward (state) and backward
(carrier / shape) movement are first-class.

**What's missing.** A formal `Tick` type (a function plus its co-step
metadata) and an explicit `Trajectory[C]` corecursive type. Currently both
live implicitly inside `Composite.run`.

## Layer 5 — Bridges and ports as lenses

Each process declares **ports**: input slots that read from the composite
state and output slots that write updates back. A port is bound to a *path*
in the composite tree. The bind operation is a **lens**:

- `view : X_C → V` — read the port's slice.
- `update_set : (V → U_V) × X_C → U_C` — lift a port-local update to a
  composite-wide update.

The classical lens laws:

- *get-put*: `update_set(λv. v, x) = ε_C` (a no-op port write produces a
  no-op composite update).
- *put-get*: reading after writing returns what was written.
- *put-put*: composing two writes equals one composite write.

The `read_bridge` / `write_bridge` machinery in `Composite` is essentially
hand-rolled lens composition over paths. Names of relevant pieces:
`apply_updates`, the `Defer`/`project` pipeline.

**What's missing.** Lenses are not first-class. If they were:

- Port composition would be lens composition — already true informally;
  formalizing it would expose the laws and give us property tests for
  port wiring correctness.
- Profunctor optics would let us mix simple lenses with traversals
  (over Map values), prisms (over Union variants), and isos (over Wrap /
  Maybe) in one uniform algebra. The current `read_bridge`/`write_bridge`
  has implicit knowledge of all these; making them lens-shaped families
  would make the abstraction explicit.

This is a refactor more than a fix — but a *suggested* one. The structure
is already there; it just isn't named.

## Layer 6 — Self-modification: structural sentinels and schema rewriting

The tick can modify the composite's *carrier*:

- `_add` of a new process changes `process_paths`.
- `_remove` of a process trims `process_paths`.
- `_divide` is `_remove(mother) ∘ _add(daughter₁) ∘ _add(daughter₂)`.
- `_type` rewrites the schema at a node.

This is the most categorically interesting (and least clean) layer.
Operationally it's handled by:

1. Apply emitting structural events (`NodeAdded`, `NodeRemoved`, `Divided`).
2. The composite consuming these events to update `process_paths`.
3. A `realize()` pass instantiating new links / process bindings.

The structure is roughly: **state and schema co-evolve**, with a
co-evolution coalgebra:

```
    (S_C, X_C, processes)  →  (S_C', X_C', processes')
```

This factors as `state-evolve · schema-rewrite` per tick.

**Categorical framing**: this is a **dialectica-style** structure. Updates
have a forward component (modify state) and a backward component (modify
schema/process index). Reconcile must compose both components correctly.

**What's missing.**

1. *Schema-rewrite as a first-class algebra.* `_type` updates currently mix
   schema-data with state-data inside the same dict. Promoting them to a
   distinct `SchemaUpdate` type, threaded through a separate
   schema-rewrite phase of the tick, would make composition rules
   explicit. Currently they're implicit in dict reconcile's
   "last-non-None-wins" treatment of `_type`.
2. *Divide as a generic combinator.* `_divide` is currently Map-specific
   in apply, but the *concept* — split one position into N — generalizes
   to any indexed container, and to processes (clone-and-perturb is a
   common pattern). Naming the combinator (`Split[F, T]: position × N →
   list[position]`) would unify these.
3. *Inverse operations / undo.* Dialectica structure suggests every
   forward-update should have a backward-update (an *undo*). Currently
   there is no first-class undo — replay-from-checkpoint is the only
   path to "back up." A formal `inverse(u)` for invertible updates would
   enable speculative tick / rollback / counterfactual analysis.

## Layer 7 — Protocols as composite homomorphisms

A protocol (`local`, `rest`, `ray`, `nextflow`) is a way to *implement* a
process. Two composites differing only in protocols are observationally
identical — same trajectory, same observable state. Mathematically, a
protocol is a **morphism** in the category of composites:

- A homomorphism `f : C₁ → C₂` preserves apply: `apply(f(x), f(u)) = f(apply(x, u))`.
- It preserves reconcile: `reconcile(s, map(f, U)) = f(reconcile(s, U))`.
- It preserves the tick: `tick(f(c)) = f(tick(c))`.

This is exactly what we want for "running the same model locally vs.
distributed gives the same answer."

**What's missing.**

1. *Tests for protocol homomorphism.* The Ray / Rest / Nextflow protocols
   are presumed-equivalent to local but this isn't formally checked.
   Property tests asserting "trajectory under protocol P = trajectory
   under local" for randomly-sampled composites would catch divergences.
2. *Composite morphisms as a first-class operation.* "Reduce a fine-grained
   composite to a coarse-grained one" (model abstraction) is a homomorphism
   currently expressed only by hand. A `coarse_grain : C → C'` operation
   with a checkable preservation law would formalize multi-scale modeling.

## Layer 8 — Reconcile as a free-monoid quotient

`reconcile : list[U] → U` is the universal arrow making `U` a quotient of
the free monoid `list[U]` modulo the per-type laws. This means:

- Any monoid homomorphism `list[U] → M` factors uniquely through reconcile.
- Reconcile *is* the algebraic structure of `U`, made explicit.

**Suggestion: dual operation `split : U → list[U]`.** The categorical dual
exists but isn't implemented. Use cases:

- **Distribution**: split an update across replicas / shards.
- **Refinement**: re-apply parts of a failed update.
- **Diff visualization**: decompose a complex update for inspection /
  audit logs.
- **Speculative tick**: split, apply prefix, evaluate, possibly rollback.

This would make the update algebra symmetric. Currently we have one
direction (`reconcile`) but not the other.

## Layer 9 — Initial and terminal composites

Category theory is fond of initial and terminal objects.

- **Initial composite**: empty schema, empty state, no processes. Every
  composite admits a unique morphism *from* it (the construction history).
  Useful as a base case for inductive reasoning about composites.
- **Terminal composite**: the unique composite with no transitions. This
  would be the *fixed point* of `tick` — a state from which no process can
  emit a non-trivial update. Currently the system runs forever (or until
  external `t_max`); a notion of "terminal / quiescent" would be useful for
  steady-state analysis and for shutting down distributed actors cleanly.

## Putting it together — the structure

```
                    ┌─────────────────────────────────────────────┐
                    │  Composite C = (Schema, State, Processes)   │
                    └─────────────────────────────────────────────┘
                                       │
                  ┌────────────────────┼────────────────────┐
                  │                    │                    │
              tick coalgebra      type triples         protocols
              C → F(C)            (X, U, apply)        C ↔ C′
                  │                    │                    │
            ┌─────┴─────┐         ┌────┴────┐          ┌───┴────┐
            │           │         │         │          │        │
        anamorphism  reconcile  monoid   functor    homo-     fibration
        (trajectory) (U-batch)  (U,⊕,ε)  (F:T→F(T)) morphism  (impl/spec)
                                                     (preserves)
                  │                    │                    │
                  └────────────────────┼────────────────────┘
                                       │
                          dialectica self-modification
                          (state ⨯ schema co-evolve)
                                       │
                          ┌────────────┴────────────┐
                          │                         │
                    structural sentinels      schema rewrites
                    (_add/_remove/_divide)    (_type)
```

The composite sits at the centre. Reconcile, apply, schema, processes,
protocols are all *facets* of this central object. The categorical view
makes the relationships explicit — and reveals that several of these facets
are currently *partial* implementations of structures that *should* be
uniform.

## What the structure suggests is missing — punch list

In rough order of "would clean up real problems" → "would unlock new
capabilities":

1. **Uniform `StructuralUpdate[F, T]` mixin.** One base class encoding
   `Add` / `Remove` / `RemoveAll` / `Set` with the L6 reset-then-add law,
   inherited by List / Map / Set / Tuple / Tree / Array. Eliminates the
   audit's CRASH and DROP bugs *as a class*, not one-by-one.
2. **Named update-type per schema.** Instead of duck-typed dicts, a
   discriminated union per schema. Catches malformed updates at
   construction; documents the algebra in one place.
3. **Property-based tests for laws L1–L6.** Hypothesis-driven check of
   monoid axioms and action compatibility for every schema. Each new
   schema constructor automatically gets law-checking.
4. **Reconcile returns metadata explicitly.** Replace the `ReconcileSummary`
   sink with a return-value pair; treat structural-change tracking as a
   first-class natural transformation.
5. **`split : U → list[U]`** as the dual of reconcile. Enables
   distribution, refinement, diff visualization, speculative tick.
6. **First-class lenses for ports.** Refactor `read_bridge`/`write_bridge`
   into lens composition. Surface the lens laws as port-wiring tests.
7. **`SchemaUpdate` algebra distinct from `StateUpdate`.** Threaded
   through a schema-rewrite phase of the tick. Makes self-modification
   explicit and composable.
8. **Protocol-homomorphism property tests.** Trajectory-under-protocol
   equality as a checkable law.
9. **Composite homomorphism / coarse-grain** as a first-class operation.
   Multi-scale modeling support.
10. **`inverse(u)` for invertible updates.** Speculative tick, rollback,
    counterfactual analysis.
11. **Coalgebraic `Trajectory[C]`** type — the unfold of `tick`. Replay,
    checkpointing, bisimulation testing.
12. **Quiescent / terminal composite detection.** Fixed-point of tick,
    useful for steady-state analysis and clean shutdown.

(1)–(4) are the immediate cleanup wins implied by the audit. (5)–(12) are
generalizations / new capabilities that the structure suggests but that
aren't strictly required to fix existing bugs.

## Conclusion

The composite is, structurally, a coalgebra over a base of (schema, state,
processes), where:

- States act under update monoids derived from the schema.
- Schema constructors lift these monoids functorially.
- The composite tick is one step of an unfold; trajectories are
  anamorphisms.
- Self-modification (structural sentinels, schema rewrites) introduces a
  dialectica / co-evolutionary layer.
- Protocols and bridges are morphisms preserving the composite's algebra.
- Reconcile is the universal arrow from `list[U]` to `U`; its dual `split`
  would complete the symmetry.

Most of this structure is *implicit* in the current implementation —
discoverable only by reading the dispatches and recognising the patterns.
Naming it explicitly (via the punch-list above) would consolidate scattered
type-specific machinery, eliminate whole classes of bugs (the audit's
CRASH/DROP cases), and unlock new capabilities (split, inverse,
coarse-grain, formal protocol equivalence).

The reconcile audit is the *bug-driven* view of these gaps. This document
is the *structure-driven* view. They agree on the immediate work — fixing
Tree, Array, Set, Tuple, dict-schema reconcilers — and converge on the
same suggested unification: `StructuralUpdate[F, T]` as a base behavior
shared across all containers.
