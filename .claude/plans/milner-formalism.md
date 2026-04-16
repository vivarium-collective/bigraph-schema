# Reintroducing Milner's full bigraph formalism

**Goal.** Extend `bigraph-schema` and `process-bigraph` so that they recover
the pieces of Milner's *Space and Motion of Communicating Agents* (Dec 2008
draft, `~/Downloads/Bigraphs-draft.pdf`) that the current codebase sets aside:

1. **Open interfaces** on both place and link graph — i.e. sites/roots and
   inner/outer names, so bigraphs aren't always "grounded" and can be
   *composed* and *juxtaposed*.
2. **Signatures** as first-class objects (controls with arity, and optional
   sorting disciplines / formation rules).
3. **Redex / reactum parametric reaction rules** as the primary way state
   changes — complementary to, but not replacing, the time-based `Process`
   abstraction.

This plan walks from Milner's vocabulary through what is already in our
codebase and out to concrete work items.

---

## 1. What Milner says (and what we already do)

The book splits cleanly along the **space / motion** axis that our two
repos already follow:

| Milner | In our code | Status |
| --- | --- | --- |
| Place graph (forest of nodes) | `Place` dataclass with `_subnodes` in `schema.py:35` | ✓ present — but **always grounded** |
| Link graph (hypergraph on ports) | `Link` in `schema.py:233` + `Wires` (tree of `Path`s) | ✓ present — but links are **state wires**, not first-class edges |
| Controls + arity (basic signature `K`) | Type registry + `Edge.inputs()/outputs()` port declarations | ✓ implicit — each process' input/output schema is a per-node signature |
| Reaction rule `r ⟶ r'` (redex + reactum) | ∅ | ✗ absent |
| Bigraphical reactive system (BRS) | `Composite` + `Process.update` | ≈ present as *time-based updates only*, not rule-based |
| Global clock | `Composite.run()` front-buffer scheduler (`composite.py:1937`) | ✓ our contribution; not in Milner |
| Interfaces `I = ⟨m, X⟩` | `Composite.bridge = {inputs, outputs}` (`composite.py:1142`) | △ only at composite outer boundary; no "sites below" |
| Composition `G ∘ F` | implicit when a `Composite` wires a child | △ ad hoc; not a formal operation between *schemas* |
| Tensor product `F ⊗ G` | ∅ | ✗ absent |
| Sorting discipline Σ = (Θ, K, Φ) | ∅ (type unions come closest) | ✗ absent |

In Milner's vocabulary the current `bigraph-schema` state is always an
arrow `g : ε → I` — a **ground** bigraph — and we collapse the signature
into whatever types happen to be registered. The first three chapters of
the book (Ch. 1 *Idea*, Ch. 2 *Defining bigraphs*, Ch. 3 *Algebra*) are
almost entirely about the structure we are throwing away when we force
grounding.

### 1a. The definitions we need to honour

Verbatim from the book (with page refs):

- **Concrete place graph** (Def. 2.1, p. 16):
  `F = (V, ctrl, prnt) : m → n` where `m` indexes *sites* (inner) and `n`
  indexes *roots* (outer). `prnt : m ⊎ V → V ⊎ n` is acyclic.
- **Concrete link graph** (Def. 2.2, p. 16):
  `F = (V, E, ctrl, link) : X → Y`. `X` is the inner name-set, `Y` the
  outer, `link : X ⊎ P → E ⊎ Y` where `P` is ports.
- **Concrete bigraph** (Def. 2.3, p. 16): pair of the two constituents,
  with interface `I = ⟨m, X⟩ / ⟨n, Y⟩`.
- **Basic signature** `K = (K, ar)` (Def. 1.1, p. 7): controls and arity.
- **Composition** (Def. 2.5, p. 17): `G ∘ F` connects roots-of-F to
  sites-of-G, and outer-names-of-F to inner-names-of-G; the mediating
  face `J` "ceases to exist".
- **Tensor / juxtaposition** (Def. 2.7, p. 18): side-by-side of disjoint
  bigraphs; interfaces disjoint-union.
- **Elementary bigraphs** (Def. 3.1–3.5, p. 28–29): `1` (barren root),
  `join : 2→1`, `γ_{m,n}` (symmetry), substitution `y/X`, closure `/x`,
  ion `K_⃗x`, atom, molecule. **Everything else is built by ∘ and ⊗.**
- **Reactive system** (Def. 7.1, p. 68): s-category + set `R` of rules.
  Reaction `a ⟶ a'` whenever `a ≃ c∘r` and `a' ≃ c∘r'` for some
  `(r, r') ∈ R` and context `c`.
- **Wide reactive system (WRS)** (Def. 7.2, p. 68): adds a width functor
  and an **activity relation** `act ⊆ `C(I→J) × width(I)`. Activity
  answers *where reactions may fire*.
- **Dynamic signature** (Def. 8.2, p. 81): assigns each control a status
  in `{atomic, passive, active}`. A bigraph is active at site `i` iff
  every ancestor node of `i` has an active control.
- **Parametric reaction rule** (Def. 8.5, p. 84): triple
  `(R : m → J, R' : m' → J, η : m' → m)` where `η` is the
  **instantiation map** saying which redex parameter fills each reactum
  site. Generates ground rules `(R·d, R'·η̄(d))` for discrete `d`.
- **BRS** (Def. 8.6, p. 85): `BG(Σ)` + set of parametric reaction rules.

---

## 2. Design principles

Before laying out work items, three principles that fall out of reading
the draft next to our code:

### P1. Keep the clock; add rules alongside it.

Milner's reaction relation is **untimed**: `a ⟶ a'` just says there
*exists* a context and a rule. Our contribution is the global clock and
interval scheduling. These compose rather than conflict — a rule engine
is just another kind of `Process`/`Step` whose `update` is "find a
redex-match, substitute reactum". We do **not** rip out time; we
introduce rules as a second idiom beside it.

### P2. Interfaces live on schemas, not on state.

In Milner a bigraph `F : ⟨m, X⟩ → ⟨n, Y⟩` carries interfaces as part of
its *identity*. In our code an object with holes has no obvious home —
state is concrete. The natural place for interfaces is therefore the
**schema layer** (`bigraph-schema`). A schema with open sites is a
*template* for composition; the runtime only ever sees ground instances.
This matches the split Milner draws between concrete (support-aware) and
abstract (support-forgotten) bigraphs (Def. 2.19).

### P3. Sort everything through composition.

The algebraic view (Ch. 3) says every bigraph is built from elementary
ones by `∘` and `⊗`. If we expose those two operators on schemas, users
get composition for free on every schema they write — the same way they
get `|` for merge today via `parse.py:33`. We should lean on the grammar
we already have.

### P4. Reactions are per-signature, not global.

A BRS is always indexed by a signature `Σ` (Def. 8.6). In our terms:
rules are registered against a specific `Core` / signature, not into the
runtime at large. Process-bigraph already has per-`Core` type
registration (`core.py:288`); extend that pattern.

---

## 3. Concrete work items

### Phase A — Interfaces in `bigraph-schema` (the *space* half)

**Goal.** Introduce *sites* (open place-graph holes) and *names* (open
link-graph endpoints) as first-class schema nodes, and implement
composition / tensor on schemas.

A.1 **`Site` node** — new dataclass in `bigraph_schema/schema.py` next to
`Place` (line 35). Fields:
```
class Site(Node):
    _schema_keys = Node._schema_keys | {'_sort', '_index'}
    _sort: str = ''     # place-sort label (Ch. 6)
    _index: int = 0     # which site this is (for ordered inner face)
```
A site is a "hole" in the schema — when you compose, its slot is filled
by a root of the other bigraph.

A.2 **`Name` node** — new dataclass. Two flavours, matching Milner's
*inner* / *outer* distinction:
```
class InnerName(Node):  # port-level, gets linked from outside
    _schema_keys = Node._schema_keys | {'_sort'}
    _sort: str = ''

class OuterName(Node):  # exposed upward; becomes a link endpoint
    _schema_keys = Node._schema_keys | {'_sort'}
    _sort: str = ''
```
Idle names (no points mapped to them) are permitted, echoing
Def. 2.19 (*lean* / *lean-support*).

A.3 **`Signature` node** — distinct from the type registry. A
`Signature` records `{control_name: arity}` and optionally a status
(atomic / passive / active, Def. 8.2) and a sort. It subsumes the ad
hoc shape that `Edge.inputs()/outputs()` currently fills.

```
class Signature(Node):
    _schema_keys = Node._schema_keys | {'_controls', '_sorts', '_formation'}
    _controls: dict = ...   # {name: Control(arity, status, sort)}
    _sorts: tuple = ()      # Θ from Def. 6.1
    _formation: object = None  # Φ formation rule, optional
```
The existing `BASE_TYPES` registry (`schema.py:552`) becomes one
anonymous signature; new signatures can be registered per-`Core`.

A.4 **Interface as a schema pair** — introduce
`Interface = ⟨place_face, link_face⟩`:

```
class Interface(Node):
    _schema_keys = Node._schema_keys | {'_sites', '_roots', '_inner_names', '_outer_names'}
    _sites: tuple = ()       # ordered list of Site schemas
    _roots: tuple = ()       # ordered list of region/root schemas
    _inner_names: dict = {}  # {name: sort}
    _outer_names: dict = {}  # {name: sort}
```
A schema with `_sites == () and _inner_names == {}` is **ground** — the
shape we allow today. The trivial interface `ε = ⟨0, ∅⟩` is just the
empty `Interface`.

A.5 **Composition and tensor operators** in a new
`bigraph_schema/assembly.py`:

```
def compose(outer: Schema, inner: Schema) -> Schema: ...
def tensor(left: Schema, right: Schema) -> Schema: ...
def identity(interface: Interface) -> Schema: ...
def symmetry(interface_a, interface_b) -> Schema: ...  # γ_{I,J}
```

Rules from Def. 2.5 + 2.7:
- `compose` requires `outer.inner_face == inner.outer_face`; the common
  face is erased; support must be disjoint.
- `tensor` requires disjoint supports and disjoint name-sets; interfaces
  concatenate.
- All four category axioms (C1–C3, M1–M3, S1–S4 from Defs. 2.8–2.11)
  must hold; ship property-based tests for them in `tests.py`.

A.6 **Parser extension** — `parse.py:33` already has `|` for merge. Add:
- `∘` (or `.`) for composition: `outer ∘ inner`
- `⊗` (or `||`) for disjoint juxtaposition

Strategy: introduce the new operators inside `[ ... ]` type-parameter
lists first to avoid ambiguity, e.g. `compose[outer, inner]`. Move to
infix sugar once the semantics are settled. Milner's `.` (nesting) and
`||` (parallel product) from §6.3 (p. 64) can later be derived:
`F.G = (id ⊗ merge) ∘ (F ⊗ G)` etc.

A.7 **Open vs. grounded guards** — `Composite.run()` today implicitly
assumes grounded state. Add a check: if the instantiated schema has
`_sites` or `_inner_names`, reject with "cannot run an open bigraph;
compose or instantiate its holes first".

### Phase B — Reaction rules in `process-bigraph` (the *motion* half)

**Goal.** Reaction rules as parametric `(redex, reactum, instantiation)`
triples, executed by a rule-engine driver that plays the role Milner's
reaction relation plays, while the global clock still provides the
scheduling skeleton.

B.1 **`ReactionRule` type** in a new
`process_bigraph/reactions/rule.py`:

```
class ReactionRule:
    redex: Schema         # open bigraph with sites/names (Phase A)
    reactum: Schema       # same outer face as redex
    instantiation: dict   # η : reactum_sites → redex_sites
    rate: float | None    # optional stochastic rate (Ch. 11.4)
    label: str
```
The `redex` and `reactum` are `bigraph-schema` schemas with open
interfaces — so this type *requires* Phase A.

B.2 **Matching engine** in `process_bigraph/reactions/match.py`. Given a
ground state + a parametric redex `R : m → J`, find occurrences:

- Walk the state tree looking for subtrees whose local structure is
  support-equivalent to `R` modulo name renaming (Def. 2.4 *support
  translation*).
- Link-graph matching respects arity and sort.
- Matching sites bind parameters `d_0, ..., d_{m-1}` — the concrete
  subtrees that fill each site.
- Honour the dynamic signature: a redex only matches at an **active**
  location (Def. 8.2: every ancestor of the match-root has an active
  control).

Implementation note: this is the structural-matching half of
`graph homomorphism / subgraph isomorphism`. Keep it simple first —
linear walk, no indexing — and add optimisation behind the same API.

B.3 **Instantiation** in `process_bigraph/reactions/instantiate.py`. Given
a match + a reactum + `η`, build the new subtree: plug parameter `d_η(j)`
into reactum site `j` (Def. 8.3). If `η` is non-injective, `d_η(j)` is
shared — under many-one sorting this means a `bound` link must be used,
otherwise parallel product (`||`, p. 83) duplicates.

B.4 **`ReactionProcess`** — a `Process` subclass in
`process_bigraph/reactions/process.py` that carries a rule-set and fires
applicable rules on each `update(state, interval)`. Choice of which
rule fires each tick:

- Deterministic: first match wins (for debugging / worked examples like
  the `B1–B3` built-environment rules on p. 8).
- Stochastic: weight matches by `rule.rate`, sample via Gillespie /
  exponential — matches Milner §11.4's sketch and `process-bigraph`'s
  stochastic-friendly intent.

`ReactionProcess` fits the current runtime because `Composite` already
dispatches `Process.update` through its front buffer
(`composite.py:2013`). No new scheduling path is needed.

B.5 **BRS bundle** — a top-level helper for users who want Milner's
BRS idiom without writing processes by hand:

```
brs = BRS(signature=Σ, rules=[r1, r2, ...], initial=g)
brs.run(interval=...)  # or brs.step()
```
Under the hood this constructs a `Composite` with one `ReactionProcess`
whose state is `g`.

### Phase C — Sorting / formation rules (optional, later)

Ch. 6 introduces place-sorting, link-sorting, and formation rules Φ.
Useful for encoding CCS (stratified sorting, Def. 6.5), π-calculus
(plain sorting, Def. 6.16), Petri nets (many-one sorting, Def. 6.12).

C.1 Extend `Signature` with `_sorts: tuple` and `_formation: Callable`
(sketched in A.3).

C.2 Make `compose` / `tensor` refuse combinations that violate Φ,
mirroring Milner §6.3 ("the product may violate the formation rule").

C.3 Encode the three canonical sortings as library presets:
`ccs_sorting()`, `pi_sorting()`, `petri_sorting()`. Each is a short
function producing a `Signature` with its Φ baked in.

This phase is intentionally last because the core benefit lands in A+B;
sorting is a refinement that unlocks specific process calculi but is not
needed for e.g. the built-environment example (Ch. 1 p. 8).

---

## 4. What lives where

| Concept | Repo | New file / module |
| --- | --- | --- |
| `Site`, `InnerName`, `OuterName`, `Signature`, `Interface` | `bigraph-schema` | `bigraph_schema/schema.py` (extend) |
| `compose`, `tensor`, `identity`, `symmetry`, elementary bigraphs | `bigraph-schema` | `bigraph_schema/assembly.py` (new) |
| Parser support for ∘ and ⊗ | `bigraph-schema` | `bigraph_schema/parse.py` (extend) |
| Composition property tests (C1–C3, M1–M3, S1–S4) | `bigraph-schema` | `tests.py` |
| `ReactionRule` + parametric (R, R', η) | `process-bigraph` | `process_bigraph/reactions/rule.py` |
| Redex matching | `process-bigraph` | `process_bigraph/reactions/match.py` |
| Reactum instantiation | `process-bigraph` | `process_bigraph/reactions/instantiate.py` |
| `ReactionProcess` (time-stepped rule driver) | `process-bigraph` | `process_bigraph/reactions/process.py` |
| `BRS` helper / top-level idiom | `process-bigraph` | `process_bigraph/reactions/__init__.py` |
| Worked examples (built environment A1–A3, mobile ambients, CCS fragment) | `process-bigraph` | `process_bigraph/reactions/examples/` |

---

## 5. Tradeoffs & open questions

### Q1. Should interfaces be carried on every schema or only on an "open schema" subtype?

Pro-everywhere: uniform. Con: noisy for the 99% grounded case. *Proposed:*
every schema has an `Interface` field defaulting to `ε`. Grounded remains
the default and no user has to think about interfaces unless composing.

### Q2. How do our `Wires` (paths) relate to Milner's link graph?

`Wires` (`schema.py:216`) is a tree of `Path` — i.e. at runtime a process
port gets hooked to an *absolute path in state*. That is one particular
implementation of Milner's `link` function. Our paths are always closed
(they resolve to a concrete store); Milner's `link` may map a port to an
open outer name. The natural translation: an outer name is a path that
*has not yet been bound*. `compose` is then "substitute the inner
bigraph's outer-name bindings into the outer bigraph's inner-name
unbound paths". That makes composition a specifically bigraph-flavoured
wire resolution step.

### Q3. Concrete vs. abstract bigraphs (Def. 2.19)?

Milner's *support* (node/edge identities) is not something we currently
track — our state has object identities only via Python. For now we work
at the *abstract* level (quotient by lean-support equivalence). If
tracking becomes relevant (Ch. 11.1, or stochastic history), we can add
an explicit support set as metadata.

### Q4. Should rules be first-class processes or second-class?

Treating rules as a `ReactionProcess` keeps the runtime simple and
pluggable, but costs one layer of indirection (the rule-set is hidden
inside a process config). Alternative: add a fourth top-level concept
alongside `Process`/`Step`/`Composite` called `Rule`, and have
`Composite.run()` match rules between process updates. *Proposed:* start
with `ReactionProcess`, promote to `Rule` only if the matching layer
needs tight integration with scheduling (e.g. stochastic rates that must
interact with the global clock's adaptive stepping).

### Q5. Reaction rules vs. existing `_add`/`_remove`/`_divide` machinery.

`process-bigraph` already has structural-change sentinels
(`composite.py:2151`). Those are *imperative* (a process emits "please
add this child"), whereas reaction rules are *declarative* (the engine
infers the change from a pattern). These coexist: `_add` etc. are the
*how*, a rule is the *what/when*. A rule engine, on firing, ultimately
emits `_add`/`_remove` operations into the state update.

---

## 6. Milestones

1. **M1.** `Site` + `Interface` types land; composition operator works
   on the trivial case `g ∘ id = g`. — 1 week.
2. **M2.** `compose` + `tensor` pass C1–C3, M1–M3, S1–S4 property tests.
   The Ch. 3 elementary bigraphs (`1`, `join`, `γ`, `y/X`, `/x`, `K_⃗x`)
   are all constructible. — 2 weeks.
3. **M3.** `Signature` + dynamic signature (`atomic/passive/active`)
   land. `ReactionRule` exists as a data type. — 1 week.
4. **M4.** Matching + instantiation pass the built-environment example
   (B1–B3, p. 8) end-to-end. — 2 weeks.
5. **M5.** `ReactionProcess` integrated with the global clock; stochastic
   rates working for a toy biological example. — 1 week.
6. **M6.** (Optional) CCS + mobile ambients translations as library
   presets; sorting disciplines (Phase C). — 2 weeks.

Total: ~8 weeks for A + B, another ~2 for C.

---

## 7. Reading roadmap for the reviewer

If you've not read the book, these are the minimum pages to skim before
reviewing this plan:

- **p. ix–xii** — Milner's own summary of the intent (space / motion /
  why).
- **p. 3–13** — Ch. 1 *Idea*: the picture of a bigraph, reaction rules
  B1–B3 for the built environment, Figure 1.2 *Anatomy*.
- **p. 15–18** — Defs. 2.1–2.7: concrete place graph, link graph,
  bigraph, support, composition, juxtaposition. **This is the formal
  core of Phase A.**
- **p. 27–29** — §3.1: elementary bigraphs and normal forms.
- **p. 67–69** — §7.1: reactive systems, WRS, activity relation.
- **p. 81–85** — §8.1: dynamic signatures, parametric rules, BRS. **This
  is the formal core of Phase B.**

Everything else (RPOs, IPOs, bisimilarity, sorting, CCS/ambients
translations) can be postponed until the relevant phase.
