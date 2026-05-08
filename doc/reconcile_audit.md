# Reconcile audit

> **Status (2026-05-08).** All CRASH/DROP findings below have been fixed
> (List, Tree, Set, Tuple, Array, dict-schema). Map gained `_remove: 'all'`
> for symmetry. The remaining items are documented soft-contention /
> design-decision flags. See `composite_algebra.md` for the structural view.

A per-type review of `reconcile(schema, updates)` looking for batch-invariant
bugs of the form: *given two updates whose individual semantics are clear, the
reconciler silently drops one or produces a contradictory combined update.*

The motivating example (now fixed): `reconcile(List, [[X], {'_remove': 'all'}])`
used to drop `[X]` because the plain-list fast path skipped any batch
containing `_remove: 'all'`. The fix promotes plain-list updates to `_add` so
they survive the concurrent reset.

This document enumerates the analogous gaps for every other dispatch.

## Severity legend

- **CRASH** — reconcile raises on a legitimate multi-update batch.
- **DROP** — reconcile silently discards updates from the batch.
- **AMBIGUOUS** — reconcile resolves to a deterministic result, but the rule
  is order-dependent or undocumented and likely surprises callers.
- **OK** — semantics match the apply-layer behavior and compose under batch.

## Findings

### List — **OK** (fixed)

Reconcile groups all `_add` items + plain-list elements into a combined
`_add`, unions `_remove` indexes, treats `_remove: 'all'` as absorbing.
Apply executes `_remove` first, then `_add` — irrespective of receipt order.

Covered by `TestListReconcile` in `test_matrix.py`.

### Set — **DROP**

Set's `apply` handles `_add`/`_remove` with set-union/difference semantics.
There is no `reconcile(Set, ...)` dispatch, so it falls through to `Node`'s
default, which delegates to the `dict` reconciler, which treats `_add` /
`_remove` as **last-non-None-wins** structural sentinels.

Reproducer:

```python
>>> reconcile(Set(_element=String()), [{'_add': ['a']}, {'_add': ['b']}])
{'_add': ['b']}    # expected: {'_add': ['a', 'b']}
```

The first process's contribution is silently dropped. This is the same class
of bug the List fix addressed, just on a different type.

**Fix**: add a `reconcile(Set, ...)` dispatch mirroring List — union all
`_add`, union all `_remove`, with the same "remove first then add" composition
rule. Consider whether `_remove: 'all'` should also be supported (Set's apply
does not handle it today).

### Tree — **CRASH**

`reconcile(Tree, updates)` delegates to `reconcile(schema._leaf, updates)`.
For a `Tree[Float]` (the typical use), structural updates like
`{'_add': {'x': 1.0}}` get passed to `reconcile(Float, ...)` which executes
`0.0 + {'_add': ...}` and raises `TypeError`.

Reproducer:

```python
>>> reconcile(Tree(_leaf=Float()), [{'_add': {'x': 1.0}}, {'_remove': ['y']}])
TypeError: unsupported operand type(s) for +=: 'float' and 'dict'
```

Tree's `apply`, by contrast, handles `_add`/`_remove` shapewise at the Tree
level and only recurses into the leaf reconciler for non-structural per-key
updates. Reconcile should mirror that two-layer treatment.

**Fix**: rewrite `reconcile(Tree, ...)` along the same lines as `Map`'s — carve
out `_add`/`_remove` (and `_divide` if Tree supports it), group remaining
per-key updates, and recurse into the leaf type for each key.

### Array — **CRASH** (in mixed `set` + delta batches)

Array's reconcile supports several update forms (ndarray dense, sparse list,
sparse dict, `{'set': ...}` overwrite). The loop handles a `{'set': ...}`
arrival by clobbering `result` with the set-dict directly. If a subsequent
additive delta arrives in the same batch, `_merge_array_deltas` is invoked
with `(set_dict, ndarray)` and treats the set-dict as a sparse-coordinate
update — using the literal string `'set'` as an array index, which crashes.

Reproducer:

```python
>>> schema = Array(_shape=(3,), _data=np.dtype('float64'))
>>> reconcile(schema, [
...     np.array([1., 0., 0.]),
...     {'set': np.array([10., 10., 10.])},
...     np.array([0., 1., 0.])])
IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`)
            and integer or boolean arrays are valid indices
```

Even ignoring the crash, the *intended* semantics are unclear:
- Does a `{'set': X}` mid-batch reset and discard prior deltas in the same
  batch (current attempt — broken)?
- Or, by analogy with `_remove: 'all'` for List, does it reset *pre-existing*
  state but leave deltas-this-tick to apply on top?

The user's stated rule for List ("removes first, then all adds, irrespective
of receipt order") generalises naturally: **resets first, then all deltas**.

**Fix**: track `set` separately from deltas, return
`{'set': last_set_arr, '_delta': merged_deltas}` (or similar shape that apply
can interpret as "overwrite-then-add"). Apply handles set first, then sums in
the deltas. Symmetrical with the List fix.

### Tuple — **DROP**

No `reconcile(Tuple, ...)` dispatch. Tuple updates are positional lists, not
dicts. Tuple thus falls through to `Node`'s default which, finding the updates
are not all dicts, returns last-non-None — discarding all earlier per-position
contributions.

Reproducer (sketch):

```python
>>> reconcile(Tuple(_values=[Float(), Float()]), [[1.0, 0.0], [0.0, 2.0]])
[0.0, 2.0]    # expected: [1.0, 2.0] under per-position summing
```

**Fix**: add a `reconcile(Tuple, ...)` that recurses per position into the
component schemas, summing/last-winning per the component's reconcile rule.

### Union — **AMBIGUOUS**

No `reconcile(Union, ...)` dispatch. If concurrent updates target different
variants of the union, Node's last-wins resolves but the result depends on
update-list iteration order. There's no notion of "which variant is active"
at reconcile time.

**Fix**: at minimum, document the rule (last-wins by iteration order). Better:
detect contradictory variant updates as an error, or define a precedence rule
based on schema declaration order.

### Maybe — **AMBIGUOUS**

No `reconcile(Maybe, ...)` dispatch. Falls through to Node. Cannot distinguish
"no update" from "reset to None" — `None` always means the former. There may
be no good representation for the latter today.

**Fix**: pick a sentinel for "explicit-None reset" (`{'_set_none': True}`?) or
declare `Maybe[T]` updates always require explicit form. Worth deferring until
a real use case appears.

### Frame / Link / Path / other structured types

No reconcile dispatches. Updates likely flow through `Node` default. Each
should be examined individually; structured types whose updates aren't pure
dicts will lose information through `Node`'s last-wins.

### Map — **AMBIGUOUS** (multiple subtleties)

The Map reconciler is mostly well-behaved (it carves out
`_add`/`_remove`/`_divide` and recursively merges per-key value updates), but
several edge cases deserve documentation or hardening:

1. **No `_remove: 'all'` analog.** A user who wants to wipe a map currently
   must enumerate keys. Symmetrical with List, this is the natural shape and
   should probably be supported. Same "remove pre-existing first, then apply
   adds" rule applies.

2. **`_add: {'a': X}` + `_remove: ['a']`** in the same batch. Apply runs add
   first, then per-key value updates, then remove — so the key ends up gone.
   The combined update silently lets remove win over add. Worth either
   detecting as an error or documenting.

3. **Multiple `_divide` per tick** on the same map: last-non-None-wins
   silently. A second divide on the same mother is almost certainly a bug,
   not a valid concurrent operation. Should be flagged.

4. **Sibling value updates to a divided mother**: apply runs the divide
   sentinel inside `apply(Map)`, which removes the mother and installs
   daughters; the per-key value update for the mother key is then iterated
   and skipped (mother no longer in state). The value update is silently
   dropped. Probably correct intent, but undocumented.

### dict-schema — **AMBIGUOUS** (inconsistency with Map/List)

For a static dict-shaped schema, structural sentinels (`_add`/`_remove`/
`_divide`/`_type`) are reconciled as **last-non-None-wins**:

```python
# Last non-None wins for structural directives.
for v in reversed(key_updates):
    if v is not None:
        result[key] = v
        break
```

This contradicts how List and Map handle the same sentinels (collect/union).
In practice dict-shaped schemas rarely receive `_add` updates, but the
inconsistency is a correctness pitfall. Same fix shape as List/Map: union
adds, union removes.

### Atomic (Float/Integer) — **OK** with one caveat

Reconcile sums all numeric deltas. Returns `None` when the sum is exactly
zero, which short-circuits `apply` (no-op). The semantic equivalence
`apply(s, x, sum(deltas)) == fold(apply, x, deltas)` holds. Caveat: any
hypothetical apply-time validation/sink hook would be skipped on a
sum-to-zero batch where intermediate values were nonzero. Probably fine.

### String / Boolean / Overwrite / Node default — **AMBIGUOUS**

Last-non-None-wins, where "last" = update-list iteration order, which is
the composite's process-iteration order. From a process author's perspective,
two processes writing the same scalar in one tick give a non-deterministic
result. This is contention, not a reconcile bug per se, but worth flagging:
the system silently picks a winner instead of detecting concurrent writes.

## Recommended fix order

1. **Tree** (CRASH) and **Array** (CRASH) — these are bugs you can hit on
   real data. Highest priority.
2. **Set** (DROP) — same class as the List bug, mechanical fix.
3. **Tuple** (DROP) — likely rare in practice but trivially fixable.
4. **dict-schema sentinel inconsistency** — small bug-shaped change, removes
   a footgun.
5. **Map** structural-sentinel hardening (flag divide conflicts, support
   `_remove: 'all'` if useful).
6. **Union/Maybe/Frame/Link** — defer until a concrete user need surfaces;
   they're under-specified, not actively wrong.
7. **String/Boolean contention detection** — separate design question; might
   warrant a `Conflict` type or per-port write-arbitration policy.

## Tests to add

For each fix, add multi-update reconcile tests under the relevant
`Test<Type>` class in `test_matrix.py`, mirroring `TestListReconcile`. The
shape:

- Single-update sanity for each form.
- Multi-update batches for each pair: structural × structural, structural ×
  plain, plain × plain.
- End-to-end `reconcile → apply` to verify "reset first, then all
  contributions, irrespective of receipt order."
- Receipt-order-invariance: same updates in reverse produce the same final
  state.

The Set / Tree / Array / Tuple fixes should each land with their own
`Test<Type>Reconcile` class, sized similarly to `TestListReconcile`.

## Cross-cutting principle

The List bug, the Array `set`-vs-delta crash, and the (hypothetical) Map
`_remove: 'all'` design all fit one pattern:

> A "reset" sentinel and concurrent "addition" updates in the same batch
> compose as: reset clears pre-existing state; all additions in the batch
> are retained.

This rule is type-agnostic. It's worth promoting to an explicit named
combinator (or a base mixin in the dispatch hierarchy) so each new
container type that wants reset semantics inherits the right composition
behavior automatically. That's a thread to pick up in the composite-algebra
doc.
