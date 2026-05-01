# Method API for bigraph-schema Container Types

## Problem

Updates to container-typed state carry implicit *operations*, encoded as
string-keyed sentinels mixed into the update dict:

- `{'_add': {k: v}}` — insert entries into a map.
- `{'_remove': [k]}` — delete entries from a map.
- `{'_divide': {...}}` — replace a "mother" node with "daughter" nodes.
- `{'_react': {...}}` — tree-rewrite rule.

Each is handled ad-hoc by the relevant type's `apply()` branch. There is
no shared contract, no uniform discoverability, no registration point,
and no error-of-omission when a type doesn't implement a sentinel (it
just silently passes through as if the key were a data key).

These sentinels are an **unrealized method API**: each one is
conceptually a named operation on a container type, and the apply path
already dispatches to them based on their presence in the update. What's
missing is making that dispatch first-class.

## Goal

Treat container-type operations as registered, multimethod-dispatched
peers to `apply` / `default` / `serialize` / `render` / `divide` /
`bundle`. Every container type declares which methods it supports; an
update dict carrying a method sentinel dispatches to the type's
registered implementation; types that don't support a method raise an
informative error.

The existing sentinel semantics become the spec for each method.
Nothing new is invented about what `_add` *means* — this spec only
formalizes *how it's looked up and dispatched*.

---

## 1. Method namespace

A **method** is a named operation on a container type. Methods are
identified by a name without the leading underscore:

| Update sentinel | Method name | Applies to                |
|-----------------|-------------|---------------------------|
| `_add`          | `add`       | Map, Tree, List (append)  |
| `_remove`       | `remove`    | Map, Tree, List (by index)|
| `_divide`       | `divide`    | Tree, Map (container of agents) |
| `_react`        | `react`     | Tree                       |

The leading underscore remains the on-the-wire convention in update
dicts (to distinguish from data keys). The method registry key is the
bare name.

New method names are reserved by adding them to the registry; there
is no magic recognition of arbitrary `_foo` keys.

---

## 2. Dispatch mechanism

Each method is a plum-dispatched multimethod in
`bigraph_schema.methods`, parallel to the existing
`apply` / `default` / `serialize` / ... modules:

```python
# bigraph_schema/methods/add.py

from plum import dispatch
from bigraph_schema.schema import Map, Tree, List

@dispatch
def add(schema: Map, state, args, path):
    """Insert entries into a map-typed state.

    ``args`` is a dict {key: value} or a list of (key, value) tuples;
    each entry is added to ``state``, overwriting on collision.
    Returns ``(new_state, merges)``.
    """
    if isinstance(args, dict):
        for k, v in args.items():
            state[k] = v
    elif isinstance(args, list):
        for k, v in args:
            state[k] = v
    return state, []

@dispatch
def add(schema: List, state, args, path):
    """Append items to a list-typed state."""
    if not isinstance(args, list):
        args = [args]
    state.extend(args)
    return state, []

# No @dispatch for Atom, Float, etc. — calling add on a scalar raises
# the standard plum NotFoundLookupError, which the apply router turns
# into a readable error.
```

`add`, `remove`, `divide`, `react` each become a top-level multimethod.
Adding a new method = adding a new module; supporting a new type for an
existing method = adding a new `@dispatch` branch.

---

## 3. Apply router

`apply()` remains the single entry point for all update processing. It
consults a **method table** (registered once) that maps sentinel keys
to method functions, and routes in a pre-defined order:

```python
# bigraph_schema/methods/apply.py

METHOD_SENTINELS = [
    ('_add', add),
    ('_remove', remove),
    ('_divide', divide),
    ('_react', react),
]

@dispatch
def apply(schema: Map, state, update, path):
    # ... existing normalizations (state defaults, None-update) ...
    if isinstance(update, dict):
        rest = update
        for sentinel, method in METHOD_SENTINELS:
            if sentinel in rest:
                state, submerges = method(schema, state, rest[sentinel], path)
                merges += submerges
                rest = {k: v for k, v in rest.items() if k != sentinel}
        if rest:
            # Normal key-by-key apply over the remaining data keys.
            state, submerges = _default_map_apply(schema, state, rest, path)
            merges += submerges
    return state, merges
```

The router is type-agnostic — every container type's `apply` uses the
same sentinel table. Method dispatch then specializes on the schema
type. Error-of-omission becomes a clear `NotFoundLookupError` from
plum (e.g. *"no `add` implementation for schema type `Atom`"*) rather
than silent passthrough.

### Ordering

Methods run in the order of `METHOD_SENTINELS`. `_divide` runs before
`_add` so that "remove mother, install daughters, then add something to
the remaining state" is expressible in a single update. `_remove` runs
after `_add` so `{'_add': {k: v}, '_remove': [k]}` drops `k` (last
write wins, symmetric with other apply rules). Ordering can be revised
per sentinel if a concrete case requires it.

---

## 4. Composability

Methods compose through the same recursive apply path as regular data
keys. Consider `tree[map[float]]` receiving:

```python
{
    'leaf_a': {'_add': {'new_key': 1.0}},
    'leaf_b': {'existing_key': 2.5},
}
```

- Tree's apply walks top-level keys (`leaf_a`, `leaf_b`).
- At `leaf_a`, the update is `{'_add': ...}` — map's apply sees the
  `_add` sentinel, routes to `add(Map, ...)`, inserts.
- At `leaf_b`, the update is `{'existing_key': 2.5}` — map's apply
  hands to the value schema's apply per key.

No special code for nesting — the sentinel check runs at each level
because every container type's apply consults the method table.

---

## 5. Declaring method support per type

A type's supported methods are implicit in which `@dispatch` branches
exist for it. `core.supported_methods(schema)` (new helper) returns the
list of registered methods for introspection:

```python
def supported_methods(schema):
    from bigraph_schema.methods import add, remove, divide, react
    methods = []
    for name, fn in [('add', add), ('remove', remove),
                     ('divide', divide), ('react', react)]:
        try:
            fn.resolve_method((type(schema),))
            methods.append(name)
        except NotFoundLookupError:
            pass
    return methods
```

This is a pure introspection helper; no static declaration required.

---

## 6. Per-type semantics (initial contract)

### Map

| Method   | Args                                  | Semantics                                        |
|----------|---------------------------------------|--------------------------------------------------|
| `add`    | `{k: v}` or `[(k, v), ...]`           | Insert; overwrite on collision.                   |
| `remove` | `[k, ...]`                            | Delete; ignore missing.                           |
| `divide` | `{mother, daughters: [{id, state}]}` | Container-of-agents division (existing semantics).|

### List

| Method   | Args           | Semantics                            |
|----------|----------------|--------------------------------------|
| `add`    | `v` or `[v...]` | Append at end.                      |
| `remove` | `[i, ...]` (indices) | Delete by index; descending order internally. |

### Tree

| Method   | Args               | Semantics                                     |
|----------|--------------------|-----------------------------------------------|
| `add`    | `{path: value}`    | Create sub-tree at each path.                  |
| `remove` | `[path, ...]`      | Delete sub-tree at each path.                  |
| `divide` | as in Map (at node) | Mother/daughter replacement at branch.         |
| `react`  | `{redex, reactum}`  | Existing tree-rewrite semantics.              |

### Atom / Float / Integer / String / Boolean

No methods. Calling `add`/`remove`/`divide`/`react` on a scalar
raises — these sentinels shouldn't appear in scalar updates.

---

## 7. Migration from current sentinel handling

Per existing sentinel (`_add`, `_remove`, `_divide`, `_react`):

1. Create `bigraph_schema/methods/<name>.py` with the top-level
   multimethod stub.
2. Copy the existing inline handling from each type's `apply` branch
   into a `@dispatch def <name>(schema: <Type>, ...)` implementation.
3. Replace the inline branch in `apply` with a call into the method
   router (or, for stepwise migration, keep the inline branch as a
   fast path and let the router be a fallback).
4. Add a case to the introspection tests.

Each sentinel is an independent migration; the existing apply paths
keep working throughout.

---

## 8. Non-goals

- **Inventing new methods.** This spec doesn't propose new operations;
  it formalizes the ones that already exist implicitly.
- **Changing update wire format.** Update dicts still use `_add`,
  `_remove`, etc. — the underscore prefix stays as the convention.
- **Making every method universal.** Not all types support every
  method; raising on unsupported calls is the intended behavior.
- **Replacing `apply`.** Apply is still the single public entry point;
  the method API is internal organization.

---

## 9. Open questions

1. **Method composition with data keys.** An update like
   `{'a': 1, '_add': {'b': 2}}` updates an existing `a` *and* adds a
   new `b`. Current behavior: both happen. Keep that.
2. **Recursion into method args.** When `_add` carries a map of
   values, should those values be recursively applied through their
   value-schema's apply? Current behavior: no (they're installed
   verbatim). Probably fine, but worth noting explicitly in the
   contract.
3. **List `_remove` semantics.** Remove by index, by value, or both?
   Current behavior is inconsistent across the codebase. Pick one.
4. **User-defined methods.** Should application code be able to
   register custom methods and sentinel keys? Most projects won't
   need it; defer until a concrete use case appears.
