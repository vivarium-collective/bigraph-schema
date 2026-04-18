"""
Bigraph algebraic assembly
==========================

The existing schema dict structure IS a bigraph: dict nesting is the
place graph, Link nodes with ports are the link graph, the type
registry is the signature. This module provides the categorical
operations that exploit that structure.

Schemas are already bigraphs — we don't wrap them. What we add:

  - ``interfaces(schema)`` — derive the inner and outer faces by
    walking for Sites (place-graph holes) and unwired Link ports
    (link-graph open names).
  - ``compose(outer, inner)`` — substitute ``inner``'s roots into
    ``outer``'s Sites, wire matching port names. (Milner Def. 2.5.)
  - ``tensor(a, b)`` — side-by-side merge of schemas with disjoint
    keys. (Milner Def. 2.7.)
  - Elementary bigraph constructors — ``barren()``, ``merge(n)``,
    ``ion()``, ``substitution()``, ``closure()``. (Milner Defs. 3.1–3.5.)

See ``.claude/plans/milner-formalism.md`` for the full design rationale
and Milner, *Space and Motion of Communicating Agents* (2008) for the
formal definitions.
"""

import copy
from bigraph_schema.schema import (
    Node, Empty, Site, Interface, Link, Wires, Path, Place,
    Map, List, Set, Tree, Tuple, Wrap, Union,
    is_schema_field)


EPSILON = Interface()
"""The trivial interface ``ε = ⟨0, ∅⟩``. Unit of tensor product,
domain of every ground bigraph."""


# ── Interface derivation ────────────────────────────────────────────


def _wired_ports(wires_field):
    """Return the set of port names that have explicit wire paths.

    At the schema level, a ``Wires`` Node means "no ports wired yet"
    (the default). A dict means some or all ports have been explicitly
    connected. Only dict keys count as wired.
    """
    if isinstance(wires_field, dict):
        return set(wires_field.keys())
    return set()


def interfaces(schema):
    """Derive the inner and outer faces of a schema-as-bigraph.

    Walks the schema tree and finds:

    - **Sites** → inner face places (holes in the place graph that
      composition fills with roots from another bigraph).
    - **Unwired Link input ports** → inner names (link-graph endpoints
      facing inward, awaiting connection from outside).
    - **Unwired Link output ports** → outer names (link-graph endpoints
      facing outward, exposed for composition or observation).
    - **Top-level dict keys** → outer face roots (the regions of the
      bigraph visible from above).

    A schema with no Sites and all ports wired is **ground**
    (``inner == EPSILON``) — the shape every current schema has.

    Returns ``(inner, outer)`` where each is an ``Interface``.

    ``inner._places`` is a tuple of ``(path, Site)`` pairs recording
    where each hole lives in the tree. ``outer._places`` is a tuple of
    root key strings. ``_names`` maps port name → path of the Link
    that owns it, so ``compose`` knows where to create wires.
    """

    sites = []
    inner_names = {}
    outer_names = {}

    def walk(node, path):
        if isinstance(node, Site):
            sites.append((path, node))

        elif isinstance(node, Link):
            # Extract open port names from the link graph.
            wired_in = _wired_ports(node.inputs)
            wired_out = _wired_ports(node.outputs)
            if isinstance(node._inputs, dict):
                for port in node._inputs:
                    if port not in wired_in:
                        inner_names[port] = path
            if isinstance(node._outputs, dict):
                for port in node._outputs:
                    if port not in wired_out:
                        outer_names[port] = path

        elif isinstance(node, (Wires, Path)):
            # Link-graph wiring structure — not place-graph children.
            pass

        elif isinstance(node, Wrap):
            # Maybe, Overwrite, Const, etc. — unwrap and continue.
            walk(node._value, path)

        elif isinstance(node, Union):
            for option in node._options:
                walk(option, path)

        elif isinstance(node, Tuple):
            for i, value_schema in enumerate(node._values):
                walk(value_schema, path + (str(i),))

        elif isinstance(node, Map):
            # Map children are dynamic — the value schema describes
            # every entry. A '*' path step means "any key".
            walk(node._value, path + ('*',))

        elif isinstance(node, (List, Set)):
            walk(node._element, path + ('*',))

        elif isinstance(node, Tree):
            walk(node._leaf, path + ('*',))

        elif isinstance(node, Place):
            if isinstance(node._subnodes, dict):
                for key, child in node._subnodes.items():
                    walk(child, path + (key,))

        elif isinstance(node, dict):
            for key, child in node.items():
                if isinstance(key, str) and not key.startswith('_'):
                    walk(child, path + (key,))

        # Atomic leaf nodes (Float, Integer, String, etc.) — stop.

    roots = []
    if isinstance(schema, dict):
        for key in schema:
            if isinstance(key, str) and not key.startswith('_'):
                roots.append(key)
                walk(schema[key], (key,))
    elif isinstance(schema, Node):
        walk(schema, ())

    inner = Interface(
        _places=tuple(sites),
        _names=inner_names)
    outer = Interface(
        _places=tuple(roots),
        _names=outer_names)

    return inner, outer


def is_ground(schema):
    """True if the schema is a ground bigraph — no Sites, all ports
    wired. Equivalent to ``inner == EPSILON``."""
    inner, _ = interfaces(schema)
    return inner._places == () and inner._names == {}


# ── Identity ────────────────────────────────────────────────────────


def identity(interface):
    """The identity arrow on ``interface``.

    For a ground schema (interface is EPSILON), this is just EPSILON
    itself — the trivial "do nothing" composition operand.
    """
    return interface


# ── Composition ─────────────────────────────────────────────────────


def compose(outer, inner):
    """Compose ``outer ∘ inner``: substitute ``inner``'s roots into
    ``outer``'s Sites, and wire matching port names.

    Milner Def. 2.5 (p. 17): ``G ∘ F : I → K`` requires the outer
    face of ``F`` equal the inner face of ``G``. The composite's inner
    face is ``F``'s inner face and its outer face is ``G``'s outer face.
    The mediating face ``J`` disappears.

    In our schema model:

    - **Place composition**: each ``Site`` in ``outer`` is replaced by
      the corresponding root (top-level key) from ``inner``. Sites
      are matched to roots by index (0th site ↔ 0th root, etc.).
    - **Link composition**: for each outer name of ``inner`` that
      matches an inner name of ``outer`` (by port name), create a
      wire connecting the inner Link's output port to the path that
      the outer Link's input port would read from.

    Currently handles: ground schemas (no Sites, all ports wired) and
    the identity cases. Raises ``NotImplementedError`` for cases not
    yet supported.
    """

    # Identity cases
    if isinstance(inner, Interface) and inner == EPSILON:
        return outer
    if isinstance(outer, Interface) and outer == EPSILON:
        return inner

    if not isinstance(outer, dict) or not isinstance(inner, dict):
        raise NotImplementedError(
            'compose currently operates on dict schemas')

    outer_inner, _ = interfaces(outer)
    _, inner_outer = interfaces(inner)

    # --- Place composition: substitute Sites ---
    site_list = list(outer_inner._places)  # [(path, Site), ...]
    root_keys = list(inner_outer._places)  # [key, ...]

    if len(site_list) != len(root_keys):
        raise ValueError(
            f'compose: outer has {len(site_list)} sites but inner has '
            f'{len(root_keys)} roots — faces must match')

    result = copy.deepcopy(outer)

    # Replace each site with the corresponding root from inner
    for (site_path, _site), root_key in zip(site_list, root_keys):
        if root_key not in inner:
            raise ValueError(
                f'compose: inner schema missing root {root_key!r}')
        filler = copy.deepcopy(inner[root_key])
        _set_at_path(result, site_path, filler)

    # --- Link composition: wire matching names ---
    # inner's outer names → ports on inner's Links whose outputs are
    # unwired. outer's inner names → ports on outer's Links whose
    # inputs are unwired. If port names match, create the connection.
    for port_name, inner_link_path in inner_outer._names.items():
        if port_name in outer_inner._names:
            outer_link_path = outer_inner._names[port_name]
            # Wire inner's output → the path that outer's input reads
            # from. For now: wire inner's output to the default path
            # (the port name itself, relative to the inner link).
            outer_link = _get_at_path(result, outer_link_path)
            if isinstance(outer_link, Link):
                if isinstance(outer_link.inputs, Wires):
                    outer_link.inputs = {}
                outer_link.inputs[port_name] = list(inner_link_path) + [port_name]

    return result


# ── Tensor product ──────────────────────────────────────────────────


def tensor(left, right):
    """Tensor product ``left ⊗ right``: place two schemas side by side.

    Milner Def. 2.7 (p. 18): juxtaposition of disjoint bigraphs.
    Interfaces concatenate, supports must be disjoint.

    In our schema model this is a dict merge — the two schemas must
    have disjoint top-level keys.
    """
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise NotImplementedError(
            'tensor currently operates on dict schemas')

    left_keys = {k for k in left if isinstance(k, str) and not k.startswith('_')}
    right_keys = {k for k in right if isinstance(k, str) and not k.startswith('_')}
    overlap = left_keys & right_keys
    if overlap:
        raise ValueError(
            f'tensor: schemas must have disjoint keys, but both '
            f'contain: {overlap}')

    return {**left, **right}


# ── Elementary bigraph constructors ─────────────────────────────────
# Milner Ch. 3, Defs. 3.1–3.5 (pp. 28–29).


def barren(key='region0'):
    """The barren root ``1 : 0 → 1``. One empty region, no sites,
    no nodes, no links. (Milner Def. 3.1, p. 28.)

    In our schema model: a dict with one key mapping to Empty.
    """
    return {key: Empty()}


def merge(n, root_key='region0'):
    """``merge_n : n → 1``. One root containing ``n`` sites.

    All ``n`` sites are placed under a single root, so composition
    with ``merge_n`` collapses ``n`` separate roots into one region.
    ``merge_0 = 1`` (the barren root). (Milner Def. 3.1, p. 28.)
    """
    if n == 0:
        return barren(root_key)
    return {root_key: {f'site{i}': Site() for i in range(n)}}


def ion(core, control, names=(), site_key='site0'):
    """Discrete ion ``K_⃗x : 1 → ⟨1, {⃗x}⟩``. A single node with
    control ``control``, ports linked to ``names``, and one site
    inside. (Milner Def. 3.4, p. 29.)

    In our schema model the K-node is a dict with two children:
    a ``Site`` (the place-graph hole) and a ``Link`` (the
    link-graph structure declaring ports). The ports are left
    unwired, making them outer names. This follows the existing
    convention where Links live as nodes in the place graph.

    Requires ``core`` to compile the Link.
    """
    outputs = {name: 'node' for name in names}
    return {control: {
        site_key: Site(),
        'link': core.access({
            '_type': 'link',
            '_inputs': {},
            '_outputs': outputs})}}


def substitution(core, outer_name, inner_names):
    """Elementary substitution ``y/X : X → {y}``. All inner names
    in ``X`` are mapped to a single outer name ``y``.
    (Milner Def. 3.2, p. 28.)

    In our model: a Link with input ports named by ``inner_names``
    (unwired → inner names) and one output port ``outer_name``
    (unwired → outer name).
    """
    inputs = {name: 'node' for name in inner_names}
    outputs = {outer_name: 'node'}
    return core.access({'sub': {
        '_type': 'link',
        '_inputs': inputs,
        '_outputs': outputs}})


def closure(core, name):
    """Elementary closure ``/x : {x} → ε``. Closes off inner name
    ``x`` — the name ceases to be visible from outside.
    (Milner Def. 3.2, p. 28.)

    In our model: a Link with one input port ``name`` (unwired →
    inner name) and no output ports (nothing exposed).
    """
    return core.access({'close': {
        '_type': 'link',
        '_inputs': {name: 'node'},
        '_outputs': {}}})


# ── Helpers ─────────────────────────────────────────────────────────


def _get_at_path(tree, path):
    """Navigate a nested dict by a tuple path."""
    node = tree
    for step in path:
        if isinstance(node, dict):
            node = node[step]
        elif hasattr(node, step):
            node = getattr(node, step)
        else:
            raise KeyError(f'cannot navigate to {step!r} in {type(node)}')
    return node


def _set_at_path(tree, path, value):
    """Set a value in a nested dict at the given path."""
    if len(path) == 0:
        return value
    parent = _get_at_path(tree, path[:-1])
    key = path[-1]
    if isinstance(parent, dict):
        parent[key] = value
    elif hasattr(parent, key):
        setattr(parent, key, value)
    else:
        raise KeyError(f'cannot set {key!r} on {type(parent)}')
    return tree
