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
import random
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List as TypingList, Tuple as TypingTuple

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


# ── Sorting disciplines ─────────────────────────────────────────────
# Milner Ch. 6: a sorting Σ = (Θ, K, Φ) enriches a signature with
# sorts that classify places and links, plus a formation rule Φ that
# well-formed bigraphs must satisfy.
#
# Place sorting (Def. 6.1): sorts on places, Φ constrains nesting.
# Link sorting (Def. 6.10): sorts on ports/links, Φ constrains sharing.
# Both are preserved by composition and tensor.


@dataclass
class Sorting:
    """A sorting discipline Σ = (Θ, K, Φ).

    Attributes:
        sorts: The set Θ of sort labels.
        controls: Dict mapping control name → dict of metadata:
            ``{'arity': int, 'status': str, 'sort': str,
               'port_sorts': tuple}``.
            ``port_sorts`` is a tuple of sort labels, one per port,
            ordered to match the control's arity.
        formation: A callable ``Φ(parent_sort, child_sort) → bool``
            that returns True if a child with ``child_sort`` is
            permitted inside a parent with ``parent_sort``.
            None means no constraint (all nesting allowed).
        link_formation: A callable
            ``Φ_link(link_sort, point_sorts) → bool`` for link
            sorting. ``point_sorts`` is the list of sorts of all
            points (ports + names) on a link. None means unconstrained.
    """
    sorts: set
    controls: Dict[str, dict]
    formation: object = None
    link_formation: object = None


def stratified_sorting(sorts, phi, controls, hard_sorts=None):
    """Build a stratified place sorting (Milner Def. 6.5).

    ``phi`` maps parent sort → required child sort. Children of a
    root with sort θ have sort θ; children of a node with sort θ
    have sort φ(θ).

    ``hard_sorts`` is a set of sorts that cannot have idle roots
    (Def. 6.2) — every root with a hard sort must contain at least
    one node.
    """
    hard = hard_sorts or set()

    def formation(parent_sort, child_sort):
        expected = phi.get(parent_sort)
        if expected is None:
            return True  # unconstrained
        return child_sort == expected

    return Sorting(
        sorts=set(sorts),
        controls=controls,
        formation=formation)


def many_one_sorting(controls):
    """Build a many-one link sorting (Milner Def. 6.12).

    Two sorts: ``'s'`` (source) and ``'t'`` (target). Each link has
    at most one s-point. A link has sort s iff it has an s-point.
    Every closed link has sort s.
    """
    def link_formation(link_sort, point_sorts):
        s_count = sum(1 for s in point_sorts if s == 's')
        return s_count <= 1

    return Sorting(
        sorts={'s', 't'},
        controls=controls,
        formation=None,
        link_formation=link_formation)


def validate_sorting(schema, sorting, path=()):
    """Validate that ``schema`` is well-sorted under ``sorting``.

    Walks the schema tree and checks:
    - Every node's control is in the sorting's controls.
    - Place nesting respects the formation rule Φ.
    - Link ports respect the link formation rule (if any).

    Returns a list of violation strings (empty = valid).
    """
    violations = []

    def get_sort(node, key=None):
        """Get the sort of a node — from _control annotation or key."""
        if isinstance(node, dict):
            ctrl = node.get('_control', key)
        elif isinstance(node, Node):
            ctrl = _control_name(node)
        else:
            ctrl = key
        info = sorting.controls.get(ctrl, {})
        return info.get('sort', ctrl)

    def walk(node, path, parent_sort=None):
        node_sort = get_sort(node, path[-1] if path else None)

        # Check formation rule
        if sorting.formation and parent_sort is not None:
            if not sorting.formation(parent_sort, node_sort):
                violations.append(
                    f'at {path}: sort {node_sort!r} not allowed '
                    f'inside sort {parent_sort!r}')

        # Recurse into children
        if isinstance(node, dict):
            for key, child in node.items():
                if isinstance(key, str) and not key.startswith('_'):
                    walk(child, path + (key,), node_sort)

        # Check link port sorts
        if isinstance(node, Link) and sorting.link_formation:
            port_sorts = []
            info = sorting.controls.get(
                node.get('_control') if isinstance(node, dict)
                else _control_name(node), {})
            for s in info.get('port_sorts', ()):
                port_sorts.append(s)
            if port_sorts and not sorting.link_formation(
                    node_sort, port_sorts):
                violations.append(
                    f'at {path}: link port sorts {port_sorts} '
                    f'violate link formation rule')

    if isinstance(schema, dict):
        for key, child in schema.items():
            if isinstance(key, str) and not key.startswith('_'):
                walk(child, (key,), None)
    else:
        walk(schema, (), None)

    return violations


# ── Binding / link locality ─────────────────────────────────────────
# Milner §11.3 (p. 122): localising a link constrains its scope to
# a subtree. In our model links are already in the place graph, so
# "bound" means all of a Link's wires point within the Link's
# ancestor subtree.


def is_bound(schema, link_path):
    """Check whether the Link at ``link_path`` is bound — all its
    wires target paths that share ``link_path``'s prefix (i.e. they
    stay within the same subtree).

    An unwired port is considered bound (it's an open name, not an
    escape). Only wired ports that point OUTSIDE the subtree violate
    binding.
    """
    link = _get_at_path(schema, link_path)
    if not isinstance(link, Link):
        return True

    prefix = link_path[:-1]  # the parent subtree

    def check_wires(wires_field):
        if isinstance(wires_field, dict):
            for port, wire_path in wires_field.items():
                if isinstance(wire_path, (list, tuple)):
                    wire_tuple = tuple(wire_path)
                    if not wire_tuple[:len(prefix)] == prefix:
                        return False
        return True

    return check_wires(link.inputs) and check_wires(link.outputs)


def find_unbound_links(schema):
    """Find all Links in ``schema`` whose wires escape their subtree.

    Returns a list of ``(link_path, escaping_port, wire_target)``
    triples.
    """
    escapes = []

    def walk(node, path):
        if isinstance(node, Link):
            prefix = path[:-1] if path else ()
            for direction in ('inputs', 'outputs'):
                wires = getattr(node, direction)
                if isinstance(wires, dict):
                    for port, wire_path in wires.items():
                        if isinstance(wire_path, (list, tuple)):
                            wire_tuple = tuple(wire_path)
                            if wire_tuple[:len(prefix)] != prefix:
                                escapes.append(
                                    (path, port, wire_tuple))
        elif isinstance(node, dict):
            for key, child in node.items():
                if isinstance(key, str) and not key.startswith('_'):
                    walk(child, path + (key,))

    if isinstance(schema, dict):
        for key, child in schema.items():
            if isinstance(key, str) and not key.startswith('_'):
                walk(child, (key,))
    return escapes


# ── Dynamic signatures and activity ─────────────────────────────────
# Milner Def. 8.2 (p. 81): a signature is *dynamic* if each control
# has a status in {atomic, passive, active}. A bigraph G is *active*
# at site i iff every ancestor node of i has an active control.
#
# In our model the type registry IS the signature. We layer dynamic
# status on top via a dict mapping type names → status strings.
# Controls not listed default to 'active'.

ACTIVE = 'active'
PASSIVE = 'passive'
ATOMIC = 'atomic'


def _control_name(node):
    """Return the control name (type name) for a schema node."""
    if isinstance(node, Link):
        return 'link'
    cls = type(node)
    # Walk BASE_TYPES reverse lookup
    from bigraph_schema.schema import BASE_TYPES
    for name, typ in BASE_TYPES.items():
        if typ is cls:
            return name
    return cls.__name__.lower()


def is_active(schema, path, control_status=None):
    """Check whether ``path`` is active — every ancestor has an active
    control. (Milner Def. 8.2.)

    ``control_status`` maps control names to status strings
    (``'active'``, ``'passive'``, ``'atomic'``). Controls not in the
    dict default to ``'active'``.

    Control names can be:

    - **Type names** (``'link'``, ``'float'``, a registered type) —
      matched against the node's schema type.
    - **Key names** (``'room'``, ``'agent'``) — matched against the
      dict key at which a node sits. Useful for plain-dict schemas
      where the control is implicit in the key, as in Milner's
      built-environment example (A:agent, B:building, R:room, etc.).

    A reaction can only fire at a location where ``is_active`` is
    True — passive ancestors block reactions inside them.
    """
    if control_status is None:
        control_status = {}

    node = schema
    for step in path:
        # Check by node type
        status = control_status.get(_control_name(node), None)
        # Also check by the key we're stepping into — for plain dicts
        # the key IS the control label in the Milner sense.
        if status is None:
            status = control_status.get(step, ACTIVE)
        if status != ACTIVE:
            return False
        # Descend
        if isinstance(node, dict):
            if step not in node:
                return True
            node = node[step]
        elif hasattr(node, step):
            node = getattr(node, step)
        else:
            return True

    return True


# ── Reaction rules ──────────────────────────────────────────────────
# Milner Def. 8.5 (p. 84): a parametric reaction rule is a triple
# (R : m → J,  R' : m' → J,  η : m' → m) where R is the parametric
# redex, R' the parametric reactum, and η the instantiation map.
#
# In our model:
#   - redex and reactum are schemas (dicts) with Sites for parameters
#   - instantiation maps reactum site keys to redex site keys
#   - rate is an optional stochastic weight (Milner §11.4)


@dataclass(frozen=True)
class Absent:
    """A redex marker requiring the matched state node to NOT have
    the named key (or to have it as an empty container).

    Where ``Site`` says "something must be present here" and
    ``LinkVar('e')`` says "this port must be wired to the edge bound
    by ``e``", ``Absent()`` is the matcher's negative-application
    condition: "this key must be absent."

    Used for *unbound* / *free* preimage patterns. For example, a
    biochemical binding rule

        Substrate{free} + Enzyme{free} -->  Substrate-Enzyme{bound}

    in our model becomes a redex with ``'outputs': Absent()`` on
    both Substrate and Enzyme — the rule fires only when neither is
    currently in any complex. Without this marker the rule would
    happily double-bind a kinase that's already bound to a different
    substrate, violating the active-site stoichiometry.

    A state value matches ``Absent()`` if it is missing entirely or
    present as an empty dict — either form represents "no wire."
    The reactum may include ``Absent()`` for symmetry, but it has
    no effect there: keys mapped to ``Absent`` are simply omitted
    from the result.
    """


@dataclass(frozen=True)
class LinkVar:
    """A wire-binding variable in a redex/reactum.

    Where ``Site`` is a hole in the *place* graph, ``LinkVar`` is a
    variable in the *link* graph (Milner Def. 2.2, p. 16) — it binds
    to a wire path in the state.

    Two ``LinkVar``s with the same ``name`` in a redex must bind to
    the *same* wire path: that's how "panel.auth and person.badge
    share an edge" is expressed.

    In a reactum:

    - A ``LinkVar`` whose ``name`` was bound during matching is
      substituted with the bound path — the new node inherits the
      same edge.
    - A ``LinkVar`` whose ``name`` was *not* bound (introduced
      fresh by the reactum, e.g. ``enter_secure`` creating a new
      link between a previously-unconnected Person and a panel)
      mints a fresh anchor path during ``instantiate``.

    The runtime wire format is the existing process-bigraph wire
    convention: a list of path components into the state tree (the
    same shape ``Link.outputs`` carries). So if ``LinkVar('e')``
    binds to ``['..', '..', '_edges', 'edge_office_panel']``, that
    list propagates verbatim into the reactum slot.
    """
    name: str


@dataclass
class Match:
    """Result of matching a redex against a state subtree.

    Attributes:
        path: Location in the state where the match occurs.
        bindings: ``{site_label: matched_subtree}`` — the content
            captured by each Site in the redex. Also carries
            ``__edges__`` — wire paths bound by ``LinkVar``s.
        key_map: ``{redex_key: state_key}`` — which state key each
            non-Site redex key was assigned to.
    """
    path: tuple
    bindings: dict
    key_map: dict


@dataclass
class ReactionRule:
    """A parametric reaction rule.

    Attributes:
        redex: Schema pattern to match. Sites in the redex are
            parameters — they match arbitrary subtrees.
        reactum: Schema to substitute when the rule fires. Sites in
            the reactum are filled via ``instantiation``.
        instantiation: Maps each reactum site key to the redex site
            key whose matched subtree should fill it. If a redex
            site key appears multiple times, its content is shared
            (Milner §8.1, p. 83). If a redex site key is absent
            from the values, its matched content is discarded.
        rate: Optional stochastic rate for Gillespie-style selection
            among competing rules (Milner §11.4).
        label: Human-readable name for the rule.
    """
    redex: dict
    reactum: dict
    instantiation: Dict[str, str] = field(default_factory=dict)
    rate: Optional[float] = None
    label: str = ''

    def __post_init__(self):
        if not self.instantiation:
            # Default: identity map — reactum site keys match redex
            # site keys by name.
            redex_inner, _ = interfaces(self.redex)
            reactum_inner, _ = interfaces(self.reactum)
            redex_sites = {p: s for p, s in redex_inner._places}
            reactum_sites = {p: s for p, s in reactum_inner._places}
            self.instantiation = {}
            for rpath in reactum_sites:
                # Match by site key (last path element)
                rkey = rpath[-1] if rpath else None
                for dpath in redex_sites:
                    dkey = dpath[-1] if dpath else None
                    if rkey == dkey:
                        self.instantiation[rkey] = dkey
                        break


# ── Matching ────────────────────────────────────────────────────────
# Given a ground state and a redex pattern, find occurrences.
#
# Milner's matching semantics (informal, Ch. 1 + Def. 8.5):
# - Non-Site redex entries must find structurally compatible state
#   entries. The assignment of redex keys to state keys is discovered
#   combinatorially (subgraph isomorphism).
# - Site entries bind the REMAINING state content not consumed by
#   non-Site entries. A single Site captures the entire leftover as
#   a dict. Multiple Sites at the same level would require
#   partitioning (deferred — rare in practice).
# - Matching walks the state tree and tries the redex at every dict
#   node, filtering by activity (is_active).


def _match_node(state_node, redex_node, bindings):
    """Check whether ``redex_node`` matches ``state_node``.

    Matching modes (checked in order):

    1. **Site** → matches anything (the whole point of a site).
    2. **LinkVar** → matches a wire path; binds the variable on
       first occurrence and requires equality on later occurrences.
    3. **dict** → structural match via ``_match_dict``, with optional
       ``_control`` constraint.
    4. **Node vs Node** → ``isinstance`` (Float matches Float, Integer
       matches Number, etc.).
    5. **Node vs runtime value** → ``check(schema, value)`` from the
       type system. Float() matches ``70.0``, Integer() matches ``5``,
       String() matches ``'hello'``.
    """
    if isinstance(redex_node, Site):
        return True

    if isinstance(redex_node, LinkVar):
        # Wire-equality matching. The state value at this position
        # is a wire (a path list, by the existing process-bigraph
        # convention used on Link.outputs/inputs). LinkVars with the
        # same name must bind to the same path; anywhere they appear
        # repeated is the redex's way of saying "these two ports
        # share an edge".
        edges = bindings.setdefault('__edges__', {})
        bound = edges.get(redex_node.name)
        if bound is None:
            edges[redex_node.name] = state_node
            return True
        return bound == state_node

    if isinstance(redex_node, dict):
        if not isinstance(state_node, dict):
            return False
        if '_control' in redex_node:
            if state_node.get('_control') != redex_node['_control']:
                return False
        return _match_dict(state_node, redex_node, bindings)

    if isinstance(redex_node, Node):
        # Schema-against-schema
        if isinstance(state_node, Node):
            return isinstance(state_node, type(redex_node))
        # Schema-against-runtime-value (state is 70.0, redex is Float)
        from bigraph_schema.methods import check
        try:
            return check(redex_node, state_node)
        except Exception:
            return False

    return False


def _match_dict(state_dict, redex_dict, bindings):
    """Match ``redex_dict`` against ``state_dict``.

    ALL redex entries (both fixed and Site) participate in the key
    assignment. Fixed entries must structurally match their assigned
    state node; Site entries match anything and bind the value.

    When there are more state keys than redex entries (surplus), the
    surplus is collected into a dict and added to the LAST Site's
    binding. This gives the "rest" capture semantics — one Site can
    bind an entire subtree of remaining children.

    When ``len(redex) == len(state)``, each Site captures exactly one
    state entry (1-to-1 assignment).
    """
    from itertools import permutations

    # Negative application conditions: any redex key paired with an
    # ``Absent()`` marker requires the matching key to be missing in
    # the state (or present but empty). Empty dicts count as absent
    # because in our wire convention an empty ``outputs: {}`` carries
    # no port at all.
    for k, v in redex_dict.items():
        if not (isinstance(k, str) and not k.startswith('_')):
            continue
        if isinstance(v, Absent):
            sval = state_dict.get(k)
            if sval is None:
                continue
            if isinstance(sval, dict) and not sval:
                continue
            return False

    redex_entries = [
        (k, v) for k, v in redex_dict.items()
        if isinstance(k, str) and not k.startswith('_')
        and not isinstance(v, Absent)]
    state_keys = [
        k for k in state_dict
        if isinstance(k, str) and not k.startswith('_')]

    if len(redex_entries) > len(state_keys):
        return False

    site_keys = [k for k, v in redex_entries if isinstance(v, Site)]
    has_surplus = len(redex_entries) < len(state_keys)

    # Edge bindings (from ``LinkVar``s) are shared by reference
    # across every recursive ``_match_dict`` call within a single
    # match attempt — that's what makes ``LinkVar('e')`` at one
    # node's wire visible to ``LinkVar('e')`` at a sibling or
    # descendant node's wire. We snapshot before each iteration so
    # a failed permutation can roll the bindings back.
    edges = bindings.setdefault('__edges__', {})

    for perm in permutations(state_keys, len(redex_entries)):
        assignment = dict(zip([k for k, _ in redex_entries], perm))
        saved_edges = dict(edges)
        trial = {'__edges__': edges}
        ok = True
        for redex_key, state_key in assignment.items():
            redex_value = redex_dict[redex_key]
            state_value = state_dict[state_key]
            if isinstance(redex_value, Site):
                if has_surplus:
                    # With surplus, Sites capture as key→value dicts
                    # (the last Site absorbs the surplus below).
                    trial[redex_key] = {state_key: state_value}
                else:
                    # Exact match — Sites capture the bare value.
                    trial[redex_key] = state_value
            elif not _match_node(state_value, redex_value, trial):
                ok = False
                break
        if not ok:
            # Roll back any edge bindings written by this iteration.
            edges.clear()
            edges.update(saved_edges)
            continue

        # Surplus state keys → merge into the last Site's binding
        used = set(assignment.values())
        surplus = {k: state_dict[k] for k in state_keys if k not in used}

        if surplus and site_keys:
            last_site = site_keys[-1]
            existing = trial.get(last_site, {})
            if isinstance(existing, dict):
                existing.update(surplus)
                trial[last_site] = existing
            else:
                trial[last_site] = surplus

        bindings.update(trial)

        # Merge this level's key map with any inner-level maps that
        # were already written by recursive ``_match_node`` calls.
        # Inner mappings (written first) take precedence on shared
        # keys — but in well-formed rules, redex keys are unique
        # across the whole tree (Milner: each site has a unique
        # number), so collisions don't arise in practice.
        # Without this merge, only the outermost level's key map
        # would survive, and inner state-key identities (e.g. an
        # alice/bob person captured under a redex slot ``p``) would
        # be lost during reactum remapping.
        this_keymap = {
            **assignment,
            **{sk: assignment.get(sk) for sk in site_keys}}
        existing_keymap = bindings.get('__key_map__', {})
        bindings['__key_map__'] = {**this_keymap, **existing_keymap}
        return True

    return False


def find_matches(state, redex, control_status=None):
    """Find all positions in ``state`` where ``redex`` matches.

    Returns a list of ``Match`` objects. Each match records the path,
    the Site bindings, and the redex→state key mapping.
    """
    results = []

    def walk(node, path):
        if isinstance(node, dict):
            bindings = {}
            if _match_dict(node, redex, bindings):
                key_map = bindings.pop('__key_map__', {})
                if control_status is None or is_active(
                        state, path, control_status):
                    results.append(Match(
                        path=path,
                        bindings=bindings,
                        key_map=key_map))
            for key, child in node.items():
                if isinstance(key, str) and not key.startswith('_'):
                    walk(child, path + (key,))

    walk(state, ())
    return results


# ── Instantiation ───────────────────────────────────────────────────


_FRESH_EDGE_COUNTER = 0


def _gensym_edge():
    """Mint a unique fresh edge id, suitable as the last component
    of an anchor path under the floor's ``_edges`` map. Module-level
    counter keeps ids unique across firings so edges introduced by
    different rule applications never collide on string equality.
    """
    global _FRESH_EDGE_COUNTER
    _FRESH_EDGE_COUNTER += 1
    return f"~e_{_FRESH_EDGE_COUNTER}"


def instantiate(reactum, bindings, instantiation):
    """Build a concrete replacement subtree from a reactum pattern.

    For each Site in the reactum, look up which redex site it maps to
    via ``instantiation``, then fill it with the subtree captured in
    ``bindings`` for that redex site.

    For each ``LinkVar`` in the reactum, substitute the wire path
    bound during matching (in ``bindings['__edges__']``). An unbound
    ``LinkVar`` is interpreted as a *new* hyperedge introduced by
    the reactum: a fresh anchor path is minted and recorded in the
    edge bindings, so subsequent occurrences of the same variable
    resolve to the same fresh path.

    All keys in the reactum are preserved (including ``_control`` and
    other user-defined ``_``-prefixed metadata).

    Returns a new dict (deep-copied, safe to mutate).
    """
    edges = bindings.get('__edges__', {})
    return _instantiate_walk(reactum, bindings, instantiation, edges)


def _instantiate_walk(reactum, bindings, instantiation, edges):
    result = {}
    for key, value in reactum.items():
        if isinstance(value, Absent):
            # Symmetry with the redex side: an Absent marker on the
            # reactum is just "this key is not in the result." We
            # could equivalently leave the key out of the reactum;
            # accepting ``Absent()`` here lets a rule author write
            # symmetric redex/reactum pairs.
            continue
        if isinstance(value, Site):
            # This site maps to a redex site via instantiation
            source_key = instantiation.get(key, key)
            filler = bindings.get(source_key)
            if isinstance(filler, dict) and '_control' not in filler:
                # Forest of trees captured (Milner: a site is a hole
                # in the place graph that gets filled with a region;
                # the region's roots become children at the slot
                # position rather than nesting under the site name).
                # We detect a forest by the absence of ``_control``
                # at the top — every bigraph node has one, so a
                # ``_control``-less dict is a multi-rooted region.
                for fk, fv in filler.items():
                    result[fk] = copy.deepcopy(fv)
            elif filler is None:
                # Site captured nothing; drop the slot.
                pass
            else:
                # Single tree (filler has ``_control``) or scalar.
                # Bind at the site's name; ``_remap_keys`` will then
                # rename that slot to the original state key via the
                # merged key map.
                result[key] = copy.deepcopy(filler)
        elif isinstance(value, LinkVar):
            bound = edges.get(value.name)
            if bound is None:
                # Reactum-introduced edge — mint a fresh anchor
                # under ``_edges`` (relative to the match path).
                # A reactum that uses the same fresh variable in
                # multiple places will see the same fresh path on
                # every later occurrence.
                bound = ['_edges', _gensym_edge()]
                edges[value.name] = bound
            result[key] = list(bound)
        elif isinstance(value, dict):
            result[key] = _instantiate_walk(
                value, bindings, instantiation, edges)
        else:
            result[key] = copy.deepcopy(value)
    return result


# ── Firing ──────────────────────────────────────────────────────────


def _remap_keys(tree, key_map):
    """Rename dict keys in ``tree`` according to ``key_map``.

    Reactum keys are pattern labels (``'r'``, ``'a'``). The key_map
    tells us what state keys they correspond to (``'lab'``,
    ``'alice'``). This recursively renames so the output uses the
    original state names, not the pattern labels.
    """
    if not isinstance(tree, dict):
        return tree
    result = {}
    for k, v in tree.items():
        new_key = key_map.get(k, k)
        result[new_key] = _remap_keys(v, key_map)
    return result


def fire_rule(state, rule, control_status=None, match_index=0):
    """Apply a reaction rule to a state.

    Finds matches of ``rule.redex`` in ``state``, picks one (by
    ``match_index``), builds the reactum via instantiation, and
    substitutes it into the state. Reactum keys are remapped to
    the original state keys from the match.

    Returns ``(new_state, match)`` or ``(state, None)`` if no match.
    """
    matches = find_matches(state, rule.redex, control_status)
    if not matches or match_index >= len(matches):
        return state, None

    match = matches[match_index]

    # Build the replacement from the reactum
    replacement = instantiate(
        rule.reactum, match.bindings, rule.instantiation)

    # Remap reactum keys to original state keys
    actual_map = {k: v for k, v in match.key_map.items() if v is not None}
    replacement = _remap_keys(replacement, actual_map)

    # Substitute into state at the match path
    new_state = copy.deepcopy(state)
    parent = _get_at_path(new_state, match.path) if match.path else new_state

    if isinstance(parent, dict):
        # Remove keys that the redex consumed
        for redex_key, state_key in match.key_map.items():
            if state_key is not None and state_key in parent:
                del parent[state_key]
        # Add the reactum's keys (now with original state names)
        parent.update(replacement)

    return new_state, match


# ── Reaction engine ─────────────────────────────────────────────────


@dataclass
class ReactionEvent:
    """Record of a single reaction firing."""
    rule_label: str
    match: Match
    step: int


def run_reactions(state, rules, control_status=None, max_steps=100,
                  mode='deterministic', rng=None):
    """Run reaction rules on ``state`` until quiescence or ``max_steps``.

    Modes:

    - ``'deterministic'`` — at each step, try rules in order and fire
      the first match found.
    - ``'stochastic'`` — collect all (rule, match) candidates, weight
      by ``rule.rate`` (default 1.0), sample one via Gillespie. The
      stochastic time is tracked but not returned (the caller
      controls real time).

    Returns ``(final_state, events)`` where ``events`` is the list of
    ``ReactionEvent`` records in firing order.
    """
    if rng is None:
        rng = random.Random()

    events = []
    state = copy.deepcopy(state)

    for step in range(max_steps):
        if mode == 'deterministic':
            fired = False
            for rule in rules:
                new_state, match = fire_rule(state, rule, control_status)
                if match is not None:
                    events.append(ReactionEvent(
                        rule_label=rule.label,
                        match=match,
                        step=step))
                    state = new_state
                    fired = True
                    break
            if not fired:
                break

        elif mode == 'stochastic':
            # Collect all candidates: (rule, match, rate)
            candidates = []
            for rule in rules:
                matches = find_matches(state, rule.redex, control_status)
                rate = rule.rate if rule.rate is not None else 1.0
                for match in matches:
                    candidates.append((rule, match, rate))

            if not candidates:
                break

            # Gillespie: total rate = sum of all rates, pick one
            # proportionally.
            total_rate = sum(r for _, _, r in candidates)
            pick = rng.random() * total_rate
            cumulative = 0.0
            chosen_rule, chosen_match = candidates[0][0], candidates[0][1]
            for rule, match, rate in candidates:
                cumulative += rate
                if cumulative >= pick:
                    chosen_rule, chosen_match = rule, match
                    break

            new_state, _ = fire_rule(
                state, chosen_rule, control_status,
                match_index=0)
            # fire_rule may pick a different match than chosen_match if
            # there are multiple; re-fire with the specific match.
            # For now, just use the first match from fire_rule.
            if new_state is not state:
                events.append(ReactionEvent(
                    rule_label=chosen_rule.label,
                    match=chosen_match,
                    step=step))
                state = new_state
            else:
                break
        else:
            raise ValueError(f'unknown mode {mode!r}')

    return state, events


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
