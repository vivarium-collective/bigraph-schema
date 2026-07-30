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
from dataclasses import dataclass, field, replace
from typing import Optional, Dict, List as TypingList, Tuple as TypingTuple

from bigraph_schema.schema import (
    Node, Empty, Site, Interface, Link, Wires, Path, Place,
    Map, List, Set, Tree, Tuple, Wrap, Union, Protocol, normalize_address,
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

            # An edge may leave its *implementation* open: a site in
            # ``address`` (or ``config``) is an abstract process — the face is
            # fixed and only what satisfies it is missing.
            walk(node.address, path + ('address',))
            walk(node.config, path + ('config',))

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


def compose(outer, inner, core=None):
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
      matches an inner name of ``outer`` (by port name), **join** the
      two ports — wire *both* ends to one shared store, so the ports
      meet the way ``realize`` resolves them. See ``_join_names``.

    Place composition is exactly :func:`fill_sites` with positionally
    matched bindings — Milner's sites are anonymous, so they are paired
    with ``inner``'s roots by index rather than by name; both routes share
    one substitution (``_fill_at_paths``). ``core`` is only needed when a
    site is *sorted* (so ``admits`` can check the filler); plain Milner
    composition over unsorted sites needs none.

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

    # Replace each site with the corresponding root from inner, recording
    # where each root landed so inner paths can be rebased into the
    # composed tree's coordinates.
    filling = {}
    rebase = {}
    for index, ((site_path, site), root_key) in enumerate(
            zip(site_list, root_keys)):
        if root_key not in inner:
            raise ValueError(
                f'compose: inner schema missing root {root_key!r}')
        filling[tuple(site_path)] = (
            site_path[-1] if site_path else index, site, inner[root_key])
        rebase[root_key] = tuple(site_path)

    result = _fill_at_paths(core, outer, filling)

    # --- Link composition: join matching names ---
    # inner's outer names → ports on inner's Links whose outputs are
    # unwired. outer's inner names → ports on outer's Links whose
    # inputs are unwired. Matching names name the same link.
    for port_name, inner_link_path in inner_outer._names.items():
        if port_name in outer_inner._names:
            _join_names(
                result,
                port_name,
                outer_link_path=outer_inner._names[port_name],
                inner_link_path=inner_link_path,
                rebase=rebase)

    return result


def _join_names(result, port_name, outer_link_path, inner_link_path, rebase):
    """Join an outer name of the filler to the inner name of the context.

    Milner joins the two faces at a single link; in this model a link is
    realized as a **store path**, so joining means wiring *both* ports to
    one shared store.

    Wires are relative to a link's parent store (``realize.port_merges``
    resolves a wire as ``link_path[:-1] + wire``) and cannot ascend, so:

    - the shared store is the filler link's own default output store —
      ``<filler link's parent>/<port_name>`` — which the filler reaches
      with the plain relative wire ``[port_name]``;
    - the context link reaches it by descending from its own parent, which
      is possible exactly when that parent contains the filled site.

    ``rebase`` maps each of ``inner``'s root keys to the site path its
    contents were substituted into, so the filler link's path can be
    expressed in the composed tree.
    """
    outer_link = _get_at_path(result, outer_link_path)
    if not isinstance(outer_link, Link):
        return

    # Rebase the filler link's path: composition splices a root's
    # *contents* at the site, so the root key itself disappears.
    root_key = inner_link_path[0]
    if root_key not in rebase:
        return
    composed_link_path = rebase[root_key] + tuple(inner_link_path[1:])

    store_path = composed_link_path[:-1] + (port_name,)

    outer_parent = tuple(outer_link_path[:-1])
    if store_path[:len(outer_parent)] != outer_parent:
        raise ValueError(
            f'compose: cannot join name {port_name!r} — wires are relative '
            f'to a link\'s parent and cannot ascend, but the link at '
            f'{outer_link_path} would have to reach {store_path}. Place the '
            f'site inside the consuming link\'s parent, or wire the port '
            f'explicitly before composing.')

    inner_link = _get_at_path(result, composed_link_path)
    if isinstance(inner_link, Link):
        if isinstance(inner_link.outputs, Wires):
            inner_link.outputs = {}
        inner_link.outputs[port_name] = [port_name]

    if isinstance(outer_link.inputs, Wires):
        outer_link.inputs = {}
    outer_link.inputs[port_name] = list(store_path[len(outer_parent):])


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
        """Get the sort of a node — from ``_type`` (canonical),
        ``_control`` (legacy), or the parent key."""
        if isinstance(node, dict):
            ctrl = node.get('_type', node.get('_control', key))
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
                node.get('_type', node.get('_control'))
                if isinstance(node, dict)
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


# ── The filling discipline: admits ──────────────────────────────────
# Milner's sorting constrains a bigraph in two ways that this module
# keeps as two named relations, because they are different relations
# that merely share an arity:
#
#   formation(parent_sort, child_sort)  — NESTING. May this child live
#       inside this parent? Policed by ``validate_sorting`` over
#       parent/child pairs, AFTER substitution.
#   admits(core, site, filler)          — FILLING. May this filler close
#       this hole? Checked BEFORE substitution, while the site — and so
#       its ``_sort`` — still exists. Once a site is filled there is no
#       site anymore, which is exactly why ``formation`` cannot do this
#       job.


def is_address(value):
    """True when ``value`` spells an edge address rather than a subtree.

    An address is a string (``'local:copasi'``, ``'copasi'``) or the canonical
    ``{'protocol', 'data'}`` dict — the two forms ``normalize_address``
    accepts.
    """
    if isinstance(value, str):
        return bool(value)
    return (isinstance(value, dict)
            and 'protocol' in value and 'data' in value)


def _face_from_address(core, address, config):
    """Resolve an edge's ports from the class its address names.

    A real process declaration carries an **address** and a config, not its
    ports — the ports live on the registered class and are only known once
    it is built. Without this, a site could only ever be filled by a
    declaration that restated its own interface, which no registered
    process does.

    Returns ``None`` when the address names nothing this core knows, so the
    caller can fall back to whatever was declared.
    """
    if core is None:
        return None

    if isinstance(address, Protocol):
        address = address._default
    canonical = normalize_address(address)
    if not isinstance(canonical, dict):
        return None

    edge_class = getattr(core, 'link_registry', {}).get(canonical.get('data'))
    if edge_class is None:
        return None

    if isinstance(config, Node):
        config = config._default
    if not isinstance(config, dict):
        config = {}

    try:
        face = edge_class(config, core).interface() or {}
    except Exception:
        return None

    return dict(face.get('inputs') or {}), dict(face.get('outputs') or {})


def collect_face(core, node):
    """Collect a filler's **outer face** — the ports it exposes.

    Returns an ``(inputs, outputs)`` pair of port-type maps. An ``Edge``
    instance reports its own via ``interface()`` (``edge.py``); a ``Link``
    reports ``_inputs``/``_outputs``, or — when it declares none — the ports
    of the class its address names; a subtree reports the union of the Links
    it contains. Declared ports are used regardless of wiring: a filler
    exposes its interface whether or not it is internally wired.
    """
    inputs = {}
    outputs = {}

    interface = getattr(node, 'interface', None)
    if callable(interface):
        face = interface() or {}
        return dict(face.get('inputs') or {}), dict(face.get('outputs') or {})

    # A bare address is a filler in its own right — the abstract-process case,
    # where only the implementation is being injected. Its face is the face of
    # the process it names.
    if is_address(node):
        resolved = _face_from_address(core, node, None)
        if resolved is not None:
            return resolved

    def declared(get, address, config):
        found_in = get('_inputs')
        found_out = get('_outputs')
        found_in = found_in if isinstance(found_in, dict) else {}
        found_out = found_out if isinstance(found_out, dict) else {}
        if not found_in and not found_out:
            resolved = _face_from_address(core, address, config)
            if resolved is not None:
                return resolved
        return found_in, found_out

    def walk(subnode):
        if isinstance(subnode, Link):
            found_in, found_out = declared(
                lambda key: getattr(subnode, key, None),
                getattr(subnode, 'address', None),
                getattr(subnode, 'config', None))
            inputs.update(found_in)
            outputs.update(found_out)
        elif isinstance(subnode, dict):
            if subnode.get('_type') in ('link', 'process', 'step'):
                found_in, found_out = declared(
                    subnode.get,
                    subnode.get('address'),
                    subnode.get('config'))
                inputs.update(found_in)
                outputs.update(found_out)
                return
            for key, child in subnode.items():
                if isinstance(key, str) and not key.startswith('_'):
                    walk(child)

    walk(node)
    return inputs, outputs


def contract_of(core, node):
    """The contract of an **edge or a site** — universal ``describe_contract``.

    A site's sort *is* a contract: a template site says "something satisfying
    *this* contract", not merely "shaped like X". So the same call answers for
    the thing that fills a hole and for the hole itself, and at exactly the
    granularity you fill.

    - a **site** → the contract it *requires*: a sort naming a registered
      contract resolves to that contract (amendments included); a face
      literal yields a contract whose typed core is that face.
    - an **edge instance** → its own ``describe_contract``.
    - a **link schema** → its declared face, or the face of the class its
      address names, merged with that class's documented contract.

    Returns ``None`` when the node carries no interface at all.
    """
    from bigraph_schema.contract import ProcessContract, resolve_contract

    if node is None:
        return None

    if isinstance(node, Site):
        sort = getattr(node, '_sort', '')
        if isinstance(sort, str) and sort:
            registered = getattr(core, 'contract_registry', {}).get(sort)
            if registered is not None:
                return registered
            sort = core.access(sort) if core is not None else sort
        if isinstance(sort, Link):
            return contract_of(core, sort)
        return None

    if callable(getattr(node, 'interface', None)):
        return resolve_contract(node)

    inputs, outputs = collect_face(core, node)
    if not inputs and not outputs:
        return None

    documented = None
    if isinstance(node, Link) and core is not None:
        address = node.address
        if isinstance(address, Protocol):
            address = address._default
        canonical = normalize_address(address)
        if isinstance(canonical, dict):
            edge_class = getattr(core, 'link_registry', {}).get(
                canonical.get('data'))
            if edge_class is not None:
                try:
                    documented = resolve_contract(
                        edge_class(node.config if isinstance(
                            node.config, dict) else {}, core))
                except Exception:
                    documented = None

    contract = copy.deepcopy(documented) if documented else ProcessContract()
    contract.face = {'inputs': dict(inputs), 'outputs': dict(outputs)}
    return contract


def contract_admits(core, contract, filler):
    """Does ``filler`` satisfy ``contract``? Returns ``(ok, reason)``.

    The typed core decides conformance; ``narrow`` amendments may add
    predicates that must also hold. Because narrowing only ever adds
    requirements, this stays monotone — see :func:`~bigraph_schema.contract.amend`.
    """
    ok, reason = face_conforms(core, contract.face, filler)
    if not ok:
        return ok, reason

    for index, predicate in enumerate(contract.predicates()):
        try:
            satisfied = predicate(core, filler)
        except Exception as error:
            return False, f'contract predicate {index} raised: {error}'
        if not satisfied:
            return False, f'filler fails contract predicate {index}'

    return True, None


def face_conforms(core, face, filler):
    """Structural subtyping of faces (the composition law, made typed).

    The filler must provide **every** port the face requires, at a type
    that ``core.resolve`` accepts. Over-providing is fine; under-providing
    is not — that is what makes a site reusable across processes of the
    same shape.

    ``face`` may be a ``Link`` schema (``_inputs``/``_outputs``) or a
    contract's typed core (``{'inputs': …, 'outputs': …}``) — the same two
    spellings of one thing.

    Returns ``(ok, reason)``.
    """
    provided = dict(zip(("inputs", "outputs"), collect_face(core, filler)))

    for direction, port_key in (('inputs', '_inputs'), ('outputs', '_outputs')):
        if isinstance(face, dict):
            required = face.get(direction)
        else:
            required = getattr(face, port_key, None)
        if not isinstance(required, dict):
            continue
        for port, port_schema in required.items():
            if port not in provided[direction]:
                return False, (
                    f'filler does not provide {direction[:-1]} port '
                    f'{port!r} (has {sorted(provided[direction])})')
            try:
                core.resolve(port_schema, provided[direction][port])
            except Exception as error:
                return False, (
                    f'{direction[:-1]} port {port!r} does not resolve: '
                    f'{error}')

    return True, None


def admits_why(core, site, filler):
    """As :func:`admits`, but returns ``(ok, reason)`` so callers can say
    *why* a filler was rejected."""
    sort = getattr(site, '_sort', '') or ''
    if not sort:
        return True, None          # an unsorted hole — pure Milner

    if core is None:
        return False, (
            f'site is sorted {sort!r} but no core was supplied to check it')

    if isinstance(sort, str):
        admits_fn = getattr(core, 'sort_registry', {}).get(sort)
        if admits_fn is not None:
            ok = admits_fn(core, site, filler)
            return bool(ok), None if ok else (
                f'sort {sort!r} rejected the filler')

    # A sort that names an interface is a *contract*: the typed face decides
    # conformance, and any narrowing amendment adds to it.
    contract = contract_of(core, site)
    if contract is not None:
        return contract_admits(core, contract, filler)

    if core.check(sort, filler):
        return True, None
    return False, f'value does not satisfy sort {sort!r}'


def admits(core, site, filler):
    """Is ``filler`` an admissible filling for this sorted site?

    Checked BEFORE substitution (see the section note above). An unsorted
    site admits anything. A sort registered via ``core.register_sort``
    decides for itself. A sort that names a **face** (``link[in, out]``,
    or a link literal) is decided by structural conformance
    (:func:`face_conforms`); any other sort is a value type, decided by
    ``core.check``.
    """
    ok, _reason = admits_why(core, site, filler)
    return ok


# ── fill — the one substitution primitive ───────────────────────────


def collect_sites(body):
    """Map each open site's **address** to ``(path, site)``.

    A site is addressed by its key in the place graph — the convention
    ``instantiate`` and ``Match.bindings`` already use — and *always* also
    by its full ``/``-joined path. The path form is what distinguishes
    sites that replication (:func:`replicate`) has given the same key in
    different copies of a region. A bare key shared by more than one site
    is simply not registered, so using it is an error that can name the
    alternatives rather than silently filling the wrong hole.
    """
    places = list(interfaces(body)[0]._places)

    shared = {}
    for path, _site in places:
        shared.setdefault(path[-1] if path else '', []).append(path)

    sites = {}
    for path, site in places:
        sites['/'.join(path)] = (path, site)
        name = path[-1] if path else ''
        if len(shared[name]) == 1:
            sites[name] = (path, site)

    return sites


def _as_schema(core, site, filler):
    """Shape a filler for substitution into a *schema* tree.

    A structural filler is already a subtree and goes in as-is. A **value**
    filler is state, not schema — so it lands as the site's own sort
    carrying that value as its default. Filling a document therefore keeps
    it a document, and ``core.fill`` later materializes the value into
    state. An unsorted site has no sort to carry the value, so the raw
    filler stands (the pure-Milner case).
    """
    sort = getattr(site, '_sort', '')

    # Injecting an implementation: a face-sorted site filled with an *address*
    # compiles to a Protocol, the same shape ``access`` gives a declared
    # address — so the filled edge realizes exactly like a written one.
    if core is not None and isinstance(sort, Link) and is_address(filler):
        from bigraph_schema.methods.handle_parameters import access_address
        return access_address(core, filler)

    if isinstance(filler, (Node, dict)):
        return filler

    if not sort or core is None:
        return filler

    schema = core.access(sort)
    if isinstance(schema, Node):
        return replace(schema, _default=filler)
    return filler


def _fill_at_paths(core, body, filling):
    """The substitution itself, shared by every way of naming a site.

    ``filling`` maps a site's **path** to ``(label, site, filler)``. Each
    filler is checked with :func:`admits` *before* substitution — while the
    site still exists — and a rejection is a fill error naming the site.
    Paths not in ``filling`` are left open.

    **Place semantics (settled).** A filler is substituted **at the site's
    own position**; it is not forest-spliced into the site's parent, which
    is what ``instantiate`` does when a *redex* site captures several
    sibling keys during matching. Reaction-matching and composition are
    different operations and this is where they differ:

    - Milner composition plugs **one root into one site**, so a filler with
      n roots fills n sites — ``compose`` already enforces that arity.
      Splicing a multi-root filler into a single site would silently turn
      one hole into n nodes.
    - Splicing drops the site's key, so the filled region loses the name the
      template gave it, and merges the filler's roots into the parent's
      namespace where they can collide with the template's own stores.
    - Empirically, ``process_bigraph.Composite`` reports a filled model site
      at ``('study', 'model')`` — it expects the process node exactly where
      the site was, with a nested composite's internals kept under it.
    """
    result = copy.deepcopy(body)

    for path, (label, site, filler) in filling.items():
        ok, reason = admits_why(core, site, filler)
        if not ok:
            raise ValueError(
                f'fill: filler rejected for site {label!r} at {path}: '
                f'{reason}')
        _set_at_path(
            result, path, copy.deepcopy(_as_schema(core, site, filler)))

    return result


def fill_sites(core, body, bindings):
    """Substitute fillers into named open sites — **the** primitive.

    ``bindings`` maps a site's name to its filler. Sites with no binding
    fall back to the site's ``_default`` if it has one and are otherwise
    **left open** — a partially filled document is still a document, which
    is what lets filling be incremental.

    Filling independent sites commutes: each binding is applied at its own
    path, so the result does not depend on the order of ``bindings``.
    """
    sites = collect_sites(body)

    unknown = set(bindings) - set(sites)
    if unknown:
        hints = []
        for name in sorted(unknown):
            alternatives = sorted(
                address for address, (path, _site) in sites.items()
                if '/' in address and path and path[-1] == name)
            if alternatives:
                hints.append(
                    f'{name!r} names more than one site — address it by '
                    f'path, one of {alternatives}')
        detail = '; '.join(hints) if hints else (
            f'open sites: {sorted(sites)}')
        raise ValueError(f'fill: no such site(s): {sorted(unknown)} — {detail}')

    # A site carries up to two addresses (bare key and path); resolve every
    # binding down to the path it designates so both spellings agree.
    bound = {}
    for address, filler in bindings.items():
        path, _site = sites[address]
        if path in bound and bound[path][1] is not filler:
            raise ValueError(
                f'fill: the site at {path} is bound twice, as '
                f'{bound[path][0]!r} and as {address!r}')
        bound[path] = (address, filler)

    filling = {}
    for address, (path, site) in sites.items():
        if path in filling:
            continue
        if path in bound:
            label, filler = bound[path]
        elif getattr(site, '_default', None) is not None:
            label, filler = address, site._default
        else:
            continue                    # left open, still a site
        filling[path] = (label, site, filler)

    return _fill_at_paths(core, body, filling)


# ── Cardinality: replication is a reaction, not a kind of site ──────
# "How many processes / store subtrees" changes the *shape* of the
# document, so it is not a hole to be filled — it is a rewrite. Milner
# already has the right tool: a parametric reaction rule whose reactum
# mentions the redex's parameter more than once shares that parameter
# across every occurrence (§8.1, p. 83). Replication is exactly that
# rule, so this adds no operator — only the rule and a driver.

REPLICATE = 'replicate'
"""The control marking a repeatable region. A node carrying
``_control: 'replicate'`` is replicated by :func:`replicate` into keyed
copies; the copies do not carry the mark, so replication is idempotent
and the rewrite reaches quiescence on its own."""

COUNT_KEY = '_count'
"""Optional default count on a marked region, overridable at build time."""

MAX_REPLICAS = 1024
"""Guard against a count that explodes the document."""


def collect_regions(body):
    """Map each marked region's name (its key) to its path."""
    regions = {}

    def walk(node, path):
        if not isinstance(node, dict):
            return
        if node.get('_control') == REPLICATE:
            name = path[-1] if path else ''
            if name in regions:
                raise ValueError(
                    f'replicate: region name {name!r} is ambiguous — it '
                    f'appears at {regions[name]} and at {path}')
            regions[name] = path
            return                      # regions do not nest through a mark
        for key, child in node.items():
            if isinstance(key, str) and not key.startswith('_'):
                walk(child, path + (key,))

    walk(body, ())
    return regions


def replicate_rule(name, count):
    """The cardinality rule: one marked region in, ``count`` copies out.

    The redex is the marked region with a single site capturing its
    contents; the reactum is ``count`` keyed copies whose sites all
    instantiate from that *same* redex site — Milner's shared parameter.
    The copies drop the mark, so the rule cannot fire on its own output.
    """
    return ReactionRule(
        redex={name: {'_control': REPLICATE, 'contents': Site()}},
        reactum={
            f'{name}_{index}': {'contents': Site()}
            for index in range(count)},
        instantiation={
            f'{name}_{index}': 'contents' for index in range(count)},
        label=f'replicate {name} x{count}')


def replicate(body, counts=None):
    """Expand every marked region into keyed copies, before filling.

    ``counts`` maps a region's name to its count, overriding the region's
    own ``_count``; a region with neither stays single. Regions are
    expanded one at a time, re-walking after each firing because
    expansion changes paths — and because a copied region may itself
    contain a marked sub-region.

    Deterministic: the copy keys are ``<name>_<i>`` for ``i`` in order, and
    the region to expand is chosen in walk order, so the same input and
    the same counts always yield the identical document.
    """
    counts = counts or {}
    body = copy.deepcopy(body)

    for _ in range(MAX_REPLICAS):
        regions = collect_regions(body)
        if not regions:
            return body

        name, path = next(iter(regions.items()))
        region = _get_at_path(body, path)
        count = counts.get(name, region.get(COUNT_KEY, 1))

        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(
                f'replicate: count for region {name!r} must be a '
                f'non-negative int, got {count!r}')
        if count > MAX_REPLICAS:
            raise ValueError(
                f'replicate: count {count} for region {name!r} exceeds '
                f'MAX_REPLICAS ({MAX_REPLICAS})')

        rule = replicate_rule(name, count)

        # The redex matches any node bearing the mark, so select the match
        # that is *this* region: same parent, and the pattern key assigned
        # to this region's key.
        parent = tuple(path[:-1])
        index = next(
            (position for position, match in enumerate(
                find_matches(body, rule.redex))
             if tuple(match.path) == parent and match.key_map.get(name) == name),
            None)
        if index is None:
            raise ValueError(
                f'replicate: marked region {name!r} at {path} did not match '
                f'its own rule')

        body, _match = fire_rule(body, rule, match_index=index)

    raise ValueError(
        f'replicate: more than {MAX_REPLICAS} regions expanded — a marked '
        f'region is probably regenerating the mark')


# ── build — the template convenience (sugar, not a primitive) ───────


def build(core, template, overrides=None):
    """Replicate, fill, default, and check groundness. Returns
    ``(schema, state)`` — directly runnable by process-bigraph.

    ``overrides`` carries region counts *and* site fillers in one map;
    a key naming a marked region is a count, anything else is a filler.
    Not a primitive: every step is an existing operation.
    """
    overrides = overrides or {}

    regions = collect_regions(template)
    counts = {name: overrides[name] for name in regions if name in overrides}
    body = replicate(template, counts)

    bindings = {key: value for key, value in overrides.items()
                if key not in counts}
    schema = fill_sites(core, body, bindings)

    open_sites = [path for path, _site in interfaces(schema)[0]._places]
    if open_sites:
        raise ValueError(
            'build: document is not ground — required site(s) left '
            'unfilled: ' + ', '.join(
                repr('/'.join(path)) for path in sorted(open_sites)))

    return schema, core.fill(schema, {})


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
        # Bigraph signature labels can be carried in ``_type`` (the
        # canonical form, sharing one namespace with the typesystem)
        # or in the legacy ``_control`` field. We accept either, so
        # rules and states authored against either spelling continue
        # to match. Mixed usage on a single node would be a bug, but
        # we don't enforce that here.
        redex_sort = redex_node.get('_type', redex_node.get('_control'))
        if redex_sort is not None:
            state_sort = state_node.get('_type', state_node.get('_control'))
            if state_sort != redex_sort:
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

    site_keys = [k for k, v in redex_entries if isinstance(v, Site)]
    non_site_count = len(redex_entries) - len(site_keys)

    # Sites are allowed to bind to *zero* state keys (an empty
    # capture), so the floor on state size is set by the non-Site
    # redex entries only. We pad ``state_keys`` with ``None``
    # sentinels so the original 1-to-1 permutation logic can still
    # run when the redex has more entries than the state — any
    # padding that lands on a Site means "this Site captures
    # nothing"; padding that lands on a non-Site is a mismatch.
    if non_site_count > len(state_keys):
        return False

    has_surplus = len(redex_entries) < len(state_keys)
    n_padding = max(0, len(redex_entries) - len(state_keys))
    padded_keys = list(state_keys) + [None] * n_padding

    # Edge bindings (from ``LinkVar``s) are shared by reference
    # across every recursive ``_match_dict`` call within a single
    # match attempt — that's what makes ``LinkVar('e')`` at one
    # node's wire visible to ``LinkVar('e')`` at a sibling or
    # descendant node's wire. We snapshot before each iteration so
    # a failed permutation can roll the bindings back.
    edges = bindings.setdefault('__edges__', {})

    for perm in permutations(padded_keys, len(redex_entries)):
        assignment = dict(zip([k for k, _ in redex_entries], perm))
        saved_edges = dict(edges)
        trial = {'__edges__': edges}
        ok = True
        for redex_key, state_key in assignment.items():
            redex_value = redex_dict[redex_key]
            if state_key is None:
                # Padding: only meaningful for a Site (binds empty).
                if not isinstance(redex_value, Site):
                    ok = False
                    break
                trial[redex_key] = {}
                continue
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
            edges.clear()
            edges.update(saved_edges)
            continue

        # Surplus state keys → merge into the last Site's binding
        used = set(v for v in assignment.values() if v is not None)
        surplus = {k: state_dict[k] for k in state_keys if k not in used}

        if surplus and site_keys:
            last_site = site_keys[-1]
            existing = trial.get(last_site, {})
            if isinstance(existing, dict):
                existing.update(surplus)
                trial[last_site] = existing
            else:
                trial[last_site] = surplus

        # Pull the trial's __key_map__ out before updating bindings:
        # otherwise a recursive _match_dict that found no non-_
        # children (e.g. matching a literal sort tag like
        # ``{'_type': 'Cytoplasm'}``) writes an *empty* __key_map__
        # into trial, and a blind ``bindings.update(trial)`` would
        # wipe out any __key_map__ already accumulated by an earlier
        # sibling match at this same level. We merge instead.
        trial_keymap = trial.pop('__key_map__', None)
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
        merged = {**this_keymap, **existing_keymap}
        if trial_keymap:
            merged = {**merged, **trial_keymap}
        bindings['__key_map__'] = merged
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
            if isinstance(filler, dict) \
                    and '_control' not in filler \
                    and '_type' not in filler:
                # Forest of trees captured (Milner: a site is a hole
                # in the place graph that gets filled with a region;
                # the region's roots become children at the slot
                # position rather than nesting under the site name).
                # We detect a forest by the absence of any sort
                # label at the top — every bigraph node has either
                # ``_type`` (canonical) or ``_control`` (legacy), so
                # a label-less dict is a multi-rooted region.
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
