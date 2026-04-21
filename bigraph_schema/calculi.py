"""
Canonical process calculi as bigraphical reactive systems
=========================================================

Each process calculus is encoded as a sorted signature + reaction rules,
following Milner's *Space and Motion of Communicating Agents* (2008):

  - **CCS** (Ch. 10): alternation-guarded send/get prefixes, stratified
    place sorting with sorts {p, a}, all controls passive.
  - **Mobile Ambients** (Ch. 1, Fig. 1.1): ambient boundaries with
    in/out/open commands, ``amb`` is active.
  - **Petri Nets** (§6.2, §9.2): condition-event nets with many-one
    link sorting.

Each encoder returns a ``(Sorting, [ReactionRule], initial_state)``
triple that can be fed to ``run_reactions`` or ``ReactionStep``.
"""

from bigraph_schema.schema import Site
from bigraph_schema.assembly import (
    Sorting, ReactionRule, stratified_sorting, many_one_sorting,
    ACTIVE, PASSIVE, ATOMIC)


# ── CCS ─────────────────────────────────────────────────────────────
# Milner Ch. 10 (pp. 103-112).
#
# CCS processes are built from:
#   - alt (alternation): passive, houses summands
#   - send_x (output prefix on channel x): passive, guards a continuation
#   - get_x (input prefix on channel x): passive, guards a continuation
#   - nil: the null process
#
# Stratified sorting: Θ = {p, a}, φ(p) = a, φ(a) = p
# All controls are passive (reactions only at the top level).
#
# Reaction: a matching send_x and get_x in sibling alternations
# both fire, consuming the prefixes and exposing their continuations.


def ccs_sorting(channels=('x',)):
    """Build the CCS stratified sorting.

    ``channels`` lists the channel names. Each channel x produces
    controls ``send_x`` and ``get_x``.
    """
    controls = {
        'alt': {'sort': 'a', 'status': PASSIVE, 'arity': 0},
        'nil': {'sort': 'a', 'status': PASSIVE, 'arity': 0}}
    for ch in channels:
        controls[f'send_{ch}'] = {
            'sort': 'p', 'status': PASSIVE, 'arity': 1,
            'port_sorts': ('s',)}
        controls[f'get_{ch}'] = {
            'sort': 'p', 'status': PASSIVE, 'arity': 1,
            'port_sorts': ('t',)}

    return stratified_sorting(
        sorts={'p', 'a'},
        phi={'p': 'a', 'a': 'p'},
        controls=controls,
        hard_sorts={'p'})


def ccs_reaction(channel):
    """The CCS reaction rule for channel ``channel``.

    Redex: two sibling alternations, one housing a send_x prefix,
    the other a get_x prefix. Each prefix guards a continuation (Site).

    Reactum: the prefixes are consumed; the continuations and any
    remaining alternation summands are exposed.

    In algebraic notation (Milner Example 8.1, p. 81)::

        alt.(send_x.d0 | d1) | alt.(get_x.d2 | d3) → x | d0 | d2
    """
    return ReactionRule(
        redex={
            'sender': {
                '_control': 'alt',
                'prefix': {
                    '_control': f'send_{channel}',
                    'continuation': Site()},
                'rest_s': Site()},
            'receiver': {
                '_control': 'alt',
                'prefix': {
                    '_control': f'get_{channel}',
                    'continuation': Site()},
                'rest_r': Site()}},
        reactum={
            'send_cont': Site(),
            'recv_cont': Site()},
        instantiation={
            'send_cont': 'continuation',
            'recv_cont': 'continuation'},
        label=f'CCS sync on {channel}')


def ccs_brs(channels=('x',)):
    """Build a CCS bigraphical reactive system.

    Returns ``(sorting, rules, example_state)``.
    """
    sorting = ccs_sorting(channels)
    rules = [ccs_reaction(ch) for ch in channels]

    # Example: P = send_x.nil | get_x.nil (a simple synchronisation)
    example = {
        'proc': {
            '_control': 'alt',
            'send': {
                '_control': f'send_{channels[0]}',
                'cont': {'_control': 'nil'}},
            'rest': {'_control': 'nil'}},
        'listener': {
            '_control': 'alt',
            'recv': {
                '_control': f'get_{channels[0]}',
                'cont': {'_control': 'nil'}},
            'rest': {'_control': 'nil'}}}

    return sorting, rules, example


# ── Mobile Ambients ─────────────────────────────────────────────────
# Milner Ch. 1, Fig. 1.1 (p. 12).
#
# Controls:
#   - amb (ambient): ACTIVE, arity 1 (the ambient's name port)
#   - in_y (enter ambient named y): arity 1
#   - out_y (exit parent): arity 1
#   - open_x (dissolve ambient): arity 1
#
# Three rules: A1 (in), A2 (out), A3 (open).


def ambient_sorting(names=('x', 'y')):
    """Build the mobile ambients sorting."""
    controls = {
        'amb': {'sort': 'amb', 'status': ACTIVE, 'arity': 1}}
    for n in names:
        controls[f'in_{n}'] = {
            'sort': 'cap', 'status': PASSIVE, 'arity': 1}
        controls[f'out_{n}'] = {
            'sort': 'cap', 'status': PASSIVE, 'arity': 1}
        controls[f'open_{n}'] = {
            'sort': 'cap', 'status': PASSIVE, 'arity': 1}

    return Sorting(
        sorts={'amb', 'cap'},
        controls=controls)


def ambient_rule_in(target_name):
    """A1: ``amb_x.(in_y.d0 | d1) | amb_y.d2 → amb_y.(amb_x.(d0 | d1) | d2)``

    An ambient x containing an ``in_y`` command moves inside
    ambient y (its sibling).
    """
    return ReactionRule(
        redex={
            'mover': {
                '_control': 'amb',
                'cmd': {'_control': f'in_{target_name}',
                        'cmd_content': Site()},
                'content': Site()},
            'target': {
                '_control': 'amb',
                'inside': Site()}},
        reactum={
            'target': {
                '_control': 'amb',
                'inside': Site(),
                'mover': {
                    '_control': 'amb',
                    'content': Site(),
                    'cmd_content': Site()}}},
        instantiation={
            'inside': 'inside',
            'content': 'content',
            'cmd_content': 'cmd_content'},
        label=f'A1: in_{target_name}')


def ambient_rule_out(parent_name):
    """A2: ``amb_y.(amb_x.(out_y.d0 | d1) | d2) → amb_x.(d0 | d1) | amb_y.d2``

    An ambient x inside ambient y, containing an ``out_y`` command,
    exits y and becomes its sibling.
    """
    return ReactionRule(
        redex={
            'parent': {
                '_control': 'amb',
                'child': {
                    '_control': 'amb',
                    'cmd': {'_control': f'out_{parent_name}',
                            'cmd_content': Site()},
                    'child_content': Site()},
                'parent_content': Site()}},
        reactum={
            'child': {
                '_control': 'amb',
                'child_content': Site(),
                'cmd_content': Site()},
            'parent': {
                '_control': 'amb',
                'parent_content': Site()}},
        instantiation={
            'child_content': 'child_content',
            'cmd_content': 'cmd_content',
            'parent_content': 'parent_content'},
        label=f'A2: out_{parent_name}')


def ambient_rule_open(name):
    """A3: ``open_x.d0 | amb_x.d1 → d0 | d1``

    The ``open_x`` command dissolves ambient x, exposing its contents.
    """
    return ReactionRule(
        redex={
            'opener': {
                '_control': f'open_{name}',
                'opener_content': Site()},
            'target': {
                '_control': 'amb',
                'target_content': Site()}},
        reactum={
            'opener_content': Site(),
            'target_content': Site()},
        instantiation={
            'opener_content': 'opener_content',
            'target_content': 'target_content'},
        label=f'A3: open_{name}')


def ambient_brs(names=('x', 'y')):
    """Build a mobile ambients BRS.

    Returns ``(sorting, rules, example_state)``.
    """
    sorting = ambient_sorting(names)
    rules = []
    for n in names:
        rules.append(ambient_rule_in(n))
        rules.append(ambient_rule_out(n))
        rules.append(ambient_rule_open(n))

    # Example from Milner Ch. 1 (p. 10):
    # amb_x.(in_y.d0 | d1) | amb_y.d2
    example = {
        'amb_x': {
            '_control': 'amb',
            'cmd': {'_control': f'in_{names[1]}',
                    'payload': {'data': 42}},
            'stuff': {'value': 1.0}},
        'amb_y': {
            '_control': 'amb',
            'content': {'value': 2.0}}}

    return sorting, rules, example


# ── Petri Nets ──────────────────────────────────────────────────────
# Milner §6.2 (p. 62), §9.2 (p. 95).
#
# Condition-event nets:
#   - M (marked condition, arity 1, port sort s)
#   - U (unmarked condition, arity 1, port sort s)
#   - E_{h,k} (event with h pre-conditions + k post-conditions,
#     arity h+k, port sorts t^{h+k})
#
# Many-one link sorting ensures each condition connects to at most
# one event as "source".
#
# Firing: all pre-conditions marked → flip to unmarked; all
# post-conditions unmarked → flip to marked.


def petri_sorting(events=None):
    """Build a Petri net many-one sorting.

    ``events`` is a list of ``(name, h, k)`` triples where h is the
    number of pre-conditions and k the number of post-conditions.
    """
    if events is None:
        events = [('e1', 1, 1)]

    controls = {
        'M': {'sort': 's', 'status': ACTIVE, 'arity': 1,
              'port_sorts': ('s',)},
        'U': {'sort': 's', 'status': ACTIVE, 'arity': 1,
              'port_sorts': ('s',)}}
    for name, h, k in events:
        controls[name] = {
            'sort': 't', 'status': ACTIVE, 'arity': h + k,
            'port_sorts': ('t',) * (h + k)}

    return many_one_sorting(controls)


def petri_fire_rule(event_name, n_pre, n_post):
    """A Petri net firing rule for event ``event_name``.

    Redex: the event with all n_pre pre-conditions marked (M) and
    all n_post post-conditions unmarked (U).

    Reactum: pre-conditions become U, post-conditions become M.
    """
    redex = {}
    reactum = {}

    for i in range(n_pre):
        key = f'pre_{i}'
        redex[key] = {'_control': 'M', 'token': Site()}
        reactum[key] = {'_control': 'U', 'token': Site()}

    for i in range(n_post):
        key = f'post_{i}'
        redex[key] = {'_control': 'U', 'token': Site()}
        reactum[key] = {'_control': 'M', 'token': Site()}

    instantiation = {
        **{f'pre_{i}': f'pre_{i}' for i in range(n_pre)},
        **{f'post_{i}': f'post_{i}' for i in range(n_post)}}
    # token sites preserved through
    for i in range(n_pre):
        instantiation[f'token'] = 'token'
    for i in range(n_post):
        instantiation[f'token'] = 'token'

    return ReactionRule(
        redex=redex,
        reactum=reactum,
        instantiation=instantiation,
        label=f'fire {event_name}')


def petri_brs(events=None):
    """Build a Petri net BRS.

    ``events`` is a list of ``(name, n_pre, n_post)`` triples.

    Returns ``(sorting, rules, example_state)``.
    """
    if events is None:
        events = [('e1', 2, 1)]

    sorting = petri_sorting(events)
    rules = [petri_fire_rule(name, h, k) for name, h, k in events]

    # Example: 2 pre-conditions (marked), 1 post-condition (unmarked)
    example = {
        'c1': {'_control': 'M', 'val': 1},
        'c2': {'_control': 'M', 'val': 2},
        'c3': {'_control': 'U', 'val': 0}}

    return sorting, rules, example
