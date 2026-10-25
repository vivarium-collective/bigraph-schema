"""Config-fill and port-projection fixes (vivarium-collective/bigraph-schema).

Three bugs, all surfaced while building viva-pulse:

- #203 — an explicit zero-valued config (0, 0.0, '') was replaced by the
  default because state-merge emptiness rules were applied to config.
- #202 — a list config was concatenated onto a non-empty ``_default`` instead
  of replacing it, and an unsupplied empty-list default was dropped entirely
  once another key was supplied.
- #201 — projecting ``initial_state()`` through a port wrote keys the port
  does not declare into the wired store.
"""
from bigraph_schema import Edge, allocate_core


# ---------------------------------------------------------------------------
# Config fill (#202, #203): a supplied value replaces the default; a declared
# key the caller omits (or sets to None) takes its default.
# ---------------------------------------------------------------------------

def _config(schema, value):
    class C(Edge):
        config_schema = {'k': schema}
    return C({'k': value}, core=allocate_core()).config['k']


def test_zero_valued_config_is_kept_not_replaced_by_default():  # #203
    assert _config({'_type': 'float', '_default': 5.0}, 0.0) == 0.0
    assert _config({'_type': 'float', '_default': 5.0}, -1.0) == -1.0
    assert _config({'_type': 'integer', '_default': 5}, 0) == 0
    assert _config({'_type': 'string', '_default': 'a'}, '') == ''


def test_supplied_config_still_wins_over_default():
    assert _config({'_type': 'float', '_default': 5.0}, 7.5) == 7.5
    assert _config({'_type': 'string', '_default': 'a'}, 'b') == 'b'


def test_list_config_replaces_default_rather_than_concatenating():  # #202
    assert _config({'_type': 'list[string]', '_default': ['a', 'b', 'c']},
                   ['x']) == ['x']
    assert _config({'_type': 'list[string]', '_default': []}, ['x']) == ['x']


def test_omitted_or_none_config_key_falls_back_to_default():
    assert _config({'_type': 'float', '_default': 5.0}, None) == 5.0

    class C(Edge):
        config_schema = {'k': {'_type': 'float', '_default': 5.0}}
    assert C({}, core=allocate_core()).config['k'] == 5.0


def test_every_declared_key_is_present_when_another_is_supplied():  # #202 (comment)
    class C(Edge):
        config_schema = {
            'given': {'_type': 'float', '_default': 1.0},
            'empty_list': {'_type': 'list[string]', '_default': []},
            'list_abc': {'_type': 'list[string]', '_default': ['a']},
            'zero_float': {'_type': 'float', '_default': 0.0},
            'empty_string': {'_type': 'string', '_default': ''},
            'zero_int': {'_type': 'integer', '_default': 0},
            'empty_map': {'_type': 'map[float]', '_default': {}},
        }

    cfg = C({'given': 2.0}, core=allocate_core()).config
    assert cfg['given'] == 2.0
    assert cfg['empty_list'] == []          # was dropped before the fix
    assert cfg['list_abc'] == ['a']
    assert cfg['zero_float'] == 0.0
    assert cfg['empty_string'] == ''
    assert cfg['zero_int'] == 0
    assert cfg['empty_map'] == {}


def test_nested_config_group_keeps_sibling_defaults():
    """A partially-supplied group of config fields keeps the untouched
    siblings' defaults while replacing the supplied leaf."""
    class C(Edge):
        config_schema = {
            'group': {
                'a': {'_type': 'float', '_default': 1.0},
                'b': {'_type': 'float', '_default': 2.0},
            }
        }

    cfg = C({'group': {'a': 9.0}}, core=allocate_core()).config
    assert cfg['group'] == {'a': 9.0, 'b': 2.0}


# ---------------------------------------------------------------------------
# Port projection (#201): projecting through a port keeps only declared keys.
# ---------------------------------------------------------------------------

def test_project_ports_drops_keys_the_port_does_not_declare():  # #201
    core = allocate_core()
    ports_schema = {'m': {'cmd': 'float'}}      # port m declares only 'cmd'
    wires = {'m': ['interventions']}
    view = {'m': {'cmd': 3.0, 'extra': 9.0}}

    _, state = core.project_ports(ports_schema, wires, (), view)

    assert state['interventions'] == {'cmd': 3.0}   # 'extra' must not leak


def test_project_ports_keeps_declared_keys_intact():
    core = allocate_core()
    ports_schema = {'m': {'cmd': 'float', 'gain': 'float'}}
    wires = {'m': ['store']}
    view = {'m': {'cmd': 3.0, 'gain': 2.0}}

    _, state = core.project_ports(ports_schema, wires, (), view)

    assert state['store'] == {'cmd': 3.0, 'gain': 2.0}
