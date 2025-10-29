"""
Bigraph-Schema Operation & Visitor
================================

This module provides the primary front door for bigraph-schema:

- **Core**: A registry-backed operation that parses and normalizes
  schema representations (strings, dicts, lists) into dataclass nodes
  (see `schema.py`), and exposes the core operations: `infer`, `render`,
  `default`, `resolve`, `check`, `serialize`, `deserialize`, `merge`,
  `jump`, `traverse`, `bind`, `apply`.

- **CoreVisitor**: A `parsimonious` visitor that lowers parsed bigraph
  expressions into node structures (e.g., `Union(_options=...)`,
  `Tuple(_values=...)`, typed mappings, and defaults).

- **Parameter/Access Hooks**:
  - `handle_parameters(...)` applies type parameters to nodes via multiple
    dispatch (e.g., `Array[_shape,_dtype]`, `Map[_key,_value]`, `Union[_options]`,
    `Tuple[_values]`, `Edge[_inputs,_outputs]`).
  - `post_access(...)` finalizes node construction after dict-based access,
    enforcing invariants (notably `Array._shape -> tuple[int,...]` and
    `Array._data -> numpy.dtype` for round-trip stability).
"""
import typing
import numpy as np
from numpy import dtype
import numpy.lib.format as nf
import pytest
import logging

from plum import dispatch
from parsimonious.nodes import NodeVisitor
from dataclasses import dataclass, is_dataclass, replace

from bigraph_schema.schema import (
    BASE_TYPES,
    resolve_path,
    convert_jump,
    convert_path,
    blank_context,
    Node,
    Union,
    Tuple,
    Boolean,
    Number,
    Integer,
    Float,
    Delta,
    Nonnegative,
    String,
    Enum,
    Wrap,
    Maybe,
    Overwrite,
    List,
    Map,
    Tree,
    Array,
    Key,
    Path,
    Wires,
    Schema,
    Edge,
    Jump,
    Star,
    Index,
)

from bigraph_schema.registry import deep_merge
from bigraph_schema.parse import visit_expression
from bigraph_schema.methods import (
    reify_schema,
    handle_parameters,
    infer,
    render,
    default,
    resolve,
    check,
    serialize,
    deserialize,
    merge,
    jump,
    traverse,
    bind,
    unify,

    apply)

# view
# project


def schema_keys(schema):
    keys = []
    for key in schema.__dataclass_fields__:
        if key.startswith('_'):
            keys.append(key)
    return keys


class CoreVisitor(NodeVisitor):
    """Visitor that converts parsed bigraph expressions into schema node structures.

    Operates within a `Core` context, mapping grammar constructs
    (unions, merges, type parameters, and defaults) into dataclass-based nodes.
    Handles normalization of nested expressions (e.g. `tuple[int,float]`,
    `edge[a:int|b:string]`, `(x:y|z:w)`) into instances of `Union`, `Tuple`,
    or structured dicts.
    """

    def __init__(self, operation):
        """Initialize with the active `Core`."""
        self.operation = operation

    def visit_expression(self, node, visit):
        """Top-level entry; returns first child."""
        return visit[0]

    def visit_union(self, node, visit):
        """Parse `a~b~c` into a `Union(_options=[a,b,c])`."""
        head = [visit[0]]
        tail = [tree['visit'][1] for tree in visit[1]['visit']]
        return Union(_options=head + tail)

    def visit_merge(self, node, visit):
        """Parse `a|b|c`; dicts merge to one mapping, others form a `Tuple`."""
        head = [visit[0]]
        tail = [tree['visit'][1] for tree in visit[1]['visit']]
        nodes = head + tail
        if all(isinstance(tree, dict) for tree in nodes):
            merged = {}
            for tree in nodes:
                merged.update(tree)
            return merged
        else:
            try:
                values = tuple([int(x) for x in nodes])
                return values
            except Exception as e:
                return Tuple(_values=nodes)

    def visit_tree(self, node, visit):
        """Delegate directly to nested element."""
        return visit[0]

    def visit_bigraph(self, node, visit):
        """Alias for tree; allows recursion within nested bigraphs."""
        return visit[0]

    def visit_group(self, node, visit):
        """Handle grouped subexpression `( ... )`; return tuple or dict."""
        group_value = visit[1]
        return group_value if isinstance(group_value, (list, tuple, dict, Tuple)) else (group_value,)

    def visit_nest(self, node, visit):
        """Handle `key:subtype` pairs (used in trees/maps)."""
        return {visit[0]: visit[2]}

    def visit_type_name(self, node, visit):
        """Resolve base type, parameters, and defaults into schema nodes."""
        schema = visit[0]

        # Parse parameter list
        type_parameters = [
            parameter
            for parameter in visit[1]['visit']]

        if type_parameters:
            schema = handle_parameters(self.operation, schema, type_parameters[0])

        # Parse default value `{...}`
        default_visit = visit[2]['visit']
        if default_visit:
            default = default_visit[0]
            if isinstance(schema, Node):
                schema._default = default
            elif isinstance(schema, dict):
                schema['_default'] = default

        return schema

    def visit_parameter_list(self, node, visit):
        """Return ordered list of parameters `[A,B,C]`."""
        first = [visit[1]]
        rest = [inner['visit'][1] for inner in visit[2]['visit']]
        return first + rest

    def visit_default_block(self, node, visit):
        """Extract contents of `{...}` blocks."""
        return visit[1]

    def visit_default(self, node, visit):
        """Return text inside default braces as string."""
        return node.text

    def visit_symbol(self, node, visit):
        """Resolve bare symbol names via the operation registry or parse visitor."""
        return self.operation.access(node.text)

    def visit_nothing(self, node, visit):
        """Handle empty productions (e.g., trailing commas)."""
        return None

    def generic_visit(self, node, visit):
        """Fallback: return raw parse node and visited children."""
        return {'node': node, 'visit': visit}


class Core:
    """Bigraph-schema operation: registry, parsing, normalization, and ops.

    - Maintains a registry mapping type keys to node constructors (see `BASE_TYPES`).
    - Normalizes schema representations (strings, dicts, lists) into dataclass nodes
      via `access(...)` using the bigraph grammar (`parse.visit_expression`).
    - Exposes core methods (`infer`, `render`, `default`, `resolve`, `check`,
      `serialize`, `deserialize`, `merge`, `jump`, `traverse`, `bind`, `apply`).
    - Post-access invariants: e.g., `Array._shape -> tuple[int,...]`,
      `Array._data -> numpy.dtype`; node fields like `_values`, `_options`,
      `_key/_value`, `_inputs/_outputs` are populated.
    """

    def __init__(self, types):
        """Initialize operation with a base type registry (e.g., `BASE_TYPES`)."""
        self.registry = {}
        self.register_types(types)
        self.parse_visitor = CoreVisitor(self)

    def register_type(self, key, data):
        """Register a single type key; deep-merge if it already exists."""
        if key in self.registry:
            self.update_type(key, data)
        else:
            self.registry[key] = data

    def register_types(self, types):
        """Bulk register multiple type keys into the operation registry."""
        for key, data in types.items():
            self.register_type(key, data)

    def update_type(self, key, data):
        """Deep-merge metadata/overrides into an existing registry entry."""
        self.registry[key] = deep_merge(self.registry[key], data)

    def select_fields(self, base, schema):
        """Project dict `schema` onto dataclass `base` fields, normalizing values via `access` (except `_default`)."""
        select = {}
        for key in base.__dataclass_fields__.keys():
            schema_key = schema.get(key)
            if schema_key:
                down = schema_key if key == '_default' else self.access(schema_key)
                select[key] = down
        return select

    def make_instance(self, base, state):
        """Instantiate dataclass `base` from dict `state` after field selection/normalization."""
        fields = self.select_fields(base, state)
        instance = base(**fields)
        return instance

    def access_type(self, value):
        if isinstance(value, dict) and '_type' in value:
            schema = core.access(value['_type'])

            default_value = None
            if '_default' in value:
                default_value = value['_default']
            elif isinstance(schema, Node) and schema._default is not None:
                default_value = schema._default
            schema._default = default_value

            parameters = {}
            for key in schema_keys(schema)[1:]:
                if key in value:
                    parameters[key] = value[key]
            schema = reify_schema(core, schema, parameters)
            return schema

    def access(self, key):
        """Normalize any schema form into nodes/values.

        - Dataclass node → returned as-is.
        - String → instantiate via registry; else parse bigraph expression.
        - Dict with `_type` → resolve `_type`, normalize fields, then `post_access(...)`.
        - Untyped dict/list → recursively normalized (private `_...` keys kept raw).
        """
        if is_dataclass(key):
            return key

        elif isinstance(key, str):
            if key not in self.registry:
                return visit_expression(key, self.parse_visitor)
            else:
                return self.registry[key]()

        elif isinstance(key, dict):
            if '_type' in key:
                return self.access_type(key)

            else:
                result = {}
                for subkey in key:
                    if isinstance(subkey, str):
                        result[subkey] = key[subkey] if subkey.startswith('_') else self.access(key[subkey])
                    else:
                        result[subkey] = key[subkey]
                return result

        elif isinstance(key, list):
            return [self.access(element) for element in key]
        else:
            return key

    def infer(self, state, path=()):
        """Infer a schema from example `state` (see `infer.py`); sets `_default` where applicable."""
        return infer(self, state, path=path)

    def render(self, schema):
        """Render a node/schema to a JSON-serializable form (inverse of `access`)."""
        found = self.access(schema)
        return render(found)

    def default(self, schema):
        """Materialize default value for `schema` (`_default` → `deserialize(...)`)."""
        found = self.access(schema)
        value = default(found)
        return deserialize(found, value)

    def resolve(self, current_schema, update_schema):
        """Unify two schemas under node semantics (e.g., Map/Tree/Edge field-wise resolution)."""
        current = self.access(current_schema)
        update = self.access(update_schema)
        return resolve(current, update)

    def check(self, schema, state):
        """Validate `state` against `schema`."""
        found = self.access(schema)
        return check(found, state)

    def serialize(self, schema, state):
        """Encode `state` per `schema` (JSON-friendly)."""
        found = self.access(schema)
        return serialize(found, state)

    def deserialize(self, schema, state):
        """Decode representation into typed value per `schema`."""
        found = self.access(schema)
        return deserialize(found, state)

    def generate(self, schema, state):
        """Compute a resolved schema and defaulted state from partial inputs.

        Equivalent to: `resolved = resolve(infer(state), access(schema)); merged = default(resolved)`.
        Returns `(resolved, merged)`.
        """
        found = self.access(schema)
        inferred = self.infer(state)
        resolved = self.resolve(inferred, found)
        merged = self.default(resolved)
        return resolved, merged

    def unify(self, schema, state):
        found = self.access(schema)
        context = blank_context(found, state, ())
        
        return unify(self, found, state, context)

    def jump(self, schema, state, raw_key):
        """Navigate by logical jump (`Key`/`Index`/`Star`)."""
        found = self.access(schema)
        key = convert_jump(raw_key)
        context = blank_context(found, state, ())
        return jump(found, state, key, context)

    def traverse(self, schema, state, raw_path):
        """Traverse along a resolved path (supports `..` and wildcards) via `convert_path`."""
        found = self.access(schema)
        path = convert_path(raw_path)
        context = blank_context(found, state, path)
        return traverse(found, state, path, context)

    def bind(self, schema, state, raw_key, target):
        """Bind a logical key (jump) to a target."""
        found = self.access(schema)
        key = convert_jump(raw_key)
        return bind(found, state, key, target)

    def merge(self, schema, state, merge_state):
        """Schema-aware merge of `merge_state` into `state`."""
        found = self.access(schema)
        return merge(found, state, merge_state)

    def apply(self, schema, state, update):
        """Apply a schema-aware update/patch; provides minimal context."""
        found = self.access(schema)
        context = {schema: found, state: state, path: ()}
        return apply(schema, state, update, context)


# test data ----------------------------

@pytest.fixture
def core():
    return Core(
        BASE_TYPES)

default_a = 11.111
# represents a hash where keys a and b are required with types specified
node_schema = {
    'a': {
        '_type': 'float',
        '_default': default_a},
    'b': {
        '_type': 'string',
        '_default': 'hello world!'},
    'c': 'array[(3|4),U36]'}

map_schema = {
        '_type': 'map',
        '_key': 'string',
        '_value': 'float'}

edge_schema = {
    '_type': 'edge',
    '_inputs': {
        'mass': 'float',
        'concentrations': map_schema},
    '_outputs': {
        'mass': 'delta',
        'concentrations': {
            '_type': 'map',
            '_key': 'string',
            '_value': 'delta'}}}

edge_a = {
    'inputs': {
        'mass': ['cell', 'mass'],
        'concentrations': ['cell', 'internal']},
    'outputs': {
        'mass': ['cell', 'mass'],
        'concentrations': ['cell', 'internal']}}

# tracking datatypes that should be in the unischema
to_implement = (
    Node,
    # Union,
    # Tuple,
    # Boolean,
    Number,
    # Integer,
    # Float,
    # Delta,
    # Nonnegative,
    # String,
    # Enum,
    # Wrap,
    # Maybe,
    # Overwrite,
    # List,
    # Map,
    # Tree,
    # Array,
    Key,
    # Path,
    # Wires,
    Schema,
    # Edge,
    Jump,
    Star,
    Index,
)

uni_schema = 'outer:tuple[tuple[boolean],' \
        'enum[a,b,c],' \
        'tuple[integer,delta,nonnegative],' \
        'list[maybe[tree[path]]],' \
        'wrap[maybe[overwrite[integer]]],' \
        'union[edge[x:integer,y:string],float,string],' \
        'path,' \
        'tree[edge[x:(y:float|z:boolean)|y:integer,oo:maybe[string]]],' \
        'wires,' \
        'integer{11},' \
        'a:string|b:float,' \
        'map[a:string|c:float]]|' \
        'outest:string'
        # 'list[maybe[tree[array[(3|4),float64]]]],' \

# tests --------------------------------------

def do_round_trip(core, schema):
    # generate a schema object from string expression
    type_ = core.access(schema)
    # generate a json object representing schema
    reified = core.render(type_)
    # finally, create another schema object
    round_trip = core.access(reified)
    final = core.render(round_trip)

    return type_, reified, round_trip, final

def _test_problem_schema_0(core):
    # providing 'float' as a dtype breaks the parser
    problem_schema = 'array[3,float]'
    problem_type, reified, round_trip, final = do_round_trip(core, problem_schema)
    assert not isinstance(problem_type, str)
    assert round_trip == problem_type

def test_problem_schema_1(core):
    # this round trip is broken, shape 3 vs. (3,)
    problem_schema = 'array[3,float64]'
    problem_type, reified, round_trip, final = \
            do_round_trip(core, problem_schema)
    assert isinstance(round_trip._data, dtype)
    assert round_trip == problem_type

def test_problem_schema_2(core):
    # turns (3, int) into ('', '<i8')
    problem_schema = 'array[3,int]'
    problem_type, reified, round_trip, final = do_round_trip(core, problem_schema)
    assert not isinstance(problem_type, str)
    assert round_trip == problem_type

def test_array(core):
    complex_spec = [('name', np.str_, 16),
                    ('grades', np.float64, (2,))]
    complex_dtype = dtype(complex_spec)
    array = np.zeros((3,4), dtype=complex_dtype)
    array_schema = core.infer(array)
    rendered = core.render(array_schema)

def test_infer(core):
    default_node = core.default(node_schema)
    node_inferred = core.infer(default_node)
    assert check(node_inferred, default_node)
    # print(f"inferred {node_inferred}\nfrom {default_node}")
    # print(f'rendered schema:\n{render(node_inferred)}')
    # assert render(node_inferred)['a'] == node_schema['a']['_type']
    # assert render(node_inferred)['b'] == node_schema['b']['_type']

# render is the inverse of access
def test_render(core):
    node_type = core.access(node_schema)
    node_render = core.render(node_schema)
    assert node_render == render(node_type)

    edge_type = core.access(edge_schema)
    edge_render = core.render(edge_type)

    # can't do the same assertion as above, because two different renderings
    # exist
    assert core.access(edge_render) == edge_type
    assert edge_render == core.render(core.access(edge_render))

    map_type = core.access(map_schema)
    map_render = core.render(map_type)
    assert core.access(map_render) == core.access(map_schema)
    # fixed point is found
    assert map_render == core.render(core.access(map_render))

def test_uni_schema(core):
    uni_type = core.access(uni_schema)
    assert not isinstance(uni_type, str)

    uni_render = core.render(uni_type)
    round_trip = core.access(uni_render)

    def idx(a, b, n):
        return a['outer']._values[n], b['outer']._values[n]

    assert round_trip == uni_type
    assert uni_render == core.render(core.access(uni_type))

def test_default(core):
    node_type = core.access(node_schema)
    default_node = core.default(node_schema)
    assert 'a' in default_node
    assert isinstance(default_node['a'], float)
    assert default_node['a'] == default_a
    assert 'b' in default_node
    assert isinstance(default_node['b'], str)
    assert core.check(node_schema, default_node)

    value = 11.11
    assert core.default(core.infer(value)) == value

def test_resolve(core):
    float_number = core.resolve('float', 'number')
    assert render(float_number) == 'float'
    assert type(float_number) == BASE_TYPES['float']

    node_resolve = core.resolve(
        {'a': 'delta', 'b': 'node'},
        node_schema)
    rendered_a = render(node_resolve)['a']
    assert rendered_a['_type'] == 'delta'
    assert core.access(rendered_a)._default == node_schema['a']['_default']

    mutual = core.resolve(
        {'a': 'float', 'b': 'string'},
        {'b': 'wrap[string]', 'c': 'boolean'})
    assert 'a' in mutual
    assert 'b' in mutual
    assert 'c' in mutual

    failed = False
    try:
        core.resolve(
            {'a': 'map[string]', 'b': 'node'},
            node_schema)
    except Exception as e:
        failed = True
    assert failed

def test_check(core):
    tree_a = {
        'a': {
            'b': 5.5},
        'c': 3.3}

    tree_b = {
        'a': {
            'b': 0.111,
            'x': 444.444},
        'd': 11.11}

    tree_schema = {
        '_type': 'tree',
        '_leaf': 'float'}

    tree_parse = 'tree[float]'
    tree_type = core.access(
        tree_parse)

    assert core.check(tree_schema, tree_a)
    assert core.check(tree_parse, tree_b)
    assert not core.check(tree_schema,'not a tree')

    edge_a = {
        'inputs': {
            'mass': ['cell', 'mass'],
            'concentrations': ['cell', 'internal']},
        'outputs': {
            'mass': ['cell', 'mass'],
            'concentrations': ['cell', 'internal']}}

    edge_b = {
        'inputs': 5.0,
        'outputs': {
            'mass': ['cell', 'mass'],
            'concentrations': ['cell', 'internal']}}

    edge_c = {
        'outputs': {
            'mass': ['cell', 'mass'],
            'concentrations': ['cell', 'internal']}}

    edge_d = {
        'inputs': {
            'mass': ['cell', 11.111],
            'concentrations': ['cell', 'internal']},
        'outputs': {
            'mass': ['cell', 'mass'],
            'concentrations': ['cell', 'internal']}}

    assert core.check(edge_schema, edge_a)
    assert not core.check(edge_schema, edge_b)
    assert not core.check(edge_schema, edge_c)
    assert not core.check(edge_schema, edge_d)
    assert not core.check(edge_schema, 44.44444)

def test_serialize(core):
    edge_type = core.access(edge_schema)
    encoded_a = serialize(edge_type, edge_a)

    assert encoded_a == edge_a
    encoded_b = core.serialize(
        {'a': 'float'},
        {'a': 55.55555})
    assert encoded_b['a'] == 55.55555

def test_deserialize(core):
    encoded_edge = {
        'inputs': {
            'mass': ['cell','mass'],
            'concentrations': '["cell","internal"]'},
        'outputs': '{\
            "mass":["cell","mass"],\
            "concentrations":["cell","internal"]}'}
    decoded = core.deserialize(edge_schema, encoded_edge)
    assert decoded == edge_a

    schema = {
        'a': 'integer',
        'b': 'tuple[float,string,map[integer]]'}
    code = {
        'a': '5555',
        'b': ('1111.1', "okay", '{"x": 5, "y": "11"}')}
    decode = core.deserialize(schema, code)
    assert decode['a'] == 5555
    assert decode['b'][2]['y'] == 11

def test_infer_edge(core):
    edge_state = {
        'edge': {
            '_type': 'edge',
            '_inputs': {
                'n': 'float',
                'x': {
                    'y': 'string'}},
            '_outputs': {
                'z': 'string'},
            'inputs': {
                'n': ['A'],
                'x': ['E']},
            'outputs': {
                'z': ['F', 'f', 'ff']}}}

    edge_schema = core.infer(edge_state)
    import ipdb; ipdb.set_trace()


def test_traverse(core):
    tree_a = {
        'a': {
            'b': 5.5,
            'y': 555.55,
            'x': {'further': {'down': 111111.111}}},
        'c': 3.3}
    further_schema, further_state = core.traverse(
        'tree[float]',
        tree_a,
        ['a', 'x', 'further'])
    assert isinstance(further_schema, Tree)
    assert further_state == {'down': 111111.111}

    down_schema, down_state = core.traverse(
        'tree[float]',
        tree_a,
        ['a', 'x', 'further', 'down'])
    assert isinstance(down_schema, Float)
    assert down_state == 111111.111

    star_schema, star_state = core.traverse(
        {'_type': 'map', '_value': {'a': 'float', 'b': 'string'}},
        {'X': {'a': 5.5, 'b': 'green'},
         'Y': {'a': 11.11, 'b': 'another green'},
         'Z': {'a': 22.2222, 'b': 'yet another green'}},
        ['*', 'a'])
    assert isinstance(star_schema, Map)
    assert isinstance(star_schema._value, Float)
    assert star_state['Y'] == 11.11
    assert 'Z' in star_state

    puts = {
        'mass': 'float',
        'concentrations': 'map[float]'}

    edge_interface = {
        '_type': 'edge',
        '_inputs': puts,
        '_outputs': puts}

    edge_schema = core.access(
        edge_interface)

    edge_state = {
        'inputs': {
            'mass': ['cell', 'mass'],
            'concentrations': ['cell', 'internal']},
        'outputs': {
            'mass': ['cell', 'mass'],
            'concentrations': ['cell', 'internal']}}
    assert core.check(edge_interface, edge_state)

    default_edge = core.default(edge_schema)
    assert default_edge['inputs']['mass'] == ['mass']

    simple_interface = {
        'cell': {
            'mass': 'float',
            'internal': 'map[float]'},
        'edge': edge_interface}

    initial_mass = 11.1111

    simple_graph = {
        'cell': {
            'mass': initial_mass,
            'internal': {
                'A': 3.333,
                'B': 44.44444,
                'C': 5555.555}},
        'edge': edge_state}

    simple_schema = core.access(
        simple_interface)

    down_schema, down_state = core.jump(
        simple_interface,
        simple_graph,
        'edge')
    assert isinstance(down_schema, Edge)
    assert 'inputs' in down_state

    mass_schema, mass_state = core.traverse(
        simple_interface,
        simple_graph,
        ['edge', 'inputs', 'mass'])
    assert isinstance(mass_schema, Float)
    assert mass_state == initial_mass

    concentration_schema, concentration_state = core.traverse(
        simple_interface,
        simple_graph,
        ['edge', 'outputs', 'concentrations', 'A'])
    assert isinstance(concentration_schema, Float)
    assert concentration_state == simple_graph['cell']['internal']['A']


def test_generate(core):
    schema = {
        'A': 'float',
        'B': 'enum[one,two,three]',
        'D': 'string{hello}',
        'units': 'map[number]'}

    state = {
        'C': {
            '_type': 'enum[x,y,z]',
            '_default': 'y'},

        'concentrations': {
            'glucose': 0.5353533},

        'edge': {
            '_type': 'edge',
            '_inputs': {
                'n': 'float',
                'x': 'string'},
            '_outputs': {
                'z': 'string'},
            'inputs': {
                'n': ['A'],
                'x': ['E']},
            'outputs': {
                'z': ['F', 'f', '_ff']}},

        'units': {
            'meters': 11.1111,
            'seconds': 22.833333}}

    generated_schema, generated_state = core.generate(schema, state)
    assert generated_state['A'] == 0.0
    assert generated_state['B'] == 'one'
    assert generated_state['C'] == 'y'
    assert generated_state['units']['seconds'] == 22.833333
    assert not hasattr(generated_schema['units'], 'meters')

    rendered = core.render(generated_schema)
    assert generated_state == \
            core.deserialize(generated_schema,
                             core.serialize(generated_schema, generated_state))

def test_unify(core):
    schema = {
        'A': 'float',
        'B': 'enum[one,two,three]',
        'D': 'string{hello}',
        'units': 'map[number]',
        'inner': {
            'edge': {
                '_type': 'edge',
                '_inputs': {
                    'n': 'float',
                    'x': {
                        'xx': 'string',
                        'xy': 'boolean'}},
                '_outputs': {
                    'z': 'string'}}}}

    state = {
        'C': {
            '_type': 'enum[x,y,z]',
            '_default': 'y'},

        'concentrations': {
            'glucose': 0.5353533},

        'inner': {
            'edge': {
                'inputs': {
                    'n': ['..', 'A'],
                    'x': {
                        '_path': ['W'],
                        'xy': ['G']}},
                'outputs': {
                    'z': ['F', 'f', '_ff']}}},

        'units': {
            'meters': 11.1111,
            'seconds': 22.833333}}

    generated_schema, generated_state = core.unify(
        schema,
        state)

    assert generated_state['A'] == 0.0
    assert generated_state['B'] == 'one'
    assert generated_state['C'] == 'y'
    assert generated_state['units']['seconds'] == 22.833333

    assert not hasattr(generated_schema['units'], 'meters')

    rendered = core.render(generated_schema)

    assert generated_state == \
            core.deserialize(generated_schema,
                             core.serialize(generated_schema, generated_state))

def test_generate_coverage(core):
    # tracking datatypes that should be covered in this test
    to_implement = (
            Node,
            # Union,
            Tuple,
            Boolean,
            # Number,
            Integer,
            # Float,
            Delta,
            Nonnegative,
            # String,
            # Enum,
            Wrap,
            Maybe,
            Overwrite,
            List,
            # Map,
            Tree,
            Array,
            Key,
            Path,
            Wires,
            Schema,
            Edge,
            Jump,
            Star,
            Index,
            )
    schema = {
            'A': 'edge[x:integer,y:nonnegative]'}

    state = {
            'B': {
                '_type': 'boolean',
                '_default': True},
            'C': {
                '_type': 'tuple[number,number]',
                '_default': (0,0)}}

    generated_schema, generated_state = core.generate(schema, state)

    assert generated_state == \
            core.deserialize(generated_schema,
                             core.serialize(generated_schema, generated_state))


def broken_test_generate_tuple_default(core):
    schema = {
            'A': 'edge[x:integer,y:nonnegative]'}

    state = {
            'B': {
                '_type': 'boolean',
                '_default': True},
            'C': {
                '_type': 'tuple[number,number]',
                '_default': (0,0)}}

    generated_schema, generated_state = core.generate(schema, state)
    assert generated_state['C'] == (0,0)


def test_generate_promote_to_struct(core):
    """
    a map schema should update to a struct schema when merged with
    a struct containing incompatible fields
    """
    # TODO - test the doppleganger dict/Map vs. Map/dict
    # TODO - this should also happen to trees
    schema = {
            'A': 'edge[x:integer,y:nonnegative]'}
    state = {
            'B': {
                '_type': 'boolean',
                '_default': True}}
    generated_schema, generated_state = core.generate(schema, state)
    serialized = core.serialize(generated_schema, generated_state)
    assert generated_state == core.deserialize(
        generated_schema,
        serialized)

def test_bind(core):
    core

def test_merge(core):
    tree_a = {
        'a': {
            'b': 5.5,
            'y': 555.55,
            'x': {'further': {'down': 111111.111}}},
        'c': 3.3}

    tree_b = {
        'a': {
            'b': 0.111,
            'z': 999999.4444,
            'x': 444.444},
        'd': 11.11}

    tree_merge = core.merge('tree[float]', tree_b, tree_a)
    assert(tree_merge['a']['x']['further']['down'])

    key_merge = core.merge(
        {'a': 'float', 'b': 'string'},
        {'a': 333.333, 'c': 4444},
        {'a': 55555.555, 'd': '111111'})

    def inputs(self):
        return {
            'mass': 'wrap[float]'} 

    assert(key_merge == {
        'a': 55555.555,
        'b': '',
        'c': 4444,
        'd': '111111'})

def test_apply(core):
    core


if __name__ == '__main__':
    core = Core(
        BASE_TYPES)

    test_infer(core)
    test_render(core)
    test_default(core)
    test_resolve(core)
    test_check(core)
    test_serialize(core)
    test_deserialize(core)
    test_merge(core)
    test_traverse(core)
    test_infer_edge(core)
    test_generate(core)
    test_generate_promote_to_struct(core)
    test_uni_schema(core)
    test_array(core)
    test_bind(core)

    test_problem_schema_1(core)
    # _test_problem_schema_0(core)
    test_problem_schema_2(core)

    test_apply(core)
    test_unify(core)

