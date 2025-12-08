"""
Bigraph-Schema Core
===================

This module defines the **Core** class — the main operational interface for
`bigraph-schema`. Core manages the translation between *compiled*
(dataclass-based) and *encoded* (JSON-compatible) representations of both
**schemas** and **states**.

Core provides a consistent API for all major transformations:
- `access` / `render`: parse and serialize schema definitions
- `default` / `infer`: connect schemas to example states
- `serialize` / `deserialize`: encode and decode state data

These methods form a reversible, type-aware layer for schema construction,
validation, and data transformation.

`CoreVisitor` implements the parsing backend, converting textual bigraph
expressions into structured schema nodes (`Union`, `Tuple`, `Array`, `Link`, etc.).
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
    nest,
    Node,
    Union,
    Tuple,
    Boolean,
    Or,
    And,
    Xor,
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
    Link,
    Jump,
    Star,
    Index,
)

from bigraph_schema.registry import deep_merge, set_star_path
from bigraph_schema.parse import visit_expression
from bigraph_schema.edge import Edge
from bigraph_schema.methods import (
    reify_schema,
    handle_parameters,
    infer,
    render,
    default,
    resolve,
    generalize,
    check,
    serialize,
    deserialize,
    merge,
    jump,
    traverse,
    bind,
    unify,
    apply)


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
    `link[a:int|b:string]`, `(x:y|z:w)`) into instances of `Union`, `Tuple`,
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
        self.link_registry = {}
        self.method_registry = {}
        self.parse_visitor = CoreVisitor(self)

        self.register_link('edge', Edge)

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

    def register_link(self, key, link):
        if key in self.registry:
            self.update_link(key, link)
        else:
            self.link_registry[key] = link

    def register_links(self, links):
        """Bulk register multiple link into the operation registry."""
        for key, link in links.items():
            self.register_link(key, link)

    def update_link(self, key, data):
        """Deep-merge metadata/overrides into an existing registry entry."""
        self.link_registry[key] = data

    def register_method(self, key, data):
        self.method_registry[key] = data

    def call_method(self, key, *args, **kwargs):
        method = self.method_registry.get(key)
        if method is None:
            raise Exception(f'no method {key} in the method registry')

        return method(self, *args, **kwargs)

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
            schema = self.access(value['_type'])

            default_value = None
            if '_default' in value:
                default_value = value['_default']
            elif isinstance(schema, Node) and schema._default is not None:
                default_value = schema._default

            schema = replace(
                schema,
                **{'_default': default_value})

            parameters = {}
            for key in schema_keys(schema)[1:]:
                if key in value:
                    parameters[key] = value[key]
            schema = reify_schema(self, schema, parameters)
            return schema

    def access(self, key):
        """Interpret an encoded schema or object and produce a compiled node.

        Converts strings, dicts, or lists into dataclass-based schema instances.
        Acts as the main entry point for parsing bigraph expressions and building
        normalized in-memory representations.
        """

        # TODO: consider other terms for this?
        #   * compile
        #   * parse

        if is_dataclass(key):
            return key

        elif isinstance(key, str):
            if key not in self.registry:
                try:
                    return visit_expression(key, self.parse_visitor)
                except Exception as e:
                    import ipdb; ipdb.set_trace()
            else:
                entry = self.registry[key]
                if callable(entry):
                    return entry()
                elif isinstance(entry, dict):
                    if '_inherit' in entry:
                        inherit_key = entry['_inherit']
                        beyond_inherit = {
                            key: value
                            for key, value in entry.items()
                            if key != '_inherit'}
                        inherit = self.access(inherit_key)
                        return replace(inherit, **beyond_inherit)
                    else:
                        return self.access(entry)

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

    def infer_merges(self, state, path=()):
        """Derive a schema that matches the structure of an example state.

        Analyzes values to infer types, shapes, and nested relationships, generating
        a schema node that captures the structure of the provided data.
        """
        return infer(self, state, path=path)

    def infer(self, state, path=()):
        """Derive a schema that matches the structure of an example state.

        Analyzes values to infer types, shapes, and nested relationships, generating
        a schema node that captures the structure of the provided data.
        """
        schema, merges = infer(self, state, path=path)
        merge_schema = self.handle_merges([schema] + merges)
        return merge_schema
        # for merge in merges:
        #     schema = resolve(schema, merge)
        # return schema

    def render(self, schema, defaults=False):
        """Produce a serializable view of a compiled schema.

        Converts internal dataclass nodes into JSON-friendly dicts or strings.
        This is the inverse of `access()`, ensuring round-trip fidelity between
        code representations and stored schema definitions.
        """
        found = self.access(schema)
        return render(found, defaults=defaults)

    def default_merges(self, schema, path=()):
        """Generate a representative state that satisfies a schema.

        Uses type defaults and explicit `_default` values to instantiate an example
        state consistent with the given schema.
        """
        found = self.access(schema)
        value = default(found)
        return deserialize(self, found, value, path=path)

    def default(self, schema, path=()):
        found = self.access(schema)
        value = default(found)
        return self.deserialize(found, value, path=path)

    def resolve(self, current_schema, update_schema):
        """Unify two schemas under node semantics (e.g., Map/Tree/Link field-wise resolution)."""
        current = self.access(current_schema)
        update = self.access(update_schema)
        return resolve(current, update)

    def generalize(self, current_schema, update_schema):
        """Unify two schemas under node semantics (e.g., Map/Tree/Link field-wise resolution)."""
        current = self.access(current_schema)
        update = self.access(update_schema)
        return generalize(current, update)

    def check(self, schema, state):
        """Check that the `state` fits the `schema`."""
        found = self.access(schema)
        return check(found, state)

    def validate(self, schema, state):
        """Raise an exception if the state does not match the schema,
        which describes why"""

        pass

    def serialize(self, schema, state):
        """Convert a structured Python state into an encoded representation.

        Encodes typed values into JSON-compatible primitives while respecting the
        schema’s structure and constraints.
        """
        found = self.access(schema)
        return serialize(found, state)

    def deserialize(self, schema, state, path=()):
        """Convert an encoded representation back into structured Python values.

        Decodes strings, numbers, and nested structures into their appropriate types,
        guided by the provided schema.
        """
        found = self.access(schema)

        decode_schema, decode_state, merges = deserialize(
            self,
            found,
            state,
            path=path)

        if merges:
            merge_schema = self.handle_merges(merges)
            decode_schema = self.generalize(decode_schema, merge_schema)
            merge_state = self.fill(merge_schema, decode_state)
        else:
            merge_state = decode_state

        return decode_schema, merge_state

    def generate(self, schema, state):
        """Combine schema inference, resolution, and defaulting.

        Produces a resolved schema and a corresponding defaulted state from partial
        inputs. Equivalent to:
            resolve(infer(state), access(schema)) → default(...)
        Returns a `(resolved_schema, completed_state)` pair.
        """
        found = self.access(schema)
        inferred = self.infer(state)
        resolved = self.resolve(inferred, found)
        merge_schema, merge_state = self.default(resolved)

        return merge_schema, merge_state

    def handle_merges(self, merges):
        if len(merges) == 0:
            return None
        else:
            schema = merges[0]
            for merge in merges[1:]:
                schema = self.generalize(schema, merge)

            return schema

    def unify(self, schema, state, path=()):
        found = self.access(schema)
        
        unify_schema, unify_state, merges = unify(
            self,
            found,
            state,
            path)

        unify_schema = self.handle_merges(
            [unify_schema] + merges)
        default_schema, default_state = self.default(unify_schema)

        return default_schema, default_state

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

    def fill(self, schema, state, overwrite=False):
        found = self.access(schema)
        base_schema, base_state = self.default(found)
        if overwrite:
            return merge(base_schema, state, base_state)
        else:
            return merge(base_schema, base_state, state)

    def view_ports(self, schema, state, path, ports_schema, wires):
        if isinstance(wires, str):
            wires = [wires]

        if isinstance(wires, (list, tuple)):
            _, result = self.traverse(schema, state, list(path) + list(wires))

        elif isinstance(wires, dict):
            result = {}
            for port_key, subport in wires.items():
                subschema, subwires = self.jump(
                    schema,
                    wires,
                    port_key)

                inner_view = self.view_ports(
                    schema,
                    state,
                    path,
                    subschema,
                    subwires)

                if inner_view is not None:
                    result[port_key] = inner_view

        else:
            raise Exception(f'trying to view state at path {path} with these ports:\n{ports_schema}\nbut not sure what these wires are:\n{wires}')

        return result

    def view(self, schema, state, link_path, ports_key='inputs'):
        found = self.access(schema)
        link_schema, link_state = self.traverse(schema, state, link_path)
        ports_schema = getattr(link_schema, f'_{ports_key}')
        wires = link_state.get(ports_key) or {}
        view = self.view_ports(
            schema,
            state,
            link_path[:-1],
            ports_schema,
            wires)

        return view

    def project_ports(self, ports_schema, wires, path, view):
        project_schema = {}
        project_state = {}

        if isinstance(wires, str):
            wires = [wires]

        if isinstance(wires, (list, tuple)):
            destination = resolve_path(list(path) + list(wires))

            project_schema = set_star_path(
                project_schema,
                destination,
                ports_schema)

            project_state = set_star_path(
                project_state,
                destination,
                view)

        elif isinstance(wires, dict):
            if isinstance(view, list):
                result = [
                    self.project_ports(ports_schema, wires, path, state)
                    for state in view]
                project_schema = Tuple(_values=[
                    item[0]
                    for item in result])
                project_state = [
                    item[1]
                    for item in result]
            else:
                branches = []
                for key, subwires in wires.items():
                    subports, subview = self.jump(ports_schema, view, key)
                    if subview is None:
                        continue

                    subschema, substate = self.project_ports(
                        subports,
                        subwires,
                        path,
                        subview)

                    if substate is not None:
                        branches.append((subschema, substate))

                project_schema = Node()
                project_state = {}
                for branch_schema, branch_state in branches:
                    project_schema = resolve(project_schema, branch_schema)
                    project_state = self.merge(project_schema, project_state, branch_state)

        else:
            raise Exception(
                f'inverting state\n  {view}\naccording to ports schema\n  {ports_schema}\nbut wires are not recognized\n  {wires}')

        return project_schema, project_state

    def project(self, schema, state, link_path, view, ports_key='outputs'):
        found = self.access(schema)
        link_schema, link_state = self.traverse(schema, state, link_path)
        ports_schema = getattr(link_schema, f'_{ports_key}')
        wires = link_state.get(ports_key) or {}
        project_schema, project_state = self.project_ports(
            ports_schema,
            wires,
            link_path[:-1],
            view)

        return project_schema, project_state

    def combine(self, schema, state, update_schema, update_state):
        resolved = self.resolve(schema, update_schema)
        merged = self.merge(resolved, state, update_state)
        decode_schema, decode_state = self.deserialize(resolved, merged)

        return decode_schema, decode_state

        # return resolved, merged

    def link_state(self, link, path):
        result_schema = {}
        result_state = {}

        instance = link.get('instance')

        if instance is not None:
            initial_state = instance.initial_state()

            for ports_key in ['inputs', 'outputs']:
                ports_schema = link.get(f'_{ports_key}', {})
                wires = link.get(ports_key, {})
                project_schema, project_state = self.project_ports(ports_schema, wires, path[:-1], initial_state)
                result_schema, result_state = self.combine(
                    result_schema, result_state,
                    project_schema, project_state)

        return result_schema, result_state

    def wire_schema(self, schema, state, wires, path=None):
        outcome = {}
        path = path or []

        if isinstance(wires, dict):
            for key, subwires in wires.items():
                outcome[key] = self.wire_schema(
                    schema,
                    state,
                    wires[key],
                    path)

        else:
            outcome, _ = self.traverse(schema, state, path + wires)

        return outcome

    def apply(self, schema, state, update, path=()):
        """Apply a schema-aware update/patch; provides minimal context."""
        if update:
            found = self.access(schema)
            return apply(found, state, update, path)
        else:
            return state, []


# test data ----------------------------

# @pytest.fixture
# def core():
#     return Core(
#         BASE_TYPES)

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

link_schema = {
    '_type': 'link',
    '_inputs': {
        'mass': 'float',
        'concentrations': map_schema},
    '_outputs': {
        'mass': 'delta',
        'concentrations': {
            '_type': 'map',
            '_key': 'string',
            '_value': 'delta'}}}

link_a = {
    'address': 'local:edge',
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
    # Link,
    Jump,
    Star,
    Index,
)

uni_schema = 'outer:tuple[tuple[boolean],' \
        'enum[a,b,c],' \
        'tuple[integer,delta,nonnegative],' \
        'list[maybe[tree[path]]],' \
        'wrap[maybe[overwrite[integer]]],' \
        'path,' \
        'wires,' \
        'integer{11},' \
        'union[link[x:integer,y:string],float,string],' \
        'tree[link[x:(y:float|z:boolean)|y:integer,oo:maybe[string]]],' \
        'a:string|b:float,' \
        'map[a:string|c:float]]|' \
        'outest:string'
        # 'list[maybe[tree[array[(3|4),float64]]]],' \

# tests --------------------------------------

def do_round_trip(core, schema):
    # generate a schema object from string expression
    type_ = core.access(schema)
    # generate a json object representing schema
    reified = core.render(type_, defaults=True)
    # finally, create another schema object
    round_trip = core.access(reified)
    final = core.render(round_trip, defaults=True)

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
    default_schema, default_state = core.default(node_schema)
    node_inferred = core.infer(default_state)
    assert check(node_inferred, default_state)

    # print(f"inferred {node_inferred}\nfrom {default_node}")
    # print(f'rendered schema:\n{render(node_inferred)}')
    # assert render(node_inferred)['a'] == node_schema['a']['_type']
    # assert render(node_inferred)['b'] == node_schema['b']['_type']

# render is the inverse of access
def test_render(core):
    node_type = core.access(node_schema)
    node_render = core.render(node_schema, defaults=True)
    assert node_render == render(node_type, defaults=True)

    link_type = core.access(link_schema)
    link_render = core.render(link_type, defaults=True)

    # can't do the same assertion as above, because two different renderings
    # exist
    assert core.access(link_render) == link_type
    assert link_render == core.render(core.access(link_render), defaults=True)

    map_type = core.access(map_schema)
    map_render = core.render(map_type, defaults=True)
    assert core.access(map_render) == core.access(map_schema)
    # fixed point is found
    assert map_render == core.render(core.access(map_render), defaults=True)

def test_uni_schema(core):
    uni_type = core.access(uni_schema)
    assert not isinstance(uni_type, str)

    uni_render = core.render(uni_type, defaults=True)
    round_trip = core.access(uni_render)

    def idx(a, b, n):
        return a['outer']._values[n], b['outer']._values[n]

    # assert round_trip == uni_type
    assert uni_render == core.render(core.access(uni_type), defaults=True)

def test_default(core):
    node_type = core.access(node_schema)
    default_schema, default_state = core.default(node_schema)
    assert 'a' in default_state
    assert isinstance(default_state['a'], float)
    assert default_state['a'] == default_a
    assert 'b' in default_state
    assert isinstance(default_state['b'], str)
    assert core.check(node_schema, default_state)

    value = 11.11
    assert core.default(core.infer(value))[1] == value

def test_resolve(core):
    float_number = core.resolve('float', 'number')
    assert render(float_number) == 'float'
    assert type(float_number) == BASE_TYPES['float']

    node_resolve = core.resolve(
        {'a': 'delta', 'b': 'node'},
        node_schema)
    rendered_a = render(node_resolve, defaults=True)['a']
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

    link_a = {
        'address': 'local:edge',
        'config': {},
        'inputs': {
            'mass': ['cell', 'mass'],
            'concentrations': ['cell', 'internal']},
        'outputs': {
            'mass': ['cell', 'mass'],
            'concentrations': ['cell', 'internal']}}

    link_b = {
        'inputs': 5.0,
        'outputs': {
            'mass': ['cell', 'mass'],
            'concentrations': ['cell', 'internal']}}

    link_c = {
        'outputs': {
            'mass': ['cell', 'mass'],
            'concentrations': ['cell', 'internal']}}

    link_d = {
        'inputs': {
            'mass': ['cell', 11.111],
            'concentrations': ['cell', 'internal']},
        'outputs': {
            'mass': ['cell', 'mass'],
            'concentrations': ['cell', 'internal']}}

    assert not core.check(link_schema, link_a)
    assert not core.check(link_schema, link_b)
    assert not core.check(link_schema, link_c)
    assert not core.check(link_schema, link_d)
    assert not core.check(link_schema, 44.44444)

    _, a_instance = core.deserialize(link_schema, link_a)
    _, b_instance = core.deserialize(link_schema, link_b)
    _, c_instance = core.deserialize(link_schema, link_c)
    # _, d_instance = core.deserialize(link_schema, link_d)

    assert core.check(link_schema, a_instance)
    assert core.check(link_schema, b_instance)
    assert core.check(link_schema, c_instance)
    # assert not core.check(link_schema, d_instance)


def test_serialize(core):
    link_type = core.access(link_schema)
    encoded_a = serialize(link_type, link_a)

    assert encoded_a['address'] == 'local:edge'
    assert encoded_a['_inputs'] == {'concentrations': 'map[float]', 'mass': 'float'}

    encoded_b = core.serialize(
        {'a': 'float'},
        {'a': 55.55555})

    assert encoded_b['a'] == 55.55555

def test_deserialize(core):
    encoded_link = {
        'inputs': {
            'mass': ['cell','mass'],
            'concentrations': '["cell","internal"]'},
        'outputs': '{\
            "mass":["cell","mass"],\
            "concentrations":["cell","internal"]}'}

    decoded_schema, decoded_state = core.deserialize(link_schema, encoded_link)

    assert isinstance(decoded_state['instance'], Edge)

    schema = {
        'a': 'integer',
        'b': 'tuple[float,string,map[integer]]'}
    code = {
        'a': '5555',
        'b': ('1111.1', "okay", '{"x": 5, "y": "11"}')}

    decoded_schema, decoded_state = core.deserialize(schema, code)
    assert decoded_state['a'] == 5555
    assert decoded_state['b'][2]['y'] == 11

def test_infer_link(core):
    link_state = {
        'link': {
            '_type': 'link',
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

    link_schema = core.infer(link_state)

    assert 'A' in link_schema and isinstance(link_schema['A'], Float)
    assert 'E' in link_schema and isinstance(link_schema['E']['y'], String)


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

    link_interface = {
        '_type': 'link',
        '_inputs': puts,
        '_outputs': puts}

    link_schema = core.access(
        link_interface)

    link_state = {
        'inputs': {
            'mass': ['cell', 'mass'],
            'concentrations': ['cell', 'internal']},
        'outputs': {
            'mass': ['cell', 'mass'],
            'concentrations': ['cell', 'internal']}}

    default_schema, default_link = core.default(link_schema)
    assert default_link['inputs']['mass'] == ['mass']

    simple_interface = {
        'cell': {
            'mass': 'float',
            'internal': 'map[float]'},
        'link': link_interface}

    initial_mass = 11.1111

    simple_graph = {
        'cell': {
            'mass': initial_mass,
            'internal': {
                'A': 3.333,
                'B': 44.44444,
                'C': 5555.555}},
        'link': link_state}

    simple_schema = core.access(
        simple_interface)

    down_schema, down_state = core.jump(
        simple_interface,
        simple_graph,
        'link')
    assert isinstance(down_schema, Link)
    assert 'inputs' in down_state

    mass_schema, mass_state = core.traverse(
        simple_interface,
        simple_graph,
        ['link', 'inputs', 'mass'])
    assert isinstance(mass_schema, Float)
    assert mass_state == initial_mass

    concentration_schema, concentration_state = core.traverse(
        simple_interface,
        simple_graph,
        ['link', 'outputs', 'concentrations', 'A'])
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

        'link': {
            '_type': 'link',
            '_inputs': {
                'n': 'float{5.5}',
                'x': 'string{what}'},
            '_outputs': {
                'z': 'string{world}'},
            'inputs': {
                'n': ['A'],
                'x': ['E']},
            'outputs': {
                'z': ['F', 'f', 'ff']}},

        'units': {
            'meters': 11.1111,
            'seconds': 22.833333}}

    generated_schema, generated_state = core.unify(schema, state)

    assert generated_state['A'] == 5.5
    assert generated_state['B'] == 'one'
    assert generated_state['C'] == 'y'
    assert generated_state['units']['seconds'] == 22.833333
    assert not hasattr(generated_schema['units'], 'meters')

    view = core.view(generated_schema, generated_state, ['link'])
    assert view['n'] == 5.5

    rendered = core.render(generated_schema, defaults=True)
    # assert generated_state == \
    #         core.deserialize(generated_schema,
    #                          core.serialize(generated_schema, generated_state))

def test_unify(core):
    default_hello = 'string{hello}'

    default_hello = {
        '_type': 'string',
        '_default': 'hello'}

    schema = {
        'A': 'float',
        'B': 'enum[one,two,three]',
        'D': 'string{hello}',
        'units': 'map[number]',
        'inner': {
            'G': 'boolean{true}',
            'link': {
                '_type': 'link',
                '_inputs': {
                    'n': 'float{3.333}',
                    'v': 'overwrite[string]',
                    'x': {
                        'xx': 'string{what}',
                        'xy': 'xor'}},
                '_outputs': {
                    'z': 'string{world}'}}}}

    state = {
        'C': {
            '_type': 'enum[x,y,z]',
            '_default': 'y'},

        'concentrations': {
            'glucose': 0.5353533},

        'inner': {
            'link': {
                'inputs': {
                    'n': ['..', 'A'],
                    'v': ['..', 'D'],
                    'x': {
                        'xx': ['W', 'w'],
                        'xy': ['G']}},
                'outputs': {
                    'z': ['F', 'f', 'ffff']}}},

        'units': {
            'meters': 11.1111,
            'seconds': 22.833333}}

    generated_schema, generated_state = core.unify(
        schema,
        state)

    assert generated_state['A'] == 3.333
    assert generated_state['B'] == 'one'
    assert generated_state['C'] == 'y'
    assert generated_state['units']['seconds'] == 22.833333

    assert not hasattr(generated_schema['units'], 'meters')

    rendered = core.render(generated_schema, defaults=True)

    serialized = core.serialize(generated_schema, generated_state)
    deserialized = core.deserialize(generated_schema, serialized)

    # assert generated_state == deserialized

    link_view = core.view(
        generated_schema,
        generated_state,
        ['inner', 'link'])

    project_schema, project_state = core.project(
        generated_schema,
        generated_state,
        ['inner', 'link'],
        link_view,
        ports_key='inputs')

    assert project_state['A'] == generated_state['A']

    # project_schema['inner']['G'] = Xor()
    project_state['D'] = 'OVER'
    applied_state, merges = core.apply(project_schema, generated_state, project_state)

    assert applied_state['inner']['G'] == False
    assert applied_state['D'] == 'OVER'

    assert 'link' in applied_state['inner']

    deschema, destate = core.deserialize(schema, state)

    import ipdb; ipdb.set_trace()


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
            Link,
            Jump,
            Star,
            Index,
            )
    schema = {
            'A': 'link[x:integer,y:nonnegative]'}

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
            'A': 'link[x:integer,y:nonnegative]'}

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
            'A': 'link[x:integer,y:nonnegative]'}
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
    test_infer_link(core)
    test_generate(core)
    # test_generate_promote_to_struct(core)
    test_uni_schema(core)
    test_array(core)
    test_bind(core)

    test_problem_schema_1(core)
    # _test_problem_schema_0(core)
    test_problem_schema_2(core)

    test_apply(core)
    test_unify(core)

