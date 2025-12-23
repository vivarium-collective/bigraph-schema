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

import copy
import typing
from pprint import pformat as pf
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

from bigraph_schema.parse import visit_expression
from bigraph_schema.edge import Edge
from bigraph_schema.methods import (
    reify_schema,
    handle_parameters,
    infer,
    default,
    resolve,
    generalize,
    check,
    validate,
    render,
    serialize,
    deserialize,
    merge,
    jump,
    traverse,
    apply)

from bigraph_schema.package import discover_packages


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
        found = self.access(data)
        if key in self.registry:
            self.update_type(key, found)
        else:
            self.registry[key] = found

    def register_types(self, types):
        """Bulk register multiple type keys into the operation registry."""
        for key, data in types.items():
            self.register_type(key, data)

    def update_type(self, key, data):
        """Deep-merge metadata/overrides into an existing registry entry."""
        if self.registry[key] != data:
            self.registry[key] = self.resolve(self.registry[key], data)

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

            if not isinstance(schema, Node):
                import ipdb; ipdb.set_trace()

            schema = replace(
                schema,
                **{'_default': default_value})

            parameters = {
                key: subvalue
                for key, subvalue in value.items()
                if not key in ('_type', '_default')}

            schema = reify_schema(self, schema, parameters)
            return schema

    def resolve_inherit(self, key):
        result = {}
        if '_inherit' in key:
            inherit = key['_inherit']
            if not isinstance(inherit, list):
                inherit = [inherit]
            for ancestor in inherit:
                found = self.access(ancestor)
                if not result:
                    result = found
                else:
                    result = self.resolve(result, found)
        return result

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
                    raise e
            else:
                entry = self.registry[key]
                if callable(entry):
                    return entry()
                elif isinstance(entry, Node):
                    return entry
                elif isinstance(entry, dict):
                    return self.access(entry)

        elif isinstance(key, dict):
            if '_type' in key:
                return self.access_type(key)

            else:
                result = self.resolve_inherit(key)

                for subkey, subitem in key.items():
                    if isinstance(subkey, str):
                        subitem = subitem if subkey.startswith('_') else self.access(subitem)
                    if isinstance(result, Node):
                        if hasattr(result, subkey):
                            result = replace(result, **{subkey: subitem})
                        else:
                            setattr(result, subkey, subitem)
                    else:
                        result[subkey] = subitem 

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
        merge_schema = self.resolve_merges(schema, merges)

        return merge_schema

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

    def resolve(self, current_schema, update_schema, path=None):
        """Unify two schemas under node semantics (e.g., Map/Tree/Link field-wise resolution)."""
        current = self.access(current_schema)
        update = self.access(update_schema)

        if isinstance(current, np.ndarray) or isinstance(update, np.ndarray):
            import ipdb; ipdb.set_trace()

        if path:
            return resolve(current, update, path=path)

        try:
            if current == update:
                return current
            else:
                return resolve(current, update)
        except ValueError:
            # numpy grumble grumble
            return resolve(current, update)

    def generalize(self, current_schema, update_schema):
        """Unify two schemas under node semantics (e.g., Map/Tree/Link field-wise resolution)."""
        current = self.access(current_schema)
        update = self.access(update_schema)
        return generalize(current, update)

    def check(self, schema, state):
        """Returns True if the `state` fits the `schema`."""
        found = self.access(schema)
        return check(found, state)

    def validate(self, schema, state, message=None):
        """Returns a nested description of how the state does not match the schema"""
        found = self.access(schema)
        validation = validate(self, found, state)
        if validation:
            message = f'state failed schema validation:\\nschema: {pf(render(schema))}\n\nstate: {pf(state)}'
            raise Exception(f'{message}: {validation}')

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
            merge_schema = self.resolve_merges({}, merges)
            decode_schema = self.resolve(decode_schema, merge_schema)
            merge_state = self.fill(merge_schema, decode_state)
        else:
            merge_state = decode_state

        return decode_schema, merge_state

    def generalize_merges(self, schema, merges):
        if len(merges) > 0:
            merge_schema = {}
            for path, subschema in merges:
                merge_schema = self.resolve(
                    merge_schema,
                    subschema,
                    resolve_path(path))

            schema = self.generalize(schema, merge_schema)

        return schema

    def resolve_merges(self, schema, merges):
        if len(merges) > 0:
            merge_schema = {}
            for path, subschema in merges:
                merge_schema = self.resolve(
                    merge_schema,
                    subschema,
                    resolve_path(path))

            schema = self.resolve(schema, merge_schema)

        return schema

    def resolve_schemas(self, schemas):
        if len(schemas) > 0:
            schema = schemas[0]
            for subschema in schemas[1:]:
                schema = self.resolve(schema, subschema)
            return schema

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

    def merge(self, schema, state, merge_state, path=()):
        """Schema-aware merge of `merge_state` into `state`."""
        found = self.access(schema)
        return merge(found, state, merge_state, path=path)

    def fill(self, schema, state, overwrite=False):
        found = self.access(schema)
        base_schema, base_state, merges = self.default_merges(found)
        merge_schema = self.resolve_merges(base_schema, merges)

        if overwrite:
            return merge(merge_schema, state, base_state)
        else:
            return merge(merge_schema, base_state, state)

    def view_ports(self, schema, state, path, ports_schema, wires):
        if isinstance(wires, str):
            wires = [wires]

        if isinstance(wires, (list, tuple)):
            _, result = self.traverse(schema, state, list(path) + list(wires))

        elif isinstance(wires, dict):
            result = {}
            for port_key, subport in wires.items():
                subschema, subwires = self.jump(
                    ports_schema,
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

            project_schema = self.resolve(
                project_schema,
                ports_schema,
                path=destination)

            project_state = self.merge(
                project_schema,
                project_state,
                view,
                path=destination)

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


def allocate_core(top=None):
    core = Core(BASE_TYPES)
    core = discover_packages(core, top)

    return core
