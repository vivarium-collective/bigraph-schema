"""
===========
Type System
===========
"""

import copy
import pint
import pprint
import pytest
import random
import numbers

import numpy as np

from bigraph_schema.react import react_divide_counts
from bigraph_schema.registry import (
    Registry, TypeRegistry, RegistryRegistry,
    type_schema_keys, non_schema_keys,
    deep_merge, get_path, establish_path, set_path, transform_path, remove_path, remove_omitted
)

from bigraph_schema.units import units, render_units_type


TYPE_SCHEMAS = {
    'float': 'float'}


class TypeSystem:
    """Handles type schemas and their operation"""

    def __init__(self):
        self.apply_registry = Registry(function_keys=['current', 'update', 'bindings', 'types'])
        self.serialize_registry = Registry(function_keys=['value', 'bindings', 'types'])
        self.deserialize_registry = Registry(function_keys=['serialized', 'bindings', 'types'])
        self.divide_registry = Registry()  # TODO enforce keys for divider methods
        self.check_registry = Registry()
        self.react_registry = Registry()
        self.type_registry = TypeRegistry()

        self.registry_registry = RegistryRegistry()
        self.registry_registry.register('_type', self.type_registry)
        self.registry_registry.register('_react', self.react_registry)
        self.registry_registry.register('_check', self.check_registry)
        self.registry_registry.register('_apply', self.apply_registry)
        self.registry_registry.register('_divide', self.divide_registry)
        self.registry_registry.register('_serialize', self.serialize_registry)
        self.registry_registry.register('_deserialize', self.deserialize_registry)
        
        register_base_types(self)
        register_base_reactions(self)


    def find_registry(self, underscore_key):
        root = underscore_key.trim('_')
        registry_key = f'{root}_registry'
        return getattr(self, registry_key)


    def register(self, type_data):
        self.function_keys = [
            '_apply',
            '_check',
            '_divide',
            '_react',
            '_serialize',
            '_deserialize']

        missing_functions = []
        for function_key in self.function_keys:
            if function_key in type_data:
                looking = type_data[function_key]
                registry = self.find_registry(
                    function_key)
                found = registry.access(looking)
                if found is None:
                    missing_functions.append(
                        (function_key, looking))

        if len(missing_functions > 0):
            raise Exception(
                f'functions are missing from\n{type_data}\nnamely, {missing_functions}')

    def access(self, type_key):
        return self.type_registry.access(type_key)


    def validate_schema(self, schema, enforce_connections=False):
        # add ports and wires
        # validate ports are wired to a matching type,
        #   either the same type or a subtype (more specific type)
        # declared ports are equivalent to the presence of a process
        #   where the ports need to be looked up

        report = {}

        if schema is None:
            report = 'schema cannot be None'

        elif isinstance(schema, str):
            typ = self.access(schema)
            if typ is None:
                report = f'type: {schema} is not in the registry'

        elif isinstance(schema, dict):
            report = {}

            schema_keys = set([])
            branches = set([])

            for key, value in schema.items():
                if key == '_type':
                    typ = self.access(value)
                    if typ is None:
                        report[key] = f'type: {value} is not in the registry'
                elif key in type_schema_keys:
                    schema_keys.add(key)
                    registry = self.registry_registry.access(key)
                    if registry is None:
                        # deserialize and serialize back and check it is equal
                        pass
                    else:
                        element = registry.access(value)
                        if element is None:
                            report[key] = f'no entry in the {key} registry for: {value}'
                else:
                    branches.add(key)
                    branch_report = self.validate_schema(value)
                    if len(branch_report) > 0:
                        report[key] = branch_report

        return report

        # # We will need this when building states to check to see if we are
        # # trying to instantiate an abstract type, but we can still register
        # # register abstract types so it is not invalid
        # if len(schema_keys) > 0 and len(branches) == 0:
        #     undeclared = set(type_schema_keys) - schema_keys
        #     if len(undeclared) > 0:
        #         for key in undeclared:
        #             if not key in optional_schema_keys:
        #                 report[key] = f'missing required key: {key} for declaring atomic type'


    # TODO: if its an edge, ensure ports match wires
    def validate_state(self, original_schema, state):
        schema = self.access(original_schema)
        validation = {}

        if '_serialize' in schema:
            if '_deserialize' not in schema:
                validation = {
                    '_deserialize': f'serialize found in type without deserialize: {schema}'
                }
            else:
                serialize = self.serialize_registry.access(
                    schema['_serialize'])
                deserialize = self.deserialize_registry.access(
                    schema['_deserialize'])
                serial = serialize(state)
                pass_through = deserialize(serial)

                if state != pass_through:
                    validation = f'state and pass_through are not the same: {serial}'
        else:
            for key, subschema in schema.items():
                if key not in type_schema_keys:
                    if key not in state:
                        validation[key] = f'key present in schema but not in state: {key}\nschema: {schema}\nstate: {state}\n'
                    else:
                        subvalidation = self.validate_state(
                            subschema,
                            state[key])
                        if not (subvalidation is None or len(subvalidation) == 0):
                            validation[key] = subvalidation

        return validation


    def default(self, schema):
        default = None
        found = self.access(schema)

        if '_default' in found:
            if not '_deserialize' in found:
                raise Exception(
                    f'asking for default but no deserialize in {found}')
            default = self.deserialize(found, found['_default'])
        else:
            default = {}
            for key, subschema in found.items():
                if key not in type_schema_keys:
                    default[key] = self.default(subschema)

        return default


    def match_node(self, schema, state, pattern):
        if isinstance(pattern, dict):
            if not isinstance(state, dict):
                return False

            if '_type' in pattern and not self.is_compatible(schema, pattern):
                return False

            for key, subpattern in pattern.items():
                if key.startswith('_'):
                    continue

                if key in schema:
                    subschema = schema[key]
                else:
                    subschema = schema

                if key in state:
                    matches = self.match_node(
                        subschema,
                        state[key],
                        pattern[key])
                    if not matches:
                        return False
                else:
                    return False

            return True

        else:
            return pattern == state


    def match_recur(self, schema, state, pattern, mode='first', path=()):
        matches = []

        match = self.match_node(
            schema,
            state,
            pattern)

        if match:
            if mode == 'first' or mode == 'immediate':
                return [path]
            else:
                matches.append(path)
        elif mode == 'immediate':
            return []

        if isinstance(state, dict):
            for key, substate in state.items():
                if key.startswith('_'):
                    continue

                if key in schema:
                    subschema = schema[key]
                else:
                    subschema = schema

                submatches = self.match_recur(
                    subschema,
                    state[key],
                    pattern,
                    mode=mode,
                    path=path + (key,))

                if mode == 'first' and len(submatches) > 0:
                    return submatches[0:1]
                else:
                    matches.extend(submatches)

        return matches


    # TODO: do we need a "match" type method?
    def match(self, original_schema, state, pattern, mode='first', path=()):
        '''
        find the path or paths to any instances of a given
        given pattern in the tree.

        "mode" can be a few things:
        * immediate: only match top level
        * first: only return the first match
        * random: return a random match of all that matched
        * all (or any other value): return every match in the tree
        '''

        schema = self.access(original_schema)

        matches = self.match_recur(
            schema,
            state,
            pattern,
            mode=mode,
            path=path)

        if mode == 'random':
            matches_count = len(matches)
            if matches_count > 0:
                choice = random.randint(
                    0,
                    matches_count-1)
                matches = [matches[choice]]

        return matches


    def react(self, schema, state, reaction, mode='random'):
        # TODO: explain all this
        # TODO: after the reaction, fill in the state with missing values
        #   from the schema

        if 'redex' in reaction or 'reactum' in reaction or 'calls' in reaction:
            redex = reaction.get('redex', {})
            reactum = reaction.get('reactum', {})
            calls = reaction.get('calls', {})
        else:
            # single key with reaction name
            reaction_key = list(reaction.keys())[0]
            make_reaction = self.react_registry.access(
                reaction_key)
            react = make_reaction(
                reaction.get(
                    reaction_key, {}))

            redex = react.get('redex', {})
            reactum = react.get('reactum', {})
            calls = react.get('calls', {})

        paths = self.match(
            schema,
            state,
            redex,
            mode=mode)
        
        def merge_state(before):
            remaining = remove_omitted(
                redex,
                reactum,
                before)

            return deep_merge(
                remaining,
                reactum)

        for path in paths:
            state = transform_path(
                state,
                path,
                merge_state)

        return state


    def check_state(self, schema, state):
        if isinstance(schema, str) or isinstance(schema, list):
            # this assumes access always returns a dict
            schema = self.access(schema)
            return self.check_state(schema, state, update)

        elif '_check' in schema and schema['_check'] != 'any':
            check_function = self.check_registry.access(
                schema['_check'])
            
            return check_function(
                state,
                schema.get('_bindings'),
                self)

        elif isinstance(state, dict):
            for key, branch in state.items():
                if not key.startswith('_'):
                    if key not in schema:
                        return False
                    else:
                        check = self.check_state(
                            schema[key],
                            state[key])

                        if not check:
                            return False

        else:
            return False

        return True


    def check(self, initial_schema, state):
        schema = self.access(initial_schema)
        return self.check_state(schema, state)
    

    def apply_update(self, schema, state, update):
        if isinstance(update, dict) and '_react' in update:
            state = self.react(
                schema,
                state,
                update['_react'])

        elif '_apply' in schema and schema['_apply'] != 'any':
            apply_function = self.apply_registry.access(schema['_apply'])
            
            state = apply_function(
                state,
                update,
                schema.get('_bindings'),
                self)

        elif isinstance(schema, str) or isinstance(schema, list):
            schema = self.access(schema)
            state = self.apply_update(schema, state, update)

        elif isinstance(update, dict):
            for key, branch in update.items():
                if key not in schema:
                    raise Exception(
                        f'trying to update a key that is not in the schema'
                        f'for state: {key}\n{state}\nwith schema:\n{schema}')
                else:
                    subupdate = self.apply_update(
                        schema[key],
                        state[key],
                        branch)

                    state[key] = subupdate
        else:
            raise Exception(
                f'trying to apply update\n  {update}\nto state\n  {state}\n'
                f'with schema\n{schema}, but the update is not a dict')

        return state


    def apply(self, original_schema, initial, update):
        schema = self.access(original_schema)
        state = copy.deepcopy(initial)
        return self.apply_update(schema, state, update)


    def set_update(self, schema, state, update):
        if '_apply' in schema:
            apply_function = self.apply_registry.access('set')
            
            state = apply_function(
                state,
                update,
                schema.get('_bindings'),
                self)

        elif isinstance(schema, str) or isinstance(schema, list):
            schema = self.access(schema)
            state = self.set_update(schema, state, update)

        elif isinstance(update, dict):
            for key, branch in update.items():
                if key not in schema:
                    raise Exception(
                        f'trying to update a key that is not in the schema'
                        f'for state: {key}\n{state}\nwith schema:\n{schema}')
                else:
                    subupdate = self.set_update(
                        schema[key],
                        state[key],
                        branch)

                    state[key] = subupdate
        else:
            raise Exception(
                f'trying to apply update\n  {update}\nto state\n  {state}\n'
                f'with schema\n{schema}, but the update is not a dict')

        return state


    def set(self, original_schema, initial, update):
        schema = self.access(original_schema)
        state = copy.deepcopy(initial)
        return self.set_update(schema, initial, update)


    def serialize(self, schema, state):
        found = self.access(schema)
        if '_serialize' in found:
            serialize_function = self.serialize_registry.access(
                found['_serialize'])

            if serialize_function is None:
                raise Exception(
                    f'serialize function not in the registry: {schema}')
            else:
                return serialize_function(
                    state,
                    found.get('_bindings'),
                    self)
        else:
            tree = {
                key: self.serialize(
                    schema[key],
                    state.get(key))
                for key in non_schema_keys(schema)}

            return tree

    def deserialize(self, schema, encoded):
        found = self.access(schema)

        if '_deserialize' in found:
            deserialize = found['_deserialize']
            if isinstance(deserialize, str):
                deserialize_function = self.deserialize_registry.access(
                    deserialize)
            else:
                deserialize_function = deserialize

            if deserialize_function is None:
                raise Exception(
                    f'deserialize function not in the registry: {deserialize}')

            if encoded is None:
                encoded = self.default(schema)

            return deserialize_function(
                encoded,
                found.get('_bindings'),
                self)

        elif isinstance(encoded, dict):
            return {
                key: self.deserialize(
                    schema.get(key),
                    branch)
                for key, branch in encoded.items()}

        else:
            print(f'cannot deserialize: {encoded}')
            return encoded


    def divide(self, schema, state, ratios=(0.5, 0.5)):
        # TODO: implement
        return state


    def fill_ports(self, schema, wires=None, state=None, top=None, path=None):
        # deal with wires
        if wires is None:
            wires = {}
        if state is None:
            state = {}
        if top is None:
            top = state
        if path is None:
            path = []

        for port_key, port_schema in schema.items():
            if port_key in wires:
                subwires = wires[port_key]
                if isinstance(subwires, dict):
                    if isinstance(state, dict):
                        state[port_key] = self.fill_ports(
                            port_schema,
                            wires=subwires,
                            state=state.get(port_key),
                            top=top,
                            path=path)
                else:
                    if isinstance(subwires, str):
                        subwires = (subwires,)

                    if len(path) == 0:
                        raise Exception(
                            f'cannot wire {port_key} as we are already at the top level {schema}')

                    peer = get_path(
                        top,
                        path[:-1])

                    destination = establish_path(
                        peer,
                        subwires[:-1],
                        top=top,
                        cursor=path[:-1])

                    destination_key = subwires[-1]

                    if destination_key in destination:
                        pass
                        # validate_state(
                        #     port_schema,
                        #     destination[destination_key])
                    else:
                        destination[destination_key] = self.default(
                            port_schema)
            else:
                # handle unconnected ports
                pass

        return state

    def fill_state(self, schema, state=None, top=None, path=None, type_key=None, context=None):
        # if a port is disconnected, build a store
        # for it under the '_open' key in the current
        # node (?)

        # inform the user that they have disconnected
        # ports somehow

        if schema is None:
            return None
        if state is None:
            state = self.default(schema)
        if top is None:
            top = state
        if path is None:
            path = []

        if '_inputs' in schema:
            inputs = state.get('inputs', {})
            state = self.fill_ports(
                schema['_inputs'],
                wires=inputs,
                state=state,
                top=top,
                path=path)

        if '_outputs' in schema:
            outputs = state.get('outputs', {})
            state = self.fill_ports(
                schema['_outputs'],
                wires=outputs,
                state=state,
                top=top,
                path=path)

        if isinstance(schema, str):
            schema = self.access(schema)

        branches = non_schema_keys(schema)

        if isinstance(state, dict):
            for branch in branches:
                subpath = path + [branch]
                state[branch] = self.fill_state(
                    schema[branch],
                    state=state.get(branch),
                    top=top,
                    path=subpath)

        return state


    def fill(self, original_schema, state=None):
        # # Removing deepcopy means the state may be updated
        # if state is not None:
        #     state = copy.deepcopy(state)
        schema = self.access(original_schema)

        return self.fill_state(
            schema,
            state=state)


    def ports_schema(self, schema, instance, edge_path, ports_key='inputs'):
        found = self.access(schema)

        ports_schema = {}
        ports = {}

        edge_schema = get_path(found, edge_path)
        ports_schema = edge_schema.get(f'_{ports_key}')

        edge_state = get_path(instance, edge_path)
        ports = edge_state.get(ports_key)
        
        return ports_schema, ports


    def view(self, schema, wires, path, instance):
        result = {}
        if isinstance(wires, str):
            wires = [wires]
        if isinstance(wires, (list, tuple)):
            result = get_path(instance, list(path) + list(wires))
        elif isinstance(wires, dict):
            result = {}
            for port_key, port_path in wires.items():
                if isinstance(port_path, dict) or get_path(instance, port_path) is not None:
                    inner_view = self.view(
                        schema[port_key],
                        port_path,
                        path,
                        instance)

                    if inner_view is not None:
                        result[port_key] = inner_view
        else:
            raise Exception(f'trying to project state with these ports:\n{schema}\nbut not sure what these wires are:\n{wires}')

        return result


    def view_edge(self, schema, instance, edge_path=None, ports_key='inputs'):
        '''
        project the state of the current instance into a form
        the edge expects, based on its ports
        '''

        if schema is None:
            return None
        if instance is None:
            instance = self.default(schema)
        if edge_path is None:
            edge_path = []

        ports_schema, ports = self.ports_schema(
            schema,
            instance,
            edge_path=edge_path,
            ports_key=ports_key)

        if not ports_schema:
            return None
        if not ports:
            return None

        return self.view(
            ports_schema,
            ports,
            edge_path[:-1],
            instance)


    def project(self, ports, wires, path, states):
        result = {}

        if isinstance(wires, str):
            wires = [wires]

        if isinstance(wires, list):
            destination = list(path) + wires
            result = set_path(
                result,
                destination,
                states)

        elif isinstance(wires, dict):
            branches = [
                self.project(
                    ports.get(key),
                    wires[key],
                    path,
                    states.get(key))
                for key in wires.keys()]

            branches = [
                branch
                for branch in branches
                if branch is not None] # and list(branch)[0][1] is not None]

            result = {}
            for branch in branches:
                deep_merge(result, branch)
        else:
            raise Exception(
                f'inverting state\n  {state}\naccording to ports schema\n  {schema}\nbut wires are not recognized\n  {wires}')

        return result


    def project_edge(self, schema, instance, edge_path, states, ports_key='outputs'):
        '''
        given states from the perspective of an edge (through
          it's ports), produce states aligned to the tree
          the wires point to.
          (inverse of view)
        '''

        if schema is None:
            return None
        if instance is None:
            instance = self.default(schema)

        ports_schema, ports = self.ports_schema(
            schema,
            instance,
            edge_path,
            ports_key)

        if ports_schema is None:
            return None
        if ports is None:
            return None

        return self.project(
            ports_schema,
            ports,
            edge_path[:-1],
            states)


    def infer_wires(self, ports, state, wires, top_schema=None, path=None):
        top_schema = top_schema or {}
        path = path or ()

        for port_key, port_wires in wires.items():
            if isinstance(ports, str):
                import ipdb; ipdb.set_trace()
            port_schema = ports.get(port_key, {})

            if isinstance(port_wires, dict):
                top_schema = self.infer_wires(
                    ports,
                    state.get(port_key),
                    port_wires,
                    top_schema,
                    path + (port_key,))
            else:
                peer = get_path(
                    top_schema,
                    path[:-1])

                destination = establish_path(
                    peer,
                    port_wires[:-1],
                    top=top_schema,
                    cursor=path[:-1])

                if len(port_wires) == 0:
                    raise Exception(f'no wires at port "{port_key}" in ports {ports} with state {state}')

                destination_key = port_wires[-1]
                if destination_key in destination:
                    # TODO: validate the schema/state
                    pass
                else:
                    destination[destination_key] = port_schema

        return top_schema


    def infer_schema(self, schema, state, top_state=None, path=None):
        '''
        Given a schema fragment and an existing state with _type keys,
        return the full schema required to describe that state,
        and whatever state was hydrated (edges) during this process
        '''

        schema = schema or {}
        # TODO: deal with this
        if schema == '{}':
            schema = {}

        top_state = top_state or state
        path = path or ()

        if isinstance(state, dict):
            if '_type' in state:
                state_type = state['_type']
                state_schema = self.access(state_type)

                hydrated_state = self.deserialize(state_schema, state)
                top_state = set_path(
                    top_state,
                    path,
                    hydrated_state)

                schema = set_path(
                    schema,
                    path,
                    {'_type': state_type})

                # TODO: fix is_descendant
                # if types.type_registry.is_descendant('edge', state_schema)
                if state_type == 'edge':
                    inputs = hydrated_state.get('inputs')
                    if inputs:
                        schema = self.infer_wires(
                            state_schema,
                            hydrated_state,
                            inputs,
                            top_schema=schema,
                            path=path[:-1])

                    outputs = hydrated_state.get('outputs')
                    if outputs:
                        schema = self.infer_wires(
                            state_schema,
                            hydrated_state,
                            outputs,
                            top_schema=schema,
                            path=path[:-1])

            elif '_type' in schema:
                hydrated_state = self.deserialize(schema, state)
                top_state = set_path(
                    top_state,
                    path,
                    hydrated_state)

            else:
                for key, value in state.items():
                    inner_path = path + (key,)
                    if get_path(schema, inner_path) is None or get_path(state, inner_path) is None:
                        schema, top_state = self.infer_schema(
                            schema,
                            value,
                            top_state=top_state,
                            path=inner_path)

        elif isinstance(state, str):
            pass

        else:
            type_schema = TYPE_SCHEMAS.get(str(type(state)), schema)

            peer = get_path(schema, path)
            destination = establish_path(
                peer,
                path[:-1],
                top=schema,
                cursor=path[:-1])

            path_key = path[-1]
            if path_key in destination:
                # TODO: validate
                pass
            else:
                destination[path_key] = type_schema

        return schema, top_state
        

    def hydrate_state(self, schema, state):
        if isinstance(state, str) or '_deserialize' in schema:
            result = self.deserialize(schema, state)
        elif isinstance(state, dict):
            if isinstance(schema, str):
                schema = self.access(schema)
                return self.hydrate_state(schema, state)
            else:
                result = state.copy()
                for key, value in schema.items():
                    if key in schema:
                        subschema = schema[key]
                    else:
                        subschema = schema

                    if key in state:
                        result[key] = self.hydrate_state(
                            subschema,
                            state.get(key))
        else:
            result = state

        return result


    def hydrate(self, schema, state):
        # TODO: support partial hydration (!)
        hydrated = self.hydrate_state(schema, state)
        return self.fill(schema, hydrated)


    def complete(self, initial_schema, initial_state):
        full_schema = self.access(
            initial_schema)

        # hydrate the state given the initial composition
        state = self.hydrate(
            full_schema,
            initial_state)

        # fill in the parts of the composition schema
        # determined by the state
        schema, state = self.infer_schema(
            full_schema,
            state)

        # TODO: add flag to types.access(copy=True)
        return self.access(schema), state
        

    def link_place(self, place, link):
        pass


    def compose(self, a, b):
        pass


    def query(self, schema, instance, redex):
        subschema = {}
        return subschema


NONE_SYMBOL = 'None'


def instance_parameter(instance_type):
    def is_instance_parameter(instance, bindings=None, types=None):
        return isinstance(instance, instance_type)

    return is_instance_parameter


check_number = instance_parameter(numbers.Number)
check_boolean = instance_parameter(bool)
check_int = instance_parameter(int)
check_float = instance_parameter(float)
check_string = instance_parameter(str)


def check_list(state, bindings, types):
    element_type = bindings['element']

    if isinstance(state, list):
        for element in state:
            check = types.check(
                element_type,
                element)

            if not check:
                return False

        return True
    else:
        return False


class Edge:
    def __init__(self):
        pass


    def schema(self):
        return {
            'inputs': {},
            'outputs': {}}


base_type_library = {
    'boolean': {
        '_type': 'boolean',
        '_default': False,
        '_apply': 'apply_boolean',
        '_serialize': 'serialize_boolean',
        '_deserialize': 'deserialize_boolean',
        '_divide': 'divide_boolean',
    },

    # abstract number type
    'number': {
        '_type': 'number',
        '_apply': 'accumulate',
        '_check': 'check_number',
        '_serialize': 'to_string',
        '_description': 'abstract base type for numbers'},

    'int': {
        '_type': 'int',
        '_default': '0',
        # inherit _apply and _serialize from number type
        '_deserialize': 'deserialize_int',
        '_check': 'check_int',
        '_divide': 'divide_int',
        '_description': '64-bit integer',
        '_super': 'number'},

    'float': {
        '_type': 'float',
        '_default': '0.0',
        '_deserialize': 'float',
        '_check': 'check_float',
        '_divide': 'divide_float',
        '_description': '64-bit floating point precision number',
        '_super': 'number'},

    'string': {
        '_type': 'string',
        '_default': '',
        '_apply': 'replace',
        '_check': 'check_string',
        '_serialize': 'serialize_string',
        '_deserialize': 'deserialize_string',
        '_divide': 'divide_int',
        '_description': '64-bit integer'},

    'list': {
        '_type': 'list',
        '_default': [],
        '_apply': 'apply_list',
        '_check': 'check_list',
        '_serialize': 'serialize_list',
        '_deserialize': 'deserialize_list',
        '_divide': 'divide_list',
        '_type_parameters': ['element'],
        '_description': 'general list type (or sublists)'},

    'tree': {
        '_type': 'tree',
        '_default': {},
        '_apply': 'apply_tree',
        '_serialize': 'serialize_tree',
        '_deserialize': 'deserialize_tree',
        '_divide': 'divide_tree',
        '_check': 'check_tree',
        '_type_parameters': ['leaf'],
        '_description': 'mapping from str to some type (or nested dicts)'},

    'dict': {
        '_type': 'dict',
        '_default': {},
        '_apply': 'apply_dict',
        '_serialize': 'serialize_dict',
        '_deserialize': 'deserialize_dict',
        '_divide': 'divide_dict',
        '_check': 'check_dict',
        # TODO: create assignable type parameters?
        '_type_parameters': ['key', 'value'],
        '_description': 'mapping from keys of any type to values of any type'},

    # TODO: add native numpy array type
    'array': {
        '_type': 'array',
        '_type_parameters': ['shape', 'element']},

    'maybe': {
        '_type': 'maybe',
        '_default': 'None',
        '_apply': 'apply_maybe',
        '_serialize': 'serialize_maybe',
        '_deserialize': 'deserialize_maybe',
        '_check': 'check_maybe',
        '_divide': 'divide_maybe',
        '_type_parameters': ['value'],
        '_description': 'type to represent values that could be empty'},

    'wires': 'tree[list[string]]',

    'edge': {
        # TODO: do we need to have defaults informed by type parameters?
        '_type': 'edge',
        '_default': {
            'inputs': {},
            'outputs': {}},
        '_apply': 'apply_edge',
        '_serialize': 'serialize_edge',
        '_deserialize': 'deserialize_edge',
        '_divide': 'divide_edge',
        '_check': 'check_edge',
        '_type_parameters': ['inputs', 'outputs'],
        '_description': 'hyperedges in the bigraph, with ports as a type parameter',
        'inputs': 'wires',
        'outputs': 'wires',
    },

    'numpy_array': {
        '_type': 'numpy_array',
        '_default': np.array([]),
        '_apply': 'accumulate',
        '_serialize': 'serialize_np_array',
        '_deserialize': 'deserialize_np_array',
        '_description': 'numpy arrays'
    },

    # TODO -- this should support any type
    'union': {
        '_type': 'union',
        '_default': 'None',
        # '_apply': 'apply_maybe',
        # '_serialize': 'serialize_maybe',
        # '_deserialize': 'deserialize_maybe',
        # '_divide': 'divide_maybe',
        '_type_parameters': ['value'],
        # '_description': 'type to represent values that could be empty'
    },
}


#################
# Apply methods #
#################

def apply_boolean(current: bool, update: bool, bindings=None, types=None) -> bool:
    """Performs a bit flip if `current` does not match `update`, returning update. Returns current if they match."""
    if current != update:
        return update
    else:
        return current


def divide_boolean(value: bool, bindings=None, types=None):
    return (value, value)


def serialize_boolean(value: bool, bindings=None, types=None) -> str:
    return str(value)


def deserialize_boolean(serialized, bindings=None, types=None) -> bool:
    return bool(serialized)


def apply_any(current, update, bindings=None, types=None):
    return update


def check_any(state, bindings=None, types=None):
    return True


def serialize_any(value, bindings=None, types=None):
    return str(value)


def accumulate(current, update, bindings=None, types=None):
    if current is None:
        return update
    if update is None:
        return current
    else:
        return current + update


def set_apply(current, update, bindings=None, types=None):
    return update


def concatenate(current, update, bindings=None, types=None):
    return current + update


##################
# Divide methods #
##################
# support dividing by ratios?
# ---> divide_float({...}, [0.1, 0.3, 0.6])

def divide_float(value, ratios, bindings=None, types=None):
    half = value / 2.0
    return (half, half)


# support function types for registrys?
# def divide_int(value: int, _) -> tuple[int, int]:
def divide_int(value, bindings=None, types=None):
    half = value // 2
    other_half = half
    if value % 2 == 1:
        other_half += 1
    return half, other_half


def divide_longest(dimensions, bindings=None, types=None):
    # any way to declare the required keys for this function in the registry?
    # find a way to ask a function what type its domain and codomain are

    width = dimensions['width']
    height = dimensions['height']

    if width > height:
        a, b = divide_int(width)
        return [{'width': a, 'height': height}, {'width': b, 'height': height}]
    else:
        x, y = divide_int(height)
        return [{'width': width, 'height': x}, {'width': width, 'height': y}]


def divide_list(l, bindings, types):
    result = [[], []]
    divide_type = bindings['element']
    divide = divide_type['_divide']

    for item in l:
        if isinstance(item, list):
            divisions = divide_list(item, bindings, types)
        else:
            divisions = divide(item, divide_type, types)

        result[0].append(divisions[0])
        result[1].append(divisions[1])

    return result


def replace(current, update, bindings=None, types=None):
    return update


#####################
# Serialize methods #
#####################

def serialize_string(value, bindings=None, types=None):
    return value


def deserialize_string(serialized, bindings=None, types=None):
    return serialized


def to_string(value, bindings=None, types=None):
    return str(value)


def serialize_list(value, bindings=None, types=None):
    schema = bindings['element']
    return [types.serialize(schema, element) for element in value]

def serialize_np_array(value, bindings=None, types=None):
    ''' Serialize numpy array to bytes '''
    return {
        'bytes': value.tobytes(),
        'dtype': value.dtype,
        'shape': value.shape,
    }


#######################
# Deserialize methods #
#######################

def deserialize_int(serialized, bindings=None, types=None):
    return int(serialized)


def deserialize_float(serialized, bindings=None, types=None):
    return float(serialized)


def evaluate(serialized, bindings=None, types=None):
    return eval(serialized)


def deserialize_any(serialized, bindings=None, types=None):
    return serialized

def deserialize_list(serialized, bindings=None, types=None):
    schema = bindings['element']
    return [types.deserialize(schema, element) for element in serialized]


def deserialize_np_array(serialized, bindings=None, types=None):
    if isinstance(serialized, dict):
        np.frombuffer(serialized['bytes'], dtype=serialized['dtype']).reshape(serialized['shape'])
        return np.frombuffer(serialized)
    else:
        return  serialized


# ------------------------------
# TODO: make all of the types work


def apply_list(current, update, bindings, types):
    element_type = types.access(bindings['element'])
    
    if isinstance(update, list):
        result = []
        for current_element, update_element in zip(current, update):
            applied = types.apply(
                element_type,
                current_element,
                update_element)

            result.append(applied)

        return result
    else:
        raise Exception(f'trying to apply an update to an existing list, but the update is not a list: {update}')


def apply_tree(current, update, bindings, types):
    leaf_type = types.access(bindings['leaf'])
    bindings['leaf'] = leaf_type
    
    if isinstance(update, dict):
        current = current or {}
        
        for key, branch in update.items():
            if key == '_add':
                current.update(branch)
            elif key == '_remove':
                current = remove_path(current, branch)
            elif types.check(leaf_type, branch):
                current[key] = types.apply(
                    leaf_type,
                    current.get(key),
                    branch)
            else:
                current[key] = apply_tree(
                    current.get(key),
                    branch,
                    bindings,
                    types)

        return current
    else:
        if current is None:
            current = types.default(leaf_type)
        return types.apply(leaf_type, current, update)


def check_tree(tree, bindings, types):
    leaf_type = bindings['leaf']

    if isinstance(state, dict):
        for key, value in state.items():
            check = types.check(
                leaf_type,
                leaf)

            if not check:
                check = types.check({
                    '_type': 'tree',
                    '_leaf': leaf_type})

                if not check:
                    return False

        return True
    else:
        return False

    return types.check(leaf_type, tree)


def divide_tree(tree, bindings, types):
    result = [{}, {}]
    # get the type of the values for this dict
    divide_type = bindings['leaf']
    divide_function = divide_type['_divide']
    # divide_function = types.registry_registry.type_attribute(
    #     divide_type,
    #     '_divide')

    for key, value in tree.items():
        if isinstance(value, dict):
            divisions = divide_tree(value)
        else:
            divisions = types.divide(divide_type, value)

        result[0][key], result[1][key] = divisions

    return result


def serialize_tree(value, bindings=None, types=None):
    return value


def deserialize_tree(serialized, bindings=None, types=None):
    tree = serialized

    if isinstance(serialized, str):
        tree = types.deserialize(
            bindings['leaf'],
            serialized)

    elif isinstance(serialized, dict):
        tree = {}
        for key, value in serialized.items():
            tree[key] = deserialize_tree(value, bindings, types)

    return tree


def apply_dict(current, update, bindings=None, types=None):
    pass


def check_dict(current, update, bindings=None, types=None):
    pass


def divide_dict(value, bindings=None, types=None):
    return value


def serialize_dict(value, bindings=None, types=None):
    return value


def deserialize_dict(serialized, bindings=None, types=None):
    return serialized


def apply_maybe(current, update, bindings, types):
    if current is None or update is None:
        return update
    else:
        value_type = bindings['value']
        return types.apply(value_type, current, update)


def check_maybe(state):
    if state is None:
        return [None, None]
    else:
        pass


def divide_maybe(value, bindings):
    if value is None:
        return [None, None]
    else:
        pass


def serialize_maybe(value, bindings, types):
    if value is None:
        return NONE_SYMBOL
    else:
        value_type = bindings['value']
        return serialize(value_type, value)


def deserialize_maybe(serialized, bindings, types):
    if serialized == NONE_SYMBOL:
        return None
    else:
        value_type = bindings['value']
        return types.deserialize(value_type, serialized)


# TODO: deal with all the different unit types
def apply_units(current, update, bindings, types):
    return current + update


def check_units(state, bindings, types):
    # TODO: expand this to check the actual units for compatibility
    return isinstance(state, pint.Quantity)


def serialize_units(value, bindings, types):
    return str(value)


def deserialize_units(serialized, bindings, types):
    return units(serialized)


def divide_units(value, bindings, types):
    return [value, value]


# TODO: implement edge handling
def apply_edge(current, update, bindings, types):
    return current + update


def check_edge(state, bindings, types):
    return state


def serialize_edge(value, bindings, types):
    return value


def deserialize_edge(serialized, bindings, types):
    return serialized


def divide_edge(value, bindings, types):
    return [value, value]


def register_units(types, units):
    for unit_name in units._units:
        try:
            unit = getattr(units, unit_name)
        except:
            # print(f'no unit named {unit_name}')
            continue

        dimensionality = unit.dimensionality
        type_key = render_units_type(dimensionality)
        if types.type_registry.access(type_key) is None:
            types.type_registry.register(type_key, {
                '_default': '',
                '_apply': 'apply_units',
                '_check': 'check_units',
                '_serialize': 'serialize_units',
                '_deserialize': 'deserialize_units',
                '_divide': 'divide_units',
                '_description': 'type to represent values with scientific units'})


def register_base_types(types):

    # validate the function registered is of the right type?
    types.apply_registry.register('any', apply_any)
    types.apply_registry.register('accumulate', accumulate)
    types.apply_registry.register('set', set_apply)
    types.apply_registry.register('concatenate', concatenate)
    types.apply_registry.register('replace', replace)
    types.apply_registry.register('apply_tree', apply_tree)
    types.apply_registry.register('apply_boolean', apply_boolean)
    types.apply_registry.register('apply_list', apply_list)
    types.apply_registry.register('apply_dict', apply_dict)
    types.apply_registry.register('apply_maybe', apply_maybe)
    types.apply_registry.register('apply_units', apply_units)
    types.apply_registry.register('apply_edge', apply_edge)

    types.divide_registry.register('divide_boolean', divide_boolean)
    types.divide_registry.register('divide_float', divide_float)
    types.divide_registry.register('divide_int', divide_int)
    types.divide_registry.register('divide_longest', divide_longest)
    types.divide_registry.register('divide_list', divide_list)
    types.divide_registry.register('divide_tree', divide_tree)
    types.divide_registry.register('divide_dict', divide_dict)
    types.divide_registry.register('divide_maybe', divide_maybe)
    types.divide_registry.register('divide_units', divide_units)
    types.divide_registry.register('divide_edge', divide_edge)

    types.check_registry.register('check_boolean', check_boolean)
    types.check_registry.register('check_number', check_number)
    types.check_registry.register('check_float', check_float)
    types.check_registry.register('check_string', check_string)
    types.check_registry.register('check_int', check_int)
    types.check_registry.register('check_list', check_list)
    types.check_registry.register('check_tree', check_tree)
    types.check_registry.register('check_dict', check_dict)
    types.check_registry.register('check_maybe', check_maybe)
    types.check_registry.register('check_units', check_units)
    types.check_registry.register('check_edge', check_edge)

    types.serialize_registry.register('serialize_any', serialize_any)
    types.serialize_registry.register('serialize_boolean', serialize_boolean)
    types.serialize_registry.register('serialize_string', serialize_string)
    types.serialize_registry.register('to_string', to_string)
    types.serialize_registry.register('serialize_tree', serialize_tree)
    types.serialize_registry.register('serialize_dict', serialize_dict)
    types.serialize_registry.register('serialize_maybe', serialize_maybe)
    types.serialize_registry.register('serialize_units', serialize_units)
    types.serialize_registry.register('serialize_edge', serialize_edge)
    types.serialize_registry.register('serialize_list', serialize_list)
    types.serialize_registry.register('serialize_np_array', serialize_np_array)

    types.deserialize_registry.register('deserialize_any', deserialize_any)
    types.deserialize_registry.register('deserialize_boolean', deserialize_boolean)
    types.deserialize_registry.register('float', deserialize_float)
    types.deserialize_registry.register('deserialize_int', deserialize_int)
    types.deserialize_registry.register('deserialize_string', deserialize_string)
    types.deserialize_registry.register('evaluate', evaluate)
    types.deserialize_registry.register('deserialize_tree', deserialize_tree)
    types.deserialize_registry.register('deserialize_dict', deserialize_dict)
    types.deserialize_registry.register('deserialize_maybe', deserialize_maybe)
    types.deserialize_registry.register('deserialize_units', deserialize_units)
    types.deserialize_registry.register('deserialize_edge', deserialize_edge)
    types.deserialize_registry.register('deserialize_list', deserialize_list)
    types.deserialize_registry.register('deserialize_np_array', deserialize_np_array)

    types.type_registry.register_multiple(base_type_library)
    register_units(types, units)

    return types


def register_base_reactions(types):
    types.react_registry.register('divide_counts', react_divide_counts)


def register_cube(types):
    cube_schema = {
        'shape': {
            '_type': 'shape',
            '_description': 'abstract shape type'},
        
        'rectangle': {
            '_type': 'rectangle',
            '_divide': 'divide_longest',
            '_description': 'a two-dimensional value',
            '_super': 'shape',
            'width': {'_type': 'int'},
            'height': {'_type': 'int'},
        },
        
        # cannot override existing keys unless it is of a subtype
        'cube': {
            '_type': 'cube',
            '_super': 'rectangle',
            'depth': {'_type': 'int'},
        },
    }

    types.type_registry.register_multiple(
        cube_schema)

    return types


@pytest.fixture
def base_types():
    return TypeSystem()


@pytest.fixture
def cube_types(base_types):
    return register_cube(base_types)


def register_compartment(base_types):
    base_types.type_registry.register('compartment', {
        'counts': 'tree[float]',
        'inner': 'tree[compartment]'})

    return base_types


@pytest.fixture
def compartment_types(base_types):
    return register_compartment(base_types)


def test_generate_default(cube_types):
    int_default = cube_types.default(
        {'_type': 'int'}
    )

    assert int_default == 0

    cube_default = cube_types.default(
        {'_type': 'cube'})

    assert 'width' in cube_default
    assert 'height' in cube_default
    assert 'depth' in cube_default

    nested_default = cube_types.default(
        {'a': 'int',
         'b': {
             'c': 'float',
             'd': 'cube'},
         'e': 'string'})

    assert nested_default['b']['d']['width'] == 0


def test_apply_update(cube_types):
    schema = {'_type': 'cube'}
    state = {
        'width': 11,
        'height': 13,
        'depth': 44,
    }
    update = {
        'depth': -5
    }

    new_state = cube_types.apply(
        schema,
        state,
        update
    )

    assert new_state['width'] == 11
    assert new_state['depth'] == 39


def print_schema_validation(types, library, should_pass):
    for key, declaration in library.items():
        report = types.validate_schema(declaration)
        if len(report) == 0:
            message = f'valid schema: {key}'
            if should_pass:
                print(f'PASS: {message}')
                pprint.pprint(declaration)
            else:
                raise Exception(f'FAIL: {message}\n{declaration}\n{report}')
        else:
            message = f'invalid schema: {key}'
            if not should_pass:
                print(f'PASS: {message}')
                pprint.pprint(declaration)
            else:
                raise Exception(f'FAIL: {message}\n{declaration}\n{report}')


def test_validate_schema(base_types):
    # good schemas
    print_schema_validation(base_types, base_type_library, True)

    good = {
        'not quite int': {
            '_default': 0,
            '_apply': 'accumulate',
            '_serialize': 'to_string',
            '_deserialize': 'deserialize_int',
            '_description': '64-bit integer'
        },
        'ports match': {
            'a': {
                '_type': 'int',
                '_value': 2
            },
            'edge1': {
                '_type': 'edge[a:int]',
                # '_type': 'edge',
                # '_ports': {
                #     '1': {'_type': 'int'},
                # },
            }
        }
    }        

    # bad schemas
    bad = {
        'empty': None,
        'str?': 'not a schema',
        'branch is weird': {
            'left': {'_type': 'ogre'},
            'right': {'_default': 1, '_apply': 'accumulate'},
        },
    }

    # test for ports and wires mismatch

    print_schema_validation(base_types, good, True)
    print_schema_validation(base_types, bad, False)


def test_fill_int(base_types):
    test_schema = {
        '_type': 'int'
    }

    full_state = base_types.fill(test_schema)
    direct_state = base_types.fill('int')

    assert full_state == direct_state == 0


def test_fill_cube(cube_types):
    test_schema = {'_type': 'cube'}
    partial_state = {'height': 5}

    full_state = cube_types.fill(
        test_schema,
        state=partial_state)

    assert 'width' in full_state
    assert 'height' in full_state
    assert 'depth' in full_state
    assert full_state['height'] == 5
    assert full_state['depth'] == 0


def test_fill_in_missing_nodes(base_types):
    test_schema = {
        'edge 1': {
            '_type': 'edge',
            '_inputs': {
                'I': 'float'},
            '_outputs': {
                'O': 'float'}}}

    test_state = {
        'edge 1': {
            'inputs': {
                'I': ['a']},
            'outputs': {
                'O': ['a']}}}

    filled = base_types.fill(
        test_schema,
        test_state)

    assert filled == {
        'a': 0.0,
        'edge 1': {
            'inputs': {
                'I': ['a']},
            'outputs': {
                'O': ['a']}}}


def test_fill_from_parse(base_types):
    test_schema = {
        'edge 1': 'edge[I:float,O:float]'}

    test_state = {
        'edge 1': {
            'inputs': {
                'I': ['a']},
            'outputs': {
                'O': ['a']}}}

    filled = base_types.fill(
        test_schema,
        test_state)

    assert filled == {
        'a': 0.0,
        'edge 1': {
            'inputs': {
                'I': ['a']},
            'outputs': {
                'O': ['a']}}}


# def test_fill_in_disconnected_port(base_types):
#     test_schema = {
#         'edge1': {
#             '_type': 'edge',
#             '_ports': {
#                 '1': {'_type': 'float'}}}}

#     test_state = {}


# def test_fill_type_mismatch(base_types):
#     test_schema = {
#         'a': {'_type': 'int', '_value': 2},
#         'edge1': {
#             '_type': 'edge',
#             '_ports': {
#                 '1': {'_type': 'float'},
#                 '2': {'_type': 'float'}},
#             'wires': {
#                 '1': ['..', 'a'],
#                 '2': ['a']},
#             'a': 5}}


# def test_edge_type_mismatch(base_types):
#     test_schema = {
#         'edge1': {
#             '_type': 'edge',
#             '_ports': {
#                 '1': {'_type': 'float'}},
#             'wires': {
#                 '1': ['..', 'a']}},
#         'edge2': {
#             '_type': 'edge',
#             '_ports': {
#                 '1': {'_type': 'int'}},
#             'wires': {
#                 '1': ['..', 'a']}}}


def test_establish_path(base_types):
    tree = {}
    destination = establish_path(
        tree,
        ('some',
         'where',
         'deep',
         'inside',
         'lives',
         'a',
         'tiny',
         'creature',
         'made',
         'of',
         'light'))

    assert tree['some']['where']['deep']['inside']['lives']['a']['tiny']['creature']['made']['of']['light'] == destination


def test_expected_schema(base_types):
    # equivalent to previous schema:

    # expected = {
    #     'store1': {
    #         'store1.1': {
    #             '_value': 1.1,
    #             '_type': 'float',
    #         },
    #         'store1.2': {
    #             '_value': 2,
    #             '_type': 'int',
    #         },
    #         'process1': {
    #             '_ports': {
    #                 'port1': {'_type': 'type'},
    #                 'port2': {'_type': 'type'},
    #             },
    #             '_wires': {
    #                 'port1': 'store1.1',
    #                 'port2': 'store1.2',
    #             }
    #         },
    #         'process2': {
    #             '_ports': {
    #                 'port1': {'_type': 'type'},
    #                 'port2': {'_type': 'type'},
    #             },
    #             '_wires': {
    #                 'port1': 'store1.1',
    #                 'port2': 'store1.2',
    #             }
    #         },
    #     },
    #     'process3': {
    #         '_wires': {
    #             'port1': 'store1',
    #         }
    #     }
    # }

    dual_process_schema = {
        'process1': 'edge[input1:float|input2:int,output1:float|output2:int]',
        'process2': {
            '_type': 'edge',
            '_inputs': {
                'input1': 'float',
                'input2': 'int'},
            '_outputs': {
                'output1': 'float',
                'output2': 'int'}}}

    base_types.type_registry.register(
        'dual_process',
        dual_process_schema,
    )

    test_schema = {
        # 'store1': 'process1:edge[port1:float|port2:int]|process2[port1:float|port2:int]',
        'store1': 'dual_process',
        'process3': 'edge[input_process:dual_process,output_process:dual_process]'}

    test_state = {
        'store1': {
            'process1': {
                'inputs': {
                    'input1': ['store1.1'],
                    'input2': ['store1.2']},
                'outputs': {
                    'output1': ['store2.1'],
                    'output2': ['store2.2']}},
            'process2': {
                'inputs': {
                    'input1': ['store2.1'],
                    'input2': ['store2.2']},
                'outputs': {
                    'output1': ['store1.1'],
                    'output2': ['store1.2']}}},
        'process3': {
            'inputs': {
                'input_process': ['store1']},
            'outputs': {
                'output_process': ['store1']}}}
    
    outcome = base_types.fill(test_schema, test_state)

    # assert outcome == {
    #     'process3': {
    #         'inputs': {
    #             'input_process': ['store1']},
    #         'outputs': {
    #             'output_process': ['store1']}},
    #     'store1': {
    #         'process1': {
    #             'inputs': {
    #                 'input1': ['store1.1'],
    #                 'input2': ['store1.2']},
    #             'outputs': {
    #                 'output1': ['store2.1'],
    #                 'output2': ['store2.2']}},
    #         'process2': {
    #             'inputs': {
    #                 'input1': ['store2.1'],
    #                 'input2': ['store2.2']},
    #             'outputs': {
    #                 'output1': ['store1.1'],
    #                 'output2': ['store1.2']}},
    #         'store1.1': 0.0,
    #         'store1.2': 0,
    #         'store2.1': 0.0,
    #         'store2.2': 0}}

    assert outcome == {
        'store1': {
            'store1.1': 0.0,
            'store1.2': 0,
            'store2.1': 0.0,
            'store2.2': 0,

            'process1': {
                'inputs': {
                    'input1': ['store1.1'],
                    'input2': ['store1.2']},
                'outputs': {
                    'output1': ['store2.1'],
                    'output2': ['store2.2']}},
            'process2': {
                'inputs': {
                    'input1': ['store2.1'],
                    'input2': ['store2.2']},
                'outputs': {
                    'output1': ['store1.1'],
                    'output2': ['store1.2']}}},
        'process3': {
            'inputs': {
                'input_process': ['store1']},
            'outputs': {
                'output_process': ['store1']}}}


def test_link_place(base_types):
    # TODO: this form is more fundamental than the compressed/inline dict form,
    #   and we should probably derive that from this form

    bigraph = {
        'nodes': {
            'v0': 'int',
            'v1': 'int',
            'v2': 'int',
            'v3': 'int',
            'v4': 'int',
            'v5': 'int',
            'e0': 'edge[e0-0:int|e0-1:int|e0-2:int]',
            'e1': {
                '_type': 'edge',
                '_ports': {
                    'e1-0': 'int',
                    'e2-0': 'int'}},
            'e2': {
                '_type': 'edge[e2-0:int|e2-1:int|e2-2:int]'}},

        'place': {
            'v0': None,
            'v1': 'v0',
            'v2': 'v0',
            'v3': 'v2',
            'v4': None,
            'v5': 'v4',
            'e0': None,
            'e1': None,
            'e2': None},

        'link': {
            'e0': {
                'e0-0': 'v0',
                'e0-1': 'v1',
                'e0-2': 'v4'},
            'e1': {
                'e1-0': 'v3',
                'e1-1': 'v1'},
            'e2': {
                'e2-0': 'v3',
                'e2-1': 'v4',
                'e2-2': 'v5'}},

        'state': {
            'v0': '1',
            'v1': '1',
            'v2': '2',
            'v3': '3',
            'v4': '5',
            'v5': '8',
            'e0': {
                'wires': {
                    'e0-0': 'v0',
                    'e0-1': 'v1',
                    'e0-2': 'v4'}},
            'e1': {
                'wires': {
                    'e1-0': 'v3',
                    'e1-1': 'v1'}},
            'e2': {
                'e2-0': 'v3',
                'e2-1': 'v4',
                'e2-2': 'v5'}}}

    placegraph = { # schema
        'v0': {
            'v1': int,
            'v2': {
                'v3': int}},
        'v4': {
            'v5': int},
        'e0': 'edge',
        'e1': 'edge',
        'e2': 'edge'}

    hypergraph = { # edges
        'e0': {
            'e0-0': 'v0',
            'e0-1': 'v1',
            'e0-2': 'v4'},
        'e1': {
            'e1-0': 'v3',
            'e1-1': 'v1'},
        'e2': {
            'e2-0': 'v3',
            'e2-1': 'v4',
            'e2-2': 'v5'}}

    merged = {
        'v0': {
            'v1': {},
            'v2': {
                'v3': {}}},
        'v4': {
            'v5': {}},
        'e0': {
            'wires': {
                'e0.0': ['v0'],
                'e0.1': ['v0', 'v1'],
                'e0.2': ['v4']}},
        'e1': {
            'wires': {
                'e0.0': ['v0', 'v2', 'v3'],
                'e0.1': ['v0', 'v1']}},
        'e2': {
            'wires': {
                'e0.0': ['v0', 'v2', 'v3'],
                'e0.1': ['v4'],
                'e0.2': ['v4', 'v5']}}}

    result = base_types.link_place(placegraph, hypergraph)
    # assert result == merged


def test_units(base_types):
    schema_length = {
        'distance': {'_type': 'length'}}

    state = {'distance': 11 * units.meter}
    update = {'distance': -5 * units.feet}

    new_state = base_types.apply(
        schema_length,
        state,
        update
    )

    assert new_state['distance'] == 9.476 * units.meter


def test_unit_conversion(base_types):
    # mass * length ^ 2 / second ^ 2

    units_schema = {
        'force': 'length^2*mass/time^2'}

    force_units = units.meter ** 2 * units.kg / units.second ** 2

    instance = {
        'force': 3.333 * force_units}


def test_serialize_deserialize(cube_types):
    schema = {
        'edge1': {
            # '_type': 'edge[1:int|2:float|3:string|4:tree[int]]',
            '_type': 'edge',
            '_outputs': {
                '1': 'int',
                '2': 'float',
                '3': 'string',
                '4': 'tree[int]'}},
        'a0': {
            'a0.0': 'int',
            'a0.1': 'float',
            'a0.2': {
                'a0.2.0': 'string'}},
        'a1': 'tree[int]'}

    instance = {
        'edge1': {
            'outputs': {
                '1': ['a0', 'a0.0'],
                '2': ['a0', 'a0.1'],
                '3': ['a0', 'a0.2', 'a0.2.0'],
                '4': ['a1']}},
        'a1': {
            'branch1': {
                'branch2': 11,
                'branch3': 22},
            'branch4': 44}}
    
    instance = cube_types.fill(schema, instance)

    encoded = cube_types.serialize(schema, instance)
    decoded = cube_types.deserialize(schema, encoded)

    assert instance == decoded


# is this a lens?
def test_project(cube_types):
    schema = {
        'edge1': {
            # '_type': 'edge[1:int|2:float|3:string|4:tree[int]]',
            # '_type': 'edge',
            '_type': 'edge',
            '_inputs': {
                '1': 'int',
                '2': 'float',
                '3': 'string',
                '4': 'tree[int]'},
            '_outputs': {
                '1': 'int',
                '2': 'float',
                '3': 'string',
                '4': 'tree[int]'}},
        'a0': {
            'a0.0': 'int',
            'a0.1': 'float',
            'a0.2': {
                'a0.2.0': 'string'}},
        'a1': 'tree[int]'}

    path_format = {
        '1': 'a0>a0.0',
        '2': 'a0>a0.1',
        '3': 'a0>a0.2>a0.2.0'}

    # TODO: support separate schema/instance, and 
    #   instances with '_type' and type parameter keys
    # TODO: support overriding various type methods
    instance = {
        'a0': {
            'a0.0': 11},
        'edge1': {
            'inputs': {
                '1': ['a0', 'a0.0'],
                '2': ['a0', 'a0.1'],
                '3': ['a0', 'a0.2', 'a0.2.0'],
                '4': ['a1']},
            'outputs': {
                '1': ['a0', 'a0.0'],
                '2': ['a0', 'a0.1'],
                '3': ['a0', 'a0.2', 'a0.2.0'],
                '4': ['a1']}},
        'a1': {
            'branch1': {
                'branch2': 11,
                'branch3': 22},
            'branch4': 44}}

    instance = cube_types.fill(schema, instance)

    states = cube_types.view_edge(
        schema,
        instance,
        ['edge1'])

    update = cube_types.project_edge(
        schema,
        instance,
        ['edge1'],
        states)

    assert update == {
        'a0': {
            'a0.0': 11,
            'a0.1': 0.0,
            'a0.2': {
                'a0.2.0': ''}},
        'a1': {
            'branch1': {
                'branch2': 11,
                'branch3': 22},
            'branch4': 44}}

    # TODO: make sure apply does not mutate instance
    updated_instance = cube_types.apply(
        schema,
        instance,
        update)

    add_update = {
        '4': {
            'branch6': 111,
            'branch1': {
                '_add': {
                    'branch7': 4444},
                '_remove': ['branch2']},
            '_add': {
                'branch5': 55},
            '_remove': ['branch4']}}

    inverted_update = cube_types.project_edge(
        schema,
        updated_instance,
        ['edge1'],
        add_update)

    modified_branch = cube_types.apply(
        schema,
        updated_instance,
        inverted_update)

    assert modified_branch == {
        'a0': {
            'a0.0': 22,
            'a0.1': 0.0,
            'a0.2': {
                'a0.2.0': ''}},
        'a1': {
            'branch1': {
                'branch7': 4444,
                'branch3': 44},
            'branch5': 55,
            'branch6': 111},
        'edge1': {
            'inputs': {
                '1': ['a0', 'a0.0'],
                '2': ['a0', 'a0.1'],
                '3': ['a0', 'a0.2', 'a0.2.0'],
                '4': ['a1']},
            'outputs': {
                '1': ['a0', 'a0.0'],
                '2': ['a0', 'a0.1'],
                '3': ['a0', 'a0.2', 'a0.2.0'],
                '4': ['a1']}}}


def test_check(base_types):
    assert base_types.check('float', 1.11)
    assert base_types.check({'b': 'float'}, {'b': 1.11})


def test_foursquare(base_types):
    # TODO: need union type and self-referential types (foursquare)
    foursquare_schema = {
        '_type': 'foursquare',
        '00': 'union[bool,foursquare]',
        '01': 'union[bool,foursquare]',
        '10': 'union[bool,foursquare]',
        '11': 'union[bool,foursquare]',
        '_default': {
            '00': False,
            '01': False,
            '10': False,
            '11': False
        },
        '_description': '',
    }
    base_types.type_registry.register(
        'foursquare', foursquare_schema)

    example = {
        '00': True,
        '11': {
            '00': True,
            '11': {
                '00': True,
                '11': {
                    '00': True,
                    '11': {
                        '00': True,
                        '11': {
                            '00': True,
                        },
                    },
                },
            },
        },
    }

    example_full = {
        '_type': 'foursquare',
        '00': {
            '_value': True,
            '_type': 'bool'},
        '11': {
            '_type': 'foursquare',
            '00': {
                '_value': True,
                '_type': 'bool'},
            '11': {
                '_type': 'foursquare',
                '00': {
                    '_value': True,
                    '_type': 'bool'},
                '11': {
                    '_type': 'foursquare',
                    '00': {
                        '_value': True,
                        '_type': 'bool'},
                    '11': {
                        '_type': 'foursquare',
                        '00': {
                            '_value': True,
                            '_type': 'bool'},
                        '11': {
                            '_type': 'foursquare',
                            '00': {
                                '_value': True,
                                '_type': 'bool'},
                        },
                    },
                },
            },
        },
    }


def test_add_reaction(compartment_types):
    single_node = {
        'environment': {
            '_type': 'compartment',
            'counts': {'A': 144},
            'inner': {
                '0': {
                    'counts': {'A': 13},
                    'inner': {}}}}}

    def add_reaction(config):
        path = config.get('path')

        redex = {}
        establish_path(
            redex,
            path)

        reactum = {}
        node = establish_path(
            reactum,
            path)

        deep_merge(
            node,
            config.get('add', {}))

        return {
            'redex': redex,
            'reactum': reactum}

    compartment_types.react_registry.register(
        'add',
        add_reaction)

    add_config = {
        'path': ['environment', 'inner'],
        'add': {
            '1': {
                'counts': {
                    'A': 8}}}}

    schema, state = compartment_types.infer_schema(
        {},
        single_node)

    assert '0' in state['environment']['inner']
    assert '1' not in state['environment']['inner']

    result = compartment_types.apply(
        schema,
        state, {
            '_react': {
                'add': add_config}})

            # '_react': {
            #     'reaction': 'add',
            #     'config': add_config}})

    assert '0' in result['environment']['inner']
    assert '1' in result['environment']['inner']


def test_remove_reaction(compartment_types):
    single_node = {
        'environment': {
            '_type': 'compartment',
            'counts': {'A': 144},
            'inner': {
                '0': {
                    'counts': {'A': 13},
                    'inner': {}},
                '1': {
                    'counts': {'A': 13},
                    'inner': {}}}}}

    # TODO: register these for general access
    def remove_reaction(config):
        path = config.get('path', ())
        redex = {}
        node = establish_path(
            redex,
            path)

        for remove in config.get('remove', []):
            node[remove] = {}

        reactum = {}
        establish_path(
            reactum,
            path)

        return {
            'redex': redex,
            'reactum': reactum}

    compartment_types.react_registry.register(
        'remove',
        remove_reaction)

    remove_config = {
        'path': ['environment', 'inner'],
        'remove': ['0']}

    schema, state = compartment_types.infer_schema(
        {},
        single_node)

    assert '0' in state['environment']['inner']
    assert '1' in state['environment']['inner']

    result = compartment_types.apply(
        schema,
        state, {
            '_react': {
                'remove': remove_config}})

    assert '0' not in result['environment']['inner']
    assert '1' in state['environment']['inner']
    

def test_replace_reaction(compartment_types):
    single_node = {
        'environment': {
            '_type': 'compartment',
            'counts': {'A': 144},
            'inner': {
                '0': {
                    'counts': {'A': 13},
                    'inner': {}},
                '1': {
                    'counts': {'A': 13},
                    'inner': {}}}}}

    def replace_reaction(config):
        path = config.get('path', ())

        redex = {}
        node = establish_path(
            redex,
            path)

        for before_key, before_state in config.get('before', {}).items():
            node[before_key] = before_state

        reactum = {}
        node = establish_path(
            reactum,
            path)

        for after_key, after_state in config.get('after', {}).items():
            node[after_key] = after_state

        return {
            'redex': redex,
            'reactum': reactum}

    compartment_types.react_registry.register(
        'replace',
        replace_reaction)

    replace_config = {
        'path': ['environment', 'inner'],
        'before': {'0': {'A': '?1'}},
        'after': {
            '2': {
                'counts': {
                    'A': {'function': 'divide', 'arguments': ['?1', 0.5], }}},
            '3': {
                'counts': {
                    'A': '@1'}}}}

    replace_config = {
        'path': ['environment', 'inner'],
        'before': {'0': {}},
        'after': {
            '2': {
                'counts': {
                    'A': 3}},
            '3': {
                'counts': {
                    'A': 88}}}}

    schema, state = compartment_types.infer_schema(
        {},
        single_node)

    assert '0' in state['environment']['inner']
    assert '1' in state['environment']['inner']

    result = compartment_types.apply(
        schema,
        state, {
            '_react': {
                'replace': replace_config}})

    assert '0' not in result['environment']['inner']
    assert '1' in result['environment']['inner']
    assert '2' in result['environment']['inner']
    assert '3' in result['environment']['inner']


def test_reaction(compartment_types):
    single_node = {
        'environment': {
            'counts': {},
            'inner': {
                '0': {
                    'counts': {}}}}}

    # TODO: compartment type ends up as 'any' at leafs?

    # TODO: come at divide reaction from the other side:
    #   ie make a call for it, then figure out what the
    #   reaction needs to be
    def divide_reaction(container, mother, divider):
        daughters = divider(mother)

        return {
            'redex': mother,
            'reactum': daughters}

    embedded_tree = {
        'environment': {
            '_type': 'compartment',
            'counts': {},
            'inner': {
                'agent1': {
                    '_type': 'compartment',
                    'counts': {},
                    'inner': {
                        'agent2': {
                            '_type': 'compartment',
                            'counts': {},
                            'inner': {},
                            'transport': {
                                'wires': {
                                    'outer': ['..', '..'],
                                    'inner': ['inner']}}}},
                    'transport': {
                        'wires': {
                            'outer': ['..', '..'],
                            'inner': ['inner']}}}}}}


    mother_tree = {
        'environment': {
            '_type': 'compartment',
            'counts': {
                'A': 15},
            'inner': {
                'mother': {
                    '_type': 'compartment',
                    'counts': {
                        'A': 5}}}}}

    divide_react = {
        '_react': {
            'redex': {
                'mother': {
                    'counts': '@counts'}},
            'reactum': {
                'daughter1': {
                    'counts': '@daughter1_counts'},
                'daughter2': {
                    'counts': '@daughter2_counts'}},
            'calls': [{
                'function': 'divide_counts',
                'arguments': ['@counts', [0.5, 0.5]],
                'bindings': ['@daughter1_counts', '@daughter2_counts']}]}}

    divide_update = {
        '_react': {
            'reaction': 'divide_counts',
            'config': {
                'id': 'mother',
                'state_key': 'counts',
                'daughters': [
                    {'id': 'daughter1', 'ratio': 0.3},
                    {'id': 'daughter2', 'ratio': 0.7}]}}}

    divide_update_concise = {
        '_react': {
            'divide_counts': {
                'id': 'mother',
                'state_key': 'counts',
                'daughters': [
                    {'id': 'daughter1', 'ratio': 0.3},
                    {'id': 'daughter2', 'ratio': 0.7}]}}}


if __name__ == '__main__':
    types = TypeSystem()

    register_compartment(types)
    register_cube(types)

    test_generate_default(types)
    test_apply_update(types)
    test_validate_schema(types)
    test_fill_int(types)
    test_fill_cube(types)
    test_establish_path(types)
    test_fill_in_missing_nodes(types)
    test_fill_from_parse(types)
    test_expected_schema(types)
    test_units(types)
    test_serialize_deserialize(types)
    test_project(types)
    test_foursquare(types)
    test_add_reaction(types)
    test_remove_reaction(types)
    test_replace_reaction(types)
    test_unit_conversion(types)
