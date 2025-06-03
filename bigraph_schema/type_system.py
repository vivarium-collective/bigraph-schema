"""
===========
Type System
===========
"""

import copy
import functools
import random
import traceback
from pprint import pformat as pf

from bigraph_schema import Registry, non_schema_keys, is_schema_key, deep_merge, type_parameter_key
from bigraph_schema.parse import parse_expression
from bigraph_schema.utilities import union_keys
from bigraph_schema.registry import remove_omitted, transform_path, set_star_path

from bigraph_schema.type_functions import (
    registry_types, base_types, unit_types, register_base_reactions, is_empty,
    set_apply, TYPE_FUNCTION_KEYS, SYMBOL_TYPES, is_method_key,
    type_schema_keys, resolve_path)
from bigraph_schema.type_system_adjunct import TypeSystemAdjunct


class TypeSystem(Registry):
    """Handles type schemas and their operation"""

    def __init__(self):
        super().__init__()

        self.inherits = {}

        self.default_registry = Registry(function_keys=[
            'schema',
            'core'])

        self.check_registry = Registry(function_keys=[
            'state',
            'schema',
            'core'])

        self.apply_registry = Registry(function_keys=[
            'current',
            'update',
            'schema',
            'core'])

        self.serialize_registry = Registry(function_keys=[
            'value',
            'schema',
            'core'])

        self.deserialize_registry = Registry(function_keys=[
            'encoded',
            'schema',
            'core'])

        self.fold_registry = Registry(function_keys=[
             'method',
             'state',
             'schema',
             'core'])

        self.react_registry = Registry()
        self.method_registry = Registry()

        # register all the base methods and types
        self.apply_registry.register(
            'set',
            set_apply)

        self.register_types(registry_types)
        self.register_types(base_types)
        self.register_types(unit_types)

        # # TODO -- add a proper registration into registry
        register_base_reactions(self)


    def register_types(self, type_library):
        for type_key, type_data in type_library.items():
            if not self.exists(type_key):
                self.register(
                    type_key,
                    type_data)

        return self


    def update_types(self, type_updates):
        for type_key, type_data in type_updates.items():
            is_update = self.exists(type_key)

            self.register(
                type_key,
                type_data,
                update=is_update)

        return self


    def lookup_registry(self, underscore_key):
        """
        access the registry for the given key
        """

        if underscore_key == '_type':
            return self
        root = underscore_key.strip('_')
        registry_key = f'{root}_registry'
        if hasattr(self, registry_key):
            return getattr(self, registry_key)


    def find_registry(self, underscore_key):
        """
        access the registry for the given key
        and create if it doesn't exist
        """

        registry = self.lookup_registry(underscore_key)
        if registry is None:
            registry = Registry()
            setattr(
                self,
                f'{underscore_key[1:]}_registry',
                registry)

        return registry


    # TODO: explain this method
    def register(self, key, schema, alternate_keys=tuple(), strict=True, update=False):
        """
        register the schema under the given key in the registry
        """

        if isinstance(schema, str):
            schema = self.find(schema)
        schema = copy.deepcopy(schema)
        if self.exists(key) and update:
            if update:
                found = self.find(key)
                schema = deep_merge(
                    found,
                    schema)
                strict = False

        if '_type' not in schema:
            schema['_type'] = key

        if isinstance(schema, dict):
            inherits = schema.get('_inherit', [])  # list of immediate inherits
            if isinstance(inherits, str):
                inherits = [inherits]
                schema['_inherit'] = inherits

            self.inherits[key] = []
            for inherit in inherits:
                inherit_type = self.access(inherit)
                new_schema = copy.deepcopy(inherit_type)

                schema = self.merge_schemas(
                    new_schema,
                    schema)

                self.inherits[key].append(
                    inherit_type)

            parameters = schema.get('_type_parameters', [])
            for subkey, subschema in schema.items():
                if subkey == '_default' or subkey in TYPE_FUNCTION_KEYS or is_method_key(subkey, parameters):
                    if callable(subschema):
                        registry = self.find_registry(subkey)
                        function_name, module_key = registry.register_function(subschema)

                        schema[subkey] = function_name
                    else:
                        schema[subkey] = subschema

                elif subkey not in type_schema_keys:
                    if schema['_type'] in SYMBOL_TYPES:
                        schema[subkey] = subschema
                    else:
                        lookup = self.find(subschema)
                        if lookup is None:
                            raise Exception(
                                f'trying to register a new type ({key}), '
                                f'but it depends on a type ({subkey}) which is not in the registry')
                        else:
                            schema[subkey] = lookup
        else:
            raise Exception(
                f'all type definitions must be dicts '
                f'with the following keys: {type_schema_keys}\nnot: {schema}')

        super().register(
            key,
            schema,
            alternate_keys,
            strict=strict)


    def register_reaction(self, reaction_key, reaction):
        self.react_registry.register(
            reaction_key,
            reaction)


    def merge_schemas(self, current, update):
        if current == update:
            return update
        if current is None:
            return update
        if update is None:
            return current
        if not isinstance(current, dict):
            return update
        if not isinstance(update, dict):
            return update

        merged = {}

        for key in union_keys(current, update):
            if key in current:
                if key in update:
                    subcurrent = current[key]
                    subupdate = update[key]
                    if subcurrent == current or subupdate == update:
                        continue

                    merged[key] = self.merge_schemas(
                        subcurrent,
                        subupdate)
                else:
                    merged[key] = current[key]
            else:
                merged[key] = update[key]

        return merged


    def sort(self, schema, state):
        schema = self.access(schema)

        sort_function = self.choose_method(
            schema,
            state,
            'sort')

        return sort_function(
            self,
            schema,
            state)


    def exists(self, type_key):
        return type_key in self.registry


    def find(self, schema, strict=False):
        """
        expand the schema to its full type information from the type registry
        """

        found = None

        if schema is None:
            return self.access('any', strict=strict)

        elif isinstance(schema, dict):
            if '_description' in schema:
                return schema

            elif '_union' in schema:
                union_schema = {
                    '_type': 'union',
                    '_type_parameters': []}

                for index, element in enumerate(schema['_union']):
                    union_schema['_type_parameters'].append(str(index))
                    union_schema[f'_{index}'] = element

                return self.access(
                    union_schema,
                    strict=strict)

            elif '_type' in schema:
                registry_type = self.retrieve(
                    schema['_type'])

                found = self.merge_schemas(
                    registry_type,
                    schema)

            else:
                found = {
                   key: self.access(
                       branch,
                       strict=strict) if key != '_default' else branch
                   for key, branch in schema.items()}

        elif isinstance(schema, int):
            return schema

        elif isinstance(schema, tuple):
            tuple_schema = {
                '_type': 'tuple',
                '_type_parameters': []}

            for index, element in enumerate(schema):
                tuple_schema['_type_parameters'].append(str(index))
                tuple_schema[f'_{index}'] = element

            return self.access(
                tuple_schema,
                strict=strict)

        elif isinstance(schema, list):
            if isinstance(schema[0], int):
                return schema

            bindings = []
            if len(schema) > 1:
                schema, bindings = schema
            else:
                schema = schema[0]
            found = self.access(
                schema,
                strict=strict)

            if len(bindings) > 0:
                found = found.copy()

                if '_type_parameters' not in found:
                    found['_type_parameters'] = []
                    for index, binding in enumerate(bindings):
                        found['_type_parameters'].append(str(index))
                        found[f'_{index}'] = binding
                else:
                    for parameter, binding in zip(found['_type_parameters'], bindings):
                        binding_type = self.access(
                            binding,
                            strict=strict) or binding

                        found[f'_{parameter}'] = binding_type

        elif isinstance(schema, str):
            found = self.registry.get(schema)

            if found is None and schema not in ('', '{}'):
                try:
                    parse = parse_expression(schema)
                    if parse != schema:
                        found = self.access(
                            parse,
                            strict=strict)
                    elif not strict:
                        found = {'_type': schema}

                except Exception:
                    print(f'type did not parse: {schema}')
                    traceback.print_exc()

        return found


    def access(self, schema, strict=False):
        if isinstance(schema, str):
            return self.access_str(schema, strict=strict)
        else:
            return self.find(schema, strict=strict)


    @functools.lru_cache(maxsize=None)
    def access_str(self, schema, strict=False):
        return self.find(
            schema,
            strict)


    def retrieve(self, schema):
        """
        like access(schema) but raises an exception if nothing is found
        """

        found = self.find(
            schema,
            strict=True)

        if found is None:
            raise Exception(f'schema not found for type: {schema}')
        return found


    def find_parameter(self, schema, parameter):
        schema_key = f'_{parameter}'
        if schema_key not in schema:
            schema = self.access(schema)
        if schema_key not in schema:
            return 'any'
            # raise Exception(f'parameter {parameter} not found in schema:\n  {schema}')

        parameter_type = self.access(
            schema[schema_key])

        return parameter_type


    def parameters_for(self, initial_schema):
        '''
        find any type parameters for this schema if they are present
        '''

        if '_type_parameters' in initial_schema:
            schema = initial_schema
        else:
            schema = self.access(initial_schema)

        if '_type_parameters' not in schema:
            return []
        else:
            result = []
            for parameter in schema['_type_parameters']:
                parameter_key = f'_{parameter}'
                if parameter_key not in schema:
                    raise Exception(
                        f'looking for type parameter {parameter_key} but it is not in the schema:\n  {pf(schema)}')
                else:
                    parameter_type = schema[parameter_key]
                    result.append(
                        parameter_type)

            return result


    def default(self, schema):
        '''
        produce the default value for the provided schema
        '''

        default = None
        found = self.retrieve(schema)

        if '_default' in found:
            default_value = found['_default']
            if isinstance(default_value, str):
                default_method = self.default_registry.access(default_value)
                if default_method and callable(default_method):
                    default = default_method(found, self)
                else:
                    default = self.deserialize(
                        found,
                        default_value)

            elif not '_deserialize' in found:
                raise Exception(
                    f'asking for default but no deserialize in {found}')

            else:
                default = self.deserialize(found, found['_default'])

        else:
            default = {}
            for key, subschema in found.items():
                if not is_schema_key(key):
                    default[key] = self.default(subschema)

        return default


    def choose_method(self, schema, state, method_name):
        '''
        find in the provided state, or schema if not there,
        a method for the given method_name
        '''

        method_key = f'_{method_name}'
        found = None

        if isinstance(state, dict) and method_key in state:
            found = state[method_key]

        elif isinstance(state, dict) and '_type' in state:
            method_type = self.find(
                state['_type'])

            if method_type:
                found = method_type.get(method_key)

        if not found and isinstance(schema, dict) and method_key in schema:
            found = schema[method_key]

        if found is None:
            any_type = self.access('any')
            found = any_type[method_key]

        registry = self.lookup_registry(method_key)
        method_function = registry.access(
            found)

        if method_function is None:
            raise Exception(f'no method "{method_name}" found for state {state} and schema {schema}')

        return method_function


    def slice(self, schema, state, path):
        '''
        find the subschema and substate at a node in the place graph
        given by the provided path
        '''

        if not isinstance(path, (list, tuple)):
            path = [path]

        schema = self.access(schema)
        if '..' in path:
            path = resolve_path(path)

        slice_function = self.choose_method(
            schema,
            state,
            'slice')

        return slice_function(
            schema,
            state,
            path,
            self)


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


    def match(self, original_schema, state, pattern, mode='first', path=()):
        """
        find the path or paths to any instances of a
        given pattern in the tree.

        "mode" can be a few things:
        * immediate: only match top level
        * first: only return the first match
        * random: return a random match of all that matched
        * all (or any other value): return every match in the tree
        """

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

        # TODO: add schema to redex and reactum

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
                schema,
                state,
                reaction.get(reaction_key, {}),
                self)

            redex = react.get('redex', {})
            reactum = react.get('reactum', {})
            calls = react.get('calls', {})

        paths = self.match(
            schema,
            state,
            redex,
            mode=mode)

        # for path in paths:
        #     path_schema, path_state = self.slice(
        #         schema,
        #         state,
        #         path)

        def merge_state(before):
            remaining = remove_omitted(
                redex,
                reactum,
                before)

            merged = deep_merge(
                remaining,
                reactum)

            return merged

        for path in paths:
            state = transform_path(
                state,
                path,
                merge_state)

        return state


    def resolve(self, schema, update):
        if update is None:
            return schema
        else:
            resolve_function = self.choose_method(
                schema,
                update,
                'resolve')

            return resolve_function(
                schema,
                update,
                self)


    def resolve_schemas(self, initial_current, initial_update):
        if initial_current == initial_update:
            return initial_current

        current = self.access(initial_current)
        update = self.access(initial_update)

        if self.equivalent(current, update):
            outcome = current

        elif self.inherits_from(current, update):
            outcome = current

        elif self.inherits_from(update, current):
            outcome = update

        elif '_type' in current and '_type' in update and current['_type'] == update['_type']:
            outcome = {}

            for key in update:
                if key == '_type_parameters' and '_type_parameters' in current:
                    for parameter in update['_type_parameters']:
                        parameter_key = f'_{parameter}'
                        if parameter in current['_type_parameters']:
                            if parameter_key in current:
                                if parameter_key in update:
                                    outcome[parameter_key] = self.resolve_schemas(
                                        current[parameter_key],
                                        update[parameter_key])
                                else:
                                    outcome[parameter_key] = current[parameter_key]
                            elif parameter_key in update:
                                outcome[parameter_key] = update[parameter_key]
                        else:
                            outcome[parameter_key] = update[parameter_key]
                elif key not in current or type_parameter_key(current, key):
                    if update[key]:
                        outcome[key] = update[key]
                    else:
                        outcome[key] = current.get(key)
                elif key in current and current[key]:
                    outcome[key] = self.resolve_schemas(
                        current[key],
                        update[key])
                else:
                    outcome[key] = update[key]

        elif '_type' in update and '_type' not in current:
            outcome = self.resolve(update, current)

        else:
            outcome = self.resolve(current, update)

        return outcome


    def check_state(self, schema, state):
        schema = self.access(schema)

        check_function = self.choose_method(
            schema,
            state,
            'check')

        return check_function(
            schema,
            state,
            self)


    def check(self, initial_schema, state):
        schema = self.retrieve(initial_schema)
        return self.check_state(schema, state)


    def fold_state(self, schema, state, method, values):
        schema = self.access(schema)

        fold_function = self.choose_method(
            schema,
            state,
            'fold')

        return fold_function(
            schema,
            state,
            method,
            values,
            self)


    def fold(self, initial_schema, state, method, values=None):
        schema = self.retrieve(initial_schema)
        return self.fold_state(
            schema,
            state,
            method,
            values)


    def apply_update(self, schema, state, update, top_schema=None, top_state=None, path=None):
        schema = self.access(schema)

        top_schema = top_schema or schema
        top_state = top_state or state
        path = path or []

        if isinstance(update, dict) and '_react' in update:
            new_state = self.react(
                schema,
                state,
                update['_react'])

            state = self.deserialize(schema, new_state)

        elif isinstance(update, dict) and '_fold' in update:
            fold = update['_fold']

            if isinstance(fold, dict):
                method = fold['method']
                values = {
                    key: value
                    for key, value in fold.items()
                    if key != 'method'}

            elif isinstance(fold, str):
                method = fold
                values = {}

            else:
                raise Exception(f'unknown fold: {pf(update)}')

            state = self.fold(
                schema,
                state,
                method,
                values)

        elif '_apply' in schema and schema['_apply'] != 'any':
            apply_function = self.apply_registry.access(schema['_apply'])

            state = apply_function(
                schema,
                state,
                update,
                top_schema,
                top_state,
                path,
                self)

        elif isinstance(schema, str) or isinstance(schema, list):
            schema = self.access(schema)
            state = self.apply_update(
                schema,
                state,
                update,
                top_schema=top_schema,
                top_state=top_state,
                path=path)

        elif isinstance(update, dict):
            for key, branch in update.items():
                if key not in schema:
                    raise Exception(
                        f'trying to update a key that is not in the schema '
                        f'for state: {key}\n{state}\nwith schema:\n{schema}')
                else:
                    subupdate = self.apply_update(
                        self.access(schema[key]),
                        state[key],
                        branch,
                        top_schema=top_schema,
                        top_state=top_state,
                        path=path + [key])

                    state[key] = subupdate
        else:
            raise Exception(
                f'trying to apply update\n  {update}\nto state\n  {state}\n'
                f'with schema\n  {schema}\nbut the update is not a dict')

        return state


    def apply(self, original_schema, initial, update):
        schema = self.access(original_schema)
        state = copy.deepcopy(initial)
        return self.apply_update(
            schema,
            state,
            update)


    def merge_recur(self, schema, state, update):
        if is_empty(schema):
            merge_state = update

        elif is_empty(update):
            merge_state = state

        elif isinstance(update, dict):
            if isinstance(state, dict):
                for key, value in update.items():
                    if is_schema_key(key):
                        state[key] = value
                    else:
                        if isinstance(schema, str):
                            schema = self.access(schema)

                        state[key] = self.merge_recur(
                            schema.get(key),
                            state.get(key),
                            value)
                merge_state = state
            else:
                merge_state = update
        else:
            merge_state = update

        return merge_state


    def merge(self, schema, state, path, update_schema, update_state, defer=False):
        top_schema, top_state = self.set_slice(
            schema,
            state,
            path,
            update_schema,
            update_state,
            defer)

        return self.generate(top_schema, top_state)


    def bind(self, schema, state, key, target_schema, target_state):
        schema = self.retrieve(schema)

        bind_function = self.choose_method(
            schema,
            state,
            'bind')

        return bind_function(
            schema,
            state,
            key,
            target_schema,
            target_state,
            self)


    def set_slice(self, schema, state, path, target_schema, target_state, defer=False):
        '''
        Makes a local modification to the schema/state at the path, and
        returns the top_schema and top_state
        '''

        path = resolve_path(path)

        if len(path) == 0:
            # deal with paths of length 0
            # this should never happen?
            merged_schema = self.resolve_schemas(
                schema,
                target_schema)

            merged_state = deep_merge(
                state,
                target_state)

            return merged_schema, merged_state

        elif len(path) == 1:
            key = path[0]
            destination_schema, destination_state = self.slice(
                schema,
                state,
                key)

            final_schema = self.resolve_schemas(
                destination_schema,
                target_schema)

            if not defer:
                result_state = self.merge_recur(
                    final_schema,
                    destination_state,
                    target_state)

            else:
                result_state = self.merge_recur(
                    final_schema,
                    target_state,
                    destination_state)

            return self.bind(
                schema,
                state,
                key,
                final_schema,
                result_state)

        else:
            head = path[0]
            tail = path[1:]

            down_schema, down_state = self.slice(
                schema,
                state,
                head)

            if head == '*':
                result_schema, result_state = down_schema, down_state
                for key in down_state:
                    if key in target_state:
                        subtarget_schema, subtarget_state = self.slice(
                            target_schema,
                            target_state,
                            key)

                        try:
                            result_schema, result_state = self.set_slice(
                                result_schema,
                                result_state,
                                tail,
                                subtarget_schema,
                                subtarget_state,
                                defer=defer)

                        except Exception as e:
                            raise Exception(
                                f'failed to set_slice at path {path}\n{str(e)}')

                        schema, state = self.bind(
                            schema,
                            state,
                            key,
                            result_schema,
                            result_state)

                return schema, state

            else:
                try:
                    result_schema, result_state = self.set_slice(
                        down_schema,
                        down_state,
                        tail,
                        target_schema,
                        target_state,
                        defer=defer)
                except Exception as e:
                    raise Exception(f'failed to set_slice at path {path}\n{str(e)}')

                return self.bind(
                    schema,
                    state,
                    head,
                    result_schema,
                    result_state)


    def serialize(self, schema, state):
        schema = self.retrieve(schema)

        serialize_function = self.choose_method(
            schema,
            state,
            'serialize')

        return serialize_function(
            schema,
            state,
            self)


    def deserialize(self, schema, state):
        schema = self.retrieve(schema)

        deserialize_function = self.choose_method(
            schema,
            state,
            'deserialize')

        deserialized = deserialize_function(
            schema,
            state,
            self)

        return deserialized


    def fill_state(self, schema, state=None, top_schema=None, top_state=None, path=None, type_key=None, context=None):
        # if a port is disconnected, build a store
        # for it under the '_open' key in the current
        # node (?)

        # inform the user that they have disconnected
        # ports somehow

        if schema is None:
            return None
        if state is None:
            state = self.default(schema)
        if top_schema is None:
            top_schema = schema
        if top_state is None:
            top_state = state
        if path is None:
            path = []

        if '_inputs' in schema:
            inputs = state.get('inputs', {})
            state = self.fill_ports(
                schema['_inputs'],
                wires=inputs,
                state=state,
                top_schema=top_schema,
                top_state=top_state,
                path=path)

        if '_outputs' in schema:
            outputs = state.get('outputs', {})
            state = self.fill_ports(
                schema['_outputs'],
                wires=outputs,
                state=state,
                top_schema=top_schema,
                top_state=top_state,
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
                    top_schema=top_schema,
                    top_state=top_state,
                    path=subpath)

            state_keys = non_schema_keys(state)
            for key in set(state_keys) - set(branches):
                subschema, substate = self.slice(
                    schema,
                    state,
                    key)

                subpath = path + [key]
                state[key] = self.fill_state(
                    subschema,
                    substate,
                    top_schema=top_schema,
                    top_state=top_state,
                    path=subpath)

        return state


    def fill(self, original_schema, state=None):
        schema = self.access(original_schema)

        return self.fill_state(
            schema,
            state=state)


    def ports_schema(self, schema, instance, edge_path, ports_key='inputs'):
        found = self.access(schema)

        edge_schema, edge_state = self.slice(
            schema,
            instance,
            edge_path)

        ports_schema = edge_state.get(f'_{ports_key}',
                                      edge_schema.get(f'_{ports_key}'))
        ports = edge_state.get(ports_key)

        return ports_schema, ports


    # start here for different methods of execution
    def view(self, schema, wires, path, top_schema=None, top_state=None):
        result = {}

        if isinstance(wires, str):
            wires = [wires]

        if isinstance(wires, (list, tuple)):
            _, result = self.slice(
                top_schema,
                top_state,
                list(path) + list(wires))

        elif isinstance(wires, dict):
            result = {}
            for port_key, port_path in wires.items():
                subschema, _ = self.slice(
                    schema,
                    {},
                    port_key)

                inner_view = self.view(
                    subschema,
                    port_path,
                    path,
                    top_schema=top_schema,
                    top_state=top_state)

                if inner_view is not None:
                    result[port_key] = inner_view
        else:
            raise Exception(f'trying to view state with these ports:\n{schema}\nbut not sure what these wires are:\n{wires}')

        return result


    def view_edge(self, schema, state, edge_path=None, ports_key='inputs'):
        """
        project the current state into a form the edge expects, based on its ports.
        """

        if schema is None:
            return None
        if state is None:
            state = self.default(schema)
        if edge_path is None:
            edge_path = []

        ports_schema, ports = self.ports_schema(
            schema,
            state,
            edge_path=edge_path,
            ports_key=ports_key)

        if not ports_schema:
            return None
        if ports is None:
            return None

        return self.view(
            ports_schema,
            ports,
            edge_path[:-1],
            top_schema=schema,
            top_state=state)


    def project(self, ports, wires, path, states):
        result = {}

        if isinstance(wires, str):
            wires = [wires]

        if isinstance(wires, (list, tuple)):
            destination = resolve_path(list(path) + list(wires))
            result = set_star_path(
                result,
                destination,
                states)

        elif isinstance(wires, dict):
            if isinstance(states, list):
                result = [
                    self.project(ports, wires, path, state)
                    for state in states]
            else:
                branches = []
                for key in wires.keys():
                    subports, substates = self.slice(ports, states, key)
                    projection = self.project(
                        subports,
                        wires[key],
                        path,
                        substates)

                    if projection is not None:
                        branches.append(projection)

                branches = [
                    branch
                    for branch in branches
                    if branch is not None]

                result = {}
                for branch in branches:
                    deep_merge(result, branch)
        else:
            raise Exception(
                f'inverting state\n  {states}\naccording to ports schema\n  {ports}\nbut wires are not recognized\n  {wires}')

        return result


    def project_edge(self, schema, instance, edge_path, states, ports_key='outputs'):
        """
        Given states from the perspective of an edge (through its ports), produce states aligned to the tree
        the wires point to.
        (inverse of view)
        """

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


    def equivalent(self, icurrent, iquestion):
        if icurrent == iquestion:
            return True

        current = self.access(icurrent)
        question = self.access(iquestion)

        if current is None:
            return question is None

        if current == {}:
            return question == {}

        elif '_type' in current:
            if '_type' in question:
                if current['_type'] == question['_type']:
                    if '_type_parameters' in current:
                        if '_type_parameters' in question:
                            for parameter in current['_type_parameters']:
                                parameter_key = f'_{parameter}'
                                if parameter_key in question:
                                    if parameter_key not in current:
                                        return False

                                    if not self.equivalent(current[parameter_key], question[parameter_key]):
                                        return False
                        else:
                            return False
                else:
                    return False
            else:
                return False
        else:
            if '_type' in question:
                return False

        for key, value in current.items():
            if not is_schema_key(key): # key not in type_schema_keys:
                if key not in question or not self.equivalent(current.get(key), question[key]):
                    return False

        for key in set(question.keys()) - set(current.keys()):
            if not is_schema_key(key): # key not in type_schema_keys:
                if key not in question or not self.equivalent(current.get(key), question[key]):
                    return False

        return True


    def inherits_from(self, descendant, ancestor):
        descendant = self.access(descendant)
        ancestor = self.access(ancestor)

        if descendant == {}:
            return ancestor == {}

        if descendant is None:
            return True

        if ancestor is None:
            return False

        if isinstance(ancestor, int):
            if isinstance(descendant, int):
                return ancestor == descendant
            else:
                return False

        elif isinstance(ancestor, list):
            if isinstance(descendant, list):
                if len(ancestor) == len(descendant):
                    for a, d in zip(ancestor, descendant):
                        if not self.inherits_from(d, a):
                            return False
                else:
                    return False

        elif '_type' in ancestor and ancestor['_type'] == 'any':
            return True

        elif '_type' in descendant:
            if '_inherit' in descendant:
                for inherit in descendant['_inherit']:

                    if self.equivalent(inherit, ancestor) or self.inherits_from(inherit, ancestor):
                        return True

                return False

            elif '_type_parameters' in descendant:
                for type_parameter in descendant['_type_parameters']:
                    parameter_key = f'_{type_parameter}'
                    if parameter_key in ancestor and parameter_key in descendant:
                        if not self.inherits_from(descendant[parameter_key], ancestor[parameter_key]):
                            return False

            if '_type' not in ancestor or descendant['_type'] != ancestor['_type']:
                return False

        else:
            for key, value in ancestor.items():
                if key not in type_schema_keys:
                # if not key.startswith('_'):
                    if key in descendant:
                        if not self.inherits_from(descendant[key], value):
                            return False
                    else:
                        return False

        return True



    def wire_schema(self, schema, wires, path=None):
        outcome = {}
        path = path or []

        if isinstance(wires, dict):
            for key, subwires in wires.items():
                outcome[key] = self.wire_schema(
                    schema,
                    wires[key],
                    path + [key])

        else:
            _, outcome = self.slice('schema', schema, wires)

        return outcome


    def generate_recur(self, schema, state, top_schema=None, top_state=None, path=None):
        found = self.retrieve(
            schema)

        generate_function = self.choose_method(
            found,
            state,
            'generate')

        return generate_function(
            self,
            found,
            state,
            top_schema=top_schema,
            top_state=top_state,
            path=path)


    def generate(self, schema, state):
        merged_schema, merged_state = self.sort(
            schema,
            state)

        _, _, top_schema, top_state = self.generate_recur(
            merged_schema,
            merged_state)

        return top_schema, top_state


    def find_method(self, schema, method_key):
        if not isinstance(schema, dict) or method_key not in schema:
            schema = self.access(schema)

        if method_key in schema:
            registry = self.lookup_registry(
                method_key)

            if registry is not None:
                method_name = schema[method_key]
                method = registry.access(method_name)

                return method

    apply_slice = TypeSystemAdjunct.apply_slice
    complete = TypeSystemAdjunct.complete
    compose = TypeSystemAdjunct.compose
    dataclass = TypeSystemAdjunct.dataclass
    define = TypeSystemAdjunct.define
    fill_ports = TypeSystemAdjunct.fill_ports
    hydrate = TypeSystemAdjunct.hydrate
    import_types = TypeSystemAdjunct.import_types
    infer_wires = TypeSystemAdjunct.infer_wires
    infer_edge = TypeSystemAdjunct.infer_edge
    infer_schema = TypeSystemAdjunct.infer_schema
    link_place = TypeSystemAdjunct.link_place
    lookup = TypeSystemAdjunct.lookup
    query = TypeSystemAdjunct.query
    representation = TypeSystemAdjunct.representation
    resolve_parameters = TypeSystemAdjunct.resolve_parameters
    set = TypeSystemAdjunct.set
    set_update = TypeSystemAdjunct.set_update
    types = TypeSystemAdjunct.types
    validate = TypeSystemAdjunct.validate
    validate_schema = TypeSystemAdjunct.validate_schema
    validate_state = TypeSystemAdjunct.validate_state
