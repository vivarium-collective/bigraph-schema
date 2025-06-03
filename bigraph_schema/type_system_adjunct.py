import copy
import inspect
from bigraph_schema import non_schema_keys, is_schema_key
from bigraph_schema.type_functions import (
        apply_schema, TYPE_SCHEMAS, type_schema_keys, resolve_path)

class TypeSystemAdjunct():
    """holds implementations of defunct or not yet ready TypeSystem methods"""
    @staticmethod
    def import_types(type_system, package, strict=False):
        for type_key, type_data in package.items():
            if not (strict and type_system.exists(type_key)):
                type_system.register(
                    type_key,
                    type_data)

    @staticmethod
    def define(type_system, method_name, methods):
        method_key = f'_{method_name}'
        for type_key, method in methods.items():
            type_system.register(
                type_key,
                {method_key: method})

    @staticmethod
    def link_place(type_system, place, link):
        pass


    @staticmethod
    def compose(type_system, a, b):
        pass


    @staticmethod
    def query(type_system, schema, instance, redex):
        subschema = {}
        return subschema

    @staticmethod
    def complete(type_system, initial_schema, initial_state):
        full_schema = type_system.access(
            initial_schema)

        state = type_system.deserialize(
            full_schema,
            initial_state)

        # fill in the parts of the composition schema
        # determined by the state
        schema, state = type_system.infer_schema(
            full_schema,
            state)

        final_state = type_system.fill(schema, state)

        # TODO: add flag to types.access(copy=True)
        return type_system.access(schema), final_state

    @staticmethod
    def lookup(type_system, type_key, attribute):
        return type_system.access(type_key).get(attribute)

    @staticmethod
    def resolve_parameters(type_system, type_parameters, schema):
        """
        find the types associated with any type parameters in the schema
        """

        return {
            type_parameter: type_system.access(
                schema.get(f'_{type_parameter}'))
            for type_parameter in type_parameters}

    @staticmethod
    def types(type_system):
        return {
            type_key: type_data
            for type_key, type_data in type_system.registry.items()}

    def types(self):
        return {
            type_key: type_data
            for type_key, type_data in self.registry.items()}


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


    @staticmethod
    def validate_schema(type_system, schema, enforce_connections=False):
        # TODO:
        #   check() always returns true or false,
        #   validate() returns information about what doesn't match

        # add ports and wires
        # validate ports are wired to a matching type,
        #   either the same type or a subtype (more specific type)
        # declared ports are equivalent to the presence of a process
        #   where the ports need to be looked up

        report = {}

        if schema is None:
            report = 'schema cannot be None'

        elif isinstance(schema, str):
            typ = type_system.access(
                schema,
                strict=True)

            if typ is None:
                report = f'type: {schema} is not in the registry'

        elif isinstance(schema, dict):
            report = {}

            schema_keys = set([])
            branches = set([])

            for key, value in schema.items():
                if key == '_type':
                    typ = type_system.access(
                        value,
                        strict=True)
                    if typ is None:
                        report[key] = f'type: {value} is not in the registry'

                elif key in type_schema_keys:
                    schema_keys.add(key)
                    registry = type_system.lookup_registry(key)
                    if registry is None or key == '_default':
                        # deserialize and serialize back and check it is equal
                        pass
                    elif isinstance(value, str):
                        element = registry.access(
                            value,
                            strict=True)

                        if element is None:
                            report[key] = f'no entry in the {key} registry for: {value}'
                    elif not inspect.isfunction(value):
                        report[key] = f'unknown value for key {key}: {value}'
                else:
                    branches.add(key)
                    branch_report = type_system.validate_schema(value)
                    if len(branch_report) > 0:
                        report[key] = branch_report

        return report


    # TODO: if its an edge, ensure ports match wires
    # TODO: make this work again, return information about what is wrong
    #   with the schema
    @staticmethod
    def validate_state(type_system, original_schema, state):
        schema = type_system.access(original_schema)
        validation = {}

        if '_serialize' in schema:
            if '_deserialize' not in schema:
                validation = {
                    '_deserialize': f'serialize found in type without deserialize: {schema}'
                }
            else:
                serialize = type_system.serialize_registry.access(
                    schema['_serialize'])
                deserialize = type_system.deserialize_registry.access(
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
                        subvalidation = type_system.validate_state(
                            subschema,
                            state[key])
                        if not (subvalidation is None or len(subvalidation) == 0):
                            validation[key] = subvalidation

        return validation


    @staticmethod
    def representation(type_system, schema, path=None, parents=None):
        '''
        produce a string representation of the schema
        * intended to be the inverse of parse_expression()
        '''

        path = path or []
        parents = parents or []
        schema_id = id(schema)

        if schema_id in parents:
            index = parents.index(schema_id)
            reference = path[:index]
            output = '/'.join(reference)

            return f'/{output}'

        if isinstance(schema, str):
            return schema

        elif isinstance(schema, tuple):
            inner = [
                type_system.representation(
                    element,
                    path + [index],
                    parents + [schema_id])
                for index, element in enumerate(schema)]

            pipes = '|'.join(inner)
            return f'({pipes})'

        elif isinstance(schema, dict):
            if '_type' in schema:
                type = schema['_type']

                inner = []
                block = ''
                if '_type_parameters' in schema:
                    for parameter_key in schema['_type_parameters']:
                        schema_key = f'_{parameter_key}'
                        if schema_key in schema:
                            parameter = type_system.representation(
                                schema[schema_key],
                                path + [schema_key],
                                parents + [schema_id])
                            inner.append(parameter)
                        else:
                            inner.append('()')

                    commas = ','.join(inner)
                    block = f'[{commas}]'

                if type == 'tuple':
                    pipes = '|'.join(inner)
                    return f'({pipes})'
                else:
                    return f"{type}{block}"

            else:
                inner = {}
                for key in non_schema_keys(schema):
                    subschema = type_system.representation(
                        schema[key],
                        path + [key],
                        parents + [schema_id])

                    inner[key] = subschema

                colons = [
                    f'{key}:{value}'
                    for key, value in inner.items()]

                pipes = '|'.join(colons)
                return f'({pipes})'
        else:
            return str(schema)

    @staticmethod
    def validate(type_system, schema, state):
        # TODO:
        #   go through the state using the schema and
        #   return information about what doesn't match

        return {}

    @staticmethod
    def apply_slice(type_system, schema, state, path, update):
        path = path or ()
        if len(path) == 0:
            result = type_system.apply(
                schema,
                state,
                update)

        else:
            subschema, substate = type_system.slice(
                schema,
                state,
                path[0])

            if len(path) == 1:
                subresult = type_system.apply(
                    subschema,
                    substate,
                    update)

                result = type_system.bind(
                    schema,
                    state,
                    path[1:],
                    subschema,
                    subresult)

            else:
                subresult = type_system.apply_slice(
                    subschema,
                    substate,
                    path[1:],
                    update)

                result = state

        return result


    @staticmethod
    def set_update(type_system, schema, state, update):
        if '_apply' in schema:
            apply_function = type_system.apply_registry.access('set')

            state = apply_function(
                schema,
                state,
                update,
                type_system)

        elif isinstance(schema, str) or isinstance(schema, list):
            schema = type_system.access(schema)
            state = type_system.set_update(schema, state, update)

        elif isinstance(update, dict):
            for key, branch in update.items():
                if key not in schema:
                    raise Exception(
                        f'trying to update a key that is not in the schema'
                        f'for state: {key}\n{state}\nwith schema:\n{schema}')
                else:
                    subupdate = type_system.set_update(
                        schema[key],
                        state[key],
                        branch)

                    state[key] = subupdate
        else:
            raise Exception(
                f'trying to apply update\n  {update}\nto state\n  {state}\n'
                f'with schema\n{schema}, but the update is not a dict')

        return state


    @staticmethod
    def set(type_system, original_schema, initial, update):
        schema = type_system.access(original_schema)
        state = copy.deepcopy(initial)

        return type_system.set_update(schema, state, update)

    @staticmethod
    def fill_ports(type_system, interface, wires=None, state=None,
                   top_schema=None, top_state=None, path=None):
        # deal with wires
        if wires is None:
            wires = {}
        if state is None:
            state = {}
        if top_schema is None:
            top_schema = schema
        if top_state is None:
            top_state = state
        if path is None:
            path = []

        if isinstance(interface, str):
            interface = {'_type': interface}

        for port_key, subwires in wires.items():
            if port_key in interface:
                port_schema = interface[port_key]
            else:
                port_schema, subwires = type_system.slice(
                    interface,
                    wires,
                    port_key)

            if isinstance(subwires, dict):
                if isinstance(state, dict):
                    state = type_system.fill_ports(
                        port_schema,
                        wires=subwires,
                        state=state,
                        top_schema=top_schema,
                        top_state=top_state,
                        path=path)

            else:
                if isinstance(subwires, str):
                    subwires = [subwires]

                subschema, substate = type_system.set_slice(
                    top_schema,
                    top_state,
                    path[:-1] + subwires,
                    port_schema,
                    type_system.default(port_schema),
                    defer=True)

        return state

    # def infer_wires(self, ports, state,
    #                 wires, top_schema=None,
    #                 top_state=None, path=None, internal_path=None):
    @staticmethod
    def infer_wires(type_system, ports, wires, top_schema=None, top_state=None,
                    path=None, internal_path=None):
        top_schema = top_schema or {}
        top_state = top_state or state
        path = path or ()
        internal_path = internal_path or ()

        if isinstance(ports, str):
            ports = type_system.access(ports)

        if isinstance(wires, (list, tuple)):
            if len(wires) == 0:
                destination_schema, destination_state = top_schema, top_state

            else:
                destination_schema, destination_state = type_system.slice(
                    top_schema,
                    top_state,
                    path[:-1] + wires)

            merged_schema = apply_schema(
                'schema',
                destination_schema,
                ports,
                type_system)

            merged_state = type_system.complete(
                merged_schema,
                destination_state)

        else:
            for port_key, port_wires in wires.items():
                subschema, substate = type_system.slice(
                    ports,
                    {},
                    port_key)

                if isinstance(port_wires, dict):
                    top_schema, top_state = type_system.infer_wires(
                        subschema,
                        # substate,
                        port_wires,
                        top_schema=top_schema,
                        top_state=top_state,
                        path=path,
                        internal_path=internal_path+(port_key,))

                # port_wires must be a list
                elif len(port_wires) == 0:
                    raise Exception(f'no wires at port "{port_key}" in ports {ports} with state {state}')

                else:
                    compound_path = resolve_path(
                        path[:-1] + tuple(port_wires))

                    compound_schema, compound_state = type_system.set_slice(
                        {}, {},
                        compound_path,
                        subschema or 'any',
                        type_system.default(subschema))

                    top_schema = type_system.resolve(
                        top_schema,
                        compound_schema)

                    top_state = type_system.merge_recur(
                        top_schema,
                        compound_state,
                        top_state)

        return top_schema, top_state


    @staticmethod
    def infer_edge(type_system, schema, state,
                   top_schema=None, top_state=None, path=None):
        '''
        given the schema and state for this edge, and its path relative to
        the top_schema and top_state, make all the necessary completions to
        both the schema and the state according to the input and output schemas
        of this edge in '_inputs' and '_outputs', along the wires in its state
        under 'inputs' and 'outputs'.

        returns the top_schema and top_state, even if the edge is deeply embedded,
        as the particular wires could have implications anywhere in the tree.
        '''

        schema = schema or {}
        top_schema = top_schema or schema
        top_state = top_state or state
        path = path or ()

        if type_system.check('edge', state):
            for port_key in ['inputs', 'outputs']:
                ports = state.get(port_key)
                schema_key = f'_{port_key}'
                port_schema = schema.get(schema_key, {})
                state_schema = state.get(schema_key, {})

                schema[schema_key] = type_system.resolve(
                    port_schema,
                    type_system.access(
                        state_schema))

                if ports:
                    top_schema, top_state = type_system.infer_wires(
                        schema[schema_key],
                        # state,
                        ports,
                        top_schema=top_schema,
                        top_state=top_state,
                        path=path)

        return top_schema, top_state


    @staticmethod
    def infer_schema(type_system, schema, state,
                     top_schema=None, top_state=None, path=None):
        """
        Given a schema fragment and an existing state with _type keys,
        return the full schema required to describe that state,
        and whatever state was hydrated (edges) during this process

        """

        # during recursive call, schema is kept at the top level and the
        # path is used to access it (!)

        schema = schema or {}
        top_schema = top_schema or schema
        top_state = top_state or state
        path = path or ()

        if isinstance(state, dict):
            state_schema = None
            if '_type' in state:
                state_type = {
                    key: value
                    for key, value in state.items()
                    if is_schema_key(key)}

                schema = type_system.resolve(
                    schema,
                    state_type)

            if '_type' in schema:
                hydrated_state = type_system.deserialize(
                    schema,
                    state)

                top_schema, top_state = type_system.set_slice(
                    top_schema,
                    top_state,
                    path,
                    schema,
                    hydrated_state)

                top_schema, top_state = type_system.infer_edge(
                    schema,
                    hydrated_state,
                    top_schema,
                    top_state,
                    path)

            else:
                for key in state:
                    inner_path = path + (key,)
                    inner_schema, inner_state = type_system.slice(
                        schema,
                        state,
                        key)

                    top_schema, top_state = type_system.infer_schema(
                        inner_schema,
                        inner_state,
                        top_schema=top_schema,
                        top_state=top_state,
                        path=inner_path)

        elif isinstance(state, str):
            pass

        else:
            type_schema = TYPE_SCHEMAS.get(
                type(state).__name__,
                'any')

            top_schema, top_state = type_system.set_slice(
                top_schema,
                top_state,
                path,
                type_schema,
                state)

        return top_schema, top_state

    # TODO: maybe all fields are optional?
    @staticmethod
    def dataclass(type_system, schema, path=None):
        path = path or []

        dataclass_function = type_system.choose_method(
            schema,
            {},
            'dataclass')

        return dataclass_function(
            schema,
            path,
            type_system)

    @staticmethod
    def hydrate(type_system, schema, state):
        hydrated = type_system.deserialize(schema, state)
        return type_system.fill(schema, hydrated)

