import copy
import pprint

from bigraph_schema.registry import Registry, TypeRegistry, RegistryRegistry, type_schema_keys, optional_schema_keys, deep_merge, get_path, establish_path
from bigraph_schema.units import units, render_units_type, parse_dimensionality


class SchemaTypes():
    def __init__(self):
        self.apply_registry = Registry()
        self.serialize_registry = Registry()
        self.deserialize_registry = Registry()
        self.divide_registry = Registry()
        self.type_registry = TypeRegistry()

        self.registry_registry = RegistryRegistry()
        self.registry_registry.register('_type', self.type_registry)
        self.registry_registry.register('_apply', self.apply_registry)
        self.registry_registry.register('_divide', self.divide_registry)
        self.registry_registry.register('_serialize', self.serialize_registry)
        self.registry_registry.register('_deserialize', self.deserialize_registry)


    def validate_schema(self, schema, enforce_connections=False):
        # add ports and wires
        # validate ports are wired to a matching type,
        #   either the same type or a subtype (more specific type)
        # declared ports are equivalent to the presence of a process
        #   where the ports need to be looked up

        if not isinstance(schema, dict):
            return f'schema is not a dict: {schema}'

        report = {}

        schema_keys = set([])
        branches = set([])

        for key, value in schema.items():
            if key == '_type':
                typ = self.type_registry.access(value)
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

        # # We will need this when building states to check to see if we are
        # # trying to instantiate an abstract type, but we can still register
        # # register abstract types so it is not invalid
        # if len(schema_keys) > 0 and len(branches) == 0:
        #     undeclared = set(type_schema_keys) - schema_keys
        #     if len(undeclared) > 0:
        #         for key in undeclared:
        #             if not key in optional_schema_keys:
        #                 report[key] = f'missing required key: {key} for declaring atomic type'

        return report


    def validate_state(self, schema, state):
        schema = self.type_registry.substitute_type(schema)
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


    def generate_default(self, schema):
        default = None

        if isinstance(schema, str):
            schema = self.type_registry.access(schema)
        elif '_type' in schema:
            schema = self.type_registry.substitute_type(schema)

        if '_default' in schema:
            if not '_deserialize' in schema:
                raise Exception(
                    f'asking for default for {type_key} but no deserialize in {schema}')
            deserialize_function = self.deserialize_registry.access(
                schema['_deserialize'])
            default = deserialize_function(
                schema['_default'],
                schema.get('_type_parameters'),
                self)
        else:
            default = {}
            for key, subschema in schema.items():
                if key not in type_schema_keys:
                    default[key] = self.generate_default(subschema)

        return default
        

    def generate_default_type(self, type_key):
        schema = self.type_registry.access(type_key)
        return self.generate_default(schema)


    def apply_update(self, schema, state, update):
        # expects an expanded schema

        if '_apply' in schema:
            apply_function = self.apply_registry.access(schema['_apply'])
            
            state = apply_function(
                state,
                update,
                schema.get('_type_parameters'),
                self)

        elif isinstance(update, dict):
            for key, branch in update.items():
                if key not in schema:
                    raise Exception(f'trying to update a key that is not in the schema {key} for state:\n{state}\nwith schema:\n{schema}')
                else:
                    subupdate = self.apply_update(
                        schema[key],
                        state[key],
                        branch)

                    state[key] = subupdate

        else:
            schema = self.type_registry.expand_schema(schema)
            state = self.apply_update(schema, state, update)

        return state


    # TODO: provide an expanded form of the type registry, where everything
    #   is already pre-expanded
    def apply(self, schema, initial, update):
        expanded = self.type_registry.expand(schema)
        state = copy.deepcopy(initial)
        return self.apply_update(expanded, initial, update)


    def serialize(self, schema, state):
        if isinstance(schema, str):
            serialize_function = serialize_registery.access(schema)
            if serialize_function is None:
                raise Exception(f'serialization function not in the registry')
            else:
                return serialize_function(
                    state,
                    type_parameters,
                    self)
        # else:
        #     for key, subschema in schema:


    def fill_ports(self, schema, wires=None, state=None, top=None, path=()):
        # deal with wires
        if wires is None:
            wires = {}
        if state is None:
            state = {}
        if top is None:
            top = state

        more_wires = state.get('wires', {})
        wires = deep_merge(wires, more_wires)

        for port_key, port_schema in schema.items():
            if port_key in wires:
                subwires = wires[port_key]
                if isinstance(subwires, dict):
                    state[port_key] = fill_ports(
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
                        destination[destination_key] = self.generate_default(
                            port_schema)
            else:
                # handle unconnected ports
                pass

        return state


    def fill_state(self, schema, state=None, top=None, path=(), type_key=None, context=None):
        # if a port is disconnected, build a store
        # for it under the '_open' key in the current
        # node

        # inform the user that they have disconnected
        # ports somehow

        if top is None:
            top = state

        schema = self.type_registry.substitute_type(schema)

        if state is None:
            if '_default' in schema:
                state = self.generate_default(schema)
            else:
                state = {}

        if isinstance(schema, str):
            raise Exception(
                f'schema cannot be a str: {str}'
            )

        for key, subschema in schema.items():
            if key == '_ports':
                wires = state.get('wires', {})
                state = self.fill_ports(
                    subschema,
                    wires=wires,
                    state=state,
                    top=top,
                    path=path)

            elif key not in type_schema_keys:
                subpath = path + (key,)
                if isinstance(state, dict):
                    state[key] = self.fill_state(
                        subschema,
                        state=state.get(key),
                        top=top,
                        path=subpath)
        
        return state


    def fill(self, schema, state=None):
        if state is not None:
            state = copy.deepcopy(state)
        return self.fill_state(schema, state=state)


    def link_place(self, place, link):
        pass


    def compose(self, a, b):
        pass


    # maybe vivarium?
    def hydrate(self, schema):
        return {}
    

    def dehydrate(self, schema):
        return {}


    def query(self, schema, state, redex):
        subschema = {}
        return subschema


    def substitute(self, schema, state, reactum):
        return state


    def react(self, schema, redex, reactum):
        return {}


def accumulate(current, update, type_parameters, types):
    return current + update

def concatenate(current, update, type_parameters, types):
    return current + update

def divide_float(value, type_parameters, types):
    half = value / 2.0
    return (half, half)

# support function types for registrys?
# def divide_int(value: int, _) -> tuple[int, int]:
def divide_int(value, type_parameters, types):
    half = value // 2
    other_half = half
    if value % 2 == 1:
        other_half += 1
    return half, other_half


# class DivideRegistry(Registry):
    

# def divide_longest(dimensions: Dimension) -> Tuple[Dimension, Dimension]:
def divide_longest(dimensions, type_parameters, types):
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


def divide_list(l, type_parameters, types):
    result = [[], []]
    divide_type = type_parameters['element']
    divide = types.registry_registry.type_attribute(
        divide_type,
        '_divide')

    for item in l:
        if isinstance(item, list):
            divisions = divide_list(item)
        else:
            divisions = divide(item)

        result[0].append(divisions[0])
        result[1].append(divisions[1])

    return result


def replace(old_value, new_value, type_parameters, types):
    return new_value


def serialize_string(s, type_parameters, types):
    return f'"{s}"'

def deserialize_string(s, type_parameters, types):
    if s[0] != '"' or s[-1] != '"':
        raise Exception(f'deserializing str which requires double quotes: {s}')
    return s[1:-1]


def to_string(value, type_parameters, types):
    return str(value)

def deserialize_int(i, type_parameters, types):
    return int(i)

def deserialize_float(i, type_parameters, types):
    return float(i)

def evaluate(code, type_parameters, types):
    return eval(code)


# TODO: make these work
def apply_tree(current, update, type_parameters, types):
    pass

def divide_tree(tree, type_parameters, types):
    result = [{}, {}]
    # get the type of the values for this dict
    divide_type = type_parameters['leaf']
    divide_function = types.registry_registry.type_attribute(
        divide_type,
        '_divide')

    for key, value in tree:
        if isinstance(value, dict):
            divisions = divide_tree(value)
        else:
            divisions = types.divide(divide_type, value)

        result[0][key], result[1][key] = divisions

    return result

def serialize_tree(value, type_parameters, types):
    return value

def deserialize_tree(value, type_parameters, types):
    return value


def apply_dict(current, update, type_parameters, types):
    pass

def divide_dict(value, type_parameters, types):
    return value

def serialize_dict(value, type_parameters, types):
    return value

def deserialize_dict(value, type_parameters, types):
    return value


def apply_maybe(current, update, type_parameters, types):
    if current is None or update is None:
        return update
    else:
        maybe_type = types.type_registry.access(
            type_parameters[0])
        return apply_update(maybe_type, current, update)

def divide_maybe(value, type_parameters):
    if value is None:
        return [None, None]
    else:
        pass

def serialize_maybe(value, type_parameters, types):
    if value is None:
        return NONE_SYMBOL
    else:
        maybe_type = types.type_registry.access(
            type_parameters[0])
        return serialize(maybe_type, value)

def deserialize_maybe(encoded, type_parameters, types):
    if encoded == NONE_SYMBOL:
        return None
    else:
        maybe_type = types.type_registry.access(
            type_parameters[0])
        return deserialize(maybe_type, encoded)


# TODO: deal with all the different unit types
def apply_units(current, update, type_parameters, types):
    return current + update

def serialize_units(value, type_parameters, types):
    return str(value)

def deserialize_units(encoded, type_parameters, types):
    return units(encoded)

def divide_units(value, type_parameters, types):
    return [value, value]


# TODO: implement edge handling
def apply_edge(current, update, type_parameters, types):
    return current + update

def serialize_edge(value, type_parameters, types):
    return str(value)

def deserialize_edge(encoded, type_parameters, types):
    return encoded

def divide_edge(value, type_parameters, types):
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
                '_serialize': 'serialize_units',
                '_deserialize': 'deserialize_units',
                '_divide': 'divide_units',
                '_description': 'type to represent values with scientific units'})


base_type_library = {
    # abstract number type
    'number': {
        '_apply': 'accumulate',
        '_serialize': 'to_string',
        '_description': 'abstract base type for numbers'},

    'int': {
        '_default': '0',
        # inherit _apply and _serialize from number type
        '_deserialize': 'int',
        '_divide': 'divide_int',
        '_description': '64-bit integer',
        '_super': 'number',},

    'float': {
        '_default': '0.0',
        '_deserialize': 'float',
        '_divide': 'divide_float',
        '_description': '64-bit floating point precision number',
        '_super': 'number',}, 

    'string': {
        '_default': '""',
        '_apply': 'replace',
        '_serialize': 'serialize_string',
        '_deserialize': 'deserialize_string',
        '_divide': 'divide_int',
        '_description': '64-bit integer'},

    'list': {
        '_default': '[]',
        '_apply': 'concatenate',
        '_serialize': 'to_string',
        '_deserialize': 'evaluate',
        '_divide': 'divide_list',
        '_type_parameters': ['element'],
        '_description': 'general list type (or sublists)'},

    'tree': {
        '_default': '{}',
        '_apply': 'apply_tree',
        '_serialize': 'serialize_tree',
        '_deserialize': 'deserialize_tree',
        '_divide': 'divide_tree',
        '_type_parameters': ['leaf'],
        '_description': 'mapping from str to some type (or nested dicts)'},

    'dict': {
        '_default': '{}',
        '_apply': 'apply_dict',
        '_serialize': 'serialize_dict',
        '_deserialize': 'deserialize_dict',
        '_divide': 'divide_dict',
        '_type_parameters': ['key', 'value'],
        '_description': 'mapping from keys of any type to values of any type'},

    'maybe': {
        '_default': 'None',
        '_apply': 'apply_maybe',
        '_serialize': 'serialize_maybe',
        '_deserialize': 'deserialize_maybe',
        '_divide': 'divide_maybe',
        '_type_parameters': ['value'],
        '_description': 'type to represent values that could be empty'},

    'edge': {
        '_default': '{"wires": {}}',
        '_apply': 'apply_edge',
        '_serialize': 'serialize_edge',
        '_deserialize': 'deserialize_edge',
        '_divide': 'divide_edge',
        '_type_parameters': ['ports'],
        '_description': 'hyperedges in the bigraph, with ports as a type parameter',
        'wires': {
            '_type': 'tree[list[string]]'}}}


def generate_base_types():
    types = SchemaTypes()

    # validate the function registered is of the right type?
    types.apply_registry.register('accumulate', accumulate)
    types.apply_registry.register('concatenate', concatenate)
    types.apply_registry.register('replace', replace)
    types.apply_registry.register('apply_tree', apply_tree)
    types.apply_registry.register('apply_dict', apply_dict)
    types.apply_registry.register('apply_maybe', apply_maybe)
    types.apply_registry.register('apply_units', apply_units)
    types.apply_registry.register('apply_edge', apply_edge)

    types.divide_registry.register('divide_float', divide_float)
    types.divide_registry.register('divide_int', divide_int)
    types.divide_registry.register('divide_longest', divide_longest)
    types.divide_registry.register('divide_list', divide_list)
    types.divide_registry.register('divide_tree', divide_tree)
    types.divide_registry.register('divide_dict', divide_dict)
    types.divide_registry.register('divide_maybe', divide_maybe)
    types.divide_registry.register('divide_units', divide_units)
    types.divide_registry.register('divide_edge', divide_edge)

    types.serialize_registry.register('serialize_string', serialize_string)
    types.serialize_registry.register('to_string', to_string)
    types.serialize_registry.register('serialize_tree', serialize_tree)
    types.serialize_registry.register('serialize_dict', serialize_dict)
    types.serialize_registry.register('serialize_maybe', serialize_maybe)
    types.serialize_registry.register('serialize_units', serialize_units)
    types.serialize_registry.register('serialize_edge', serialize_edge)

    types.deserialize_registry.register('float', deserialize_float)
    types.deserialize_registry.register('int', deserialize_int)
    types.deserialize_registry.register('deserialize_string', deserialize_string)
    types.deserialize_registry.register('evaluate', evaluate)
    types.deserialize_registry.register('deserialize_tree', deserialize_tree)
    types.deserialize_registry.register('deserialize_dict', deserialize_dict)
    types.deserialize_registry.register('deserialize_maybe', deserialize_maybe)
    types.deserialize_registry.register('deserialize_units', deserialize_units)
    types.deserialize_registry.register('deserialize_edge', deserialize_edge)

    types.type_registry.register_multiple(base_type_library)
    register_units(types, units)

    return types


def schema_zoo():
    mitochondria_schema = {
        'mitochondria': {
            'volume': {'_type': 'float'},
            'membrane': {
                'surface_proteins': {'_type': 'tree[protein]'},
                'potential': {'_type': 'microvolts'}},
            'mass': {'_type': 'membrane?'},
        }
    }

    cytoplasm_schema = {
        'cytoplasm': {
            'mitochondria': {'_type': 'tree[mitochondria]'},
            'proteins': {'_type': 'tree[mitochondria]'},
            'nucleus': {'_type': 'tree[mitochondria]'},
            'transcripts': {'_type': 'tree[mitochondria]'},
        }
    }

    cell_schema = {
        'cell': {
            'shape': {'_type': 'mesh'},
            'volume': {'_type': 'mL'},
            'temperature': {'_type': 'K'},
        }
    }

    cell_composite = {
        'environment': {
            'outer_shape': {
                '_type': 'mesh', '_value': []},
            'cellA': {
                'cytoplasm': {
                    'external_ions': {'_type': 'ions'},
                    'internal_ions': {'_type': 'ions'},
                    'other_ions': {'_type': {
                        '_default': 0.0,
                        '_apply': accumulate,
                        '_serialize': str,
                        '_deserialize': float,
                        '_divide': divide_float,
                        '_description': '64-bit floating point precision number'
                    }},
                    'electron_transport': {
                        '_type': 'process',
                        '_value': 'ElectronTransport',
                        '_ports': {
                            'external_ions': 'ions',
                            'internal_ions': 'ions'},
                        '_wires': {
                            'external_ions': ['..', 'external_ions'],
                            'internal_ions': ['..', 'internal_ions']}
                        }
                    },
                'inner_shape': {'_type': 'mesh', '_value': []},
                '_ports': {
                    'shape': 'mesh',
                    'volume': 'mL',
                    'temperature': 'K'
                },
                '_channel': {
                    'shape': ['inner_shape'],
                },
                '_wires': {
                    'shape': ['..', 'outer_shape']
                }
            }
        }
    }

    compose({
        'cell': {
            'membrane': cell_schema,
            'cytoplasm': cytoplasm_schema
        }
    }, {
        
    })


types = generate_base_types()


def test_cube():
    cube_schema = {
        'shape': {},
        
        'rectangle': {
            'width': {'_type': 'int'},
            'height': {'_type': 'int'},
            '_divide': 'divide_longest',
            '_description': 'a two-dimensional value',
            '_super': 'shape',
        },
        
        # cannot override existing keys unless it is of a subtype
        'cube': {
            'depth': {'_type': 'int'},
            '_super': 'rectangle',
        },
    }

    types.type_registry.register_multiple(
        cube_schema)


def test_generate_default():
    int_default = types.generate_default(
        {'_type': 'int'}
    )
    assert int_default == 0

    cube_default = types.generate_default(
        {'_type': 'cube'})

    assert 'width' in cube_default
    assert 'height' in cube_default
    assert 'depth' in cube_default


def test_expand_schema():
    schema = {'_type': 'cube'}
    expanded = types.type_registry.expand(schema)

    assert len(schema) == 1
    assert 'height' in expanded


def test_apply_update():
    schema = {'_type': 'cube'}
    state = {
        'width': 11,
        'height': 13,
        'depth': 44,
    }
    update = {
        'depth': -5
    }

    new_state = types.apply(
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


def test_validate_schema():
    # good schemas
    print_schema_validation(types, base_type_library, True)

    good = {
        'not quite int': {
            '_default': 0,
            '_apply': 'accumulate',
            '_serialize': 'to_string',
            '_deserialize': 'int',
            '_description': '64-bit integer'
        },
        'ports match': {
            'a': {
                '_type': 'int',
                '_value': 2
            },
            'edge1': {
                '_type': 'edge',
                # '_type': 'edge[a:int]',
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

    print_schema_validation(types, good, True)
    print_schema_validation(types, bad, False)


def test_fill_int():
    test_schema = {
        '_type': 'int'
    }

    full_state = types.fill(test_schema)

    assert full_state == 0


def test_fill_cube():
    test_schema = {
        '_type': 'cube'
    }

    partial_state = {
        'height': 5,
    }

    full_state = types.fill(
        test_schema,
        state=partial_state)

    assert 'width' in full_state
    assert 'height' in full_state
    assert 'depth' in full_state
    assert full_state['height'] == 5
    assert full_state['depth'] == 0


def test_fill_in_missing_nodes():
    test_schema = {
        'edge 1': {
            # this could become a process_edge type
            '_type': 'edge',
            '_ports': {
                'port A': {'_type': 'float'},
            },
        }
    }

    test_state = {
        'edge 1': {
            'wires': {
                'port A': ['a'],
            }
        }
    }

    filled = types.fill(
        test_schema,
        test_state
    )

    assert filled == {
        'a': 0.0,
        'edge 1': {
            'wires': {
                'port A': ['a']
            }
        }
    }

def test_fill_in_disconnected_port():
    test_schema = {
        'edge1': {
            '_type': 'edge',
            '_ports': {
                '1': {'_type': 'float'},
            },
        }
    }

    test_state = {}


def test_fill_type_mismatch():
    test_schema = {
        'a': {'_type': 'int', '_value': 2},
        'edge1': {
            '_type': 'edge',
            '_ports': {
                '1': {'_type': 'float'},
                '2': {'_type': 'float'}
            },
            'wires': {
                '1': ['..', 'a'],
                '2': ['a'],
            },
            'a': 5
        },
    }


def test_edge_type_mismatch():
    test_schema = {
        'edge1': {
            '_type': 'edge',
            '_ports': {
                '1': {'_type': 'float'},
            },
            'wires': {
                '1': ['..', 'a']
            },
        },
        'edge2': {
            '_type': 'edge',
            '_ports': {
                '1': {'_type': 'int'},
            },
            'wires': {
                '1': ['..', 'a']
            },
        },
    }


def test_fill_nested_store():
    test_schema = {
        'edge1': {
            '_type': 'edge',
            '_ports': {
                '1': {'_type': 'float'},
            },
            'wires': {
                '1': ['somewhere', 'down', 'this', 'path']
            },
        },
    }    


def test_establish_path():
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


def test_expected_schema():
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
        'process1': {
            '_ports': {
                'port1': 'float',
                'port2': 'int',
            },
        },
        'process2': {
            '_ports': {
                'port1': 'float',
                'port2': 'int',
            },
        },
    }    

    types.type_registry.register(
        'dual_process',
        dual_process_schema,
    )

    test_schema = {
        'store1': 'dual_process',
        'process3': {
            '_ports': {
                'port1': 'dual_process'
            }
        }
    }

    test_state = {
        'store1': {
            'process1': {
                'wires': {
                    'port1': ['store1.1'],
                    'port2': ['store1.2'],
                }
            },
            'process2': {
                'wires': {
                    'port1': ['store1.1'],
                    'port2': ['store1.2'],
                }
            }
        },
        'process3': {
            'wires': {
                'port1': ['store1'],
            }
        },
    }
    
    outcome = types.fill(test_schema, test_state)

    assert outcome == {
        'process3': {
            'wires': {
                'port1': ['store1']
            }
        },
        'store1': {
            'process1': {
                'wires': {
                    'port1': ['store1.1'],
                    'port2': ['store1.2']
                }
            },
            'process2': {
                'wires': {
                    'port1': ['store1.1'],
                    'port2': ['store1.2']
                }
            },
            'store1.1': 0.0,
            'store1.2': 0
        }
    }


def test_link_place():
    bigraph = {
        'nodes': {
            'v0': {
                '_type': 'int',
                '_value': 0},
            'v1': {
                '_type': 'int',
                '_value': 1},
            'v2': {
                '_type': 'int',
                '_value': 2},
            'v3': {
                '_type': 'int',
                '_value': 3},
            'v4': {
                '_type': 'int',
                '_value': 4},
            'v5': {
                '_type': 'int',
                '_value': 5},
            'e0': {
                '_type': 'edge[e0-0:int|e0-1:int|e0-2:int]',
                'wires': {
                    'e0-0': 'v0',
                    'e0-1': 'v1',
                    'e0-2': 'v4'}},
            'e1': {
                '_type': 'edge[e1-0:int|e2-0:int]',
                'wires': {
                    'e1-0': 'v3',
                    'e1-1': 'v1'}},
            'e2': {
                '_type': 'edge[e2-0:int|e2-1:int|e2-2:int]',
                'wires': {
                    'e2-0': 'v3',
                    'e2-1': 'v4',
                    'e2-2': 'v5'}}},
        'place': {
            'v0': {
                'v1': {},
                'v2': {
                    'v3': {}}},
            'v4': {
                'v5': {}},
            'e0': {},
            'e1': {},
            'e2': {}},

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
                'e2-2': 'v5'}}}

    placegraph = {
        'v0': {
            'v1': {},
            'v2': {
                'v3': {}}},
        'v4': {
            'v5': {}},
        'e0': {},
        'e1': {},
        'e2': {}}

    hypergraph = {
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

    result = types.link_place(placegraph, hypergraph)
    # assert result == merged


def test_units():
    schema_length = {
        'distance': {'_type': 'length'}}

    state = {'distance': 11 * units.meter}
    update = {'distance': -5 * units.feet}

    new_state = types.apply(
        schema_length,
        state,
        update
    )

    assert new_state['distance'] == 9.476 * units.meter


if __name__ == '__main__':
    test_cube()
    test_generate_default()
    test_expand_schema()
    test_apply_update()
    test_validate_schema()
    test_fill_int()
    test_fill_cube()
    test_establish_path()
    test_fill_in_missing_nodes()
    test_expected_schema()
    test_units()

