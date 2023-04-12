import copy
import pprint
from bigraph_schema.registry import registry_registry, type_schema_keys, optional_schema_keys, type_library, deep_merge


type_registry = registry_registry.access('_type')
serialize_registry = registry_registry.access('_serialize')
deserialize_registry = registry_registry.access('_deserialize')


def validate_schema(schema, enforce_connections=False):
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
            typ = type_registry.access(value)
            if typ is None:
                report[key] = f'type: {value} is not in the registry'
        elif key in type_schema_keys:
            schema_keys.add(key)
            registry = registry_registry.access(key)
            if registry is None:
                # deserialize and serialize back and check it is equal
                pass
            else:
                element = registry.access(value)
                if element is None:
                    report[key] = f'no entry in the {key} registry for: {value}'
        else:
            branches.add(key)
            branch_report = validate_schema(value)
            if len(branch_report) > 0:
                report[key] = branch_report

    # # We will need this when building instances to check to see if we are
    # # trying to instantiate an abstract type, but we can still register
    # # register abstract types so it is not invalid
    # if len(schema_keys) > 0 and len(branches) == 0:
    #     undeclared = set(type_schema_keys) - schema_keys
    #     if len(undeclared) > 0:
    #         for key in undeclared:
    #             if not key in optional_schema_keys:
    #                 report[key] = f'missing required key: {key} for declaring atomic type'

    return report


def validate_instance(schema, instance):
    schema = type_registry.substitute_type(schema)
    validation = {}

    if '_serialize' in schema:
        if '_deserialize' not in schema:
            validation = {
                '_deserialize': f'serialize found in type without deserialize: {schema}'
            }
        else:
            serialize = serialize_registry.access(schema['_serialize'])
            deserialize = deserialize_registry.access(schema['_deserialize'])
            serial = serialize(instance)
            pass_through = deserialize(serial)

            if instance != pass_through:
                validation = f'instance and pass_through are not the same: {serial}'
    else:
        for key, subschema in schema.items():
            if key not in type_schema_keys:
                if key not in instance:
                    validation[key] = f'key present in schema but not in instance: {key}\nschema: {schema}\ninstance: {instance}\n'
                else:
                    subvalidation = validate_instance(subschema, instance[key])
                    if not (subvalidation is None or len(subvalidation) == 0):
                        validation[key] = subvalidation

    return validation


def get_path(tree, path):
    if len(path) == 0:
        return tree
    else:
        head = path[0]
        if head not in tree:
            return None
        else:
            return get_path(tree[head], path[1:])


def establish_path(tree, path, top=None, cursor=()):
    if top is None:
        top = tree
    if path is None or path == ():
        return tree
    elif len(path) == 0:
        return tree
    else:
        if isinstance(path, str):
            path = (path,)

        head = path[0]
        if head == '..':
            if cursor == ():
                raise Exception(
                    f'trying to travel above the top of the tree: {path}')
            else:
                return establish_path(
                    top,
                    cursor[:-1])
        else:
            if head not in tree:
                tree[head] = {}
            return establish_path(
                tree[head],
                path[1:],
                top=top,
                cursor=cursor + (head,))


def fill_ports(schema, wires=None, instance=None, top=None, path=()):
    # deal with wires
    if wires is None:
        wires = {}
    if instance is None:
        instance = {}
    if top is None:
        top = instance

    more_wires = instance.get('wires', {})
    wires = deep_merge(wires, more_wires)

    for port_key, port_schema in schema.items():
        if port_key in wires:
            subwires = wires[port_key]
            if isinstance(subwires, dict):
                instance[port_key] = fill_ports(
                    port_schema,
                    wires=subwires,
                    instance=instance.get(port_key),
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
                    path[:-1]
                )

                destination = establish_path(
                    peer,
                    subwires[:-1],
                    top=top,
                    cursor=path[:-1])

                destination_key = subwires[-1]

                if destination_key in destination:
                    pass
                    # validate_instance(
                    #     port_schema,
                    #     destination[destination_key])
                else:
                    destination[destination_key] = type_registry.generate_default(
                        port_schema)
        else:
            # handle unconnected ports
            pass

    return instance


def fill_instance(schema, instance=None, top=None, path=(), type_key=None, context=None):
    # if a port is disconnected, build a store
    # for it under the '_open' key in the current
    # node

    # inform the user that they have disconnected
    # ports somehow

    if top is None:
        top = instance

    schema = type_registry.substitute_type(schema)

    if instance is None:
        if '_default' in schema:
            instance = type_registry.generate_default(schema)
        else:
            instance = {}

    if isinstance(schema, str):
        raise Exception(
            f'schema cannot be a str: {str}'
        )

    for key, subschema in schema.items():
        if key == '_ports':
            wires = instance.get('wires', {})
            instance = fill_ports(
                subschema,
                wires=wires,
                instance=instance,
                top=top,
                path=path)

        elif key not in type_schema_keys:
            subpath = path + (key,)
            if isinstance(instance, dict):
                instance[key] = fill_instance(
                    subschema,
                    instance=instance.get(key),
                    top=top,
                    path=subpath)
        
    return instance


def fill(schema, instance=None):
    if instance is not None:
        instance = copy.deepcopy(instance)
    return fill_instance(schema, instance=instance)


def merge(a, b):
    pass


def compose(a, b):
    pass


# maybe vivarium?
def hydrate(schema):
    return {}
    

def dehydrate(schema):
    return {}


def query(schema, redex):
    subschema = {}
    return subschema


def react(schema, redex, reactum):
    return {}


def print_schema_validation(library, should_pass):
    for key, declaration in library.items():
        report = validate_schema(declaration)
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
    print_schema_validation(type_library, True)

    good = {
        'not quite int': {
            '_default': 0,
            '_apply': 'accumulate',
            '_serialize': 'str',
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
                '_ports': {
                    '1': {'_type': 'int'},
                },
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

    print_schema_validation(good, True)
    print_schema_validation(bad, False)


def test_fill_int():
    test_schema = {
        '_type': 'int'
    }

    full_instance = fill(test_schema)

    assert full_instance == 0


def test_fill_cube():
    test_schema = {
        '_type': 'cube'
    }

    partial_instance = {
        'height': 5,
    }

    full_instance = fill(
        test_schema,
        instance=partial_instance)

    assert 'width' in full_instance
    assert 'height' in full_instance
    assert 'depth' in full_instance
    assert full_instance['height'] == 5
    assert full_instance['depth'] == 0


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

    test_instance = {
        'edge 1': {
            'wires': {
                'port A': 'a',
            }
        }
    }

    filled = fill(
        test_schema,
        test_instance
    )

    assert filled == {
        'a': 0.0,
        'edge 1': {
            'wires': {
                'port A': 'a'
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

    test_instance = {}

    import ipdb; ipdb.set_trace()


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


def test_expand_schema():
    schema = {'_type': 'cube'}
    expanded = type_registry.expand(schema)

    assert len(schema) == 1
    assert 'height' in expanded


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

    type_registry.register(
        'dual_process',
        dual_process_schema,
    )

    expected_schema = {
        'store1': 'dual_process',
        'process3': {
            '_ports': {
                'port1': 'dual_process'
            }
        }
    }

    expected_instance = {
        'store1': {
            'process1': {
                'wires': {
                    'port1': 'store1.1',
                    'port2': 'store1.2',
                }
            },
            'process2': {
                'wires': {
                    'port1': 'store1.1',
                    'port2': 'store1.2',
                }
            }
        },
        'process3': {
            'wires': {
                'port1': 'store1',
            }
        },
    }
    
    outcome = fill(expected_schema, expected_instance)

    assert outcome == {
        'process3': {
            'wires': {
                'port1': 'store1'
            }
        },
        'store1': {
            'process1': {
                'wires': {
                    'port1': 'store1.1',
                    'port2': 'store1.2'
                }
            },
            'process2': {
                'wires': {
                    'port1': 'store1.1',
                    'port2': 'store1.2'
                }
            },
            'store1.1': 0.0,
            'store1.2': 0
        }
    }




if __name__ == '__main__':
    test_validate_schema()
    test_fill_int()
    test_fill_cube()
    test_establish_path()
    test_fill_in_missing_nodes()
    test_expected_schema()
    test_expand_schema()











