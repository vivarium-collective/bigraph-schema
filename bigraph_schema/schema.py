import pprint
from bigraph_schema.registry import registry_registry, type_schema_keys, optional_schema_keys, type_library


type_registry = registry_registry.access('_type')


def validate_schema(schema, enforce_connections=False):
    # add ports and wires
    # validate ports are wired to a matching type,
    #   either the same type or a subtype (more specific type)
    # declared ports are equivalent to the presence of a process
    #   where the ports need to be looked up
    # if _wires key, must have _ports key (later we will look up
    #   processes by key and find their ports)

    if not isinstance(schema, dict):
        return f'schema is not a dict: {schema}'

    # must be a dict
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


def fill_schema(schema, top=None, path=(), type_key=None, context=None):
    # if a port is disconnected, build a store
    # for it under the '_open' key in the current
    # node

    # inform the user that they have disconnected
    # ports somehow

    if type_key is None:
        if '_type' in schema:
            type_key = schema['_type']
        else:
            raise Exception(
                f'no _type known or inferred at path {path} for {schema}'
            )

    if context is None:
        context = type_registry.access(type_key)

    result = {
        '_type': type_key
    }

    if top is None:
        top = result

    for key, subcontext in context.items():
        if key not in schema:
            raise Exception(
                f'branch of type {type_key} not present in schema {schema}'
            )
        if key not in type_schema_keys:
            result[key] = fill_schema(schema[key])

    return result


# def validate_edges(state, schema, enforce_connections=False):
#     for key, subschema in schema.items():
        


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
                # this could become a process_edge type
                '_type': 'edge',
                '_ports': {
                    '1': {'_type': 'int'},
                    # '2': {'_type': 'float'}
                },
                # 'process': {
                #     '_type': 'process instance',
                #     '_value': 'process:location/somewhere',
                # },
                # 'config': {
                #     '_type': 'hash[any]',
                #     '_value': {},
                # },
                # 'wires': {
                #     '_type': 'hash[list[string]]',
                #     '_value': {
                #         '1': ['..', 'a'],
                #         # '2': ['..', 'b']
                #     }
                # }
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

    import ipdb; ipdb.set_trace()


def test_fill_in_missing_nodes():
    test_schema = {
        # 'a': {'_type': 'int', '_value': 2},
        'edge1': {
            # this could become a process_edge type
            '_type': 'edge',
            '_ports': {
                '1': {'_type': 'float'},
                # '2': {'_type': 'float'}
            },
            # 'process': 'process:location/somewhere',
            # 'config': {},
            'wires': {
                '1': ['..', 'a'],
                # '2': ['..', 'b']
            },
            # 'process': {
            #     '_type': 'process instance',
            #     '_value': 'process:location/somewhere',
            # },
            # 'config': {
            #     '_type': 'dict',
            #     '_value': {},
            # },
        }
    }

    filled = fill_schema(test_schema)
    unenforced_report = validate_schema(filled)
    enforced_report = validate_schema(
        filled, enforce_connections=True)

    assert not unenforced_report
    assert enforced_report
    assert enforced_report['a']


def test_fill_in_disconnected_port():
    test_schema = {
        # 'a': {'_type': 'int', '_value': 2},
        'edge1': {
            '_type': 'edge',
            '_ports': {
                '1': {'_type': 'float'},
                # '2': {'_type': 'float'}
            },
        }
    }

def test_fill_type_mismatch():
    test_schema = {
        'a': {'_type': 'int', '_value': 2},
        'edge1': {
            '_type': 'edge',
            '_ports': {
                '1': {'_type': 'float'},
                # '2': {'_type': 'float'}
            },
            'wires': {
                '1': ['..', 'a']
            },
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
        # 'a': {'_type': 'int', '_value': 2},
    test_schema = {
        'edge1': {
            '_type': 'edge',
            '_ports': {
                '1': {'_type': 'float'},
                # '2': {'_type': 'float'}
            },
            'wires': {
                '1': ['somewhere', 'down', 'this', 'path']
            },
        },
    }    



if __name__ == '__main__':
    test_validate_schema()
