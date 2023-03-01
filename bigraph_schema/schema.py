import pprint
from bigraph_schema.registry import registry_registry, type_schema_keys, type_library # , optional_schema_keys

type_registry = registry_registry.access('_type')

def validate_schema(schema):
    if not isinstance(schema, dict):
        return f'schema is not a dict: {schema}'

    # must be a dict
    report = {}

    schema_keys = set([])
    branches = set([])

    for key, value in schema.items():
        if key == '_type':
            type = type_registry.access(value)
            if type is None:
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

    if len(schema_keys) > 0 and len(branches) == 0:
        undeclared = set(type_schema_keys) - schema_keys
        if len(undeclared) > 0:
            for key in undeclared:
                report[key] = f'missing required key: {key} for declaring atomic type'

    return report


def compose_schemas(schemas, wires):
    return {} # new schema


def query(schema, redex):
    subschema = {}
    return subschema


def react(schema, redex, reactum):
    return {}





def print_schema_validation(library):
    for key, declaration in library.items():
        report = validate_schema(declaration)
        if len(report) == 0:
            print(f'valid schema: {key}')
            pprint.pprint(declaration)
            print('')
        else:
            print(f'INVALID SCHEMA: {key}')
            pprint.pprint(declaration)
            print('')
            print('validation report:')
            pprint.pprint(report)
            print('')


def test_validate_schema():
    # good schemas
    print_schema_validation(type_library)

    # bad schemas
    bad = {
        'empty': None,
        'str?': 'not a schema',
        'not quite int': {
            '_default': 0,
            '_apply': 'accumulate',
            '_serialize': 'str',
            '_deserialize': 'int',
            '_description': '64-bit integer'
        },
        'branch is weird': {
            'left': {'_type': 'ogre'},
            'right': {'_default': 1, '_apply': 'accumulate'},
        }
    }

    print_schema_validation(bad)


if __name__ == '__main__':
    test_validate_schema()
