from bigraph_schema.registry import registry_registry, type_schema_keys # , optional_schema_keys


def validate_schema(schema):
    # must be a dict
    report = {}

    all_keys = ['_type'] + type_schema_keys # required_schema_keys + optional_schema_keys

    for key, value in schema.items():
        if key == '_type':
            type = registry_registry.access('type').access(value)
            if type is None:
                report[key] = f'type {value} is not in the registry'
        elif key in type_schema_keys:
            registry = registry_registry.access(key)
            if registry is None:
                # deserialize and serialize back and check it is equal
                pass
            else:
                element = registry.access(value)
                if element is None:
                    report[key] = f'no entry in the {key} registry for {value}'
        else:
            report[key] = validate_schema(value)

    return report


def compose_schemas(schemas, wires):
    return {} # new schema

def query(schema, redex):
    subschema = {}
    return subschema

def react(schema, redex, reactum):
    return {}





def test_validate_schema():
    validate_schema()

if __name__ == '__main__':
    test_validate_schema()
