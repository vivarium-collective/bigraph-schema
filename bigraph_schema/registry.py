"""
========
Registry
========
"""

import inspect
import copy
import collections
import pytest
import traceback

from bigraph_schema.parse import parse_expression


NONE_SYMBOL = ''

required_schema_keys = (
    '_default',
    '_apply',
    '_serialize',
    '_deserialize',
)

optional_schema_keys = (
    '_type',
    '_value',
    '_divide',
    '_description',
    '_ports',
    '_type_parameters',
    '_super',
)

type_schema_keys = required_schema_keys + optional_schema_keys

overridable_schema_keys = (
    '_type',
    '_default',
    '_apply',
    '_serialize',
    '_deserialize',
    '_value',
    '_divide',
    '_description',
)

merge_schema_keys = (
    '_ports',
    '_type_parameters',
)

# check to see where are not adding in supertypes of types
# already present
concatenate_schema_keys = (
    '_super',
)


def non_schema_keys(schema):
    return [
        element
        for element in schema.keys()
        if not element.startswith('_')]

            
def type_merge(dct, merge_dct, path=tuple(), merge_supers=True):
    """Recursively merge type definitions, never overwrite.
    Args:
        dct: The dictionary to merge into. This dictionary is mutated
            and ends up being the merged dictionary.  If you want to
            keep dct you could call it like
            ``deep_merge_check(copy.deepcopy(dct), merge_dct)``.
        merge_dct: The dictionary to merge into ``dct``.
        path: If the ``dct`` is nested within a larger dictionary, the
            path to ``dct``. This is normally an empty tuple (the
            default) for the end user but is used for recursive calls.
    Returns:
        ``dct``
    """
    for k in merge_dct:
        if not k in dct or k in overridable_schema_keys:
            dct[k] = merge_dct[k]
        elif k in merge_schema_keys or isinstance(
            dct[k], dict
        ) and isinstance(
            merge_dct[k], collections.abc.Mapping
        ):
            type_merge(dct[k], merge_dct[k], path + (k,))
        elif k in concatenate_schema_keys:
            # this check may not be necessary if we check
            # for merging super types
            if k != '_super' or merge_supers:
                dct[k].extend(merge_dct[k])
        else:
            raise ValueError(
                f'cannot merge types at path {path + (k,)}:\n'
                f'{dct}\noverwrites \'{k}\' from\n{merge_dct}')
            
    return dct


def deep_merge(dct, merge_dct):
    """ Recursive dict merge
    This mutates dct - the contents of merge_dct are added to dct (which is also returned).
    If you want to keep dct you could call it like deep_merge(copy.deepcopy(dct), merge_dct)
    """
    if dct is None:
        dct = {}
    if merge_dct is None:
        merge_dct = {}
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.abc.Mapping)):
            deep_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]
    return dct


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


def set_path(tree, path, value, top=None, cursor=None):
    if value is None:
        return None
    final = path[-1]
    towards = path[:-1]
    destination = establish_path(tree, towards)
    destination[final] = value
    return tree


def remove_path(tree, path):
    if path is None or len(path) == 0:
        return None

    upon = get_path(tree, path[:-1])
    if upon is not None:
        del upon[path[-1]]
    return tree


class Registry(object):
    """A Registry holds a collection of functions or objects."""

    def __init__(self, function_keys=None):
        function_keys = function_keys or []
        self.registry = {}
        self.main_keys = set([])
        self.function_keys = set(function_keys)

    def register(self, key, item, alternate_keys=tuple(), force=False):
        """Add an item to the registry.

        Args:
            key: Item key.
            item: The item to add.
            alternate_keys: Additional keys under which to register the
                item. These keys will not be included in the list
                returned by ``Registry.list()``.

                This may be useful if you want to be able to look up an
                item in the registry under multiple keys.
            force (bool): Force the registration, overriding existing keys. False by default.
        """

        # check that registered function have the required function keys
        if callable(item) and self.function_keys:
            sig = inspect.signature(item)
            sig_keys = set(sig.parameters.keys())
            assert all(
                key in self.function_keys for key in sig_keys), f"Function '{item.__name__}' keys {sig_keys} are not all " \
                                                                f"in the function_keys {self.function_keys}"

        keys = [key]
        keys.extend(alternate_keys)
        for registry_key in keys:
            if registry_key in self.registry and not force:
                if item != self.registry[registry_key]:
                    raise Exception(
                        'registry already contains an entry for {}: {} --> {}'.format(
                            registry_key, self.registry[key], item))
            else:
                self.registry[registry_key] = item
        self.main_keys.add(key)

    def register_multiple(self, schemas, force=False):
        for key, schema in schemas.items():
            self.register(key, schema, force=force)

    def access(self, key):
        """Get an item by key from the registry."""
        return self.registry.get(key)

    def list(self):
        return list(self.main_keys)

    def validate(self, item):
        return True


class TypeRegistry(Registry):
    """Type Registry

    Holds type schema in one object for easy access
    """
    def __init__(self):
        super().__init__()

        self.supers = {}
        self.register('any', {})

    def register(self, key, schema, alternate_keys=tuple(), force=False):
        schema = copy.deepcopy(schema)
        if isinstance(schema, dict):
            supers = schema.get('_super', ['any'])  # list of immediate supers
            if isinstance(supers, str):
                supers = [supers]
                schema['_super'] = supers
            for su in supers:
                assert isinstance(
                    su, str), f"super for {key} must be a string, not {su}"
            self.supers[key] = supers

            for su in supers:
                su_type = self.registry.get(su, {})
                new_schema = copy.deepcopy(su_type)
                schema = type_merge(
                    new_schema,
                    schema,
                    merge_supers=False)

            for subkey, original_subschema in schema.items():
                if not subkey in type_schema_keys:
                    subschema = self.access(original_subschema)
                    if subschema is None:
                        raise Exception(f'trying to register a new type ({key}), '
                                        f'but it depends on a type ({subkey}) which is not in the registry')
                    else:
                        schema[subkey] = subschema
        else:
            raise Exception(f'all type definitions must be dicts '
                            f'with the following keys: {type_schema_keys}\nnot: {schema}')

        super().register(key, schema, alternate_keys, force)

    def resolve_parameters(self, type_parameters, schema):
        return {
            type_parameter: self.access(
                schema.get(f'_{type_parameter}'))
            for type_parameter in type_parameters}

    def access(self, schema):
        """Retrieve all types in the schema"""

        found = None

        if isinstance(schema, dict):
            if '_default' in schema:
                return schema
            elif '_type' in schema:
                found = self.access(schema['_type'])
                if '_type_parameters' in found:
                    found = copy.deepcopy(found)
                    found = deep_merge(found, schema)

                    for type_parameter in found['_type_parameters']:
                        parameter_key = f'_{type_parameter}'
                        if parameter_key in found:
                            if not '_bindings' in found:
                                found['_bindings'] = {}
                            found['_bindings'][parameter_key] = found[parameter_key]
                        elif '_bindings' in found and type_parameter in found['_bindings']:
                            found[parameter_key] = found['_bindings'][parameter_key]
            else:
                found = {
                   key: self.access(branch)
                   for key, branch in schema.items()}

        elif isinstance(schema, list):
            bindings = []
            if len(schema) > 1:
                schema, bindings = schema
            else:
                schema = schema[0]
            found = self.access(schema)

            if len(bindings) > 0:
                found = found.copy()
                found['_bindings'] = dict(zip(
                    found['_type_parameters'],
                    bindings))

                for type_parameter, binding in found['_bindings'].items():
                    found[f'_{type_parameter}'] = self.access(binding)

        elif isinstance(schema, str):
            found = self.registry.get(schema)

            if found is None and schema is not None and schema != '':
                try:
                    parse = parse_expression(schema)
                    if parse != schema:
                        found = self.access(parse)
                except Exception:
                    print(f'type did not parse: {schema}')
                    traceback.print_exc()
                    
        return found

    def lookup(self, type_key, attribute):
        return self.access(type_key).get(attribute)

    # description should come from type
    def is_descendant(self, key, ancestor):
        for sup in self.supers.get(key, []):
            if sup == ancestor:
                return True
            else:
                found = self.is_descendant(sup, ancestor)
                if found:
                    return True
        return False


class RegistryRegistry(Registry):
    def type_attribute(self, type_key, attribute):
        type_registry = self.access('_type')
        type_value = type_registry.access(type_key)
        attribute_key = type_value.get(attribute)
        if attribute_key is not None:
            attribute_registry = self.access(attribute)
            return attribute_registry.access(attribute_key)


def test_reregister_type():
    type_registry = TypeRegistry()
    type_registry.register('A', {'_default': 'a'})
    with pytest.raises(Exception) as e:
        type_registry.register('A', {'_default': 'b'})

    type_registry.register('A', {'_default': 'b'}, force=True)


if __name__ == '__main__':
    test_reregister_type()
