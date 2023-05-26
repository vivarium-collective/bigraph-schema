import copy
import random
import collections
import pytest
from typing import Any

from bigraph_schema.parse import parse_expression
from bigraph_schema.units import units, render_units_type


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
                f'cannot merge types at path {path + (k,)}: '
                f'{dct} overwrites {k} from {merge_dct}'
            )
            
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


class Registry(object):
    def __init__(self):
        """A Registry holds a collection of functions or objects."""
        self.registry = {}
        self.main_keys = set([])

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
        """
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
    def __init__(self):
        super().__init__()

        self.supers = {}
        self.register('any', {})


    def register(self, key, item, alternate_keys=tuple(), force=False):
        item = copy.deepcopy(item)
        if isinstance(item, dict):
            supers = item.get('_super', ['any']) # list of immediate supers
            if isinstance(supers, str):
                supers = [supers]
                item['_super'] = supers
            for su in supers:
                assert isinstance(
                    su, str), f"super for {key} must be a string, not {su}"
            self.supers[key] = supers
            for su in supers:
                su_type = self.registry.get(su, {})
                new_item = copy.deepcopy(su_type)
                item = type_merge(
                    new_item,
                    item,
                    merge_supers=False)

        super().register(key, item, alternate_keys, force)


    def resolve_parameters(self, qualified_type):
        if isinstance(qualified_type, str):
            type_name = qualified_type
            parameter_types = []
        elif isinstance(qualified_type, dict):
            return {
                key: self.resolve_parameters(branch)
                for key, branch in qualified_type.items()}
        else:
            type_name, parameter_types = qualified_type
        outer_type = self.registry.get(type_name)

        if outer_type is None:
            try:
                outer_type = parse_expression(type_name)
            except:
                import ipdb; ipdb.set_trace()
                raise ValueError(f'type {qualified_type} is looking for type {type_name} but that is not in the registry')

        type_parameters = {}
        if '_type_parameters' in outer_type:
            parameter_names = outer_type['_type_parameters']
            type_parameters = {
                parameter_name: self.resolve_parameters(parameter_type)
                for parameter_name, parameter_type in zip(
                    parameter_names,
                    parameter_types)}

        result = {
            '_type': type_name,
        }

        if type_parameters:
            result['_type_parameters'] = type_parameters

        return result


    def substitute_type(self, schema):
        if isinstance(schema, str):
            schema = self.access(schema)
            if schema is None:
                raise Exception(f'trying to substitute a type that is unrecognized {schema}')
        elif '_type' in schema:
            type_key = schema['_type']
            type_schema = copy.deepcopy(self.access(type_key))
            schema = deep_merge(type_schema, schema)

        return schema


    def expand_schema(self, schema):
        # make this only show the types at the leaves

        step = self.substitute_type(schema)
        for key, subschema in step.items():
            if key not in type_schema_keys:
                step[key] = self.expand_schema(subschema)
        return step
        

    def expand(self, schema):
        duplicate = copy.deepcopy(schema)
        return self.expand_schema(duplicate)


    def access(self, key):
        """Get an item by key from the registry."""
        typ = self.registry.get(key)

        if typ is None and key is not None and key != '':
            try:
                parse = parse_expression(key)
                if parse[0] in self.registry:
                    typ = self.resolve_parameters(parse)
            except Exception as e:
                import ipdb; ipdb.set_trace()
                print(f'type did not parse: {key}')

        return typ


    def lookup(type_key, attribute):
        return self.access(type_key).get(attribute)


    # description should come from type
    def is_descendent(self, key, ancestor):
        for sup in self.supers.get(key, []):
            if sup == ancestor:
                return True
            else:
                found = self.is_descendent(sup, ancestor)
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
