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

from pprint import pformat as pf

from bigraph_schema.parse import parse_expression
from bigraph_schema.protocols import local_lookup_module, function_module


NONE_SYMBOL = '!nil'


required_schema_keys = set([
    '_default',
    '_apply',
    '_check',
    '_serialize',
    '_deserialize',
    '_fold',
])

optional_schema_keys = set([
    '_type',
    '_value',
    '_description',
    '_type_parameters',
    '_inherit',
    '_divide',
])

type_schema_keys = required_schema_keys | optional_schema_keys

TYPE_FUNCTION_KEYS = [
    '_apply',
    '_check',
    '_fold',
    '_divide',
    '_react',
    '_serialize',
    '_deserialize']

overridable_schema_keys = set([
    '_type',
    '_default',
    '_check',
    '_apply',
    '_serialize',
    '_deserialize',
    '_fold',
    '_divide',
    '_type_parameters',
    '_value',
    '_description',
    '_inherit',
])

nonoverridable_schema_keys = type_schema_keys - overridable_schema_keys

merge_schema_keys = (
    '_ports',
    '_type_parameters',
)



def non_schema_keys(schema):
    return [
        element
        for element in schema.keys()
        if not element.startswith('_')]

            
def type_merge(dct, merge_dct, path=tuple(), merge_supers=False):
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
            type_merge(
                dct[k],
                merge_dct[k],
                path + (k,),
                merge_supers)

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


def validate_merge(state, dct, merge_dct):
    """ Recursive dict merge
    This mutates dct - the contents of merge_dct are added to dct (which is also returned).
    If you want to keep dct you could call it like deep_merge(copy.deepcopy(dct), merge_dct)
    """
    dct = dct or {}
    merge_dct = merge_dct or {}
    state = state or {}

    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.abc.Mapping)):
            if k not in state:
                state[k] = {}

            validate_merge(
                state[k],
                dct[k],
                merge_dct[k])
        else:
            if k in state:
                dct[k] = state[k]
            elif k in dct:
                if dct[k] != merge_dct[k]:
                    raise Exception(f'cannot merge dicts at key "{k}":\n{dct}\n{merge_dct}')
            else:
                dct[k] = merge_dct[k]
    return dct


def get_path(tree, path):
    '''
    given a tree and a path, find the subtree at that path
    Args:
        tree: the tree we are looking in (a nested dict)
        path: a list/tuple of keys we follow down the tree
            to find the subtree we are looking for
    Returns:
        subtree: the subtree found by following the list of keys
            down the tree
    '''

    if len(path) == 0:
        return tree
    else:
        head = path[0]
        if not tree or head not in tree:
            return None
        else:
            return get_path(tree[head], path[1:])


def establish_path(tree, path, top=None, cursor=()):
    '''
    given a tree and a path in the tree that may or may not yet exist,
    add nodes along the path and return the final node which is now at the
    given path.
    Args:
        tree: the tree we are establishing a path in
        path: where the new subtree will be located in the tree
        top: (None) a reference to the top of the tree
        cursor: (()) the current location we are visiting in the tree
    Returns:
        node: the new node of the tree that exists at the given path
    '''

    if tree is None:
        tree = {}

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
                cursor=tuple(cursor) + (head,))


def set_path(tree, path, value, top=None, cursor=None):
    '''
    given a tree, a path, and a value, sets the location
    in the tree corresponding to the path to the given value
    Args:
        tree: the tree we are setting a value in
        path: where the new value will be located in the tree
        value: the value to set at the given path in the tree
        top: (None) a reference to the top of the tree
        cursor: (()) the current location we are visiting in the tree
    Returns:
        node: the new node of the tree that exists at the given path
    '''

    if value is None:
        return None
    if len(path) == 0:
        return value

    final = path[-1]
    towards = path[:-1]
    destination = establish_path(tree, towards)
    destination[final] = value
    return tree


def transform_path(tree, path, transform):
    '''
    given a tree, a path, and a transform (function), 
    mutate the tree by replacing the subtree at the path by
    whatever is returned from applying the transform to the
    existing value
    Args:
        tree: the tree we are setting a value in
        path: where the new value will be located in the tree
        transform: the function to apply to whatever currently lives
            at the given path in the tree
    Returns:
        node: the node of the tree that exists at the given path
    '''
    before = establish_path(tree, path)
    after = transform(before)

    return set_path(tree, path, after)


def remove_omitted(before, after, tree):
    '''
    removes anything in tree that was in before but not in after
    '''

    if isinstance(before, dict):
        if not isinstance(tree, dict):
            raise Exception(
                f'trying to remove an entry from something that is not a dict: {tree}')

        if not isinstance(after, dict):
            return after

        for key, down in before.items():
            if not key.startswith('_'):
                if key not in after:
                    if key in tree:
                        del tree[key]
                else:
                    tree[key] = remove_omitted(
                        down,
                        after[key],
                        tree[key])

    return tree


def remove_path(tree, path):
    '''
    removes whatever subtree lives at the given path
    '''

    if path is None or len(path) == 0:
        return None

    upon = get_path(tree, path[:-1])
    if upon is not None:
        del upon[path[-1]]
    return tree


class Registry(object):
    '''A Registry holds a collection of functions or objects'''

    def __init__(self, function_keys=None):
        function_keys = function_keys or []
        self.registry = {}
        self.main_keys = set([])
        self.function_keys = set(function_keys)

    def register(self, key, item, alternate_keys=tuple(), strict=False):
        '''
        Add an item to the registry.

        Args:
            key: Item key.
            item: The item to add.
            alternate_keys: Additional keys under which to register the
                item. These keys will not be included in the list
                returned by ``Registry.list()``.

                This may be useful if you want to be able to look up an
                item in the registry under multiple keys.
            strict (bool): Disallow re-registration, overriding existing keys. False by default.
        '''

        # check that registered function have the required function keys
        if callable(item) and self.function_keys:
            sig = inspect.signature(item)
            sig_keys = set(sig.parameters.keys())
            # assert all(
            #     key in self.function_keys for key in sig_keys), f"Function '{item.__name__}' keys {sig_keys} are not all " \
            #                                                     f"in the function_keys {self.function_keys}"

        keys = [key]
        keys.extend(alternate_keys)
        for registry_key in keys:
            if registry_key in self.registry:
                if item != self.registry[registry_key]:
                    if strict:
                        raise Exception(
                            'registry already contains an entry for {}: {} --> {}'.format(
                                registry_key, self.registry[key], item))

                    else:
                        self.registry[registry_key] = deep_merge(
                            self.registry[registry_key],
                            item)
            else:
                self.registry[registry_key] = item
        self.main_keys.add(key)


    def register_function(self, function):
        if isinstance(function, str):
            module_key = function
            found = self.access(module_key)

            if found is None:
                found = local_lookup_module(
                    module_key)

                if found is None:
                    raise Exception(
                        f'function "{subschema}" not found for type data:\n  {pf(schema)}')

        elif inspect.isfunction(function):
            found = function
            module_key = function_module(found)
        
        function_name = module_key.split('.')[-1]
        self.register(function_name, found)
        self.register(module_key, found)

        return function_name, module_key


    def register_multiple(self, schemas, force=False):
        for key, schema in schemas.items():
            self.register(key, schema, force=force)

    def access(self, key):
        '''
        get an item by key from the registry.
        '''

        return self.registry.get(key)

    def list(self):
        return list(self.main_keys)

    def validate(self, item):
        return True


def visit_method(schema, state, method, values, core):
    schema = core.access(schema)
    method_key = f'_{method}'

    # TODO: we should probably cache all this
    if isinstance(state, dict) and method_key in state:
        visit = core.find_method(
            {method_key: state[method_key]},
            method_key)

    elif method_key in schema:
        visit = core.find_method(
            schema,
            method_key)

    else:
        visit = core.find_method(
            'any',
            method_key)

    result = visit(
        schema,
        state,
        values,
        core)

    return result


def fold_any(schema, state, method, values, core):
    if isinstance(state, dict):
        result = {}
        for key, value in state.items():
            if key.startswith('_'):
                result[key] = value
            else:
                if key in schema:
                    fold = core.fold_state(
                        schema[key],
                        value,
                        method,
                        values)
                    result[key] = fold

    else:
        result = state

    visit = visit_method(
        schema,
        result,
        method,
        values,
        core)

    return visit


def fold_tuple(schema, state, method, values, core):
    if not isinstance(state, (tuple, list)):
        return visit_method(
            schema,
            state,
            method,
            values,
            core)
    else:
        parameters = core.parameters_for(schema)
        result = []
        for parameter, element in zip(parameters, state):
            fold = core.fold(
                parameter,
                element,
                method,
                values)
            result.append(fold)

        result = tuple(result)

        return visit_method(
            schema,
            result,
            method,
            values,
            core)


def fold_union(schema, state, method, values, core):
    union_type = find_union_type(
        core,
        schema,
        state)

    result = core.fold(
        union_type,
        state,
        method,
        values)

    return result


def divide_any(schema, state, values, core):
    divisions = values.get('divisions', 2)

    if isinstance(state, dict):
        result = [
            {}
            for _ in range(divisions)]

        for key, value in state.items():
            for index in range(divisions):
                result[index][key] = value[index]

        return result

    else:
        return [
            copy.deepcopy(state)
            for _ in range(divisions)]


def divide_tuple(schema, state, values, core):
    divisions = values.get('divisions', 2)

    return [
        tuple([item[index] for item in state])
        for index in range(divisions)]


def apply_tree(schema, current, update, core):
    leaf_type = core.find_parameter(
        schema,
        'leaf')

    if current is None:
        current = core.default(leaf_type)

    if isinstance(current, dict) and isinstance(update, dict):
        for key, branch in update.items():
            if key == '_add':
                current.update(branch)
            elif key == '_remove':
                current = remove_path(current, branch)
            elif isinstance(branch, dict):
                subschema = schema
                if key in schema:
                    subschema = schema[key]

                current[key] = core.apply(
                    subschema,
                    current.get(key),
                    branch)

            elif core.check(leaf_type, branch):
                current[key] = core.apply(
                    leaf_type,
                    current.get(key),
                    branch)

            else:
                raise Exception(f'state does not seem to be of leaf type:\n  state: {state}\n  leaf type: {leaf_type}')

        return current

    elif core.check(leaf_type, current):
        return core.apply(
            leaf_type,
            current,
            update)

    else:
        raise Exception(f'trying to apply an update to a tree but the values are not trees or leaves of that tree\ncurrent:\n  {pf(current)}\nupdate:\n  {pf(update)}\nschema:\n  {pf(schema)}')


def apply_any(schema, current, update, core):
    if isinstance(current, dict):
        return apply_tree(
            current,
            update,
            'tree[any]',
            core)
    else:
        return update


def slice_any(schema, state, path, core):
    if not isinstance(path, (list, tuple)):
        if path is None:
            path = ()
        else:
            path = [path]

    if len(path) > 0:
        head = path[0]
        tail = path[1:]
        step = None

        if isinstance(state, dict):
            if head not in state:
                state[head] = {}
            step = state[head]

        elif hasattr(state, head):
            step = getattr(state, head)

        return core.slice(
            schema.get(head, schema),
            step,
            tail)

    else:
        return schema, state


def check_any(schema, state, core):
    if isinstance(schema, dict):
        for key, subschema in schema.items():
            if not key.startswith('_'):
                if isinstance(state, dict):
                    if key in state:
                        check = core.check_state(
                            subschema,
                            state[key])

                        if not check:
                            return False
                else:
                    return False

        return True
    else:
        return True


def serialize_any(schema, state, core):
    if isinstance(state, dict):
        tree = {}

        for key in non_schema_keys(schema):
            encoded = core.serialize(
                schema.get(key, schema),
                state.get(key))
            tree[key] = encoded

        return tree

    else:
        return str(state)


def deserialize_any(schema, state, core):
    if isinstance(state, dict):
        tree = {}

        for key in non_schema_keys(schema):
            decoded = core.deserialize(
                schema.get(key, schema),
                state.get(key))

            tree[key] = decoded

        return tree

    else:
        return state


def apply_tuple(schema, current, update, core):
    parameters = core.parameters_for(schema)
    result = []

    for parameter, current_value, update_value in zip(parameters, current, update):
        element = core.apply(
            parameter,
            current_value,
            update_value)

        result.append(element)

    return tuple(result)


def check_tuple(schema, state, core):
    if not isinstance(state, (tuple, list)):
        return False

    parameters = core.parameters_for(schema)
    for parameter, element in zip(parameters, state):
        if not core.check(parameter, element):
            return False

    return True


def slice_tuple(schema, state, path, core):
    if len(path) > 0:
        head = path[0]
        tail = path[1:]

        if str(head) in schema['_type_parameters']:
            try:
                index = schema['_type_parameters'].index(str(head))
            except:
                raise Exception(f'step {head} in path {path} is not a type parameter of\n  schema: {pf(schema)}\n  state: {pf(state)}')
            index_key = f'_{index}'
            subschema = core.access(schema[index_key])

            return core.slice(subschema, state[head], tail)
        else:
            raise Exception(f'trying to index a tuple with a key that is not an index: {state} {head}')
    else:
        return schema, state


def serialize_tuple(schema, value, core):
    parameters = core.parameters_for(schema)
    result = []

    for parameter, element in zip(parameters, value):
        encoded = core.serialize(
            parameter,
            element)

        result.append(encoded)

    return tuple(result)


def deserialize_tuple(schema, state, core):
    parameters = core.parameters_for(schema)
    result = []

    if isinstance(state, str):
        if (state[0] == '(' and state[-1] == ')') or (state[0] == '[' and state[-1] == ']'):
            state = state[1:-1].split(',')
        else:
            return None

    for parameter, code in zip(parameters, state):
        element = core.deserialize(
            parameter,
            code)

        result.append(element)

    return tuple(result)


def find_union_type(core, schema, state):
    parameters = core.parameters_for(schema)

    for possible in parameters:
        if core.check(possible, state):
            return core.access(possible)

    return None


def apply_union(schema, current, update, core):
    current_type = find_union_type(
        core,
        schema,
        current)

    update_type = find_union_type(
        core,
        schema,
        update)

    if current_type is None:
        raise Exception(f'trying to apply update to union value but cannot find type of value in the union\n  value: {current}\n  update: {update}\n  union: {list(bindings.values())}')
    elif update_type is None:
        raise Exception(f'trying to apply update to union value but cannot find type of update in the union\n  value: {current}\n  update: {update}\n  union: {list(bindings.values())}')

    # TODO: throw an exception if current_type is incompatible with update_type

    return core.apply(
        update_type,
        current,
        update)


def check_union(schema, state, core):
    found = find_union_type(
        core,
        schema,
        state)

    return found is not None and len(found) > 0


def slice_union(schema, state, path, core):
    union_type = find_union_type(
        core,
        schema,
        state)

    return core.slice(
        union_type,
        state,
        path)


def serialize_union(schema, value, core):
    union_type = find_union_type(
        core,
        schema,
        value)

    return core.serialize(
        union_type,
        value)


def deserialize_union(schema, encoded, core):
    if encoded == NONE_SYMBOL:
        return None
    else:
        parameters = core.parameters_for(schema)

        for parameter in parameters:
            value = core.deserialize(
                parameter,
                encoded)


            if value is not None:
                return value


registry_types = {
    'any': {
        '_type': 'any',
        '_slice': slice_any,
        '_apply': apply_any,
        '_check': check_any,
        '_serialize': serialize_any,
        '_deserialize': deserialize_any,
        '_fold': fold_any,
        '_divide': divide_any},

    'tuple': {
        '_type': 'tuple',
        '_default': (),
        '_apply': apply_tuple,
        '_check': check_tuple,
        '_slice': slice_tuple,
        '_serialize': serialize_tuple,
        '_deserialize': deserialize_tuple,
        '_fold': fold_tuple,
        '_divide': divide_tuple,
        '_description': 'tuple of an ordered set of typed values'},

    'union': {
        '_type': 'union',
        '_default': NONE_SYMBOL,
        '_apply': apply_union,
        '_check': check_union,
        '_slice': slice_union,
        '_serialize': serialize_union,
        '_deserialize': deserialize_union,
        '_fold': fold_union,
        '_description': 'union of a set of possible types'}}


def is_method_key(key, parameters):
    return key.startswith('_') and key not in type_schema_keys and key not in [
        f'_{parameter}' for parameter in parameters]


class TypeRegistry(Registry):
    """
    registry for holding type information
    """

    def __init__(self):
        super().__init__()

        # inheritance tracking
        self.inherits = {}

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

        for type_key, type_data in registry_types.items():
            self.register(
                type_key,
                type_data)


    def lookup_registry(self, underscore_key):
        '''
        access the registry for the given key
        '''

        if underscore_key == '_type':
            return self
        root = underscore_key.strip('_')
        registry_key = f'{root}_registry'
        if hasattr(self, registry_key):
            return getattr(self, registry_key)


    def find_registry(self, underscore_key):
        '''
        access the registry for the given key
        and create if it doesn't exist
        '''

        registry = self.lookup_registry(underscore_key)
        if registry is None:
            registry = Registry()
            setattr(
                self,
                f'{underscore_key[1:]}_registry',
                registry)

        return registry


    def register(self, key, schema, alternate_keys=tuple(), force=False):
        '''
        register the schema under the given key in the registry
        '''

        if isinstance(schema, str):
            schema = self.access(schema)
        schema = copy.deepcopy(schema)

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
                schema = type_merge(
                    new_schema,
                    schema)

                self.inherits[key].append(
                    inherit_type)

            for subkey, subschema in schema.items():
                parameters = schema.get('_type_parameters', [])
                if subkey in TYPE_FUNCTION_KEYS or is_method_key(subkey, parameters):
                    registry = self.find_registry(
                        subkey)
                    function_name, module_key = registry.register_function(subschema)

                    schema[subkey] = function_name

                elif subkey not in type_schema_keys:
                    lookup = self.access(subschema)
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

        super().register(key, schema, alternate_keys, force)


    def resolve_parameters(self, type_parameters, schema):
        '''
        find the types associated with any type parameters in the schema
        '''

        return {
            type_parameter: self.access(
                schema.get(f'_{type_parameter}'))
            for type_parameter in type_parameters}


    def access(self, schema):
        '''
        expand the schema to its full type information from the type registry
        '''

        found = None

        if isinstance(schema, dict):
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
                    union_schema)

            elif '_type' in schema:
                registry_type = self.retrieve(schema['_type'])
                found = schema.copy()
                for key, value in registry_type.items():
                    if  key == '_type' or key not in found:
                        found[key] = value

            else:
                found = {
                   key: self.access(branch)
                   for key, branch in schema.items()}

        elif isinstance(schema, tuple):
            tuple_schema = {
                '_type': 'tuple',
                '_type_parameters': []}

            for index, element in enumerate(schema):
                tuple_schema['_type_parameters'].append(str(index))
                tuple_schema[f'_{index}'] = element

            return self.access(
                tuple_schema)

        elif isinstance(schema, list):
            bindings = []
            if len(schema) > 1:
                schema, bindings = schema
            else:
                schema = schema[0]
            found = self.access(schema)

            if len(bindings) > 0:
                found = found.copy()

                if '_type_parameters' not in found:
                    found['_type_parameters'] = []
                    for index, binding in enumerate(bindings):
                        found['_type_parameters'].append(str(index))
                        found[f'_{index}'] = binding
                else:
                    for parameter, binding in zip(found['_type_parameters'], bindings):
                        binding_type = self.access(binding) or binding
                        found[f'_{parameter}'] = binding_type

        elif isinstance(schema, str):
            found = self.registry.get(schema)

            if found is None and schema is not None and schema not in ('', '{}'):
                try:
                    parse = parse_expression(schema)
                    if parse != schema:
                        found = self.access(parse)
                except Exception:
                    print(f'type did not parse: {schema}')
                    traceback.print_exc()
                    
        return found
    
    def retrieve(self, schema):
        '''
        like access(schema) but raises an exception if nothing is found
        '''

        found = self.access(schema)
        if found is None:
            raise Exception(f'schema not found for type: {schema}')
        return found

    def lookup(self, type_key, attribute):
        return self.access(type_key).get(attribute)


def test_reregister_type():
    type_registry = TypeRegistry()
    type_registry.register('A', {'_default': 'a'})
    with pytest.raises(Exception) as e:
        type_registry.register('A', {'_default': 'b'}, strict=True)

    type_registry.register('A', {'_default': 'b'})

    assert type_registry.access('A')['_default'] == 'b'


def test_remove_omitted():
    result = remove_omitted(
        {'a': {}, 'b': {'c': {}, 'd': {}}},
        {'b': {'c': {}}},
        {'a': {'X': 1111}, 'b': {'c': {'Y': 4444}, 'd': {'Z': 99999}}})

    assert 'a' not in result
    assert result['b']['c']['Y'] == 4444
    assert 'd' not in result['b']


if __name__ == '__main__':
    test_reregister_type()
    test_remove_omitted()
