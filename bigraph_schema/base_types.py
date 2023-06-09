"""
==========
Base Types
==========
"""

from bigraph_schema.registry import remove_path
from bigraph_schema.units import units


base_type_library = {
    # abstract number type
    'number': {
        '_type': 'number',
        '_apply': 'accumulate',
        '_serialize': 'to_string',
        '_description': 'abstract base type for numbers'},

    'int': {
        '_type': 'int',
        '_default': '0',
        # inherit _apply and _serialize from number type
        '_deserialize': 'deserialize_int',
        '_divide': 'divide_int',
        '_description': '64-bit integer',
        '_super': 'number'},

    'float': {
        '_type': 'float',
        '_default': '0.0',
        '_deserialize': 'float',
        '_divide': 'divide_float',
        '_description': '64-bit floating point precision number',
        '_super': 'number'},

    'string': {
        '_type': 'string',
        '_default': '""',
        '_apply': 'replace',
        '_serialize': 'serialize_string',
        '_deserialize': 'deserialize_string',
        '_divide': 'divide_int',
        '_description': '64-bit integer'},

    'list': {
        '_type': 'list',
        '_default': '[]',
        '_apply': 'concatenate',
        '_serialize': 'to_string',
        '_deserialize': 'evaluate',
        '_divide': 'divide_list',
        '_type_parameters': ['element'],
        '_description': 'general list type (or sublists)'},

    'tree': {
        '_type': 'tree',
        '_default': '{}',
        '_apply': 'apply_tree',
        '_serialize': 'serialize_tree',
        '_deserialize': 'deserialize_tree',
        '_divide': 'divide_tree',
        '_type_parameters': ['leaf'],
        '_description': 'mapping from str to some type (or nested dicts)'},

    'dict': {
        '_type': 'dict',
        '_default': '{}',
        '_apply': 'apply_dict',
        '_serialize': 'serialize_dict',
        '_deserialize': 'deserialize_dict',
        '_divide': 'divide_dict',
        # TODO: create assignable type parameters?
        '_type_parameters': ['key', 'value'],
        '_description': 'mapping from keys of any type to values of any type'},

    # TODO: add native numpy array type
    'array': {
        '_type': 'array',
        '_type_parameters': ['shape', 'element']},

    'maybe': {
        '_type': 'maybe',
        '_default': 'None',
        '_apply': 'apply_maybe',
        '_serialize': 'serialize_maybe',
        '_deserialize': 'deserialize_maybe',
        '_divide': 'divide_maybe',
        '_type_parameters': ['value'],
        '_description': 'type to represent values that could be empty'},

    'edge': {
        # TODO: do we need to have defaults informed by type parameters?
        '_type': 'edge',
        '_default': '{"wires": {}}',
        '_apply': 'apply_edge',
        '_serialize': 'serialize_edge',
        '_deserialize': 'deserialize_edge',
        '_divide': 'divide_edge',
        '_type_parameters': ['ports'],
        '_description': 'hyperedges in the bigraph, with ports as a type parameter',
        'wires': 'tree[list[string]]'
    },

    # TODO -- this should support any type
    'union': {
        '_type': 'union',
        '_default': 'None',
        # '_apply': 'apply_maybe',
        # '_serialize': 'serialize_maybe',
        # '_deserialize': 'deserialize_maybe',
        # '_divide': 'divide_maybe',
        '_type_parameters': ['value'],
        # '_description': 'type to represent values that could be empty'
    },
}


#################
# Apply methods #
#################

def accumulate(current, update, bindings=None, types=None):
    if update is None:
        return current
    return current + update


def concatenate(current, update, bindings=None, types=None):
    return current + update


##################
# Divide methods #
##################
# support dividing by ratios?
# ---> divide_float({...}, [0.1, 0.3, 0.6])

def divide_float(value, ratios, bindings=None, types=None):
    half = value / 2.0
    return (half, half)


# support function types for registrys?
# def divide_int(value: int, _) -> tuple[int, int]:
def divide_int(value, bindings=None, types=None):
    half = value // 2
    other_half = half
    if value % 2 == 1:
        other_half += 1
    return half, other_half


def divide_longest(dimensions, bindings=None, types=None):
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


def divide_list(l, bindings, types):
    result = [[], []]
    divide_type = bindings['element']
    divide = divide_type['_divide']

    for item in l:
        if isinstance(item, list):
            divisions = divide_list(item, bindings, types)
        else:
            divisions = divide(item, divide_type, types)

        result[0].append(divisions[0])
        result[1].append(divisions[1])

    return result


def replace(current, update, bindings=None, types=None):
    return update


#####################
# Serialize methods #
#####################

def serialize_string(value, bindings=None, types=None):
    return f'"{value}"'


def deserialize_string(serialized, bindings=None, types=None):
    if serialized[0] != '"' or serialized[-1] != '"':
        raise Exception(f'deserializing str which requires double quotes: {serialized}')
    return serialized[1:-1]


def to_string(value, bindings=None, types=None):
    return str(value)


#######################
# Deserialize methods #
#######################

def deserialize_int(serialized, bindings=None, types=None):
    return int(serialized)


def deserialize_float(serialized, bindings=None, types=None):
    return float(serialized)


def evaluate(serialized, bindings=None, types=None):
    return eval(serialized)







# In Progress?
##############


# TODO: make these work
def apply_tree(current, update, bindings, types):
    if isinstance(update, dict):
        if current is None:
            current = {}
        for key, branch in update.items():
            if key == '_add':
                current.update(branch)
            elif key == '_remove':
                current = remove_path(current, branch)
            else:
                current[key] = apply_tree(
                    current.get(key),
                    branch,
                    bindings,
                    types)

        return current
    else:
        leaf_type = bindings['leaf']
        if current is None:
            current = types.default(leaf_type)
        return types.apply(leaf_type, current, update)


def divide_tree(tree, bindings, types):
    result = [{}, {}]
    # get the type of the values for this dict
    divide_type = bindings['leaf']
    divide_function = divide_type['_divide']
    # divide_function = types.registry_registry.type_attribute(
    #     divide_type,
    #     '_divide')

    for key, value in tree:
        if isinstance(value, dict):
            divisions = divide_tree(value)
        else:
            divisions = types.divide(divide_type, value)

        result[0][key], result[1][key] = divisions

    return result


def serialize_tree(value, bindings=None, types=None):
    return value


def deserialize_tree(serialized, bindings=None, types=None):
    return serialized


def apply_dict(current, update, bindings=None, types=None):
    pass


def divide_dict(value, bindings=None, types=None):
    return value


def serialize_dict(value, bindings=None, types=None):
    return value


def deserialize_dict(serialized, bindings=None, types=None):
    return serialized


def apply_maybe(current, update, bindings, types):
    if current is None or update is None:
        return update
    else:
        value_type = bindings['value']
        return types.apply(value_type, current, update)


def divide_maybe(value, bindings):
    if value is None:
        return [None, None]
    else:
        pass


def serialize_maybe(value, bindings, types):
    if value is None:
        return NONE_SYMBOL
    else:
        value_type = bindings['value']
        return serialize(value_type, value)


def deserialize_maybe(serialized, bindings, types):
    if serialized == NONE_SYMBOL:
        return None
    else:
        value_type = bindings['value']
        return deserialize(value_type, serialized)


# TODO: deal with all the different unit types
def apply_units(current, update, bindings, types):
    return current + update


def serialize_units(value, bindings, types):
    return str(value)


def deserialize_units(serialized, bindings, types):
    return units(serialized)


def divide_units(value, bindings, types):
    return [value, value]


# TODO: implement edge handling
def apply_edge(current, update, bindings, types):
    return current + update


def serialize_edge(value, bindings, types):
    return str(value)


def deserialize_edge(serialized, bindings, types):
    return eval(serialized)


def divide_edge(value, bindings, types):
    return [value, value]
