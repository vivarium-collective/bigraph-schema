from plum import dispatch
import numpy as np

from bigraph_schema.schema import (
    Node,
    Atom,
    Empty,
    Union,
    Tuple,
    Boolean,
    Number,
    Integer,
    Float,
    Delta,
    Nonnegative,
    String,
    Enum,
    Wrap,
    Maybe,
    Overwrite,
    List,
    Map,
    Tree,
    Dtype,
    Array,
    Key,
    Path,
    Wires,
    Schema,
    Edge,
)


@dispatch
def check(schema: Empty, state):
    return state is None


@dispatch
def check(schema: Maybe, state):
    if state is None:
        return True
    else:
        return check(schema._value, state)


@dispatch
def check(schema: Wrap, state):
    return check(schema._value, state)


@dispatch
def check(schema: Union, state):
    for option in schema._options:
        if check(option, state):
            return True

    return False


@dispatch
def check(schema: Tuple, state):
    if not isinstance(state, (list, tuple)):
        return False

    elif len(schema._values) == len(state):
        return all([
            check(subschema, value)
            for subschema, value in zip(schema._values, state)])

    else:
        return False


@dispatch
def check(schema: Boolean, state):
    return isinstance(state, bool)


@dispatch
def check(schema: Integer, state):
    return isinstance(state, int)


@dispatch
def check(schema: Float, state):
    return isinstance(state, float)


@dispatch
def check(schema: Nonnegative, state):
    return state >= 0


@dispatch
def check(schema: String, state):
    return isinstance(state, str)


@dispatch
def check(schema: Enum, state):
    if not isinstance(state, str):
        return False

    return state in schema._values


@dispatch
def check(schema: List, state):
    if not isinstance(state, (list, tuple)):
        return False

    return all([
        check(schema._element, element)
        for element in state])


@dispatch
def check(schema: Map, state):
    if not isinstance(state, dict):
        return False

    all_values = all([
        check(schema._value, value)
        for value in state.values()])

    if isinstance(schema._key, String):
        return all_values

    else:
        # if the keys are not strings, we must deserialize
        # them all to tell if they pass the check?
        # - this seems expensive?
        all_keys = all([
            check(schema._key, deserialize(key))
            for key in state.keys()])

        return all_keys and all_values


@dispatch
def check(schema: Tree, state):
    if check(schema._leaf, state):
        return True

    elif isinstance(state, dict):
        return all([
            isinstance(key, str) and check(schema, branch)
            for key, branch in state.items()])
    else:
        return False


@dispatch
def check(schema: Dtype, state):
    if not isinstance(state, np.dtype):
        return False

    return np.dtype(schema._fields) == state


@dispatch
def check(schema: Array, state):
    if not isinstance(state, np.ndarray):
        return False

    shape_match = tuple(schema._shape) == state.shape
    data_match = check(schema._data, state.dtype)

    return shape_match and data_match


@dispatch
def check(schema: Key, state):
    return isinstance(state, int) or isinstance(state, str)


@dispatch
def check(schema: Node, state):
    fields = [
        field
        for field in schema.__dataclass_fields__
        if not field.startswith('_')]

    if fields:
        if isinstance(state, dict):
            for key in schema.__dataclass_fields__:
                if not key.startswith('_'):
                    if key not in state:
                        return False
                    else:
                        down = check(
                            getattr(schema, key),
                            state[key])
                        if down is False:
                            return False
            return True
        else:
            return False
    else:
        return True


@dispatch
def check(schema: dict, state):
    for key, subschema in schema.items():
        if key not in state:
            return False
        elif not check(subschema, state[key]):
            return False

    return True


@dispatch
def check(schema, state):
    raise Exception(f'not a valid schema: {schema}')
