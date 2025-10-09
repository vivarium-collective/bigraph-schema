from plum import dispatch
import numpy as np
from numpy.random.mtrand import RandomState

from bigraph_schema.utilities import NONE_SYMBOL

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
    NPRandom,
    String,
    Enum,
    Wrap,
    Maybe,
    Overwrite,
    List,
    Map,
    Tree,
    Array,
    Key,
    Path,
    Wires,
    Schema,
    Edge,
)

from bigraph_schema.methods.check import check


def render_associated(assoc):
    if all([isinstance(value, str) for value in assoc.values()]):
        parts = [f'{key}:{value}' for key, value in assoc.items()]
        assoc = '|'.join(parts)
    return assoc


@dispatch
def serialize(schema: Empty, state):
    return NONE_SYMBOL

@dispatch
def serialize(schema: Maybe, state):
    if state is None:
        return NONE_SYMBOL
    else:
        return serialize(
            schema._value,
            state)

@dispatch
def serialize(schema: Wrap, state):
    return serialize(schema._value, state)

@dispatch
def serialize(schema: Union, state):
    match = None
    for option in schema._options:
        if check(option, state):
            match = serialize(option, state)

            break
    return match

@dispatch
def serialize(schema: Tuple, state):
    return [
        serialize(subschema, value)
        for subschema, value in zip(schema._values, state)]

@dispatch
def serialize(schema: Boolean, state):
    if state:
        return 'true'
    else:
        return 'false'

@dispatch
def serialize(schema: NPRandom, state):
    if isinstance(state, RandomState):
        return serialize(
            schema.state,
            state.get_state())
    elif isinstance(state, (list, tuple)):
        return state
    else:
        import ipdb; ipdb.set_trace()

@dispatch
def serialize(schema: String, state):
    return state

@dispatch
def serialize(schema: np.str_, state):
    return str(state)

@dispatch
def serialize(schema: List, state):
    return [
        serialize(schema._element, element)
        for element in state]

@dispatch
def serialize(schema: Map, state):
    return {
        key: serialize(schema._value, value)
        for key, value in state.items()}

@dispatch
def serialize(schema: Tree, state):
    if check(schema._leaf, state):
        return serialize(schema._leaf, state)
    else:
        return {
            key: serialize(schema, branch)
            for key, branch in state.items()}

@dispatch
def serialize(schema: dict, state):
    if not isinstance(state, dict):
        import ipdb; ipdb.set_trace()

    result = {}

    for key, subschema in schema.items():
        if not key.startswith('_'):
            result[key] = serialize(
                subschema,
                state.get(key))

    return result


@dispatch
def serialize(schema: Number, state):
    return state


@dispatch
def serialize(schema: Atom, state):
    return str(state)


@dispatch
def serialize(schema: Array, state: np.ndarray):
    return state.tolist()

@dispatch
def serialize(schema: Array, state: list):
    return state

@dispatch
def serialize(schema: Array, state: dict):
    return state

@dispatch
def serialize(schema: Array, state):
    raise Exception(f'serializing array:\n  {schema}\nbut state is not an array?\n  {state}')

@dispatch
def serialize(schema: Node, state):
    if isinstance(state, dict):
        result = {}

        for key in schema.__dataclass_fields__:
            if not key in ('_default',):
                if key in state:
                    result[key] = serialize(
                        getattr(schema, key),
                        state[key])

        return render_associated(result)
    else:
        return str(state)

