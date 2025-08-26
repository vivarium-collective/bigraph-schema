from plum import dispatch
import numpy as np

from bigraph_schema.utilities import NONE_SYMBOL

from bigraph_schema.schema import (
    Node,
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

from bigraph_schema.methods import check

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
def serialize(schema: String, state):
    return state

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
    return {
        key: serialize(
            subschema,
            state.get(key))
        for key, subschema in schema.items()}

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
        return result
    else:
        return str(state)

