from ast import literal_eval
from plum import dispatch
import numpy as np
from numpy.random.mtrand import RandomState

from bigraph_schema.utilities import NONE_SYMBOL

from bigraph_schema.schema import (
    Node,
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
    Dtype,
    Array,
    Key,
    Path,
    Wires,
    Schema,
    Edge,
)

from bigraph_schema.methods import infer, check


@dispatch
def unify(core, schema: Empty, state, context):
    schema = infer(core, state)
    return schema, state

@dispatch
def unify(core, schema: Maybe, state, context):
    if state is None:
        return schema, state
    else:
        subcontext = walk_path(context, '_value')
        schema._value, result = unify(core, schema._value, state, subcontext)
        return schema, result

@dispatch
def unify(core, schema: Wrap, state, context):
    return unify(core, schema._value, state, context)

@dispatch
def unify(core, schema: Union, state, context):
    for option in schema._options:
        if check(option, state):
            subcontext = walk_path(context, '_options')
            _, inner = unify(core, option, state, subcontext)
            return schema, inner

    initial = default(schema)
    return schema, initial

@dispatch
def unify(core, schema: Tuple, state, context):
    result = []
    for index, value in enumerate(schema._values):
        schema._values[index], result[index] = unify(
            core,
            schema._values[index],
            state.get(index),
            walk_path(context, index))

    schema_len = len(schema._values)
    for index, over in enumerate(state[schema_len:]):
        overindex = schema_len + index
        result[overindex] = over
        schema._values[overindex] = infer(result[overindex])

    return schema, tuple(result)

@dispatch
def unify(core, schema: Atom, state, context):
    return schema, state
        
@dispatch
def unify(core, schema: NPRandom, state, context):
    return schema, state

@dispatch
def unify(core, schema: List, state, context):
    result = []
    for index, element in enumerate(state):
        schema._element, result[index] = unify(
            core,
            schema._element,
            element,
            walk_path(context, index))

    return schema, result

@dispatch
def deserialize(schema: Map, encode):
    if isinstance(encode, str):
        encode = literal_eval(encode)

    if isinstance(encode, dict):
        result = {
            key: deserialize(schema._value, value)
            for key, value in encode.items()}

        if not isinstance(schema._key, String):
            result = [(deserialize(schema._key, key), value)
                for key, value in encode.items()]

        return result

@dispatch
def deserialize(schema: Tree, encode):
    if isinstance(encode, str):
        encode = literal_eval(encode)

    leaf_code = deserialize(schema._leaf, encode)
    if leaf_code is not None:
        return leaf_code
    elif isinstance(encode, dict):
        return {
            key: deserialize(schema, value)
            for key, value in encode.items()}

@dispatch
def deserialize(schema: Dtype, encode):
    return encode

@dispatch
def deserialize(schema: Array, encode):
    state = np.array(
        encode,
        dtype=schema._data)

    if state.shape != schema._shape:
        state.reshape(schema._shape)

    return state

@dispatch
def deserialize(schema: Node, encode):
    if isinstance(encode, str):
        try:
            encode = literal_eval(encode)
        except Exception as e:
            return encode

    result = {}
    if isinstance(encode, dict):
        for key in schema.__dataclass_fields__:
            if key in encode:
                result[key] = deserialize(
                    getattr(schema, key),
                    encode.get(key))
        return result
    else:
        for key in schema.__dataclass_fields__:
            if hasattr(encode, key):
                result[key] = deserialize(
                    getattr(schema, key),
                    getattr(encode, key))

    if result:
        return result

@dispatch
def deserialize(schema: dict, encode):
    if isinstance(encode, str):
        try:
            encode = literal_eval(encode)
        except Exception as e:
            return encode

    if isinstance(encode, dict):
        result = {}
        
        for key, subschema in schema.items():
            if subschema == ['A']:
                import ipdb; ipdb.set_trace()
            if key in encode:
                outcome = deserialize(
                    subschema,
                    encode[key])

                if outcome is not None:
                    result[key] = outcome

        if result:
            return result

