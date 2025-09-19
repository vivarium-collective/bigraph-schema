from ast import literal_eval
from plum import dispatch
import numpy as np

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
def deserialize(schema: Empty, encode):
    return None

@dispatch
def deserialize(schema: Maybe, encode):
    if encode is not None and encode != NONE_SYMBOL:
        return deserialize(schema._value, encode)

@dispatch
def deserialize(schema: Wrap, encode):
    return deserialize(schema._value, encode)

@dispatch
def deserialize(schema: Union, encode):
    for option in schema._options:
        decode = deserialize(option, encode)
        if decode is not None:
            return decode

@dispatch
def deserialize(schema: Tuple, encode):
    if isinstance(encode, str):
        encode = literal_eval(encode)

    if isinstance(encode, (list, tuple)):
        return tuple([
            deserialize(value, code)
            for value, code in zip(
                schema._values, encode)])

@dispatch
def deserialize(schema: Boolean, encode):
    if encode == 'true':
        return True
    elif encode == 'false':
        return False
        
@dispatch
def deserialize(schema: Integer, encode):
    try:
        result = int(encode)
        return result
    except Exception:
        pass

@dispatch
def deserialize(schema: Float, encode):
    try:
        result = float(encode)
        return result
    except Exception:
        pass

@dispatch
def deserialize(schema: String, encode):
    return encode

@dispatch
def deserialize(schema: List, encode):
    if isinstance(encode, str):
        encode = literal_eval(encode)

    if isinstance(encode, (list, tuple)):
        return [
            deserialize(schema._element, element)
            for element in encode]

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
            if key in encode:
                outcome = deserialize(
                    subschema,
                    encode[key])

                if outcome is not None:
                    result[key] = outcome

        if result:
            return result

