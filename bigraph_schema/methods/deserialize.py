from ast import literal_eval
from plum import dispatch
import numpy as np
from numpy.random.mtrand import RandomState

from bigraph_schema.protocols import local_lookup
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
    Array,
    Key,
    Path,
    Wires,
    Protocol,
    LocalProtocol,
    Schema,
    Link,
)


@dispatch
def deserialize(core, schema: Empty, encode):
    return None

@dispatch
def deserialize(core, schema: Maybe, encode):
    if encode is not None and encode != NONE_SYMBOL:
        return deserialize(core, schema._value, encode)

@dispatch
def deserialize(core, schema: Wrap, encode):
    return deserialize(core, schema._value, encode)

@dispatch
def deserialize(core, schema: Union, encode):
    for option in schema._options:
        decode = deserialize(core, option, encode)
        if decode is not None:
            return decode

@dispatch
def deserialize(core, schema: Tuple, encode):
    if isinstance(encode, str):
        encode = literal_eval(encode)

    if isinstance(encode, (list, tuple)):
        return tuple([
            deserialize(core, value, code)
            for value, code in zip(
                schema._values, encode)])

@dispatch
def deserialize(core, schema: Boolean, encode):
    if encode == 'true':
        return True
    elif encode == 'false':
        return False
        
@dispatch
def deserialize(core, schema: Integer, encode):
    try:
        result = int(encode)
        return result
    except Exception:
        pass

@dispatch
def deserialize(core, schema: Float, encode):
    try:
        result = float(encode)
        return result
    except Exception:
        pass

@dispatch
def deserialize(core, schema: NPRandom, encode):
    if isinstance(encode, RandomState):
        return encode
    else:
        state = deserialize(core, schema.state, encode)
        random = RandomState()
        random.set_state(state)

        return random

@dispatch
def deserialize(core, schema: String, encode):
    return encode

@dispatch
def deserialize(core, schema: List, encode):
    if isinstance(encode, str):
        encode = literal_eval(encode)

    if isinstance(encode, (list, tuple)):
        return [
            deserialize(core, schema._element, element)
            for element in encode]

@dispatch
def deserialize(core, schema: Map, encode):
    if isinstance(encode, str):
        encode = literal_eval(encode)

    if isinstance(encode, dict):
        result = {
            key: deserialize(core, schema._value, value)
            for key, value in encode.items()}

        if not isinstance(schema._key, String):
            result = [(deserialize(core, schema._key, key), value)
                for key, value in encode.items()]

        return result

@dispatch
def deserialize(core, schema: Tree, encode):
    if isinstance(encode, str):
        encode = literal_eval(encode)

    leaf_code = deserialize(core, schema._leaf, encode)
    if leaf_code is not None:
        return leaf_code
    elif isinstance(encode, dict):
        return {
            key: deserialize(core, schema, value)
            for key, value in encode.items()}


@dispatch
def deserialize(core, schema: Array, encode):
    state = np.array(
        encode,
        dtype=schema._data)

    if state.shape != schema._shape:
        state.reshape(schema._shape)

    return state


@dispatch
def load_protocol(core, protocol: LocalProtocol, data):
    if isinstance(data, str):
        return local_lookup(core, data)
    else:
        raise Exception(f'address must be str, not {data}')


@dispatch
def load_protocol(core, protocol: Protocol, data):
    raise Exception(f'protocol {protocol} with data {data} not implemented (!)')


@dispatch
def load_protocol(core, protocol, data):
    raise Exception(f'value is not a protocol: {protocol}')


@dispatch
def deserialize(core, schema: Link, encode):
    address = encode.get('address', 'local:edge')
    if isinstance(address, str):
        if ':' not in address:
            import ipdb; ipdb.set_trace()
        protocol, data = address.split(':', 1)
        address = {
            'protocol': protocol,
            'data': data}

    protocol = address.get('protocol', 'local')
    protocol_schema = core.access(protocol)
    edge_class = load_protocol(core, protocol_schema, address['data'])
    config_schema = edge_class.config_schema
    config = core.fill(config_schema, encode.get('config', {}))
    if not core.check(config_schema, config):
        raise Exception(f'config {config} provided to {address} does not match the config_schema {config_schema}')
    edge_instance = edge_class(config, core)
    interface = edge_instance.interface()

    inputs_schema = core.resolve(
        schema._inputs,
        interface['inputs'])

    outputs_schema = core.resolve(
        schema._outputs,
        interface['outputs'])

    return {
        'address': address,
        'config': config,
        '_inputs': inputs_schema,
        '_outputs': outputs_schema,
        'instance': edge_instance,
        'inputs': deserialize(core, schema.inputs, encode.get('inputs', {})),
        'outputs': deserialize(core, schema.outputs, encode.get('outputs', {}))}


@dispatch
def deserialize(core, schema: Node, encode):
    if isinstance(encode, str):
        try:
            encode = literal_eval(encode)
        except Exception as e:
            return encode

    result = {}
    if isinstance(encode, dict):
        for key in schema.__dataclass_fields__:
            if key in encode:
                result[key] = deserialize(core, 
                    getattr(schema, key),
                    encode.get(key))
        return result
    else:
        for key in schema.__dataclass_fields__:
            if hasattr(encode, key):
                result[key] = deserialize(core, 
                    getattr(schema, key),
                    getattr(encode, key))

    if result:
        return result

@dispatch
def deserialize(core, schema: dict, encode):
    # use access_type and fix for links (!)
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
                    core, 
                    subschema,
                    encode[key])

                if outcome is not None:
                    result[key] = outcome

        if result:
            return result

