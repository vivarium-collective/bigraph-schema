from ast import literal_eval
from plum import dispatch
import numpy as np
from numpy.random.mtrand import RandomState
from dataclasses import replace

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
    nest_schema,
)


@dispatch
def deserialize(core, schema: Empty, encode, path=()):
    return schema, None, []

@dispatch
def deserialize(core, schema: Maybe, encode, path=()):
    if encode is not None and encode != NONE_SYMBOL:
        return deserialize(core, schema._value, encode)

@dispatch
def deserialize(core, schema: Wrap, encode, path=()):
    return deserialize(core, schema._value, encode)

@dispatch
def deserialize(core, schema: Union, encode, path=()):
    for option in schema._options:
        decode_schema, decode_state, merges = deserialize(core, option, encode)
        if decode_state is not None:
            return decode_schema, decode_state, merges
    return schema, None, []

@dispatch
def deserialize(core, schema: Tuple, encode, path=()):
    merges = []

    if isinstance(encode, str):
        encode = literal_eval(encode)

    if isinstance(encode, (list, tuple)):
        subvalues = []
        subtuple = []
        for value, code, index in zip(schema._values, encode, range(len(encode))):
            subvalue, subcode, submerges = deserialize(core, value, code, path+(index,))
            subvalues.append(subvalue)
            subtuple.append(subcode)
            merges += submerges

        result_schema = replace(schema, **{'_values': subvalues})
        result_state = tuple(subtuple)

        return result_schema, result_state, merges

    else:
        return schema, None, []

@dispatch
def deserialize(core, schema: Boolean, encode, path=()):
    if encode == 'true':
        return schema, True, []
    elif encode == 'false':
        return schema, False, []
    else:
        return schema, None, []
        
@dispatch
def deserialize(core, schema: Integer, encode, path=()):
    try:
        result = int(encode)
        return schema, result, []
    except Exception:
        return schema, None, []

@dispatch
def deserialize(core, schema: Float, encode, path=()):
    try:
        result = float(encode)
        return schema, result, []
    except Exception:
        return schema, None, []

@dispatch
def deserialize(core, schema: NPRandom, encode, path=()):
    if isinstance(encode, RandomState):
        return schema, encode, []
    else:
        state = deserialize(core, schema.state, encode)
        random = RandomState()
        random.set_state(state)

        return schema, random, []

@dispatch
def deserialize(core, schema: String, encode, path=()):
    return schema, encode, []

@dispatch
def deserialize(core, schema: List, encode, path=()):
    decode = []
    merges = []

    if isinstance(encode, str):
        encode = literal_eval(encode)

    if isinstance(encode, (list, tuple)):
        for index, element in enumerate(encode):
            subschema, substate, submerges = deserialize(core, schema._element, element, path+(index,))
            element_schema = core.resolve(schema._element, subschema)
            schema = replace(schema, **{'_element': element_schema})
            decode.append(substate)
            merges += submerges

        return schema, decode, merges

    else:
        return schema, None, []

@dispatch
def deserialize(core, schema: Map, encode, path=()):
    if isinstance(encode, str):
        encode = literal_eval(encode)

    if isinstance(encode, dict):
        decode = {}
        merges = []

        for key, value in encode.items():
            subschema, substate, submerges = deserialize(core, schema._value, value, path+(key,))
            value_schema = core.resolve(schema._value, subschema)
            schema = replace(schema, **{
                '_value': value_schema})
            decode[key] = substate
            merges += submerges

        return schema, decode, merges

    else:
        return schema, None, []

@dispatch
def deserialize(core, schema: Tree, encode, path=()):
    decode = {}

    if isinstance(encode, str):
        try:
            encode = literal_eval(encode)
        except:
            pass

    leaf_schema, leaf_state, merges = deserialize(core, schema._leaf, encode)
    if leaf_state is not None:
        return leaf_schema, leaf_state, merges

    elif isinstance(encode, dict):
        decode = {}
        for key, value in encode.items():
            subschema, substate, submerges = deserialize(core, schema, value, path+(key,))
            schema = core.resolve(schema, subschema)
            decode[key] = substate
            merges += submerges

        return schema, decode, merges

    else:
        return schema, None, []

@dispatch
def deserialize(core, schema: Array, encode, path=()):
    state = np.array(
        encode,
        dtype=schema._data)

    if state.shape != schema._shape:
        state.reshape(schema._shape)

    return schema, state, []


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


def port_merges(port_schema, wires, path):
    if isinstance(wires, (list, tuple)):
        subpath = path[:-1] + tuple(wires)
        submerges = nest_schema(
            port_schema,
            subpath)
        return [submerges]
    else:
        merges = []
        for key, subwires in wires.items():
            down = port_schema[key]
            submerges = port_merges(
                down,
                subwires,
                path)
            merges += submerges

        return merges


def default_wires(schema):
    return {
        key: [key]
        for key in schema}


def deserialize_link(core, schema: Link, encode, path=()):
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

    if edge_class is None:
        import ipdb; ipdb.set_trace()

    config_schema = edge_class.config_schema
    encode_config = encode.get('config', {})
    _, decode_config = core.deserialize(config_schema, encode_config)
    config = core.fill(config_schema, decode_config)

    if not core.check(config_schema, config):
        raise Exception(f'config {config} provided to {address} does not match the config_schema {config_schema}')
    edge_instance = edge_class(config, core)
    interface = edge_instance.interface()

    decode = {
        'address': address,
        'config': config,
        'instance': edge_instance}
    merges = []

    for port in ['inputs', 'outputs']:
        port_key = f'_{port}'
        port_schema = getattr(schema, port_key)

        if port_key in encode:
            port_schema = core.resolve(
                port_schema,
                encode[port_key])
        else:
            port_schema = core.resolve(
                port_schema,
                interface[port])

        decode[port_key] = port_schema
        schema = replace(schema, **{port_key: port_schema})

        if port not in encode or encode[port] is None:
            decode[port] = default_wires(port_schema)
        else:
            subschema = getattr(schema, port)
            subschema._default = encode[port]
            wires_schema, wires_state, submerges = deserialize(core, subschema, encode[port], path+(port,))
            if wires_state is None:
                decode[port] = default_wires(port_schema)
            else:
                decode[port] = wires_state
            merges += submerges

        if port_schema is None:
            import ipdb; ipdb.set_trace()

        submerges = port_merges(
            port_schema,
            decode[port],
            path)

        merges += submerges

    for key, value in encode.items():
        if not key.startswith('_') and hasattr(schema, key):
            getattr(schema, key)._default = value

    return schema, decode, merges

@dispatch
def deserialize(core, schema: Link, encode, path=()):
    return deserialize_link(core, schema, encode, path=path)

@dispatch
def deserialize(core, schema: Node, encode, path=()):
    if isinstance(encode, str):
        try:
            encode = literal_eval(encode)
        except Exception as e:
            return schema, encode, []

    result = {}
    merges = []

    if isinstance(encode, dict):
        for key in schema.__dataclass_fields__:
            if key in encode:
                attr = getattr(schema, key)
                subschema, substate, submerges = deserialize(
                    core,
                    attr,
                    encode.get(key),
                    path+(key,))
                schema = replace(schema, **{
                    key: core.resolve(attr, subschema)})
                result[key] = substate
                merges += submerges

    # else:
    #     for key in schema.__dataclass_fields__:
    #         if hasattr(encode, key):
    #             attr = getattr(schema, key)
    #             subschema, substate, submerges = deserialize(
    #                 core,
    #                 attr,
    #                 getattr(encode, key))
    #             schema = replace(schema, **{
    #                 key: core.resolve(attr, subschema)})
    #             result[key] = substate
    #             merges += submerges

    if result:
        return schema, result, merges
    else:
        return schema, None, []

@dispatch
def deserialize(core, schema: dict, encode, path=()):
    if isinstance(encode, str):
        try:
            encode = literal_eval(encode)
        except Exception as e:
            return schema, encode, []

    result_schema = {}
    result_state = {}
    merges = []

    if isinstance(encode, dict):
        for key, subschema in schema.items():
            if key in encode:
                outcome_schema, outcome_state, submerges = deserialize(
                    core, 
                    subschema,
                    encode[key],
                    path+(key,))

                if outcome_state is not None:
                    result_schema[key] = core.resolve(subschema, outcome_schema)
                    result_state[key] = outcome_state
                    merges += submerges

        if result_state:
            return result_schema, result_state, merges

    return schema, encode, []
            
    

