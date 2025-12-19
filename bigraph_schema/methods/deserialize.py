from ast import literal_eval
from pprint import pformat as pf
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
)


from bigraph_schema.methods.serialize import render
from bigraph_schema.methods.default import default


@dispatch
def deserialize(core, schema: Empty, encode, path=()):
    return schema, encode, []

@dispatch
def deserialize(core, schema: Maybe, encode, path=()):
    if encode is not None and encode != NONE_SYMBOL:
        return deserialize(core, schema._value, encode)
    else:
        return schema, encode, []

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
        default_schema, default_state, merges = core.default_merges(schema, path=path)
        return default_schema, default_state, merges


def deserialize_default(core, schema, encode: dict, path=()):
    default_state = encode.get('_default', default(schema))
    return deserialize(core, schema, default_state, path=path)


@dispatch
def deserialize(core, schema: Boolean, encode, path=()):
    if encode is None:
        schema, state = core.default(schema, path=path)
        return schema, state, []
    elif isinstance(encode, dict):
        return deserialize_default(core, schema, encode, path=path)
    elif encode == 'true':
        return schema, True, []
    elif encode == 'false':
        return schema, False, []
    else:
        return schema, encode, []
        
@dispatch
def deserialize(core, schema: Integer, encode, path=()):
    if encode is None:
        schema, state = core.default(schema, path=path)
        return schema, state, []
    if isinstance(encode, dict):
        return deserialize_default(core, schema, encode, path=path)

    try:
        result = int(encode)
        return schema, result, []
    except Exception:
        return schema, None, []

@dispatch
def deserialize(core, schema: Float, encode, path=()):
    if encode is None:
        schema, state = core.default(schema, path=path)
        return schema, state, []
    elif isinstance(encode, dict):
        return deserialize_default(core, schema, encode, path=path)
    else:
        try:
            result = float(encode)
            return schema, result, []
        except Exception:
            return schema, None, []

@dispatch
def deserialize(core, schema: String, encode, path=()):
    if isinstance(encode, dict):
        return deserialize_default(core, schema, encode, path=path)

    return schema, encode, []

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
        schema = replace(schema, **{'_default': encode})
        value_schemas = []

        for key, value in encode.items():
            if key.startswith('_'):
                continue
            else:
                subschema, substate, submerges = deserialize(core, schema._value, value, path+(key,))
                value_schemas.append(subschema)
                decode[key] = substate
                merges += submerges

        if value_schemas:
            value_schema = core.resolve_schemas(
                value_schemas)
            value_schema = core.resolve(schema._value, value_schema)

            schema = replace(schema, **{
                '_value': value_schema})

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

def dict_values(d):
    result = []
    for key, value in d.items():
        if isinstance(value, dict):
            value = dict_values(value)
        result.append(value)
    return result

@dispatch
def deserialize(core, schema: Array, encode, path=()):
    if isinstance(encode, np.ndarray):
        return schema, encode, []
    elif isinstance(encode, dict):
        encode = dict_values(encode)

    state = np.array(
        encode,
        dtype=schema._data)

    if state.shape != schema._shape:
        state.reshape(schema._shape)

    return schema, state, []


def load_local_protocol(core, protocol, data):
    if isinstance(data, str):
        return local_lookup(core, data)
    else:
        raise Exception(f'address must be str, not {data}')


@dispatch
def load_protocol(core, protocol: LocalProtocol, data):
    return load_local_protocol(core, protocol, data)


@dispatch
def load_protocol(core, protocol: Protocol, data):
    raise Exception(f'protocol {protocol} with data {data} not implemented (!)')


@dispatch
def load_protocol(core, protocol, data):
    raise Exception(f'value is not a protocol: {protocol}')


def port_merges(core, port_schema, wires, path):
    if isinstance(wires, (list, tuple)):
        subpath = path[:-1] + tuple(wires)
        return [(subpath, port_schema)]

    else:
        merges = []
        for key, subwires in wires.items():
            down_schema, _ = core.jump(
                port_schema,
                {},
                key)

            submerges = port_merges(
                core,
                down_schema,
                subwires,
                path)
            merges += submerges

        return merges


def default_wires(schema):
    return {
        key: [key]
        for key in schema}


def deserialize_link(core, schema: Link, encode, path=()):
    if 'instance' in encode:
        return schema, encode, []

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
        raise Exception(f'no link found at address: {address}')

    config_schema = edge_class.config_schema
    encode_config = encode.get('config', {})
    _, decode_config = core.deserialize(config_schema, encode_config)
    config = core.fill(config_schema, decode_config)

    # validate the config against the config_schema
    message = f'config provided to {address} does not match the config_schema!\n\nconfig_schema: {pf(render(config_schema))}\n\nconfig: {pf(config)}\n\n'
    core.validate(config_schema, config, message)

    edge_instance = encode.get('instance', edge_class(config, core))
    interface = edge_instance.interface()

    decode = {
        'address': address,
        'config': config,
        'instance': edge_instance}
    merges = []

    for port in ['inputs', 'outputs']:
        port_key = f'_{port}'
        port_schema = getattr(schema, port_key)

        port_schema = core.resolve(
            port_schema,
            interface[port])
        if port_key in encode and encode[port_key]:
            port_schema = core.resolve(
                port_schema,
                encode[port_key])

        decode[port_key] = port_schema
        schema = replace(schema, **{port_key: port_schema})

        if port_schema is None:
            continue

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

        submerges = port_merges(
            core,
            port_schema,
            decode[port],
            path)

        merges += submerges

    # if 'shared' in encode and encode['shared'] is not None:
    #     decode['shared'] = {}
    #     for shared_name, shared_state in encode['shared'].items():
    #         link_schema, link_state, submerges = deserialize_link(core, schema, shared_state, path+('shared',))
    #         merges += submerges

    #         link_state['instance'].register_shared(edge_instance)
    #         decode['shared'][shared_name] = link_state

    for key, value in encode.items():
        if not key.startswith('_'):
            if hasattr(schema, key):
                getattr(schema, key)._default = value
            else:
                attr, decode[key], submerges = deserialize(
                    core, {}, value, path+(key,))
                setattr(schema, key, attr)
                merges += submerges

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

    if result:
        return schema, result, merges
    else:
        return schema, encode, []

@dispatch
def deserialize(core, schema: None, encode, path=()):
    if isinstance(encode, dict) and '_type' in encode:
        schema_keys = {
            key: value
            for key, value in encode.items()
            if key.startswith('_')}
        schema = core.access_type(schema_keys)
        result_schema, result_state, merges = deserialize(
            core, schema, encode, path=path)

        return result_schema, result_state, merges
    else:
        infer_schema, merges = core.infer_merges(
            encode, path=path)
        return infer_schema, encode, merges

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
            else:
                result_schema[key], result_state[key], submerges = core.default_merges(
                    subschema,
                    path=path+(key,))
                merges += submerges

        for key in set(encode.keys()).difference(set(schema.keys())):
            if not key.startswith('_'):
                result_schema[key], result_state[key], submerges = deserialize(
                    core, None, encode[key], path+(key,))
                merges += submerges

        if result_state:
            return result_schema, result_state, merges

    return schema, encode, merges
            
    

