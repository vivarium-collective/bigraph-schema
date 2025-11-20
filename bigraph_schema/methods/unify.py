from ast import literal_eval
from plum import dispatch
import numpy as np
from numpy.random.mtrand import RandomState

from bigraph_schema.utilities import NONE_SYMBOL

from bigraph_schema.schema import (
    Node,
    Empty,
    Atom,
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
    Link,
    nest_schema,
    convert_path,
)

from bigraph_schema.methods.default import default
from bigraph_schema.methods.check import check
from bigraph_schema.methods.resolve import resolve



def walk(path, to):
    return tuple(path) + (to,)


@dispatch
def unify(core, schema: Empty, state, path):
    schema, merges = core.infer_merges(state, path=path)
    return schema, state, []

@dispatch
def unify(core, schema: None, state, path):
    schema, merges = core.infer_merges(state, path=path)
    return schema, state, merges

@dispatch
def unify(core, schema: Maybe, state, path):
    if state is None:
        return schema, state, []
    else:
        schema._value, result, merges = unify(
            core,
            schema._value,
            state,
            path)
        return schema, result, merges

@dispatch
def unify(core, schema: Wrap, state, path):
    schema._value, result, merges = unify(
        core,
        schema._value,
        state,
        path)
    return schema, result, merges

@dispatch
def unify(core, schema: Union, state, path):
    for option in schema._options:
        if check(option, state):
            _, inner, merges = unify(
                core,
                option,
                state,
                path)
            return schema, inner, merges

    initial = default(schema)
    return schema, initial, merges

@dispatch
def unify(core, schema: Tuple, state, path):
    state = state or []
    result = []
    merges = []

    for index, value in enumerate(schema._values):
        if index >= len(state):
            result[index] = default(value)
        else:
            schema._values[index], result[index], submerges = unify(
                core,
                value,
                state[index],
                walk(path, index))
            merges += submerges

    schema_len = len(schema._values)
    for index, over in enumerate(state[schema_len:]):
        overindex = schema_len + index
        result[overindex] = over
        schema._values[overindex], submerges = core.infer_merges(
            result[overindex],
            walk(path, index))
        merges += submerges

    return schema, tuple(result), merges

def state_keys(state):
    return {
        key: value
        for key, value in state.items()
        if not key.startswith('_')}

@dispatch
def unify(core, schema: Enum, state, path):
    if isinstance(state, dict):
        state = state_keys(state)

    if not state:
        state = default(schema)

    return schema, state, []
    
@dispatch
def unify(core, schema: Atom, state, path):
    if isinstance(state, dict):
        state = state_keys(state)

    if not state:
        state = default(schema)

    return schema, state, []
        
@dispatch
def unify(core, schema: NPRandom, state, path):
    return schema, state, []

@dispatch
def unify(core, schema: List, state, path):
    result = []
    state = state or []
    merges = []

    for index, element in enumerate(state):
        schema._element, result[index], submerges = unify(
            core,
            schema._element,
            element,
            walk(path, index))
        merges += submerges

    return schema, result, merges

@dispatch
def unify(core, schema: Map, state, path):
    value_schema = schema._value
    merges = []
    if state and isinstance(state, dict):
        result = {}
        for key, value in state.items():
            try:
                value_schema, result[key], submerges = unify(
                    core,
                    value_schema,
                    value,
                    walk(path, key))
                merges += submerges

            except Exception as e:
                # schemas did not unify which means we need to
                # make this a generic struct instead of a map
                return unify(core, {}, state, path)

        schema._value = value_schema
        return schema, result, merges
    else:
        return schema, default(schema), merges

@dispatch
def unify(core, schema: Tree, state, path):
    leaf_schema = schema._leaf
    merges = []
    if check(leaf_schema, state):
        return unify(core, leaf_schema, state, path)

    elif state and isinstance(state, dict):
        for key, value in state.items():
            try:
                subschema, result[key], submerges = unify(
                    core,
                    schema,
                    value,
                    walk(path, key))
                merges += submerges
            except Exception as e:
                # schemas did not unify which means we need to
                # make this a generic struct instead of a tree
                return unify(core, {}, state, path)

        return schema, result, merges

    else:
        return schema, default(schema), merges

@dispatch
def unify(core, schema: Array, state, path):
    if state is None:
        return schema, default(schema), []

    if not schema._shape:
        schema._shape = state.shape
    elif schema._shape != state.shape:
        state.reshape(schema._shape)

    if not schema._data:
        schema._data = state.dtype

    return schema, state, []


def default_wires(schema):
    return {
        key: [key]
        for key in schema}

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


@dispatch
def unify(core, schema: Link, state, path):
    merges = []

    for port in ['inputs', 'outputs']:
        port_key = f'_{port}'
        port_schema = getattr(schema, port_key)

        if port_key in state:
            state_schema = core.access(state[port_key])
            port_schema = core.resolve(
                port_schema,
                state[port_key])

            state[port_key] = port_schema

        if port not in state:
            state[port] = default_wires(port_schema)
        else:
            getattr(schema, port)._default = state[port]

        submerges = port_merges(
            port_schema,
            state[port],
            path)

        merges += submerges

    return schema, state, merges

@dispatch
def unify(core, schema: Node, state, path):
    merges = []
    if state is None:
        return schema, default(schema), merges

    elif isinstance(state, dict):
        result = {}
        for key in schema.__dataclass_fields__:
            if key in state:
                if not key.startswith('_'):
                    subschema, result[key], submerges = unify(
                        core,
                        getattr(schema, key),
                        state[key],
                        walk(path, key))
                    merges += submerges

                    setattr(schema, key, subschema)
            else:
                result[key] = default(
                    getattr(schema, key))

        for key in state:
            if key not in result:
                subschema, submerges = core.infer_merges(
                    state[key],
                    walk(path, key))
                merges += submerges

                setattr(schema, key, subschema)

        return schema, result, merges

    else:
        for key in schema.__dataclass_fields__:
            if hasattr(state, key) and not key.startswith('_'):
                subschema, substate, submerges = unify(
                    core,
                    getattr(schema, key),
                    getattr(state, key),
                    walk(path, key))
                merges += submerges
                setattr(schema, key, subschema)
                setattr(state, key, substate)
            else:
                substate = default(
                    getattr(schema, key))
                setattr(state, key, substate)

        return schema, result, merges

@dispatch
def unify(core, schema: dict, state, path):
    merges = []

    if not schema:
        schema, merges = core.infer_merges(state, path)
        return schema, state, merges

    elif not state:
        return schema, default(schema), merges

    elif isinstance(state, dict):
        result_schema = {}
        result_state = {}
        merges = []
        
        for key in state:
            infer_schema, submerges = core.infer_merges(
                state[key],
                walk(path, key))
            merges += submerges

            if key in schema:
                if not key.startswith('_'):
                    subschema = resolve(schema[key], infer_schema)
                    result_schema[key], result_state[key], submerges = unify(
                        core,
                        subschema,
                        state[key],
                        walk(path, key))
                    merges += submerges
                else:
                    result_schema[key] = schema[key]
            else:
                result_schema[key] = infer_schema
                result_state[key] = default(infer_schema)

        schema_only = list(set(schema.keys()).difference(set(state.keys())))
        for schema_key in schema_only:
            result_schema[schema_key] = schema[schema_key]
            result_state[schema_key] = default(
                schema[schema_key])

        return result_schema, result_state, merges

    else:
        raise Exception(f'could not unify state as struct schema:\n{state}\n{schema}')
