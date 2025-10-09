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
    Edge,
    convert_path,
    walk_path,
)

from bigraph_schema.methods import default, check, resolve, jump, traverse, bind


@dispatch
def unify(core, schema: Empty, state, context):
    schema = infer(core, state)
    return schema, state

@dispatch
def unify(core, schema: None, state, context):
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
    schema._value, result = unify(core, schema._value, state, context)
    return schema, result

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
    state = state or []
    result = []

    for index, value in enumerate(schema._values):
        if index >= len(state):
            result[index] = default(value)
        else:
            schema._values[index], result[index] = unify(
                core,
                value,
                state[index],
                walk_path(context, index))

    schema_len = len(schema._values)
    for index, over in enumerate(state[schema_len:]):
        overindex = schema_len + index
        result[overindex] = over
        schema._values[overindex] = core.infer(result[overindex])

    return schema, tuple(result)

@dispatch
def unify(core, schema: Atom, state, context):
    if state is None:
        resolve_schema = schema
    else:
        infer_schema = core.infer(state)
        resolve_schema = resolve(schema, infer_schema)

    return resolve_schema, default(resolve_schema)
        
@dispatch
def unify(core, schema: NPRandom, state, context):
    return schema, state

@dispatch
def unify(core, schema: List, state, context):
    result = []
    state = state or []

    for index, element in enumerate(state):
        schema._element, result[index] = unify(
            core,
            schema._element,
            element,
            walk_path(context, index))

    return schema, result

@dispatch
def unify(core, schema: Map, state, context):
    value_schema = schema._value
    if state and isinstance(state, dict):
        result = {}
        for key, value in state.items():
            try:
                value_schema, result[key] = unify(
                    core,
                    value_schema,
                    value,
                    walk_path(context, key))

            except Exception as e:
                # schemas did not unify which means we need to
                # make this a generic struct instead of a map
                return unify(core, {}, state, context)

        schema._value = value_schema
        return schema, result
    else:
        return schema, default(schema)

@dispatch
def unify(core, schema: Tree, state, context):
    leaf_schema = schema._leaf
    if check(leaf_schema, state):
        return unify(core, leaf_schema, state, context)

    elif state and isinstance(state, dict):
        for key, value in state.items():
            try:
                leaf_schema, result[key] = unify(
                    core,
                    leaf_schema,
                    value,
                    walk_path(context, key))
            except Exception as e:
                # schemas did not unify which means we need to
                # make this a generic struct instead of a tree
                return unify(core, {}, state, context)

    else:
        return schema, default(schema)

@dispatch
def unify(core, schema: Array, state, context):
    if state is None:
        return schema, default(schema)

    if not schema._shape:
        schema._shape = state.shape
    elif schema._shape != state.shape:
        state.reshape(schema._shape)

    if not schema._data:
        schema._data = state.dtype

    return schema, state


def default_wires(schema):
    return {
        key: [key]
        for key in schema}

def unify_path(core, schema, path, context):
    outer_path = convert_path(context['edge_path'])
    subpath = tuple(outer_path) + tuple(convert_path(path))
    subcontext = dict(context, **{
        'path': outer_path,
        'subpath': subpath})

    subschema, substate = traverse(
        context['schema'],
        context['state'],
        subpath[:-1],
        subcontext)

    import ipdb; ipdb.set_trace()

    target_schema, target_state = jump(
        subschema,
        substate,
        subpath[-1],
        subcontext)

    resolved = resolve(target_schema, schema)
    if target_state is None:
        target_state = default(target_schema)
    substate = bind(subschema, substate, subpath[-1], target_state)

    return context


def unify_wires(core, schema, wires, context):
    import ipdb; ipdb.set_trace()

    if isinstance(wires, list):
        return unify_path(
            core,
            schema,
            wires,
            context)

    else:
        for key, subschema in schema.items():
            subpath = []
            if 'subpath' in context:
                subpath = context['subpath'] or subpath
            subcontext = dict(context, **{
                'subpath': subpath + [key]})

            if key in wires:
                context = unify_wires(
                    core,
                    subschema,
                    wires[key],
                    subcontext)

    return context

@dispatch
def unify(core, schema: Edge, state, context):
    import ipdb; ipdb.set_trace()

    for port in ['inputs', 'outputs']:
        port_schema = getattr(schema, f'_{port}')
        if port not in state:
            state[port] = default_wires(port_schema)

        context = dict(context, **{
            'ports_key': port,
            'edge_path': context['path'][:-1],
            f'_{port}': port_schema})

        context = unify_wires(core, port_schema, state[port], context)

    return schema, state

@dispatch
def unify(core, schema: Node, state, context):
    if state is None:
        return schema, default(schema)

    elif isinstance(state, dict):
        result = {}
        for key in schema.__dataclass_fields__:
            if key in state:
                if not key.startswith('_'):
                    subschema, result[key] = unify(
                        core,
                        getattr(schema, key),
                        state[key],
                        walk_path(context, key))
                    setattr(schema, key, subschema)
            else:
                result[key] = default(
                    getattr(schema, key))
        for key in state:
            if key not in result:
                subschema = core.infer(state[key])
                setattr(schema, key, subschema)

        return schema, result

    else:
        for key in schema.__dataclass_fields__:
            if hasattr(state, key) and not key.startswith('_'):
                subschema, substate = unify(
                    core,
                    getattr(schema, key),
                    getattr(state, key),
                    walk_path(context, key))
                setattr(schema, key, subschema)
                setattr(state, key, substate)
            else:
                substate = default(
                    getattr(schema, key))
                setattr(state, key, substate)

        return schema, result

@dispatch
def unify(core, schema: dict, state, context):
    if not schema:
        return core.infer(state), state

    if not state:
        return schema, default(schema)

    elif isinstance(state, dict):
        result = {}
        
        for key in state:
            if key not in schema:
                schema[key] = core.infer(state[key])

        for key, subschema in schema.items():
            if key in state:
                if not key.startswith('_'):
                    schema[key], result[key] = unify(
                        core,
                        subschema,
                        state[key],
                        walk_path(context, key))
            else:
                result[key] = default(
                    subschema)

        return schema, result

    else:
        raise Exception(f'could not unify state as struct schema:\n{state}\n{schema}')
