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
    Star,
    Index,
    Jump,
    convert_path,
)

from bigraph_schema.methods import check, serialize, resolve


def walk_path(context, to, subpath):
    return {
        **context,
        'path': context['path'] + (to,),
        'subpath': subpath}


@dispatch
def jump(schema: Empty, state, to, context):
    return schema, None


@dispatch
def jump(schema: Maybe, state, to, context):
    if state is None:
        return Empty(), state
    else:
        return jump(schema._value, state, to, context)


@dispatch
def jump(schema: Wrap, state, to, context):
    return jump(schema._value, state, to, context)


@dispatch
def jump(schema: Union, state, to, context):
    for option in schema._options:
        if check(option, state):
            return jump(option, state, to, context)
    return Empty(), None


@dispatch
def jump(schema: Tuple, state, to: Key, context):
    index = Index(int(to._value))
    return jump(schema, state, index, context)


@dispatch
def jump(schema: Tuple, state, to: Index, context):
    return traverse(
        schema._values[to._value],
        state[to._value],
        context['subpath'],
        context)


@dispatch
def jump(schema: Tuple, state, to: Star, context):
    value_schemas = []
    values = []

    for index, value in enumerate(schema._values):
        subvalue_schema, subvalue = traverse(
            value,
            state[index],
            context['subpath'],
            context)

        value_schemas.append(subvalue_schema)
        values.append(subvalue)

    subschema = Tuple(_values=value_schemas)
    substate = tuple(values)

    return subschema, substate


@dispatch
def jump(schema: Tuple, state, to: Jump, context):
    # TODO: find general way to format exceptions (!)
    raise Exception(f'cannot lookup index "{to._value}" in tuple {state}\ncontext:\n{context}')


@dispatch
def jump(schema: List, state, to: Key, context):
    index = Index(int(to._value))
    return jump(schema, state, index, context)


@dispatch
def jump(schema: List, state, to: Index, context):
    return traverse(
        schema._element,
        state[to._value],
        context['subpath'],
        context)


@dispatch
def jump(schema: List, state, to: Star, context):
    subelement = Node()
    elements = []

    for index, value in state:
        subvalue_schema, subvalue = traverse(
            schema._element,
            state[index],
            context['subpath'],
            context)

        subelement = resolve(subelement, subvalue_schema)
        elements.append(subvalue)

    subschema = List(_element=subelement)
    return subschema, elements


@dispatch
def jump(schema: List, state, to: Jump, context):
    raise Exception(f'cannot lookup index "{to._value}" in list {state}\ncontext:\n{context}')


@dispatch
def jump(schema: Map, state, to: Index, context):
    key = Key(str(to._value))
    return jump(schema, state, key, context)


@dispatch
def jump(schema: Map, state, to: Key, context):
    return traverse(
        schema._value,
        state[to._value],
        context['subpath'],
        context)


@dispatch
def jump(schema: Map, state, to: Star, context):
    value_schema = Node()
    values = {}

    for key, value in state.items():
        index = serialize(schema._key, key)
        subvalue_schema, subvalue = traverse(
            schema._value,
            state[index],
            context['subpath'],
            context)

        value_schema = resolve(value_schema, subvalue_schema)
        values[index] = subvalue

    subschema = Map(_key=schema._key, _value=value_schema)
    return subschema, values


@dispatch
def jump(schema: Map, state, to: Jump, context):
    key = serialize(schema._key, to._value)
    return jump(schema, state, Key(_value=key), context)


@dispatch
def jump(schema: Tree, state, to: Key, context):
    down = state[to._value]

    subschema = schema
    if check(schema._leaf, down):
        subschema = schema._leaf

    return traverse(
        subschema,
        down,
        context['subpath'],
        context)


@dispatch
def jump(schema: Tree, state, to: Star, context):
    leaf_schema = Node()
    branches = {}

    for key, branch in state.items():
        subschema = schema
        if check(schema._leaf, branch):
            subschema = schema._leaf

        branch_schema, branch_value = traverse(
            subschema,
            branch,
            context['subpath'],
            context)

        leaf_schema = resolve(leaf_schema, branch_schema)
        branches[key] = branch_value

    subschema = Tree(_leaf=leaf_schema)
    return subschema, branches


@dispatch
def jump(schema: Tree, state, to: Jump, context):
    raise Exception(f'cannot lookup key "{to._value}" in tree {state}\ncontext:\n{context}')


@dispatch
def jump(schema: Atom, state, to, context):
    if to._value:
        raise Exception(f'cannot jump in atom - key is "{to._value}" but state is an atom:\n{state}')
    else:
        return schema, state


@dispatch
def jump(schema: Edge, state, to: Key, context):
    key = to._value
    if key in ['inputs', 'outputs']:
        if not key in state:
            raise Exception(f'no "{key}" key in state to jump to:\n{state}')

        puts_schema = getattr(schema, f'_{key}')
        wires_schema = getattr(schema, key)
        subcontext = dict(context, **{
            'ports_key': key,
            'edge_path': context['path'][:-1],
            f'_{key}': puts_schema})

        return traverse(
            wires_schema,
            state[key],
            context['subpath'],
            subcontext)
    else:
        return jump(Node(), state, to, context)


@dispatch
def jump(schema: Wires, state, to: Key, context):
    key = to._value

    if not key in state:
        raise Exception(f'no entry "{key}" for wires:\n{state}')

    substate = state[key]
    if isinstance(substate, list):
        outer_path = context['edge_path'][:-1]
        subpath = tuple(convert_path(substate)) + tuple(context['subpath'])
        target_path = outer_path + subpath
        subcontext = dict(context, **{
            'path': outer_path,
            'subpath': subpath})

        return traverse(
            context['schema'],
            context['state'],
            target_path,
            subcontext)
        
    else:
        return traverse(
            schema,
            substate,
            context['subpath'],
            context)

@dispatch
def jump(schema: Node, state, to: Star, context):
    value_schema = {}
    values = {}

    for key, value in schema.__dataclass_fields__:
        if key in state:
            subschema, subvalue = traverse(
                getattr(schema, key),
                state[key],
                context['subpath'],
                context)

            value_schema[key] = subschema
            values[key] = subvalue
        else:
            raise Exception(f'traverse: no key "{key}" in state {state} at path {context["path"]}')

    return value_schema, values


@dispatch
def jump(schema: Node, state, to: Jump, context):
    key = to._value
    if key in state:
        return traverse(
            getattr(schema, key),
            state[key],
            context['subpath'],
            context)
    else:
        raise Exception(f'traverse: no key "{key}" in state {state} at path {context["path"]}')


@dispatch
def jump(schema: Node, state, to, context):
    raise Exception(f'cannot lookup key "{to._value}" in state {state}\ncontext:\n{context}')


@dispatch
def jump(schema: dict, state, to: Key, context):
    key = to._value
    if key in schema and key in state:
        return traverse(
            schema[key],
            state[key],
            context['subpath'],
            context)
    else:
        raise Exception(f'no key "{key}" in state {state} at path {context["path"]}')


@dispatch
def jump(schema: dict, state, to: Star, context):
    value_schema = {}
    values = {}

    for key in schema:
        if key in state:
            subschema, subvalue = traverse(
                schema[key],
                state[key],
                context['subpath'],
                context)

            value_schema[key] = subschema
            values[key] = subvalue

    return value_schema, values


@dispatch
def jump(schema: dict, state, to: Jump, context):
    raise Exception(f'cannot lookup key "{to._value}" in state {state}\ncontext:\n{context}')


def traverse(schema, state, path, context):
    if path:
        to = path[0]
        subpath = path[1:]
        subcontext = walk_path(context, to, subpath)

        return jump(schema, state, to, subcontext)
    else:
        return schema, state
