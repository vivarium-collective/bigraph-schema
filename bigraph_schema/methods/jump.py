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
)

from bigraph_schema.methods import check, serialize, resolve


def walk_path(context, to):
    return {
        **context,
        'path': context['path'] + (to,)}


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
    return context['continue'](
        schema._values[to._value],
        state[to._value])


@dispatch
def jump(schema: Tuple, state, to: Star, context):
    value_schemas = []
    values = []

    for index, value in enumerate(schema._values):
        subvalue_schema, subvalue = context['continue'](
            value,
            state[index])

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
    return context['continue'](
        schema._element,
        state[to._value])


@dispatch
def jump(schema: List, state, to: Star, context):
    subelement = Node()
    elements = []

    for index, value in state:
        subvalue_schema, subvalue = context['continue'](
            schema._element,
            state[index])

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
    return context['continue'](
        schema._value,
        state[to._value])


@dispatch
def jump(schema: Map, state, to: Star, context):
    value_schema = Node()
    values = {}

    for key, value in state.items():
        index = serialize(schema._key, key)
        subvalue_schema, subvalue = context['continue'](
            schema._value,
            state[index])

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

    if check(schema._leaf, down):
        return context['continue'](
            schema._leaf,
            down)
    else:
        return context['continue'](
            schema,
            down)

@dispatch
def jump(schema: Tree, state, to: Star, context):
    leaf_schema = Node()
    branches = {}

    for key, branch in state.items():
        if check(schema._leaf, branch):
            branch_schema, branch_value = context['continue'](
                schema._leaf,
                branch)
        else:
            branch_schema, branch_value = context['continue'](
                schema,
                branch)

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
def jump(schema: Node, state, to: Key, context):
    key = to._value
    if key in state:
        return context['continue'](
            getattr(schema, key),
            state[key])
    else:
        raise Exception(f'traverse: no key "{key}" in state {state} at path {context["path"]}')


@dispatch
def jump(schema: Node, state, to: Star, context):
    value_schema = {}
    values = {}

    for key, value in schema.__dataclass_fields__:
        if key in state:
            subschema, subvalue = context['continue'](
                getattr(schema, key),
                state[key])

            value_schema[key]= subschema
            values[key] = subvalue
        else:
            raise Exception(f'traverse: no key "{key}" in state {state} at path {context["path"]}')

    return value_schema, values


@dispatch
def jump(schema: Node, state, to: Jump, context):
    raise Exception(f'cannot lookup key "{to._value}" in state {state}\ncontext:\n{context}')


@dispatch
def jump(schema: dict, state, to: Key, context):
    key = to._value
    if key in schema and key in state:
        return context['continue'](
            schema[key],
            state[key])
    else:
        raise Exception(f'traverse: no key "{key}" in state {state} at path {context["path"]}')


@dispatch
def jump(schema: dict, state, to: Star, context):
    value_schema = {}
    values = {}

    for key in schema:
        if key in state:
            subcontext = walk_path(context, key)
            subschema, subvalue = traverse(
                schema[key],
                state[key],
                subpath,
                subcontext)

            value_schema[key]= subschema
            values[key] = subvalue

    return value_schema, values


@dispatch
def jump(schema: dict, state, to: Jump, context):
    raise Exception(f'cannot lookup key "{to._value}" in state {state}\ncontext:\n{context}')


@dispatch
def jump(schema: Edge, state, to: Jump, context):
    pass


def traverse(schema, state, path, context):
    if path:
        to = path[0]
        subpath = path[1:]
        subcontext = walk_path(context, to)

        def subcontinue(schema, state):
            return traverse(schema, state, subpath, subcontext)

        subcontext['continue'] = subcontinue
        return jump(schema, state, to, subcontext)
    else:
        return schema, state


