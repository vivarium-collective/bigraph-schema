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
)

from bigraph_schema.methods import check, serialize, resolve


def walk_path(context, step):
    return {
        **context,
        'path': context['path'] + (step,)}


@dispatch
def traverse(schema: Empty, state, path, context):
    return schema, None


@dispatch
def traverse(schema: Maybe, state, path, context):
    if state is None:
        return Empty(), state
    else:
        return traverse(schema._value, state, path, context)


@dispatch
def traverse(schema: Wrap, state, path, context):
    return traverse(schema._value, state, path, context)


@dispatch
def traverse(schema: Union, state, path, context):
    for option in schema._options:
        if check(option, state):
            return traverse(option, state, path, context)
    return Empty(), None


@dispatch
def traverse(schema: Tuple, state, path, context):
    if path:
        step = path[0]
        subpath = path[1:]

        if step == '*':
            value_schemas = []
            values = []

            for index, value in enumerate(schema._values):
                subcontext = walk_path(context, index)
                subvalue_schema, subvalue = traverse(
                    value,
                    state[index],
                    subpath,
                    subcontext)

                value_schemas.append(subvalue_schema)
                values.append(subvalue)

            subschema = Tuple(_values=value_schemas)
            substate = tuple(values)

            return subschema, substate

        else:
            if isinstance(step, str):
                step = int(step)

            return traverse(
                schema._values[step],
                state[step],
                subpath,
                walk_path(
                    context, step))
    else:
        return schema, state


@dispatch
def traverse(schema: List, state, path, context):
    if path:
        step = path[0]
        subpath = path[1:]

        if step == '*':
            subelement = Node()
            elements = []

            for index, value in state:
                subcontext = walk_path(context, index)
                subvalue_schema, subvalue = traverse(
                    schema._element,
                    state[index],
                    subpath,
                    subcontext)

                subelement = resolve(subelement, subvalue_schema)
                elements.append(subvalue)

            subschema = List(_element=subelement)
            return subschema, elements

        else:
            if isinstance(step, str):
                step = int(step)

            return traverse(
                schema._values[step],
                state[step],
                subpath,
                walk_path(context, step))
    else:
        return schema, state


@dispatch
def traverse(schema: Map, state, path, context):
    if path:
        step = path[0]
        subpath = path[1:]

        if step == '*':
            value_schema = Node()
            values = {}

            for key, value in state.items():
                index = serialize(schema._key, key)
                subcontext = walk_path(context, key)
                subvalue_schema, subvalue = traverse(
                    schema._value,
                    state[index],
                    subpath,
                    subcontext)

                value_schema = resolve(value_schema, subvalue_schema)
                values[index] = subvalue

            subschema = Map(_key=schema._key, _value=value_schema)
            return subschema, values

        else:
            index = serialize(schema._key, key)
            subcontext = walk_path(context, key)
            return traverse(
                schema._value,
                state[index],
                subpath,
                subcontext)
    else:
        return schema, state


@dispatch
def traverse(schema: Tree, state, path, context):
    if path:
        step = path[0]
        subpath = path[1:]

        if step == '*':
            leaf_schema = Node()
            branches = {}

            for key, branch in state.items():
                subcontext = walk_path(context, key)
                if check(schema._leaf, branch):
                    branch_schema, branch_value = traverse(
                        schema._leaf,
                        branch,
                        subpath,
                        subcontext)
                else:
                    branch_schema, branch_value = traverse(
                        schema,
                        branch,
                        path,
                        subcontext)

                leaf_schema = resolve(leaf_schema, branch_schema)
                branches[key] = branch_value

            subschema = Tree(_leaf=leaf_schema)
            return subschema, branches

        else:
            down = state[step]
            subcontext = walk_path(context, step)

            if check(schema._leaf, down):
                return traverse(
                    schema._leaf,
                    down,
                    subpath,
                    subcontext)
            else:
                return traverse(
                    schema,
                    state[step],
                    subpath,
                    subcontext)
    else:
        return schema, state


@dispatch
def traverse(schema: Atom, state, path, context):
    if path:
        raise Exception(f'more path to traverse: {path}\nbut state is an atom: {state}')
    else:
        return schema, state


# @dispatch
# def traverse(schema: Edge, state, path, context):
#     pass


@dispatch
def traverse(schema: Node, state, path, context):
    if path:
        step = path[0]
        subpath = path[1:]

        if step == '*':
            value_schema = {}
            values = {}

            for key, value in schema.__dataclass_fields__:
                if key in state:
                    subcontext = walk_path(context, key)
                    subschema, subvalue = traverse(
                        getattr(schema, key),
                        state[key],
                        subpath,
                        subcontext)

                    value_schema[key]= subschema
                    values[key] = subvalue
                else:
                    raise Exception(f'traverse: no key "{key}" in state {state} at path {context["path"]}')

            return value_schema, values

        else:
            subcontext = walk_path(context, key)
            if key in state:
                return traverse(
                    getattr(schema, key),
                    state[key],
                    subpath,
                    subcontext)
            else:
                raise Exception(f'traverse: no key "{key}" in state {state} at path {context["path"]}')
    else:
        return schema, state


@dispatch
def traverse(schema: dict, state, path, context):
    if path:
        step = path[0]
        subpath = path[1:]

        if step == '*':
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

        else:
            if step in schema and step in state:
                subcontext = walk_path(context, step)
                return traverse(
                    schema[step],
                    state[step],
                    subpath,
                    subcontext)
            else:
                raise Exception(f'traverse: no key "{step}" in state {state} at path {context["path"]}')
    else:
        return schema, state
