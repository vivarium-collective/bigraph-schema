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
    Array,
    Key,
    Path,
    Wires,
    Schema,
    Link,
    Star,
    Index,
    Jump,
    convert_path,
)

from bigraph_schema.methods import check


@dispatch
def bind(schema: Empty, state, to, to_state):
    return schema, None


@dispatch
def bind(schema: Maybe, state, to, to_state):
    if state is None:
        return Empty(), state
    else:
        return bind(schema._value, state, to, to_state)


@dispatch
def bind(schema: Wrap, state, to, to_state):
    return bind(schema._value, state, to, to_state)


@dispatch
def bind(schema: Union, state, to, to_state):
    for option in schema._options:
        if check(option, state):
            return bind(option, state, to, to_state)
    return Empty(), None


@dispatch
def bind(schema: Tuple, state, to: Key, to_state):
    index = Index(int(to._value))
    return bind(schema, state, index, to_state)


@dispatch
def bind(schema: Tuple, state, to: Index, to_state):
    if not check(schema._values[to._value], to_state):
        raise Exception(f'trying to bind at index {to._value}\n{to_state}\nbut does match tuple entry schema\n{schema[to._value]}')

    state[to._value] = to_state
    return state


@dispatch
def bind(schema: List, state, to: Key, to_state):
    index = Index(int(to._value))
    return bind(schema, state, index, to_state)


@dispatch
def bind(schema: List, state, to: Index, to_state):
    if not check(schema._element, to_state):
        raise Exception(f'trying to bind at index {to._value}\n{to_state}\nbut does match list element schema\n{schema._element}')

    state[to._value] = to_state
    return state


@dispatch
def bind(schema: Map, state, to: Index, to_state):
    key = Key(str(to._value))
    return bind(schema, state, key, to_state)


@dispatch
def bind(schema: Map, state, to: Key, to_state):
    if not check(schema._value, to_state):
        raise Exception(f'trying to bind with key {to._value}\n{to_state}\nbut does match map value schema\n{schema._value}')
    
    state[key] = to_state
    return state


@dispatch
def bind(schema: Tree, state, to: Key, to_state):
    if not check(schema, to_state):
        raise Exception(f'trying to bind with key {to._value}\n{to_state}\nbut does match tree schema\n{schema}')
    
    state[key] = to_state
    return state


@dispatch
def bind(schema: Atom, state, to: Key, to_state):
    if to._value:
        raise Exception(f'cannot bind to atom - key is "{to._value}" but state is an atom:\n{state}')
    else:
        return schema, state


@dispatch
def bind(schema: Node, state, to: Jump, to_state):
    key = to._value
    if not hasattr(schema, key):
        raise Exception(f'schema is missing key "{key}"\n{schema}')

    branch = getattr(schema, key)
    
    if not check(branch, to_state):
        raise Exception(f'trying to bind key "{key}" but state is does not check as {branch}\n{to_state}')

    state[key] = to_state
    return state


@dispatch
def bind(schema: dict, state, to: Jump, to_state):
    key = to._value
    if not key in schema:
        raise Exception(f'schema is missing key "{key}"\n{schema} for binding state\n{state}')

    branch = schema[key]
    
    if not check(branch, to_state):
        raise Exception(f'trying to bind key "{key}" but state is does not check as {branch}\n{to_state}')

    state[key] = to_state
    return state
