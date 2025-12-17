from __future__ import annotations

import typing
import numpy as np
import numpy.lib.format as nf

from plum import dispatch
from dataclasses import dataclass, is_dataclass, field


@dataclass(kw_only=True)
class Node():
    _default: object = None

@dataclass(kw_only=True)
class Place(Node):
    _subnodes: dict = field(default_factory=dict)

@dataclass(kw_only=True)
class Atom(Node):
    pass

@dataclass(kw_only=True)
class Empty(Atom):
    pass

@dataclass(kw_only=True)
class Union(Node):
    _options: typing.Tuple[Node] = field(default_factory=tuple)

@dataclass(kw_only=True)
class Tuple(Node):
    _values: typing.List[Node] = field(default_factory=list)

@dataclass(kw_only=True)
class Boolean(Atom):
    pass

@dataclass(kw_only=True)
class Or(Boolean):
    _default: bool = False

@dataclass(kw_only=True)
class And(Boolean):
    _default: bool = True

@dataclass(kw_only=True)
class Xor(Boolean):
    _default: bool = False

@dataclass(kw_only=True)
class Number(Atom):
    pass

@dataclass(kw_only=True)
class Integer(Number):
    pass

@dataclass(kw_only=True)
class Float(Number):
    pass

@dataclass(kw_only=True)
class Complex(Float):
    pass

@dataclass(kw_only=True)
class Delta(Float):
    pass

@dataclass(kw_only=True)
class Nonnegative(Float):
    pass

@dataclass(kw_only=True)
class NPRandom(Node):
    state: Tuple() = field(default_factory=tuple)

@dataclass(kw_only=True)
class String(Atom):
    pass

@dataclass(kw_only=True)
class Enum(String):
    _values: typing.Tuple[str] = field(default_factory=tuple)

@dataclass(kw_only=True)
class Wrap(Node):
    _value: Node = field(default_factory=Node)

@dataclass(kw_only=True)
class Maybe(Wrap):
    pass

@dataclass(kw_only=True)
class Overwrite(Wrap):
    pass

@dataclass(kw_only=True)
class List(Node):
    _element: Node = field(default_factory=Node)

@dataclass(kw_only=True)
class Map(Node):
    _key: Node = field(default_factory=String)
    _value: Node = field(default_factory=Node)

@dataclass(kw_only=True)
class Tree(Node):
    _leaf: Node = field(default_factory=Node)

@dataclass(kw_only=True)
class Array(Node):
    _shape: typing.Tuple[int] = field(default_factory=tuple)
    _data: np.dtype = field(default_factory=lambda:np.dtype('float64'))

@dataclass(kw_only=True)
class Path(List):
    _element: Node = field(default_factory=Node)

# @dataclass(kw_only=True)
# class Wire(Union):
#     _options: typing.Tuple[Node] = (Key(), Jump())

@dataclass(kw_only=True)
class Wires(Tree):
    _leaf: Node = field(default_factory=Path)

@dataclass(kw_only=True)
class Schema(Tree):
    _leaf: Node = field(default_factory=Node)

@dataclass(kw_only=True)
class Protocol(Node):
    protocol: String = field(default_factory=String)
    data: Node = field(default_factory=Node)

@dataclass(kw_only=True)
class LocalProtocol(Protocol):
    data: String = field(default_factory=String)

@dataclass(kw_only=True)
class Link(Node):
    address: Protocol = field(default_factory=Protocol)
    config: Node = field(default_factory=Node)
    _inputs: dict = field(default_factory=dict)
    _outputs: dict = field(default_factory=dict)
    inputs: Wires = field(default_factory=Wires)
    outputs: Wires = field(default_factory=Wires)

# types for jumps in traversals

@dataclass(kw_only=True)
class Jump():
    _value: object

@dataclass(kw_only=True)
class Key(Jump):
    _value: str

@dataclass(kw_only=True)
class Index(Jump):
    _value: int

@dataclass(kw_only=True)
class Slice(Jump):
    _value: slice

@dataclass(kw_only=True)
class Star(Jump):
    _value: str = '*'

@dataclass(kw_only=True)
class Match(Jump):
    _match = str # regex

def convert_jump(key):
    if isinstance(key, Jump):
        return key

    convert_key = None
    if isinstance(key, str):
        if key == '*':
            convert_key = Star(_value=key)
        else:
            convert_key = Key(_value=key)
    elif isinstance(key, int):
        convert_key = Index(_value=key)

    return convert_key or Jump(_value=key)

def resolve_path(path):
    """
    Given a path that includes '..' steps, resolve the path to a canonical form
    """
    resolve = []

    for step in path:
        if step == '..':
            if len(resolve) == 0:
                raise Exception(f'cannot go above the top in path: "{path}"')
            else:
                resolve = resolve[:-1]
        else:
            resolve.append(step)

    return resolve

def deep_merge(dct, merge_dct):
    """
    Deep merge `merge_dct` into `dct`, modifying `dct` in-place.

    Nested dictionaries are recursively merged.
    """
    if dct is None:
        dct = {}
    if merge_dct is None:
        merge_dct = {}
    if not isinstance(merge_dct, dict):
        return merge_dct

    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(v, collections.abc.Mapping)):
            deep_merge(dct[k], v)
        else:
            dct[k] = v
    return dct

def is_empty(value):
    return value is None or value == {} or value == []

def convert_path(path):
    resolved = resolve_path(path)
    return [
        convert_jump(key)
        for key in resolved]

def blank_context(schema, state, path):
    return {
        'schema': schema,
        'state': state,
        'path': (),
        'subpath': path}

def walk_path(context, to, subpath=None):
    return {
        **context,
        'path': context['path'] + (to,),
        'subpath': subpath}

def dtype_schema(dtype: np.dtype):
    data = nf.dtype_to_descr(dtype)
    if isinstance(data, str):
        if 'f' in data or 'd' in data:
            return Float()
        elif 'U' in data:
            return String()
        elif 'i' in data or 'b' in data or 'h' in data:
            return Integer()
        elif 'F' in data or 'D' in data:
            return Complex()
    elif isinstance(data, list):
        result = {}
        for group in data:
            key = group[0]
            subschema = dtype_schema(group[1])
            if len(group) > 2:
                shape = group[2]
                subschema = Array(_shape=shape, _data=subschema)
            result[key] = subschema

        return result
    else:
        raise Exception('do not know how to interpret dtype as schema:\n\n{dtype}\n\n')
    

@dispatch
def schema_dtype(schema: Complex):
    return np.dtype('complex128')

@dispatch
def schema_dtype(schema: Float):
    return np.dtype('float64')

@dispatch
def schema_dtype(schema: Integer):
    return np.dtype('int32')

@dispatch
def schema_dtype(schema: Boolean):
    return np.dtype('bool')

@dispatch
def schema_dtype(schema: String):
    return np.dtype('unicode')

@dispatch
def schema_dtype(schema: str):
    return np.dtype(schema)

@dispatch
def schema_dtype(schema: list):
    return np.dtype(schema)

@dispatch
def schema_dtype(schema: dict):
    result = []
    if not schema:
        return None

    for key, value in schema.items():
        subschema = schema_dtype(value)
        subresult = (key, subschema)
        if isinstance(subschema, Array):
            subshape = subschema._shape
            subresult = subresult + (subshape,)

        result.append(subresult)

    if all([isinstance(key, int) for key in schema.keys()]):
        shape = max(schema.keys())
        return Array(_shape=(shape,), _data=result[0][1])
    else:
        return np.dtype(result)

@dispatch
def schema_dtype(schema):
    raise Exception(f'schema dtype not implemented for:\n\n{schema}\n\n')
    

BASE_TYPES = {
    'node': Node,
    'atom': Atom,
    'empty': Empty,
    'union': Union,
    'tuple': Tuple,
    'boolean': Boolean,
    'or': Or,
    'and': And,
    'xor': Xor,
    'number': Number,
    'integer': Integer,
    'float': Float,
    'delta': Delta,
    'nonnegative': Nonnegative,
    'random_state': NPRandom,
    'string': String,
    'enum': Enum,
    'wrap': Wrap,
    'maybe': Maybe,
    'overwrite': Overwrite,
    'list': List,
    'map': Map,
    'tree': Tree,
    # 'dtype': Dtype,
    'array': Array,
    'path': Path,
    # 'jump': Jump,
    # 'wire': Wire,
    'wires': Wires,
    'protocol': Protocol,
    'local': LocalProtocol,
    'schema': Schema,
    'link': Link}



def test_schema():
    pass


if __name__ == '__main__':
    test_schema()
