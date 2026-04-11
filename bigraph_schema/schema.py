from __future__ import annotations

import typing
import numpy as np
import numpy.lib.format as nf
import collections

from plum import dispatch
from dataclasses import dataclass, is_dataclass, field


NONE_SYMBOL = '__nil__'


def is_schema_field(schema, key):
    """Check whether a _ prefixed key on a schema is a schema field (vs metadata).

    For Node types, checks the _schema_keys class variable.
    For dicts, all _ prefixed keys are metadata (e.g. _default, _link_path).
    Non _ prefixed keys are always schema fields.
    """
    if not isinstance(key, str) or not key.startswith('_'):
        return True
    if isinstance(schema, Node):
        return key in schema._schema_keys
    return False


@dataclass(kw_only=True)
class Node():
    _schema_keys = frozenset()
    _default: object = None

@dataclass(kw_only=True)
class Place(Node):
    _schema_keys =Node._schema_keys | frozenset({'_subnodes'})
    _subnodes: dict = field(default_factory=dict)

@dataclass(kw_only=True)
class Atom(Node):
    pass

@dataclass(kw_only=True)
class Empty(Atom):
    pass

@dataclass(kw_only=True)
class Union(Node):
    _schema_keys =Node._schema_keys | frozenset({'_options'})
    _options: typing.Tuple[Node] = field(default_factory=tuple)

@dataclass(kw_only=True)
class Tuple(Node):
    _schema_keys =Node._schema_keys | frozenset({'_values'})
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
    """Numeric types with optional precision and unit annotations.

    Parameters:
        _bits: Bit width (8, 16, 32, 64, 128). 0 means language default
            (Python arbitrary-precision int / 64-bit float).
        _units: Pint-parseable unit string. Empty means "no unit declared".

    Schema syntax::

        integer          # Python int (arbitrary precision)
        integer[64]      # numpy int64
        integer[8]       # numpy int8
        float            # Python float (64-bit)
        float[32]        # numpy float32
        float[64,fg]     # numpy float64, femtograms
        float[fg]        # Python float, femtograms (backward compatible)
    """
    _schema_keys = Atom._schema_keys | frozenset({'_units', '_bits'})
    _units: str = ''
    _bits: int = 0

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
class Range(Float):
    """Bounded float with min/max constraints."""
    _schema_keys = Float._schema_keys | frozenset({'_min', '_max'})
    _min: float = float('-inf')
    _max: float = float('inf')

@dataclass(kw_only=True)
class NPRandom(Node):
    state: Tuple() = field(default_factory=tuple)

@dataclass(kw_only=True)
class String(Atom):
    pass

@dataclass(kw_only=True)
class Enum(String):
    _schema_keys =String._schema_keys | frozenset({'_values'})
    _values: typing.Tuple[str] = field(default_factory=tuple)

@dataclass(kw_only=True)
class Wrap(Node):
    _schema_keys =Node._schema_keys | frozenset({'_value'})
    _value: Node = field(default_factory=Node)

@dataclass(kw_only=True)
class Maybe(Wrap):
    pass

@dataclass(kw_only=True)
class Overwrite(Wrap):
    pass

@dataclass(kw_only=True)
class Const(Wrap):
    """Immutable wrapper - merge and apply preserve the current value."""
    pass

@dataclass(kw_only=True)
class List(Node):
    _schema_keys =Node._schema_keys | frozenset({'_element'})
    _element: Node = field(default_factory=Node)

@dataclass(kw_only=True)
class Set(Node):
    """Unordered collection of unique elements."""
    _schema_keys =Node._schema_keys | frozenset({'_element'})
    _element: Node = field(default_factory=Node)

@dataclass(kw_only=True)
class Map(Node):
    _schema_keys =Node._schema_keys | frozenset({'_key', '_value'})
    _key: Node = field(default_factory=String)
    _value: Node = field(default_factory=Node)

@dataclass(kw_only=True)
class Tree(Node):
    _schema_keys =Node._schema_keys | frozenset({'_leaf'})
    _leaf: Node = field(default_factory=Node)

@dataclass(kw_only=True)
class Array(Node):
    """Numerical array. Optional _units applies to all elements (one
    bulk scale at wire crossing — no per-element cost)."""
    _schema_keys =Node._schema_keys | frozenset({'_data', '_units'})
    _shape: typing.Tuple[int] = field(default_factory=tuple)
    _data: np.dtype = field(default_factory=lambda:np.dtype('float64'))
    _units: str = ''

@dataclass(kw_only=True)
class Frame(Node):
    _schema_keys =Node._schema_keys | frozenset({'_columns'})
    _columns: dict = field(default_factory=dict)

@dataclass(kw_only=True)
class Path(List):
    _element: Node = field(default_factory=Node)

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
    _schema_keys =Node._schema_keys | frozenset({'_inputs', '_outputs'})
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

    return tuple(resolve)

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
        elif 'b1' in data:
            return Boolean()
        elif 'i' in data or 'b' in data or 'h' in data or 'u' in data:
            return Integer()
        elif 'F' in data or 'D' in data:
            return Complex()
        else:
            raise Exception(f'unknown dtype {data}')

    elif isinstance(data, list):
        result = {}
        for group in data:
            key = group[0]
            subschema = np.dtype(group[1])
            if len(group) > 2:
                shape = group[2]
                subschema = Array(_shape=shape, _data=subschema)
            result[key] = subschema

        return result
    else:
        raise Exception('do not know how to interpret dtype as schema:\n\n{dtype}\n\n')
    
def get_frame_schema(df):
    schema = {}
    for column in df.columns:
        schema[column] = dtype_schema(df.loc[:, column].dtype)
    return schema

def make_default(schema, state):
    return {
        '_type': schema,
        '_default': state}


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
def schema_dtype(schema: Array):
    # For sub-array fields in structured dtypes, return the Array itself
    # so that schema_dtype(dict) can extract shape + inner dtype.
    return schema

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
        if isinstance(subschema, Array):
            # Sub-array field: (name, inner_dtype, shape)
            subresult = (key, subschema._data, subschema._shape)
        else:
            subresult = (key, subschema)

        result.append(subresult)

    if all([isinstance(key, int) for key in schema.keys()]):
        shape = max(schema.keys())
        return Array(_shape=(shape,), _data=result[0][1])
    else:
        return np.dtype(result)

@dispatch
def schema_dtype(schema):
    raise Exception(f'schema dtype not implemented for:\n\n{schema}\n\n')
    

@dataclass(kw_only=True)
class Quote(Wrap):
    """Opaque value — passes through realize and apply untouched.

    Inherits from Wrap so ``_value`` records the inner type for
    introspection, but realize/apply treat the value as opaque.
    Use for values that should be carried as-is: process instances,
    simData objects, binary blobs, etc.

    Usage: ``'quote[float]'``, ``'quote[node]'``, ``'quote'``
    """
    pass


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
    'float64': Float,
    'delta': Delta,
    'nonnegative': Nonnegative,
    'complex': Complex,
    'range': Range,
    'random_state': NPRandom,
    'string': String,
    'enum': Enum,
    'wrap': Wrap,
    'maybe': Maybe,
    'overwrite': Overwrite,
    'const': Const,
    'list': List,
    'set': Set,
    'map': Map,
    'tree': Tree,
    'array': Array,
    'dataframe': Frame,
    'path': Path,
    'wires': Wires,
    'protocol': Protocol,
    'local': LocalProtocol,
    'schema': Schema,
    'link': Link,
    'quote': Quote}


