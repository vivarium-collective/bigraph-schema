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
class Dtype(Atom):
    """Numpy dtype descriptor.  Serializes as its string form
    (e.g. ``'float64'``, ``'[(\\'x\\', \\'<f8\\'), (\\'y\\', \\'<i4\\')]'``).
    Realize reconstructs the ``np.dtype`` from that string."""
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

    # Subclasses that override the inner type's divide semantic should set
    # this to True. Used by container-wrapper dispatches (e.g. Overwrite)
    # to decide whether to delegate to the inner or apply their own
    # default divide. Avoids hard-coded ``isinstance(..., (DivideReset,
    # DivideShare, ...))`` lists that go stale when new wrappers land.
    _customizes_divide: typing.ClassVar[bool] = False

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
class DivideReset(Wrap):
    """Wrapper that resets the inner value to its default on division.

    Mirrors v1's ``_divider: {set_value: <default>}`` behavior. Use this
    for fields that should not propagate from mother to daughter unchanged
    — e.g. ``divide`` flags, ``has_triggered_division`` markers, or any
    flag whose semantics tie it to the mother's pre-division phase.

    All other dispatched methods (apply, merge, serialize, realize)
    delegate to the inner type — only ``divide`` differs.
    """
    _customizes_divide: typing.ClassVar[bool] = True

@dataclass(kw_only=True)
class DivideShare(Wrap):
    """Wrapper that shares the inner value across both daughters on division.

    Mirrors v1's ``_divider: "set"`` behavior — both daughters receive
    the mother's value (no copy, no halving, no reset). Use this for
    intensive quantities the framework would otherwise halve by default
    (notably ``array[float]``, which the framework treats as extensive
    and divides by 2 — wrong for rate-typed arrays like
    ``aa_exchange_rates``).

    All other dispatched methods delegate to the inner type — only
    ``divide`` differs.
    """
    _customizes_divide: typing.ClassVar[bool] = True

@dataclass(kw_only=True)
class LineageSeed(Wrap):
    """Marker wrapper for integer seeds that are scoped to a lineage.

    State holds the seed as an integer. Divide shares the seed across
    both daughters (matches v1's ``_divider:"set"`` for seeds — both
    daughters inherit the mother's value, then the project's load
    pipeline can recompute per-generation seeds from the cli_seed).

    The class carries no derivation logic itself: project integrations
    that need per-generation seed reseeding should compute fresh
    seeds in their bundle-load pipeline rather than at realize time.
    Keeping derivation out of the framework avoids coupling
    bigraph-schema to project-specific derivation formulas
    (e.g. v1's ``crc32(process_name, cli_seed)``).
    """
    _customizes_divide: typing.ClassVar[bool] = True

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
        # Preserve precision via _bits parameter
        bits = dtype.itemsize * 8
        if 'f' in data or 'd' in data:
            return Float(_bits=bits if bits != 64 else 0)
        elif 'U' in data:
            return String()
        elif 'b1' in data:
            return Boolean()
        elif 'i' in data or 'b' in data or 'h' in data or 'u' in data:
            return Integer(_bits=bits if bits != 32 else 0)
        elif 'F' in data or 'D' in data:
            return Complex(_bits=bits if bits != 128 else 0)
        elif 'O' in data or 'V' in data:
            return Node()  # object dtype — heterogeneous elements
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
    bits = getattr(schema, '_bits', 0)
    if bits == 64:
        return np.dtype('complex64')
    return np.dtype('complex128')

@dispatch
def schema_dtype(schema: Float):
    bits = getattr(schema, '_bits', 0)
    if bits == 16:
        return np.dtype('float16')
    if bits == 32:
        return np.dtype('float32')
    return np.dtype('float64')

@dispatch
def schema_dtype(schema: Integer):
    bits = getattr(schema, '_bits', 0)
    if bits == 8:
        return np.dtype('int8')
    if bits == 16:
        return np.dtype('int16')
    if bits == 64:
        return np.dtype('int64')
    return np.dtype('int32')

@dispatch
def schema_dtype(schema: Boolean):
    return np.dtype('bool')

@dispatch
def schema_dtype(schema: String):
    return np.dtype('unicode')

@dispatch
def schema_dtype(schema: Node):
    return np.dtype('object')

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


@dataclass(kw_only=True)
class Quantity(Node):
    """Pint ``Quantity`` — runtime value carries units alongside magnitude.

    Different from ``Number._units`` which is metadata only (the runtime
    value is a plain float and the unit string lives on the schema). A
    ``Quantity`` value is a ``pint.Quantity`` instance — ``.to(other)``,
    ``.dimensionality``, etc. are available at runtime.

    Type parameters:
        units: Dict of unit symbol → exponent (e.g. ``{'mol': 1, 'L': -1}``).
        magnitude: Schema for the numeric magnitude (typically ``Float``).
    """
    _schema_keys = Node._schema_keys | frozenset({'units', 'magnitude'})
    units: typing.Dict = field(default_factory=dict)
    magnitude: 'Node' = field(default_factory=lambda: Float())


@dataclass(kw_only=True)
class Function(Node):
    """Reference to a standalone callable, serialized by import path.

    Type parameters:
        module: Fully qualified module name (e.g. ``numpy.random``).
        instance: Optional class name when the callable is a classmethod
            or staticmethod attached to a class. ``None`` for module-level
            functions.
        attribute: Function/method name on the module or class.

    Realize imports the module and looks up the attribute. Bound methods
    are not supported here — use a domain-specific ``Method`` type for
    those (instance state isn't carried in the schema).

    Usage::

        function          # serialized as {'module', 'instance', 'attribute'}
    """
    _schema_keys = Node._schema_keys | frozenset({'module', 'instance', 'attribute'})
    module: str = ''
    instance: typing.Optional[str] = None
    attribute: str = ''


@dataclass(kw_only=True)
class Object(Node):
    """Serializable Python object — reconstructed from its ``__dict__``.

    Type parameters:
        _class: Fully qualified class name (e.g.
            ``reconstruction.ecoli.dataclasses.process.metabolism.Metabolism``)
        _schema: Dict mapping field names to their schema types.  Determined
            at save time from ``infer()`` on each field of ``__dict__``.

    Serialize/bundle walks ``__dict__`` using ``_schema`` to serialize each
    field.  Realize imports the class, creates a blank instance via
    ``__new__``, then sets ``__dict__`` from the realized fields.

    Usage::

        object[reconstruction.ecoli.dataclasses.process.metabolism.Metabolism]

    The ``_schema`` is populated at infer/serialize time and stored in the
    document alongside the data, so realize knows exactly how to
    reconstruct every field.
    """
    _schema_keys = Node._schema_keys | frozenset({'_class', '_schema'})
    _class: str = ''
    _schema: typing.Dict = field(default_factory=dict)


# Milner bigraph structural types
# ================================
# These introduce open interfaces on both the place graph and the link
# graph, turning a schema from a ground bigraph (g : ε → I) into an
# arrow (F : I → J) that can be composed with another arrow.
# See Milner, *Space and Motion of Communicating Agents* (2008),
# §2.1 Defs. 2.1–2.3 (pp. 16–17) for the formal definitions, and
# .claude/plans/milner-formalism.md for the design rationale.


@dataclass(kw_only=True)
class Site(Empty):
    """A hole in the place graph — an open inner-face position.

    In Milner's Def. 2.1 (p. 16), a place graph ``F : m → n`` has an
    inner face ``m`` indexing *sites*. Each site is a slot that a root
    of another bigraph plugs into during composition. A schema that
    contains ``Site``s is ungrounded: it describes a *context* into
    which another bigraph can be placed, not a runnable state tree.

    Inherits from ``Empty``: a site carries no state on its own.
    Composition substitutes the filler's schema into the site's
    position, so ``Site`` never coexists with runtime state — once a
    site is filled, there is no site anymore.

    ``_sort`` carries an optional place-sort label for signatures that
    classify places (Milner Ch. 6). Empty string means unsorted, which
    is the only option for a basic signature.
    """
    _schema_keys = Empty._schema_keys | frozenset({'_sort'})
    _sort: str = ''


@dataclass(kw_only=True)
class InnerName(Empty):
    """An open link-graph endpoint facing inward.

    Inner names are the domain side of the link map (Def. 2.2, p. 16):
    ``link : X ⊎ P → E ⊎ Y`` sends each inner name in ``X`` to an edge
    or an outer name. During composition ``G ∘ F``, each outer name of
    ``F`` is connected to the inner name of ``G`` of the same name; the
    common face disappears.

    Inherits from ``Empty``: the name itself is the identifier, held
    on the schema. The schema marks "there is a link endpoint here
    addressable by this name" — the link's binding is held on a
    ``Link`` node, not here.

    ``_sort`` is the optional link-sort label (Milner §6.2). Empty
    means unsorted.
    """
    _schema_keys = Empty._schema_keys | frozenset({'_sort'})
    _sort: str = ''


@dataclass(kw_only=True)
class OuterName(Empty):
    """An open link-graph endpoint facing outward.

    Outer names are the codomain side of the link map alongside edges
    (Def. 2.2, p. 16). A link ending at an outer name escapes the
    bigraph and can be joined to another bigraph's inner name of the
    same name during composition. By convention Milner draws outer
    names above a bigraph diagram and inner names below.

    Inherits from ``Empty`` — see ``InnerName`` for the reasoning.
    """
    _schema_keys = Empty._schema_keys | frozenset({'_sort'})
    _sort: str = ''


@dataclass(kw_only=True)
class Interface(Empty):
    """A bigraphical interface ``I = ⟨m, X⟩``.

    Def. 2.3 (p. 16): an interface pairs a place-graph face — here an
    ordered tuple of sites or roots of width ``m`` — with a link-graph
    face, a finite name-set ``X``. The trivial interface
    ``ε = ⟨0, ∅⟩`` is ``Interface()`` with neither places nor names.

    The same dataclass is used for both inner and outer faces: which
    side it represents is determined by how it is attached to a
    bigraph (``_inner_face`` vs ``_outer_face`` on a composite schema).
    For an inner face the ``_places`` are typically ``Site`` schemas;
    for an outer face they are the schemas of the root regions.

    Inherits from ``Empty``: an ``Interface`` is a structural
    descriptor of shape; the state lives on the bigraph that bears
    this face and on the sub-schemas in ``_places`` / ``_names``.

    ``_names`` maps name strings to optional sort labels.
    """
    _schema_keys = Empty._schema_keys | frozenset({'_places', '_names'})
    _places: typing.Tuple = field(default_factory=tuple)
    _names: typing.Dict = field(default_factory=dict)


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
    'divide_reset': DivideReset,
    'divide_share': DivideShare,
    'lineage_seed': LineageSeed,
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
    'quote': Quote,
    'object': Object,
    'function': Function,
    'quantity': Quantity,
    'dtype': Dtype,
    'site': Site,
    'inner_name': InnerName,
    'outer_name': OuterName,
    'face': Interface}


