from pprint import pformat as pf
from plum import dispatch
import numpy as np
import pandas as pd
from numpy.random.mtrand import RandomState
from dataclasses import replace

from bigraph_schema.protocols import local_lookup


from bigraph_schema.schema import (
    Node,
    Empty,
    NONE_SYMBOL,
    Union,
    Tuple,
    Boolean,
    Number,
    Integer,
    Float,
    Complex,
    Delta,
    Nonnegative,
    Range,
    NPRandom,
    String,
    Enum,
    Wrap,
    Quote,
    Maybe,
    Overwrite,
    Const,
    List,
    Set,
    Map,
    Tree,
    Array,
    Frame,
    Key,
    Path,
    Wires,
    Protocol,
    LocalProtocol,
    Schema,
    Link,
    Object,
    Function,
    Quantity,
    Dtype,
    is_schema_field,
)


from bigraph_schema.methods.serialize import render
from bigraph_schema.methods.default import default


def _enrich_defaults(port_schema, v1_ports):
    """Walk a port schema and inject _default values from v1 ports_schema().

    The v1 ports_schema has runtime defaults (with correct shapes from simData)
    that the inferred schema lacks. This enriches the schema with those defaults
    so that realize/fill creates state with the right shapes.
    """
    if isinstance(port_schema, dict):
        for key, subschema in port_schema.items():
            if key in v1_ports and isinstance(v1_ports[key], dict):
                v1_port = v1_ports[key]
                if '_default' in v1_port and hasattr(subschema, '_default'):
                    if subschema._default is None or (
                        isinstance(subschema._default, (list, tuple)) and len(subschema._default) == 0
                    ):
                        subschema._default = v1_port['_default']
                elif isinstance(v1_port, dict) and not '_default' in v1_port:
                    _enrich_defaults(subschema, v1_port)
    elif hasattr(port_schema, '__dataclass_fields__'):
        for key in port_schema.__dataclass_fields__:
            if key.startswith('_'):
                continue
            attr = getattr(port_schema, key)
            if key in v1_ports and isinstance(v1_ports[key], dict):
                v1_port = v1_ports[key]
                if '_default' in v1_port and hasattr(attr, '_default'):
                    if attr._default is None or (
                        isinstance(attr._default, (list, tuple)) and len(attr._default) == 0
                    ):
                        attr._default = v1_port['_default']
                elif isinstance(v1_port, dict) and not '_default' in v1_port:
                    _enrich_defaults(attr, v1_port)


@dispatch
def realize(core, schema: tuple, encode, path=()):
    return schema, encode, []

@dispatch
def realize(core, schema: Quote, encode, path=()):
    # Opaque — pass through without walking or inferring the value
    return schema, encode, []

@dispatch
def realize(core, schema: Empty, encode, path=()):
    return schema, encode, []

@dispatch
def realize(core, schema: Maybe, encode, path=()):
    if encode is not None and encode != NONE_SYMBOL:
        return realize(core, schema._value, encode)
    return schema, None, []

@dispatch
def realize(core, schema: Wrap, encode, path=()):
    if encode is None:
        encode = default(schema)
    outschema, outstate, merges = realize(core, schema._value, encode, path=path)
    schema._value = outschema
    return schema, outstate, merges

@dispatch
def realize(core, schema: Union, encode, path=()):
    for option in schema._options:
        decode_schema, decode_state, merges = realize(core, option, encode)
        if decode_state is not None:
            return decode_schema, decode_state, merges
    return schema, None, []

@dispatch
def realize(core, schema: Tuple, encode, path=()):
    merges = []

    if isinstance(encode, str):
        import json
        if encode.startswith('['):
            try:
                encode = json.loads(encode)
            except (json.JSONDecodeError, ValueError):
                pass
        elif encode.startswith('('):
            try:
                encode = json.loads('[' + encode[1:-1] + ']')
            except (json.JSONDecodeError, ValueError):
                pass

    if isinstance(encode, (list, tuple)):
        subvalues = []
        subtuple = []
        for value, code, index in zip(schema._values, encode, range(len(encode))):
            subvalue, subcode, submerges = realize(core, value, code, path+(index,))
            subvalues.append(subvalue)
            subtuple.append(subcode)
            merges += submerges

        result_schema = replace(schema, **{'_values': subvalues})
        result_state = tuple(subtuple)

        return result_schema, result_state, merges

    else:
        default_schema, default_state, merges = core.default_merges(schema, path=path)
        return default_schema, default_state, merges


def realize_default(core, schema, encode: dict, path=()):
    default_state = encode.get('_default', default(schema))
    return realize(core, schema, default_state, path=path)


@dispatch
def realize(core, schema: Boolean, encode, path=()):
    if encode is None:
        schema, state = core.default(schema, path=path)
        return schema, state, []
    elif isinstance(encode, dict):
        return realize_default(core, schema, encode, path=path)
    elif isinstance(encode, (bool, np.bool_)):
        return schema, bool(encode), []
    elif encode == 'true':
        return schema, True, []
    elif encode == 'false':
        return schema, False, []
    else:
        # Signal "not a boolean" so Union can try other options.
        return schema, None, []


_INT_DTYPES = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}
_FLOAT_DTYPES = {16: np.float16, 32: np.float32, 64: np.float64}
_COMPLEX_DTYPES = {64: np.complex64, 128: np.complex128}

@dispatch
def realize(core, schema: Integer, encode, path=()):
    if encode is None:
        schema, state = core.default(schema, path=path)
        return schema, state, []
    if isinstance(encode, dict):
        return realize_default(core, schema, encode, path=path)

    # Reject bool: in Python ``bool`` is an ``int`` subclass, but we
    # want Union[Boolean, Integer] to route True/False to Boolean.
    # Everything else that ``int()`` can coerce is accepted so schemas
    # that mis-type a float as int don't silently blow up in save_bundle
    # with None-valued numerics.
    if isinstance(encode, bool):
        return schema, None, []

    try:
        bits = getattr(schema, '_bits', 0)
        if bits and bits in _INT_DTYPES:
            result = _INT_DTYPES[bits](encode)
        else:
            result = int(encode)
        return schema, result, []
    except Exception:
        return schema, None, []

@dispatch
def realize(core, schema: Float, encode, path=()):
    if encode is None:
        schema, state = core.default(schema, path=path)
        return schema, state, []
    elif isinstance(encode, dict):
        return realize_default(core, schema, encode, path=path)
    else:
        # Reject bool so unions can distinguish boolean from float.
        if isinstance(encode, bool):
            return schema, None, []
        if not isinstance(encode, (int, float, np.integer, np.floating, str)):
            return schema, None, []
        try:
            bits = getattr(schema, '_bits', 0)
            if bits and bits in _FLOAT_DTYPES:
                result = _FLOAT_DTYPES[bits](encode)
            else:
                result = float(encode)
            return schema, result, []
        except Exception:
            return schema, None, []

@dispatch
def realize(core, schema: Complex, encode, path=()):
    if encode is None:
        schema, state = core.default(schema, path=path)
        return schema, state, []
    elif isinstance(encode, dict):
        return realize_default(core, schema, encode, path=path)
    else:
        try:
            bits = getattr(schema, '_bits', 0)
            if bits and bits in _COMPLEX_DTYPES:
                result = _COMPLEX_DTYPES[bits](encode)
            else:
                result = complex(encode)
            return schema, result, []
        except Exception:
            return schema, None, []

@dispatch
def realize(core, schema: Range, encode, path=()):
    if encode is None:
        schema, state = core.default(schema, path=path)
        return schema, state, []
    elif isinstance(encode, dict):
        return realize_default(core, schema, encode, path=path)
    else:
        try:
            result = float(encode)
            result = max(schema._min, min(schema._max, result))
            return schema, result, []
        except Exception:
            return schema, None, []

@dispatch
def realize(core, schema: Dtype, encode, path=()):
    """Realize a numpy dtype from its string representation."""
    if isinstance(encode, np.dtype):
        return schema, encode, []
    if encode is None:
        return schema, None, []
    try:
        return schema, np.dtype(encode), []
    except Exception:
        return schema, encode, []

@dispatch
def realize(core, schema: String, encode, path=()):
    if isinstance(encode, dict):
        return realize_default(core, schema, encode, path=path)
    if encode is None:
        schema, state = core.default(schema, path=path)
        return schema, state, []
    if isinstance(encode, str):
        return schema, encode, []
    # Non-string: signal miss so Union dispatch can try other options.
    return schema, None, []

@dispatch
def realize(core, schema: NPRandom, encode, path=()):
    import numpy as _np

    if isinstance(encode, RandomState):
        return schema, encode, []

    if encode is None:
        return schema, RandomState(), []

    # Dict produced by serialize(NPRandom): direct reconstruction.
    if isinstance(encode, dict) and 'alg' in encode and 'key' in encode:
        rs = RandomState()
        key_arr = _np.asarray(encode['key'], dtype=_np.uint32)
        rs.set_state((
            encode['alg'],
            key_arr,
            int(encode['pos']),
            int(encode['has_gauss']),
            float(encode['cached_gauss']),
        ))
        return schema, rs, []

    # Legacy: nested {'state': ...} wrapper with empty state → fresh RNG.
    if isinstance(encode, dict) and (
        not encode.get('state') or
        (isinstance(encode.get('state'), (tuple, list)) and len(encode['state']) == 0)
    ):
        return schema, RandomState(), []

    if isinstance(encode, tuple) and len(encode) == 0:
        return schema, RandomState(), []

    # Legacy: raw 5-tuple as produced by RandomState.get_state().
    if isinstance(encode, (tuple, list)) and len(encode) == 5:
        alg, key, pos, has_gauss, cached = encode
        rs = RandomState()
        rs.set_state((
            alg, _np.asarray(key, dtype=_np.uint32),
            int(pos), int(has_gauss), float(cached),
        ))
        return schema, rs, []

    print(f'NPRandom realize: unexpected encode type={type(encode).__name__}, path={path}', flush=True)
    return schema, RandomState(), []

@dispatch
def realize(core, schema: List, encode, path=()):
    decode = []
    merges = []

    if isinstance(encode, str) and encode.startswith('['):
        import json
        try:
            encode = json.loads(encode)
        except (json.JSONDecodeError, ValueError):
            pass

    if isinstance(encode, (list, tuple)):
        for index, element in enumerate(encode):
            subschema, substate, submerges = realize(core, schema._element, element, path+(index,))
            element_schema = core.resolve(schema._element, subschema)
            schema = replace(schema, **{'_element': element_schema})
            decode.append(substate)
            merges += submerges

        return schema, decode, merges

    else:
        return schema, None, []

@dispatch
def realize(core, schema: Set, encode, path=()):
    if isinstance(encode, (list, tuple, set)):
        decode = set()
        merges = []
        for element in encode:
            subschema, substate, submerges = realize(core, schema._element, element)
            decode.add(substate)
            merges += submerges
        return schema, decode, merges
    else:
        return schema, None, []

@dispatch
def _realize_map_key(core, key_schema, key_str):
    """Realize a map key from its serialized string form.

    Inverts ``_serialize_map_key``: passes the string through
    ``realize(key_schema, key_str)`` which knows how to convert
    ``'42'`` → ``42`` for Integer, ``'[0, "minimal"]'`` → ``(0, 'minimal')``
    for Tuple, etc.
    """
    _, realized_key, _ = realize(core, key_schema, key_str, path=())
    return realized_key


@dispatch
def realize(core, schema: Map, encode, path=()):
    if encode is None:
        encode = default(schema)

    if isinstance(encode, str) and encode.startswith('{'):
        import json
        try:
            encode = json.loads(encode)
        except (json.JSONDecodeError, ValueError):
            pass

    if isinstance(encode, dict):
        decode = {}
        merges = []
        schema = replace(schema, **{'_default': encode})
        value_schemas = []

        for key, value in encode.items():
            if isinstance(key, str) and key.startswith('_'):
                continue
            else:
                realized_key = _realize_map_key(core, schema._key, key)
                subschema, substate, submerges = realize(core, schema._value, value, path+(key,))
                value_schemas.append(subschema)
                decode[realized_key] = substate
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
def realize(core, schema: Tree, encode, path=()):
    decode = {}

    leaf_schema, leaf_state, merges = realize(core, schema._leaf, encode)
    if leaf_state is not None:
        return leaf_schema, leaf_state, merges

    elif isinstance(encode, dict):
        decode = {}
        for key, value in encode.items():
            subschema, substate, submerges = realize(core, schema, value, path+(key,))
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
def realize(core, schema: Array, encode, path=()):
    if isinstance(encode, np.ndarray):
        return schema, encode, []
    elif isinstance(encode, dict):
        encode = dict_values(encode)

    # Structured dtypes (named fields) need tuples, not lists
    if (isinstance(schema._data, np.dtype) and schema._data.names
            and isinstance(encode, list) and encode
            and isinstance(encode[0], list)):
        encode = [tuple(row) for row in encode]

    try:
        state = np.array(
            encode,
            dtype=schema._data)
    except OverflowError as _oe:
        inferred = np.array(encode)
        raise OverflowError(
            f'realize Array: saved dtype {inferred.dtype} cannot fit into '
            f'declared schema dtype {schema._data} at path={path}. Fix the '
            f'schema to match the actual runtime dtype.'
        ) from _oe

    if state.size > 0 and state.shape != schema._shape:
        try:
            state = state.reshape(schema._shape)
        except ValueError:
            # Shape mismatch (e.g., enriched default has different size
            # than inferred schema shape). Use the actual data shape.
            schema._shape = state.shape

    return schema, state, []


@realize.dispatch
def realize(core, schema: Frame, encode, path=()):
    if isinstance(encode, pd.DataFrame):
        return schema, encode, []
    elif not encode:
        return schema, {}, []
    else:
        return schema, pd.DataFrame(encode), []


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
        return [(subpath, port_schema, path)]

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


def realize_link(core, schema: Link, encode, path=()):
    # Invalidate any cached compiled link structure so it will be
    # rebuilt with the new wiring after realization.
    core.invalidate_link(path)

    address = encode.get('address', 'local:edge')

    if isinstance(address, str):
        if ':' not in address:
            protocol = 'local'
            data = address
        else:
            protocol, data = address.split(':', 1)

        address = {
            'protocol': protocol,
            'data': data}

    if 'instance' in encode:
        # Instance already exists — skip instantiation but still
        # compute port merges so wired state paths get created.
        # Use the instance's ports_schema() defaults to populate
        # the merge schema with correctly-shaped initial values.
        edge_instance = encode['instance']
        decode = dict(encode)

        # Enrich the port schemas with runtime defaults from ports_schema()
        if hasattr(edge_instance, 'ports_schema'):
            try:
                v1_ports = edge_instance.ports_schema()
                for port_key in ['_inputs', '_outputs']:
                    port_schema = getattr(schema, port_key, None)
                    if port_schema is not None and isinstance(v1_ports, dict):
                        _enrich_defaults(port_schema, v1_ports)
            except Exception:
                pass

    else:
        protocol = address.get('protocol', 'local')
        protocol_schema = core.access(protocol)
        edge_class = load_protocol(core, protocol_schema, address['data'])

        if edge_class is None:
            raise Exception(f'no link found at address: {address}')

        config_schema = getattr(edge_class, 'config_schema', None) or {}
        encode_config = encode.get('config', {})

        if config_schema:
            _, config = core.realize(config_schema, encode_config)
        else:
            config = encode_config

        # Try (config, core) first (standard bigraph signature), then
        # fall back to (config) only (vivarium-style processes whose
        # __init__ doesn't accept core).
        try:
            edge_instance = edge_class(config, core)
        except TypeError:
            edge_instance = edge_class(config)
        # Ensure all instances have core for config_schema resolution
        # (needed by serialize). Vivarium-style processes don't accept
        # core in __init__ so we set it after construction.
        if not hasattr(edge_instance, 'core') or edge_instance.core is None:
            edge_instance.core = core
        decode = {
            'address': address,
            'config': config,
            'instance': edge_instance}

    interface = edge_instance.interface()

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
            if isinstance(port_schema, dict):
                decode[port] = default_wires(port_schema)
            elif hasattr(port_schema, '__dataclass_fields__'):
                # Leaf schema (e.g. Float) — default wire is identity
                decode[port] = {}
            else:
                decode[port] = {}

        else:
            subschema = getattr(schema, port)

            subschema._default = encode[port]
            wires_schema, wires_state, submerges = realize(core, subschema, encode[port], path+(port,))
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

    # Realize remaining non-port keys (config, priority, interval, etc.)
    # When instance already exists, pass through config and instance
    # without expensive realization/inference.
    has_instance = 'instance' in encode
    for key, value in encode.items():
        if not key.startswith('_'):
            if has_instance and key in ('config', 'instance'):
                decode[key] = value
            elif hasattr(schema, key):
                getattr(schema, key)._default = value
            else:
                attr, decode[key], submerges = realize(
                    core, {}, value, path+(key,))
                setattr(schema, key, attr)
                merges += submerges

    # if 'shared' in encode and encode['shared'] is not None:
    #     decode['shared'] = {}
    #     for shared_name, shared_state in encode['shared'].items():
    #         link_schema, link_state, submerges = realize_link(core, schema, shared_state, path+('shared',))
    #         merges += submerges

    #         link_state['instance'].register_shared(edge_instance)
    #         decode['shared'][shared_name] = link_state

    return schema, decode, merges

@dispatch
def realize(core, schema: Link, encode, path=()):
    return realize_link(core, schema, encode, path=path)

@dispatch
def realize(core, schema: Node, encode, path=()):
    if isinstance(encode, str):
        if encode.startswith('{') or encode.startswith('['):
            import json
            try:
                encode = json.loads(encode)
            except (json.JSONDecodeError, ValueError):
                return schema, encode, []
        else:
            return schema, encode, []

    result = {}
    merges = []

    if isinstance(encode, dict):
        for key in schema.__dataclass_fields__:
            if not is_schema_field(schema, key):
                continue
            attr = getattr(schema, key)
            if key in encode:
                subschema, substate, submerges = realize(
                    core,
                    attr,
                    encode.get(key),
                    path+(key,))
                schema = replace(schema, **{
                    key: core.resolve(attr, subschema)})
                result[key] = substate
                merges += submerges
            else:
                # Schema field missing from state — fill with default
                subschema, substate, submerges = core.default_merges(
                    attr, path=path+(key,))
                schema = replace(schema, **{key: subschema})
                result[key] = substate
                merges += submerges

    if result:
        return schema, result, merges
    else:
        return schema, encode, []

@dispatch
def realize(core, schema: None, encode, path=()):
    if isinstance(encode, dict):
        if '_type' in encode:
            schema_keys = {
                key: value
                for key, value in encode.items()
                if key.startswith('_')}
            schema = core.access_type(schema_keys)
            result_schema, result_state, merges = realize(
                core, schema, encode, path=path)

            return result_schema, result_state, merges
        else:
            merges = []
            schema = {}
            state = {}
            for key, value in encode.items():
                schema[key], state[key], submerges = realize(
                    core,
                    None,
                    value,
                    path+(key,))
                merges += submerges
            return schema, state, merges
    else:
        infer_schema, merges = core.infer_merges(
            encode, path=path)
        return infer_schema, encode, merges

@dispatch
def realize(core, schema: Quantity, encode, path=()):
    """Realize a ``pint.Quantity`` from one of:
    - existing ``pint.Quantity`` (returned as-is)
    - ``{units: {...}, magnitude: ...}`` dict
    - bare numeric (uses ``schema.units``)
    - parseable string like ``'2.1 millimolar'``
    Uses the registry from ``bigraph_schema.units.set_quantity_registry``."""
    import pint
    from bigraph_schema.units import get_quantity_registry
    if isinstance(encode, pint.Quantity):
        return schema, encode, []
    ureg = get_quantity_registry()
    if isinstance(encode, dict) and 'magnitude' in encode:
        magnitude = encode['magnitude']
        if isinstance(magnitude, str):
            magnitude = float(magnitude)
        decode = (magnitude, tuple(encode.get('units', schema.units).items()))
        return schema, ureg.Quantity.from_tuple(decode), []
    if isinstance(encode, (int, float)):
        decode = (encode, tuple(schema.units.items()))
        return schema, ureg.Quantity.from_tuple(decode), []
    if isinstance(encode, str):
        try:
            return schema, ureg.parse_expression(encode), []
        except Exception:
            pass
    return schema, encode, []


@dispatch
def realize(core, schema: Function, encode, path=()):
    """Resolve a Function reference to its actual callable.

    Encode forms accepted:
      - already-callable: returned as-is
      - dict with ``module`` + ``attribute`` (and optional ``instance``):
        import the module, optionally drill into the class, return the
        attribute.
    Anything else is passed through unchanged."""
    import importlib
    if callable(encode):
        return schema, encode, []
    if not isinstance(encode, dict):
        return schema, encode, []
    module_name = encode.get('module')
    attribute_name = encode.get('attribute')
    if not module_name or not attribute_name:
        return schema, encode, []
    instance_name = encode.get('instance')
    mod = importlib.import_module(module_name)
    if instance_name and instance_name != 'None':
        cls = getattr(mod, instance_name)
        func = getattr(cls, attribute_name)
    else:
        func = getattr(mod, attribute_name)
    return schema, func, []


@dispatch
def realize(core, schema: Object, encode, path=()):
    """Realize a Python object from its serialized form.

    The serialized form is::

        {"_class": "module.Class",
         "_schema": {"field": "type_string", ...},
         "fields": {"field": serialized_value, ...}}

    Imports the class, creates a blank instance via ``__new__``,
    realizes each field through its schema, then sets ``__dict__``.
    """
    # Already an instance of the target class
    if encode is not None and not isinstance(encode, dict):
        return schema, encode, []

    if encode is None:
        return schema, None, []

    class_path = encode.get('_class') or schema._class
    field_schemas = encode.get('_schema', {})
    fields = encode.get('fields', {})

    if not class_path:
        return schema, encode, []

    # Import the class
    import importlib
    module_path, class_name = class_path.rsplit('.', 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)

    # Create blank instance (bypasses __init__).
    # Use object.__new__ for classes that override __new__ with required args.
    try:
        instance = cls.__new__(cls)
    except TypeError:
        instance = object.__new__(cls)

    # Realize each field through its schema.
    # DerivedFunction fields are deferred until after all other fields
    # are realized, since they depend on sibling field values.
    all_merges = []
    realized_dict = {}
    derived_fields = {}  # key -> recipe dict
    for key, value in fields.items():
        field_schema_str = field_schemas.get(key)
        if field_schema_str == 'derived_function':
            derived_fields[key] = value
            continue
        if field_schema_str:
            field_schema = core.access(field_schema_str)
            _, realized_value, merges = realize(
                core, field_schema, value, path + (key,))
            all_merges += merges
        else:
            realized_value = value
        realized_dict[key] = realized_value

    # Rebuild derived functions from their sympy source fields
    for key, recipe in derived_fields.items():
        if isinstance(recipe, dict):
            source_field = recipe.get('source_field')
            builder_path = recipe.get('builder')
            is_tuple = recipe.get('is_tuple', False)

            source_value = realized_dict.get(source_field)
            if source_value is not None and builder_path:
                module_path, func_name = builder_path.rsplit('.', 1)
                mod = importlib.import_module(module_path)
                builder = getattr(mod, func_name)
                result = builder(source_value)
                if is_tuple and not isinstance(result, tuple):
                    result = (result,)
                realized_dict[key] = result

    instance.__dict__ = realized_dict

    # Let classes restore lazy/derived attributes that weren't serialized
    # (compiled lambdas, caches, etc.). Called after __dict__ is populated
    # so the hook can read from realized fields.
    post = getattr(cls, '__post_realize__', None)
    if callable(post):
        try:
            post(instance)
        except Exception as _pr:
            print(f'[realize Object] __post_realize__ failed for '
                  f'{class_path}: {_pr}', flush=True)

    # Update schema with inferred class and schema info
    schema = Object(_class=class_path, _schema=field_schemas)

    return schema, instance, all_merges


@dispatch
def realize(core, schema: dict, encode, path=()):
    return _realize_dict(core, schema, encode, path=path, fill_missing=True)


def _realize_dict(core, schema, encode, path=(), fill_missing=True):
    """Shared implementation for dict realize and discover.

    When ``fill_missing=True`` (normal realize), missing schema keys
    get their schema defaults via ``default_merges``. When
    ``fill_missing=False`` (discover phase), missing keys stay missing
    so port_merges can supply better defaults in a follow-up pass.

    In discover mode, recursive dict calls use discover; non-dict
    sub-schemas delegate to realize (they just coerce existing values
    and don't fill anything missing on their own).
    """
    if encode is None and schema and fill_missing:
        encode = default(schema)

    if isinstance(encode, str):
        if encode.startswith('{') or encode.startswith('['):
            import json
            try:
                encode = json.loads(encode)
            except (json.JSONDecodeError, ValueError):
                return schema, encode, []
        else:
            return schema, encode, []

    result_schema = {}
    result_state = {}
    merges = []

    if isinstance(encode, dict):
        if '_type' in encode:
            schema = core.access_type(encode)

        for key, subschema in schema.items():
            if is_schema_field(schema, key):
                if isinstance(subschema, str):
                    subschema = core.access(subschema)
                if key in encode:
                    if fill_missing:
                        outcome_schema, outcome_state, submerges = realize(
                            core, subschema, encode[key], path+(key,))
                    else:
                        outcome_schema, outcome_state, submerges = _discover_dispatch(
                            core, subschema, encode[key], path+(key,))

                    if outcome_state is not None:
                        result_schema[key] = core.resolve(subschema, outcome_schema)
                        result_state[key] = outcome_state
                        merges += submerges

                else:
                    # Always collect merges here — Link subschemas
                    # emit port_merges even when their state is
                    # absent (default_merges realizes the link's
                    # default and its port wires contribute paths).
                    # In discovery mode we keep the schema + merges
                    # but discard the filled state; realize's second
                    # phase does the filling with port-enhanced
                    # defaults.
                    default_schema, default_state, submerges = core.default_merges(
                        subschema, path=path+(key,))
                    merges += submerges
                    if fill_missing:
                        result_schema[key] = default_schema
                        result_state[key] = default_state
                    else:
                        result_schema[key] = subschema

        for key in encode.keys():
            if (isinstance(key, str) and not key.startswith('_')) and not key in schema:
                if fill_missing:
                    result_schema[key], result_state[key], submerges = realize(
                        core, None, encode[key], path+(key,))
                else:
                    result_schema[key], result_state[key], submerges = _discover_dispatch(
                        core, None, encode[key], path+(key,))
                merges += submerges

        if result_state or not fill_missing:
            return result_schema, result_state, merges

    return schema, encode, merges


def _discover_dispatch(core, subschema, value, path):
    """Route to the discover dispatch for dict subschemas, else fall
    through to realize for non-dict types (they don't need a
    discovery-mode variant since they only coerce existing values)."""
    if isinstance(subschema, dict):
        return _realize_dict(core, subschema, value, path=path, fill_missing=False)
    return realize(core, subschema, value, path=path)


def discover(core, schema, state, path=()):
    """Walk ``state`` against ``schema``, coercing existing values and
    collecting port_merges, but leaving missing schema keys missing.
    Used as the first phase of ``core.realize`` so port_merges can
    supply better defaults than the bare schema defaults.
    """
    if isinstance(schema, dict):
        return _realize_dict(core, schema, state, path=path, fill_missing=False)
    return realize(core, schema, state, path=path)




