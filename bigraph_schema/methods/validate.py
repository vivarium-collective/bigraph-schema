from pprint import pformat as pf
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
)

from bigraph_schema.methods.check import check
from bigraph_schema.methods.serialize import render


@dispatch
def validate(core, schema: Empty, state):
    if state is not None:
        return f'Empty schema is not empty:\n\n{pf(state)}\n\n'


@dispatch
def validate(core, schema: Maybe, state):
    if state is not None:
        return validate(core, schema._value, state)


@dispatch
def validate(core, schema: Wrap, state):
    return validate(core, schema._value, state)


@dispatch
def validate(core, schema: Union, state):
    for option in schema._options:
        if check(option, state):
            return

    return f'Union values did not match state:\n\n{pf(render(schema))}\n\n{pf(state)}\n\n'


def filter_nones(results):
    return [
        result
        for result in results
        if result is not None]

@dispatch
def validate(core, schema: Tuple, state):
    if not isinstance(state, (list, tuple)):
        return f'Tuple schema requires tuple values:\n\n{pf(render(schema))}\n\nnot:\n\n{pf(state)}\n\n'

    elif len(schema._values) == len(state):
        results = filter_nones([
            check(subschema, value)
            for subschema, value in zip(schema._values, state)])

        if results:
            return results

    else:
        return f'Tuple schema and state are different lengths:\n\n{pf(render(schema))}\n\n{pf(state)}\n\n'


@dispatch
def validate(core, schema: Boolean, state):
    if not isinstance(state, bool):
        return f'Boolean schema but state is not boolean:\n\nschema: {pf(render(schema))}\n\nstate: {pf(state)}\n\n'


@dispatch
def validate(core, schema: Integer, state):
    if not isinstance(state, int):
        return f'Integer schema but state is not an integer:\n\nschema: {pf(render(schema))}\n\nstate: {pf(state)}\n\n'


@dispatch
def validate(core, schema: Float, state):
    if not isinstance(state, float):
        return f'Float schema but state is not a float:\n\nschema: {pf(render(schema))}\n\nstate: {pf(state)}\n\n'


@dispatch
def validate(core, schema: Nonnegative, state):
    if state < 0:
        return f'Nonnegative schema but state is negative:\n\nschema: {pf(render(schema))}\n\nstate: {pf(state)}\n\n'


@dispatch
def validate(core, schema: String, state):
    if not isinstance(state, str):
        return f'Float schema but state is not a float:\n\nschema: {pf(render(schema))}\n\nstate: {pf(state)}\n\n'


@dispatch
def validate(core, schema: Enum, state):
    if not isinstance(state, str):
        return f'Enum schema but state is not a string:\n\nschema: {pf(render(schema))}\n\nstate: {pf(state)}\n\n'

    if not state in schema._values:
        return f'Enum schema but state is not in the enumeration:\n\nschema: {pf(render(schema))}\n\nstate: {pf(state)}\n\n'


@dispatch
def validate(core, schema: List, state):
    if not isinstance(state, (list, tuple)):
        return f'List schema but state is not a list:\n\nschema: {pf(render(schema))}\n\nstate: {pf(state)}\n\n'

    results = filter_nones([
        validate(core, schema._element, element)
        for element in state])

    if results:
        return results

@dispatch
def validate(core, schema: Map, state):
    if not isinstance(state, dict):
        return f'Map schema but state is not a map:\n\nschema: {pf(render(schema))}\n\nstate: {pf(state)}\n\n'

    results = filter_nones([
        validate(core, schema._value, value)
        for value in state.values()])

    if results:
        return results

    elif not isinstance(schema._key, String):
        # if the keys are not strings, we must deserialize
        # them all to tell if they pass the check?
        # - this seems expensive?
        results = filter_nones([
            # TODO: if deserialize needs core this will fail
            #   does that matter?
            validate(
                core,
                schema._key,
                deserialize(
                    core,
                    schema._key,
                    key))
            for key in state.keys()])

        if results:
            return results


@dispatch
def validate(core, schema: Tree, state):
    leaf_validate = validate(core, schema._leaf, state)

    if leaf_validate:
        if isinstance(state, dict):
            results = filter_nones([
                validate(core, schema, branch)
                for key, branch in state.items()])
            if results:
                return f'Tree schema but state matches neither leaf nor tree:\n\nschema: {pf(render(schema))}\n\nstate: {pf(state)}\n\n'

        else:
            return f'Tree schema but state matches neither leaf nor tree:\n\nschema: {pf(render(schema))}\n\nstate: {pf(state)}\n\n'


@dispatch
def validate(core, schema: Array, state):
    if not isinstance(state, np.ndarray):
        return f'Array schema but state is not an array:\n\nschema: {pf(render(schema))}\n\nstate: {pf(state)}\n\n'

    shape_match = tuple(schema._shape) == state.shape
    data_match = schema._data == state.dtype

    if not shape_match:
        return f'Array schema but shape does not match:\n\nschema: {pf(render(schema))}\n\nstate: {pf(state)}\n\n'
    if not data_match:
        return f'Array schema but data does not match:\n\nschema: {pf(render(schema))}\n\nstate: {pf(state)}\n\n'


@dispatch
def validate(core, schema: Key, state):
    if not isinstance(state, int) or isinstance(state, str):
        return f'Key schema but state is not a key:\n\nschema: {pf(render(schema))}\n\nstate: {pf(state)}\n\n'


@dispatch
def validate(core, schema: Node, state):
    fields = [
        field
        for field in schema.__dataclass_fields__
        if not field.startswith('_')]

    if fields:
        if isinstance(state, dict):
            result = {}
            for key in schema.__dataclass_fields__:
                if not key.startswith('_'):
                    if key not in state:
                        return f'Node schema but key "{key}" is not in state:\n\nschema: {pf(render(schema))}\n\nstate: {pf(state)}\n\n'
                    else:
                        down = validate(
                            core, 
                            getattr(schema, key),
                            state[key])

                        if down:
                            result[key] = down
            if result:
                return result
        else:
            subcheck = check(schema, state)
            if not subcheck:
                return f'Node schema but state does not match:\n\nschema: {pf(render(schema))}\n\nstate: {pf(state)}\n\n'


@dispatch
def validate(core, schema: dict, state):
    result = {}
    for key, subschema in schema.items():
        if key not in state:
            continue
            # result[key] = f'Schema has key "{key}" but state does not:\n\nschema: {pf(render(schema))}\n\nstate: {pf(state)}\n\n'
        else:
            subresult = validate(core, subschema, state[key])
            if subresult:
                result[key] = subresult

    if result:
        return result


@dispatch
def validate(core, schema, state):
    return f'Schema and state are not known:\n\nschema: {pf(render(schema))}\n\nstate: {pf(state)}\n\n'
