"""
==============
Type Functions
==============

This module includes various type functions that are essential for handling and manipulating different types of schemas and states. These functions are categorized based on their functionality and the type of schema they operate on. Below is an overview of the type functions included in this module:

1. **Apply Functions**:
   - Responsible for applying updates to various types of schemas.
   - Each function handles a specific type of schema and ensures that updates are applied correctly.

2. **Check Functions**:
   - Responsible for validating the state against various types of schemas.
   - Each function ensures that the state conforms to the expected schema type.

3. **Fold Functions**:
   - Responsible for folding the state based on the schema and a given method.
   - Each function handles a specific type of schema and ensures that the folding is done correctly.

4. **Divide Functions**:
   - Responsible for dividing the state into a number of parts based on the schema.
   - Each function handles a specific type of schema and divides the state accordingly.

5. **Serialize Functions**:
   - Responsible for converting the state into a serializable format based on the schema.
   - Each function handles a specific type of schema and ensures that the state is serialized correctly.

6. **Deserialize Functions**:
   - Responsible for converting serialized data back into the state based on the schema.
   - Each function handles a specific type of schema and ensures that the data is deserialized correctly.

7. **Slice Functions**:
   - Responsible for extracting a part of the state based on the schema and path.
   - Each function handles a specific type of schema and ensures that the correct part of the state is sliced.

8. **Bind Functions**:
   - Responsible for binding a key and its corresponding schema and state to the main schema and state.
   - Each function handles a specific type of schema and ensures that the binding is done correctly.

9. **Resolve Functions**:
   - Responsible for resolving updates to the schema.
   - Each function handles a specific type of schema and ensures that updates are resolved correctly.

10. **Dataclass Functions**:
    - Responsible for generating dataclass representations of various types of schemas.
    - Each function handles a specific type of schema and ensures that the dataclass is generated correctly.

11. **Default Functions**:
    - Responsible for providing default values for various types of schemas.
    - Each function handles a specific type of schema and ensures that the default value is generated correctly.

12. **Generate Functions**:
    - Responsible for generating schemas and states based on the provided schema and state.
    - Each function handles a specific type of schema and ensures that the generation is done correctly.

13. **Sort Functions**:
    - Responsible for sorting schemas and states.
    - Each function handles a specific type of schema and ensures that the sorting is done correctly.

14. **Reaction Functions**:
    - Responsible for handling reactions within the schema and state.
    - Each function processes a specific type of reaction and ensures that the state is updated accordingly.

"""

import sys
import types
import copy
import numbers
import numpy as np
from pint import Quantity
from pprint import pformat as pf

import typing
from typing import NewType, Union, Mapping, List, Dict, Optional, Callable
from dataclasses import field, make_dataclass

from bigraph_schema import get_path, set_path
from bigraph_schema.units import units, render_units_type
from bigraph_schema.registry import (
    is_schema_key, non_schema_keys, type_parameter_key, deep_merge, hierarchy_depth, establish_path
)
from bigraph_schema.utilities import (
    is_empty, union_keys, tuple_from_type, array_shape, read_datatype, read_shape, remove_path,
    type_parameters_for, visit_method, NONE_SYMBOL
)


# Create a new module dynamically for the dataclasses
module_name = 'bigraph_schema.data'
if module_name not in sys.modules:
    data_module = types.ModuleType(module_name)
    sys.modules[module_name] = data_module
else:
    data_module = sys.modules[module_name]


# =========================
# Apply Functions Overview
# =========================
# These functions are responsible for applying updates to various types of schemas.
# Each function handles a specific type of schema and ensures that updates are applied correctly.
# Function signature: (schema, current, update, core)

def apply_any(schema, current, update, top_schema, top_state, path, core):
    if isinstance(current, dict):
        return apply_tree(
            current,
            update,
            'tree[any]',
            top_schema=top_schema,
            top_state=top_state,
            path=path,
            core=core)
    else:
        return update

def apply_tuple(schema, current, update, top_schema, top_state, path, core):
    parameters = core.parameters_for(schema)
    result = []

    for parameter, current_value, update_value in zip(parameters, current, update):
        element = core.apply_update(
            parameter,
            current_value,
            update_value,
            top_schema=top_schema,
            top_state=top_state,
            path=path)

        result.append(element)

    return tuple(result)

def apply_union(schema, current, update, top_schema, top_state, path, core):
    current_type = find_union_type(
        core,
        schema,
        current)

    update_type = find_union_type(
        core,
        schema,
        update)

    if current_type is None:
        raise Exception(f'trying to apply update to union value but cannot find type of value in the union\n  value: {current}\n  update: {update}\n  union: {list(bindings.values())}')
    elif update_type is None:
        raise Exception(f'trying to apply update to union value but cannot find type of update in the union\n  value: {current}\n  update: {update}\n  union: {list(bindings.values())}')

    # TODO: throw an exception if current_type is incompatible with update_type

    return core.apply_update(
        update_type,
        current,
        update,
        top_schema=top_schema,
        top_state=top_state,
        path=path)

def set_apply(schema, current, update, top_schema, top_state, path, core):
    if isinstance(current, dict) and isinstance(update, dict):
        for key, value in update.items():
            # TODO: replace this with type specific functions (??)
            if key in schema:
                subschema = schema[key]
            elif '_leaf' in schema:
                if core.check(schema['_leaf'], value):
                    subschema = schema['_leaf']
                else:
                    subschema = schema
            elif '_value' in schema:
                subschema = schema['_value']

            current[key] = set_apply(
                subschema,
                current.get(key),
                value,
                core)

        return current
    else:
        return update

def accumulate(schema, current, update, top_schema, top_state, path, core):
    if current is None:
        return update
    if update is None:
        return current
    else:
        return current + update

def concatenate(schema, current, update, top_schema, top_state, path, core=None):
    return current + update

def replace(schema, current, update, top_schema, top_state, path, core=None):
    return update

def apply_schema(schema, current, update, top_schema, top_state, path, core):
    """
    Apply an update to a schema, returning the new schema
    """
    outcome = core.resolve_schemas(current, update)
    return outcome

def apply_tree(schema, current, update, top_schema, top_state, path, core):
    leaf_type = core.find_parameter(
        schema,
        'leaf')

    if current is None:
        current = core.default(leaf_type)

    if isinstance(current, dict) and isinstance(update, dict):
        for key, branch in update.items():
            if key == '_add':
                current.update(branch)
            elif key == '_remove':
                for removed_path in branch:
                    if isinstance(removed_path, str):
                        removed_path = [removed_path]
                    current = remove_path(current, removed_path)
            elif isinstance(branch, dict):
                subschema = schema
                if key in schema:
                    subschema = schema[key]

                current[key] = core.apply_update(
                    subschema,
                    current.get(key),
                    branch,
                    top_schema=top_schema,
                    top_state=top_state,
                    path=path + [key])

            elif core.check(leaf_type, branch):
                current[key] = core.apply_update(
                    leaf_type,
                    current.get(key),
                    branch,
                    top_schema=top_schema,
                    top_state=top_state,
                    path=path + [key])

            else:
                raise Exception(f'state does not seem to be of leaf type:\n  state: {state}\n  leaf type: {leaf_type}')

        return current

    elif core.check(leaf_type, current):
        return core.apply_update(
            leaf_type,
            current,
            update,
            top_schema=top_schema,
            top_state=top_state,
            path=path)

    else:
        raise Exception(f'trying to apply an update to a tree but the values are not trees or leaves of that tree\ncurrent:\n  {pf(current)}\nupdate:\n  {pf(update)}\nschema:\n  {pf(schema)}')

def apply_boolean(schema, current: bool, update: bool, top_schema, top_state, path, core=None) -> bool:
    """Performs a bit flip if `current` does not match `update`, returning update. Returns current if they match."""
    if current != update:
        return update
    else:
        return current

def apply_list(schema, current, update, top_schema, top_state, path, core):
    element_type = core.find_parameter(
        schema,
        'element')

    if current is None:
        current = []

    if core.check(element_type, update):
        result = current + [update]
        return result

    elif isinstance(update, list):
        result = current + update
        # for current_element, update_element in zip(current, update):
        #     applied = core.apply(
        #         element_type,
        #         current_element,
        #         update_element)
        #     result.append(applied)

        return result
    else:
        raise Exception(f'trying to apply an update to an existing list, but the update is not a list or of element type:\n  update: {update}\n  element type: {pf(element_type)}')

def apply_map(schema, current, update, top_schema, top_state, path, core=None):
    if update is None:
        return current

    if not isinstance(current, dict):
        raise Exception(f'trying to apply an update to a value that is not a map:\n  value: {current}\n  update: {update}')
    if not isinstance(update, dict):
        raise Exception(f'trying to apply an update that is not a map:\n  value: {current}\n  update: {update}')

    value_type = core.find_parameter(
        schema,
        'value')

    result = current.copy()

    for key, update_value in update.items():
        if key == '_add':
            for addition_key, addition in update_value.items():

                _, generated_state, top_schema, top_state = core.generate_recur(
                    value_type,
                    addition,
                    top_schema=top_schema,
                    top_state=top_state,
                    path=path + [addition_key])

                result[addition_key] = generated_state

        elif key == '_remove':
            for remove_key in update_value:
                if remove_key in result:
                    del result[remove_key]

        elif key not in current:
            # This supports adding without the '_add' key, if the key is not in the state
            _, generated_state, top_schema, top_state = core.generate_recur(
                value_type,
                update_value,
                top_schema=top_schema,
                top_state=top_state,
                path=path + [key])

            result[key] = generated_state

            # generated_schema, generated_state = core.generate(
            #     value_type,
            #     update_value)

            # result[key] = generated_state

            # raise Exception(f'trying to update a key that does not exist:\n  value: {current}\n  update: {update}')
        else:
            result[key] = core.apply_update(
                value_type,
                result[key],
                update_value,
                top_schema=top_schema,
                top_state=top_state,
                path=path + [key])

    return result


def apply_maybe(schema, current, update, top_schema, top_state, path, core):
    if current is None or update is None:
        return update
    else:
        value_type = core.find_parameter(
            schema,
            'value')

        return core.apply_update(
            value_type,
            current,
            update,
            top_schema=top_schema,
            top_state=top_state,
            path=path)


def apply_path(schema, current, update, top_schema, top_state, path, core):
    # paths replace previous paths
    return update


def apply_edge(schema, current, update, top_schema, top_state, path, core):
    result = current.copy()
    result['inputs'] = core.apply_update(
        'wires',
        current.get('inputs'),
        update.get('inputs'),
        top_schema=top_schema,
        top_state=top_state,
        path=path)

    result['outputs'] = core.apply_update(
        'wires',
        current.get('outputs'),
        update.get('outputs'),
        top_schema=top_schema,
        top_state=top_state,
        path=path)

    return result

# TODO: deal with all the different unit core
def apply_units(schema, current, update, top_schema, top_state, path, core):
    return current + update

def apply_enum(schema, current, update, top_schema, top_state, path, core):
    parameters = core.parameters_for(schema)
    if update in parameters:
        return update
    else:
        raise Exception(f'{update} is not in the enum, options are: {parameters}')

def apply_array(schema, current, update, top_schema, top_state, path, core):
    if isinstance(update, dict):
        paths = hierarchy_depth(update)
        for path, inner_update in paths.items():
            if len(path) > len(schema['_shape']):
                raise Exception(f'index is too large for array update: {path}\n  {schema}')
            else:
                index = tuple(path)
                current[index] += inner_update

        return current
    else:
        return current + update


# =========================
# Check Functions Overview
# =========================
# These functions are responsible for validating the state against various types of schemas.
# Each function ensures that the state conforms to the expected schema type.
# Function signature: (schema, state, core)

def check_any(schema, state, core):
    if isinstance(schema, dict):
        for key, subschema in schema.items():
            if not key.startswith('_'):
                if isinstance(state, dict):
                    if key in state:
                        check = core.check_state(
                            subschema,
                            state[key])

                        if not check:
                            return False
                    else:
                        return False
                else:
                    return False

        return True
    else:
        return True

def check_tuple(schema, state, core):
    if not isinstance(state, (tuple, list)):
        return False

    parameters = core.parameters_for(schema)
    for parameter, element in zip(parameters, state):
        if not core.check(parameter, element):
            return False

    return True

def check_union(schema, state, core):
    found = find_union_type(
        core,
        schema,
        state)

    return found is not None and len(found) > 0

def check_number(schema, state, core=None):
    return isinstance(state, numbers.Number)

def check_boolean(schema, state, core=None):
    return isinstance(state, bool)

def check_integer(schema, state, core=None):
    return isinstance(state, int) and not isinstance(state, bool)

def check_float(schema, state, core=None):
    return isinstance(state, float)

def check_string(schema, state, core=None):
    return isinstance(state, str)

def check_list(schema, state, core):
    element_type = core.find_parameter(
        schema,
        'element')

    if isinstance(state, list):
        for element in state:
            check = core.check(
                element_type,
                element)

            if not check:
                return False

        return True
    else:
        return False

def check_tree(schema, state, core):
    leaf_type = core.find_parameter(
        schema,
        'leaf')

    if isinstance(state, dict):
        for key, value in state.items():
            check = core.check({
                '_type': 'tree',
                '_leaf': leaf_type},
                value)

            if not check:
                return core.check(
                    leaf_type,
                    value)

        return True
    else:
        return core.check(leaf_type, state)

def check_map(schema, state, core=None):
    value_type = core.find_parameter(
        schema,
        'value')

    if not isinstance(state, dict):
        return False

    for key, substate in state.items():
        if not core.check(value_type, substate):
            return False

    return True

def check_ports(state, core, key):
    return key in state and core.check(
        'wires',
        state[key])

def check_edge(schema, state, core):
    return isinstance(state, dict) and check_ports(state, core, 'inputs') and check_ports(state, core, 'outputs')

def check_maybe(schema, state, core):
    if state is None:
        return True
    else:
        value_type = core.find_parameter(
            schema,
            'value')

        return core.check(value_type, state)

def check_array(schema, state, core):
    shape_type = core.find_parameter(
        schema,
        'shape')

    return isinstance(state, np.ndarray) and state.shape == array_shape(core, shape_type) # and state.dtype == bindings['data'] # TODO align numpy data types so we can validate the types of the arrays

def check_enum(schema, state, core):
    if not isinstance(state, str):
        return False

    parameters = core.parameters_for(schema)
    return state in parameters

def check_units(schema, state, core):
    # TODO: expand this to check the actual units for compatibility
    return isinstance(state, Quantity)


# =========================
# Fold Functions Overview
# =========================
# These functions are responsible for folding the state based on the schema and a given method.
# Each function handles a specific type of schema and ensures that the folding is done correctly.
# In functional programming, a fold is a higher-order function that processes a data structure
# in some order and builds a return value.
# Function signature: (schema, state, method, values, core)

def fold_any(schema, state, method, values, core):
    if isinstance(state, dict):
        result = {}
        for key, value in state.items():
            if key.startswith('_'):
                result[key] = value
            else:
                if key in schema:
                    fold = core.fold_state(
                        schema[key],
                        value,
                        method,
                        values)
                    result[key] = fold

    else:
        result = state

    visit = visit_method(
        schema,
        result,
        method,
        values,
        core)

    return visit

def fold_tuple(schema, state, method, values, core):
    if not isinstance(state, (tuple, list)):
        return visit_method(
            schema,
            state,
            method,
            values,
            core)
    else:
        parameters = core.parameters_for(schema)
        result = []
        for parameter, element in zip(parameters, state):
            fold = core.fold(
                parameter,
                element,
                method,
                values)
            result.append(fold)

        result = tuple(result)

        return visit_method(
            schema,
            result,
            method,
            values,
            core)

def fold_union(schema, state, method, values, core):
    union_type = find_union_type(
        core,
        schema,
        state)

    result = core.fold(
        union_type,
        state,
        method,
        values)

    return result

def fold_list(schema, state, method, values, core):
    element_type = core.find_parameter(
        schema,
        'element')

    if core.check(element_type, state):
        result = core.fold(
            element_type,
            state,
            method,
            values)

    elif isinstance(state, list):
        subresult = [
            fold_list(
                schema,
                element,
                method,
                values,
                core)
            for element in state]

        result = visit_method(
            schema,
            subresult,
            method,
            values,
            core)

    else:
        raise Exception(f'state does not seem to be a list or an eelement:\n  state: {state}\n  schema: {schema}')

    return result

def fold_tree(schema, state, method, values, core):
    leaf_type = core.find_parameter(
        schema,
        'leaf')

    if core.check(leaf_type, state):
        result = core.fold(
            leaf_type,
            state,
            method,
            values)

    elif isinstance(state, dict):
        subresult = {}

        for key, branch in state.items():
            if key.startswith('_'):
                subresult[key] = branch
            else:
                subresult[key] = fold_tree(
                    schema[key] if key in schema else schema,
                    branch,
                    method,
                    values,
                    core)

        result = visit_method(
            schema,
            subresult,
            method,
            values,
            core)

    else:
        raise Exception(f'state does not seem to be a tree or a leaf:\n  state: {state}\n  schema: {schema}')

    return result

def fold_map(schema, state, method, values, core):
    value_type = core.find_parameter(
        schema,
        'value')

    subresult = {}

    for key, value in state.items():
        subresult[key] = core.fold(
            value_type,
            value,
            method,
            values)

    result = visit_method(
        schema,
        subresult,
        method,
        values,
        core)

    return result

def fold_maybe(schema, state, method, values, core):
    value_type = core.find_parameter(
        schema,
        'value')

    if state is None:
        result = core.fold(
            'any',
            state,
            method,
            values)

    else:
        result = core.fold(
            value_type,
            state,
            method,
            values)

    return result

def fold_enum(schema, state, method, values, core):
    if not isinstance(state, (tuple, list)):
        return visit_method(
            schema,
            state,
            method,
            values,
            core)
    else:
        parameters = core.parameters_for(schema)
        result = []
        for parameter, element in zip(parameters, state):
            fold = core.fold(
                parameter,
                element,
                method,
                values)
            result.append(fold)

        result = tuple(result)

        return visit_method(
            schema,
            result,
            method,
            values,
            core)


# ==========================
# Divide Functions Overview
# ==========================
# These functions are responsible for dividing the state into a number of parts based on the schema.
# Each function handles a specific type of schema and divides the state accordingly.
# Function signature: (schema, state, values, core)

def divide_any(schema, state, values, core):
    divisions = values.get('divisions', 2)

    if isinstance(state, dict):
        result = [
            {}
            for _ in range(divisions)]

        for key, value in state.items():
            for index in range(divisions):
                result[index][key] = value[index]

        return result

    else:
        # TODO: division operates on and returns dictionaries
#         return {
#             id: copy.deepcopy(state),
#             for generate_new_id(existing_id, division) in range(divisions)}
# ?????

        return [
            copy.deepcopy(state)
            for _ in range(divisions)]

def divide_tuple(schema, state, values, core):
    divisions = values.get('divisions', 2)

    return [
        tuple([item[index] for item in state])
        for index in range(divisions)]

def divide_float(schema, state, values, core):
    divisions = values.get('divisions', 2)
    portion = float(state) / divisions
    return [
        portion
        for _ in range(divisions)]

# support function core for registries?
def divide_integer(schema, value, values, core):
    half = value // 2
    other_half = half
    if value % 2 == 1:
        other_half += 1
    return [half, other_half]

def divide_longest(schema, dimensions, values, core):
    # any way to declare the required keys for this function in the registry?
    # find a way to ask a function what type its domain and codomain are

    width = dimensions['width']
    height = dimensions['height']

    if width > height:
        a, b = divide_integer(width)
        return [{'width': a, 'height': height}, {'width': b, 'height': height}]
    else:
        x, y = divide_integer(height)
        return [{'width': width, 'height': x}, {'width': width, 'height': y}]

def divide_reaction(schema, state, reaction, core):
    mother = reaction['mother']
    daughters = reaction['daughters']

    mother_schema, mother_state = core.slice(
        schema,
        state,
        mother)

    division = core.fold(
        mother_schema,
        mother_state,
        'divide', {
            'divisions': len(daughters),
            'daughter_configs': [daughter[1] for daughter in daughters]})

    after = {
        daughter[0]: daughter_state
        for daughter, daughter_state in zip(daughters, division)}

    replace = {
        'before': {
            mother: {}},
        'after': after}

    return replace_reaction(
        schema,
        state,
        replace,
        core)

def divide_list(schema, state, values, core):
    element_type = core.find_parameter(
        schema,
        'element')

    if core.check(element_type, state):
        return core.fold(
            element_type,
            state,
            'divide',
            values)

    elif isinstance(state, list):
        divisions = values.get('divisions', 2)
        result = [[] for _ in range(divisions)]

        for elements in state:
            for index in range(divisions):
                result[index].append(
                    elements[index])

        return result

    else:
        raise Exception(
            f'trying to divide list but state does not resemble a list or an element.\n  state: {pf(state)}\n  schema: {pf(schema)}')

def divide_tree(schema, state, values, core):
    leaf_type = core.find_parameter(
        schema,
        'leaf')

    if core.check(leaf_type, state):
        return core.fold(
            leaf_type,
            state,
            'divide',
            values)

    elif isinstance(state, dict):
        divisions = values.get('divisions', 2)
        division = [{} for _ in range(divisions)]

        for key, value in state.items():
            for index in range(divisions):
                division[index][key] = value[index]

        return division

    else:
        raise Exception(
            f'trying to divide tree but state does not resemble a leaf or a tree.\n  state: {pf(state)}\n  schema: {pf(schema)}')

def divide_map(schema, state, values, core):
    if isinstance(state, dict):
        divisions = values.get('divisions', 2)
        division = [{} for _ in range(divisions)]
        for key, value in state.items():
            for index in range(divisions):
                division[index][key] = value[index]

        return division
    else:
        raise Exception(
            f'trying to divide a map but state is not a dict.\n  state: {pf(state)}\n  schema: {pf(schema)}')

def divide_enum(schema, state, values, core):
    divisions = values.get('divisions', 2)

    return [
        tuple([item[index] for item in state])
        for index in range(divisions)]


# =============================
# Serialize Functions Overview
# =============================
# These functions are responsible for converting the state into a serializable format based on the schema.
# Each function handles a specific type of schema and ensures that the state is serialized correctly.
# Function signature: (schema, state, core)

def serialize_any(schema, state, core):
    if isinstance(state, dict):
        tree = {}

        for key in non_schema_keys(schema):
            encoded = core.serialize(
                schema.get(key, schema),
                state.get(key))
            tree[key] = encoded

        return tree

    else:
        return str(state)

def serialize_tuple(schema, value, core):
    parameters = core.parameters_for(schema)
    result = []

    for parameter, element in zip(parameters, value):
        encoded = core.serialize(
            parameter,
            element)

        result.append(encoded)

    return tuple(result)

def serialize_union(schema, value, core):
    union_type = find_union_type(
        core,
        schema,
        value)

    return core.serialize(
        union_type,
        value)

def serialize_string(schema, value, core=None):
    return value

def serialize_boolean(schema, value: bool, core) -> str:
    return str(value)

def serialize_list(schema, value, core=None):
    element_type = core.find_parameter(
        schema,
        'element')

    return [
        core.serialize(
            element_type,
            element)
        for element in value]

def serialize_tree(schema, value, core):
    if isinstance(value, dict):
        encoded = {}
        for key, subvalue in value.items():
            encoded[key] = serialize_tree(
                schema,
                subvalue,
                core)

    else:
        leaf_type = core.find_parameter(
            schema,
            'leaf')

        if core.check(leaf_type, value):
            encoded = core.serialize(
                leaf_type,
                value)
        else:
            raise Exception(f'trying to serialize a tree but unfamiliar with this form of tree: {value} - current schema:\n {pf(schema)}')

    return encoded

def serialize_units(schema, value, core):
    return str(value)

def serialize_maybe(schema, value, core):
    if value is None:
        return NONE_SYMBOL
    else:
        value_type = core.find_parameter(
            schema,
            'value')

        return core.serialize(
            value_type,
            value)

def serialize_map(schema, value, core=None):
    value_type = core.find_parameter(
        schema,
        'value')

    return {
        key: core.serialize(
            value_type,
            subvalue) if not is_schema_key(key) else subvalue
        for key, subvalue in value.items()}

def serialize_edge(schema, value, core):
    return value

def serialize_enum(schema, value, core):
    return value

def recur_serialize_schema(schema, core, path=None, parents=None):
    """ Serialize schema to a string """
    path = path or []
    parents = parents or []
    schema_id = id(schema)

    if schema_id in parents:
        index = parents.index(schema_id)
        reference = path[:index]
        output = '/'.join(reference)
        return f'/{output}'

    if isinstance(schema, str):
        return schema

    elif isinstance(schema, tuple):
        inner = [
            recur_serialize_schema(
                schema=element,
                core=core,
                path=path+[index],
                parents=parents+[schema_id])
            for index, element in enumerate(schema)]

        return inner

    elif isinstance(schema, dict):
        inner = {}
        for key in schema:
            subschema = recur_serialize_schema(
                schema=schema[key],
                core=core,
                path=path+[key],
                parents=parents+[schema_id])
            inner[key] = subschema

        return inner

    else:
        return schema

def serialize_schema(schema, state, core):
    """ Serialize schema to a string """
    return recur_serialize_schema(schema=state, core=core)

def serialize_array(schema, value, core):
    """ Serialize numpy array to list """

    if isinstance(value, dict):
        return value
    elif isinstance(value, str):
        import ipdb; ipdb.set_trace()
    else:
        array_data = 'string'
        dtype = value.dtype.name
        if dtype.startswith('int'):
            array_data = 'integer'
        elif dtype.startswith('float'):
            array_data = 'float'

        return {
            'list': value.tolist(),
            'data': array_data,
            'shape': list(value.shape)}


# ===============================
# Deserialize Functions Overview
# ===============================
# These functions are responsible for converting serialized data back into the state based on the schema.
# Each function handles a specific type of schema and ensures that the data is deserialized correctly.
# Function signature: (schema, state, core)

def to_string(schema, value, core=None):
    return str(value)

# def evaluate(schema, encoded, core=None):
#     return eval(encoded)

def deserialize_any(schema, state, core):
    if isinstance(state, dict):
        tree = {}

        for key, value in state.items():
            if is_schema_key(key):
                decoded = value
            else:
                decoded = core.deserialize(
                    schema.get(key, 'any'),
                    value)

            tree[key] = decoded

        for key in non_schema_keys(schema):
            if key not in tree:
                # if key not in state:
                #     decoded = core.default(
                #         schema[key])
                # else:
                if key in state:
                    decoded = core.deserialize(
                        schema[key],
                        state[key])

                    tree[key] = decoded

        return tree

    else:
        return state

def deserialize_tuple(schema, state, core):
    parameters = core.parameters_for(schema)
    result = []

    if isinstance(state, str):
        if (state[0] == '(' and state[-1] == ')') or (state[0] == '[' and state[-1] == ']'):
            state = state[1:-1].split(',')
        else:
            return None

    for parameter, code in zip(parameters, state):
        element = core.deserialize(
            parameter,
            code)

        result.append(element)

    return tuple(result)

def deserialize_union(schema, encoded, core):
    if encoded == NONE_SYMBOL:
        return None
    else:
        parameters = core.parameters_for(schema)

        for parameter in parameters:
            value = core.deserialize(
                parameter,
                encoded)

            if value is not None:
                return value

def deserialize_string(schema, encoded, core=None):
    if isinstance(encoded, str):
        return encoded

def deserialize_integer(schema, encoded, core=None):
    value = None
    try:
        value = int(encoded)
    except:
        pass

    return value

def deserialize_float(schema, encoded, core=None):
    value = None
    try:
        value = float(encoded)
    except:
        pass

    return value

def deserialize_list(schema, encoded, core=None):
    if isinstance(encoded, list):
        element_type = core.find_parameter(
            schema,
            'element')

        return [
            core.deserialize(
                element_type,
                element)
            for element in encoded]

def deserialize_maybe(schema, encoded, core):
    if encoded == NONE_SYMBOL or encoded is None:
        return None
    else:
        value_type = core.find_parameter(
            schema,
            'value')

        return core.deserialize(value_type, encoded)

def deserialize_quote(schema, state, core):
    return state

def deserialize_boolean(schema, encoded, core) -> bool:
    if encoded == 'true':
        return True
    elif encoded == 'false':
        return False
    elif encoded == True or encoded == False:
        return encoded

def deserialize_tree(schema, encoded, core):
    if isinstance(encoded, dict):
        tree = {}
        for key, value in encoded.items():
            if key.startswith('_'):
                tree[key] = value
            else:
                tree[key] = deserialize_tree(schema, value, core)

        return tree

    else:
        leaf_type = core.find_parameter(
            schema,
            'leaf')

        if leaf_type:
            return core.deserialize(
                leaf_type,
                encoded)
        else:
            return encoded

def deserialize_units(schema, encoded, core):
    if isinstance(encoded, Quantity):
        return encoded
    else:
        return units(encoded)

def deserialize_map(schema, encoded, core=None):
    if isinstance(encoded, dict):
        value_type = core.find_parameter(
            schema,
            'value')

        return {
            key: core.deserialize(
                value_type,
                subvalue) if not is_schema_key(key) else subvalue
            for key, subvalue in encoded.items()}


def enum_list(enum_schema):
    return [
        enum_schema[f'_{parameter}']
        for parameter in enum_schema['_type_parameters']]


def deserialize_enum(schema, state, core):
    enum = enum_list(schema)
    if state in enum:
        return state
    else:
        raise Exception(f'{state} not in enum: {enum}')


def deserialize_array(schema, encoded, core):
    if isinstance(encoded, np.ndarray):
        return encoded

    elif isinstance(encoded, dict):
        if 'value' in encoded:
            return encoded['value']
        else:
            found = core.retrieve(
                encoded.get(
                    'data',
                    schema['_data']))

            dtype = read_datatype(
                found)

            shape = read_shape(
                schema['_shape'])

            if 'list' in encoded:
                return np.array(
                    encoded['list'],
                    dtype=dtype).reshape(
                        shape)
            else:
                return np.zeros(
                    tuple(shape),
                    dtype=dtype)

def deserialize_edge(schema, encoded, core):
    return encoded

def recur_deserialize_schema(schema, core, top_state=None, path=None):
    top_state = top_state or schema
    path = path or []

    if isinstance(schema, dict):
        subschema = {}
        for key, value in schema.items():
            subschema[key] = recur_deserialize_schema(
                value,
                core,
                top_state=top_state,
                path=path+[key])

        return subschema

    elif isinstance(schema, list):
        subschema = []
        for index, value in enumerate(schema):
            subschema.append(
                recur_deserialize_schema(
                    value,
                    core,
                    top_state=top_state,
                    path=path+[index]))

        return tuple(subschema)

    elif isinstance(schema, str):
        if schema.startswith('/'):  # this is a reference to another schema
            local_path = schema.split('/')[1:]
            reference = get_path(top_state, local_path)

            set_path(
                tree=top_state,
                path=path,
                value=reference)

            return reference
        else:
            return schema
    else:
        return schema


def deserialize_schema(schema, state, core):
    return recur_deserialize_schema(schema=state, core=core)


# =========================
# Slice Functions Overview
# =========================
# These functions are responsible for extracting a part of the state based on the schema and path.
# Each function handles a specific type of schema and ensures that the correct part of the state is sliced.
# Function signature: (schema, state, path, core)

def slice_any(schema, state, path, core):
    if not isinstance(path, (list, tuple)):
        if path is None:
            path = ()
        else:
            path = [path]

    if len(path) == 0:
        return schema, state

    elif len(path) > 0:
        head = path[0]
        tail = path[1:]
        step = None

        if isinstance(state, dict):
            if head == '*':
                step_schema = {}
                step_state = {}

                for key, value in state.items():
                    if key in schema:
                        step_schema[key], step_state[key] = core.slice(
                            schema[key],
                            value,
                            tail)

                    else:
                        step_schema[key], step_state[key] = slice_any(
                            {},
                            value,
                            tail,
                            core)

                return step_schema, step_state

            elif head not in state:
                state[head] = core.default(
                    schema.get(head))

            step = state[head]

        elif isinstance(head, str) and hasattr(state, head):
            step = getattr(state, head)

        if head in schema:
            return core.slice(
                schema[head],
                step,
                tail)
        else:
            return slice_any(
                {},
                step,
                tail,
                core)

def slice_tuple(schema, state, path, core):
    if len(path) > 0:
        head = path[0]
        tail = path[1:]

        if head == '*':
            result_schema = {}
            result_state = {}
            for index, position in enumerate(schema['_type_parameters']):
                result_schema[position], result_state[position] = core.slice(
                    schema[position],
                    state[index],
                    tail)
            return result_schema, result_state
        elif str(head) in schema['_type_parameters']:
            try:
                index = schema['_type_parameters'].index(str(head))
            except:
                raise Exception(f'step {head} in path {path} is not a type parameter of\n  schema: {pf(schema)}\n  state: {pf(state)}')
            index_key = f'_{index}'
            subschema = core.access(schema[index_key])

            return core.slice(subschema, state[head], tail)
        else:
            raise Exception(f'trying to index a tuple with a key that is not an index: {state} {head}')
    else:
        return schema, state

def slice_union(schema, state, path, core):
    union_type = find_union_type(
        core,
        schema,
        state)

    return core.slice(
        union_type,
        state,
        path)

def slice_list(schema, state, path, core):
    element_type = core.find_parameter(
        schema,
        'element')

    if len(path) > 0:
        head = path[0]
        tail = path[1:]

        if not isinstance(head, int) or head >= len(state):
            raise Exception(f'bad index for list: {path} for {state}')

        step = state[head]
        return core.slice(element_type, step, tail)
    else:
        return schema, state

def slice_tree(schema, state, path, core):
    leaf_type = core.find_parameter(
        schema,
        'leaf')

    if len(path) > 0:
        head = path[0]
        tail = path[1:]

        if head == '*':
            slice_schema = {}
            slice_state = {}
            for key, value in state.items():
                if core.check(leaf_type, value):
                    slice_schema[key], slice_state[key] = core.slice(
                        leaf_type,
                        value,
                        tail)
                else:
                    slice_schema[key], slice_state[key] = core.slice(
                        schema,
                        value,
                        tail)

            return slice_schema, slice_state

        if not state:
            default = core.default(
                leaf_type)
            try:
                down_schema, down_state = core.slice(
                    leaf_type,
                    default,
                    path)

                if down_state:
                    return down_schema, down_state
            except:
                state = {}
        if not head in state:
            state[head] = {}

        step = state[head]
        if core.check(leaf_type, step):
            return core.slice(leaf_type, step, tail)
        else:
            return core.slice(schema, step, tail)
    else:
        return schema, state


def slice_edge(schema, state, path, core):
    if len(path) > 0:
        head = path[0]
        tail = path[1:]

        if head == '_inputs' or head == '_outputs':
            pass

        return slice_any(schema, state, path, core)
    else:
        return schema, state
    

def slice_map(schema, state, path, core):
    value_type = core.find_parameter(
        schema,
        'value')

    if len(path) > 0:
        head = path[0]
        tail = path[1:]

        if head == '*':
            slice_schema = {'_type': 'map'}
            slice_state = {}

            for key, value in state.items():
                tail_schema, slice_state[key] = core.slice(
                    value_type,
                    value,
                    tail)

                if not '_value' in slice_schema:
                    slice_schema['_value'] = tail_schema
                else:
                    slice_schema['_value'] = core.resolve_schemas(
                        slice_schema['_value'],
                        tail_schema)

            return slice_schema, slice_state

        if not head in state:
            state[head] = core.default(
                value_type)

        step = state[head]
        return core.slice(
            value_type,
            step,
            tail)
    else:
        return schema, state


def slice_maybe(schema, state, path, core):
    if state is None:
        return schema, None

    else:
        value_type = core.find_parameter(
            schema,
            'value')

        return core.slice(
            value_type,
            state,
            path)


def slice_array(schema, state, path, core):
    if len(path) > 0:
        head = path[0]
        tail = path[1:]
        if isinstance(head, str):
            head = int(head)
        step = state[head]

        if isinstance(step, np.ndarray):
            sliceschema = schema.copy()
            sliceschema['_shape'] = step.shape
            return core.slice(
                sliceschema,
                step,
                tail)
        else:
            data_type = core.find_parameter(
                schema,
                'data')

            return core.slice(
                data_type,
                step,
                tail)

    else:
        return schema, state


def slice_string(schema, state, path, core):
    raise Exception(f'cannot slice into an string: {path}\n{state}\n{schema}')



# ========================
# Bind Functions Overview
# ========================
# These functions are responsible for binding a key and its corresponding schema and state to the main schema and state.
# Each function handles a specific type of schema and ensures that the binding is done correctly.
# Function signature: (schema, state, key, subschema, substate, core)

def bind_any(schema, state, key, subschema, substate, core):
    result_schema = core.resolve_schemas(
          schema,
          {key: subschema})

    if state is None:
        state = {}

    state[key] = substate

    return result_schema, state

def bind_tuple(schema, state, key, subschema, substate, core):
    new_schema = schema.copy()
    new_schema[f'_{key}'] = subschema
    open = list(state)
    open[key] = substate

    return new_schema, tuple(open)

def bind_union(schema, state, key, subschema, substate, core):
    union_type = find_union_type(
        core,
        schema,
        state)

    return core.bind(
        union_type,
        state,
        key,
        subschema,
        substate)

def bind_enum(schema, state, key, subschema, substate, core):
    new_schema = schema.copy()
    new_schema[f'_{key}'] = subschema
    open = list(state)
    open[key] = substate

    return new_schema, tuple(open)

def bind_array(schema, state, key, subschema, substate, core):
    if state is None:
        state = core.default(schema)
    if isinstance(key, str):
        key = int(key)
    state[key] = substate

    return schema, state


# ==========================
# Resolve Functions Overview
# ==========================
# These functions are responsible for resolving updates to the schema.
# Each function handles a specific type of schema and ensures that updates are resolved correctly.
# Function signature: (schema, update, core)

def resolve_maybe(schema, update, core):
    value_schema = core.find_parameter(
        schema,
        'value')

    inner_value = core.resolve_schemas(
        value_schema,
        update)

    schema['_value'] = inner_value


def resolve_map(schema, update, core):
    if isinstance(update, dict):
        value_schema = update.get(
            '_value',
            schema.get('_value', {}))

        for key, subschema in update.items():
            if not is_schema_key(key):
                value_schema = core.resolve_schemas(
                    value_schema,
                    subschema)

        schema['_type'] = update.get(
            '_type',
            schema.get('_type', 'map'))
        schema['_value'] = value_schema

    return schema

def resolve_array(schema, update, core):
    if not '_shape' in schema:
        schema = core.access(schema)
    if not '_shape' in schema:
        raise Exception(f'array must have a "_shape" key, not {schema}')

    data_schema = schema.get('_data', {})

    if '_type' in update:
        data_schema = core.resolve_schemas(
            data_schema,
            update.get('_data', {}))

        if update['_type'] == 'array':
            if '_shape' in update:
                if update['_shape'] != schema['_shape']:
                    raise Exception(f'arrays must be of the same shape, not \n  {schema}\nand\n  {update}')

        elif core.inherits_from(update, schema):
            schema.update(update)

        elif not core.inherits_from(schema, update):
            raise Exception(f'cannot resolve incompatible array schemas:\n  {schema}\n  {update}')

    else:
        for key, subschema in update.items():
            if isinstance(key, int):
                key = (key,)

            if len(key) > len(schema['_shape']):
                raise Exception(f'key is longer than array dimension: {key}\n{schema}\n{update}')
            elif len(key) == len(schema['_shape']):
                data_schema = core.resolve_schemas(
                    data_schema,
                    subschema)
            else:
                shape = tuple_from_type(
                    schema['_shape'])

                subshape = shape[len(key):]
                inner_schema = schema.copy()
                inner_schema['_shape'] = subshape
                inner_schema = core.resolve_schemas(
                    inner_schema,
                    subschema)

                data_schema = inner_schema['_data']

    schema['_data'] = data_schema

    return schema


# ============================
# Dataclass Functions Overview
# ============================
# These functions are responsible for generating dataclass representations of various types of schemas.
# Each function handles a specific type of schema and ensures that the dataclass is generated correctly.
# Function signature: (schema, path, core)

def dataclass_any(schema, path, core):
    parts = path
    if not parts:
        parts = ['top']
    dataclass_name = '_'.join(parts)

    if isinstance(schema, dict):
        type_name = schema.get('_type', 'any')

        branches = {}
        for key, subschema in schema.items():
            if not key.startswith('_'):
                branch = core.dataclass(
                    subschema,
                    path + [key])

                def default(subschema=subschema):
                    return core.default(subschema)

                branches[key] = (
                    key,
                    branch,
                    field(default_factory=default))

        dataclass = make_dataclass(
            dataclass_name,
            branches.values(),
            namespace={
                '__module__': 'bigraph_schema.data'})

        setattr(
            data_module,
            dataclass_name,
            dataclass)

    else:
        schema = core.access(schema)
        dataclass = core.dataclass(schema, path)

    return dataclass

def dataclass_tuple(schema, path, core):
    parameters = type_parameters_for(schema)
    subtypes = []

    for index, key in enumerate(schema['type_parameters']):
        subschema = schema.get(key, 'any')
        subtype = core.dataclass(
            subschema,
            path + [index])

        subtypes.append(subtype)

    parameter_block = ', '.join(subtypes)
    return eval(f'tuple[{parameter_block}]')

def dataclass_union(schema, path, core):
    parameters = type_parameters_for(schema)
    subtypes = []
    for parameter in parameters:
        dataclass = core.dataclass(
            parameter,
            path)

        if isinstance(dataclass, str):
            subtypes.append(dataclass)
        elif isinstance(dataclass, type):
            subtypes.append(dataclass.__name__)
        else:
            subtypes.append(str(dataclass))

    parameter_block = ', '.join(subtypes)
    return eval(f'Union[{parameter_block}]')

def dataclass_float(schema, path, core):
    return float

def dataclass_integer(schema, path, core):
    return int

def dataclass_list(schema, path, core):
    element_type = core.find_parameter(
        schema,
        'element')

    dataclass = core.dataclass(
        element_type,
        path + ['element'])

    return list[dataclass]

def dataclass_tree(schema, path, core):
    leaf_type = core.find_parameter(schema, 'leaf')
    leaf_dataclass = core.dataclass(leaf_type, path + ['leaf'])

    dataclass_name = '_'.join(path)
    block = f"NewType('{dataclass_name}', Union[{leaf_dataclass}, Mapping[str, '{dataclass_name}']])"

    dataclass = eval(block, {
        'typing': typing,  # Add typing to the context
        'NewType': NewType,
        'Union': Union,
        'Mapping': Mapping,
        'List': List,
        'Dict': Dict,
        'Optional': Optional,
        'str': str
    })
    setattr(data_module, dataclass_name, dataclass)

    return dataclass

def dataclass_map(schema, path, core):
    value_type = core.find_parameter(
        schema,
        'value')

    dataclass = core.dataclass(
        value_type,
        path + ['value'])

    return Mapping[str, dataclass]

def dataclass_maybe(schema, path, core):
    value_type = core.find_parameter(
        schema,
        'value')

    dataclass = core.dataclass(
        value_type,
        path + ['value'])

    return Optional[dataclass]

def dataclass_edge(schema, path, core):
    inputs = schema.get('_inputs', {})
    inputs_dataclass = core.dataclass(
        inputs,
        path + ['inputs'])

    outputs = schema.get('_outputs', {})
    outputs_dataclass = core.dataclass(
        outputs,
        path + ['outputs'])

    return Callable[[inputs_dataclass], outputs_dataclass]

def dataclass_boolean(schema, path, core):
    return bool

def dataclass_string(schema, path, core):
    return str

def dataclass_enum(schema, path, core):
    parameters = type_parameters_for(schema)
    subtypes = []

    for index, key in enumerate(schema['type_parameters']):
        subschema = schema.get(key, 'any')
        subtype = core.dataclass(
            subschema,
            path + [index])

        subtypes.append(subtype)

    parameter_block = ', '.join(subtypes)
    return eval(f'tuple[{parameter_block}]')

def dataclass_array(schema, path, core):
    return np.ndarray


# ===========================
# Default Functions Overview
# ===========================
# These functions are responsible for providing default values for various types of schemas.
# Each function handles a specific type of schema and ensures that the default value is generated correctly.
# Absent a default function, the type could provide a default value directly.

def default_any(schema, core):
    default = {}

    for key, subschema in schema.items():
        if not is_schema_key(key):
            default[key] = core.default(
                subschema)

    return default

def default_tuple(schema, core):
    parts = []
    for parameter in schema['_type_parameters']:
        subschema = schema[f'_{parameter}']
        part = core.default(subschema)
        parts.append(part)

    return tuple(parts)

def default_union(schema, core):
    final_parameter = schema['_type_parameters'][-1]
    subschema = schema[f'_{final_parameter}']

    return core.default(subschema)

def default_tree(schema, core):
    leaf_schema = core.find_parameter(
        schema,
        'leaf')

    default = {}

    non_schema_keys = [
        key
        for key in schema
        if not is_schema_key(key)]

    if non_schema_keys:
        base_schema = {
            key: subschema
            for key, subschema in schema.items()
            if is_schema_key(key)}

        for key in non_schema_keys:
            subschema = core.merge_schemas(
                base_schema,
                schema[key])

            subdefault = core.default(
                subschema)

            if subdefault:
                default[key] = subdefault

    return default

def default_array(schema, core):
    data_schema = core.find_parameter(
        schema,
        'data')

    dtype = read_datatype(
        data_schema)

    shape = read_shape(
        schema['_shape'])

    return np.zeros(
        shape,
        dtype=dtype)

def default_enum(schema, core):
    parameter = schema['_type_parameters'][0]
    return schema[f'_{parameter}']

def default_edge(schema, core):
    edge = {}
    for key in schema:
        if not is_schema_key(key):
            edge[key] = core.default(
                schema[key])

    return edge


# ============================
# Generate Functions Overview
# ============================
# These functions are responsible for generating schemas and states based on the provided schema and state.
# Each function handles a specific type of schema and ensures that the generation is done correctly.

def generate_any(core, schema, state, top_schema=None, top_state=None, path=None):
    schema = schema or {}
    if is_empty(state):
        state = core.default(schema)
    top_schema = top_schema or schema
    top_state = top_state or state
    path = path or []

    generated_schema = {}
    generated_state = {}

    if isinstance(state, dict):
        visited = set([])

        all_keys = union_keys(
            schema,
            state)

        non_schema_keys = [
            key
            for key in all_keys
            if not is_schema_key(key)]

        for key in all_keys:
            if is_schema_key(key):
                generated_schema[key] = state.get(
                    key,
                    schema.get(key))

            else:
                subschema, substate, top_schema, top_state = core.generate_recur(
                    schema.get(key),
                    state.get(key),
                    top_schema=top_schema,
                    top_state=top_state,
                    path=path+[key])

                generated_schema[key] = core.resolve_schemas(
                    schema.get(key, {}),
                    subschema)

                generated_state[key] = substate

    else:
        if not core.check(schema, state):
            deserialized_state = core.deserialize(schema, state)
            if core.check(schema, deserialized_state):
                state = deserialized_state
            else:
                raise Exception(f'cannot generate {state} as {schema}')
        generated_schema, generated_state = schema, state

    if path:
        top_schema, top_state = core.set_slice(
            top_schema,
            top_state,
            path,
            generated_schema,
            generated_state)
    else:
        top_state = core.merge_recur(
            top_schema,
            top_state,
            generated_state)

    return generated_schema, generated_state, top_schema, top_state

def generate_quote(core, schema, state, top_schema=None, top_state=None, path=None):
    return schema, state, top_schema, top_state


def default_quote(schema, core):
    if '_default' in schema:
        return copy.deepcopy(schema['_default'])
    else:
        return None


def generate_map(core, schema, state, top_schema=None, top_state=None, path=None):
    schema = schema or {}
    state = state or core.default(schema)
    top_schema = top_schema or schema
    top_state = top_state or state
    path = path or []

    value_type = core.find_parameter(
        schema,
        'value')

    # TODO: can we assume this was already sorted at the top level?
    generated_schema, generated_state = core.sort(
        schema,
        state)

    try:
        all_keys = union_keys(schema, state)  # set(schema.keys()).union(state.keys())
    except Exception as e:
        # provide the path at which the error occurred
        raise Exception(
            f"Error at path {path}:\n"
            f"Expected schema: {core.representation(schema)}\n"
            f"Provided state: {state}") from e

    for key in all_keys:
        if is_schema_key(key):
            generated_schema[key] = state.get(
                key,
                schema.get(key))

        else:
            subschema = schema.get(key, value_type)
            substate = state.get(key)

            subschema = core.merge_schemas(
                value_type,
                subschema)

            subschema, generated_state[key], top_schema, top_state = core.generate_recur(
                subschema,
                substate,
                top_schema=top_schema,
                top_state=top_state,
                path=path + [key])

    return generated_schema, generated_state, top_schema, top_state

def generate_tree(core, schema, state, top_schema=None, top_state=None, path=None):
    schema = schema or {}
    state = state or core.default(schema)
    top_schema = top_schema or schema
    top_state = top_state or state
    path = path or []

    leaf_type = core.find_parameter(
        schema,
        'leaf')

    leaf_is_any = leaf_type == 'any' or (isinstance(leaf_type, dict) and leaf_type.get('_type') == 'any')

    if not leaf_is_any and core.check(leaf_type, state):
        generate_schema, generate_state, top_schema, top_state = core.generate_recur(
            leaf_type,
            state,
            top_schema=top_schema,
            top_state=top_state,
            path=path)

    elif isinstance(state, dict):
        generate_schema = {}
        generate_state = {}

        all_keys = union_keys(schema, state)  # set(schema.keys()).union(state.keys())
        non_schema_keys = [
            key
            for key in all_keys
            if not is_schema_key(key)]

        if non_schema_keys:
            base_schema = {
                key: subschema
                for key, subschema in schema.items()
                if is_schema_key(key)}
        else:
            base_schema = schema

        for key in all_keys:
            if not is_schema_key(key):
                subschema = schema.get(key)
                substate = state.get(key)

                if not substate or core.check(leaf_type, substate):
                    base_schema = leaf_type

                subschema = core.merge_schemas(
                    base_schema,
                    subschema)

                subschema, generate_state[key], top_schema, top_state = core.generate_recur(
                    subschema,
                    substate,
                    top_schema=top_schema,
                    top_state=top_state,
                    path=path + [key])

            elif key in state:
                generate_schema[key] = state[key]
            elif key in schema:
                generate_schema[key] = schema[key]
            else:
                raise Exception('the impossible has occurred now is the time for celebration')
    else:
        generate_schema, generate_state, top_schema, top_state = core.generate_recur(
            leaf_type,
            state,
            top_schema=top_schema,
            top_state=top_state,
            path=path)

    return generate_schema, generate_state, top_schema, top_state


def generate_ports(core, schema, wires, top_schema=None, top_state=None, path=None):
    schema = schema or {}
    wires = wires or {}
    top_schema = top_schema or schema
    top_state = top_state or {}
    path = path or []

    if isinstance(schema, str):
        schema = {'_type': schema}

    for port_key, subwires in wires.items():
        if port_key in schema:
            port_schema = schema[port_key]
        else:
            port_schema, subwires = core.slice(
                schema,
                wires,
                port_key)

        if isinstance(subwires, dict):
            top_schema, top_state = generate_ports(
                core,
                port_schema,
                subwires,
                top_schema=top_schema,
                top_state=top_state,
                path=path)

        else:
            if isinstance(subwires, str):
                subwires = [subwires]

            default_state = core.default(
                port_schema)

            top_schema, top_state = core.set_slice(
                top_schema,
                top_state,
                path[:-1] + subwires,
                port_schema,
                default_state,
                defer=True)

    return top_schema, top_state

def generate_edge(core, schema, state, top_schema=None, top_state=None, path=None):
    schema = schema or {}
    state = state or {}
    top_schema = top_schema or schema
    top_state = top_state or state
    path = path or []

    generated_schema, generated_state, top_schema, top_state = generate_any(
        core,
        schema,
        state,
        top_schema=top_schema,
        top_state=top_state,
        path=path)

    deserialized_state = core.deserialize(
        generated_schema,
        generated_state)

    merged_schema, merged_state = core.sort(
        generated_schema,
        deserialized_state)

    top_schema, top_state = core.set_slice(
        top_schema,
        top_state,
        path,
        merged_schema,
        merged_state)

    for port_key in ['inputs', 'outputs']:
        port_schema = merged_schema.get(
            f'_{port_key}', {})
        ports = merged_state.get(
            port_key, {})

        top_schema, top_state = generate_ports(
            core,
            port_schema,
            ports,
            top_schema=top_schema,
            top_state=top_state,
            path=path)

    return merged_schema, merged_state, top_schema, top_state


# =========================
# Sort Functions Overview
# =========================
# These functions are responsible for sorting schemas and states.
# Each function handles a specific type of schema and ensures that the sorting is done correctly.

def sort_any(core, schema, state):
    if not isinstance(schema, dict):
        schema = core.find(schema)
    if not isinstance(state, dict):
        return schema, state

    merged_schema = {}
    merged_state = {}

    for key in union_keys(schema, state):
        if is_schema_key(key):
            if key in state:
                merged_schema[key] = core.merge_schemas(
                    schema.get(key, {}),
                    state[key])
            else:
                merged_schema[key] = schema[key]
        else:
            subschema, merged_state[key] = core.sort(
                schema.get(key, {}),
                state.get(key, None))
            if subschema:
                merged_schema[key] = subschema

    return merged_schema, merged_state


def sort_quote(core, schema, state):
    return schema, state


def sort_map(core, schema, state):
    if not isinstance(schema, dict):
        schema = core.find(schema)
    if not isinstance(state, dict):
        return schema, state

    merged_schema = {}
    merged_state = {}

    value_schema = core.find_parameter(
        schema,
        'value')

    for key in union_keys(schema, state):
        if is_schema_key(key):
            if key in state:
                merged_schema[key] = core.merge_schemas(
                    schema.get(key, {}),
                    state[key])
            else:
                merged_schema[key] = schema[key]
        else:
            subschema, merged_state[key] = core.sort(
                schema.get(key, {}),
                state.get(key, None))
            if subschema:
                value_schema = core.merge_schemas(
                    value_schema,
                    subschema)
                # merged_schema[key] = subschema

    return merged_schema, merged_state


def find_union_type(core, schema, state):
    parameters = core.parameters_for(schema)

    for possible in parameters:
        if core.check(possible, state):
            return core.access(possible)
    return None


# ==========================
# Resolve Functions Overview
# ==========================
# These functions are responsible for resolving updates to the schema.
# Each function handles a specific type of schema and ensures that updates are resolved correctly.

def resolve_any(schema, update, core):
    if not schema or schema == 'any':
        return update
    if not update or update == 'any':
        return schema

    if isinstance(schema, str):
        schema = core.access(schema)

    outcome = schema.copy()

    for key, subschema in update.items():
        if key == '_type' and key in outcome:
            if schema[key] != subschema:
                if core.inherits_from(schema[key], subschema):
                    continue
                elif core.inherits_from(subschema, schema[key]):
                    outcome[key] = subschema
                else:
                    raise Exception(f'cannot resolve types when updating\ncurrent type: {schema}\nupdate type: {update}')

        elif not key in schema or type_parameter_key(schema, key):
            if subschema:
                outcome[key] = subschema
        else:
            outcome[key] = core.resolve_schemas(
                schema.get(key),
                subschema)

    return outcome


def resolve_union(schema, update, core):
    if '_type' in schema and schema['_type'] == 'union':
        union_type, resolve_type = schema, update
    elif '_type' in update and update['_type'] == 'union':
        union_type, resolve_type = update, schema
    else:
        raise Exception(f'empty union?\n{schema}\n{update}')

    if '_type_parameters' in union_type:
        parameters = union_type['_type_parameters']
    else:
        raise Exception(f'no type parameters in union?\n{union_type}')

    for parameter in parameters:
        parameter_key = f'_{parameter}'
        parameter_type = union_type[parameter_key]
        try:
            resolved = core.resolve(
                parameter_type,
                resolve_type)
            return union_type
        except Exception as e:
            pass

    raise Exception(f'could not resolve type with union:\n{update}\nunion:\n{schema}')


def resolve_tree(schema, update, core):
    if not schema or schema == 'any':
        return update
    if not update or update == 'any':
        return schema

    outcome = schema.copy()

    for key, subschema in update.items():
        if key == '_type' and key in outcome:
            if outcome[key] != subschema:
                if core.inherits_from(outcome[key], subschema):
                    continue
                elif core.inherits_from(subschema, outcome[key]):
                    outcome[key] = subschema
                else:
                    leaf_type = core.find_parameter(
                        schema,
                        'leaf')

                    return core.resolve(
                        leaf_type,
                        update)

                    # raise Exception(f'cannot resolve types when updating\ncurrent type: {schema}\nupdate type: {update}')

        elif not key in outcome or type_parameter_key(update, key):
            if subschema:
                outcome[key] = subschema
        else:
            outcome[key] = core.resolve_schemas(
                outcome.get(key),
                subschema)

    return outcome


# ==========================
# Reaction Functions Overview
# ==========================
# These functions are responsible for handling reactions within the schema and state.
# Each function processes a specific type of reaction and ensures that the state is updated accordingly.
# Function signature: (schema, state, reaction, core)

def add_reaction(schema, state, reaction, core):
    path = reaction.get('path')

    redex = {}
    establish_path(
        redex,
        path)

    reactum = {}
    node = establish_path(
        reactum,
        path)

    deep_merge(
        node,
        reaction.get('add', {}))

    return {
        'redex': redex,
        'reactum': reactum}

def remove_reaction(schema, state, reaction, core):
    path = reaction.get('path', ())
    redex = {}
    node = establish_path(
        redex,
        path)

    for remove in reaction.get('remove', []):
        node[remove] = {}

    reactum = {}
    establish_path(
        reactum,
        path)

    return {
        'redex': redex,
        'reactum': reactum}

def replace_reaction(schema, state, reaction, core):
    path = reaction.get('path', ())

    redex = {}
    node = establish_path(
        redex,
        path)

    for before_key, before_state in reaction.get('before', {}).items():
        node[before_key] = before_state

    reactum = {}
    node = establish_path(
        reactum,
        path)

    for after_key, after_state in reaction.get('after', {}).items():
        node[after_key] = after_state

    return {
        'redex': redex,
        'reactum': reactum}

def register_base_reactions(core):
    core.register_reaction('add', add_reaction)
    core.register_reaction('remove', remove_reaction)
    core.register_reaction('replace', replace_reaction)
    core.register_reaction('divide', divide_reaction)


# ===============================
# Types with their type functions
# ===============================
# These dictionaries define the types and their corresponding type functions.

def add_units_to_library(units, type_library):
    for unit_name in units._units:
        try:
            unit = getattr(units, unit_name)
        except:
            # print(f'no unit named {unit_name}')
            continue

        dimensionality = unit.dimensionality
        type_key = render_units_type(dimensionality)
        if not type_library.get(type_key):
            type_library[type_key] = {
                '_default': '',
                '_apply': apply_units,
                '_check': check_units,
                '_serialize': serialize_units,
                '_deserialize': deserialize_units,
                '_description': 'type to represent values with scientific units'}

    return type_library

unit_types = {}
unit_types = add_units_to_library(units, unit_types)

base_types = {
    'boolean': {
        '_type': 'boolean',
        '_default': False,
        '_check': check_boolean,
        '_apply': apply_boolean,
        '_serialize': serialize_boolean,
        '_deserialize': deserialize_boolean,
        '_dataclass': dataclass_boolean},

    # abstract number type
    'number': {
        '_type': 'number',
        '_check': check_number,
        '_apply': accumulate,
        '_serialize': to_string,
        '_description': 'abstract base type for numbers'},

    'integer': {
        '_type': 'integer',
        '_default': 0,
        # inherit _apply and _serialize from number type
        '_check': check_integer,
        '_deserialize': deserialize_integer,
        '_dataclass': dataclass_integer,
        '_description': '64-bit integer',
        '_inherit': 'number'},

    'float': {
        '_type': 'float',
        '_default': 0.0,
        '_check': check_float,
        '_deserialize': deserialize_float,
        '_divide': divide_float,
        '_dataclass': dataclass_float,
        '_description': '64-bit floating point precision number',
        '_inherit': 'number'},

    'string': {
        '_type': 'string',
        '_default': '',
        '_check': check_string,
        '_apply': replace,
        '_serialize': serialize_string,
        '_deserialize': deserialize_string,
        '_dataclass': dataclass_string,
        '_description': '64-bit integer'},

    'enum': {
        '_type': 'enum',
        '_default': default_enum,
        '_apply': apply_enum,
        '_check': check_enum,
        '_serialize': serialize_string,
        '_deserialize': deserialize_string,
        '_dataclass': dataclass_string,
        '_description': 'enumeration type for a selection of key values'},

    'list': {
        '_type': 'list',
        '_default': [],
        '_check': check_list,
        '_slice': slice_list,
        '_apply': apply_list,
        '_serialize': serialize_list,
        '_deserialize': deserialize_list,
        '_dataclass': dataclass_list,
        '_fold': fold_list,
        '_divide': divide_list,
        '_type_parameters': ['element'],
        '_description': 'general list type (or sublists)'},

    'map': {
        '_type': 'map',
        '_default': {},
        '_generate': generate_map,
        '_apply': apply_map,
        '_serialize': serialize_map,
        '_deserialize': deserialize_map,
        '_resolve': resolve_map,
        '_dataclass': dataclass_map,
        '_check': check_map,
        '_slice': slice_map,
        '_fold': fold_map,
        '_divide': divide_map,
        '_sort': sort_map,
        '_type_parameters': ['value'],
        '_description': 'flat mapping from keys of strings to values of any type'},

    'tree': {
        '_type': 'tree',
        '_default': default_tree,
        '_generate': generate_tree,
        '_check': check_tree,
        '_slice': slice_tree,
        '_apply': apply_tree,
        '_serialize': serialize_tree,
        '_deserialize': deserialize_tree,
        '_dataclass': dataclass_tree,
        '_fold': fold_tree,
        '_divide': divide_tree,
        '_resolve': resolve_tree,
        '_type_parameters': ['leaf'],
        '_description': 'mapping from str to some type in a potentially nested form'},

    'array': {
        '_type': 'array',
        '_default': default_array,
        '_check': check_array,
        '_slice': slice_array,
        '_apply': apply_array,
        '_serialize': serialize_array,
        '_deserialize': deserialize_array,
        '_dataclass': dataclass_array,
        '_resolve': resolve_array,
        '_bind': bind_array,
        '_type_parameters': [
            'shape',
            'data'],
        '_description': 'an array of arbitrary dimension'},

    'maybe': {
        '_type': 'maybe',
        '_default': None,
        '_apply': apply_maybe,
        '_check': check_maybe,
        '_slice': slice_maybe,
        '_serialize': serialize_maybe,
        '_deserialize': deserialize_maybe,
        '_dataclass': dataclass_maybe,
        '_resolve': resolve_maybe,
        '_fold': fold_maybe,
        '_type_parameters': ['value'],
        '_description': 'type to represent values that could be empty'},

    'path': {
        '_type': 'path',
        '_inherit': 'list[string~integer]',
        '_apply': apply_path},

    'wires': {
        '_type': 'wires',
        '_inherit': 'tree[path]'},

    'schema': {
        '_type': 'schema',
        '_inherit': 'tree[any]',
        '_apply': apply_schema,
        '_serialize': serialize_schema,
        '_deserialize': deserialize_schema},

    'edge': {
        '_type': 'edge',
        '_default': default_edge,
        '_generate': generate_edge,
        '_apply': apply_edge,
        '_serialize': serialize_edge,
        '_deserialize': deserialize_edge,
        '_dataclass': dataclass_edge,
        '_check': check_edge,
        '_slice': slice_edge,
        # '_merge': merge_edge,
        '_type_parameters': ['inputs', 'outputs'],
        '_description': 'hyperedges in the bigraph, with inputs and outputs as type parameters',
        'inputs': 'wires',
        'outputs': 'wires'}}

registry_types = {
    'any': {
        '_type': 'any',
        '_default': default_any,
        '_slice': slice_any,
        '_apply': apply_any,
        '_check': check_any,
        '_sort': sort_any,
        '_generate': generate_any,
        '_serialize': serialize_any,
        '_deserialize': deserialize_any,
        '_dataclass': dataclass_any,
        '_resolve': resolve_any,
        '_fold': fold_any,
        '_bind': bind_any,
        '_divide': divide_any},

    'quote': {
        '_type': 'quote',
        '_deserialize': deserialize_quote,
        '_default': default_quote,
        '_generate': generate_quote,
        '_sort': sort_quote,
        '_description': 'protect a schema from generation, ie in the config for a nested composite which has type information we only want to evaluate inside of the composite'},

    'tuple': {
        '_type': 'tuple',
        '_default': default_tuple,
        '_apply': apply_tuple,
        '_check': check_tuple,
        '_slice': slice_tuple,
        '_serialize': serialize_tuple,
        '_deserialize': deserialize_tuple,
        '_dataclass': dataclass_tuple,
        '_fold': fold_tuple,
        '_divide': divide_tuple,
        '_bind': bind_tuple,
        '_description': 'tuple of an ordered set of typed values'},

    'union': {
        '_type': 'union',
        '_default': default_union,
        '_apply': apply_union,
        '_check': check_union,
        '_slice': slice_union,
        '_serialize': serialize_union,
        '_deserialize': deserialize_union,
        '_dataclass': dataclass_union,
        '_fold': fold_union,
        '_resolve': resolve_union,
        '_description': 'union of a set of possible types'}}
