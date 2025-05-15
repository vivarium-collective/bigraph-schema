"""
Tests for the type system and schema manipulation functions
"""

import pytest
import pprint
import numpy as np
from dataclasses import asdict

from bigraph_schema.type_functions import (
    divide_longest, base_types, accumulate, to_string, deserialize_integer, apply_schema, data_module)
from bigraph_schema.utilities import compare_dicts, NONE_SYMBOL
from bigraph_schema import TypeSystem
from bigraph_schema.units import units
from bigraph_schema.registry import establish_path, remove_omitted


@pytest.fixture
def core():
    core = TypeSystem()
    return register_test_types(core)


def register_cube(core):
    cube_schema = {
        'shape': {
            '_type': 'shape',
            '_description': 'abstract shape type'},

        'rectangle': {
            '_type': 'rectangle',
            '_divide': divide_longest,
            '_description': 'a two-dimensional value',
            '_inherit': 'shape',
            'width': {'_type': 'integer'},
            'height': {'_type': 'integer'},
        },

        # cannot override existing keys unless it is of a subtype
        'cube': {
            '_type': 'cube',
            '_inherit': 'rectangle',
            'depth': {'_type': 'integer'},
        },
    }

    for type_key, type_data in cube_schema.items():
        core.register(type_key, type_data)

    return core


def register_test_types(core):
    register_cube(core)

    core.register('compartment', {
        'counts': 'tree[float]',
        'inner': 'tree[compartment]'})

    core.register('metaedge', {
        '_inherit': 'edge',
        '_inputs': {
            'before': 'metaedge'},
        '_outputs': {
            'after': 'metaedge'}})

    return core


def test_reregister_type(core):
    core.register('A', {'_default': 'a'})
    with pytest.raises(Exception) as e:
        core.register(
            'A', {'_default': 'b'},
            strict=True)

    core.register('A', {'_default': 'b'}, strict=False)

    assert core.access('A')['_default'] == 'b'


def test_generate_default(core):
    int_default = core.default(
        {'_type': 'integer'}
    )

    assert int_default == 0

    cube_default = core.default(
        {'_type': 'cube'})

    assert 'width' in cube_default
    assert 'height' in cube_default
    assert 'depth' in cube_default

    nested_default = core.default(
        {'a': 'integer',
         'b': {
             'c': 'float',
             'd': 'cube'},
         'e': 'string'})

    assert nested_default['b']['d']['width'] == 0


def test_apply_update(core):
    schema = {'_type': 'cube'}
    state = {
        'width': 11,
        'height': 13,
        'depth': 44,
    }

    update = {
        'depth': -5
    }

    new_state = core.apply(
        schema,
        state,
        update
    )

    assert new_state['width'] == 11
    assert new_state['depth'] == 39


def print_schema_validation(core, library, should_pass):
    for key, declaration in library.items():
        report = core.validate_schema(declaration)
        if len(report) == 0:
            message = f'valid schema: {key}'
            if should_pass:
                print(f'PASS: {message}')
                pprint.pprint(declaration)
            else:
                raise Exception(f'FAIL: {message}\n{declaration}\n{report}')
        else:
            message = f'invalid schema: {key}'
            if not should_pass:
                print(f'PASS: {message}')
                pprint.pprint(declaration)
            else:
                raise Exception(f'FAIL: {message}\n{declaration}\n{report}')


def test_validate_schema(core):
    # good schemas
    print_schema_validation(core, base_types, True)

    good = {
        'not quite int': {
            '_default': 0,
            '_apply': accumulate,
            '_serialize': to_string,
            '_deserialize': deserialize_integer,
            '_description': '64-bit integer'
        },
        'ports match': {
            'a': {
                '_type': 'integer',
                '_value': 2
            },
            'edge1': {
                '_type': 'edge[a:integer]',
                # '_type': 'edge',
                # '_ports': {
                #     '1': {'_type': 'integer'},
                # },
            }
        }
    }

    # bad schemas
    bad = {
        'empty': None,
        'str?': 'not a schema',
        'branch is weird': {
            'left': {'_type': 'ogre'},
            'right': {'_default': 1, '_apply': accumulate},
        },
    }

    # test for ports and wires mismatch

    print_schema_validation(core, good, True)
    print_schema_validation(core, bad, False)


def test_fill_integer(core):
    test_schema = {
        '_type': 'integer'
    }

    full_state = core.fill(test_schema)
    direct_state = core.fill('integer')
    generated_schema, generated_state = core.generate(
        test_schema, None)

    assert generated_schema['_type'] == 'integer'
    assert full_state == direct_state == 0 == generated_state


def test_fill_cube(core):
    test_schema = {'_type': 'cube'}
    partial_state = {'height': 5}

    full_state = core.fill(
        test_schema,
        state=partial_state)

    assert 'width' in full_state
    assert 'height' in full_state
    assert 'depth' in full_state
    assert full_state['height'] == 5
    assert full_state['depth'] == 0


def test_fill_in_missing_nodes(core):
    test_schema = {
        'edge 1': {
            '_type': 'edge',
            '_inputs': {
                'I': 'float'},
            '_outputs': {
                'O': 'float'}}}

    test_state = {
        'edge 1': {
            'inputs': {
                'I': ['a']},
            'outputs': {
                'O': ['a']}}}

    filled = core.fill(
        test_schema,
        test_state)

    assert filled == {
        'a': 0.0,
        'edge 1': {
            'inputs': {
                'I': ['a']},
            'outputs': {
                'O': ['a']}}}


def test_overwrite_existing(core):
    test_schema = {
        'edge 1': {
            '_type': 'edge',
            '_inputs': {
                'I': 'float'},
            '_outputs': {
                'O': 'float'}}}

    test_state = {
        'a': 11.111,
        'edge 1': {
            'inputs': {
                'I': ['a']},
            'outputs': {
                'O': ['a']}}}

    filled = core.fill(
        test_schema,
        test_state)

    assert filled == {
        'a': 11.111,
        'edge 1': {
            'inputs': {
                'I': ['a']},
            'outputs': {
                'O': ['a']}}}


def test_fill_from_parse(core):
    test_schema = {
        'edge 1': 'edge[I:float,O:float]'}

    test_state = {
        'edge 1': {
            'inputs': {
                'I': ['a']},
            'outputs': {
                'O': ['a']}}}

    filled = core.fill(
        test_schema,
        test_state)

    assert filled == {
        'a': 0.0,
        'edge 1': {
            'inputs': {
                'I': ['a']},
            'outputs': {
                'O': ['a']}}}


# def test_fill_in_disconnected_port(core):
#     test_schema = {
#         'edge1': {
#             '_type': 'edge',
#             '_ports': {
#                 '1': {'_type': 'float'}}}}

#     test_state = {}


# def test_fill_type_mismatch(core):
#     test_schema = {
#         'a': {'_type': 'integer', '_value': 2},
#         'edge1': {
#             '_type': 'edge',
#             '_ports': {
#                 '1': {'_type': 'float'},
#                 '2': {'_type': 'float'}},
#             'wires': {
#                 '1': ['..', 'a'],
#                 '2': ['a']},
#             'a': 5}}


# def test_edge_type_mismatch(core):
#     test_schema = {
#         'edge1': {
#             '_type': 'edge',
#             '_ports': {
#                 '1': {'_type': 'float'}},
#             'wires': {
#                 '1': ['..', 'a']}},
#         'edge2': {
#             '_type': 'edge',
#             '_ports': {
#                 '1': {'_type': 'integer'}},
#             'wires': {
#                 '1': ['..', 'a']}}}


def test_establish_path(core):
    tree = {}
    destination = establish_path(
        tree,
        ('some',
         'where',
         'deep',
         'inside',
         'lives',
         'a',
         'tiny',
         'creature',
         'made',
         'of',
         'light'))

    assert tree['some']['where']['deep']['inside']['lives']['a']['tiny']['creature']['made']['of'][
               'light'] == destination


def test_fill_ports(core):
    cell_state = {
        'cell1': {
            'nucleus': {
                'transcription': {
                    '_type': 'edge',
                    'inputs': {'DNA': ['chromosome']},
                    'outputs': {
                        'RNA': ['..', 'cytoplasm']}}}}}

    schema, state = core.complete(
        {},
        cell_state)

    assert 'chromosome' in schema['cell1']['nucleus']


def test_expected_schema(core):
    # equivalent to previous schema:

    # expected = {
    #     'store1': {
    #         'store1.1': {
    #             '_value': 1.1,
    #             '_type': 'float',
    #         },
    #         'store1.2': {
    #             '_value': 2,
    #             '_type': 'integer',
    #         },
    #         'process1': {
    #             '_ports': {
    #                 'port1': {'_type': 'type'},
    #                 'port2': {'_type': 'type'},
    #             },
    #             '_wires': {
    #                 'port1': 'store1.1',
    #                 'port2': 'store1.2',
    #             }
    #         },
    #         'process2': {
    #             '_ports': {
    #                 'port1': {'_type': 'type'},
    #                 'port2': {'_type': 'type'},
    #             },
    #             '_wires': {
    #                 'port1': 'store1.1',
    #                 'port2': 'store1.2',
    #             }
    #         },
    #     },
    #     'process3': {
    #         '_wires': {
    #             'port1': 'store1',
    #         }
    #     }
    # }

    dual_process_schema = {
        'process1': 'edge[input1:float|input2:integer,output1:float|output2:integer]',
        'process2': {
            '_type': 'edge',
            '_inputs': {
                'input1': 'float',
                'input2': 'integer'},
            '_outputs': {
                'output1': 'float',
                'output2': 'integer'}}}

    core.register(
        'dual_process',
        dual_process_schema,
    )

    test_schema = {
        # 'store1': 'process1.edge[port1.float|port2.int]|process2[port1.float|port2.int]',
        'store1': 'dual_process',
        'process3': 'edge[input_process:dual_process,output_process:dual_process]'}

    test_state = {
        'store1': {
            'process1': {
                'inputs': {
                    'input1': ['store1.1'],
                    'input2': ['store1.2']},
                'outputs': {
                    'output1': ['store2.1'],
                    'output2': ['store2.2']}},
            'process2': {
                'inputs': {
                    'input1': ['store2.1'],
                    'input2': ['store2.2']},
                'outputs': {
                    'output1': ['store1.1'],
                    'output2': ['store1.2']}}},
        'process3': {
            'inputs': {
                'input_process': ['store1']},
            'outputs': {
                'output_process': ['store1']}}}

    outcome = core.fill(test_schema, test_state)

    assert outcome == {
        'process3': {
            'inputs': {
                'input_process': ['store1']},
            'outputs': {
                'output_process': ['store1']}},
        'store1': {
            'process1': {
                'inputs': {
                    'input1': ['store1.1'],
                    'input2': ['store1.2']},
                'outputs': {
                    'output1': ['store2.1'],
                    'output2': ['store2.2']}},
            'process2': {
                'inputs': {'input1': ['store2.1'],
                           'input2': ['store2.2']},
                'outputs': {'output1': ['store1.1'],
                            'output2': ['store1.2']}},
            'store1.1': 0.0,
            'store1.2': 0,
            'store2.1': 0.0,
            'store2.2': 0}}


def test_link_place(core):
    # TODO: this form is more fundamental than the compressed/inline dict form,
    #   and we should probably derive that from this form

    bigraph = {
        'nodes': {
            'v0': 'integer',
            'v1': 'integer',
            'v2': 'integer',
            'v3': 'integer',
            'v4': 'integer',
            'v5': 'integer',
            'e0': 'edge[e0-0:int|e0-1:int|e0-2:int]',
            'e1': {
                '_type': 'edge',
                '_ports': {
                    'e1-0': 'integer',
                    'e2-0': 'integer'}},
            'e2': {
                '_type': 'edge[e2-0:int|e2-1:int|e2-2:int]'}},

        'place': {
            'v0': None,
            'v1': 'v0',
            'v2': 'v0',
            'v3': 'v2',
            'v4': None,
            'v5': 'v4',
            'e0': None,
            'e1': None,
            'e2': None},

        'link': {
            'e0': {
                'e0-0': 'v0',
                'e0-1': 'v1',
                'e0-2': 'v4'},
            'e1': {
                'e1-0': 'v3',
                'e1-1': 'v1'},
            'e2': {
                'e2-0': 'v3',
                'e2-1': 'v4',
                'e2-2': 'v5'}},

        'state': {
            'v0': '1',
            'v1': '1',
            'v2': '2',
            'v3': '3',
            'v4': '5',
            'v5': '8',
            'e0': {
                'wires': {
                    'e0-0': 'v0',
                    'e0-1': 'v1',
                    'e0-2': 'v4'}},
            'e1': {
                'wires': {
                    'e1-0': 'v3',
                    'e1-1': 'v1'}},
            'e2': {
                'e2-0': 'v3',
                'e2-1': 'v4',
                'e2-2': 'v5'}}}

    placegraph = {  # schema
        'v0': {
            'v1': int,
            'v2': {
                'v3': int}},
        'v4': {
            'v5': int},
        'e0': 'edge',
        'e1': 'edge',
        'e2': 'edge'}

    hypergraph = {  # edges
        'e0': {
            'e0-0': 'v0',
            'e0-1': 'v1',
            'e0-2': 'v4'},
        'e1': {
            'e1-0': 'v3',
            'e1-1': 'v1'},
        'e2': {
            'e2-0': 'v3',
            'e2-1': 'v4',
            'e2-2': 'v5'}}

    merged = {
        'v0': {
            'v1': {},
            'v2': {
                'v3': {}}},
        'v4': {
            'v5': {}},
        'e0': {
            'wires': {
                'e0.0': ['v0'],
                'e0.1': ['v0', 'v1'],
                'e0.2': ['v4']}},
        'e1': {
            'wires': {
                'e0.0': ['v0', 'v2', 'v3'],
                'e0.1': ['v0', 'v1']}},
        'e2': {
            'wires': {
                'e0.0': ['v0', 'v2', 'v3'],
                'e0.1': ['v4'],
                'e0.2': ['v4', 'v5']}}}

    result = core.link_place(placegraph, hypergraph)
    # assert result == merged


def test_units(core):
    schema_length = {
        'distance': {'_type': 'length'}}

    state = {'distance': 11 * units.meter}
    update = {'distance': -5 * units.feet}

    new_state = core.apply(
        schema_length,
        state,
        update
    )

    assert new_state['distance'] == 9.476 * units.meter


def test_unit_conversion(core):
    # mass * length ^ 2 / second ^ 2

    units_schema = {
        'force': 'length^2*mass/time^2'}

    force_units = units.meter ** 2 * units.kg / units.second ** 2

    instance = {
        'force': 3.333 * force_units}


def test_serialize_deserialize(core):
    schema = {
        'edge1': {
            # '_type': 'edge[1:int|2:float|3:string|4:tree[int]]',
            '_type': 'edge',
            '_outputs': {
                '1': 'integer',
                '2': 'float',
                '3': 'string',
                '4': 'tree[integer]'}},
        'a0': {
            'a0.0': 'integer',
            'a0.1': 'float',
            'a0.2': {
                'a0.2.0': 'string'}},
        'a1': 'tree[integer]'}

    instance = {
        'edge1': {
            'outputs': {
                '1': ['a0', 'a0.0'],
                '2': ['a0', 'a0.1'],
                '3': ['a0', 'a0.2', 'a0.2.0'],
                '4': ['a1']}},
        'a1': {
            'branch1': {
                'branch2': 11,
                'branch3': 22},
            'branch4': 44}}

    instance = core.fill(schema, instance)

    encoded = core.serialize(schema, instance)
    decoded = core.deserialize(schema, encoded)

    assert instance == decoded


# is this a lens?
def test_project(core):
    schema = {
        'edge1': {
            # '_type': 'edge[1:int|2:float|3:string|4:tree[int]]',
            # '_type': 'edge',
            '_type': 'edge',
            '_inputs': {
                '1': 'integer',
                '2': 'float',
                '3': 'string',
                'inner': {
                    'chamber': 'tree[integer]'},
                '4': 'tree[integer]'},
            '_outputs': {
                '1': 'integer',
                '2': 'float',
                '3': 'string',
                'inner': {
                    'chamber': 'tree[integer]'},
                '4': 'tree[integer]'}},
        'a0': {
            'a0.0': 'integer',
            'a0.1': 'float',
            'a0.2': {
                'a0.2.0': 'string'}},
        'a1': {
            '_type': 'tree[integer]'}}

    path_format = {
        '1': 'a0>a0.0',
        '2': 'a0>a0.1',
        '3': 'a0>a0.2>a0.2.0'}

    # TODO: support separate schema/instance, and 
    #   instances with '_type' and type parameter keys
    # TODO: support overriding various type methods
    instance = {
        'a0': {
            'a0.0': 11},
        'edge1': {
            'inputs': {
                '1': ['a0', 'a0.0'],
                '2': ['a0', 'a0.1'],
                '3': ['a0', 'a0.2', 'a0.2.0'],
                'inner': {
                    'chamber': ['a1', 'a1.0']},
                '4': ['a1']},
            'outputs': {
                '1': ['a0', 'a0.0'],
                '2': ['a0', 'a0.1'],
                '3': ['a0', 'a0.2', 'a0.2.0'],
                'inner': {
                    'chamber': {
                        'X': ['a1', 'a1.0', 'Y']}},
                '4': ['a1']}},
        'a1': {
            'a1.0': {
                'X': 555},
            'branch1': {
                'branch2': 11,
                'branch3': 22},
            'branch4': 44}}

    instance = core.fill(schema, instance)

    states = core.view_edge(
        schema,
        instance,
        ['edge1'])

    update = core.project_edge(
        schema,
        instance,
        ['edge1'],
        states)

    assert update == {
        'a0': {
            'a0.0': 11,
            'a0.1': 0.0,
            'a0.2': {
                'a0.2.0': ''}},
        'a1': {
            'a1.0': {
                'X': 555,
                'Y': {}},
            'branch1': {
                'branch2': 11,
                'branch3': 22},
            'branch4': 44}}

    # TODO: make sure apply does not mutate instance
    updated_instance = core.apply(
        schema,
        instance,
        update)

    add_update = {
        '4': {
            'branch6': 111,
            'branch1': {
                '_add': {
                    'branch7': 4444,
                    'branch8': 555,
                },
                '_remove': ['branch2']},
            '_add': {
                'branch5': 55},
            '_remove': ['branch4']}}

    inverted_update = core.project_edge(
        schema,
        updated_instance,
        ['edge1'],
        add_update)

    modified_branch = core.apply(
        schema,
        updated_instance,
        inverted_update)

    assert modified_branch == {
        'a0': {
            'a0.0': 22,
            'a0.1': 0.0,
            'a0.2': {
                'a0.2.0': ''}},
        'edge1': {'inputs': {'1': ['a0', 'a0.0'],
                             '2': ['a0', 'a0.1'],
                             '3': ['a0', 'a0.2', 'a0.2.0'],
                             'inner': {
                                 'chamber': ['a1', 'a1.0']},
                             '4': ['a1']},
                  'outputs': {'1': ['a0', 'a0.0'],
                              '2': ['a0', 'a0.1'],
                              '3': ['a0', 'a0.2', 'a0.2.0'],
                              'inner': {
                                  'chamber': {
                                      'X': ['a1', 'a1.0', 'Y']}},
                              '4': ['a1']}},
        'a1': {
            'a1.0': {
                'X': 1110,
                'Y': {}},
            'branch1': {
                'branch3': 44,
                'branch7': 4444,
                'branch8': 555, },
            'branch6': 111,
            'branch5': 55}}


def test_check(core):
    assert core.check('float', 1.11)
    assert core.check({'b': 'float'}, {'b': 1.11})


def test_inherits_from(core):
    assert core.inherits_from(
        'float',
        'number')

    assert core.inherits_from(
        'tree[float]',
        'tree[number]')

    assert core.inherits_from(
        'tree[path]',
        'tree[list[string~integer]]')

    assert not core.inherits_from(
        'tree[path]',
        'tree[list[number]]')

    assert not core.inherits_from(
        'tree[float]',
        'tree[string]')

    assert not core.inherits_from(
        'tree[float]',
        'list[float]')

    assert core.inherits_from({
        'a': 'float',
        'b': 'schema'}, {

        'a': 'number',
        'b': 'tree'})

    assert not core.inherits_from({
        'a': 'float',
        'b': 'schema'}, {

        'a': 'number',
        'b': 'number'})


def test_resolve_schemas(core):
    resolved = core.resolve_schemas({
        'a': 'float',
        'b': 'map[list[string]]'}, {
        'a': 'number',
        'b': 'map[path]',
        'c': 'string'})

    assert resolved['a']['_type'] == 'float'
    assert resolved['b']['_value']['_type'] == 'path'
    assert resolved['c']['_type'] == 'string'

    raises_on_incompatible_schemas = False
    try:
        core.resolve_schemas({
            'a': 'string',
            'b': 'map[list[string]]'}, {
            'a': 'number',
            'b': 'map[path]',
            'c': 'string'})
    except:
        raises_on_incompatible_schemas = True

    assert raises_on_incompatible_schemas


def test_apply_schema(core):
    current = {
        'a': 'number',
        'b': 'map[path]',
        'd': ('float', 'number', 'list[string]')}

    update = {
        'a': 'float',
        'b': 'map[list[string]]',
        'c': 'string',
        'd': ('number', 'float', 'path')}

    applied = core.apply(
        'schema',
        current,
        update)

    assert applied['a']['_type'] == 'float'
    assert applied['b']['_value']['_type'] == 'path'
    assert applied['c']['_type'] == 'string'
    assert applied['d']['_0']['_type'] == 'float'
    assert applied['d']['_1']['_type'] == 'float'
    assert applied['d']['_2']['_type'] == 'path'


def apply_foursquare(schema, current, update, top_schema, top_state, path, core):
    if isinstance(current, bool) or isinstance(update, bool):
        return update
    else:
        for key, value in update.items():
            current[key] = apply_foursquare(
                schema,
                current[key],
                value,
                top_schema=top_schema,
                top_state=top_state,
                path=path,
                core=core)

        return current


def test_foursquare(core):
    foursquare_schema = {
        '_apply': apply_foursquare,
        '00': 'boolean~foursquare',
        '01': 'boolean~foursquare',
        '10': 'boolean~foursquare',
        '11': 'boolean~foursquare'}

    core.register(
        'foursquare',
        foursquare_schema)

    example = {
        '00': True,
        '01': False,
        '10': False,
        '11': {
            '00': True,
            '01': False,
            '10': False,
            '11': {
                '00': True,
                '01': False,
                '10': False,
                '11': {
                    '00': True,
                    '01': False,
                    '10': False,
                    '11': {
                        '00': True,
                        '01': False,
                        '10': False,
                        '11': {
                            '00': True,
                            '01': False,
                            '10': False,
                            '11': False}}}}}}

    assert core.check(
        'foursquare',
        example)

    example['10'] = 5

    assert not core.check(
        'foursquare',
        example)

    update = {
        '01': True,
        '11': {
            '01': True,
            '11': {
                '11': True,
                '10': {
                    '10': {
                        '00': True,
                        '11': False}}}}}

    result = core.apply(
        'foursquare',
        example,
        update)

    assert result == {
        '00': True,
        '01': True,
        '10': 5,
        '11': {'00': True,
               '01': True,
               '10': False,
               '11': {'00': True,
                      '01': False,
                      '10': {
                          '10': {
                              '00': True,
                              '11': False}},
                      '11': True}}}


def test_add_reaction(core):
    single_node = {
        'environment': {
            '_type': 'compartment',
            'counts': {'A': 144},
            'inner': {
                '0': {
                    'counts': {'A': 13},
                    'inner': {}}}}}

    add_config = {
        'path': ['environment', 'inner'],
        'add': {
            '1': {
                'counts': {
                    'A': 8}}}}

    schema, state = core.infer_schema(
        {},
        single_node)

    assert '0' in state['environment']['inner']
    assert '1' not in state['environment']['inner']

    result = core.apply(
        schema,
        state, {
            '_react': {
                'add': add_config}})

    # '_react': {
    #     'reaction': 'add',
    #     'config': add_config}})

    assert '0' in result['environment']['inner']
    assert '1' in result['environment']['inner']


def test_remove_reaction(core):
    single_node = {
        'environment': {
            '_type': 'compartment',
            'counts': {'A': 144},
            'inner': {
                '0': {
                    'counts': {'A': 13},
                    'inner': {}},
                '1': {
                    'counts': {'A': 13},
                    'inner': {}}}}}

    remove_config = {
        'path': ['environment', 'inner'],
        'remove': ['0']}

    schema, state = core.infer_schema(
        {},
        single_node)

    assert '0' in state['environment']['inner']
    assert '1' in state['environment']['inner']

    result = core.apply(
        schema,
        state, {
            '_react': {
                'remove': remove_config}})

    assert '0' not in result['environment']['inner']
    assert '1' in state['environment']['inner']


def test_replace_reaction(core):
    single_node = {
        'environment': {
            '_type': 'compartment',
            'counts': {'A': 144},
            'inner': {
                '0': {
                    'counts': {'A': 13},
                    'inner': {}},
                '1': {
                    'counts': {'A': 13},
                    'inner': {}}}}}

    # replace_config = {
    #     'path': ['environment', 'inner'],
    #     'before': {'0': {'A': '?1'}},
    #     'after': {
    #         '2': {
    #             'counts': {
    #                 'A': {'function': 'divide', 'arguments': ['?1', 0.5], }}},
    #         '3': {
    #             'counts': {
    #                 'A': '@1'}}}}

    replace_config = {
        'path': ['environment', 'inner'],
        'before': {'0': {}},
        'after': {
            '2': {
                'counts': {
                    'A': 3}},
            '3': {
                'counts': {
                    'A': 88}}}}

    schema, state = core.infer_schema(
        {},
        single_node)

    assert '0' in state['environment']['inner']
    assert '1' in state['environment']['inner']

    result = core.apply(
        schema,
        state, {
            '_react': {
                'replace': replace_config}})

    assert '0' not in result['environment']['inner']
    assert '1' in result['environment']['inner']
    assert '2' in result['environment']['inner']
    assert '3' in result['environment']['inner']


def test_reaction(core):
    single_node = {
        'environment': {
            'counts': {},
            'inner': {
                '0': {
                    'counts': {}}}}}

    # TODO: compartment type ends up as 'any' at leafs?

    # TODO: come at divide reaction from the other side:
    #   ie make a call for it, then figure out what the
    #   reaction needs to be
    def divide_reaction(container, mother, divider):
        daughters = divider(mother)

        return {
            'redex': mother,
            'reactum': daughters}

    embedded_tree = {
        'environment': {
            '_type': 'compartment',
            'counts': {},
            'inner': {
                'agent1': {
                    '_type': 'compartment',
                    'counts': {},
                    'inner': {
                        'agent2': {
                            '_type': 'compartment',
                            'counts': {},
                            'inner': {},
                            'transport': {
                                'wires': {
                                    'outer': ['..', '..'],
                                    'inner': ['inner']}}}},
                    'transport': {
                        'wires': {
                            'outer': ['..', '..'],
                            'inner': ['inner']}}}}}}

    mother_tree = {
        'environment': {
            '_type': 'compartment',
            'counts': {
                'A': 15},
            'inner': {
                'mother': {
                    '_type': 'compartment',
                    'counts': {
                        'A': 5}}}}}

    divide_react = {
        '_react': {
            'redex': {
                'mother': {
                    'counts': '@counts'}},
            'reactum': {
                'daughter1': {
                    'counts': '@daughter1_counts'},
                'daughter2': {
                    'counts': '@daughter2_counts'}},
            'calls': [{
                'function': 'divide_counts',
                'arguments': ['@counts', [0.5, 0.5]],
                'bindings': ['@daughter1_counts', '@daughter2_counts']}]}}

    divide_update = {
        '_react': {
            'reaction': 'divide_counts',
            'config': {
                'id': 'mother',
                'state_key': 'counts',
                'daughters': [
                    {'id': 'daughter1', 'ratio': 0.3},
                    {'id': 'daughter2', 'ratio': 0.7}]}}}

    divide_update_concise = {
        '_react': {
            'divide_counts': {
                'id': 'mother',
                'state_key': 'counts',
                'daughters': [
                    {'id': 'daughter1', 'ratio': 0.3},
                    {'id': 'daughter2', 'ratio': 0.7}]}}}


def test_map_type(core):
    schema = 'map[integer]'

    state = {
        'a': 12,
        'b': 13,
        'c': 15,
        'd': 18}

    update = {
        'b': 44,
        'd': 111}

    assert core.check(schema, state)
    assert core.check(schema, update)
    assert not core.check(schema, 15)

    result = core.apply(
        schema,
        state,
        update)

    assert result['a'] == 12
    assert result['b'] == 57
    assert result['d'] == 129

    encode = core.serialize(schema, update)
    assert encode['d'] == '111'

    decode = core.deserialize(schema, encode)
    assert decode == update


def test_tree_type(core):
    schema = 'tree[maybe[integer]]'

    state = {
        'a': 12,
        'b': 13,
        'c': {
            'e': 5555,
            'f': 111},
        'd': None}

    update = {
        'a': None,
        'c': {
            'e': 88888,
            'f': 2222,
            'G': None},
        'd': 111}

    assert core.check(schema, state)
    assert core.check(schema, update)
    assert core.check(schema, 15)
    assert core.check(schema, None)
    assert core.check(schema, {'c': {'D': None, 'e': 11111}})
    assert not core.check(schema, 'yellow')
    assert not core.check(schema, {'a': 5, 'b': 'green'})
    assert not core.check(schema, {'c': {'D': False, 'e': 11111}})

    result = core.apply(
        schema,
        state,
        update)

    assert result['a'] == None
    assert result['b'] == 13
    assert result['c']['f'] == 2333
    assert result['d'] == 111

    encode = core.serialize(schema, update)
    assert encode['a'] == NONE_SYMBOL
    assert encode['d'] == '111'

    decode = core.deserialize(schema, encode)
    assert decode == update


def test_maybe_type(core):
    schema = 'map[maybe[integer]]'

    state = {
        'a': 12,
        'b': 13,
        'c': None,
        'd': 18}

    update = {
        'a': None,
        'c': 44,
        'd': 111}

    assert core.check(schema, state)
    assert core.check(schema, update)
    assert not core.check(schema, 15)

    result = core.apply(
        schema,
        state,
        update)

    assert result['a'] == None
    assert result['b'] == 13
    assert result['c'] == 44
    assert result['d'] == 129

    encode = core.serialize(schema, update)
    assert encode['a'] == NONE_SYMBOL
    assert encode['d'] == '111'

    decode = core.deserialize(schema, encode)
    assert decode == update


def test_tuple_type(core):
    schema = {
        '_type': 'tuple',
        '_type_parameters': ['0', '1', '2'],
        '_0': 'string',
        '_1': 'int',
        '_2': 'map[maybe[float]]'}

    schema = ('string', 'int', 'map[maybe[float]]')
    schema = 'tuple[string,int,map[maybe[float]]]'
    schema = 'string|integer|map[maybe[float]]'

    state = (
        'aaaaa',
        13, {
            'a': 1.1,
            'b': None})

    update = (
        'bbbbbb',
        10, {
            'a': 33.33,
            'b': 4.44444})

    assert core.check(schema, state)
    assert core.check(schema, update)
    assert not core.check(schema, 15)

    result = core.apply(
        schema,
        state,
        update)

    assert len(result) == 3
    assert result[0] == update[0]
    assert result[1] == 23
    assert result[2]['a'] == 34.43
    assert result[2]['b'] == update[2]['b']

    encode = core.serialize(schema, state)
    assert encode[2]['b'] == NONE_SYMBOL
    assert encode[1] == '13'

    decode = core.deserialize(schema, encode)
    assert decode == state

    tuple_type = core.access('(3|4|10)')
    assert '_2' in tuple_type
    assert tuple_type['_2'] == '10'

    tuple_type = core.access('tuple[9,float,7]')
    assert '_2' in tuple_type
    assert tuple_type['_2'] == '7'


def test_union_type(core):
    schema = {
        '_type': 'union',
        '_type_parameters': ['0', '1', '2'],
        '_0': 'string',
        '_1': 'integer',
        '_2': 'map[maybe[float]]'}

    schema = 'string~integer~map[maybe[float]]'

    state = {
        'a': 1.1,
        'b': None}

    update = {
        'a': 33.33,
        'b': 4.44444}

    assert core.check(schema, state)
    assert core.check(schema, update)
    assert core.check(schema, 15)

    wrong_state = {
        'a': 1.1,
        'b': None}

    wrong_update = 'a different type'

    assert core.check(schema, wrong_state)
    assert core.check(schema, wrong_update)

    # TODO: deal with union apply of different types

    result = core.apply(
        schema,
        state,
        update)

    assert result['a'] == 34.43
    assert result['b'] == update['b']

    encode = core.serialize(schema, state)
    assert encode['b'] == NONE_SYMBOL

    decode = core.deserialize(schema, encode)
    assert decode == state


def test_union_values(core):
    schema = 'map[integer~string~map[maybe[float]]]'

    state = {
        'a': 'bbbbb',
        'b': 15}

    update = {
        'a': 'aaaaa',
        'b': 22}

    assert core.check(schema, state)
    assert core.check(schema, update)
    assert not core.check(schema, 15)

    result = core.apply(
        schema,
        state,
        update)

    assert result['a'] == 'aaaaa'
    assert result['b'] == 37

    encode = core.serialize(schema, state)
    decode = core.deserialize(schema, encode)

    assert decode == state


def test_array_type(core):
    shape = (3, 4, 10)
    shape_representation = core.representation(shape)
    shape_commas = ','.join([
        str(x)
        for x in shape])

    schema = {
        '_type': 'map',
        '_value': {
            '_type': 'array',
            # '_shape': '(3|4|10)',
            '_shape': shape_representation,
            '_data': 'float'}}

    schema = f'map[array[tuple[{shape_commas}],float]]'
    schema = f'map[array[{shape_representation},float]]'

    state = {
        'a': np.zeros(shape),
        'b': np.ones(shape)}

    update = {
        'a': np.full(shape, 5.555),
        'b': np.full(shape, 9.999)}

    assert core.check(schema, state)
    assert core.check(schema, update)
    assert not core.check(schema, 15)

    result = core.apply(
        schema,
        state,
        update)

    assert result['a'][0, 0, 0] == 5.555
    assert result['b'][0, 0, 0] == 10.999

    encode = core.serialize(schema, state)
    assert encode['b']['shape'] == list(shape)
    assert encode['a']['data'] == 'float'

    decode = core.deserialize(schema, encode)

    for key in state:
        assert np.equal(
            decode[key],
            state[key]).all()

    found = core.find(
        schema)

    default = core.default(
        found['_value'])

    assert default.shape == shape


def test_infer_edge(core):
    initial_schema = {}
    initial_state = {
        'fade': {
            '_type': 'edge',
            '_inputs': {
                'yellow': 'array[(3|4|10),float]'},
            '_outputs': {
                'green': 'array[(11|5|8),float]'},
            'inputs': {
                'yellow': ['yellow']},
            'outputs': {
                'green': ['green']}}}

    update = {
        'yellow': np.ones((3, 4, 10)),
        'fade': {
            'inputs': {
                'yellow': ['red']},
            'outputs': {
                'green': ['green', 'green', 'green']}}}

    schema, state = core.complete(
        initial_schema,
        initial_state)

    assert core.check(schema, state)
    assert not core.check(schema, 15)

    result = core.apply(
        schema,
        state,
        update)

    assert result['yellow'][0, 0, 0] == 1.0
    assert result['fade']['inputs']['yellow'] == ['red']

    encode = core.serialize(schema, state)
    decode = core.deserialize(schema, encode)

    assert np.equal(
        decode['yellow'],
        state['yellow']).all()


def test_edge_type(core):
    schema = {
        'fade': {
            '_type': 'edge',
            '_inputs': {
                'yellow': {
                    '_type': 'array',
                    '_shape': 'tuple(3,4,10)',
                    '_data': 'float'}},
            '_outputs': {
                'green': {
                    '_type': 'array',
                    '_shape': 'tuple(11,5,8)',
                    '_data': 'float'}}}}

    initial_schema = {
        'fade': 'edge[yellow:array[(3|4|10),float],green:array[(11|5|8),float]]'}

    initial_state = {
        # 'yellow': np.zeros((3, 4, 10)),
        # 'green': np.ones((11, 5, 8)),
        'fade': {
            'inputs': {
                'yellow': ['yellow']},
            'outputs': {
                'green': ['green']}}}

    schema, state = core.complete(
        initial_schema,
        initial_state)

    update = {
        'yellow': np.ones((3, 4, 10)),
        'fade': {
            'inputs': {
                'yellow': ['red']},
            'outputs': {
                'green': ['green', 'green', 'green']}}}

    assert core.check(schema, state)
    assert not core.check(schema, 15)

    result = core.apply(
        schema,
        state,
        update)

    assert result['yellow'][0, 0, 0] == 1.0
    assert result['fade']['inputs']['yellow'] == ['red']

    encode = core.serialize(schema, state)
    decode = core.deserialize(schema, encode)

    assert np.equal(
        decode['yellow'],
        state['yellow']).all()


def test_edge_complete(core):
    edge_schema = {
        '_type': 'edge',
        '_inputs': {
            'concentration': 'float',
            'field': 'map[boolean]'},
        '_outputs': {
            'target': 'boolean',
            # 'inner': {
            #     'nested': 'boolean'},
            'total': 'integer',
            'delta': 'float'}}

    edge_state = {
        'inputs': {
            'concentration': ['molecules', 'glucose'],
            'field': ['states']},
        'outputs': {
            'target': ['states', 'X'],
            # 'inner': {
            #     'nested': ['states', 'A']},
            'total': ['emitter', 'total molecules'],
            'delta': ['molecules', 'glucose']}}

    # edge_state = {
    #     'inputs': {
    #         'concentration': ['..', 'molecules', 'glucose'],
    #         'field': ['..', 'states']},
    #     'outputs': {
    #         'target': ['..', 'states', 'X'],
    #         'total': ['..', 'emitter', 'total molecules'],
    #         'delta': ['..', 'molecules', 'glucose']}}

    full_schema, full_state = core.complete(
        {'edge': edge_schema},
        {'edge': edge_state})

    assert full_schema['states']['_type'] == 'map'


def test_divide(core):
    schema = {
        'a': 'tree[maybe[float]]',
        'b': 'float~list[string]',
        'c': {
            'd': 'map[edge[GGG:float,OOO:float]]',
            'e': 'array[(3|4|10),float]'}}

    state = {
        'a': {
            'x': {
                'oooo': None,
                'y': 1.1,
                'z': 33.33},
            'w': 44.444},
        'b': ['1', '11', '111', '1111'],
        'c': {
            'd': {
                'A': {
                    'inputs': {
                        'GGG': ['..', '..', 'a', 'w']},
                    'outputs': {
                        'OOO': ['..', '..', 'a', 'x', 'y']}},
                'B': {
                    'inputs': {
                        'GGG': ['..', '..', 'a', 'x', 'y']},
                    'outputs': {
                        'OOO': ['..', '..', 'a', 'w']}}},
            'e': np.zeros((3, 4, 10))}}

    divisions = 3
    division = core.fold(
        schema,
        state,
        'divide',
        {'divisions': divisions})

    assert len(division) == divisions
    assert 'a' in division[0].keys()
    assert len(division[1]['b']) == len(state['b'])


def test_merge(core):
    current_schema = {
        'a': 'tree[maybe[float]]',
        'b': 'float~list[string]',
        'c': {
            'd': 'map[edge[GGG:float,OOO:float]]',
            'e': 'array[(3|4|10),float]'}}

    current_state = {
        'a': {
            'x': {
                'oooo': None,
                'y': 1.1,
                'z': 33.33},
            'w': 44.444},
        'b': ['1', '11', '111', '1111'],
        'c': {
            'd': {
                'A': {
                    'inputs': {
                        'GGG': ['..', '..', 'a', 'w']},
                    'outputs': {
                        'OOO': ['..', '..', 'a', 'x', 'y']}},
                'B': {
                    'inputs': {
                        'GGG': ['..', '..', 'a', 'x', 'y']},
                    'outputs': {
                        'OOO': ['..', '..', 'a', 'w']}}},
            'e': np.zeros((3, 4, 10))}}

    merge_state = {
        'z': 555.55,
        'b': ['333333333'],
        'a': {
            'x': {
                'x': {
                    'o': 99999.11}}}}

    result = core.merge_recur(
        current_schema,
        current_state,
        merge_state)

    assert result['z'] == merge_state['z']
    assert result['b'] == merge_state['b']
    assert result['a']['x']['x']['o'] == merge_state['a']['x']['x']['o']


def test_bind(core):
    current_schema = {
        'a': 'tree[maybe[float]]',
        'b': 'float~list[string]',
        'c': {
            'd': 'map[edge[GGG:float,OOO:float]]',
            'e': 'array[(3|4|10),float]'}}

    current_state = {
        'a': {
            'x': {
                'oooo': None,
                'y': 1.1,
                'z': 33.33},
            'w': 44.444},
        'b': ['1', '11', '111', '1111'],
        'c': {
            'd': {
                'A': {
                    'inputs': {
                        'GGG': ['..', '..', 'a', 'w']},
                    'outputs': {
                        'OOO': ['..', '..', 'a', 'x', 'y']}},
                'B': {
                    'inputs': {
                        'GGG': ['..', '..', 'a', 'x', 'y']},
                    'outputs': {
                        'OOO': ['..', '..', 'a', 'w']}}},
            'e': np.zeros((3, 4, 10))}}

    result_schema, result_state = core.bind(
        current_schema,
        current_state,
        'z',
        'float',
        555.55)

    assert result_schema['z']['_type'] == 'float'
    assert result_state['z'] == 555.55


def test_slice(core):
    schema, state = core.slice(
        'map[float]',
        {'aaaa': 55.555},
        ['aaaa'])

    schema, state = core.complete({}, {
        'top': {
            '_type': 'tree[list[maybe[(float|integer)~string]]]',
            'AAAA': {
                'BBBB': {
                    'CCCC': [
                        (1.3, 5),
                        'okay',
                        (55.555, 1),
                        None,
                        'what',
                        'is']}},
            'DDDD': [
                (3333.1, 88),
                'in',
                'between',
                (66.8, -3),
                None,
                None,
                'later']}})

    float_schema, float_state = core.slice(
        schema,
        state,
        ['top', 'AAAA', 'BBBB', 'CCCC', 2, 0])

    assert float_schema['_type'] == 'float'
    assert float_state == 55.555

    assert core.slice(
        schema,
        state,
        ['top', 'AAAA', 'BBBB', 'CCCC', 3])[1] is None


def test_star_path(core):
    nested_schema = 'map[map[green:float|yellow:integer|blue:string]]'
    nested_state = {
        'aaa': {
            'bbb': {
                'green': 1.1,
                'yellow': 55,
                'blue': 'what'},
            'ccc': {
                'green': 9999.4,
                'yellow': 11,
                'blue': 'umbrella'}}}

    # TODO: can you do everything the * is doing here with _path instead?
    nested_path = ['aaa', '*', 'green']

    schema, state = core.slice(
        nested_schema,
        nested_state,
        nested_path)

    assert schema['_value']['_type'] == 'float'
    assert state['ccc'] == 9999.4


def test_star_view_project(core):
    schema = {
        'edges': 'map[edge[view:map[float],project:map[string]]]',
        'stores': 'map[map[green:float|yellow:integer|blue:string]]'}

    state = {
        'edges': {
            'edge': {
                'inputs': {
                    'view': ['..', 'stores', 'aaa', '*', 'green']},
                'outputs': {
                    'project': ['..', 'stores', 'aaa', '*', 'blue']}}},
        'stores': {
            'aaa': {
                'bbb': {
                    'green': 1.1,
                    'yellow': 55,
                    'blue': 'what'},
                'ccc': {
                    'green': 9999.4,
                    'yellow': 11,
                    'blue': 'umbrella'}}}}

    edge_path = ['edges', 'edge']

    view = core.view_edge(
        schema,
        state,
        edge_path)

    internal = {
        'project': {
            'bbb': 'everything',
            'ccc': 'inside out'}}

    project = core.project_edge(
        schema,
        state,
        edge_path,
        internal)

    assert view['view']['bbb'] == state['stores']['aaa']['bbb']['green']
    assert project['stores']['aaa']['ccc']['blue'] == internal['project']['ccc']


def test_set_slice(core):
    float_schema, float_state = core.set_slice(
        'map[float]',
        {'aaaa': 55.555},
        ['bbbbb'],
        'float',
        888.88888)

    assert float_schema['_type'] == 'map'
    assert float_state['bbbbb'] == 888.88888

    schema, state = core.complete({}, {
        'top': {
            '_type': 'tree[list[maybe[(float|integer)~string]]]',
            'AAAA': {
                'BBBB': {
                    'CCCC': [
                        (1.3, 5),
                        'okay',
                        (55.555, 1),
                        None,
                        'what',
                        'is']}},
            'DDDD': [
                (3333.1, 88),
                'in',
                'between',
                (66.8, -3),
                None,
                None,
                'later']}})

    leaf_schema, leaf_state = core.set_slice(
        schema,
        state,
        ['top', 'AAAA', 'BBBB', 'CCCC', 2, 1],
        'integer',
        33)

    assert core.slice(
        leaf_schema,
        leaf_state, [
            'top',
            'AAAA',
            'BBBB',
            'CCCC',
            2,
            1])[1] == 33


def from_state(dataclass, state):
    if hasattr(dataclass, '__dataclass_fields__'):
        fields = dataclass.__dataclass_fields__
        state = state or {}

        init = {}
        for key, field in fields.items():
            substate = from_state(
                field.type,
                state.get(key))
            init[key] = substate
        instance = dataclass(**init)
    # elif get_origin(dataclass) in [typing.Union, typing.Mapping]:
    #     instance = state
    else:
        instance = state
        # instance = dataclass(state)

    return instance


def test_dataclass(core):
    simple_schema = {
        'a': 'float',
        'b': 'integer',
        'c': 'boolean',
        'x': 'string'}

    # TODO: accept just a string instead of only a path
    simple_dataclass = core.dataclass(
        simple_schema,
        ['simple'])

    simple_state = {
        'a': 88.888,
        'b': 11111,
        'c': False,
        'x': 'not a string'}

    simple_new = simple_dataclass(
        a=1.11,
        b=33,
        c=True,
        x='what')

    simple_from = from_state(
        simple_dataclass,
        simple_state)

    nested_schema = {
        'a': {
            'a': {
                'a': 'float',
                'b': 'float'},
            'x': 'float'}}

    nested_dataclass = core.dataclass(
        nested_schema,
        ['nested'])

    nested_state = {
        'a': {
            'a': {
                'a': 13.4444,
                'b': 888.88},
            'x': 111.11111}}

    nested_new = data_module.nested(
        data_module.nested_a(
            data_module.nested_a_a(
                a=222.22,
                b=3.3333),
            5555.55))

    nested_from = from_state(
        nested_dataclass,
        nested_state)

    complex_schema = {
        'a': 'tree[maybe[float]]',
        'b': 'float~list[string]',
        'c': {
            'd': 'map[edge[GGG:float,OOO:float]]',
            'e': 'array[(3|4|10),float]'}}

    complex_dataclass = core.dataclass(
        complex_schema,
        ['complex'])

    complex_state = {
        'a': {
            'x': {
                'oooo': None,
                'y': 1.1,
                'z': 33.33},
            'w': 44.444},
        'b': ['1', '11', '111', '1111'],
        'c': {
            'd': {
                'A': {
                    'inputs': {
                        'GGG': ['..', '..', 'a', 'w']},
                    'outputs': {
                        'OOO': ['..', '..', 'a', 'x', 'y']}},
                'B': {
                    'inputs': {
                        'GGG': ['..', '..', 'a', 'x', 'y']},
                    'outputs': {
                        'OOO': ['..', '..', 'a', 'w']}}},
            'e': np.zeros((3, 4, 10))}}

    complex_from = from_state(
        complex_dataclass,
        complex_state)

    complex_dict = asdict(complex_from)

    # assert complex_dict == complex_state ? 

    assert complex_from.a['x']['oooo'] is None
    assert len(complex_from.c.d['A']['inputs']['GGG'])
    assert isinstance(complex_from.c.e, np.ndarray)


def test_enum_type(core):
    core.register(
        'planet',
        'enum[mercury,venus,earth,mars,jupiter,saturn,neptune]')

    # core.register('planet', {
    #     '_type': 'enum',
    #     '_type_parameters': ['0', '1', '2', '3', '4', '5', '6'],
    #     '_0': 'mercury',
    #     '_1': 'venus',
    #     '_2': 'earth',
    #     '_3': 'mars',
    #     '_4': 'jupiter',
    #     '_5': 'saturn',
    #     '_6': 'neptune'})

    assert core.default('planet') == 'mercury'

    solar_system_schema = {
        'planets': 'map[planet]'}

    solar_system = {
        'planets': {
            '3': 'earth',
            '4': 'mars'}}

    jupiter_update = {
        'planets': {
            '5': 'jupiter'}}

    pluto_update = {
        'planets': {
            '7': 'pluto'}}

    assert core.check(
        solar_system_schema,
        solar_system)

    assert core.check(
        solar_system_schema,
        jupiter_update)

    assert not core.check(
        solar_system_schema,
        pluto_update)

    with_jupiter = core.apply(
        solar_system_schema,
        solar_system,
        jupiter_update)

    try:
        core.apply(
            solar_system_schema,
            solar_system,
            pluto_update)

        assert False
    except Exception as e:
        print(e)
        assert True


def test_map_schema(core):
    schema = {
        'greetings': 'map[hello:string]',
        'edge': {
            '_type': 'edge',
            '_inputs': {
                'various': {
                    '_type': 'map',
                    '_value': {
                        'world': 'string'}}},
            '_outputs': {
                'referent': 'float'}}}

    state = {
        'edge': {
            'inputs': {
                'various': ['greetings']},
            'outputs': {
                'referent': ['where']}},

        'greetings': {
            'a': {
                'hello': 'yes'},
            'b': {
                'hello': 'again',
                'world': 'present'},
            'c': {
                'other': 'other'}}}

    complete_schema, complete_state = core.complete(
        schema,
        state)

    assert complete_schema['greetings']['_value']['hello']['_type'] == 'string'
    assert complete_schema['greetings']['_value']['world']['_type'] == 'string'

    assert 'world' in complete_state['greetings']['a']
    assert complete_schema['greetings']['_value']['world']['_type'] == 'string'


def test_representation(core):
    schema_examples = [
        'map[float]',
        '(string|float)',
        'tree[(a:float|b:map[string])]',
        'array[(5|11),maybe[integer]]',
        'edge[(x:float|y:tree[(z:float)]),(w:(float|float|float))]']

    for example in schema_examples:
        full_type = core.access(example)
        representation = core.representation(full_type)

        if example != representation:
            raise Exception(
                f'did not receive the same type after parsing and finding the representation:\n  {example}\n  {representation}')


def test_generate(core):
    schema = {
        'A': 'float',
        'B': 'enum[one,two,three]',
        'units': 'map[float]'}

    state = {
        'C': {
            '_type': 'enum[x,y,z]',
            '_default': 'y'},
        'units': {
            'x': 11.1111,
            'y': 22.833333}}

    generated_schema, generated_state = core.generate(
        schema,
        state)

    assert generated_state['A'] == 0.0
    assert generated_state['B'] == 'one'
    assert generated_state['C'] == 'y'
    assert generated_state['units']['y'] == 22.833333
    assert 'x' not in generated_schema['units']


def test_edge_cycle(core):
    empty_schema = {}
    empty_state = {}

    A_schema = {
        'A': {
            '_type': 'metaedge',
            '_inputs': {
                'before': {
                    'inputs': {'before': {'_default': ['B']}},
                    'outputs': {'after': {'_default': ['A']}}}},
            '_outputs': {
                'after': {
                    'inputs': {'before': {'_default': ['A']}},
                    'outputs': {'after': {'_default': ['C']}}}},
            'inputs': {'before': {'_default': ['C']}},
            'outputs': {'after': {'_default': ['B']}}}}

    A_state = {
        'A': {
            '_type': 'metaedge',
            '_inputs': {
                'before': {
                    'inputs': {'before': {'_default': ['B']}},
                    'outputs': {'after': {'_default': ['A']}}}},
            '_outputs': {
                'after': {
                    'inputs': {'before': {'_default': ['A']}},
                    'outputs': {'after': {'_default': ['C']}}}},
            'inputs': {'before': {'_default': ['C']}},
            'outputs': {'after': {'_default': ['B']}}}}

    schema_from_schema, state_from_schema = core.generate(
        A_schema,
        empty_state)

    schema_from_state, state_from_state = core.generate(
        empty_schema,
        A_state)

    # print(compare_dicts(schema_from_schema, schema_from_state))
    # print(compare_dicts(state_from_schema, state_from_state))

    if schema_from_schema != schema_from_state:
        print(compare_dicts(schema_from_schema, schema_from_state))

    if state_from_schema != state_from_state:
        print(compare_dicts(state_from_schema, state_from_state))

    assert schema_from_schema == schema_from_state
    assert state_from_schema == state_from_state

    for key in ['A', 'B', 'C']:
        for result in [schema_from_schema, state_from_schema, schema_from_state, state_from_state]:
            assert key in result


def test_merge(core):
    schema = {
        'A': 'float',
        'B': 'enum[one,two,three]',
        'units': 'map[float]'}

    state = {
        'C': {
            '_type': 'enum[x,y,z]',
            '_default': 'y'},
        'units': {
            'x': 11.1111,
            'y': 22.833333}}

    generated_schema, generated_state = core.generate(
        schema,
        state)

    edge_state = {
        '_type': 'edge',
        '_inputs': {
            'input': 'float'},
        '_outputs': {
            'output': 'float'},
        'inputs': {
            'input': ['A']},
        'outputs': {
            'output': ['D']}}

    top_schema, top_state = core.merge(
        generated_schema,
        generated_state,
        ['edge'],
        {},
        edge_state)

    assert 'D' in top_state
    assert top_schema['D']['_type'] == 'float'

def test_remove_omitted(core=None):
    result = remove_omitted(
        {'a': {}, 'b': {'c': {}, 'd': {}}},
        {'b': {'c': {}}},
        {'a': {'X': 1111}, 'b': {'c': {'Y': 4444}, 'd': {'Z': 99999}}})

    assert 'a' not in result
    assert result['b']['c']['Y'] == 4444
    assert 'd' not in result['b']


def test_union_key_error(core):
    schema = core.access('map[map[float]]')
    state = {
        'a': {'b': 1.1},
        'c': {'d': 2.2},
        'e': 3.3  # this should be an error
    }
    generate_method = core.choose_method(schema, state, 'generate')

    # assert that the Exception is raised
    with pytest.raises(Exception):
        result = generate_method(core, schema, state)


def fix_test_slice_edge(core):
    initial_schema = {
        'edge': {
            '_type': 'edge',
            '_inputs': {
                'a': 'float',
                'b': {'c': 'float', 'd': 'string'},
                'e': {'f': 'array[(3|3),float]'}},
            '_outputs': {
                'g': 'float',
                'h': {'i': {'j': 'map[integer]'}},
                'k': {'l': 'array[(3|3),float]'}}}}

    initial_state = {
        'JJJJ': {'MMMM': 55555},
        'edge': {
            'inputs': {
                'a': ['AAAA'],
                'b': {
                    'c': ['CCCC'],
                    'd': ['DDDD']},
                'e': ['EEEE']},
            'outputs': {
                'g': ['GGGG'],
                'h': {'i': {'j': ['JJJJ']}},
                'k': {'l': ['LLLL', 'LLLLL', 'LLLLLL']}}}}

    schema, state = core.generate(initial_schema, initial_state)

    inner_schema, inner_state = core.slice(
        schema,
        state,
        ['edge', 'outputs', 'h', 'i', 'j', 'MMMM'])

    assert inner_schema['_type'] == 'integer'
    assert inner_state == 55555


def fix_test_complex_wiring(core):
    initial_schema = {
        'edge': {
            '_type': 'edge',
            '_inputs': {
                'a': {
                    'b': 'float',
                    'c': 'float',
                    'd': 'float'}},
            '_outputs': {}}}

    initial_state = {
        'edge': {
            'inputs': {
                'a': {
                    '_path': ['AAAA', 'AAAAA'],
                    'b': ['BBBB'],
                    'c': ['CCCC', 'CCCCC']}}}}

    schema, state = core.generate(
        initial_schema,
        initial_state)

    assert state['AAAA']['AAAAA']['BBBB'] == 0.0
    assert state['AAAA']['AAAAA']['CCCC']['CCCCC'] == 0.0
    assert state['AAAA']['AAAAA']['d'] == 0.0


def test_tree_equivalence(core):
    initial_state = {
        'store1': {
            'store1.1': '1.0'}}

    # create a nested store type and register it
    store11 = {
        'store1.1': {
            '_type': 'float',
            '_default': '1.0'}}

    core.register('store1.1', store11)

    # create a tree schema that uses this type
    store_tree = {
        '_type': 'tree',
        '_leaf': 'store1.1'}

    # interpret the state as a simple tree of float
    store_schema, store_state = core.generate(
        'tree[float]',
        initial_state)

    # use the nested type to fill in the state
    tree_schema, tree_state = core.generate(
        store_tree,
        {'store1': None})

    # use the nested type but with an empty dict instead
    fill_schema, fill_state = core.generate(
        store_tree,
        {'store1': {}})

    # supply the whole schema at once instead of registering
    inline_schema, inline_state = core.generate({
        '_type': 'tree',
        '_leaf': {
            'store1.1': {
                '_type': 'float',
                '_default': '1.0'}}},
        {'store1': {}})

    # here is the state we expect from each of these calls
    # to generate
    target_state = {
        'store1': {
            'store1.1': 1.0}}

    # all of the resulting generated states are the same
    assert store_state == tree_state == fill_state == inline_state == target_state


if __name__ == '__main__':
    core = TypeSystem()
    core = register_test_types(core)

    test_reregister_type(core)
    test_generate_default(core)
    test_apply_update(core)
    test_validate_schema(core)
    test_fill_integer(core)
    test_fill_cube(core)
    test_establish_path(core)
    test_overwrite_existing(core)
    test_fill_in_missing_nodes(core)
    test_fill_from_parse(core)
    test_fill_ports(core)
    test_expected_schema(core)
    test_units(core)
    test_serialize_deserialize(core)
    test_project(core)
    test_inherits_from(core)
    test_apply_schema(core)
    test_resolve_schemas(core)
    test_add_reaction(core)
    test_remove_reaction(core)
    test_replace_reaction(core)
    test_unit_conversion(core)
    test_map_type(core)
    test_tree_type(core)
    test_maybe_type(core)
    test_tuple_type(core)
    test_array_type(core)
    test_union_type(core)
    test_union_values(core)
    test_infer_edge(core)
    test_edge_type(core)
    test_edge_complete(core)
    test_foursquare(core)
    test_divide(core)
    test_merge(core)
    test_bind(core)
    test_slice(core)
    test_set_slice(core)
    test_dataclass(core)
    test_enum_type(core)
    test_map_schema(core)
    test_representation(core)
    test_generate(core)
    test_edge_cycle(core)
    test_merge(core)
    test_remove_omitted(core)
    test_union_key_error(core)
    test_tree_equivalence(core)
    test_star_view_project(core)

    # test_slice_edge(core)
    # test_complex_wiring(core)
