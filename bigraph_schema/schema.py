import typing
import numpy as np

from plum import dispatch
from dataclasses import dataclass, is_dataclass


@dataclass(kw_only=True)
class Node():
    _default: object = None

@dataclass(kw_only=True)
class Maybe(Node):
    _value: Node

@dataclass(kw_only=True)
class Union(Node):
    _options: typing.List[
        Node]

@dataclass(kw_only=True)
class Tuple(Node):
    _values: typing.List[
        Node]

@dataclass(kw_only=True)
class Boolean(Node):
    pass

@dataclass(kw_only=True)
class Number(Node):
    pass

@dataclass(kw_only=True)
class Integer(Number):
    pass

@dataclass(kw_only=True)
class Float(Number):
    pass

@dataclass(kw_only=True)
class Delta(Float):
    pass

@dataclass(kw_only=True)
class Nonnegative(Float):
    pass

@dataclass(kw_only=True)
class String(Node):
    pass

@dataclass(kw_only=True)
class Enum(String):
    _values: typing.List[
        str]

@dataclass(kw_only=True)
class List(Node):
    _item: Node
    
@dataclass(kw_only=True)
class Map(Node):
    _key: Node
    _value: Node

@dataclass(kw_only=True)
class Tree(Node):
    _leaf: Node

@dataclass(kw_only=True)
class Dtype(Node):
    _fields: typing.Union[
        str,
        typing.List[
            typing.Tuple[
                str,
                'Dtype']]]

@dataclass(kw_only=True)
class Array(Node):
    _shape: typing.List[
        int]
    _data: Dtype


BASE_TYPES = {
    'node': Node,
    'maybe': Maybe,
    'union': Union,
    'tuple': Tuple,
    'boolean': Boolean,
    'number': Number,
    'integer': Integer,
    'float': Float,
    'delta': Delta,
    'nonnegative': Nonnegative,
    'enum': Enum,
    'list': List,
    'map': Map,
    'tree': Tree,
    'dtype': Dtype,
    'array': Array}


# @dataclass
# class Cell(Node):
#     position: typing.Tuple[Float, Float, Float]
#     radius: Float
#     def __init__(self, cell_data):
#         self.position = cell_data['position']
#         self.radius = cell_data['radius']


# @dispatch
# def merge(current: float, update: Delta):
#     return current + update.value

# @dispatch
# def merge(current: Cell, update: Cell):
#     current.position = merge(current.position, update.position)
#     current.radius = merge(current.radius, update.radius)

#     return current

# @dispatch
# def merge(current, update):
#     return update


# def ColonyProcess(Process):
#     config_schema = {}


#     def inputs(self):
#         return {
#             'environment': 'map[cell]'}

    
#     def outputs(self):
#         return {
#             'environment': 'map[cell]',
#             'mass_delta': 'delta',
#             'energy': 'set_float',
#             'concentrations': 'array[(20|10),float]'}

#             # 'environment': {
#             #     '_type': 'map',
#             #     '_value': {
#             #         '_type': 'cell',
#             #         '_merge': 'merge_cell!'}},
#             # 'mass': {
#             #     '_type': 'float',
#             #     '_divide': 'binomial'}}

    
#     def update(self, state, interval):
#         delta = evolve(
#             state['environment'])

#         return {
#             'environment': delta,
#             'mass_delta': 0.0333}


# def ConstResolver(Resolver):
#     def resolve(self, current, updates):
#         return current


# def MergeAdapter(Adapter):
#     def adapt(self, current, update):
#         return current + update


def test_schema():
    pass


if __name__ == '__main__':
    test_schema()
