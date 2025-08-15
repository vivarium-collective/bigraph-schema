from __future__ import annotations

import typing
import numpy as np

from plum import dispatch
from dataclasses import dataclass, is_dataclass, field


@dataclass(kw_only=True)
class Node():
    _default: object = None

@dataclass(kw_only=True)
class Union(Node):
    _options: typing.Tuple[Node] = field(default_factory=tuple)

@dataclass(kw_only=True)
class Tuple(Node):
    _values: typing.Tuple[Node] = field(default_factory=tuple)

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
class Dtype(Node):
    _fields: typing.Union[
        str,
        typing.Tuple[
            typing.Tuple[
                str,
                'Dtype']]] = field(default_factory=lambda: 'float64')

@dataclass(kw_only=True)
class Array(Node):
    _shape: typing.Tuple[int] = field(default_factory=tuple)
    _data: Dtype = field(default_factory=lambda: 'float64')

@dataclass(kw_only=True)
class Key(Union):
    _options: typing.Tuple[Node] = (String(), Integer())

@dataclass(kw_only=True)
class Path(List):
    _element: Node = field(default_factory=Key)

# @dataclass(kw_only=True)
# class Jump(Node):
#     path: Path = field(default_factory=Path)
#     wires: Wires = field(default_factory=make_wires)

# @dataclass(kw_only=True)
# class Wire(Union):
#     _options: typing.Tuple[Node] = (Key(), Jump())

def make_wires():
    return Tree(_leaf=Path())

@dataclass(kw_only=True)
class Wires(Tree):
    _leaf: Node = field(default_factory=make_wires)

@dataclass(kw_only=True)
class Schema(Tree):
    _leaf: Node = field(default_factory=Node)

@dataclass(kw_only=True)
class Edge(Node):
    _inputs: Schema = field(default_factory=Schema)
    _outputs: Schema = field(default_factory=Schema)
    inputs: Wires = field(default_factory=Wires)
    outputs: Wires = field(default_factory=Wires)


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
    'string': String,
    'enum': Enum,
    'wrap': Wrap,
    'overwrite': Overwrite,
    'list': List,
    'map': Map,
    'tree': Tree,
    'dtype': Dtype,
    'array': Array,
    'key': Key,
    'path': Path,
    # 'jump': Jump,
    # 'wire': Wire,
    'wires': Wires,
    'schema': Schema,
    'edge': Edge,
}


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
#             'energy': 'float',
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
