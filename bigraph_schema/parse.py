"""
======================
Parse Bigraph Notation
======================
"""

from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor


parameter_examples = {
    'no-parameters': 'simple',
    'one-parameter': 'parameterized[A]',
    'three-parameters': 'parameterized[A,B,C]',
    'nested-parameters': 'nested[outer[inner]]',
    'multiple-nested-parameters': 'nested[outer[inner],other,later[on,there[is],more]]',
    'bars': 'a|b|c|zzzz',
    'union': 'a~b~c~zzzz',
    'typed': 'a:field[yellow,tree,snake]|b:(x:earth|y:cloud|z:sky)',
    'typed_parameters': 'edge[a:int|b:(x:length|y:float),v[zz:float|xx:what]]',
    'inputs_and_outputs': 'edge[input1:float|input2:int,output1:float|output2:int]',
    'tuple': 'what[is,happening|(with:yellow|this:green)|this:now]',
    'units_type': 'length^2*mass/time^1_5'}


parameter_grammar = Grammar(
    """
    expression = merge / union / tree
    merge = tree (bar tree)+
    union = tree (tilde tree)+
    tree = bigraph / type_name
    bigraph = group / nest
    group = paren_left expression paren_right
    nest = symbol colon tree
    type_name = symbol parameter_list?
    parameter_list = square_left expression (comma expression)* square_right
    symbol = ~r"[\w\d-_/*&^%$#@!`+ ]+"
    dot = "."
    colon = ":"
    bar = "|"
    paren_left = "("
    paren_right = ")"
    square_left = "["
    square_right = "]"
    comma = ","
    tilde = "~"
    not_newline = ~r"[^\\n\\r]"*
    newline = ~"[\\n\\r]+"
    ws = ~"\s*"
    """)


class ParameterVisitor(NodeVisitor):
    def visit_expression(self, node, visit):
        return visit[0]


    def visit_union(self, node, visit):
        head = [visit[0]]
        tail = [
            tree['visit'][1]
            for tree in visit[1]['visit']]

        return {
            '_union': head + tail}


    def visit_merge(self, node, visit):
        head = [visit[0]]
        tail = [
            tree['visit'][1]
            for tree in visit[1]['visit']]

        nodes = head + tail

        if all([
            isinstance(tree, dict)
            for tree in nodes]):

            merge = {}
            for tree in nodes:
                merge.update(tree)

            return merge

        else:
            values = []
            for tree in head + tail:
                values.append(tree)

            return tuple(values)


    def visit_tree(self, node, visit):
        return visit[0]

    def visit_bigraph(self, node, visit):
        return visit[0]

    def visit_group(self, node, visit):
        return visit[1]

    def visit_nest(self, node, visit):
        return {visit[0]: visit[2]}

    def visit_type_name(self, node, visit):
        type_name = visit[0]
        type_parameters = visit[1]['visit']
        if len(type_parameters) > 0:
            type_parameters = type_parameters[0]
            return [type_name, type_parameters]
        else:
            return type_name

    def visit_parameter_list(self, node, visit):
        first_type = [visit[1]]
        rest_types = [
            inner['visit'][1]
            for inner in visit[2]['visit']]

        parameters = first_type + rest_types

        return parameters

    def visit_symbol(self, node, visit):
        return node.text

    def generic_visit(self, node, visit):
        return {
            'node': node,
            'visit': visit,
        }


def parse_expression(expression):
    parse = parameter_grammar.parse(expression)
    visitor = ParameterVisitor()
    type_parameters = visitor.visit(parse)

    return type_parameters


def is_type_expression(expression):
    return len(expression) == 2 and isinstance(expression[1], list)


def render_expression(expression):
    if isinstance(expression, str):
        return expression

    elif isinstance(expression, list):
        type_name, parameters = expression
        render = ','.join([
            render_expression(parameter)
            for parameter in parameters])
        return f'{type_name}[{render}]'

    elif isinstance(expression, tuple):
        render = '|'.join([
            render_expression(subexpression)
            for subexpression in expression])
        return f'({render})'

    elif isinstance(expression, dict):
        parts = []
        if '_union' in expression:
            parts = [
                render_expression(part)
                for part in expression['_union']]
            return '~'.join(parts)
        else:
            for key, tree in expression.items():
                render = render_expression(tree)
                parts.append(f'{key}:{render}')

            inner = '|'.join(parts)
            return f'({inner})'


# Test the functions
def test_parse_parameters():
    for key, example in parameter_examples.items():
        types = parse_expression(example)

        print(f'{key}: {example}')
        if types:
            print(f'  {types}')
            print(f'  {render_expression(types)}')


if __name__ == '__main__':
    test_parse_parameters()
