import fire
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor


examples = {
    'no-parameters': 'simple',
    'one-parameter': 'parameterized[A]',
    'three-parameters': 'parameterized[A,B,C]',
    'nested-parameters': 'nested[outer[inner]]',
    'multiple-nested-parameters': 'nested[outer[inner],other,later[on,there[is],more]]',
}


parameter_grammar = Grammar(
    """
    qualified_type = type_name parameter_list? comma?
    parameter_list = square_left qualified_type+ square_right
    type_name = ~r"[\w\d-_|/()*&^%$#@!~`+]+"
    square_left = "["
    square_right = "]"
    comma = ","
    not_newline = ~r"[^\\n\\r]"*
    newline = ~"[\\n\\r]+"
    ws = ~"\s*"
    """)


class ParameterVisitor(NodeVisitor):
    def visit_qualified_type(self, node, visit):
        outer_type = visit[0]
        parameters = []
        if len(visit[1]['visit']) > 0:
            parameters = visit[1]['visit'][0]

        return [outer_type, parameters]

    def visit_parameter_list(self, node, visit):
        return visit[1]['visit']

    def visit_type_name(self, node, visit):
        return node.text

    def generic_visit(self, node, visit):
        return {
            'node': node,
            'visit': visit,
        }


def parse_type_parameters(expression):
    parse = parameter_grammar.parse(expression)
    visitor = ParameterVisitor()
    type_parameters = visitor.visit(parse)

    return type_parameters


def render_type_parameters(type_parameters):
    # inverse of parse_type_parameters
    type_name, parameters = type_parameters
    parameters_render = ','.join([
        render_type_parameters(parameter)
        for parameter in parameters
    ])

    render = type_name
    if len(parameters) > 0:
        render = f'{render}[{parameters_render}]'

    return render


def test_parse_parameters():
    for key, example in examples.items():
        types = parse_type_parameters(example)

        print(f'{key}: {example}')
        if types:
            print(types)
            print(render_type_parameters(types))


if __name__ == '__main__':
    fire.Fire(test_parse_parameters)
