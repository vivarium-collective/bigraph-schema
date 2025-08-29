"""
======================
Parse Bigraph Notation
======================

Parses and renders bigraph-style type expressions used in schema definitions,
such as parameterized types, nested trees, merges (`|`), and unions (`~`).
"""

from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

# --- Example Expressions -----------------------------------------------------
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
    'single': 'hello[(3),over]',
    'double': 'hello[(3|4),over]',
    'units_type': 'length^2*mass/time^1_5',
    'nothing': '()'}

# --- Grammar -----------------------------------------------------------------
parameter_grammar = Grammar(
    """
    expression = merge / union / tree / nothing
    merge = tree (bar tree)+
    union = tree (tilde tree)+
    tree = bigraph / type_name
    bigraph = group / nest
    group = paren_left expression paren_right
    nest = symbol colon tree
    type_name = symbol parameter_list?
    parameter_list = square_left expression (comma expression)* square_right
    symbol = ~r"[\\w\\d-_/*&^%$#@!`+ ]+"
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
    ws = ~r"\\s*"
    nothing = ""
    """)


# --- Visitor -----------------------------------------------------------------
class ParameterVisitor(NodeVisitor):
    """Visitor that walks a parsed tree and builds structured type expressions."""

    def visit_expression(self, node, visit):
        return visit[0]

    def visit_union(self, node, visit):
        head = [visit[0]]
        tail = [tree['visit'][1] for tree in visit[1]['visit']]
        return {'_union': head + tail}

    def visit_merge(self, node, visit):
        head = [visit[0]]
        tail = [tree['visit'][1] for tree in visit[1]['visit']]
        nodes = head + tail

        if all(isinstance(tree, dict) for tree in nodes):
            merged = {}
            for tree in nodes:
                merged.update(tree)
            return merged
        else:
            return tuple(nodes)

    def visit_tree(self, node, visit):
        return visit[0]

    def visit_bigraph(self, node, visit):
        return visit[0]

    def visit_group(self, node, visit):
        group_value = visit[1]
        return group_value if isinstance(group_value, (list, tuple, dict)) else (group_value,)

    def visit_nest(self, node, visit):
        return {visit[0]: visit[2]}

    def visit_type_name(self, node, visit):
        type_name = visit[0]
        type_parameters = visit[1]['visit']
        if type_parameters:
            return [type_name, type_parameters[0]]
        return type_name

    def visit_parameter_list(self, node, visit):
        first = [visit[1]]
        rest = [inner['visit'][1] for inner in visit[2]['visit']]
        return first + rest

    def visit_symbol(self, node, visit):
        return node.text

    def visit_nothing(self, node, visit):
        return {}

    def generic_visit(self, node, visit):
        return {'node': node, 'visit': visit}

# --- API ---------------------------------------------------------------------
def visit_expression(expression, visitor):
    parsed = parameter_grammar.parse(expression)
    return visitor.visit(parsed)

def parse_expression(expression):
    """
    Parse a bigraph-style type expression into a structured Python object.
    """
    visitor = ParameterVisitor()
    return visit_expression(expression, visitor)

def is_type_expression(expression):
    """
    Return True if the expression is a parameterized type list.
    """
    return isinstance(expression, list) and len(expression) == 2 and isinstance(expression[1], list)

def render_expression(expression):
    """
    Render a structured type expression back into a bigraph-style string.
    """
    if isinstance(expression, str):
        return expression
    elif isinstance(expression, list):
        type_name, parameters = expression
        inner = ','.join(render_expression(p) for p in parameters)
        return f'{type_name}[{inner}]'
    elif isinstance(expression, tuple):
        return '(' + '|'.join(render_expression(e) for e in expression) + ')'
    elif isinstance(expression, dict):
        if '_union' in expression:
            return '~'.join(render_expression(p) for p in expression['_union'])
        else:
            parts = [f'{k}:{render_expression(v)}' for k, v in expression.items()]
            return f'({ "|".join(parts) })'
    return str(expression)  # fallback for unknown structures

# --- Debugging/Testing ---------------------------------------------------------
def test_parse_parameters():
    """Test parsing and rendering for all example expressions."""
    for label, expr in parameter_examples.items():
        parsed = parse_expression(expr)
        print(f'{label}: {expr}')
        if parsed:
            print(f'  Parsed: {parsed}')
            print(f'  Rendered: {render_expression(parsed)}')

if __name__ == '__main__':
    test_parse_parameters()
