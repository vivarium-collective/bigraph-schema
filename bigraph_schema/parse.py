"""
======================
Parse Bigraph Notation
======================

Parses and renders bigraph-style type expressions used in schema definitions,
such as parameterized types, nested trees, merges (`|`), and unions (`~`).
"""

from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor
from bigraph_schema.schema import Union, Tuple, Node
from bigraph_schema.methods import handle_parameters

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
    'double': 'hello[(3|4),over{owiehf;a832hf9237fh!@#(&$A(HFO@I#}]',
    'double_default': 'hello[(3|4),over]{3,3,3,3,3}',
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
    type_name = symbol parameter_list? default_block?
    parameter_list = square_left expression (comma expression)* square_right
    default_block = curly_left default curly_right
    default = ~r"[^}]*"
    symbol = ~r"[\\w\\d-_/<>*&^%$#@!`+. ]+"
    dot = "."
    colon = ":"
    bar = "|"
    paren_left = "("
    paren_right = ")"
    square_left = "["
    square_right = "]"
    curly_left = "{"
    curly_right = "}"
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
class CoreVisitor(NodeVisitor):
    """Visitor that converts parsed bigraph expressions into schema node structures.

    Operates within a `Core` context, mapping grammar constructs
    (unions, merges, type parameters, and defaults) into dataclass-based nodes.
    Handles normalization of nested expressions (e.g. `tuple[int,float]`,
    `link[a:int|b:string]`, `(x:y|z:w)`) into instances of `Union`, `Tuple`,
    or structured dicts.
    """

    def __init__(self, operation):
        """Initialize with the active `Core`."""
        self.operation = operation

    def visit_expression(self, node, visit):
        """Top-level entry; returns first child."""
        return visit[0]

    def visit_union(self, node, visit):
        """Parse `a~b~c` into a `Union(_options=[a,b,c])`."""
        head = [visit[0]]
        tail = [tree['visit'][1] for tree in visit[1]['visit']]
        return Union(_options=head + tail)

    def visit_merge(self, node, visit):
        """Parse `a|b|c`; dicts merge to one mapping, others form a `Tuple`."""
        head = [visit[0]]
        tail = [tree['visit'][1] for tree in visit[1]['visit']]
        nodes = head + tail
        if all(isinstance(tree, dict) for tree in nodes):
            merged = {}
            for tree in nodes:
                merged.update(tree)
            return merged
        else:
            try:
                values = tuple([int(x) for x in nodes])

                return values
            except Exception:
                return Tuple(_values=nodes)

    def visit_tree(self, node, visit):
        """Delegate directly to nested element."""
        return visit[0]

    def visit_bigraph(self, node, visit):
        """Alias for tree; allows recursion within nested bigraphs."""
        return visit[0]

    def visit_group(self, node, visit):
        """Handle grouped subexpression `( ... )`; return tuple or dict."""
        group_value = visit[1]
        return group_value if isinstance(group_value, (list, tuple, dict, Tuple)) else (group_value,)

    def visit_nest(self, node, visit):
        """Handle `key:subtype` pairs (used in trees/maps).

        The left side of the colon is a dict key — always a plain string,
        not a resolved type.  We use the raw text from the parse node
        rather than the visited (type-resolved) value, since key names
        like ``process`` may collide with registered type names.
        """
        key = node.children[0].text
        return {key: visit[2]}

    def visit_type_name(self, node, visit):
        """Resolve base type, parameters, and defaults into schema nodes."""
        schema = visit[0]

        # Parse parameter list
        type_parameters = [
            parameter
            for parameter in visit[1]['visit']]

        if type_parameters:
            schema = handle_parameters(self.operation, schema, type_parameters[0])

        # Parse default value `{...}`
        default_visit = visit[2]['visit']
        if default_visit:
            default = default_visit[0]
            if isinstance(schema, Node):
                schema._default = default
            elif isinstance(schema, dict):
                schema['_default'] = default

        return schema

    def visit_parameter_list(self, node, visit):
        """Return ordered list of parameters `[A,B,C]`."""
        first = [visit[1]]
        rest = [inner['visit'][1] for inner in visit[2]['visit']]
        return first + rest

    def visit_default_block(self, node, visit):
        """Extract contents of `{...}` blocks."""
        return visit[1]

    def visit_default(self, node, visit):
        """Return text inside default braces as string."""
        return node.text

    def visit_symbol(self, node, visit):
        """Resolve bare symbol names via the operation registry or parse visitor."""
        return self.operation.access(node.text)

    def visit_nothing(self, node, visit):
        """Handle empty productions (e.g., trailing commas)."""
        return None

    def generic_visit(self, node, visit):
        """Fallback: return raw parse node and visited children."""
        return {'node': node, 'visit': visit}


def parsed_leftmost_leaf(parsed):
    if len(parsed.children) == 0:
        return parsed
    else:
        return parsed_leftmost_leaf(parsed.children[0])

def visit_expression(expression, visitor):
    parsed = parameter_grammar.parse(expression)
    leaf = parsed_leftmost_leaf(parsed)
    if hasattr(leaf, 'match') and leaf.match[0] == expression:
        return expression
    else:
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
def _demo_parse_parameters():
    """Test parsing and rendering for all example expressions."""
    for label, expr in parameter_examples.items():
        parsed = parse_expression(expr)
        print(f'{label}: {expr}')
        if parsed:
            print(f'  Parsed: {parsed}')
            print(f'  Rendered: {render_expression(parsed)}')

if __name__ == '__main__':
    _demo_parse_parameters()
