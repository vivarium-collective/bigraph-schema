import fire
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

# from bigraph.bigraph import Control, Node, One, Id, Edge, EdgeGroup, Parallel, Merge, Big, InGroup, Condition, Reaction, Range, Assign, Init, Param, RuleGroup, Rules, Preds, System, BigraphicalReactiveSystem, PARAMETER_SYMBOLS


examples = {
    'no-parameters': 'simple',
    # 'empty-parameters': 'empty[]',
    'one-parameter': 'parameterized[A]',
    'three-parameters': 'parameterized[A,B,C]',
    'nested-parameters': 'nested[outer[inner]]',
    'multiple-nested-parameters': 'nested[outer[inner],other,later[on,there[is],more]]',
}


parameter_grammar = Grammar(
    """
    qualified_type = type_name parameter_list? comma?
    parameter_list = square_left qualified_type+ square_right
    parameter_name = type_name comma?
    type_name = ~r"[\w\d-_|/()*&^%$#@!~`+]+"
    square_left = "["
    square_right = "]"
    comma = ","
    not_newline = ~r"[^\\n\\r]"*
    newline = ~"[\\n\\r]+"
    ws = ~"\s*"
    """)













big = """
    big_source = (cws big_expression semicolon? cws)*
    big_expression = control_declare / bigraph_expression / react_expression / reactive_system / expression

    control_declare = atomic? fun? ctrl control_invoke equals number
    control_invoke = control_symbol control_params?
    control_params = paren_left edge_commas paren_right

    reactive_system = begin system_type system_declaration end
    system_type = brs / pbrs / sbrs
    system_declaration = (cws system_expression semicolon? cws)*
    system_expression = system_assign / system_init / system_rules / system_preds
    system_assign = type variable_name equals system_value
    system_value = range / array / param_name
    system_init = init param_name
    system_rules = rules equals square_left system_rule? (comma system_rule)* square_right
    system_rule = deterministic_rule / nondeterministic_rule
    system_preds = preds equals nondeterministic_rule
    deterministic_rule = paren_left param_commas paren_right
    nondeterministic_rule = squirrel_left param_commas squirrel_right
    type = int / string / float
    range = square_left integer_symbol colon integer_symbol colon integer_symbol square_right
    array = squirrel_left param_commas squirrel_right

    react_expression = fun? react variable_name param_group? equals expression arrow expression instantiation? condition?
    arrow = cws dash square_params? dash arrowhead
    instantiation = at square_params cws
    condition = if in_group (comma in_group)*
    in_group = bang? bigraph in (ctx / param)

    bigraph_expression = big variable_name equals expression
    expression = group / merge / parallel / bigraph
    merge = bigraph (merge_pipe bigraph)+
    parallel = bigraph (parallel_pipe bigraph)+
    bigraph = group / nest / control / edge_group / id / one
    group = paren_left expression paren_right
    nest = control (dot bigraph)+
    control = (control_symbol param_group? edge_group?)
    control_symbol = control_start name_tail

    square_params = square_left param_commas? square_right
    param_group = paren_left param_commas paren_right
    param_commas = param_name additional_param*
    additional_param = comma param_name
    param_name = number / string_name / param_call / variable_name
    param_call = variable_name param_group
    integer_symbol = integer / variable_name

    edge_group = squirrel_left edge_commas squirrel_right
    edge_commas = variable_name additional_edge*
    additional_edge = comma variable_name
    variable_name = variable_start name_tail

    begin = "begin" cws
    brs = "brs" cws
    pbrs = "pbrs" cws
    sbrs = "sbrs" cws
    int = "int" cws
    init = "init" cws
    string = "string" cws
    float = "float" cws
    rules = "rules" cws
    colon = cws ":" cws
    preds = "preds" cws
    end = "end" cws

    id = "id" cws
    one = "1" cws
    if = cws "if" cws
    in = cws "in" cws
    param = "param" cws
    ctx = "ctx" cws
    at = cws "@"
    bang = cws "!" cws
    dash = "-"
    plus = "+"
    arrowhead = ">" cws
    react = "react" cws
    big = "big" cws
    atomic = "atomic" cws
    fun = "fun" cws
    ctrl = "ctrl" cws
    equals = cws "=" cws
    cws = (ws comment? ws)*
    comment = octothorpe not_newline newline?
    octothorpe = "#"
    string_name = quote not_quote quote
    quote = "\\""
    not_quote = ~r"[^\\"]"*
    number = digit+ (dot digit+)?
    integer = digit+
    digit = ~r"[\d]"
    control_start = ~r"[\w]"
    variable_start = ~r"[\w]"
    squirrel_left = "{"
    squirrel_right = "}"
    merge_pipe = cws "|" cws
    parallel_pipe = cws "||" cws
    paren_left = cws "(" cws
    paren_right = cws ")" cws
    square_left = cws "[" cws
    square_right = cws "]" cws
    comma = "," cws
    dot = "."
    semicolon = cws ";" cws
    name_tail = ~r"[-+_'\w\d]"*
    not_newline = ~r"[^\\n\\r]"*
    newline = ~"[\\n\\r]+"
    ws = ~"\s*"
    """






class ParameterVisitor(NodeVisitor):
    def visit_qualified_type(self, node, visit):
        outer_type = visit[0]
        parameters = []
        if len(visit[1]['visit']) > 0:
            parameters = visit[1]['visit'][0]

        return [outer_type, parameters]

    def visit_parameter_list(self, node, visit):
        return visit[1]['visit']

    def visit_parameter_name(self, node, visit):
        return visit[0] # ['node'].text

    def visit_type_name(self, node, visit):
        return node.text

    def generic_visit(self, node, visit):
        return {
            'node': node,
            'visit': visit,
        }












    #     expressions = [
    #         node['visit'][1]
    #         for node in visit]

    #     if len(expressions) == 1:
    #         return expressions[0]

    #     controls = {}
    #     bigraphs = {}
    #     reactions = {}
    #     system = None

    #     for expression in expressions:
    #         if isinstance(expression, Control):
    #             controls[expression.symbol] = expression
    #         elif isinstance(expression, Big):
    #             bigraphs[expression.symbol] = expression
    #         elif isinstance(expression, Reaction):
    #             reactions[expression.symbol] = expression
    #         elif isinstance(expression, System):
    #             system = expression

    #     brs = BigraphicalReactiveSystem(
    #         controls=controls,
    #         bigraphs=bigraphs,
    #         reactions=reactions,
    #         system=system)

    #     return brs

    # def visit_big_expression(self, node, visit):
    #     return visit[0]

    # def visit_reactive_system(self, node, visit):
    #     system_type = visit[1]
    #     declarations = visit[2]

    #     bindings = []
    #     init = None
    #     rules = None
    #     preds = None
    #     for declaration in declarations:
    #         if isinstance(declaration, Assign):
    #             bindings.append(declaration)
    #         elif isinstance(declaration, Init):
    #             init = declaration
    #         elif isinstance(declaration, Rules):
    #             rules = declaration
    #         elif isinstance(declaration, Preds):
    #             preds = declaration

    #     return System(
    #         system_type=system_type,
    #         bindings=bindings,
    #         init=init,
    #         rules=rules,
    #         preds=preds)

    # def visit_system_declaration(self, node, visit):
    #     declarations = [
    #         line['visit'][1]
    #         for line in visit]

    #     return declarations

    # def visit_system_expression(self, node, visit):
    #     return visit[0]

    # def visit_system_assign(self, node, visit):
    #     assign_type = visit[0]
    #     symbol = visit[1]
    #     value = visit[3]
    #     return Assign(
    #         assign_type=assign_type,
    #         symbol=symbol,
    #         value=value)

    # def visit_system_value(self, node, visit):
    #     return visit[0]

    # def visit_system_init(self, node, visit):
    #     return Init(symbol=visit[1])

    # def visit_system_rules(self, node, visit):
    #     rule_groups = visit[3]['visit']
    #     rest = [
    #         rule['visit'][1]
    #         for rule in visit[4]['visit']]
    #     rule_groups.extend(rest)
    #     return Rules(
    #         rule_groups=rule_groups)

    # def visit_system_rule(self, node, visit):
    #     return visit[0]

    # def visit_deterministic_rule(self, node, visit):
    #     rule = RuleGroup(
    #         deterministic=True,
    #         rules=visit[1])
    #     return rule

    # def visit_nondeterministic_rule(self, node, visit):
    #     rule = RuleGroup(
    #         deterministic=False,
    #         rules=visit[1])
    #     return rule

    # def visit_system_preds(self, node, visit):
    #     return Preds(
    #         rules=visit[2])

    # def visit_system_type(self, node, visit):
    #     return node.text.strip()

    # def visit_control_declare(self, node, visit):
    #     atomic = bool(visit[0]['visit'])
    #     fun = bool(visit[1]['visit'])
    #     control = visit[3]['visit']
    #     symbol = control[0]
    #     params = control[1]['visit']
    #     if params:
    #         params = params[0]
    #     arity = visit[5]

    #     return Control(
    #         symbol=symbol,
    #         arity=arity,
    #         atomic=atomic,
    #         fun=tuple(params))

    # def visit_react_expression(self, node, visit):
    #     symbol = visit[2]
    #     param_names = visit[3]['visit']
    #     if len(param_names) > 0:
    #         param_names = param_names[0]
    #     redex = visit[5]
    #     arrow = visit[6]
    #     reactum = visit[7]

    #     instantiation = visit[8]['visit']
    #     if instantiation:
    #         instantiation = instantiation[0]

    #     condition = None
    #     condition_visit = visit[9]['visit']
    #     if condition_visit:
    #         condition = Condition(
    #             groups=condition_visit[0])

    #     return Reaction(
    #         symbol=symbol,
    #         params=param_names,
    #         redex=redex,
    #         arrow=arrow,
    #         reactum=reactum,
    #         instantiation=instantiation,
    #         condition=condition)

    # def visit_arrow(self, node, visit):
    #     params = visit[2]['visit']
    #     if len(params) > 0:
    #         return params[0]
    #     else:
    #         return params

    # def visit_instantiation(self, node, visit):
    #     return visit[1]

    # def visit_condition(self, node, visit):
    #     groups = [visit[1]]
    #     rest = [
    #         inner_visit['visit'][1]
    #         for inner_visit in visit[2]['visit']]
    #     groups.extend(rest)
    #     return groups

    # def visit_in_group(self, node, visit):
    #     negate = len(visit[0]['visit']) > 0
    #     control = visit[1]
    #     target = visit[3]['visit'][0]
    #     return InGroup(
    #         control=control,
    #         target=target,
    #         negate=negate)

    # def visit_bigraph_expression(self, node, visit):
    #     return Big(visit[1], visit[3])

    # def visit_atomic(self, node, visit):
    #     return node.text

    # def visit_fun(self, node, visit):
    #     return node.text

    # def visit_control_params(self, node, visit):
    #     params = [visit[1]['visit'][0]]
    #     tail = visit[1]['visit'][1]['visit']
    #     params.extend(tail)
    #     return params

    # def visit_bang(self, node, visit):
    #     return True

    # def visit_number(self, node, visit):
    #     return node.text

    # def visit_expression(self, node, visit):
    #     return visit[0]

    # def visit_bigraph(self, node, visit):
    #     return visit[0]

    # def visit_id(self, node, visit):
    #     return Id()

    # def visit_one(self, node, visit):
    #     return One()

    # def visit_merge(self, node, visit):
    #     merge = [visit[0]]
    #     tail_nodes = visit[1]['visit']
    #     tail = [
    #         node['visit'][1]
    #         for node in tail_nodes]
    #     merge.extend(tail)
    #     return Merge(merge)

    # def visit_parallel(self, node, visit):
    #     parallel = [visit[0]]
    #     tail_nodes = visit[1]['visit']
    #     tail = [
    #         node['visit'][1]
    #         for node in tail_nodes]
    #     parallel.extend(tail)
    #     return Parallel(parallel)

    # def visit_group(self, node, visit):
    #     return visit[1]

    # def visit_nest(self, node, visit):
    #     root = visit[0]
    #     child = visit[1]['visit'][0]['visit'][1]
    #     root.nest(child)
    #     return root

    # def visit_control(self, node, visit):
    #     control_symbol = visit[0]

    #     param_names = visit[1]['visit']
    #     param_symbols = []
    #     if len(param_names) > 0:
    #         param_names = param_names[0]
    #         param_symbols = [
    #             PARAMETER_SYMBOLS[index]
    #             for index, _ in enumerate(param_names)]

    #     edges = visit[2]['visit']
    #     if len(edges) > 0:
    #         edges = edges[0]
    #     else:
    #         edges = EdgeGroup(edges=[])

    #     return Node(
    #         control=Control(
    #             symbol=control_symbol,
    #             arity=len(edges.edges),
    #             fun=param_symbols),
    #         ports=edges,
    #         params=param_names)

    # def visit_control_symbol(self, node, visit):
    #     return node.text

    # def visit_square_params(self, node, visit):
    #     param_names = visit[1]['visit']
    #     if param_names:
    #         param_names = param_names[0]
    #     return param_names

    # def visit_param_group(self, node, visit):
    #     param_names = visit[1]
    #     return param_names

    # def visit_integer_symbol(self, node, visit):
    #     return visit[0]

    # def visit_param_name(self, node, visit):
    #     return visit[0]

    # def visit_param_call(self, node, visit):
    #     return Param(
    #         symbol=visit[0],
    #         params=visit[1])

    # def visit_param_commas(self, node, visit):
    #     params = [visit[0]]
    #     rest = visit[1]['visit']
    #     params.extend(rest)
    #     return params

    # def visit_additional_param(self, node, visit):
    #     return visit[1]

    # def visit_edge_group(self, node, visit):
    #     edge_symbols = [visit[1]['visit'][0]]
    #     additional_edges = visit[1]['visit'][1]['visit']
    #     edge_symbols.extend(additional_edges)

    #     return EdgeGroup(edges=[
    #         Edge(symbol=symbol)
    #         for symbol in edge_symbols])

    # def visit_additional_edge(self, node, visit):
    #     return visit[1]

    # def visit_range(self, node, visit):
    #     return Range(
    #         start=visit[1],
    #         step=visit[3],
    #         stop=visit[5])

    # def visit_array(self, node, visit):
    #     return visit[1]

    # def visit_variable_name(self, node, visit):
    #     return node.text

    # def visit_string_name(self, node, visit):
    #     return node.text

    # def visit_number(self, node, visit):
    #     if node.text.find('.') >= 0:
    #         return float(node.text)
    #     else:
    #         return int(node.text)

    # def visit_integer(self, node, visit):
    #     return int(node.text)

    # def visit_type(self, node, visit):
    #     return visit[0]

    # def visit_int(self, node, visit):
    #     return node.text.strip()

    # def visit_float(self, node, visit):
    #     return node.text.strip()

    # def visit_string(self, node, visit):
    #     return node.text.strip()

    # def visit_ctx(self, node, visit):
    #     return 'ctx'

    # def visit_param(self, node, visit):
    #     return 'param'

    # def visit_cws(self, node, visit):
    #     return '#'

    # def generic_visit(self, node, visit):
    #     return {
    #         'node': node,
    #         'visit': visit}


def parse_type_parameters(expression):
    parse = parameter_grammar.parse(expression)
    visitor = ParameterVisitor()
    type_parameters = visitor.visit(parse)

    return type_parameters


def test_parse_bigraph():
    for key, example in examples.items():
        types = parse_type_parameters(example)

        print(f'{key}: {example}')
        if types:
            print(types)


if __name__ == '__main__':
    fire.Fire(test_parse_bigraph)
