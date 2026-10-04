import pytest
from bigraph_schema.contract_expr import parse, ExprError

def test_parses_a_conservation_expression():
    ast = parse("abs(sum(outputs.mass) - sum(inputs.mass)) <= tol")
    assert ast is not None  # structure asserted via names_in/evaluate in later tasks

def test_parses_a_name_path():
    assert parse("inputs.biomass > 0") is not None

def test_rejects_unbalanced_parens():
    with pytest.raises(ExprError):
        parse("sum(outputs.mass ==")

def test_rejects_dunder_and_calls_outside_the_whitelist():
    with pytest.raises(ExprError):
        parse("__import__('os')")
    with pytest.raises(ExprError):
        parse("outputs.mass.conjugate()")  # attribute call, not a whitelisted reducer

def test_rejects_an_unknown_reducer():
    with pytest.raises(ExprError):
        parse("total(outputs.mass) == 0")

from bigraph_schema.contract_expr import names_in

def test_names_in_collects_paths_and_excludes_tol():
    names = names_in(parse("abs(sum(outputs.mass) - sum(inputs.mass)) <= tol"))
    assert names == {('outputs', 'mass'), ('inputs', 'mass')}

from bigraph_schema.contract_expr import evaluate

def test_evaluate_conservation_true_within_tol():
    ast = parse("abs(sum(outputs.flux) - sum(inputs.flux)) <= tol")
    env = {('outputs', 'flux'): [1.0, 2.0], ('inputs', 'flux'): [3.0]}
    assert evaluate(ast, env, tol=1e-9) is True

def test_evaluate_precondition():
    assert evaluate(parse("inputs.biomass > 0"), {('inputs', 'biomass'): 0.5}) is True
    assert evaluate(parse("inputs.biomass > 0"), {('inputs', 'biomass'): 0.0}) is False

def test_evaluate_missing_name_raises():
    with pytest.raises(ExprError):
        evaluate(parse("inputs.x > 0"), {})
