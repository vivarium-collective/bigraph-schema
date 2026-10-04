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
