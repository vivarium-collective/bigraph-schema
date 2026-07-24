# tests/test_process_contract.py
from bigraph_schema.contract import ProcessContract, resolve_contract

DESC = """Distributes activated RNAPs across TUs by weighted multinomial sampling.

    n_to_activate = round(f_active · n_total_RNAP) - n_active
    p_i = max(0, basal_prob_i + sum_j delta_prob[i,j] · bound_TF_j)
"""

def test_mutable_defaults_isolated():
    a, b = ProcessContract(summary="a"), ProcessContract(summary="b")
    a.inputs["x"] = "y"
    assert b.inputs == {}

def test_from_description_splits_summary_and_math():
    c = ProcessContract.from_description(DESC)
    assert c.summary.startswith("Distributes activated RNAPs")
    assert len(c.math) == 2 and c.math[0].startswith("n_to_activate =")

def test_from_description_none():
    assert ProcessContract.from_description("") is None
    assert ProcessContract.from_description(None) is None

def test_merged_preserves_authored_rows_fills_math():
    authored = ProcessContract(inputs={"RNAs": "reads transcripts"})
    merged = authored.merged_with_description(DESC)
    assert merged.inputs == {"RNAs": "reads transcripts"}   # untouched
    assert merged.math and merged.summary                    # filled from desc

def test_merged_does_not_override_authored_math():
    authored = ProcessContract(summary="mine", math=["x = 1"])
    merged = authored.merged_with_description(DESC)
    assert merged.summary == "mine" and merged.math == ["x = 1"]

class _Declared:
    contract = ProcessContract(inputs={"p": "reads p"})
    description = DESC

class _DescOnly:
    description = DESC

class _Bare:
    """Plain docstring, no math."""

def test_resolve_declared_merges_description():
    c = resolve_contract(_Declared())
    assert c.inputs == {"p": "reads p"}       # authored
    assert c.math                              # merged from description

def test_resolve_description_only():
    c = resolve_contract(_DescOnly())
    assert c.math and not c.inputs

def test_resolve_docstring_fallback_then_none_safe():
    assert resolve_contract(_Bare()).summary == "Plain docstring, no math."
    assert resolve_contract(None) is None
    assert resolve_contract(object()) is None
