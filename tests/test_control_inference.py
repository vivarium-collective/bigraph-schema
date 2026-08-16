"""A dict carrying a node control (``_control``) is inferred and realized as a
node, so ``_control`` (node identity) survives on an UNTYPED store — it used to be
silently dropped, because a plain-dict schema treats every ``_``-key as metadata."""
import pytest

from bigraph_schema import allocate_core
from bigraph_schema.assembly import ReactionRule, run_reactions

try:
    from bigraph_schema.assembly import Site
except Exception:  # pragma: no cover
    from bigraph_schema.schema import Site


@pytest.fixture
def core():
    return allocate_core()


def _cell():
    return {"cell": {"_control": "cell", "contents": {"biomass": 1.0}}}


def test_discover_types_a_control_dict_as_node(core):
    # the _control-bearing dict is a node, not a structural {_control: string, ...}
    assert "node" in core.render(core.discover({}, _cell())[0])


def test_realize_preserves_control_on_an_untyped_dict(core):
    _, state, _ = core.realize({}, _cell())
    assert state == _cell()          # _control preserved (was dropped before the fix)


def test_reaction_divides_the_untyped_realized_node(core):
    # after realizing an untyped node-dict, a division rule genuinely creates two
    # daughter nodes — the reactive system works end to end on the untyped state.
    _, state, _ = core.realize({}, _cell())
    rule = ReactionRule(
        redex={"cell": {"_control": "cell", "contents": Site()}},
        reactum={"a": {"_control": "cell", "contents": Site()},
                 "b": {"_control": "cell", "contents": Site()}},
        instantiation={"contents": "contents"}, label="divide")
    final, events = run_reactions(state, [rule], max_steps=1)
    assert [e.rule_label for e in events] == ["divide"]
    daughters = {k for k, v in final.items()
                 if isinstance(v, dict) and v.get("_control") == "cell"}
    assert daughters == {"a", "b"}
