"""Tests for the typed Translator kernel (bigraph_schema/translator.py).

A Translator is a typed, *declared* crossing between two schema types.
Applying it via ``Core.cross`` returns a ``Crossed`` value or a
``Refusal`` — never a silent ``None`` or a silently substituted default.
That is the entire point, and test 3 below demonstrates the contrast
directly against the existing (unmodified) ``coerce`` method, which
*does* silently return a default on mismatch.

These tests are purely additive: they exercise new API surface only
(``Translator``, ``Refusal``, ``Crossed``, ``Core.register_translator``,
``Core.cross``, ``Core.compose_translators``, plus ``translator_registry``
on ``Core``) and do not modify or replace any existing test.
"""

import pytest

from bigraph_schema.core import allocate_core
from bigraph_schema.methods.coerce import coerce as existing_coerce
from bigraph_schema.translator import Translator, Refusal, Crossed


@pytest.fixture
def core():
    return allocate_core()


# ---------------------------------------------------------------------------
# 1. register_translator validates well-typedness
# ---------------------------------------------------------------------------

def test_register_translator_rejects_unknown_source_type(core):
    bad = Translator(
        id='bad_source',
        source='nonexistent_type_xyz',
        target='float',
        mode='total',
        cross_fn=float,
    )
    with pytest.raises(ValueError):
        core.register_translator(bad)


def test_register_translator_rejects_unknown_target_type(core):
    bad = Translator(
        id='bad_target',
        source='integer',
        target='nonexistent_type_xyz',
        mode='total',
        cross_fn=float,
    )
    with pytest.raises(ValueError):
        core.register_translator(bad)


def test_register_translator_accepts_valid_types(core):
    t = Translator(
        id='int_to_float',
        source='integer',
        target='float',
        mode='total',
        cross_fn=float,
    )
    registered = core.register_translator(t)
    assert registered is t
    assert core.translator_registry['int_to_float'] is t


# ---------------------------------------------------------------------------
# 2. a total translator crosses successfully
# ---------------------------------------------------------------------------

def test_total_translator_crosses(core):
    core.register_translator(Translator(
        id='int_to_float',
        source='integer',
        target='float',
        mode='total',
        cross_fn=float,
    ))
    result = core.cross('int_to_float', 5)
    assert result == Crossed(5.0)


# ---------------------------------------------------------------------------
# 3. THE EIGENIUS DEMONSTRATION: translator refuses where coerce silently
#    passes a default.
# ---------------------------------------------------------------------------

def test_translator_refuses_where_coerce_silently_passes(core):
    core.register_translator(Translator(
        id='int_to_float',
        source='integer',
        target='float',
        mode='total',
        cross_fn=float,
    ))

    # The translator: applying it to a value that doesn't fit the source
    # type returns a Refusal — a value, not None, and not a silently
    # substituted default.
    refusal = core.cross('int_to_float', 'abc')
    assert isinstance(refusal, Refusal)
    assert 'source' in refusal.reason
    assert refusal.offending == 'abc'

    # The existing (unmodified) coerce method: applying it to the exact
    # same mismatched input silently returns a default int (0), with no
    # indication anything went wrong.
    schema = core.access('integer')
    coerced = existing_coerce(schema, 'abc')
    assert not isinstance(coerced, Refusal)
    assert isinstance(coerced, int)
    assert coerced == 0

    # Same input, same conceptual crossing (string -> integer-ish),
    # opposite epistemics: the translator refuses exactly where coerce
    # silently passes.


# ---------------------------------------------------------------------------
# 4. a refusing cross is a Refusal instance, has a reason, and is not None
# ---------------------------------------------------------------------------

def test_refusal_is_a_value_not_none(core):
    core.register_translator(Translator(
        id='int_to_float',
        source='integer',
        target='float',
        mode='total',
        cross_fn=float,
    ))
    refusal = core.cross('int_to_float', 'abc')
    assert refusal is not None
    assert isinstance(refusal, Refusal)
    assert isinstance(refusal.reason, str) and refusal.reason


# ---------------------------------------------------------------------------
# 5. lossy translator requires a declared loss
# ---------------------------------------------------------------------------

def test_lossy_translator_requires_loss_declaration(core):
    without_loss = Translator(
        id='float_to_int',
        source='float',
        target='integer',
        mode='lossy',
        cross_fn=int,
    )
    with pytest.raises(ValueError):
        core.register_translator(without_loss)

    with_loss = Translator(
        id='float_to_int',
        source='float',
        target='integer',
        mode='lossy',
        cross_fn=int,
        loss='truncates fractional part',
    )
    registered = core.register_translator(with_loss)
    assert registered.loss == 'truncates fractional part'

    result = core.cross('float_to_int', 3.7)
    assert result == Crossed(3)


# ---------------------------------------------------------------------------
# 6. anti-silent-None law: cross_fn returning None becomes a Refusal
# ---------------------------------------------------------------------------

def test_cross_fn_returning_none_becomes_refusal(core):
    core.register_translator(Translator(
        id='swallow',
        source='integer',
        target='integer',
        mode='total',
        cross_fn=lambda value: None,
    ))
    result = core.cross('swallow', 5)
    assert result is not None
    assert isinstance(result, Refusal)
    assert 'None' in result.reason


# ---------------------------------------------------------------------------
# 7. compose_translators: mode = worst of the two, loss concatenated
# ---------------------------------------------------------------------------

def test_compose_translators_total_then_lossy(core):
    core.register_translator(Translator(
        id='int_to_float',
        source='integer',
        target='float',
        mode='total',
        cross_fn=float,
    ))
    core.register_translator(Translator(
        id='float_to_int',
        source='float',
        target='integer',
        mode='lossy',
        cross_fn=int,
        loss='truncates fractional part',
    ))

    composed = core.compose_translators('float_to_int', 'int_to_float')

    assert composed.mode == 'lossy'
    assert 'truncates fractional part' in composed.loss

    result = core.cross(composed.id, 4)
    assert result == Crossed(4)


def test_compose_translators_refusal_threads_through(core):
    core.register_translator(Translator(
        id='int_to_float',
        source='integer',
        target='float',
        mode='total',
        cross_fn=float,
    ))
    core.register_translator(Translator(
        id='float_to_int',
        source='float',
        target='integer',
        mode='lossy',
        cross_fn=int,
        loss='truncates fractional part',
    ))
    composed = core.compose_translators('float_to_int', 'int_to_float')

    result = core.cross(composed.id, 'not an int')
    assert isinstance(result, Refusal)


# ---------------------------------------------------------------------------
# 8. translator_registry survives the core copy performed by allocate_core
# ---------------------------------------------------------------------------

def test_translator_registry_survives_core_copy():
    core_a = allocate_core()
    core_a.register_translator(Translator(
        id='int_to_float',
        source='integer',
        target='float',
        mode='total',
        cross_fn=float,
    ))

    # cross must find the translator registered on the SAME returned core.
    assert core_a.cross('int_to_float', 5) == Crossed(5.0)

    # A freshly allocated core must NOT see translators registered on a
    # different core (proves translator_registry is copied per-instance,
    # like method_registry, and not shared/mutated on a common base).
    core_b = allocate_core()
    assert 'int_to_float' not in core_b.translator_registry
    with pytest.raises(ValueError):
        core_b.cross('int_to_float', 5)


# ---------------------------------------------------------------------------
# 9. existing behavior unchanged (additive-only proof)
# ---------------------------------------------------------------------------

def test_existing_core_behavior_unchanged():
    core = allocate_core()
    assert core.check('float', 5.0) is True
    assert core.check('integer', 5) is True
    assert core.check('integer', 'abc') is False

    schema = core.access('integer')
    assert existing_coerce(schema, '42') == 42
    assert existing_coerce(schema, 'abc') == 0

    resolved = core.resolve(core.access('integer'), core.access('integer'))
    assert resolved is not None


# ---------------------------------------------------------------------------
# 10. a raising cross_fn becomes a Refusal, never propagates
# ---------------------------------------------------------------------------

def test_cross_fn_raising_becomes_refusal_not_propagated(core):
    core.register_translator(Translator(
        id='divides_by_zero',
        source='integer',
        target='integer',
        mode='total',
        cross_fn=lambda value: 1 // 0,
    ))

    # Must NOT raise: cross() catches it and returns a Refusal.
    result = core.cross('divides_by_zero', 5)
    assert isinstance(result, Refusal)
    assert 'raised' in result.reason
    assert result.offending == 5


# ---------------------------------------------------------------------------
# 11. target-type mismatch: cross_fn returns a wrong-typed non-None value
# ---------------------------------------------------------------------------

def test_target_type_mismatch_becomes_refusal(core):
    core.register_translator(Translator(
        id='int_to_bad_float',
        source='integer',
        target='float',
        mode='total',
        cross_fn=lambda value: 'not a float',
    ))

    result = core.cross('int_to_bad_float', 5)
    assert isinstance(result, Refusal)
    assert 'target' in result.reason
    assert result.offending == 'not a float'


# ---------------------------------------------------------------------------
# 12. compose_translators raises ValueError when source/target don't line up
# ---------------------------------------------------------------------------

def test_compose_translators_incompatible_types_raises(core):
    core.register_translator(Translator(
        id='int_to_float',
        source='integer',
        target='float',
        mode='total',
        cross_fn=float,
    ))
    core.register_translator(Translator(
        id='string_to_int',
        source='string',
        target='integer',
        mode='total',
        cross_fn=int,
    ))

    # int_to_float.target ('float') != string_to_int.source ('string')
    with pytest.raises(ValueError):
        core.compose_translators('string_to_int', 'int_to_float')


# ---------------------------------------------------------------------------
# 13. registration/crossing works beyond scalars (a complex type)
# ---------------------------------------------------------------------------

def test_translator_with_complex_map_type(core):
    core.register_translator(Translator(
        id='map_float_to_map_float_scaled',
        source='map[float]',
        target='map[float]',
        mode='total',
        cross_fn=lambda value: {key: item * 2.0 for key, item in value.items()},
    ))

    result = core.cross('map_float_to_map_float_scaled', {'a': 1.0, 'b': 2.5})
    assert result == Crossed({'a': 2.0, 'b': 5.0})

    # Source mismatch still refuses correctly for the complex type too.
    refusal = core.cross('map_float_to_map_float_scaled', 'not a map')
    assert isinstance(refusal, Refusal)
    assert refusal.offending == 'not a map'
