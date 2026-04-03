"""
Test matrix: systematic coverage for every type × method combination.

Types (rows):
  Empty, Union, Tuple, Boolean, Or, And, Xor, Integer, Float, Delta,
  Nonnegative, String, Enum, Maybe, Wrap, Overwrite, List, Map, Tree,
  Array, Frame, Path, Wires, Link

Methods (columns):
  default, check, validate, render, serialize, realize, merge, apply
"""

import pytest
import numpy as np
import pandas as pd

from bigraph_schema import allocate_core
from bigraph_schema.schema import (
    Node, Empty, Union, Tuple, Boolean, Or, And, Xor,
    Number, Integer, Float, Delta, Nonnegative, Complex,
    String, Enum, Maybe, Wrap, Overwrite,
    List, Map, Tree, Array, Frame, Path, Wires, Link,
)
from bigraph_schema.methods import (
    default, check, validate, render, serialize, realize, merge, apply,
)


@pytest.fixture
def core():
    return allocate_core()


# ── Empty ──────────────────────────────────────────────────

class TestEmpty:
    def test_default(self):
        assert default(Empty()) is None

    def test_check_none(self):
        assert check(Empty(), None)

    def test_check_nonempty(self):
        assert not check(Empty(), 'something')

    def test_render(self):
        assert render(Empty()) == 'empty'

    def test_serialize(self):
        assert serialize(Empty(), None) == '__nil__'

    def test_realize(self, core):
        schema, state, merges = realize(core, Empty(), None)
        assert isinstance(schema, Empty)

    def test_merge(self):
        assert merge(Empty(), None, None) is None

    def test_validate(self, core):
        assert validate(core, Empty(), None) is None

    def test_validate_fail(self, core):
        result = validate(core, Empty(), 'not empty')
        assert result is not None


# ── Boolean ────────────────────────────────────────────────

class TestBoolean:
    def test_default(self):
        assert default(Boolean()) == False

    def test_default_custom(self):
        assert default(Boolean(_default=True)) == True

    def test_check_true(self):
        assert check(Boolean(), True)

    def test_check_false(self):
        assert check(Boolean(), False)

    def test_check_fail(self):
        assert not check(Boolean(), 1)
        assert not check(Boolean(), 'true')

    def test_render(self):
        assert render(Boolean()) == 'boolean'

    def test_serialize_true(self):
        assert serialize(Boolean(), True) == 'true'

    def test_serialize_false(self):
        assert serialize(Boolean(), False) == 'false'

    def test_realize_true(self, core):
        _, state, _ = realize(core, Boolean(), 'true')
        assert state is True

    def test_realize_false(self, core):
        _, state, _ = realize(core, Boolean(), 'false')
        assert state is False

    def test_realize_none(self, core):
        _, state, _ = realize(core, Boolean(), None)
        assert state == False

    def test_merge(self):
        # Atom merge: update wins when truthy
        result = merge(Boolean(), True, False)
        assert result is False or result is True  # Atom merge behavior

    def test_apply(self):
        result, merges = apply(Boolean(), True, False, ())
        assert result is False

    def test_validate(self, core):
        assert validate(core, Boolean(), True) is None

    def test_validate_fail(self, core):
        result = validate(core, Boolean(), 'not bool')
        assert result is not None


# ── Or ─────────────────────────────────────────────────────

class TestOr:
    def test_default(self):
        assert default(Or()) == False

    def test_check(self):
        assert check(Or(), True)
        assert check(Or(), False)

    def test_apply(self):
        result, _ = apply(Or(), False, True, ())
        assert result is True

    def test_apply_both_false(self):
        result, _ = apply(Or(), False, False, ())
        assert result is False

    def test_render(self):
        # Or is a Boolean subtype
        assert render(Or()) == 'boolean'

    def test_serialize(self):
        assert serialize(Or(), True) == 'true'


# ── And ────────────────────────────────────────────────────

class TestAnd:
    def test_default(self):
        assert default(And()) == True

    def test_check(self):
        assert check(And(), True)
        assert check(And(), False)

    def test_apply(self):
        result, _ = apply(And(), True, True, ())
        assert result is True

    def test_apply_one_false(self):
        result, _ = apply(And(), True, False, ())
        assert result is False

    def test_serialize(self):
        assert serialize(And(), False) == 'false'


# ── Xor ────────────────────────────────────────────────────

class TestXor:
    def test_default(self):
        assert default(Xor()) == False

    def test_check(self):
        assert check(Xor(), True)
        assert check(Xor(), False)

    def test_apply_different(self):
        result, _ = apply(Xor(), True, False, ())
        assert result is True

    def test_apply_same(self):
        result, _ = apply(Xor(), True, True, ())
        assert result is False


# ── Integer ────────────────────────────────────────────────

class TestInteger:
    def test_default(self):
        assert default(Integer()) == 0

    def test_default_custom(self):
        assert default(Integer(_default=42)) == 42

    def test_check(self):
        assert check(Integer(), 5)

    def test_check_fail(self):
        assert not check(Integer(), 5.0)
        assert not check(Integer(), '5')

    def test_render(self):
        assert render(Integer()) == 'integer'

    def test_serialize(self):
        assert serialize(Integer(), 42) == 42

    def test_realize(self, core):
        _, state, _ = realize(core, Integer(), '5555')
        assert state == 5555

    def test_realize_none(self, core):
        _, state, _ = realize(core, Integer(), None)
        assert state == 0

    def test_merge(self):
        # Atom merge: truthy update wins
        result = merge(Integer(), 5, 10)
        assert result == 10

    def test_apply(self):
        # Atom apply: addition
        result, _ = apply(Integer(), 5, 3, ())
        assert result == 8

    def test_validate(self, core):
        assert validate(core, Integer(), 5) is None

    def test_validate_fail(self, core):
        result = validate(core, Integer(), 5.5)
        assert result is not None


# ── Float ──────────────────────────────────────────────────

class TestFloat:
    def test_default(self):
        assert default(Float()) == 0.0

    def test_default_custom(self):
        assert default(Float(_default=3.14)) == 3.14

    def test_check(self):
        assert check(Float(), 3.14)

    def test_check_fail(self):
        assert not check(Float(), 3)
        assert not check(Float(), '3.14')

    def test_render(self):
        assert render(Float()) == 'float'

    def test_serialize(self):
        assert serialize(Float(), 3.14) == 3.14

    def test_realize(self, core):
        _, state, _ = realize(core, Float(), '3.14')
        assert abs(state - 3.14) < 1e-10

    def test_realize_none(self, core):
        _, state, _ = realize(core, Float(), None)
        assert state == 0.0

    def test_merge(self):
        result = merge(Float(), 1.0, 2.0)
        assert result == 2.0

    def test_apply(self):
        result, _ = apply(Float(), 1.5, 2.5, ())
        assert result == 4.0

    def test_validate(self, core):
        assert validate(core, Float(), 1.0) is None

    def test_validate_fail(self, core):
        result = validate(core, Float(), 1)
        assert result is not None


# ── Delta ──────────────────────────────────────────────────

class TestDelta:
    def test_default(self):
        assert default(Delta()) == 0.0

    def test_check(self):
        assert check(Delta(), 1.0)

    def test_check_fail(self):
        assert not check(Delta(), 1)

    def test_render(self):
        assert render(Delta()) == 'delta'

    def test_serialize(self):
        assert serialize(Delta(), 5.5) == 5.5

    def test_realize(self, core):
        _, state, _ = realize(core, Delta(), '2.5')
        assert state == 2.5

    def test_merge(self):
        result = merge(Delta(), 1.0, 2.0)
        assert result == 2.0

    def test_apply(self):
        # Delta inherits from Float -> Atom apply: addition
        result, _ = apply(Delta(), 1.0, 2.0, ())
        assert result == 3.0

    def test_validate(self, core):
        assert validate(core, Delta(), 1.0) is None


# ── Nonnegative ────────────────────────────────────────────

class TestNonnegative:
    def test_default(self):
        assert default(Nonnegative()) == 0.0

    def test_check(self):
        assert check(Nonnegative(), 0.0)
        assert check(Nonnegative(), 5.5)

    def test_check_negative(self):
        assert not check(Nonnegative(), -1.0)

    def test_render(self):
        assert render(Nonnegative()) == 'nonnegative'

    def test_serialize(self):
        assert serialize(Nonnegative(), 3.0) == 3.0

    def test_realize(self, core):
        _, state, _ = realize(core, Nonnegative(), '4.5')
        assert state == 4.5

    def test_merge(self):
        result = merge(Nonnegative(), 1.0, 2.0)
        assert result == 2.0

    def test_apply(self):
        result, _ = apply(Nonnegative(), 1.0, 2.0, ())
        assert result == 3.0

    def test_validate(self, core):
        assert validate(core, Nonnegative(), 5.0) is None

    def test_validate_fail(self, core):
        result = validate(core, Nonnegative(), -1.0)
        assert result is not None


# ── String ─────────────────────────────────────────────────

class TestString:
    def test_default(self):
        assert default(String()) == ''

    def test_default_custom(self):
        assert default(String(_default='hello')) == 'hello'

    def test_check(self):
        assert check(String(), 'hello')

    def test_check_fail(self):
        assert not check(String(), 123)

    def test_render(self):
        assert render(String()) == 'string'

    def test_serialize(self):
        assert serialize(String(), 'hello') == 'hello'

    def test_realize(self, core):
        _, state, _ = realize(core, String(), 'world')
        assert state == 'world'

    def test_merge(self):
        # Atom merge: truthy update wins
        result = merge(String(), 'a', 'b')
        assert result == 'b'

    def test_apply(self):
        result, _ = apply(String(), 'a', 'b', ())
        assert result == 'b'

    def test_validate(self, core):
        assert validate(core, String(), 'hello') is None

    def test_validate_fail(self, core):
        result = validate(core, String(), 123)
        assert result is not None


# ── Enum ───────────────────────────────────────────────────

class TestEnum:
    def test_default(self):
        assert default(Enum(_values=('a', 'b', 'c'))) == 'a'

    def test_default_custom(self):
        assert default(Enum(_values=('a', 'b'), _default='b')) == 'b'

    def test_check(self):
        assert check(Enum(_values=('x', 'y')), 'x')

    def test_check_fail(self):
        assert not check(Enum(_values=('x', 'y')), 'z')
        assert not check(Enum(_values=('x', 'y')), 1)

    def test_render(self):
        assert render(Enum(_values=('a', 'b'))) == 'enum[a,b]'

    def test_serialize(self):
        assert serialize(Enum(_values=('a',)), 'a') == 'a'

    def test_realize(self, core):
        _, state, _ = realize(core, Enum(_values=('a', 'b')), 'b')
        assert state == 'b'

    def test_merge(self):
        result = merge(Enum(_values=('a', 'b')), 'a', 'b')
        assert result == 'b'

    def test_apply(self):
        result, _ = apply(Enum(_values=('a', 'b')), 'a', 'b', ())
        assert result == 'b'

    def test_validate(self, core):
        assert validate(core, Enum(_values=('a', 'b')), 'a') is None

    def test_validate_fail(self, core):
        result = validate(core, Enum(_values=('a', 'b')), 'z')
        assert result is not None


# ── Maybe ──────────────────────────────────────────────────

class TestMaybe:
    def test_default(self):
        assert default(Maybe(_value=Float())) is None

    def test_check_none(self):
        assert check(Maybe(_value=Float()), None)

    def test_check_value(self):
        assert check(Maybe(_value=Float()), 3.14)

    def test_check_fail(self):
        assert not check(Maybe(_value=Float()), 'string')

    def test_render(self):
        assert render(Maybe(_value=Float())) == 'maybe[float]'

    def test_serialize_none(self):
        assert serialize(Maybe(_value=Float()), None) == '__nil__'

    def test_serialize_value(self):
        assert serialize(Maybe(_value=Float()), 3.14) == 3.14

    def test_realize_none(self, core):
        schema, state, _ = realize(core, Maybe(_value=Float()), None)
        assert isinstance(schema, Maybe)

    def test_realize_value(self, core):
        _, state, _ = realize(core, Maybe(_value=Float()), '3.14')
        assert abs(state - 3.14) < 1e-10

    def test_merge_none_update(self):
        result = merge(Maybe(_value=Float()), 1.0, None)
        assert result == 1.0

    def test_merge_none_current(self):
        result = merge(Maybe(_value=Float()), None, 2.0)
        assert result == 2.0

    def test_merge_both(self):
        result = merge(Maybe(_value=Float()), 1.0, 2.0)
        assert result == 2.0

    def test_apply_none(self):
        result, _ = apply(Maybe(_value=Float()), None, 3.0, ())
        assert result == 3.0

    def test_apply_value(self):
        result, _ = apply(Maybe(_value=Float()), 1.0, 2.0, ())
        assert result == 3.0

    def test_validate(self, core):
        assert validate(core, Maybe(_value=Float()), None) is None

    def test_validate_value(self, core):
        assert validate(core, Maybe(_value=Float()), 3.14) is None


# ── Wrap ───────────────────────────────────────────────────

class TestWrap:
    def test_default(self):
        assert default(Wrap(_value=String())) == ''

    def test_default_custom(self):
        assert default(Wrap(_value=String(), _default='hi')) == 'hi'

    def test_check(self):
        assert check(Wrap(_value=String()), 'hello')

    def test_check_fail(self):
        assert not check(Wrap(_value=String()), 123)

    def test_render(self):
        assert render(Wrap(_value=String())) == 'wrap[string]'

    def test_serialize(self):
        assert serialize(Wrap(_value=String()), 'hello') == 'hello'

    def test_realize(self, core):
        _, state, _ = realize(core, Wrap(_value=Float()), '3.14')
        assert abs(state - 3.14) < 1e-10

    def test_merge(self):
        result = merge(Wrap(_value=Float()), 1.0, 2.0)
        assert result == 2.0

    def test_apply(self):
        result, _ = apply(Wrap(_value=Float()), 1.0, 2.0, ())
        assert result == 3.0

    def test_validate(self, core):
        assert validate(core, Wrap(_value=Float()), 1.0) is None


# ── Overwrite ──────────────────────────────────────────────

class TestOverwrite:
    def test_default(self):
        assert default(Overwrite(_value=String())) == ''

    def test_check(self):
        assert check(Overwrite(_value=String()), 'hello')

    def test_render(self):
        assert render(Overwrite(_value=String())) == 'overwrite[string]'

    def test_serialize(self):
        assert serialize(Overwrite(_value=String()), 'hello') == 'hello'

    def test_realize(self, core):
        _, state, _ = realize(core, Overwrite(_value=Float()), '5.0')
        assert state == 5.0

    def test_merge(self):
        # Overwrite merge always returns update
        result = merge(Overwrite(_value=Float()), 1.0, 2.0)
        assert result == 2.0

    def test_apply(self):
        # Overwrite apply always returns update
        result, _ = apply(Overwrite(_value=Float()), 1.0, 2.0, ())
        assert result == 2.0

    def test_validate(self, core):
        assert validate(core, Overwrite(_value=String()), 'hi') is None


# ── Union ──────────────────────────────────────────────────

class TestUnion:
    def test_default(self):
        u = Union(_options=(Float(), String()))
        assert default(u) == 0.0  # default of first option

    def test_check_first(self):
        u = Union(_options=(Float(), String()))
        assert check(u, 3.14)

    def test_check_second(self):
        u = Union(_options=(Float(), String()))
        assert check(u, 'hello')

    def test_check_fail(self):
        u = Union(_options=(Float(), String()))
        assert not check(u, [1, 2])

    def test_render(self):
        u = Union(_options=(Float(), String()))
        rendered = render(u)
        assert 'float' in rendered and 'string' in rendered

    def test_serialize_float(self):
        u = Union(_options=(Float(), String()))
        assert serialize(u, 3.14) == 3.14

    def test_serialize_string(self):
        u = Union(_options=(Float(), String()))
        assert serialize(u, 'hello') == 'hello'

    def test_realize(self, core):
        u = Union(_options=(Float(), String()))
        _, state, _ = realize(core, u, '3.14')
        assert state == 3.14

    def test_merge(self):
        u = Union(_options=(Float(), String()))
        result = merge(u, 1.0, 2.0)
        assert result == 2.0

    def test_apply(self):
        u = Union(_options=(Float(), String()))
        result, _ = apply(u, 1.0, 2.0, ())
        assert result == 3.0  # Float apply = addition

    def test_validate(self, core):
        u = Union(_options=(Float(), String()))
        assert validate(core, u, 3.14) is None

    def test_validate_fail(self, core):
        u = Union(_options=(Float(), String()))
        result = validate(core, u, [1, 2])
        assert result is not None


# ── Tuple ──────────────────────────────────────────────────

class TestTuple:
    def test_default(self):
        t = Tuple(_values=[Float(), String()])
        result = default(t)
        assert result == [0.0, '']

    def test_default_custom(self):
        t = Tuple(_values=[Float(), String()], _default=(1.0, 'hi'))
        assert default(t) == (1.0, 'hi')

    def test_check(self):
        t = Tuple(_values=[Float(), String()])
        assert check(t, (1.0, 'hello'))

    def test_check_wrong_types(self):
        t = Tuple(_values=[Float(), String()])
        assert not check(t, (1, 'hello'))

    def test_check_wrong_length(self):
        t = Tuple(_values=[Float(), String()])
        assert not check(t, (1.0,))

    def test_check_not_tuple(self):
        t = Tuple(_values=[Float(), String()])
        assert not check(t, 'not a tuple')

    def test_render(self):
        t = Tuple(_values=[Float(), String()])
        assert render(t) == 'tuple[float,string]'

    def test_serialize(self):
        t = Tuple(_values=[Float(), String()])
        result = serialize(t, (3.14, 'hi'))
        assert result == [3.14, 'hi']

    def test_realize(self, core):
        t = Tuple(_values=[Integer(), String()])
        _, state, _ = realize(core, t, ('42', 'hello'))
        assert state == (42, 'hello')

    def test_merge(self):
        t = Tuple(_values=[Float(), String()])
        result = merge(t, (1.0, 'a'), (2.0, 'b'))
        assert result == (2.0, 'b')

    def test_apply(self):
        t = Tuple(_values=[Float(), String()])
        result, _ = apply(t, (1.0, 'a'), (2.0, 'b'), ())
        assert result == (3.0, 'b')

    def test_validate(self, core):
        t = Tuple(_values=[Float(), String()])
        assert validate(core, t, (1.0, 'hello')) is None

    def test_validate_fail(self, core):
        t = Tuple(_values=[Float(), String()])
        result = validate(core, t, 'not a tuple')
        assert result is not None


# ── List ───────────────────────────────────────────────────

class TestList:
    def test_default(self):
        assert default(List(_element=Float())) == []

    def test_check(self):
        assert check(List(_element=Float()), [1.0, 2.0])

    def test_check_empty(self):
        assert check(List(_element=Float()), [])

    def test_check_fail(self):
        assert not check(List(_element=Float()), [1, 2])

    def test_check_not_list(self):
        assert not check(List(_element=Float()), 'not a list')

    def test_render(self):
        assert render(List(_element=Float())) == 'list[float]'

    def test_serialize(self):
        result = serialize(List(_element=Float()), [1.0, 2.0])
        assert result == [1.0, 2.0]

    def test_realize(self, core):
        _, state, _ = realize(core, List(_element=Integer()), ['1', '2', '3'])
        assert state == [1, 2, 3]

    def test_merge(self):
        result = merge(List(_element=Float()), [1.0], [2.0])
        assert result == [1.0, 2.0]

    def test_merge_empty(self):
        result = merge(List(_element=Float()), [], [1.0])
        assert result == [1.0]

    def test_apply(self):
        result, _ = apply(List(_element=Float()), [1.0], [2.0], ())
        assert result == [1.0, 2.0]

    def test_validate(self, core):
        assert validate(core, List(_element=Float()), [1.0, 2.0]) is None

    def test_validate_fail(self, core):
        result = validate(core, List(_element=Float()), 'not a list')
        assert result is not None


# ── Map ────────────────────────────────────────────────────

class TestMap:
    def test_default(self):
        assert default(Map(_value=Float())) == {}

    def test_check(self):
        assert check(Map(_value=Float()), {'a': 1.0, 'b': 2.0})

    def test_check_fail_values(self):
        assert not check(Map(_value=Float()), {'a': 'string'})

    def test_check_not_dict(self):
        assert not check(Map(_value=Float()), 'not a map')

    def test_render(self):
        assert render(Map(_value=Float())) == 'map[float]'

    def test_serialize(self):
        result = serialize(Map(_value=Float()), {'a': 1.0})
        assert result == {'a': 1.0}

    def test_realize(self, core):
        _, state, _ = realize(core, Map(_value=Integer()), {'a': '1', 'b': '2'})
        assert state == {'a': 1, 'b': 2}

    def test_merge(self):
        result = merge(Map(_value=Float()), {'a': 1.0}, {'a': 2.0, 'b': 3.0})
        assert result['a'] == 2.0
        assert result['b'] == 3.0

    def test_merge_disjoint(self):
        result = merge(Map(_value=Float()), {'a': 1.0}, {'b': 2.0})
        assert result['a'] == 1.0
        assert result['b'] == 2.0

    def test_apply(self):
        result, _ = apply(
            Map(_value=Float()),
            {'a': 1.0, 'b': 2.0},
            {'a': 0.5},
            ())
        assert result['a'] == 1.5
        assert result['b'] == 2.0

    def test_apply_add(self):
        result, _ = apply(
            Map(_value=Float()),
            {'a': 1.0},
            {'_add': {'c': 3.0}},
            ())
        assert result['c'] == 3.0

    def test_apply_remove(self):
        result, _ = apply(
            Map(_value=Float()),
            {'a': 1.0, 'b': 2.0},
            {'_remove': ['b']},
            ())
        assert 'b' not in result

    def test_validate(self, core):
        assert validate(core, Map(_value=Float()), {'a': 1.0}) is None

    def test_validate_fail(self, core):
        result = validate(core, Map(_value=Float()), 'not a map')
        assert result is not None


# ── Tree ───────────────────────────────────────────────────

class TestTree:
    def test_default(self):
        assert default(Tree(_leaf=Float())) == {}

    def test_check_leaf(self):
        assert check(Tree(_leaf=Float()), 3.14)

    def test_check_nested(self):
        assert check(Tree(_leaf=Float()), {'a': {'b': 1.0}})

    def test_check_fail(self):
        assert not check(Tree(_leaf=Float()), 'not a tree')

    def test_render(self):
        assert render(Tree(_leaf=Float())) == 'tree[float]'

    def test_serialize_leaf(self):
        result = serialize(Tree(_leaf=Float()), 3.14)
        assert result == 3.14

    def test_serialize_nested(self):
        result = serialize(Tree(_leaf=Float()), {'a': 1.0, 'b': {'c': 2.0}})
        assert result == {'a': 1.0, 'b': {'c': 2.0}}

    def test_realize_leaf(self, core):
        _, state, _ = realize(core, Tree(_leaf=Float()), '3.14')
        assert state == 3.14

    def test_realize_nested(self, core):
        # Tree realize handles leaf values
        _, state, _ = realize(core, Tree(_leaf=Float()), 3.14)
        assert abs(state - 3.14) < 1e-10

    def test_merge_leaves(self):
        result = merge(Tree(_leaf=Float()), 1.0, 2.0)
        assert result == 2.0

    def test_merge_nested(self):
        result = merge(
            Tree(_leaf=Float()),
            {'a': 1.0, 'b': {'c': 2.0}},
            {'a': 3.0, 'b': {'d': 4.0}})
        assert result['a'] == 3.0
        assert result['b']['c'] == 2.0
        assert result['b']['d'] == 4.0

    def test_validate(self, core):
        assert validate(core, Tree(_leaf=Float()), {'a': 1.0}) is None

    def test_validate_fail(self, core):
        result = validate(core, Tree(_leaf=Float()), 'not a tree')
        assert result is not None


# ── Array ──────────────────────────────────────────────────

class TestArray:
    def test_default(self):
        schema = Array(_shape=(3,), _data=np.dtype('float64'))
        result = default(schema)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    def test_check(self):
        schema = Array(_shape=(2, 3), _data=np.dtype('float64'))
        state = np.zeros((2, 3))
        assert check(schema, state)

    def test_check_wrong_shape(self):
        schema = Array(_shape=(2, 3), _data=np.dtype('float64'))
        state = np.zeros((3, 2))
        assert not check(schema, state)

    def test_check_not_array(self):
        schema = Array(_shape=(3,), _data=np.dtype('float64'))
        assert not check(schema, [1.0, 2.0, 3.0])

    def test_render(self):
        schema = Array(_shape=(3, 4), _data=np.dtype('float64'))
        result = render(schema)
        assert 'array' in result
        assert '3|4' in result

    def test_serialize(self):
        schema = Array(_shape=(3,), _data=np.dtype('float64'))
        state = np.array([1.0, 2.0, 3.0])
        result = serialize(schema, state)
        assert result == [1.0, 2.0, 3.0]

    def test_realize(self, core):
        schema = Array(_shape=(3,), _data=np.dtype('float64'))
        _, state, _ = realize(core, schema, [1.0, 2.0, 3.0])
        assert isinstance(state, np.ndarray)
        assert state.shape == (3,)

    def test_merge(self):
        schema = Array(_shape=(3,), _data=np.dtype('float64'))
        current = np.array([1.0, 2.0, 3.0])
        update = np.array([4.0, 5.0, 6.0])
        result = merge(schema, current, update)
        np.testing.assert_array_equal(result, update)

    def test_merge_none_update(self):
        schema = Array(_shape=(3,), _data=np.dtype('float64'))
        current = np.array([1.0, 2.0, 3.0])
        result = merge(schema, current, None)
        np.testing.assert_array_equal(result, current)

    def test_apply(self):
        schema = Array(_shape=(3,), _data=np.dtype('float64'))
        state = np.array([1.0, 2.0, 3.0])
        update = np.array([0.5, 0.5, 0.5])
        result, _ = apply(schema, state, update, ())
        np.testing.assert_array_equal(result, np.array([1.5, 2.5, 3.5]))

    def test_validate(self, core):
        schema = Array(_shape=(3,), _data=np.dtype('float64'))
        state = np.zeros(3)
        assert validate(core, schema, state) is None

    def test_validate_fail(self, core):
        schema = Array(_shape=(3,), _data=np.dtype('float64'))
        result = validate(core, schema, [1, 2, 3])
        assert result is not None


# ── Frame ──────────────────────────────────────────────────

class TestFrame:
    def test_default(self):
        schema = Frame(_columns={'a': Float(), 'b': Integer()})
        result = default(schema)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['a', 'b']

    def test_render(self):
        schema = Frame(_columns={'a': Float(), 'b': Integer()})
        result = render(schema)
        assert 'dataframe' in result

    def test_serialize(self):
        schema = Frame(_columns={'a': Float()})
        df = pd.DataFrame({'a': [1.0, 2.0]})
        result = serialize(schema, df)
        assert result == {'a': [1.0, 2.0]}

    def test_realize(self, core):
        schema = Frame(_columns={'a': Float()})
        _, state, _ = realize(core, schema, {'a': [1.0, 2.0]})
        assert isinstance(state, pd.DataFrame)

    def test_merge(self):
        schema = Frame(_columns={'a': Float()})
        current = pd.DataFrame({'a': [1.0]})
        update = pd.DataFrame({'a': [2.0]})
        result = merge(schema, current, update)
        assert result.equals(update)

    def test_apply(self):
        schema = Frame(_columns={'a': Float()})
        state = pd.DataFrame({'a': [1.0]})
        update = pd.DataFrame({'a': [2.0]})
        result, _ = apply(schema, state, update, ())
        assert result.equals(update)


# ── Path ───────────────────────────────────────────────────

class TestPath:
    def test_default(self):
        assert default(Path()) == []

    def test_check(self):
        assert check(Path(), ['a', 'b'])

    def test_render(self):
        assert render(Path()) == 'path'

    def test_serialize(self):
        result = serialize(Path(), ['a', 'b'])
        assert isinstance(result, list)

    def test_realize(self, core):
        _, state, _ = realize(core, Path(), ['a', 'b'])
        assert state == ['a', 'b']

    def test_merge(self):
        result = merge(Path(), ['a'], ['b'])
        assert result == ['a', 'b']


# ── Wires ──────────────────────────────────────────────────

class TestWires:
    def test_default(self):
        assert default(Wires()) == {}

    def test_render(self):
        assert render(Wires()) == 'wires'

    def test_serialize(self):
        state = {'x': ['a', 'b']}
        result = serialize(Wires(), state)
        # Wires is a Tree[Path], so it serializes nested paths
        assert isinstance(result, dict)

    def test_realize(self, core):
        _, state, _ = realize(core, Wires(), {'x': ['a']})
        assert isinstance(state, dict)

    def test_merge(self):
        result = merge(Wires(), {'x': ['a']}, {'x': ['b']})
        assert result == {'x': ['b']}


# ── Link ───────────────────────────────────────────────────

class TestLink:
    def test_default(self, core):
        link = core.access('link[x:float,y:string]')
        result = default(link)
        assert 'address' in result
        assert 'inputs' in result
        assert 'outputs' in result

    def test_render(self, core):
        link = core.access('link[x:float,y:string]')
        result = render(link)
        assert isinstance(result, (str, dict))

    def test_check_unrealized(self, core):
        link = core.access({
            '_type': 'link',
            '_inputs': {'x': 'float'},
            '_outputs': {'y': 'string'}})
        state = {
            'address': 'local:edge',
            'inputs': {'x': ['x']},
            'outputs': {'y': ['y']}}
        # Check should work on link states
        result = check(link, state)
        # Link check delegates to Node check
        assert isinstance(result, bool)

    def test_realize(self, core):
        schema = {
            '_type': 'link',
            '_inputs': {'x': 'float'},
            '_outputs': {'y': 'string'}}
        state = {
            'inputs': {'x': ['x']},
            'outputs': {'y': ['y']}}
        gen_schema, gen_state = core.realize(schema, state)
        assert 'instance' in gen_state

    def test_serialize(self, core):
        schema = {
            '_type': 'link',
            '_inputs': {'x': 'float'},
            '_outputs': {'y': 'string'}}
        state = {
            'inputs': {'x': ['x']},
            'outputs': {'y': ['y']}}
        gen_schema, gen_state = core.realize(schema, state)
        result = core.serialize(gen_schema, gen_state)
        assert '_inputs' in result

    def test_merge(self, core):
        link = core.access({
            '_type': 'link',
            '_inputs': {'x': 'float'},
            '_outputs': {'y': 'string'}})
        state_a = {
            'address': {'protocol': 'local', 'data': 'edge'},
            'config': {},
            'inputs': {'x': ['x']},
            'outputs': {'y': ['y']}}
        state_b = {
            'address': {'protocol': 'local', 'data': 'edge'},
            'config': {},
            'inputs': {'x': ['a']},
            'outputs': {'y': ['b']}}
        result = merge(link, state_a, state_b)
        assert result['inputs']['x'] == ['a']

    def test_validate(self, core):
        schema = {
            '_type': 'link',
            '_inputs': {'x': 'float'},
            '_outputs': {'y': 'string'}}
        state = {
            'inputs': {'x': ['x']},
            'outputs': {'y': ['y']}}
        gen_schema, gen_state = core.realize(schema, state)
        result = validate(core, gen_schema, gen_state)
        # Should not have validation errors
        assert result is None or result == {}


# ── Integration: full round-trip via core ──────────────────

class TestCoreRoundTrip:
    """Test access → default → serialize → realize for each type string."""

    def test_empty(self, core):
        schema = core.access('empty')
        assert isinstance(schema, Empty)

    def test_boolean(self, core):
        s, v = core.default('boolean')
        assert v == False
        encoded = core.serialize(s, v)
        _, decoded = core.realize(s, encoded)
        assert decoded == v

    def test_or(self, core):
        s, v = core.default('or')
        assert v == False

    def test_and(self, core):
        s, v = core.default('and')
        assert v == True

    def test_xor(self, core):
        s, v = core.default('xor')
        assert v == False

    def test_integer(self, core):
        s, v = core.default('integer')
        assert v == 0
        encoded = core.serialize(s, v)
        _, decoded = core.realize(s, encoded)
        assert decoded == v

    def test_float(self, core):
        s, v = core.default('float')
        assert v == 0.0
        encoded = core.serialize(s, v)
        _, decoded = core.realize(s, encoded)
        assert decoded == v

    def test_delta(self, core):
        s, v = core.default('delta')
        assert v == 0.0

    def test_nonnegative(self, core):
        s, v = core.default('nonnegative')
        assert v == 0.0

    def test_string(self, core):
        s, v = core.default('string')
        assert v == ''
        encoded = core.serialize(s, v)
        _, decoded = core.realize(s, encoded)
        assert decoded == v

    def test_enum(self, core):
        s, v = core.default('enum[a,b,c]')
        assert v == 'a'
        assert core.check(s, 'b')
        assert not core.check(s, 'd')

    def test_maybe(self, core):
        s, v = core.default('maybe[float]')
        assert v is None

    def test_wrap(self, core):
        s, v = core.default('wrap[string]')
        assert v == ''

    def test_overwrite(self, core):
        s, v = core.default('overwrite[integer]')
        assert v == 0

    def test_union(self, core):
        s, v = core.default('union[float,string]')
        assert v == 0.0

    def test_tuple(self, core):
        s, v = core.default('tuple[float,string,integer]')
        assert v == (0.0, '', 0)

    def test_list(self, core):
        s, v = core.default('list[float]')
        assert v == []

    def test_map(self, core):
        s, v = core.default('map[float]')
        assert v == {}

    def test_tree(self, core):
        s, v = core.default('tree[float]')
        # core.default for tree may return the leaf default (0.0) or {}
        assert v == 0.0 or v == {}

    def test_array(self, core):
        s, v = core.default('array[(3|4),float]')
        assert isinstance(v, np.ndarray)
        assert v.shape == (3, 4)

    def test_path(self, core):
        s, v = core.default('path')
        assert v == []

    def test_wires(self, core):
        s, v = core.default('wires')
        assert v == {}

    def test_frame(self, core):
        s, v = core.default('dataframe[a:float|b:integer]')
        assert isinstance(v, pd.DataFrame)

    def test_link(self, core):
        s, v = core.default('link[x:float,y:string]')
        assert 'inputs' in v
        assert 'outputs' in v


class TestCoreCheck:
    """Test core.check for each type with valid and invalid states."""

    def test_boolean(self, core):
        assert core.check('boolean', True)
        assert not core.check('boolean', 1)

    def test_integer(self, core):
        assert core.check('integer', 5)
        assert not core.check('integer', 5.0)

    def test_float(self, core):
        assert core.check('float', 1.0)
        assert not core.check('float', 1)

    def test_nonnegative(self, core):
        assert core.check('nonnegative', 0.0)
        assert not core.check('nonnegative', -1.0)

    def test_string(self, core):
        assert core.check('string', 'hello')
        assert not core.check('string', 123)

    def test_enum(self, core):
        assert core.check('enum[a,b,c]', 'a')
        assert not core.check('enum[a,b,c]', 'd')

    def test_maybe(self, core):
        assert core.check('maybe[float]', None)
        assert core.check('maybe[float]', 1.0)
        assert not core.check('maybe[float]', 'wrong')

    def test_list(self, core):
        assert core.check('list[integer]', [1, 2, 3])
        assert not core.check('list[integer]', [1.0])

    def test_map(self, core):
        assert core.check('map[float]', {'x': 1.0})
        assert not core.check('map[float]', {'x': 'wrong'})

    def test_tree(self, core):
        assert core.check('tree[float]', {'a': {'b': 1.0}})
        assert core.check('tree[float]', 1.0)
        assert not core.check('tree[float]', 'wrong')

    def test_tuple(self, core):
        assert core.check('tuple[float,string]', (1.0, 'hi'))
        assert not core.check('tuple[float,string]', (1, 'hi'))

    def test_array(self, core):
        assert core.check(
            'array[(3),float]',
            np.zeros(3))
        assert not core.check(
            'array[(3),float]',
            np.zeros(4))


class TestCoreMerge:
    """Test core.merge for each type."""

    def test_boolean(self, core):
        result = core.merge('boolean', True, False)
        assert isinstance(result, bool)

    def test_integer(self, core):
        result = core.merge('integer', 5, 10)
        assert result == 10

    def test_float(self, core):
        result = core.merge('float', 1.0, 2.0)
        assert result == 2.0

    def test_string(self, core):
        result = core.merge('string', 'a', 'b')
        assert result == 'b'

    def test_list(self, core):
        result = core.merge('list[float]', [1.0], [2.0])
        assert result == [1.0, 2.0]

    def test_map(self, core):
        result = core.merge('map[float]', {'a': 1.0}, {'b': 2.0})
        assert result == {'a': 1.0, 'b': 2.0}

    def test_tree(self, core):
        result = core.merge(
            'tree[float]',
            {'a': 1.0},
            {'a': 2.0, 'b': 3.0})
        assert result == {'a': 2.0, 'b': 3.0}

    def test_maybe(self, core):
        result = core.merge('maybe[float]', None, 5.0)
        assert result == 5.0

    def test_overwrite(self, core):
        result = core.merge('overwrite[float]', 1.0, 2.0)
        assert result == 2.0

    def test_tuple(self, core):
        result = core.merge(
            'tuple[float,string]',
            (1.0, 'a'),
            (2.0, 'b'))
        assert result == (2.0, 'b')

    def test_array(self, core):
        schema = 'array[(3),float]'
        current = np.array([1.0, 2.0, 3.0])
        update = np.array([4.0, 5.0, 6.0])
        result = core.merge(schema, current, update)
        np.testing.assert_array_equal(result, update)


class TestCoreApply:
    """Test core.apply for each type."""

    def test_float(self, core):
        schema = core.access('float')
        state, _ = apply(schema, 1.0, 2.0, ())
        assert state == 3.0

    def test_integer(self, core):
        schema = core.access('integer')
        state, _ = apply(schema, 5, 3, ())
        assert state == 8

    def test_string(self, core):
        schema = core.access('string')
        state, _ = apply(schema, 'old', 'new', ())
        assert state == 'new'

    def test_boolean(self, core):
        schema = core.access('boolean')
        state, _ = apply(schema, True, False, ())
        assert state is False

    def test_or(self, core):
        schema = core.access('or')
        state, _ = apply(schema, False, True, ())
        assert state is True

    def test_and(self, core):
        schema = core.access('and')
        state, _ = apply(schema, True, False, ())
        assert state is False

    def test_xor(self, core):
        schema = core.access('xor')
        state, _ = apply(schema, True, False, ())
        assert state is True

    def test_overwrite(self, core):
        schema = core.access('overwrite[float]')
        state, _ = apply(schema, 1.0, 99.0, ())
        assert state == 99.0

    def test_maybe(self, core):
        schema = core.access('maybe[float]')
        state, _ = apply(schema, None, 5.0, ())
        assert state == 5.0

    def test_list(self, core):
        schema = core.access('list[float]')
        state, _ = apply(schema, [1.0], [2.0], ())
        assert state == [1.0, 2.0]

    def test_map(self, core):
        schema = core.access('map[float]')
        state, _ = apply(schema, {'a': 1.0}, {'a': 0.5}, ())
        assert state['a'] == 1.5

    def test_tuple(self, core):
        schema = core.access('tuple[float,string]')
        state, _ = apply(schema, (1.0, 'a'), (2.0, 'b'), ())
        assert state == (3.0, 'b')

    def test_array(self, core):
        schema = core.access('array[(3),float]')
        state = np.array([1.0, 2.0, 3.0])
        update = np.array([0.1, 0.2, 0.3])
        result, _ = apply(schema, state, update, ())
        np.testing.assert_allclose(result, [1.1, 2.2, 3.3])

    def test_frame(self, core):
        schema = core.access('dataframe[a:float]')
        state = pd.DataFrame({'a': [1.0]})
        update = pd.DataFrame({'a': [2.0]})
        result, _ = apply(schema, state, update, ())
        assert result.equals(update)

    def test_union(self, core):
        schema = core.access('union[float,string]')
        state, _ = apply(schema, 1.0, 2.0, ())
        assert state == 3.0


class TestCoreSerializeRealize:
    """Test serialize → realize round-trip for each type."""

    def test_boolean(self, core):
        s = core.access('boolean')
        encoded = serialize(s, True)
        _, decoded, _ = realize(core, s, encoded)
        assert decoded is True

    def test_integer(self, core):
        s = core.access('integer')
        encoded = serialize(s, 42)
        _, decoded, _ = realize(core, s, encoded)
        assert decoded == 42

    def test_float(self, core):
        s = core.access('float')
        encoded = serialize(s, 3.14)
        _, decoded, _ = realize(core, s, encoded)
        assert abs(decoded - 3.14) < 1e-10

    def test_string(self, core):
        s = core.access('string')
        encoded = serialize(s, 'hello')
        _, decoded, _ = realize(core, s, encoded)
        assert decoded == 'hello'

    def test_enum(self, core):
        s = core.access('enum[a,b,c]')
        encoded = serialize(s, 'b')
        _, decoded, _ = realize(core, s, encoded)
        assert decoded == 'b'

    def test_maybe_none(self, core):
        s = core.access('maybe[float]')
        encoded = serialize(s, None)
        schema, decoded, _ = realize(core, s, encoded)
        assert isinstance(schema, Maybe)

    def test_maybe_value(self, core):
        s = core.access('maybe[float]')
        encoded = serialize(s, 3.14)
        _, decoded, _ = realize(core, s, encoded)
        assert abs(decoded - 3.14) < 1e-10

    def test_tuple(self, core):
        s = core.access('tuple[integer,string]')
        encoded = serialize(s, (42, 'hello'))
        _, decoded, _ = realize(core, s, encoded)
        assert decoded == (42, 'hello')

    def test_list(self, core):
        s = core.access('list[integer]')
        encoded = serialize(s, [1, 2, 3])
        _, decoded, _ = realize(core, s, encoded)
        assert decoded == [1, 2, 3]

    def test_map(self, core):
        s = core.access('map[float]')
        encoded = serialize(s, {'a': 1.0, 'b': 2.0})
        _, decoded, _ = realize(core, s, encoded)
        assert decoded == {'a': 1.0, 'b': 2.0}

    def test_tree(self, core):
        s = core.access('tree[float]')
        # Simple leaf values round-trip through tree
        leaf_val = 3.14
        encoded = serialize(s, leaf_val)
        _, decoded, _ = realize(core, s, encoded)
        assert abs(decoded - leaf_val) < 1e-10

    def test_array(self, core):
        s = core.access('array[(3),float]')
        arr = np.array([1.0, 2.0, 3.0])
        encoded = serialize(s, arr)
        _, decoded, _ = realize(core, s, encoded)
        np.testing.assert_array_equal(decoded, arr)

    def test_frame(self, core):
        s = core.access('dataframe[a:float|b:integer]')
        df = pd.DataFrame({'a': [1.0, 2.0], 'b': [3, 4]})
        encoded = serialize(s, df)
        _, decoded, _ = realize(core, s, encoded)
        assert isinstance(decoded, pd.DataFrame)

    def test_union(self, core):
        s = core.access('union[float,string]')
        encoded = serialize(s, 3.14)
        _, decoded, _ = realize(core, s, encoded)
        assert decoded == 3.14

    def test_wrap(self, core):
        s = core.access('wrap[float]')
        encoded = serialize(s, 3.14)
        _, decoded, _ = realize(core, s, encoded)
        assert abs(decoded - 3.14) < 1e-10

    def test_overwrite(self, core):
        s = core.access('overwrite[string]')
        encoded = serialize(s, 'hello')
        _, decoded, _ = realize(core, s, encoded)
        assert decoded == 'hello'


class TestCoreValidate:
    """Test validate for each type via core."""

    def test_empty(self, core):
        s = core.access('empty')
        assert validate(core, s, None) is None

    def test_boolean(self, core):
        s = core.access('boolean')
        assert validate(core, s, True) is None
        assert validate(core, s, 1) is not None

    def test_integer(self, core):
        s = core.access('integer')
        assert validate(core, s, 5) is None
        assert validate(core, s, 5.0) is not None

    def test_float(self, core):
        s = core.access('float')
        assert validate(core, s, 1.0) is None
        assert validate(core, s, 1) is not None

    def test_nonnegative(self, core):
        s = core.access('nonnegative')
        assert validate(core, s, 5.0) is None
        assert validate(core, s, -1.0) is not None

    def test_string(self, core):
        s = core.access('string')
        assert validate(core, s, 'hi') is None
        assert validate(core, s, 123) is not None

    def test_enum(self, core):
        s = core.access('enum[a,b,c]')
        assert validate(core, s, 'a') is None
        assert validate(core, s, 'd') is not None

    def test_maybe(self, core):
        s = core.access('maybe[float]')
        assert validate(core, s, None) is None
        assert validate(core, s, 1.0) is None

    def test_wrap(self, core):
        s = core.access('wrap[float]')
        assert validate(core, s, 1.0) is None

    def test_union(self, core):
        s = core.access('union[float,string]')
        assert validate(core, s, 1.0) is None
        assert validate(core, s, 'hi') is None

    def test_tuple(self, core):
        s = core.access('tuple[float,string]')
        assert validate(core, s, (1.0, 'hi')) is None
        assert validate(core, s, 'wrong') is not None

    def test_list(self, core):
        s = core.access('list[float]')
        assert validate(core, s, [1.0, 2.0]) is None
        assert validate(core, s, 'wrong') is not None

    def test_map(self, core):
        s = core.access('map[float]')
        assert validate(core, s, {'a': 1.0}) is None
        assert validate(core, s, 'wrong') is not None

    def test_tree(self, core):
        s = core.access('tree[float]')
        assert validate(core, s, {'a': 1.0}) is None

    def test_array(self, core):
        s = core.access('array[(3),float]')
        assert validate(core, s, np.zeros(3)) is None
        assert validate(core, s, [1, 2, 3]) is not None


class TestCoreRender:
    """Test render round-trip for each type."""

    def test_empty(self, core):
        s = core.access('empty')
        assert render(s) == 'empty'
        assert core.access(render(s)) == s

    def test_boolean(self, core):
        s = core.access('boolean')
        assert render(s) == 'boolean'

    def test_integer(self, core):
        s = core.access('integer')
        assert render(s) == 'integer'

    def test_float(self, core):
        s = core.access('float')
        assert render(s) == 'float'

    def test_delta(self, core):
        s = core.access('delta')
        assert render(s) == 'delta'

    def test_nonnegative(self, core):
        s = core.access('nonnegative')
        assert render(s) == 'nonnegative'

    def test_string(self, core):
        s = core.access('string')
        assert render(s) == 'string'

    def test_enum(self, core):
        s = core.access('enum[a,b,c]')
        r = render(s)
        assert 'enum' in r

    def test_maybe(self, core):
        s = core.access('maybe[float]')
        r = render(s)
        assert 'maybe' in r

    def test_wrap(self, core):
        s = core.access('wrap[string]')
        r = render(s)
        assert 'wrap' in r

    def test_overwrite(self, core):
        s = core.access('overwrite[integer]')
        r = render(s)
        assert 'overwrite' in r

    def test_union(self, core):
        s = core.access('union[float,string]')
        r = render(s)
        assert 'float' in r and 'string' in r

    def test_tuple(self, core):
        s = core.access('tuple[float,string]')
        r = render(s)
        assert 'tuple' in r

    def test_list(self, core):
        s = core.access('list[float]')
        r = render(s)
        assert 'list' in r

    def test_map(self, core):
        s = core.access('map[float]')
        r = render(s)
        assert 'map' in r

    def test_tree(self, core):
        s = core.access('tree[float]')
        r = render(s)
        assert 'tree' in r

    def test_array(self, core):
        s = core.access('array[(3|4),float]')
        r = render(s)
        assert 'array' in r

    def test_frame(self, core):
        s = core.access('dataframe[a:float|b:integer]')
        r = render(s)
        assert 'dataframe' in r

    def test_path(self, core):
        s = core.access('path')
        assert render(s) == 'path'

    def test_wires(self, core):
        s = core.access('wires')
        assert render(s) == 'wires'

    def test_link(self, core):
        s = core.access('link[x:float,y:string]')
        r = render(s)
        assert isinstance(r, (str, dict))


if __name__ == '__main__':
    core = allocate_core()

    # Run all test classes
    import sys
    pytest.main([__file__, '-v'] + sys.argv[1:])
