"""
Bigraph algebraic assembly
==========================

Implements the categorical operations on bigraph schemas that have
open interfaces (sites / inner names on their inner face, roots /
outer names on their outer face). See Milner, *Space and Motion of
Communicating Agents* (2008):

  - Def. 2.5 (p. 17): composition ``G ∘ F : I → K`` glues the roots
    of ``F`` to the sites of ``G`` and the outer names of ``F`` to
    the inner names of ``G``. The mediating face ``J`` disappears.
  - Def. 2.7 (p. 18): juxtaposition / tensor ``F ⊗ G`` places two
    disjoint bigraphs side by side; interfaces concatenate.
  - Def. 3.1 (p. 28): the identity placing ``id_m : m → m``; at the
    bigraph level the identity ``id_I : I → I`` is the arrow that
    lets every composition satisfy ``f ∘ id = f`` and ``id ∘ f = f``.

Incremental scope. M1.4 lands the trivial identity laws; M2 (elementary
bigraphs and full composition) builds out the full algebra. For the
design rationale see ``.claude/plans/milner-formalism.md``.
"""

from bigraph_schema.schema import Interface


EPSILON = Interface()
"""The trivial interface ``ε = ⟨0, ∅⟩`` — the unit of tensor product
and the domain of every ground bigraph. ``Interface()`` with no
places and no names (Def. 2.7, p. 18)."""


def identity(interface: Interface) -> Interface:
    """The identity arrow on ``interface``.

    In Milner's category of bigraphs, ``id_I : I → I`` is the arrow
    whose place graph is the bijection from sites to roots and whose
    link graph is the bijection from inner names to outer names. Its
    defining property is the identity law (C3 of Def. 2.8): for any
    composable ``f``, ``id ∘ f = f`` and ``f ∘ id = f``.

    We don't yet carry a separate *Bigraph* schema distinct from its
    interface, so for now ``identity(I)`` is represented as ``I``
    itself — equality up to the face tells us we have an identity.
    A future ``Bigraph`` type wrapping ⟨inner, outer, body⟩ will
    replace this with a proper identity arrow whose body is the
    trivial placing/linking.
    """
    return interface


def compose(outer: Interface, inner: Interface) -> Interface:
    """Compose ``outer ∘ inner``.

    Composition law (Def. 2.5): ``G ∘ F : I → K`` requires that the
    outer face of ``F`` equal the inner face of ``G``; the composite's
    inner face is ``F``'s inner face and its outer face is ``G``'s.

    M1.4 only implements the **identity** cases: anything composed
    with ``EPSILON`` (the identity on the origin) returns the other
    operand unchanged. This is enough to demonstrate the right- and
    left-identity laws. Full composition requires a Bigraph schema
    whose body can be glued, and lands in M2.
    """
    if inner == EPSILON:
        return outer
    if outer == EPSILON:
        return inner
    raise NotImplementedError(
        'compose is currently only defined for the identity cases '
        '(g ∘ id_ε = g and id_ε ∘ g = g). Full composition of '
        'non-trivial bigraph schemas lands in M2 — see '
        '.claude/plans/milner-formalism.md.')
