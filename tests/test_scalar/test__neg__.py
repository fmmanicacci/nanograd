"""Test suit for the negation operator."""

from nanograd.scalar import Scalar
from nanograd.enums import Operation
from ordered_set import OrderedSet

def test_negation() -> None:
    """Test the negation operator."""
    x = Scalar(2.0, requires_grad=True)
    y = -x
    y._grad = 1.0
    y._backward_fn()
    assert y.data == -x.data
    assert y._op == Operation.NEGATION
    assert y._prev == OrderedSet([x])
    assert x._grad == -1.0
