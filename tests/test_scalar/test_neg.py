"""Test suit for the neg method of the Scalar object."""

from nanograd.scalar import Scalar
from nanograd.enums import Operation

def test_neg() -> None:
    """Test the neg method of the Scalar object."""
    x = Scalar(2.0, requires_grad=True)
    y = x.neg(label="-x")
    y._grad = 1.0
    y._backward_fn()
    assert y.data == -x.data
    assert y.label == "-x"
    assert y._op == Operation.NEGATION
    assert y._prev == {x}
    assert x._grad == -1.0