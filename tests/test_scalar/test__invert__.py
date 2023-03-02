"""Test suit for the __invert__ method of the Scalar object."""

from math import isclose
from nanograd.scalar import Scalar
from nanograd.enums import Operation

def test__invert() -> None:
    """Test the __invert__ method."""
    x = Scalar(42.0, requires_grad=True)
    z = ~x
    z._grad = 1.0
    z._backward_fn()
    
    assert isclose(z.data, x.data ** (-1.0))
    assert z._prev == {x}
    assert z._op == Operation.INVERTION
    assert isclose(x._grad, (-1.0)/(x.data**2))
    