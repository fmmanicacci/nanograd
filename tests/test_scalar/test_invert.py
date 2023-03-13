"""Test suite for the invert method of the Scalar object."""

from math import isclose
from nanograd.scalar import Scalar
from nanograd.enums import Operation
from ordered_set import OrderedSet

def test_invert_with_label() -> None:
    """Test the invert method with a label."""
    x = Scalar(42.0, requires_grad=True)
    z = x.invert(label="z")
    z._grad = 1.0
    z._backward_fn()
    
    assert isclose(z.data, x.data**-1)
    assert z.label == 'z'
    assert z._prev == OrderedSet([x])
    assert z._op == Operation.INVERTION
    assert isclose(x._grad, -1.0/(x.data**2))
    