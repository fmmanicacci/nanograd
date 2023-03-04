"""Test suite for the tanh method of the Scalar object."""

from math import isclose, tanh
from nanograd.scalar import Scalar
from nanograd.enums import Operation

def test_tanh_with_label() -> None:
    """Test the tanh method of the Scalar object with a label."""
    x = Scalar(0.5, label='x', requires_grad=True)
    y = x.tanh(label='y')
    y._grad = 1.0
    y._backward_fn()

    assert isclose(y.data, tanh(0.5))
    assert y.label == 'y'
    assert y._prev == {x}
    assert y._op == Operation.HYPERBOLIC_TANGENT
    assert isclose(x._grad, 1.0 - y.data**2)
