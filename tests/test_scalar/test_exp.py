"""Test suite for the exp method of the Scalar object."""

from math import isclose, log
from nanograd.scalar import Scalar
from nanograd.enums import Operation

def test_exp_with_label() -> None:
    """Test the exp method of the Scalar object with a label."""
    x = Scalar(log(2.0), label='x', requires_grad=True)
    y = x.exp(label='y')
    y._grad = 1.0
    y._backward_fn()

    assert y.data == 2.0
    assert y.label == 'y'
    assert y._prev == {x}
    assert y._op == Operation.EXPONENTIAL
    assert isclose(x._grad, y.data)
