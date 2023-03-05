"""Test suite for the ReLU method of the Scalar object."""

from nanograd.scalar import Scalar
from nanograd.enums import Operation
from ordered_set import OrderedSet

def test_relu_with_label() -> None:
    """Test of the relu method with a label."""
    x = Scalar(1.0, requires_grad=True)
    y = x.relu(label="y")
    y._grad = 1.0
    y._backward_fn()

    assert y.data == 1.0
    assert y.label == "y"
    assert y._prev == OrderedSet([x])
    assert y._op == Operation.RELU
    assert x._grad == 1.0
