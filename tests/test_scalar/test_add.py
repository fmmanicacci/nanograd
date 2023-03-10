"""Test suit for the add method of the Scalar object."""

from nanograd.scalar import Scalar
from nanograd.enums import Operation
from ordered_set import OrderedSet

def test_add() -> None:
    """Test that the add method works properly."""
    x = Scalar(1.0, requires_grad=True)
    y = Scalar(2.0, requires_grad=True)
    label = 'a+b'
    z = x.add(y, label=label)
    z._grad = 1.0
    z._backward_fn()
    assert z.data == 3.0
    assert z.label == label
    assert z.requires_grad
    assert z._op == Operation.ADDITION
    assert z._prev == OrderedSet([x, y])
    assert x._grad == 1.0
    assert y._grad == 1.0
