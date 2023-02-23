"""Test suit for the __radd__ method of the Scalar object."""

from pytest import raises
from nanograd.scalar import Scalar
from nanograd.enums import Operation

def test__radd__int() -> None:
    """Test that the right addition operator works when the argument is an int."""
    x = Scalar(1.0, requires_grad=True)
    z = 1 + x
    y = (y._prev - {x}).pop()
    # Check that the output is correct.
    assert z.data == 2.0
    assert z.requires_grad
    assert z._op == Operation.ADDITION
    # Check that the children of the output are correct.
    assert len(z._prev) == 2
    assert x in z._prev
    assert y.data == 1
    assert not y.requires_grad
    assert y._op == Operation.NONE
    assert len(y._prev) == 0
    assert y.label == None


def test__radd__float() -> None:
    """Test that the right addition operator works when the argument is a float."""
    x = Scalar(1.0, requires_grad=True)
    z = 1.0 + x
    y = (z._prev - {x}).pop()
    # Check that the output is correct.
    assert z.data == 2.0
    assert z.requires_grad
    assert z._op == Operation.ADDITION
    # Check that the children of the output are correct.
    assert len(z._prev) == 2
    assert x in z._prev
    assert y.data == 1.0
    assert not y.requires_grad
    assert y._op == Operation.NONE
    assert len(y._prev) == 0
    assert y.label == None


def test__radd__unsupported_type() -> None:
    """Test that the right addition operator raises a TypeError when the argument is not a supported type."""
    x = Scalar(1.0, requires_grad=True)
    with raises(TypeError):
        x + '1'


def test__radd__backward_fn() -> None:
    """Test that the backward function works for the right addition operator."""
    x = Scalar(1.0, requires_grad=True)
    z = 1 + x
    y = (z._prev - {x}).pop()
    y.requires_grad_(True)
    z._grad = 1.0
    z._backward_fn()
    assert x._grad == 1.0
    assert y._grad == 1.0
