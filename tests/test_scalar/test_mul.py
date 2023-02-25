"""Test suit for the mul method of the Scalar object."""

from pytest import raises
from nanograd.scalar import Scalar
from nanograd.enums import Operation

def test_mul_int() -> None:
    """Test the mul method with an int as argument."""
    x = Scalar(2.0, requires_grad=True)
    z = x.mul(2)
    y = (z._prev - {x}).pop()
    y.requires_grad_(True)
    z._grad = 1.0
    z._backward_fn()
    assert z.data == 4.0
    assert z._op == Operation.MULTIPLICATION
    assert len(z._prev) == 2
    assert x in z._prev
    assert y in z._prev
    assert x._grad == y.data
    assert y._grad == x.data


def test_mul_float() -> None:
    """Test the mul method with a float as argument."""
    x = Scalar(2.0, requires_grad=True)
    z = x.mul(2.0)
    y = (z._prev - {x}).pop()
    y.requires_grad_(True)
    z._grad = 1.0
    z._backward_fn()
    assert z.data == 4.0
    assert z._op == Operation.MULTIPLICATION
    assert len(z._prev) == 2
    assert x in z._prev
    assert y in z._prev
    assert x._grad == y.data
    assert y._grad == x.data


def test_mul_scalar() -> None:
    """Test the mul method with a Scalar as argument."""
    x = Scalar(2.0, requires_grad=True)
    y = Scalar(2.0, requires_grad=True)
    z = x.mul(y)
    z._grad = 1.0
    z._backward_fn()
    assert z.data == 4.0
    assert z._op == Operation.MULTIPLICATION
    assert len(z._prev) == 2
    assert x in z._prev
    assert y in z._prev
    assert x._grad == y.data
    assert y._grad == x.data


def test_mul_not_supported() -> None:
    """Test the mul method with a not supported argument."""
    x = Scalar(2.0, requires_grad=True)
    with raises(TypeError):
        x.mul("2")


def test_mul_with_label() -> None:
    """Test the mul method with a label."""
    x = Scalar(2.0, requires_grad=True)
    z = x.mul(2, label="z")
    y = (z._prev - {x}).pop()
    y.requires_grad_(True)
    z._grad = 1.0
    z._backward_fn()
    assert z.data == 4.0
    assert z._op == Operation.MULTIPLICATION
    assert len(z._prev) == 2
    assert x in z._prev
    assert y in z._prev
    assert x._grad == y.data
    assert y._grad == x.data
    assert z.label == "z"
