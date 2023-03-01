"""Test suite for the floordiv method of the Scalar object."""

from pytest import raises
from nanograd.scalar import Scalar
from nanograd.enums import Operation

def test_floordiv_int() -> None:
    """Test the floordiv method of the Scalar object with an int."""
    x = Scalar(2.143498, requires_grad=True)
    z = x.floordiv(2)
    y = (z._prev - {x}).pop()
    y.requires_grad_(True)
    z._grad = 1.0
    z._backward_fn()

    assert z.data == 1.0
    assert z._op == Operation.FLOOR_DIVISION
    assert z._prev == {y, x}
    assert x._grad == 0.0
    assert y._grad == 0.0


def test_floordiv_float() -> None:
    """Test the floordiv method of the Scalar object with a float."""
    x = Scalar(2.143498, requires_grad=True)
    z = x.floordiv(2.0)
    y = (z._prev - {x}).pop()
    y.requires_grad_(True)
    z._grad = 1.0
    z._backward_fn()

    assert z.data == 1.0
    assert z._op == Operation.FLOOR_DIVISION
    assert z._prev == {y, x}
    assert x._grad == 0.0
    assert y._grad == 0.0


def test_floordiv_scalar() -> None:
    """Test the floordiv method of the Scalar object with a Scalar."""
    x = Scalar(2.143498, requires_grad=True)
    y = Scalar(2.0, requires_grad=True)
    z = x.floordiv(y)
    z._grad = 1.0
    z._backward_fn()

    assert z.data == 1.0
    assert z._op == Operation.FLOOR_DIVISION
    assert z._prev == {y, x}
    assert x._grad == 0.0
    assert y._grad == 0.0


def test_floordiv_unsupported() -> None:
    """Test the floordiv method of the Scalar object with an unsupported type."""
    x = Scalar(2.143498, requires_grad=True)
    with raises(TypeError):
        x.floordiv("2.0")


def test_floordiv_label() -> None:
    """Test the floordiv method of the Scalar object with a label."""
    x = Scalar(2.143498, requires_grad=True)
    y = Scalar(2.0, requires_grad=True)
    z = x.floordiv(y, label="z")
    z._grad = 1.0
    z._backward_fn()

    assert z.data == 1.0
    assert z.label == "z"
    assert z._op == Operation.FLOOR_DIVISION
    assert z._prev == {y, x}
    assert x._grad == 0.0
    assert y._grad == 0.0
