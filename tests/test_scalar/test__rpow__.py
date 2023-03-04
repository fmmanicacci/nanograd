"""Test suite for the __rpow__ method of the Scalar object."""

from math import isclose, log
from pytest import raises
from nanograd.scalar import Scalar
from nanograd.enums import Operation

def test__rpow__int() -> None:
    """Test the __rpow__ method of the Scalar object with an int."""
    x = Scalar(2.0, requires_grad=True)
    z = 2 ** x
    y = (z._prev - {x}).pop()
    y.requires_grad_(True)
    z._grad = 1.0
    z._backward_fn()

    assert z.data == 4.0
    assert z._op == Operation.EXPONENTIATION
    assert z._prev == {y, x}
    assert isclose(x._grad, (y.data**x.data) * log(y.data))
    assert isclose(y._grad, (y.data**(x.data-1))* x.data)


def test__rpow__float() -> None:
    """Test the __rpow__ method of the Scalar object with a float."""
    x = Scalar(2.0, requires_grad=True)
    z = 2.0 ** x
    y = (z._prev - {x}).pop()
    y.requires_grad_(True)
    z._grad = 1.0
    z._backward_fn()

    assert z.data == 4.0
    assert z._op == Operation.EXPONENTIATION
    assert z._prev == {y, x}
    assert isclose(x._grad, (y.data**x.data) * log(y.data))
    assert isclose(y._grad, (y.data**(x.data-1))* x.data)


def test__rpow__unsupported() -> None:
    """Test the __rpow__ method of the Scalar object with an unsupported type."""
    x = Scalar(2.0, requires_grad=True)
    with raises(TypeError):
        "2.0" ** x
