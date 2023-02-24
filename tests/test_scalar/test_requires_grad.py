"""Test suite for the requires_grad_ method of the Scalar object."""

from nanograd.scalar import Scalar

def test_requires_grad_true() -> None:
    """Test that the requires_grad_ method works when the argument is True."""
    x = Scalar(1.0, requires_grad=True)
    assert x.requires_grad
    assert x._backward == 1.0


def test_requires_grad_false() -> None:
    """Test that the requires_grad_ method works when the argument is False."""
    x = Scalar(1.0, requires_grad=False)
    assert not x.requires_grad
    assert x._backward == 0.0
