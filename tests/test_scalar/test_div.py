"""Test suite for the div method of the Scalar object."""

from pytest import raises
from nanograd.scalar import Scalar
from nanograd.enums import Operation

def test_div_label() -> None:
    """Test the label of the output of the div method."""
    x = Scalar(1.0, requires_grad=True)
    y = Scalar(2.0, requires_grad=True)
    z = x.div(y, label='z')
    assert z.label == 'z'
