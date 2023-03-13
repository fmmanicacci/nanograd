"""Test suite for the backward method of the Scalar object."""

from math import isclose
from nanograd.scalar import Scalar

def test_backward() -> None:
    """Test of the backward method of the Scalar object."""
    x1 = Scalar(-3.0)
    w1 = Scalar(1.0)
    x1w1 = x1 * w1
    x2 = Scalar(1.0)
    w2 = Scalar(0.5)
    x2w2 = x2 * w2
    x1w1x2w2 = x1w1 + x2w2
    b = Scalar(3.0)
    x1w1x2w2b = x1w1x2w2 + b
    out = x1w1x2w2b.tanh()
    out.backward()
    
    x1w1x2w2b_expected_grad = 1.0 - out.data**2
    x1w1x2w2_expected_grad = 1.0 * x1w1x2w2b_expected_grad
    x1w1_expected_grad = 1.0 * x1w1x2w2_expected_grad
    x2w2_expected_grad = x1w1_expected_grad

    assert isclose(out.data, 0.4621171573)
    assert out._grad == 1.0
    assert isclose(x1w1x2w2b._grad, x1w1x2w2b_expected_grad)
    assert isclose(x1w1x2w2._grad, x1w1x2w2_expected_grad)
    assert b._grad == 0.0
    assert isclose(x1w1._grad, x1w1_expected_grad)
    assert isclose(x2w2._grad, x2w2_expected_grad)
    assert x1._grad == 0.0
    assert w1._grad == 0.0
    assert x2._grad == 0.0
    assert w2._grad == 0.0
