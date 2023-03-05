"""Test suite for the topological sort function."""

from nanograd.scalar import Scalar
from nanograd.utils import topological_sort

def test_topological_sort() -> None:
    """Test of the topological sort function."""
    x1 = Scalar(1.0)
    w1 = Scalar(-0.5)
    x1w1 = x1 * w1
    x2 = Scalar(-3.0)
    w2 = Scalar(0.1)
    x2w2 = x2 * w2
    x1w1x2w2 = x1w1 + x2w2
    b = Scalar(2.5)
    x1w1x2w2b = x1w1x2w2 + b
    out = x1w1x2w2b.tanh()
    topo = topological_sort(out)

    assert topo == {x1, w1, x1w1, x2, w2, x2w2, x1w1x2w2, b, x1w1x2w2b, out}
