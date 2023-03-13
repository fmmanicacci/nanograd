"""Test suite for the topological sort function."""

from nanograd.scalar import Scalar
from nanograd.utils import topological_sort
from ordered_set import OrderedSet

def test_topological_sort() -> None:
    """Test of the topological sort function."""
    x1 = Scalar(1.0, label='x1')
    w1 = Scalar(-0.5, label='w1')
    x1w1 = x1.mul(w1, label='x1w1')
    x2 = Scalar(-3.0, label='x2')
    w2 = Scalar(0.1, label='w2')
    x2w2 = x2.mul(w2, label='x2w2')
    x1w1x2w2 = x1w1.add(x2w2, label='x1w1x2w2')
    b = Scalar(2.5, label='b')
    x1w1x2w2b = x1w1x2w2.add(b, label='x1w1x2w2b')
    out = x1w1x2w2b.tanh(label='out')
    topo = topological_sort(out)
    
    assert topo == OrderedSet([x1, w1, x1w1, x2, w2, x2w2, x1w1x2w2, b, x1w1x2w2b, out])
