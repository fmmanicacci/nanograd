"""Module containing utility functions for nanograd."""

from typing import TYPE_CHECKING
from ordered_set import OrderedSet

if TYPE_CHECKING:
    from .scalar import Scalar

def topological_sort(root: 'Scalar') -> OrderedSet['Scalar']:
        """Topological sort of the computational graph."""
        topo, visited = OrderedSet(), set()
        def _topological_sort(node: 'Scalar') -> None:
            if node not in visited:
                visited.add(node)
                for prev in node._prev:
                    _topological_sort(prev)
                topo.append(node)
        _topological_sort(root)
        return topo
