"""Module containing utility functions for nanograd."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .scalar import Scalar

def topological_sort(root: 'Scalar') -> set['Scalar']:
        """Topological sort of the computational graph."""
        topo, visited = set(), set()
        def _topological_sort(node: 'Scalar') -> None:
            if node not in visited:
                visited.add(node)
                for prev in node._prev:
                    _topological_sort(prev)
                topo.add(node)
        _topological_sort(root)
        return topo
