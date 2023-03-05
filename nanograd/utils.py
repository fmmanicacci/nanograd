"""Module containing utility functions for nanograd."""

from .types import _NodeLike

def topological_sort(root: _NodeLike) -> set[_NodeLike]:
        """Topological sort of the computational graph."""
        topo, visited = set(), set()
        def _topological_sort(node: _NodeLike) -> None:
            if node not in visited:
                visited.add(node)
                for prev in node._prev:
                    _topological_sort(prev)
                topo.add(node)
        _topological_sort(root)
        return topo
