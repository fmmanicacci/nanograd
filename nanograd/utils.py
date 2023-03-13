"""Module containing utility functions for nanograd."""

from ordered_set import OrderedSet

def topological_sort(root) -> OrderedSet:
        """Topological sort of the computational graph."""
        topo, visited = OrderedSet(), set()
        def _topological_sort(node) -> None:
            if node not in visited:
                visited.add(node)
                for prev in node._prev:
                    _topological_sort(prev)
                topo.append(node)
        _topological_sort(root)
        return topo
