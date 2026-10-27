"""Tests for communication graphs."""

from affective_contagion.topology import inbound_neighbors


def test_full_excludes_self_edges() -> None:
    """Every agent should observe every other agent only."""
    result = inbound_neighbors("full", 3)
    assert result == {0: (1, 2), 1: (0, 2), 2: (0, 1)}
