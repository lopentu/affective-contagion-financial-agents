"""Communication topology: who reads whom."""

from __future__ import annotations

from affective_contagion.config import Topology


def inbound_neighbors(topology: Topology, n_agents: int) -> dict[int, tuple[int, ...]]:
    """Return senders whose previous-round messages each recipient observes.

    The paper's teams are fully connected: every agent reads every other agent, never itself.
    """
    if n_agents <= 0:
        msg = "n_agents must be positive"
        raise ValueError(msg)
    if topology == "full":
        return {
            recipient: tuple(sender for sender in range(n_agents) if sender != recipient)
            for recipient in range(n_agents)
        }
    msg = f"unsupported topology: {topology}"
    raise ValueError(msg)
