"""Seed blocks, defined once so a config generator and its test cannot drift apart.

Each block belongs to one plan. Blocks never overlap, because an experiment reusing another's seeds
would reuse its runs, and a replication would become a re-analysis.
"""

from __future__ import annotations

#: PR1 and PR3, the first-hop and second-hop blocks.
PR1_SEEDS: tuple[int, ...] = tuple(range(6, 68))

#: PR7, the belief pilot. Disjoint from every other block.
PR7_SEEDS: tuple[int, ...] = tuple(range(400, 424))

#: PR8, the second-scenario replication. Disjoint from every other block.
PR8_SEEDS: tuple[int, ...] = tuple(range(500, 524))

#: Every seed the PR1-style config generator writes a cell for.
GENERATED_SEEDS: tuple[int, ...] = PR1_SEEDS + PR7_SEEDS + PR8_SEEDS
