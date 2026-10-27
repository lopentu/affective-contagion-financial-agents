"""Write a standalone HTML replay of one recorded run.

Thin CLI over `affective_contagion.replay`. The page needs no server and no dependencies at view
time, so it can be attached to an issue or shared as one file. A committed example lives in
docs/examples/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from affective_contagion.replay import embed_as_javascript, load_run


def render(data: dict[str, Any], template: Path) -> str:
    """Inline the run data into the page template."""
    return template.read_text(encoding="utf-8").replace("__RUN_DATA__", embed_as_javascript(data))


def main() -> None:
    """Write a standalone replay page for one run directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, help="A completed run directory.")
    parser.add_argument("--output", type=Path, required=True, help="Destination .html file.")
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(__file__).parent / "replay_template.html",
    )
    arguments = parser.parse_args()
    data = load_run(arguments.run)
    arguments.output.write_text(render(data, arguments.template), encoding="utf-8")
    agent_steps = sum(step["kind"] == "agent" for step in data["steps"])
    print(
        f"replay for {data['run_id']}: {len(data['steps'])} steps "
        f"({agent_steps} agent turns, {data['n_rounds']} rounds) -> {arguments.output}"
    )


if __name__ == "__main__":
    main()
