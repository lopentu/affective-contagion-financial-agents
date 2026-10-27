"""Turn a recorded run directory into the data a replay page renders.

This runs after a run and off the experimental path. It reads only saved artifacts and computes
nothing scientific, so it cannot affect a result. `scripts/replay_run.py` is a thin CLI that
inlines the output into a self-contained HTML page.

Stages come from `trace.jsonl`, which `simulation.run_simulation` emits as it executes, so a page
shows what actually ran. `PIPELINE_STAGES` below is a fallback for runs without a trace, and a page
built from it reports itself as reconstructed.

It lives in the package so that tests load the committed sample run in `docs/examples/`, and a
schema change that breaks the page fails a test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PIPELINE_STAGES: tuple[tuple[str, str, str], ...] = (
    (
        "topology.inbound_neighbors",
        "inbox",
        "Which peers this agent can hear, and only their previous-round messages.",
    ),
    (
        "GenerationContext",
        "context",
        "Assembles stimulus (only if treated), price, and inbox into the observable state.",
    ),
    (
        "generator.generate",
        "raw_output",
        "Prompt is rendered and the model samples a decision, under the decoding grammar\n"
        "when constrained_decision_format is set.",
    ),
    (
        "messages.parse_agent_decision",
        "decision",
        "Strict parse. On failure messages.fallback_hold substitutes HOLD at size 0.",
    ),
    (
        "simulation._fill_within_position_limit",
        "fill",
        "The part of the requested order the position bound can absorb; this is what clears.",
    ),
    (
        "intervention.apply",
        "delivery",
        "What peers will receive next round: the message unchanged, or the injected report.",
    ),
    (
        "text_affect.VADLexicon.score",
        "message_affect",
        "Primary outcome measurement, scored on the message the agent produced (not on the\n"
        "delivered text, which differs when the injected report replaces it).",
    ),
)


def embed_as_javascript(data: dict[str, Any]) -> str:
    r"""Serialise run data for inlining inside an HTML ``<script>`` block.

    Plain `json.dumps` is not safe here. JSON and JavaScript-in-HTML disagree in two ways that
    model-generated text can trigger.

    - `</script>` inside any string closes the block early, so the page fails with a syntax error
      and renders blank. An agent message containing "</script>" would break the replay of its
      own run.
    - U+2028 and U+2029 are legal in a JSON string but are line terminators in JavaScript source
      prior to ES2019, and remain a portability hazard.

    Escaping `<` to `\\u003c` covers the first case without changing the decoded value. JSON string
    escapes are resolved by the JS parser, so `"\\u003c/script>"` decodes back to `"</script>"`.
    """
    payload = json.dumps(data, separators=(",", ":"))
    return payload.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def load_run(run: Path) -> dict[str, Any]:
    """Read one run directory into the structure the page consumes."""
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (run / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    market = [
        json.loads(line)
        for line in (run / "market.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    trace_path = run / "trace.jsonl"
    trace = (
        [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if trace_path.exists()
        else []
    )
    experiment = manifest["experiment"]
    roles: list[str] = list(experiment["roles"])
    # System prompts are constant per role, so store them once instead of on all N x R steps.
    system_prompts: dict[str, str] = {}

    steps: list[dict[str, Any]] = []
    positions = dict.fromkeys(roles, 0.0)
    for round_index in range(experiment["n_rounds"]):
        rows = sorted(
            (row for row in records if row["round_index"] == round_index),
            key=lambda row: row["agent_index"],
        )
        for row in rows:
            role = row["role"]
            before = positions[role]
            positions[role] = row["position_after"]
            affect = row.get("message_affect") or {}
            prompt = row["generation"].get("prompt_messages") or []
            for message in prompt:
                if message["role"] == "system":
                    system_prompts.setdefault(role, message["content"])
            user_prompt = next(
                (m["content"] for m in prompt if m["role"] == "user"),
                "",
            )
            steps.append(
                {
                    "kind": "agent",
                    "round": round_index,
                    "agent": row["agent_index"],
                    "role": role,
                    "treated": bool(row["context"]["news"]),
                    "stimulus": row["context"]["news"],
                    "inbox": [
                        {"from": item["sender_role"], "text": item["text"]}
                        for item in row["context"]["inbox"]
                    ],
                    "price": row["context"]["price"],
                    # The exact text sent to the model, and the exact text it returned before
                    # parsing. Both are needed to tell a prompt bug from a parsing bug.
                    "user_prompt": user_prompt,
                    "raw": row["generation"]["raw_output"],
                    "parsed": row["parse_succeeded"],
                    "message": row["decision"]["message"],
                    "action": row["decision"]["action"],
                    "size": row["decision"]["size"],
                    "delivered": row["delivery"]["delivered_text"],
                    "intervention": row["delivery"]["intervention_name"],
                    "position_before": before,
                    "position_after": row["position_after"],
                    "fill": round(row["position_after"] - before, 6),
                    "arousal": affect.get("arousal"),
                    "valence": affect.get("valence"),
                    "coverage": affect.get("coverage"),
                    "seed": row["generation"]["generation_seed"],
                    "positions": dict(positions),
                }
            )
        if round_index < len(market):
            step = market[round_index]
            steps.append(
                {
                    "kind": "market",
                    "round": round_index,
                    "imbalance": step["order_imbalance"],
                    "previous_price": step["previous_price"],
                    "price": step["price"],
                    "noise": step["noise_draw"],
                    "positions": dict(positions),
                }
            )
    for step in steps:
        if step["kind"] == "agent":
            step["trace"] = [
                {"stage": e["stage"], "detail": e["detail"]}
                for e in trace
                if e["round_index"] == step["round"] and e["agent_index"] == step["agent"]
            ]
        else:
            step["trace"] = [
                {"stage": e["stage"], "detail": e["detail"]}
                for e in trace
                if e["round_index"] == step["round"] and e["agent_index"] is None
            ]
    return {
        "run_id": manifest["run_id"],
        "roles": roles,
        "shock_round": experiment["shock_round"],
        "shock_duration": experiment["shock_duration"],
        "treated_role": experiment.get("treated_role"),
        "topology": experiment["topology"],
        "condition": experiment["shock_type"],
        "intervention": experiment["message_intervention"],
        "fundamental": experiment["fundamental_value"],
        "n_rounds": experiment["n_rounds"],
        "stimuli": manifest.get("stimuli"),
        "payload": manifest.get("payload"),
        "arousal_delta": manifest.get("stimulus_arousal_delta"),
        "parse_failure_rate": manifest["parse_failure_rate"],
        "lexicon": manifest.get("vad_lexicon_source"),
        "system_prompts": system_prompts,
        "traced": bool(trace),
        "stages": [
            {"fn": name, "field": field, "note": note} for name, field, note in PIPELINE_STAGES
        ],
        "steps": steps,
    }
