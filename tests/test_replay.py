"""Tests for turning a recorded run into replay page data.

The load path is covered by `test_sample_run.py` against the committed example. What is tested here
is the HTML embedding step, which is the one place where model-generated text reaches a different
language's parser.
"""

from __future__ import annotations

import json
from pathlib import Path

from affective_contagion.replay import embed_as_javascript


def test_run_data_cannot_break_out_of_the_script_block() -> None:
    """An agent that writes "</script>" must not destroy the replay of the run that captured it.

    `json.dumps` leaves `</script>` literal, which closes the HTML block early and leaves a page
    that fails to parse and renders blank. Escaping `<` keeps the decoded value identical.
    """
    hostile = {"raw": "SELL </script><script>alert(1)</script>", "note": "a\u2028b"}
    embedded = embed_as_javascript(hostile)
    assert "</script>" not in embedded
    assert "\u2028" not in embedded
    assert "\\u003c/script>" in embedded
    # The escaping must be lossless: JSON unescapes \u003c back to "<".
    assert json.loads(embedded) == hostile


def test_sample_page_embeds_no_closing_script_tag_in_its_data() -> None:
    """The committed example must itself be free of the breakout."""
    page = Path("docs/examples/sample-run.html").read_text(encoding="utf-8")
    data_line = page.split("const D = ", 1)[1].split("\n", 1)[0]
    assert "</script>" not in data_line
