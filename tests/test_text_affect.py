"""Tests for lexical message-level affect measurement."""

from pathlib import Path
from typing import cast

import pytest

from affective_contagion.text_affect import (
    VADLexicon,
    VADScale,
    validate_affect_separation,
)


def test_vad_score_reports_coverage_and_neutral_oov() -> None:
    """Coverage should reveal how much text the lexical score observed."""
    lexicon = VADLexicon({"panic": (0.1, 0.9, 0.2), "calm": (0.8, 0.1, 0.7)})
    score = lexicon.score("panic unknown")
    assert score.arousal == 0.9
    assert score.coverage == 0.5
    assert lexicon.score("unknown").coverage == 0


def test_nrc_style_tsv_loading_skips_headers_and_bad_rows(tmp_path: Path) -> None:
    """External lexicon parsing skips the header and malformed rows and keeps valid entries."""
    source = tmp_path / "vad.tsv"
    source.write_text(
        "Word\tValence\tArousal\tDominance\npanic\t0.1\t0.9\t0.2\nbad\trow\n",
        encoding="utf-8",
    )
    assert VADLexicon.from_tsv(source).score("panic").arousal == 0.9


def test_signed_scale_rescales_and_unit_scale_rejects_signed_input(tmp_path: Path) -> None:
    """NRC VAD v2.x ships on [-1, 1] and must be declared, not inferred.

    With `scale="signed"` values are rescaled to [0, 1]. Without it, negative entries raise an
    error rather than loading on the wrong scale.
    """
    source = tmp_path / "signed.tsv"
    source.write_text(
        "term\tvalence\tarousal\tdominance\n"
        "terrified\t-0.800\t0.800\t-0.200\n"
        "calm\t0.700\t-0.900\t-0.400\n",
        encoding="utf-8",
    )
    lexicon = VADLexicon.from_tsv(source, scale="signed")
    terrified = lexicon.score("terrified")
    assert terrified.arousal == pytest.approx((0.800 + 1) / 2)
    assert terrified.valence == pytest.approx((-0.800 + 1) / 2)
    calm = lexicon.score("calm")
    assert calm.arousal == pytest.approx((-0.900 + 1) / 2)
    # Anxious wording must still score as higher arousal after rescaling.
    assert terrified.arousal > calm.arousal

    with pytest.raises(ValueError, match=r'scale="signed"'):
        VADLexicon.from_tsv(source)


def test_unknown_scale_is_rejected(tmp_path: Path) -> None:
    """An unrecognized scale must fail loudly rather than silently defaulting.

    The annotation restricts callers, but the scale also arrives from CLI and config strings, so the
    runtime boundary is checked deliberately here with an unchecked value.
    """
    source = tmp_path / "lex.tsv"
    source.write_text("word\t0.1\t0.2\t0.3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown VAD scale"):
        VADLexicon.from_tsv(source, scale=cast("VADScale", "percent"))


def test_header_and_malformed_rows_are_skipped(tmp_path: Path) -> None:
    """The NRC distributions ship a header row, which must not become an entry."""
    source = tmp_path / "lex.tsv"
    source.write_text(
        "term\tvalence\tarousal\tdominance\npanic\t0.1\t0.9\t0.2\nbroken\t0.1\t0.2\n",
        encoding="utf-8",
    )
    lexicon = VADLexicon.from_tsv(source)
    assert lexicon.score("panic").coverage == pytest.approx(1.0)
    assert lexicon.score("term").coverage == pytest.approx(0.0)


def test_surface_form_wins_over_the_suffix_fallback(tmp_path: Path) -> None:
    """Inflections can carry distinct scores, so an exact match must never be overridden.

    Normalising every form to one lemma would replace a listed value with a wrong one, which is why
    this is a fallback and not a lemmatizer. The values here are illustrative.
    """
    source = tmp_path / "lex.tsv"
    source.write_text("term\tv\ta\td\nfear\t0.1\t0.800\t0.2\nfearing\t0.1\t0.600\t0.2\n", "utf-8")
    lexicon = VADLexicon.from_tsv(source, match="surface_then_suffix")
    assert lexicon.score("fearing").arousal == pytest.approx(0.600)
    assert lexicon.score("fear").arousal == pytest.approx(0.800)


def test_suffix_fallback_recovers_absent_inflections_and_possessives(tmp_path: Path) -> None:
    """Absent inflected forms should resolve to their stem rather than score nothing."""
    source = tmp_path / "lex.tsv"
    source.write_text("term\tv\ta\td\nconcern\t0.3\t0.7\t0.4\nbank\t0.5\t0.5\t0.5\n", "utf-8")
    surface = VADLexicon.from_tsv(source, match="surface")
    fallback = VADLexicon.from_tsv(source, match="surface_then_suffix")
    assert surface.score("concerns").coverage == pytest.approx(0.0)
    assert fallback.score("concerns").arousal == pytest.approx(0.7)
    assert fallback.score("bank's").coverage == pytest.approx(1.0)
    # A short token must not be stripped down to nothing.
    assert fallback.score("is").coverage == pytest.approx(0.0)


def test_affect_separation_gate_rejects_a_non_manipulating_pair(tmp_path: Path) -> None:
    """A pair whose anxiety arm is not more aroused must fail before a run starts."""
    source = tmp_path / "lex.tsv"
    source.write_text("term\tv\ta\td\ncalm\t0.9\t0.05\t0.3\nalarmed\t0.2\t0.82\t0.5\n", "utf-8")
    lexicon = VADLexicon.from_tsv(source)
    with pytest.raises(ValueError, match="not measurable"):
        validate_affect_separation(lexicon, {"neutral": "alarmed", "anxiety_like": "calm"})
    delta = validate_affect_separation(lexicon, {"neutral": "calm", "anxiety_like": "alarmed"})
    assert delta > 0


def test_signed_scale_on_a_unit_file_fails_loudly(tmp_path: Path) -> None:
    """A [0, 1] lexicon read as signed maps into [0.5, 1], halving every contrast silently."""
    source = tmp_path / "unit.tsv"
    source.write_text(
        "term\tv\ta\td\npanic\t0.1\t0.9\t0.2\ncalm\t0.8\t0.1\t0.7\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r'scale="unit"'):
        VADLexicon.from_tsv(source, scale="signed")
