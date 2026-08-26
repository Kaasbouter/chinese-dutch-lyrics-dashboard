from __future__ import annotations

from dataclasses import replace

import pytest

import lyrics_dashboard.parser as parser_module
from lyrics_dashboard.errors import PairingError
from lyrics_dashboard.models import Language, ParsedLyrics, Section
from lyrics_dashboard.parser import apply_section_languages, parse_lyrics


def _detected_bilingual_lyrics() -> ParsedLyrics:
    return parse_lyrics(
        "[Title]\n"
        "\u4e2d\u6587\u6b4c\u540d | Nederlands lied\n\n"
        "Verse 1\n"
        "\u9019\u662f\u4e2d\u6587\u6b4c\u8a5e\n\n"
        "Chorus 1\n"
        "Dit is Nederlandse liedtekst\n\n"
        "Bridge 1\n"
        "Nog een Nederlandse regel\n"
    )


def _section(index: int, language: Language, line: str) -> Section:
    return Section(
        label=f"Verse {index}",
        kind="Verse",
        number=index,
        lines=(line,),
        language=language,
        original_index=index,
    )


def test_default_assignments_are_an_immutable_no_op() -> None:
    parsed = _detected_bilingual_lyrics()
    assignments = {
        section.original_index: section.language for section in parsed.sections
    }

    updated = apply_section_languages(parsed, assignments)

    assert updated == parsed
    assert updated is not parsed
    assert all(
        updated_section is not original_section
        for updated_section, original_section in zip(
            updated.sections, parsed.sections, strict=True
        )
    )


def test_multiple_languages_change_by_stable_original_index() -> None:
    parsed = _detected_bilingual_lyrics()
    reordered_indices = (41, 7, 23)
    parsed = replace(
        parsed,
        sections=tuple(
            replace(section, original_index=original_index)
            for section, original_index in zip(
                parsed.sections, reordered_indices, strict=True
            )
        ),
    )

    updated = apply_section_languages(parsed, {23: "zh", 41: "nl", 7: "zh"})

    assert [section.original_index for section in updated.sections] == [41, 7, 23]
    assert [section.language for section in updated.sections] == ["nl", "zh", "zh"]


def test_bilingual_warnings_and_language_counts_are_recomputed() -> None:
    parsed = ParsedLyrics(
        raw_title="",
        chinese_title="",
        dutch_title="",
        sections=(
            _section(0, "zh", "\u4e2d\u6587\u4e00"),
            _section(1, "zh", "\u4e2d\u6587\u4e8c"),
            _section(2, "nl", "Nederlandse drie"),
            _section(3, "nl", "Nederlandse vier"),
        ),
        warnings=("stale warning",),
        mode="bilingual",
    )

    interleaved = apply_section_languages(
        parsed,
        {0: "zh", 1: "nl", 2: "zh", 3: "nl"},
    )

    assert len(interleaved.warnings) == 1
    assert "interleaved" in interleaved.warnings[0]
    assert "stale warning" not in interleaved.warnings

    uneven = apply_section_languages(
        parsed,
        {0: "zh", 1: "zh", 2: "zh", 3: "nl"},
    )

    assert len(uneven.warnings) == 1
    assert "3 Chinese sections and 1 Dutch sections" in uneven.warnings[0]


def test_override_preserves_section_order_text_titles_and_markers_exactly() -> None:
    parsed = ParsedLyrics(
        raw_title="Raw | title // exact",
        chinese_title="\u4e2d\u6587 | //",
        dutch_title="Dutch | //",
        sections=(
            Section(
                label="Bridge 9",
                kind="Bridge",
                number=9,
                lines=("\u4e2d\u6587 // \u4fdd\u7559 | \u6a19\u8a18", "  exact spacing  "),
                language="zh",
                original_index=90,
            ),
            Section(
                label="Verse 2",
                kind="Verse",
                number=2,
                lines=("Dutch // stays | exact",),
                language="nl",
                original_index=20,
            ),
        ),
        warnings=(),
        mode="bilingual",
    )

    updated = apply_section_languages(parsed, {20: "zh", 90: "nl"})

    assert updated.raw_title == parsed.raw_title
    assert updated.chinese_title == parsed.chinese_title
    assert updated.dutch_title == parsed.dutch_title
    assert [section.original_index for section in updated.sections] == [90, 20]
    assert [section.lines for section in updated.sections] == [
        section.lines for section in parsed.sections
    ]
    restored_sections = [
        replace(updated_section, language=original_section.language)
        for updated_section, original_section in zip(
            updated.sections, parsed.sections, strict=True
        )
    ]
    assert restored_sections == list(parsed.sections)


def test_applying_assignments_never_reruns_language_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = _detected_bilingual_lyrics()

    def fail_if_called(_text: str) -> None:
        raise AssertionError("automatic language detection was rerun")

    monkeypatch.setattr(parser_module, "detect_language", fail_if_called)

    updated = apply_section_languages(parsed, {0: "nl", 1: "zh", 2: "zh"})

    assert [section.language for section in updated.sections] == ["nl", "zh", "zh"]


def test_all_one_language_uses_existing_strict_validation_before_mode_change() -> None:
    valid = ParsedLyrics(
        raw_title="Song title",
        chinese_title="",
        dutch_title="Song title",
        sections=(
            _section(0, "zh", "This line was misclassified"),
            _section(1, "nl", "Another Dutch or Latin-script line"),
        ),
        warnings=("old",),
        mode="bilingual",
    )

    updated = apply_section_languages(valid, {0: "nl", 1: "nl"})

    assert updated.mode == "single-language"
    assert updated.single_language == "nl"
    assert updated.warnings == ()

    unsafe = replace(
        valid,
        raw_title="",
        sections=(
            _section(0, "zh", "\u9019\u662f\u4e2d\u6587\u6b4c\u8a5e"),
            _section(1, "nl", "A Latin-script lyric"),
        ),
    )
    with pytest.raises(PairingError, match="evidence of a second"):
        apply_section_languages(unsafe, {0: "nl", 1: "nl"})


@pytest.mark.parametrize(
    ("assignments", "message"),
    (
        ({0: "zh"}, "missing original_index values 1"),
        ({0: "zh", 1: "nl", 99: "zh"}, "unexpected original_index values 99"),
        ({0: "unknown", 1: "nl"}, "must be 'zh' or 'nl'"),
        ({0: "en", 1: "nl"}, "must be 'zh' or 'nl'"),
    ),
)
def test_assignments_must_be_complete_and_use_supported_languages(
    assignments: dict[int, str],
    message: str,
) -> None:
    parsed = ParsedLyrics(
        raw_title="",
        chinese_title="",
        dutch_title="",
        sections=(
            _section(0, "zh", "\u4e2d\u6587\u6b4c\u8a5e"),
            _section(1, "nl", "Nederlandse tekst"),
        ),
    )

    with pytest.raises(ValueError, match=message):
        apply_section_languages(parsed, assignments)  # type: ignore[arg-type]


def test_duplicate_original_indexes_are_rejected_clearly() -> None:
    parsed = ParsedLyrics(
        raw_title="",
        chinese_title="",
        dutch_title="",
        sections=(
            _section(4, "zh", "\u4e2d\u6587\u6b4c\u8a5e"),
            replace(_section(5, "nl", "Nederlandse tekst"), original_index=4),
        ),
    )

    with pytest.raises(ValueError, match="original_index values are not unique"):
        apply_section_languages(parsed, {4: "zh"})
