from __future__ import annotations

import importlib
from collections.abc import Callable

import pytest

from lyrics_dashboard.alignment import build_exact_manual_plan
from lyrics_dashboard.converter import ConversionSettings, convert_lyrics
from lyrics_dashboard.models import (
    AlignedLine,
    ParsedLyrics,
    Section,
    TranslationReference,
)
from lyrics_dashboard.splitter import split_lyric


@pytest.fixture
def clean_content_text() -> Callable[[str, str], str]:
    """Load the revised cleaner without preventing the old suite from collecting."""
    module = importlib.import_module("lyrics_dashboard.text_processing")
    return module.clean_content_text


@pytest.fixture
def split_lyric_result():
    """Load the revised detailed splitter while the legacy string API remains public."""
    module = importlib.import_module("lyrics_dashboard.splitter")
    return module.split_lyric_result


def _single_pair(
    *,
    chinese: str,
    dutch: str,
    chinese_title: str = "中文歌名",
    dutch_title: str = "Nederlandse titel",
) -> tuple[ParsedLyrics, object]:
    parsed = ParsedLyrics(
        raw_title=f"{chinese_title} {dutch_title}",
        chinese_title=chinese_title,
        dutch_title=dutch_title,
        sections=(
            Section(
                label="Verse 1",
                kind="verse",
                number=1,
                lines=(chinese,),
                language="zh",
                original_index=0,
            ),
            Section(
                label="Verse 2",
                kind="verse",
                number=2,
                lines=(dutch,),
                language="nl",
                original_index=1,
            ),
        ),
    )
    plan = build_exact_manual_plan(
        parsed,
        selections={0: (1,)},
        line_groups_by_chinese={
            0: (
                AlignedLine(
                    source_line_indices=(0,),
                    translation_references=(
                        TranslationReference(section_index=1, line_indices=(0,)),
                    ),
                ),
            )
        },
    )
    return parsed, plan


# Requirement 1
def test_western_punctuation_is_removed_from_title_and_lyric_content() -> None:
    parsed, plan = _single_pair(
        chinese="祢信實,何廣大!",
        dutch='Groot-is "Uw" trouw, o Heer.',
        chinese_title="中文(歌名)!",
        dutch_title='Nederlandse: "titel".',
    )

    output = convert_lyrics(
        parsed,
        plan,
        ConversionSettings(
            switch_index=1,
            chinese_max_length=40,
            dutch_max_length=80,
        ),
    )

    assert "[Title]\n中文歌名 Nederlandse titel" in output
    assert "祢信實何廣大|Groot is Uw trouw o Heer" in output
    assert "Groot is Uw trouw o Heer|祢信實何廣大" in output


# Requirement 2
def test_chinese_and_unicode_punctuation_is_removed(clean_content_text) -> None:
    source = "祢，信。實！何？廣；大：、—…「愛」『恩』（）《歌》"

    assert clean_content_text(source, "zh") == "祢信實何廣大愛恩歌"


# Requirement 3
def test_generated_structural_brackets_pipe_and_double_slash_remain_present() -> None:
    parsed, plan = _single_pair(
        chinese="天地玄黃宇宙洪荒",
        dutch="een lange Nederlandse liedregel",
    )

    output = convert_lyrics(
        parsed,
        plan,
        ConversionSettings(
            switch_index=1,
            chinese_max_length=4,
            dutch_max_length=12,
        ),
    )

    assert output.startswith("[Title]\n")
    assert "[Verse 1]\n" in output
    assert "[Verse 2]\n" in output
    lyric_rows = [line for line in output.splitlines() if "|" in line]
    assert lyric_rows
    assert all(line.count("|") == 1 for line in lyric_rows)
    assert all("//" in line for line in lyric_rows)


# Requirement 4
def test_punctuation_is_removed_before_length_and_split_decisions() -> None:
    assert split_lyric("天地，，玄黃", "zh", 4) == "天地玄黃"


# Requirement 5
def test_spaces_left_by_punctuation_are_normalized_without_merging_dutch_words(
    clean_content_text,
) -> None:
    source = "  Groot---is,\t\tUw   trouw...  "

    assert clean_content_text(source, "nl") == "Groot is Uw trouw"
    assert split_lyric("Alpha-beta gamma", "nl", 10) == "Alpha beta//gamma"
    assert split_lyric("Alpha-beta-gamma", "nl", 10) == "Alpha beta gamma"


# Requirement 6
@pytest.mark.parametrize(
    ("source", "language", "maximum", "expected"),
    (
        ("短歌", "zh", 4, "短歌"),
        ("kort lied", "nl", 20, "kort lied"),
    ),
)
def test_short_cleaned_text_receives_no_double_slash(
    source: str,
    language: str,
    maximum: int,
    expected: str,
) -> None:
    result = split_lyric(source, language, maximum)

    assert result == expected
    assert "//" not in result


# Requirement 7
def test_long_chinese_text_receives_exactly_one_double_slash() -> None:
    result = split_lyric("祢信實何廣大哦神我的父", "zh", 4)

    assert result.count("//") == 1
    assert all(result.split("//"))


# Requirement 8
def test_long_dutch_text_receives_exactly_one_double_slash() -> None:
    result = split_lyric(
        "Groot is Uw trouw o Heer mijn God en Vader",
        "nl",
        10,
    )

    assert result.count("//") == 1
    assert all(result.split("//"))


# Requirement 9
def test_dutch_split_never_cuts_through_a_word(clean_content_text) -> None:
    source = "Groot is Uw trouw o Heer mijn God en Vader"
    cleaned = clean_content_text(source, "nl")
    result = split_lyric(source, "nl", 18)

    assert result.count("//") == 1
    assert result.replace("//", " ").split() == cleaned.split()
    assert "trou//w" not in result


# Requirement 10
def test_recognised_two_character_chinese_word_is_not_split_internally() -> None:
    result = split_lyric("天地玄中国人民", "zh", 6)

    assert result.count("//") == 1
    assert "中//国" not in result
    assert result.replace("//", "") == "天地玄中国人民"


# Requirement 11
def test_recognised_multi_character_chinese_word_is_not_split_internally() -> None:
    source = "甲乙中华人民共和国丙"
    result = split_lyric(source, "zh", 8)
    split_index = len(result.split("//", 1)[0])
    word_start = source.index("中华人民共和国")
    word_end = word_start + len("中华人民共和国")

    assert result.count("//") == 1
    assert not word_start < split_index < word_end
    assert result.replace("//", "") == source


# Requirement 12
def test_closest_balanced_chinese_word_boundary_is_selected() -> None:
    assert (
        split_lyric("祢信實何廣大哦神我的父", "zh", 10)
        == "祢信實何廣大//哦神我的父"
    )


# Requirement 13
def test_chinese_character_split_is_used_only_after_word_and_whitespace_boundaries(
    split_lyric_result,
) -> None:
    whitespace_result = split_lyric_result("ABCDEFGHIJK LMNOP", "zh", 11)
    fallback_result = split_lyric_result("ABCDEFGHIJKLMNOP", "zh", 8)

    assert whitespace_result.text == "ABCDEFGHIJK//LMNOP"
    assert whitespace_result.used_character_fallback is False
    assert fallback_result.text == "ABCDEFGH//IJKLMNOP"
    assert fallback_result.used_character_fallback is True


# Requirement 14
def test_character_fallback_is_reported_with_conversion_context() -> None:
    parsed, plan = _single_pair(
        chinese="中华人民共和国",
        dutch="Nederlandse regel",
    )
    warnings: list[str] = []

    output = convert_lyrics(
        parsed,
        plan,
        ConversionSettings(
            switch_index=1,
            chinese_max_length=4,
            dutch_max_length=80,
        ),
        warnings=warnings,
    )

    assert output
    assert any(
        "character" in warning.lower() and "split" in warning.lower()
        for warning in warnings
    )


# Requirement 15
@pytest.mark.parametrize(
    ("source", "language"),
    (
        ("天" * 200, "zh"),
        (" ".join(f"woord{index}" for index in range(80)), "nl"),
    ),
)
def test_extremely_long_text_still_has_at_most_one_double_slash(
    source: str,
    language: str,
) -> None:
    assert split_lyric(source, language, 10).count("//") == 1


# Requirement 16
def test_complete_output_row_has_at_most_one_split_per_language() -> None:
    parsed, plan = _single_pair(
        chinese="天" * 200,
        dutch=" ".join(f"woord{index}" for index in range(80)),
    )
    output = convert_lyrics(
        parsed,
        plan,
        ConversionSettings(
            switch_index=1,
            chinese_max_length=10,
            dutch_max_length=10,
        ),
    )

    for row in (line for line in output.splitlines() if "|" in line):
        left, right = row.split("|")
        assert left.count("//") <= 1
        assert right.count("//") <= 1


# Requirement 17
def test_splitting_loses_reorders_or_rewrites_no_cleaned_content(
    clean_content_text,
) -> None:
    chinese_source = "祢信實，何廣大！哦神我的父"
    dutch_source = "Groot-is Uw trouw, o Heer mijn God en Vader."
    chinese_cleaned = clean_content_text(chinese_source, "zh")
    dutch_cleaned = clean_content_text(dutch_source, "nl")
    chinese_result = split_lyric(chinese_source, "zh", 6)
    dutch_result = split_lyric(dutch_source, "nl", 18)

    assert chinese_result.replace("//", "") == chinese_cleaned
    assert dutch_result.replace("//", " ").split() == dutch_cleaned.split()


# Requirement 18 is covered by the pre-existing parser, alignment, conversion,
# switch-point, AppTest preview, and download tests. Those tests remain unchanged.
