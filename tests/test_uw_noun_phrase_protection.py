from __future__ import annotations

import pytest

from lyrics_dashboard.alignment import build_exact_manual_plan
from lyrics_dashboard.converter import ConversionSettings, convert_lyrics
from lyrics_dashboard.models import (
    AlignedLine,
    AlignmentPlan,
    ParsedLyrics,
    TranslationReference,
)
from lyrics_dashboard.parser import parse_lyrics
from lyrics_dashboard.splitter import split_lyric, split_lyric_result


@pytest.mark.parametrize(
    ("source", "maximum", "expected", "forbidden"),
    (
        (
            "Wij zingen Uw Naam is de grootste",
            32,
            "Wij zingen//Uw Naam is de grootste",
            "Uw Naam//is",
        ),
        (
            "Wij zingen uw naam is de grootste",
            32,
            "Wij zingen//uw naam is de grootste",
            "uw naam//is",
        ),
        (
            "WIJ ZINGEN UW NAAM IS DE GROOTSTE",
            32,
            "WIJ ZINGEN//UW NAAM IS DE GROOTSTE",
            "UW NAAM//IS",
        ),
        (
            "Wij zingen Uw liefde blijft voor altijd",
            32,
            "Wij zingen//Uw liefde blijft voor altijd",
            "Uw liefde//blijft",
        ),
        (
            "Wij zingen Uw grote Naam is heilig",
            32,
            "Wij zingen//Uw grote Naam is heilig",
            "Uw grote Naam//is",
        ),
    ),
)
def test_uw_noun_phrase_stays_with_its_following_verb(
    source: str,
    maximum: int,
    expected: str,
    forbidden: str,
) -> None:
    result = split_lyric_result(
        source,
        "nl",
        maximum,
        minimum_fragment_length=2,
        minimum_fragment_ratio=0.25,
    )

    assert result.text == expected
    assert forbidden not in result.text
    assert "uw//" not in result.text.casefold()
    assert result.text.replace("//", " ").split() == source.split()


def test_repeated_uw_construction_splits_before_the_second_complete_phrase() -> None:
    source = "Uw Naam is de hoogste Uw Naam is de grootste"
    result = split_lyric_result(
        source,
        "nl",
        20,
        minimum_fragment_length=2,
        minimum_fragment_ratio=0.25,
    )

    assert result.text == "Uw Naam is de hoogste//Uw Naam is de grootste"
    assert "Uw Naam//is" not in result.text
    assert "Uw Naam is//de grootste" not in result.text


@pytest.mark.parametrize(
    "verb",
    (
        "is",
        "zijn",
        "bent",
        "ben",
        "was",
        "waren",
        "wordt",
        "worden",
        "blijft",
        "blijven",
    ),
)
def test_all_recognised_uw_following_verbs_are_protected(verb: str) -> None:
    source = f"Wij zingen Uw Naam {verb} altijd"

    assert split_lyric(source, "nl", 20) == (
        f"Wij zingen//Uw Naam {verb} altijd"
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (
            "Wij zingen Uw Naam is heilig",
            "Wij zingen//Uw Naam is heilig",
        ),
        (
            "Wij zingen Uw grote Naam is heilig",
            "Wij zingen//Uw grote Naam is heilig",
        ),
        (
            "Wij prijzen steeds Uw heilige grote Naam blijft eeuwig",
            "Wij prijzen steeds//Uw heilige grote Naam blijft eeuwig",
        ),
    ),
)
def test_uw_accepts_one_to_three_noun_phrase_tokens(
    source: str,
    expected: str,
) -> None:
    assert split_lyric(source, "nl", 20) == expected


def test_uw_protection_never_creates_a_new_split() -> None:
    source = "Wij zingen Uw Naam is groot"

    assert split_lyric(source, "nl", 40) == source


def test_sentence_initial_uw_keeps_existing_non_leading_behavior() -> None:
    source = "Uw Naam is de grootste"
    result = split_lyric(source, "nl", 20)

    assert result == "Uw Naam is//de grootste"
    assert not result.startswith("//")


def test_unbalanced_boundary_before_uw_leaves_the_line_unsplit() -> None:
    source = "Hi Uw Naam is de grootste"
    result = split_lyric_result(
        source,
        "nl",
        20,
        minimum_fragment_length=2,
        minimum_fragment_ratio=0.25,
    )

    assert result.text == source
    assert "//" not in result.text


def test_invalid_before_uw_move_uses_closest_other_balanced_safe_boundary() -> None:
    source = "Hi Uw Naam is groot vandaag zingen wij verder samen"
    result = split_lyric_result(
        source,
        "nl",
        20,
        minimum_fragment_length=2,
        minimum_fragment_ratio=0.25,
    )

    assert result.text == (
        "Hi Uw Naam is groot vandaag//zingen wij verder samen"
    )
    assert not result.text.startswith("Hi//Uw")
    assert "Uw Naam//is" not in result.text
    assert "Uw Naam is//groot" not in result.text


def test_existing_uw_with_following_word_protection_is_preserved() -> None:
    result = split_lyric("dit is uw belofte", "nl", 10)

    assert result == "dit is//uw belofte"
    assert "uw//belofte" not in result


def test_uw_construction_elsewhere_does_not_move_an_already_safe_boundary() -> None:
    source = "the melody rises brightly Uw Naam is groot"

    assert split_lyric(source, "nl", 20) == (
        "the melody rises//brightly Uw Naam is groot"
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (
            "grote lof Jouw Naam is de grootste",
            "grote lof Jouw//Naam is de grootste",
        ),
        (
            "grote lof uwnaam is de grootste",
            "grote lof uwnaam//is de grootste",
        ),
        (
            "alle eer klinkt Uw zeer heilige grote Naam blijft altijd",
            "alle eer klinkt Uw zeer heilige//grote Naam blijft altijd",
        ),
        (
            "grote lof Uw is Naam blijft altijd",
            "grote lof Uw is//Naam blijft altijd",
        ),
    ),
)
def test_uw_rule_does_not_expand_beyond_its_exact_targeted_scope(
    source: str,
    expected: str,
) -> None:
    result = split_lyric_result(
        source,
        "nl",
        10,
        minimum_fragment_length=2,
        minimum_fragment_ratio=0.25,
    )

    assert result.text == expected


def _bilingual_pair(dutch_line: str) -> tuple[ParsedLyrics, AlignmentPlan]:
    parsed = parse_lyrics(
        "中文歌名 | Nederlandse titel\n\n"
        "Verse 1\n"
        "短歌\n\n"
        "Verse 2\n"
        f"{dutch_line}\n"
    )
    plan = build_exact_manual_plan(
        parsed,
        {0: (1,)},
        {
            0: (
                AlignedLine(
                    source_line_indices=(0,),
                    translation_references=(TranslationReference(1, (0,)),),
                ),
            ),
        },
    )
    return parsed, plan


def test_uw_rule_works_before_and_after_pipe_across_language_switch() -> None:
    dutch_line = "grote lof Uw Naam is de grootste"
    parsed, plan = _bilingual_pair(dutch_line)
    output = convert_lyrics(
        parsed,
        plan,
        ConversionSettings(
            switch_index=1,
            chinese_max_length=40,
            dutch_max_length=12,
        ),
    )

    assert "短歌|grote lof//Uw Naam is de grootste" in output
    assert "grote lof//Uw Naam is de grootste|短歌" in output
    assert "Uw Naam//is" not in output
    assert "Uw Naam is//de grootste" not in output


def test_uw_rule_works_in_stricter_single_language_flow() -> None:
    dutch_line = "grote lof Uw Naam is de grootste"
    parsed = parse_lyrics(
        "Song title\n\n"
        "Verse 1\n"
        f"{dutch_line}\n"
    )
    output = convert_lyrics(
        parsed,
        None,
        ConversionSettings(dutch_max_length=12),
    )
    lyric = output.split("[Verse 1]\n", 1)[1].strip()

    assert lyric == "grote lof//Uw Naam is de grootste"
    assert "Uw Naam//is" not in lyric
    assert "Uw Naam is//de grootste" not in lyric
