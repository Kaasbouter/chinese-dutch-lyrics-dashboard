from __future__ import annotations

import math

import pytest

from lyrics_dashboard.alignment import build_exact_manual_plan
from lyrics_dashboard.converter import (
    ConversionSettings,
    convert_lyrics,
    encode_utf8_txt,
)
from lyrics_dashboard.models import AlignedLine, TranslationReference
from lyrics_dashboard.parser import parse_lyrics
from lyrics_dashboard.splitter import split_lyric_result


def _first_side(source: str, maximum: int = 20) -> str:
    return split_lyric_result(
        source,
        "nl",
        maximum,
        minimum_fragment_length=2,
        minimum_fragment_ratio=0.25,
    ).text


def _assert_words_and_balance_preserved(source: str, result: str) -> None:
    assert result.count("//") <= 1
    assert result.replace("//", " ").split() == source.split()
    if "//" in result:
        left, right = result.split("//")
        assert min(len(left), len(right)) >= max(2, math.ceil(len(source) * 0.25))


def test_life_gate_and_that_clause_use_the_requested_boundary() -> None:
    source = "And opened the life gate that all may go in"
    result = _first_side(source, 40)

    assert result == "And opened the life gate//that all may go in"
    assert "the life//gate" not in result
    assert "opened the life//gate" not in result
    _assert_words_and_balance_preserved(source, result)


def test_taught_us_and_repeated_phrase_use_the_requested_boundary() -> None:
    source = "Great things He has taught us great things He has done"
    result = _first_side(source, 40)

    assert result == "Great things He has taught us//great things He has done"
    assert "taught//us" not in result
    _assert_words_and_balance_preserved(source, result)


@pytest.mark.parametrize(
    ("source", "forbidden"),
    (
        ("We praise the holy name for every day", "the holy//name"),
        ("We praise Your great love for every day", "Your great//love"),
        ("We praise Your love forever", "Your//love"),
        ("We sing a new song for all the world", "a new//song"),
        ("Wij zien de eeuwige hoop voor iedere dag", "de eeuwige//hoop"),
        ("Wij prijzen het heilige licht voor altijd", "het heilige//licht"),
        ("Wij zingen Uw grote Naam voor ieder mens", "Uw grote//Naam"),
        ("Wij horen een nieuw lied voor iedere dag", "een nieuw//lied"),
    ),
)
def test_short_english_and_dutch_noun_phrases_stay_intact(
    source: str,
    forbidden: str,
) -> None:
    result = _first_side(source)

    assert "//" in result
    assert forbidden not in result
    _assert_words_and_balance_preserved(source, result)


@pytest.mark.parametrize(
    ("source", "forbidden"),
    (
        ("We sing the silver river forever", "silver//river"),
        ("We see the quiet morning coming", "quiet//morning"),
        ("We sing the river bank forever", "river//bank"),
    ),
)
def test_short_content_modifier_and_head_remain_together(
    source: str,
    forbidden: str,
) -> None:
    result = split_lyric_result(source, "nl", 12).text

    assert "//" in result
    assert forbidden not in result
    assert result.replace("//", " ").split() == source.split()


@pytest.mark.parametrize(
    ("source", "forbidden"),
    (
        ("Great things He loves me great things He has done", "loves//me"),
        ("Great things He loves her and great things He has done", "loves//her"),
        ("In all the world we see You and sing for joy", "see//You"),
        ("Every day He gives strength to carry on", "gives//strength"),
        ("All the world knows Him and sings for joy", "knows//Him"),
        ("We rejoice as He gave the new promise to all", "gave//the new promise"),
        ("Great things He showed His great love to us", "showed//His great love"),
        ("Hij toont Zijn grote liefde aan de wereld", "Zijn grote//liefde"),
    ),
)
def test_verb_and_direct_object_stay_together(
    source: str,
    forbidden: str,
) -> None:
    result = _first_side(source)

    assert "//" in result
    assert forbidden not in result
    _assert_words_and_balance_preserved(source, result)


@pytest.mark.parametrize(
    ("verb", "object_word"),
    (
        ("blesses", "us"),
        ("guides", "me"),
        ("saves", "me"),
        ("hears", "me"),
        ("calls", "us"),
        ("forgives", "us"),
        ("follows", "Him"),
        ("praises", "You"),
        ("receives", "me"),
    ),
)
def test_common_lyric_verbs_keep_object_pronouns(
    verb: str,
    object_word: str,
) -> None:
    source = f"Great things He {verb} {object_word} great things He has done"
    result = _first_side(source)

    assert "//" in result
    assert f"{verb}//{object_word}" not in result
    _assert_words_and_balance_preserved(source, result)


def test_safe_boundary_before_complete_verb_object_phrase_can_win() -> None:
    source = "We sing because He opened the life gate today"
    result = _first_side(source)

    assert result == "We sing because//He opened the life gate today"
    assert "opened//the" not in result
    assert "the life//gate" not in result
    _assert_words_and_balance_preserved(source, result)


@pytest.mark.parametrize(
    "source",
    (
        "the life gate",
        "the holy name",
        "Your great love",
        "a new song",
        "He taught us",
        "He loves me",
        "We see You",
    ),
)
def test_protected_phrase_alone_does_not_create_a_split(source: str) -> None:
    assert _first_side(source, 40) == source


@pytest.mark.parametrize(
    "source",
    (
        "Hi the holy name end",
        "Hi the life gate end",
        "Hi loves me now",
    ),
)
def test_no_balanced_phrase_edge_leaves_line_unsplit(source: str) -> None:
    assert _first_side(source, 10) == source


def test_nearby_that_clause_boundary_gets_a_modest_preference() -> None:
    source = "Our voices rise aloud that all may know Your love"
    result = _first_side(source)

    assert result == "Our voices rise aloud//that all may know Your love"
    _assert_words_and_balance_preserved(source, result)


def test_nearby_repeated_phrase_can_beat_an_already_safe_midpoint() -> None:
    source = "New hope fills our hearts now new hope fills the earth"
    result = _first_side(source)

    assert result == "New hope fills our hearts now//new hope fills the earth"
    _assert_words_and_balance_preserved(source, result)


def test_distant_unbalanced_repeated_phrase_cannot_pull_the_split() -> None:
    source = (
        "New hope fills our hearts and carries us through every dark night new hope"
    )
    result = _first_side(source)
    second_phrase = source.rfind("new hope")
    distant_boundary = source[:second_phrase].rstrip() + "//" + source[second_phrase:]

    assert "//" in result
    assert result != distant_boundary
    _assert_words_and_balance_preserved(source, result)


def test_distant_clause_marker_cannot_override_balance() -> None:
    source = "Hi that all the people can sing forever in peace"
    result = _first_side(source)

    assert "//" in result
    assert not result.startswith("Hi//that")
    _assert_words_and_balance_preserved(source, result)


def test_single_language_conversion_keeps_phrase_and_utf8_output_equal() -> None:
    source_line = "And opened the life gate that all may go in"
    assert _first_side(source_line, 50) == source_line
    parsed = parse_lyrics(f"Song title\n\nVerse 1\n{source_line}\n")
    output = convert_lyrics(
        parsed,
        None,
        ConversionSettings(dutch_max_length=50),
    )

    assert parsed.mode == "single-language"
    assert output == (
        "[Title]\nSong title\n\n"
        "[Verse 1]\n"
        "And opened the life gate//that all may go in\n"
    )
    assert "|" not in output
    assert encode_utf8_txt(output).decode("utf-8") == output


def test_bilingual_conversion_protects_phrase_on_both_pipe_sides_after_switch() -> None:
    source_line = "And opened the life gate that all may go in"
    parsed = parse_lyrics(
        "中文歌名 | Nederlandse titel\n\n"
        "Verse 1\n"
        "短歌\n\n"
        "Verse 2\n"
        f"{source_line}\n"
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
    output = convert_lyrics(
        parsed,
        plan,
        ConversionSettings(
            switch_index=1,
            chinese_max_length=40,
            dutch_max_length=40,
        ),
    )
    rows = [line for line in output.splitlines() if "|" in line]
    protected = "And opened the life gate//that all may go in"

    assert rows == [f"短歌|{protected}", f"{protected}|短歌"]
    assert all(side.count("//") <= 1 for row in rows for side in row.split("|"))
    assert encode_utf8_txt(output).decode("utf-8") == output
