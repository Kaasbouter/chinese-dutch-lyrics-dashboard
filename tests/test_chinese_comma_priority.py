from __future__ import annotations

import importlib

import pytest

from lyrics_dashboard.splitter import split_lyric, split_lyric_result
from lyrics_dashboard.text_processing import clean_content_result, clean_content_text


def _first_side(source: str, maximum: int = 8) -> str:
    return split_lyric_result(
        source,
        "zh",
        maximum,
        minimum_fragment_length=2,
        minimum_fragment_ratio=0.25,
    ).text


@pytest.mark.parametrize("comma", ("，", ","))
def test_original_comma_position_survives_cleaning(comma: str) -> None:
    cleaned = clean_content_result(f"天地玄黃{comma}祢是永遠君王萬歲", "zh")

    assert cleaned.text == "天地玄黃祢是永遠君王萬歲"
    assert cleaned.original_comma_boundaries == (4,)
    assert cleaned.original_whitespace_boundaries == ()
    assert cleaned.punctuation_separator_boundaries == ()


def test_multiple_comma_positions_follow_cleaned_text_and_ignore_edges() -> None:
    cleaned = clean_content_result("！  甲乙，，丙丁；戊己,庚辛 ？", "zh")

    assert cleaned.text == "甲乙丙丁戊己庚辛"
    assert cleaned.original_comma_boundaries == (2, 6)
    assert cleaned.original_whitespace_boundaries == ()


@pytest.mark.parametrize("comma", ("，", ","))
def test_safe_comma_takes_priority_over_more_balanced_jieba(comma: str) -> None:
    # This source boundary is 4/9, while ordinary candidates can balance better.
    result = _first_side(f"天地玄黃{comma}祢是永遠君王萬歲", 7)

    assert result == "天地玄黃//祢是永遠君王萬歲"
    assert _first_side("天地玄黃祢是永遠君王萬歲", 7) == (
        "天地玄黃祢是//永遠君王萬歲"
    )
    assert result.count("//") == 1
    assert comma not in result


@pytest.mark.parametrize("separator", ("，", ",", " "))
def test_source_boundary_after_coverb_keeps_existing_authority(
    separator: str,
) -> None:
    assert _first_side(f"甲乙在{separator}祢裡面丙丁", 4) == (
        "甲乙在//祢裡面丙丁"
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (
            "甲乙丙丁 戊己，庚辛壬癸子丑",
            "甲乙丙丁 戊己//庚辛壬癸子丑",
        ),
        (
            "甲乙丙丁，戊己 庚辛壬癸子丑",
            "甲乙丙丁戊己//庚辛壬癸子丑",
        ),
    ),
)
def test_space_and_comma_share_ranking_and_either_type_can_win(
    source: str, expected: str
) -> None:
    cleaned = clean_content_result(source, "zh")

    assert len(cleaned.original_whitespace_boundaries) == 1
    assert len(cleaned.original_comma_boundaries) == 1
    assert _first_side(source, 4) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (
            "甲乙，丙丁戊己，庚辛壬癸，子丑",
            "甲乙丙丁戊己//庚辛壬癸子丑",
        ),
        (
            "甲 乙丙丁，戊己庚辛 壬癸，子丑",
            "甲 乙丙丁//戊己庚辛 壬癸子丑",
        ),
    ),
)
def test_all_source_candidates_are_tried_and_unselected_commas_are_removed(
    source: str, expected: str
) -> None:
    result = _first_side(source, 4)

    assert result == expected
    assert "," not in result and "，" not in result
    assert result.count("//") == 1
    assert result.replace("//", "").replace(" ", "") == (
        clean_content_text(source, "zh").replace(" ", "")
    )


@pytest.mark.parametrize("separator", ("， ", " ，", "\t，  \u3000", ", \t， "))
def test_adjacent_comma_and_whitespace_coalesce_to_one_boundary(
    separator: str,
) -> None:
    source = f"天地玄黃{separator}祢是永遠君王萬歲"
    cleaned = clean_content_result(source, "zh")

    assert cleaned.text == "天地玄黃 祢是永遠君王萬歲"
    assert cleaned.original_whitespace_boundaries == (5,)
    assert cleaned.original_comma_boundaries == (5,)
    assert len(
        set(cleaned.original_whitespace_boundaries)
        | set(cleaned.original_comma_boundaries)
    ) == 1
    assert _first_side(source) == "天地玄黃//祢是永遠君王萬歲"


@pytest.mark.parametrize(
    ("source", "maximum"),
    (
        ("祢愛我，我愛祢", 6),
        ("祢愛我,我愛祢", 6),
        ("祢愛我， 我愛祢", 6),
        ("甲乙，丙丁，戊己", 6),
    ),
)
def test_commas_never_create_a_split_within_existing_threshold(
    source: str, maximum: int
) -> None:
    assert _first_side(source, maximum) == clean_content_text(source, "zh")


@pytest.mark.parametrize("comma", ("，", ","))
def test_two_eight_comma_is_rejected_and_ordinary_candidates_remain(
    comma: str,
) -> None:
    source = f"甲乙{comma}丙丁戊己庚辛壬癸"
    result = _first_side(source, 4)

    assert result == _first_side("甲乙丙丁戊己庚辛壬癸", 4)
    assert result != "甲乙//丙丁戊己庚辛壬癸"
    assert min(map(len, result.split("//"))) >= 3


@pytest.mark.parametrize("separator", ("，", " "))
def test_invalid_comma_tries_other_source_types_before_ordinary_candidates(
    separator: str,
) -> None:
    assert _first_side(f"甲乙，丙丁戊己{separator}庚辛壬癸子丑", 4) == (
        "甲乙丙丁戊己//庚辛壬癸子丑"
    )


def test_rejected_comma_keeps_ordinary_chinese_grammar_protection() -> None:
    assert _first_side("甲，乙在祢裡面丙丁", 4) == "甲乙//在祢裡面丙丁"
    assert _first_side("甲乙在祢裡面丙丁", 4) == "甲乙//在祢裡面丙丁"


def test_rejected_comma_keeps_two_character_fallback_protection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    splitter = importlib.import_module("lyrics_dashboard.splitter")
    monkeypatch.setattr(splitter, "_chinese_word_boundaries", lambda _text: ())

    result = splitter.split_lyric_result(
        "賜，我永平安直到永遠",
        "zh",
        4,
        minimum_fragment_length=2,
        minimum_fragment_ratio=0.25,
    )

    assert result.text == "賜我永平安//直到永遠"
    assert "平//安" not in result.text
    assert result.used_character_fallback is True


def test_extremely_long_line_keeps_one_split_and_removes_every_comma() -> None:
    source = ("天" * 50) + "，" + ("地" * 50) + " , " + ("玄" * 50)
    result = _first_side(source)

    assert result.count("//") == 1
    assert "," not in result and "，" not in result
    assert result.replace("//", "").replace(" ", "") == (
        "天" * 50 + "地" * 50 + "玄" * 50
    )


@pytest.mark.parametrize(
    ("source", "maximum", "expected"),
    (
        ("Alpha-beta gamma", 10, "Alpha beta//gamma"),
        ("dit is uw belofte", 10, "dit is//uw belofte"),
        ("wij leven door Uw genade", 12, "wij leven//door Uw genade"),
        (
            "genade,trouw you are the light",
            4,
            "genade trouw//you are the light",
        ),
        (
            "genade trouw，you are the light",
            4,
            "genade trouw you are//the light",
        ),
    ),
)
def test_latin_splitting_and_bytes_remain_unchanged(
    source: str, maximum: int, expected: str
) -> None:
    assert clean_content_result(source, "nl").original_comma_boundaries == ()
    assert split_lyric(source, "nl", maximum).encode("utf-8") == expected.encode(
        "utf-8"
    )
