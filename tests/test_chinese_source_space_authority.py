from __future__ import annotations

import pytest

from lyrics_dashboard.converter import ConversionSettings, convert_lyrics
from lyrics_dashboard.parser import parse_lyrics
from lyrics_dashboard.splitter import split_lyric_result
from lyrics_dashboard.text_processing import clean_content_result


def _first_side(source: str, maximum: int = 8) -> str:
    return split_lyric_result(
        source,
        "zh",
        maximum,
        minimum_fragment_length=2,
        minimum_fragment_ratio=0.25,
    ).text


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (
            "我要永遠愛祢 永遠要堅定",
            "我要永遠愛祢//永遠要堅定",
        ),
        (
            "神啊 我的心切慕祢 如鹿切慕溪水",
            "神啊 我的心切慕祢//如鹿切慕溪水",
        ),
    ),
)
def test_reported_source_spaces_override_chinese_grammar(
    source: str, expected: str
) -> None:
    result = _first_side(source)

    assert result == expected
    assert result.count("//") == 1


def test_reported_case_uses_converter_first_side_limit() -> None:
    parsed = parse_lyrics("Verse 1\n我要永遠愛祢 永遠要堅定\n")

    assert convert_lyrics(
        parsed, None, ConversionSettings(chinese_max_length=10)
    ) == "[Verse 1]\n我要永遠愛祢//永遠要堅定\n"


@pytest.mark.parametrize("pronoun", ("祢", "我", "你", "祂", "他", "她", "您"))
def test_source_space_after_protected_pronoun_is_authoritative(
    pronoun: str,
) -> None:
    source = f"天地玄黃{pronoun} 宇宙洪荒萬歲"

    assert _first_side(source) == f"天地玄黃{pronoun}//宇宙洪荒萬歲"


def test_source_space_after_coverb_overrides_chain_protection() -> None:
    assert _first_side("甲乙在 祢裡面丙丁", 4) == "甲乙在//祢裡面丙丁"


def test_all_source_spaces_rank_before_jieba_and_best_balance_wins() -> None:
    # The first space follows protected 我. Both source spaces pass the 25%
    # rule, and the earlier one wins the existing balance tie.
    assert _first_side("甲乙丙我 相信戊己 庚辛壬癸", 9) == (
        "甲乙丙我//相信戊己 庚辛壬癸"
    )
    # A 4/9 source-space split still takes priority over a non-space word
    # boundary that could put both fragments under the configured limit.
    assert _first_side("天地玄黃 祢是永遠君王萬歲", 7) == (
        "天地玄黃//祢是永遠君王萬歲"
    )


def test_invalid_early_space_does_not_hide_balanced_later_space() -> None:
    assert _first_side("甲 乙丙丁戊己 庚辛壬癸", 9) == (
        "甲 乙丙丁戊己//庚辛壬癸"
    )


def test_unbalanced_source_space_falls_back_to_ordinary_chinese_candidates() -> None:
    source = "甲乙 丙丁戊己庚辛壬癸子丑"
    result = _first_side(source, 4)

    assert result == "甲乙 丙丁戊己//庚辛壬癸子丑"
    assert result.count("//") == 1


def test_all_unbalanced_source_spaces_allow_ordinary_chinese_candidates() -> None:
    source = "甲 乙丙丁戊己庚辛 壬"
    result = _first_side(source, 4)

    assert result == "甲 乙丙//丁戊己庚辛 壬"
    assert result.count("//") == 1


def test_source_space_does_not_trigger_split_under_existing_limit() -> None:
    assert _first_side("祢愛我 我愛祢", 6) == "祢愛我 我愛祢"


def test_nonspace_chinese_chain_protection_remains_active() -> None:
    assert _first_side("甲乙在祢裡面丙丁", 4) == "甲乙//在祢裡面丙丁"


def test_punctuation_generated_separator_is_not_a_source_space() -> None:
    cleaned = clean_content_result("甲乙ABC，DEF戊己庚辛", "zh")

    assert cleaned.text == "甲乙ABC DEF戊己庚辛"
    assert cleaned.original_whitespace_boundaries == ()
    assert cleaned.punctuation_separator_boundaries == (5, 6)
