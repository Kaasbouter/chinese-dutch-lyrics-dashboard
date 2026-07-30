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


@pytest.mark.parametrize(
    ("source", "expected", "forbidden"),
    (
        (
            "groot U bent de bron",
            "groot//U bent de bron",
            "U//bent",
        ),
        (
            "groot ik geloof in U",
            "groot//ik geloof in U",
            "ik//geloof",
        ),
        (
            "genade trouw wij vertrouwen op U",
            "genade trouw//wij vertrouwen op U",
            "wij//vertrouwen",
        ),
    ),
)
def test_dutch_personal_pronouns_move_with_their_following_word(
    source: str,
    expected: str,
    forbidden: str,
) -> None:
    result = split_lyric(source, "nl", 4)

    assert result == expected
    assert forbidden not in result


@pytest.mark.parametrize(
    ("source", "expected", "forbidden"),
    (
        (
            "genade trouw you are the light",
            "genade trouw//you are the light",
            "you//are",
        ),
        (
            "genade trouw we believe in You",
            "genade trouw//we believe in You",
            "we//believe",
        ),
        (
            "groot I trust in You",
            "groot//I trust in You",
            "I//trust",
        ),
    ),
)
def test_english_personal_pronouns_move_with_their_following_word(
    source: str,
    expected: str,
    forbidden: str,
) -> None:
    result = split_lyric(source, "nl", 4)

    assert result == expected
    assert forbidden not in result


@pytest.mark.parametrize(
    ("source", "maximum", "expected", "forbidden"),
    (
        ("zing van hoop vervuld", 4, "zing//van hoop vervuld", "van//hoop"),
        (
            "genade trouw in het licht",
            10,
            "genade trouw//in het licht",
            "in//het licht",
        ),
        ("zing met U wandelen", 4, "zing//met U wandelen", "met//U"),
        ("ja voor altijd zingen", 4, "ja//voor altijd zingen", "voor//altijd"),
        ("wij gaan naar U", 8, "wij gaan//naar U", "naar//U"),
        (
            "wij leven door Uw genade",
            12,
            "wij leven//door Uw genade",
            "door//Uw genade",
        ),
    ),
)
def test_dutch_prepositions_stay_with_their_local_phrase(
    source: str,
    maximum: int,
    expected: str,
    forbidden: str,
) -> None:
    result = split_lyric(source, "nl", maximum)

    assert result == expected
    assert forbidden not in result


@pytest.mark.parametrize(
    ("source", "maximum", "expected", "forbidden"),
    (
        ("zing of hope filled", 4, "zing//of hope filled", "of//hope"),
        (
            "genade trouw in the light",
            10,
            "genade trouw//in the light",
            "in//the light",
        ),
        ("ja with You walking", 4, "ja//with You walking", "with//You"),
        ("zing for ever singing", 4, "zing//for ever singing", "for//ever"),
        ("we sing to You", 7, "we sing//to You", "to//You"),
        (
            "genade trouw through Your grace",
            10,
            "genade trouw//through Your grace",
            "through//Your grace",
        ),
    ),
)
def test_english_prepositions_stay_with_their_local_phrase(
    source: str,
    maximum: int,
    expected: str,
    forbidden: str,
) -> None:
    result = split_lyric(source, "nl", maximum)

    assert result == expected
    assert forbidden not in result


@pytest.mark.parametrize(
    "chain",
    (
        "in het licht",
        "van de Heer",
        "of the Lord",
        "with a promise",
        "voor een nieuw begin",
        "with Your strength",
        "for a new beginning",
    ),
)
def test_consecutive_lead_words_form_one_short_protected_chain(
    chain: str,
) -> None:
    source = f"genade trouw {chain}"

    assert split_lyric(source, "nl", 10) == f"genade trouw//{chain}"


def test_balanced_boundary_before_a_mid_line_chain_is_preferred(
    split_lyric_result,
) -> None:
    source = "aaaa bbbb of the Lord zzzz yyyy"
    result = split_lyric_result(
        source,
        "nl",
        13,
        minimum_fragment_length=2,
        minimum_fragment_ratio=0.25,
    )

    assert result.text == "aaaa bbbb//of the Lord zzzz yyyy"
    assert "of//the Lord" not in result.text
    assert "of the//Lord" not in result.text


def test_before_chain_preference_preserves_within_limit_ranking(
    split_lyric_result,
) -> None:
    source = "aaaa bbbb in the light zzzz yyyy qqq"
    result = split_lyric_result(
        source,
        "nl",
        22,
        minimum_fragment_length=2,
        minimum_fragment_ratio=0.25,
    )

    assert result.text == "aaaa bbbb in the light//zzzz yyyy qqq"
    assert all(len(part) <= 22 for part in result.text.split("//"))


def test_before_chain_preference_does_not_worsen_balance() -> None:
    assert (
        split_lyric("ja in het licht staan", "nl", 4)
        == "ja in het licht//staan"
    )


@pytest.mark.parametrize(
    ("source", "maximum", "forbidden"),
    (
        (
            "aa voor een nieuw begin zz",
            4,
            "voor een nieuw//begin",
        ),
        (
            "we worship with Your strength today",
            17,
            "with Your//strength",
        ),
        (
            "we live through Your grace",
            13,
            "through Your//grace",
        ),
        ("de new ik alpha", 4, "ik//alpha"),
        ("for a new we believe", 4, "we//believe"),
    ),
)
def test_local_determiner_and_modifier_chains_have_no_internal_boundary(
    source: str,
    maximum: int,
    forbidden: str,
) -> None:
    result = split_lyric(source, "nl", maximum)

    assert forbidden not in result
    assert result.replace("//", " ").split() == source.split()


@pytest.mark.parametrize(
    ("source", "expected", "forbidden"),
    (
        (
            "\u7532\u4e59\u6211\u76f8\u4fe1\u4e19\u4e01",
            "\u7532\u4e59//\u6211\u76f8\u4fe1\u4e19\u4e01",
            "\u6211//\u76f8\u4fe1",
        ),
        (
            "\u7532\u4e59\u5728\u7962\u88e1\u9762\u4e19\u4e01",
            "\u7532\u4e59//\u5728\u7962\u88e1\u9762\u4e19\u4e01",
            "\u5728//\u7962\u88e1\u9762",
        ),
        (
            "\u7532\u4e59\u5979\u5011\u76f8\u4fe1\u4e19\u4e01",
            "\u7532\u4e59//\u5979\u5011\u76f8\u4fe1\u4e19\u4e01",
            "\u5979\u5011//\u76f8\u4fe1",
        ),
        (
            "\u7532\u4e59\u70ba\u4e86\u76fc\u671b\u4e19\u4e01",
            "\u7532\u4e59//\u70ba\u4e86\u76fc\u671b\u4e19\u4e01",
            "\u70ba\u4e86//\u76fc\u671b",
        ),
    ),
)
def test_chinese_pronouns_and_coverbs_use_existing_jieba_token_spans(
    source: str,
    expected: str,
    forbidden: str,
) -> None:
    result = split_lyric(source, "zh", 4)

    assert result == expected
    assert forbidden not in result
    assert result.replace("//", "") == source


def test_chinese_protected_chain_at_start_never_creates_a_leading_split() -> None:
    source = "\u5728\u7962\u88e1\u9762\u6211\u6709\u5e73\u5b89"
    result = split_lyric(source, "zh", 4)

    assert result == "\u5728\u7962\u88e1\u9762//\u6211\u6709\u5e73\u5b89"
    assert not result.startswith("//")
    assert "\u5728//" not in result
    assert result.replace("//", "") == source


def test_character_fallback_applies_grammar_and_two_character_safety_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    splitter = importlib.import_module("lyrics_dashboard.splitter")
    monkeypatch.setattr(splitter, "_chinese_word_boundaries", lambda _text: ())
    source = "\u7532\u4e59\u6211\u76f8\u4fe1\u4e19\u4e01"

    result = splitter.split_lyric_result(source, "zh", 4)

    assert result.text == "\u7532\u4e59//\u6211\u76f8\u4fe1\u4e19\u4e01"
    assert "\u6211//\u76f8\u4fe1" not in result.text
    assert "\u76f8//\u4fe1" not in result.text
    assert result.text.replace("//", "") == source
    assert result.used_character_fallback is True


@pytest.mark.parametrize(
    ("source", "language", "maximum"),
    (
        ("U bent", "nl", 20),
        ("the light", "nl", 20),
        ("in het licht", "nl", 20),
        ("\u6211\u76f8\u4fe1", "zh", 8),
        ("\u5728\u7962\u88e1\u9762", "zh", 8),
    ),
)
def test_grammatical_protection_never_creates_a_new_split(
    source: str,
    language: str,
    maximum: int,
) -> None:
    assert split_lyric(source, language, maximum) == source


def test_protected_word_elsewhere_does_not_change_a_safe_planned_boundary() -> None:
    source = "the melody rises brightly in our hearts"

    assert split_lyric(source, "nl", 16) == (
        "the melody rises//brightly in our hearts"
    )


def test_invalid_before_chain_move_uses_the_closest_balanced_safe_boundary(
    split_lyric_result,
) -> None:
    result = split_lyric_result(
        "x wij ga huis",
        "nl",
        4,
        minimum_fragment_length=2,
        minimum_fragment_ratio=0.25,
    )

    assert result.text == "x wij ga//huis"
    assert "wij//ga" not in result.text
    assert min(map(len, result.text.split("//"))) >= 4


def test_line_remains_unsplit_when_no_grammatical_and_balanced_boundary_exists(
    split_lyric_result,
) -> None:
    source = "x wij superlangwoord"
    result = split_lyric_result(
        source,
        "nl",
        4,
        minimum_fragment_length=2,
        minimum_fragment_ratio=0.25,
    )

    assert result.text == source
    assert "//" not in result.text


@pytest.mark.parametrize("source", ("aa bbbbbbbb", "aaaaaaaa bb"))
def test_existing_quarter_fragment_guard_rejects_two_eight_style_splits(
    split_lyric_result,
    source: str,
) -> None:
    result = split_lyric_result(
        source,
        "nl",
        4,
        minimum_fragment_length=2,
        minimum_fragment_ratio=0.25,
    )

    assert result.text == source
    assert "//" not in result.text


def test_protected_matching_is_case_insensitive_but_never_partial() -> None:
    assert (
        split_lyric("genade trouw WE believe in You", "nl", 4)
        == "genade trouw//WE believe in You"
    )
    assert (
        split_lyric("genade trouw IN the light", "nl", 4)
        == "genade trouw//IN the light"
    )
    assert (
        split_lyric("we sing insideout melodies now", "nl", 12)
        == "we sing insideout//melodies now"
    )


@pytest.mark.parametrize(
    "protected_word",
    (
        "de",
        "het",
        "een",
        "uw",
        "a",
        "an",
        "the",
        "De",
        "HET",
        "Een",
        "Uw",
        "UW",
        "The",
        "AN",
        "A",
    ),
)
def test_middle_protected_words_stay_with_the_following_word(
    protected_word: str,
) -> None:
    assert (
        split_lyric(f"dit is {protected_word} nieuw lied", "nl", 8)
        == f"dit is//{protected_word} nieuw lied"
    )


def test_boundaries_before_complete_chains_and_after_their_follower_are_allowed() -> None:
    assert split_lyric("ik kom tot de Heer", "nl", 10) == "ik kom//tot de Heer"
    assert (
        split_lyric("ik kom tot de Heer om Hem te aanbidden", "nl", 10)
        == "ik kom tot de Heer//om Hem te aanbidden"
    )
    assert (
        split_lyric("dit is uw belofte voor altijd", "nl", 10)
        == "dit is uw belofte//voor altijd"
    )


def test_protection_uses_exact_standalone_tokens() -> None:
    assert (
        split_lyric("dit is uwbelofte voor altijd", "nl", 10)
        == "dit is uwbelofte//voor altijd"
    )
    assert (
        split_lyric("dit uw is een mooie belofte", "nl", 10)
        == "dit uw is//een mooie belofte"
    )
    assert split_lyric("dit is u belofte", "nl", 10) == "dit is//u belofte"


def test_sentence_initial_protected_words_never_create_a_leading_split() -> None:
    assert split_lyric("The Lord", "nl", 4) == "The Lord"
    assert split_lyric("Uw belofte", "nl", 4) == "Uw belofte"
    assert (
        split_lyric("dit theater is mooi", "nl", 4)
        == "dit theater//is mooi"
    )


@pytest.mark.parametrize("protected_word", ("de", "uw"))
def test_text_remains_unsplit_when_only_boundary_would_separate_a_protected_word(
    split_lyric_result,
    protected_word: str,
) -> None:
    result = split_lyric_result(
        f"x {protected_word} woord",
        "nl",
        4,
        minimum_fragment_length=3,
    )

    assert result.text == f"x {protected_word} woord"
    assert f"{protected_word}//" not in result.text


def test_planned_split_moves_to_closest_balanced_edge_of_two_character_word(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    splitter = importlib.import_module("lyrics_dashboard.splitter")
    monkeypatch.setattr(splitter, "_chinese_word_boundaries", lambda _text: ())
    source = "\u8cdc\u6211\u6c38\u5e73\u5b89\u76f4\u5230\u6c38\u9060"
    result = splitter.split_lyric_result(source, "zh", 4)

    assert result.text == "\u8cdc\u6211\u6c38\u5e73\u5b89//\u76f4\u5230\u6c38\u9060"
    assert "\u5e73//\u5b89" not in result.text
    assert result.text.replace("//", "") == source
    assert result.used_character_fallback is True


def test_subpair_inside_a_longer_jieba_token_gets_no_new_special_treatment(
    split_lyric_result,
) -> None:
    source = "\u7532\u4e59\u4e2d\u534e\u4eba\u6c11\u5171\u548c\u56fd\u4e19"
    result = split_lyric_result(
        source,
        "zh",
        4,
        minimum_fragment_length=4,
    )

    assert result.text == "\u7532\u4e59\u4e2d\u534e\u4eba//\u6c11\u5171\u548c\u56fd\u4e19"
    assert result.text.replace("//", "") == source
    assert result.used_character_fallback is True


def test_two_character_word_elsewhere_does_not_change_the_planned_split() -> None:
    source = "\u7532\u4fe1\u5be6\u4e59\u6211\u4f60\u4e19\u4e01"

    assert (
        split_lyric(source, "zh", 4)
        == "\u7532\u4fe1\u5be6\u4e59//\u6211\u4f60\u4e19\u4e01"
    )


def test_two_character_words_do_not_cause_short_text_to_split() -> None:
    source = "\u7532\u4fe1\u5be6\u4e59"

    assert split_lyric(source, "zh", 4) == source


def test_unrecognised_character_pair_is_not_specially_protected(
    split_lyric_result,
) -> None:
    source = "\u7532\u4e59\u58ec\u7678\u4e19\u4e01"
    result = split_lyric_result(
        source,
        "zh",
        4,
        minimum_fragment_length=3,
    )

    assert result.text == "\u7532\u4e59\u58ec//\u7678\u4e19\u4e01"
    assert result.text.replace("//", "") == source


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (
            "\u5e73\u5b89\u7532\u4e59\u4e19\u4e01\u620a\u5df1",
            "\u5e73\u5b89//\u7532\u4e59\u4e19\u4e01\u620a\u5df1",
        ),
        (
            "\u7532\u4e59\u4e19\u4e01\u620a\u5df1\u5e73\u5b89",
            "\u7532\u4e59\u4e19\u4e01//\u620a\u5df1\u5e73\u5b89",
        ),
    ),
)
def test_two_character_words_at_text_edges_get_no_new_special_treatment(
    source: str,
    expected: str,
) -> None:
    assert split_lyric(source, "zh", 4) == expected


def test_chinese_text_remains_unsplit_when_no_safe_adjustment_exists(
    split_lyric_result,
) -> None:
    source = "\u7532\u4e59\u5e73\u5b89\u4e19\u4e01"
    result = split_lyric_result(
        source,
        "zh",
        4,
        minimum_fragment_length=3,
    )

    assert result.text == source
    assert "\u5e73//\u5b89" not in result.text
    assert result.used_character_fallback is False


# Requirement 18 is covered by the pre-existing parser, alignment, conversion,
# switch-point, AppTest preview, and download tests. Those tests remain unchanged.
