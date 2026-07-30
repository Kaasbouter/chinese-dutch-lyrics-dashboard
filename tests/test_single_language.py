from __future__ import annotations

import pytest

from lyrics_dashboard.converter import (
    FIRST_SIDE_LIMIT_RATIO,
    ConversionSettings,
    convert_lyrics,
    derive_first_side_limit,
    encode_utf8_txt,
)
from lyrics_dashboard.errors import PairingError
from lyrics_dashboard.parser import parse_lyrics
from lyrics_dashboard.splitter import split_lyric
from lyrics_dashboard.text_processing import clean_content_text


def _convert(source: str, settings: ConversionSettings | None = None) -> str:
    parsed = parse_lyrics(source)
    return convert_lyrics(parsed, None, settings or ConversionSettings())


def test_chinese_only_input_parses_and_converts_without_placeholders() -> None:
    source = (
        "[Title]\n"
        "\u4e2d\u6587\u6b4c\u540d\n\n"
        "Verse 1\n"
        "\u9019\u662f\u4e00\u6bb5\u9700\u8981\u5206\u958b\u986f\u793a\u7684"
        "\u4e2d\u6587\u6b4c\u8a5e\n"
        "\u77ed\u6b4c\n\n"
        "Chorus 1\n"
        "\u8cdc\u6211\u6c38\u5e73\u5b89\u76f4\u5230\u6c38\u9060\n"
    )

    parsed = parse_lyrics(source)
    output = convert_lyrics(parsed, None, ConversionSettings())

    assert parsed.mode == "single-language"
    assert parsed.single_language == "zh"
    assert [section.label for section in parsed.sections] == ["Verse 1", "Chorus 1"]
    assert "|" not in output
    assert "\n|" not in output
    assert "|\n" not in output
    assert output.index("[Verse 1]") < output.index("[Chorus 1]")
    assert "[Title]\n\u4e2d\u6587\u6b4c\u540d" in output
    assert "\u77ed\u6b4c" in output


@pytest.mark.parametrize(
    ("title", "line"),
    (
        ("Genade", "Wij zingen samen voor de Heer"),
        ("Grace", "We sing together for the Lord"),
        (
            "\u0141aska",
            "\u0141aska prowadzi nas przez ka\u017cdy dzie\u0144",
        ),
        (
            "B\u00ecnh an",
            "Ch\u00fang ta c\u00f9ng h\u00e1t v\u1edbi ni\u1ec1m vui",
        ),
    ),
)
def test_latin_script_single_language_inputs_parse_and_convert(
    title: str,
    line: str,
) -> None:
    parsed = parse_lyrics(f"{title}\n\nVerse 1\n{line}\n")
    output = convert_lyrics(parsed, None, ConversionSettings())

    assert parsed.mode == "single-language"
    assert parsed.single_language == "nl"
    assert "|" not in output
    assert clean_content_text(line, "nl").replace(" ", "") == (
        output.split("[Verse 1]\n", 1)[1]
        .strip()
        .replace("//", "")
        .replace(" ", "")
    )


def test_single_language_keeps_original_section_and_line_order_exactly_once() -> None:
    source = (
        "Song title\n\n"
        "Bridge\n"
        "first unique line\n"
        "second unique line\n\n"
        "Verse 2\n"
        "third unique line\n\n"
        "Chorus 1\n"
        "fourth unique line\n"
    )

    output = _convert(source)

    expected_order = (
        "[Bridge]",
        "first unique line",
        "second unique line",
        "[Verse 2]",
        "third unique line",
        "[Chorus 1]",
        "fourth unique line",
    )
    positions = [output.index(item) for item in expected_order]
    assert positions == sorted(positions)
    assert all(output.count(line) == 1 for line in expected_order[1:])
    assert "|" not in output


def test_alignment_and_language_switch_are_not_required_in_single_mode() -> None:
    parsed = parse_lyrics("Song title\n\nVerse 1\nshort lyric line\n")

    output = convert_lyrics(
        parsed,
        None,
        ConversionSettings(title_separator="|"),
    )

    assert parsed.mode == "single-language"
    assert output == "[Title]\nSong title\n\n[Verse 1]\nshort lyric line\n"
    assert "|" not in output


def test_short_line_stays_unsplit_and_medium_line_uses_first_side_limit() -> None:
    short_line = "short lyric"
    medium_line = "een middelgrote Nederlandse liedregel"
    parsed = parse_lyrics(
        f"Title\n\nVerse 1\n{short_line}\n{medium_line}\n"
    )

    output = convert_lyrics(parsed, None, ConversionSettings())
    lyric_block = output.split("[Verse 1]\n", 1)[1].splitlines()

    assert FIRST_SIDE_LIMIT_RATIO == 0.80
    assert derive_first_side_limit(40) == 32
    assert len(clean_content_text(medium_line, "nl")) <= 40
    assert split_lyric(medium_line, "nl", 40) == medium_line
    assert lyric_block[0] == short_line
    assert "//" not in lyric_block[0]
    assert lyric_block[1] == "een middelgrote//Nederlandse liedregel"
    assert lyric_block[1].count("//") == 1


@pytest.mark.parametrize(
    ("source_line", "expected"),
    (
        ("dit is een nieuw lied", "dit is//een nieuw lied"),
        ("we sing the holy song", "we sing//the holy song"),
    ),
)
def test_single_latin_splitting_is_word_safe_and_protects_articles(
    source_line: str,
    expected: str,
) -> None:
    output = _convert(
        f"Title\n\nVerse 1\n{source_line}\n",
        ConversionSettings(dutch_max_length=10),
    )
    lyric = output.split("[Verse 1]\n", 1)[1].strip()

    assert lyric == expected
    assert lyric.replace("//", " ").split() == source_line.split()
    assert "een//" not in lyric
    assert "the//" not in lyric


def test_single_language_uw_stays_with_the_following_word() -> None:
    output = _convert(
        "Title\n\nVerse 1\ndit is uw belofte\n",
        ConversionSettings(dutch_max_length=10),
    )
    lyric = output.split("[Verse 1]\n", 1)[1].strip()

    assert lyric == "dit is//uw belofte"
    assert "uw//belofte" not in lyric


@pytest.mark.parametrize(
    ("source_line", "expected", "forbidden"),
    (
        (
            "genade trouw we believe in You",
            "genade trouw//we believe in You",
            "we//believe",
        ),
        (
            "genade trouw in het licht",
            "genade trouw//in het licht",
            "in//het licht",
        ),
    ),
)
def test_single_latin_flow_keeps_grammatical_chains_together(
    source_line: str,
    expected: str,
    forbidden: str,
) -> None:
    output = _convert(
        f"Title\n\nVerse 1\n{source_line}\n",
        ConversionSettings(dutch_max_length=13),
    )
    lyric = output.split("[Verse 1]\n", 1)[1].strip()

    assert lyric == expected
    assert forbidden not in lyric
    assert lyric.replace("//", " ").split() == source_line.split()


def test_single_chinese_flow_keeps_a_pronoun_with_its_following_token() -> None:
    source_line = "\u7532\u4e59\u6211\u76f8\u4fe1\u4e19\u4e01"
    output = _convert(
        f"\u6b4c\u540d\n\nVerse 1\n{source_line}\n",
        ConversionSettings(chinese_max_length=5),
    )
    lyric = output.split("[Verse 1]\n", 1)[1].strip()

    assert lyric == "\u7532\u4e59//\u6211\u76f8\u4fe1\u4e19\u4e01"
    assert "\u6211//\u76f8\u4fe1" not in lyric
    assert lyric.replace("//", "") == source_line


def test_single_chinese_uses_existing_targeted_word_protection() -> None:
    source_line = "\u8cdc\u6211\u6c38\u5e73\u5b89\u76f4\u5230\u6c38\u9060"
    parsed = parse_lyrics(
        f"\u6b4c\u540d\n\nVerse 1\n{source_line}\n"
    )
    warnings: list[str] = []

    output = convert_lyrics(
        parsed,
        None,
        ConversionSettings(chinese_max_length=5),
        warnings=warnings,
    )
    lyric = output.split("[Verse 1]\n", 1)[1].strip()

    assert lyric == "\u8cdc\u6211\u6c38\u5e73\u5b89//\u76f4\u5230\u6c38\u9060"
    assert "\u5e73//\u5b89" not in lyric
    assert lyric.replace("//", "") == source_line
    assert lyric.count("//") == 1


def test_single_chinese_preserves_character_fallback_warning() -> None:
    source_line = "\u4e2d\u534e\u4eba\u6c11\u5171\u548c\u56fd"
    parsed = parse_lyrics(f"Verse 1\n{source_line}\n")
    warnings: list[str] = []

    output = convert_lyrics(
        parsed,
        None,
        ConversionSettings(chinese_max_length=5),
        warnings=warnings,
    )

    assert output
    assert any("character split" in warning for warning in warnings)


def test_single_language_preserves_cleaning_and_utf8_export_identity() -> None:
    source_line = "  Groot---is,\t\tUw   trouw...  "
    parsed = parse_lyrics(f"Title!!!\n\nVerse 1\n{source_line}\n")

    preview = convert_lyrics(parsed, None, ConversionSettings())
    downloaded = encode_utf8_txt(preview)

    assert preview == "[Title]\nTitle\n\n[Verse 1]\nGroot is Uw trouw\n"
    assert downloaded.decode("utf-8") == preview
    assert "|" not in preview


def test_single_chinese_punctuation_removal_keeps_all_lyric_characters() -> None:
    source_line = (
        "\u7962\uff0c\u4fe1\u3002\u5be6\uff01\u4f55\uff1f"
        "\u5ee3\uff1b\u5927\uff1a\u611b\u6069"
    )
    cleaned = "\u7962\u4fe1\u5be6\u4f55\u5ee3\u5927\u611b\u6069"

    output = _convert(
        f"Verse 1\n{source_line}\n",
        ConversionSettings(chinese_max_length=40),
    )
    lyric = output.split("[Verse 1]\n", 1)[1].strip()

    assert lyric == cleaned
    assert lyric.replace("//", "") == clean_content_text(source_line, "zh")


@pytest.mark.parametrize(
    ("source", "settings"),
    (
        (
            " ".join(f"word{index}" for index in range(40)),
            ConversionSettings(dutch_max_length=10),
        ),
        (
            "\u5929" * 80,
            ConversionSettings(chinese_max_length=4),
        ),
    ),
)
def test_single_language_line_gets_at_most_one_double_slash(
    source: str,
    settings: ConversionSettings,
) -> None:
    output = _convert(f"Verse 1\n{source}\n", settings)
    lyric = output.split("[Verse 1]\n", 1)[1].strip()

    assert lyric.count("//") <= 1


def test_malformed_bilingual_lines_are_not_accepted_as_single_language() -> None:
    source = (
        "Verse 1\n"
        "\u9019\u662f\u4e2d\u6587\u6b4c\u8a5e\n"
        "This is the translated lyric\n"
    )

    with pytest.raises(PairingError, match="evidence of a second"):
        parse_lyrics(source)


def test_bilingual_title_with_one_detected_lyric_side_is_rejected() -> None:
    source = (
        "[Title]\n"
        "\u4e2d\u6587\u6b4c\u540d | Dutch title\n\n"
        "Verse 1\n"
        "\u9019\u662f\u4e2d\u6587\u6b4c\u8a5e\n"
    )

    with pytest.raises(PairingError, match="title contains bilingual evidence"):
        parse_lyrics(source)


@pytest.mark.parametrize(
    ("source", "message"),
    (
        (
            "[Title]\nSong |\n\nVerse 1\nsingle lyric line\n",
            "title contains bilingual evidence",
        ),
        (
            "Verse 1\nsingle lyric | translated lyric\n",
            r"contains `\|`",
        ),
    ),
)
def test_single_language_input_cannot_leak_a_pipe(
    source: str,
    message: str,
) -> None:
    with pytest.raises(PairingError, match=message):
        parse_lyrics(source)


def test_merged_unsupported_script_is_not_accepted_as_single_language() -> None:
    source = (
        "Verse 1\n"
        "This is the English lyric\n"
        "\u042d\u0442\u043e \u0440\u0443\u0441\u0441\u043a\u0438\u0439 "
        "\u043f\u0435\u0440\u0435\u0432\u043e\u0434\n"
    )

    with pytest.raises(PairingError, match="second or unsupported language"):
        parse_lyrics(source)


def test_missing_alignment_is_still_rejected_for_bilingual_input() -> None:
    parsed = parse_lyrics(
        "Verse 1\n"
        "\u9019\u662f\u4e2d\u6587\u6b4c\u8a5e\n\n"
        "Verse 2\n"
        "Dit is de Nederlandse liedtekst\n"
    )

    assert parsed.mode == "bilingual"
    with pytest.raises(PairingError, match="validated alignment plan"):
        convert_lyrics(parsed, None, ConversionSettings())
