from __future__ import annotations

import pytest

from lyrics_dashboard.alignment import build_exact_manual_plan
from lyrics_dashboard.converter import ConversionSettings, convert_lyrics, encode_utf8_txt
from lyrics_dashboard.extractors import TEXT_EXTENSIONS, extract_text
from lyrics_dashboard.models import AlignedLine, TranslationReference
from lyrics_dashboard.parser import parse_lyrics


@pytest.mark.parametrize("explicit_title", [False, True])
@pytest.mark.parametrize(
    ("title", "lyric", "expected_title"),
    [
        ("135 人算什麼 (分2)", "祢愛我們", "135 人算什麼 分2"),
        ("001 中文歌名", "祢愛我們", "001 中文歌名"),
        ("001 Groot is Uw trouw", "Uw trouw is groot", "001 Groot is Uw trouw"),
        ("42 Holy Forever", "We sing", "42 Holy Forever"),
        ("001.Groot is Uw trouw", "Uw trouw is groot", "001 Groot is Uw trouw"),
        ("42-中文歌名", "祢愛我們", "42 中文歌名"),
        ("001中文歌名", "祢愛我們", "001中文歌名"),
        ("人算什麼 (分2)", "祢愛我們", "人算什麼 分2"),
        ("Groot is Uw trouw", "Uw trouw is groot", "Groot is Uw trouw"),
        ("Holy Forever", "We sing", "Holy Forever"),
        ("Psalm 23", "We sing", "Psalm 23"),
    ],
)
def test_single_language_title_preserves_digits_and_existing_cleaning(
    explicit_title: bool,
    title: str,
    lyric: str,
    expected_title: str,
) -> None:
    source = ("[Title]\n" if explicit_title else "") + f"{title}\n\nVerse 1\n{lyric}\n"
    parsed = parse_lyrics(source)
    output = convert_lyrics(parsed, None, ConversionSettings())

    assert parsed.raw_title == title
    assert (parsed.chinese_title or parsed.dutch_title) == title
    assert output.splitlines()[:2] == ["[Title]", expected_title]
    assert encode_utf8_txt(output).decode("utf-8") == output


@pytest.mark.parametrize("separator", [" ", "|"])
@pytest.mark.parametrize("switch_index", [None, 0, 1])
@pytest.mark.parametrize(
    ("title", "expected_parts"),
    [
        ("135 中文歌名 | Holy Forever", ("135 中文歌名", "Holy Forever")),
        ("001 Holy Forever | 中文歌名", ("001 中文歌名", "Holy Forever")),
        ("001 中文歌名 Holy Forever", ("001 中文歌名", "Holy Forever")),
        ("001 Holy Forever 中文歌名", ("001 中文歌名", "Holy Forever")),
        ("001 中文歌名\nHoly Forever", ("001 中文歌名", "Holy Forever")),
        ("001 Holy Forever\n中文歌名", ("001 中文歌名", "Holy Forever")),
        ("001. Holy Forever | 中文歌名", ("001 中文歌名", "Holy Forever")),
        ("001：Holy Forever | 中文歌名", ("001 中文歌名", "Holy Forever")),
        ("001—Holy Forever | 中文歌名", ("001 中文歌名", "Holy Forever")),
        ("001/Holy Forever | 中文歌名", ("001 中文歌名", "Holy Forever")),
        ("001Holy Forever | 中文歌名", ("001中文歌名", "Holy Forever")),
        ("中文歌名 | Holy Forever", ("中文歌名", "Holy Forever")),
        ("Holy Forever | 中文歌名", ("中文歌名", "Holy Forever")),
    ],
)
def test_bilingual_title_keeps_number_first_without_changing_language_order(
    title: str,
    expected_parts: tuple[str, str],
    separator: str,
    switch_index: int | None,
) -> None:
    parsed = parse_lyrics(f"{title}\n\nVerse 1\n祢愛我們\n\nChorus 2\nWe sing\n")
    plan = build_exact_manual_plan(
        parsed,
        {0: [1]},
        {0: (AlignedLine((0,), (TranslationReference(1, (0,)),)),)},
    )
    output = convert_lyrics(
        parsed,
        plan,
        ConversionSettings(title_separator=separator, switch_index=switch_index),
    )

    assert parsed.raw_title == title.replace("\n", " ")
    assert output.splitlines()[:2] == ["[Title]", separator.join(expected_parts)]
    if title.startswith(("135", "001")):
        number = title[:3]
        assert output.splitlines()[1].startswith(number)
        assert output.splitlines()[1].count(number) == 1
    assert "[Verse 1]" in output
    assert "[Chorus 2]" in output


def test_title_number_does_not_change_section_numbers_or_numeric_lyrics() -> None:
    parsed = parse_lyrics(
        "135 Holy Forever\n\n"
        "Verse 2\n"
        "2026 We sing\n\n"
        "Chorus 3\n"
        "001 We praise\n\n"
        "[End 4]\n"
        "42 Amen\n"
    )
    output = convert_lyrics(parsed, None, ConversionSettings())

    assert [section.number for section in parsed.sections] == [2, 3, 4]
    assert [section.lines for section in parsed.sections] == [
        ("2026 We sing",), ("001 We praise",), ("42 Amen",),
    ]
    assert output == (
        "[Title]\n135 Holy Forever\n\n"
        "[Verse 2]\n2026 We sing\n\n"
        "[Chorus 3]\n001 We praise\n\n"
        "[End 4]\n42 Amen\n"
    )


def test_chinese_first_title_does_not_take_number_from_its_translation() -> None:
    parsed = parse_lyrics(
        "001 中文歌名 | 001 Holy Forever\n\n"
        "Verse 1\n祢愛我們\n\nChorus 2\nWe sing\n"
    )
    plan = build_exact_manual_plan(
        parsed,
        {0: [1]},
        {0: (AlignedLine((0,), (TranslationReference(1, (0,)),)),)},
    )
    assert convert_lyrics(parsed, plan, ConversionSettings()).startswith(
        "[Title]\n001 中文歌名 001 Holy Forever\n"
    )


@pytest.mark.parametrize("extension", sorted(TEXT_EXTENSIONS))
def test_supported_text_uploads_and_paste_keep_identical_numbered_title(
    extension: str,
) -> None:
    source = "https://example.invalid/song\n\n001 Groot is Uw trouw\n\nVerse 1\nUw trouw is groot\n"
    pasted = parse_lyrics(source)
    uploaded = parse_lyrics(extract_text(f"song{extension}", source.encode("utf-8")))

    assert uploaded == pasted
    assert pasted.raw_title == "001 Groot is Uw trouw"
    assert convert_lyrics(uploaded, None, ConversionSettings()).startswith(
        "[Title]\n001 Groot is Uw trouw\n"
    )


def test_source_without_a_title_does_not_generate_a_number_or_title() -> None:
    parsed = parse_lyrics("Verse 42\nWe sing\n")
    assert parsed.raw_title == ""
    assert convert_lyrics(parsed, None, ConversionSettings()) == "[Verse 42]\nWe sing\n"
