from __future__ import annotations

import math

from lyrics_dashboard.alignment import build_exact_manual_plan
from lyrics_dashboard.converter import (
    FIRST_SIDE_LIMIT_RATIO,
    ConversionSettings,
    convert_lyrics,
    derive_first_side_limit,
)
from lyrics_dashboard.models import (
    AlignedLine,
    ParsedLyrics,
    Section,
    TranslationReference,
)
from lyrics_dashboard.text_processing import clean_content_text


def _single_pair(chinese: str, dutch: str) -> tuple[ParsedLyrics, object]:
    parsed = ParsedLyrics(
        raw_title="Chinese title Nederlandse titel",
        chinese_title="Chinese title",
        dutch_title="Nederlandse titel",
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


def _lyric_rows(output: str) -> list[str]:
    return [line for line in output.splitlines() if "|" in line]


def test_first_side_limit_is_named_eighty_percent_with_safe_minimum() -> None:
    assert FIRST_SIDE_LIMIT_RATIO == 0.80
    assert derive_first_side_limit(50) == 40
    assert derive_first_side_limit(20) == 16
    assert derive_first_side_limit(10) == 8
    assert derive_first_side_limit(40) == 32
    assert derive_first_side_limit(4) == 4


def test_same_text_uses_stricter_limit_only_before_pipe_after_switch() -> None:
    chinese = "\u4eca\u5929\u6211\u4eec\u4e00\u8d77\u8d5e\u7f8e\u4e0a\u5e1d"
    dutch = "een middelgrote Nederlandse liedregel"
    parsed, plan = _single_pair(chinese, dutch)
    warnings: list[str] = []

    output = convert_lyrics(
        parsed,
        plan,
        ConversionSettings(
            switch_index=1,
            chinese_max_length=10,
            dutch_max_length=40,
        ),
        warnings=warnings,
    )
    rows = _lyric_rows(output)

    assert rows == [
        "\u4eca\u5929\u6211\u4eec//\u4e00\u8d77\u8d5e\u7f8e\u4e0a\u5e1d|"
        "een middelgrote Nederlandse liedregel",
        "een middelgrote//Nederlandse liedregel|"
        "\u4eca\u5929\u6211\u4eec\u4e00\u8d77\u8d5e\u7f8e\u4e0a\u5e1d",
    ]
    assert warnings == []
    assert all(side.count("//") <= 1 for row in rows for side in row.split("|"))

    chinese_first, dutch_second = rows[0].split("|")
    dutch_first, chinese_second = rows[1].split("|")
    assert chinese_first.replace("//", "") == clean_content_text(chinese, "zh")
    assert chinese_second == clean_content_text(chinese, "zh")
    assert dutch_first.replace("//", " ") == clean_content_text(dutch, "nl")
    assert dutch_second == clean_content_text(dutch, "nl")

    chinese_parts = chinese_first.split("//")
    dutch_parts = dutch_first.split("//")
    assert min(map(len, chinese_parts)) >= math.ceil(len(chinese) * 0.25)
    assert min(map(len, dutch_parts)) >= math.ceil(len(dutch) * 0.25)
    assert "\u6211//\u4eec" not in chinese_first
    assert "\u4e00//\u8d77" not in chinese_first
    assert all(part in dutch.split() for part in dutch_first.replace("//", " ").split())


def test_short_first_side_and_unbalanced_dutch_boundary_remain_unsplit() -> None:
    short_parsed, short_plan = _single_pair("\u77ed\u6b4c", "kort lied")
    short_output = convert_lyrics(
        short_parsed,
        short_plan,
        ConversionSettings(
            switch_index=1,
            chinese_max_length=10,
            dutch_max_length=40,
        ),
    )
    assert _lyric_rows(short_output) == [
        "\u77ed\u6b4c|kort lied",
        "kort lied|\u77ed\u6b4c",
    ]

    dutch = "a supercalifragilisticexpialidocious"
    parsed, plan = _single_pair("\u77ed\u6b4c", dutch)
    output = convert_lyrics(
        parsed,
        plan,
        ConversionSettings(
            switch_index=1,
            chinese_max_length=10,
            dutch_max_length=40,
        ),
    )
    rows = _lyric_rows(output)

    assert rows == [
        f"\u77ed\u6b4c|{dutch}",
        f"{dutch}|\u77ed\u6b4c",
    ]
    assert "//" not in rows[1]
