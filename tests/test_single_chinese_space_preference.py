from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from lyrics_dashboard.alignment import build_exact_manual_plan
from lyrics_dashboard.converter import (
    ConversionSettings,
    convert_lyrics,
    encode_utf8_txt,
)
from lyrics_dashboard.models import (
    AlignedLine,
    ParsedLyrics,
    Section,
    TranslationReference,
)
from lyrics_dashboard.splitter import split_lyric, split_lyric_result
from lyrics_dashboard.text_processing import (
    clean_content_result,
    clean_content_text,
)


PREFERRED_SOURCE = "天地玄黃 祢是永遠君王萬歲"
PREFERRED_RESULT = "天地玄黃//祢是永遠君王萬歲"
APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


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
                        TranslationReference(
                            section_index=1,
                            line_indices=(0,),
                        ),
                    ),
                ),
            )
        },
    )
    return parsed, plan


def _lyric_rows(output: str) -> list[str]:
    return [line for line in output.splitlines() if "|" in line]


def _first_side_result(
    source: str,
    max_length: int,
) -> str:
    return split_lyric_result(
        source,
        "zh",
        max_length,
        minimum_fragment_length=2,
        minimum_fragment_ratio=0.25,
    ).text


def test_long_chinese_line_prefers_its_only_internal_space() -> None:
    requested_example = "祢名超乎萬名之上 祢是永遠君王"

    assert _first_side_result(requested_example, 10) == (
        "祢名超乎萬名之上//祢是永遠君王"
    )

    result = _first_side_result(PREFERRED_SOURCE, 8)
    cleaned = clean_content_text(PREFERRED_SOURCE, "zh")

    assert result == PREFERRED_RESULT
    assert " //" not in result
    assert "// " not in result
    assert result.count("//") == 1
    assert result.replace("//", "") == cleaned.replace(" ", "")


@pytest.mark.parametrize(
    ("source", "max_length"),
    (
        ("短歌 短句", 5),
        ("短歌 短句", 4),
        ("天地玄黃 宇宙洪荒", 8),
    ),
)
def test_only_space_never_makes_an_in_limit_chinese_line_split(
    source: str,
    max_length: int,
) -> None:
    result = split_lyric(source, "zh", max_length)

    assert result == clean_content_text(source, "zh")
    assert "//" not in result


def test_trimmed_and_repeated_whitespace_is_one_normalized_boundary() -> None:
    source = "\t  天地玄黃 \t\u3000 祢是永遠君王萬歲  "
    cleaning = clean_content_result(source, "zh")

    assert cleaning.text == PREFERRED_SOURCE
    assert cleaning.original_whitespace_boundaries == (5,)
    assert _first_side_result(source, 8) == PREFERRED_RESULT


def test_zero_and_multiple_spaces_keep_ordinary_chinese_behavior() -> None:
    no_space = "祢名超乎萬名之上祢是永遠君王"
    multiple_spaces = "第一部分 第二部分 第三部分"

    assert split_lyric(no_space, "zh", 10) == (
        "祢名超乎萬名//之上祢是永遠君王"
    )
    assert split_lyric(multiple_spaces, "zh", 6) == (
        "第一部分 第二//部分 第三部分"
    )


def test_single_space_must_pass_fragment_and_maximum_length_ranking() -> None:
    minimum_rejected = split_lyric_result(
        "甲 乙丙丁戊己庚辛",
        "zh",
        4,
        minimum_fragment_length=2,
    )
    ratio_rejected = _first_side_result(
        "甲乙 丙丁戊己庚辛壬癸子丑",
        4,
    )
    normal_side_rejected = split_lyric_result(
        "甲乙 丙丁戊己庚辛壬癸子丑",
        "zh",
        10,
    ).text
    no_word_alternative = split_lyric_result(
        "甲乙 中华人民共和国",
        "zh",
        4,
    )
    over_limit_rejected = _first_side_result(PREFERRED_SOURCE, 7)

    assert minimum_rejected.text == "甲 乙丙//丁戊己庚辛"
    assert ratio_rejected == "甲乙 丙丁戊己//庚辛壬癸子丑"
    assert normal_side_rejected == "甲乙 丙丁戊己//庚辛壬癸子丑"
    assert no_word_alternative.text == "甲乙 中华//人民共和国"
    assert no_word_alternative.used_character_fallback is True
    assert over_limit_rejected == "天地玄黃 祢是//永遠君王萬歲"
    assert all(
        result != unsafe
        for result, unsafe in (
            (minimum_rejected.text, "甲//乙丙丁戊己庚辛"),
            (ratio_rejected, "甲乙//丙丁戊己庚辛壬癸子丑"),
            (normal_side_rejected, "甲乙//丙丁戊己庚辛壬癸子丑"),
            (no_word_alternative.text, "甲乙//中华人民共和国"),
            (over_limit_rejected, PREFERRED_RESULT),
        )
    )


@pytest.mark.parametrize(
    ("source", "expected", "protected_content"),
    (
        (
            "甲乙我 相信丙丁戊己",
            "甲乙我 相信//丙丁戊己",
            "我 相信",
        ),
        (
            "甲乙在 祢裡面丙丁",
            "甲乙在 祢裡面丙丁",
            "在 祢裡面",
        ),
        (
            "甲乙 丙丁平安戊己庚辛",
            "甲乙 丙丁//平安戊己庚辛",
            "平安",
        ),
        (
            "甲乙 丙丁中华人民共和国戊己",
            "甲乙 丙丁//中华人民共和国戊己",
            "中华人民共和国",
        ),
    ),
)
def test_rejected_space_preserves_existing_chinese_protections(
    source: str,
    expected: str,
    protected_content: str,
) -> None:
    result = _first_side_result(source, 4)

    assert result == expected
    assert protected_content in result
    assert result.replace("//", "").replace(" ", "") == (
        clean_content_text(source, "zh").replace(" ", "")
    )


def test_rejected_space_preserves_two_character_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    splitter = importlib.import_module("lyrics_dashboard.splitter")
    monkeypatch.setattr(splitter, "_chinese_word_boundaries", lambda _text: ())

    result = splitter.split_lyric_result(
        "賜 我永平安直到永遠",
        "zh",
        4,
        minimum_fragment_length=2,
        minimum_fragment_ratio=0.25,
    )

    assert result.text == "賜 我永//平安直到永遠"
    assert "平//安" not in result.text
    assert result.used_character_fallback is True


def test_single_space_preference_works_on_both_pipe_sides_after_switch() -> None:
    dutch = "Nederlandse regel"
    parsed, plan = _single_pair(PREFERRED_SOURCE, dutch)

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
        f"{PREFERRED_RESULT}|{dutch}",
        f"{dutch}|{PREFERRED_RESULT}",
    ]
    assert all(
        side.count("//") <= 1
        for row in rows
        for side in row.split("|")
    )

    unsafe = "甲乙 丙丁戊己庚辛壬癸子丑"
    unsafe_parsed, unsafe_plan = _single_pair(unsafe, dutch)
    unsafe_output = convert_lyrics(
        unsafe_parsed,
        unsafe_plan,
        ConversionSettings(
            switch_index=1,
            chinese_max_length=10,
            dutch_max_length=40,
        ),
    )
    safe_fallback = "甲乙 丙丁戊己//庚辛壬癸子丑"

    assert _lyric_rows(unsafe_output) == [
        f"{safe_fallback}|{dutch}",
        f"{dutch}|{safe_fallback}",
    ]


def test_chinese_only_editable_preview_matches_utf8_txt_export() -> None:
    source = (
        "[Title]\n"
        "歌名\n\n"
        "Verse 1\n"
        f"{PREFERRED_SOURCE}\n"
    )
    expected = (
        "[Title]\n"
        "歌名\n\n"
        "[Verse 1]\n"
        f"{PREFERRED_RESULT}\n"
    )
    app = AppTest.from_file(str(APP_PATH), default_timeout=10).run()
    app.file_uploader[0].set_value(
        ("chinese-space.txt", source.encode("utf-8"), "text/plain")
    ).run(timeout=10)

    assert not app.exception
    assert not app.error
    assert len(app.text_area) == 1
    preview = app.text_area[0].value

    assert preview == expected
    assert "|" not in preview
    assert app.session_state["edited_output"] == preview
    assert encode_utf8_txt(preview).decode("utf-8") == preview

    edited_preview = preview.replace("歌名", "新歌名")
    app.text_area[0].set_value(edited_preview).run(timeout=10)

    assert app.text_area[0].value == edited_preview
    assert app.session_state["edited_output"] == edited_preview
    assert encode_utf8_txt(app.text_area[0].value).decode("utf-8") == (
        edited_preview
    )
    download_buttons = app.get("download_button")
    assert len(download_buttons) == 1
    assert download_buttons[0].label == "Download final TXT"
