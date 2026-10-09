from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from lyrics_dashboard.alignment import (
    build_exact_manual_plan,
    suggest_manual_line_groups,
    suggest_manual_selections,
)
from lyrics_dashboard.converter import ConversionSettings, convert_lyrics
from lyrics_dashboard.errors import PairingError
from lyrics_dashboard.parser import parse_lyrics


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
CHINESE_SOURCE = (
    "135 人算什麼\n\n"
    "Verse\n"
    "耶和華我們的主 你的名在地何美\n"
    "你將榮耀彰顯於天\n\n"
    "Chorus\n"
    "便說 人算什麼 你竟顧念他\n"
    "他比天使微小一點\n\n"
    "End\n"
    "我要永遠愛祢\n"
    "我願一生跟隨祢\n"
)


@pytest.mark.parametrize(
    ("heading", "label", "number"),
    [
        ("End", "End", None),
        ("END", "End", None),
        ("end", "End", None),
        ("[End]", "End", None),
        ("End 1", "End 1", 1),
        ("End 2", "End 2", 2),
        ("[End 1]", "End 1", 1),
    ],
)
def test_end_variants_are_headings_and_keep_their_own_lyrics(
    heading: str, label: str, number: int | None
) -> None:
    parsed = parse_lyrics(
        f"Verse\n中文主歌\n{heading}\n我要永遠愛祢\n我願一生跟隨祢\n"
        "Bridge\n中文橋段\n"
    )

    assert [section.label for section in parsed.sections] == ["Verse", label, "Bridge"]
    end_section = parsed.sections[1]
    assert end_section.kind == "Outro"
    assert end_section.number == number
    assert end_section.language == "zh"
    assert end_section.lines == ("我要永遠愛祢", "我願一生跟隨祢")
    assert parsed.sections[2].lines == ("中文橋段",)

    output = convert_lyrics(parsed, None, ConversionSettings())
    assert f"[{label}]\n我要永遠愛祢\n我願一生跟隨祢\n" in output
    assert output.count(f"[{label}]") == 1


def test_chinese_only_song_with_end_converts_without_false_language_error() -> None:
    parsed = parse_lyrics(CHINESE_SOURCE)
    output = convert_lyrics(parsed, None, ConversionSettings())

    assert parsed.mode == "single-language"
    assert parsed.single_language == "zh"
    assert [section.language for section in parsed.sections] == ["zh", "zh", "zh"]
    assert [section.label for section in parsed.sections] == ["Verse", "Chorus", "End"]
    assert parsed.sections[2].lines == ("我要永遠愛祢", "我願一生跟隨祢")
    assert not parsed.warnings
    assert output.startswith("[Title]\n")
    assert output.index("[Verse]") < output.index("[Chorus]") < output.index("[End]")
    assert output.endswith("[End]\n我要永遠愛祢\n我願一生跟隨祢\n")
    assert "|" not in output


def test_end_inside_an_ordinary_lyric_sentence_is_not_a_heading() -> None:
    parsed = parse_lyrics(
        "Song title\n\nVerse 1\nWe sing until the end of the night\n"
        "The end of the road is near\nOutro\nThe final line\n"
    )

    assert [section.label for section in parsed.sections] == ["Verse 1", "Outro"]
    assert parsed.sections[0].lines == (
        "We sing until the end of the night",
        "The end of the road is near",
    )
    assert parsed.sections[1].lines == ("The final line",)


@pytest.mark.parametrize(
    ("heading", "kind", "number"),
    [
        ("Verse", "Verse", None),
        ("Chorus", "Chorus", None),
        ("Bridge", "Bridge", None),
        ("Outro", "Outro", None),
        ("Verse 2", "Verse", 2),
        ("Chorus 2", "Chorus", 2),
        ("Bridge 2", "Bridge", 2),
        ("Outro 2", "Outro", 2),
    ],
)
def test_existing_section_headings_keep_their_labels_and_kinds(
    heading: str, kind: str, number: int | None
) -> None:
    parsed = parse_lyrics(f"{heading}\nA lyric line\n")

    assert len(parsed.sections) == 1
    assert parsed.sections[0].label == heading
    assert parsed.sections[0].kind == kind
    assert parsed.sections[0].number == number
    assert parsed.sections[0].lines == ("A lyric line",)


def test_bilingual_end_matches_outro_by_kind_and_converts() -> None:
    parsed = parse_lyrics(
        "中文歌名 | Dutch title\n\n"
        "Verse 1\n中文主歌\n"
        "End\n中文尾聲\n"
        "Verse 2\nDutch verse line\n"
        "Outro\nDutch ending line\n"
    )
    suggestions = suggest_manual_selections(parsed)
    groups = {
        source_index: suggest_manual_line_groups(parsed, source_index, selected)
        for source_index, selected in suggestions.items()
    }
    plan = build_exact_manual_plan(parsed, suggestions, groups)
    output = convert_lyrics(parsed, plan, ConversionSettings(switch_index=2))

    assert parsed.mode == "bilingual"
    assert [section.language for section in parsed.sections] == ["zh", "zh", "nl", "nl"]
    assert parsed.sections[1].label == "End"
    assert parsed.sections[1].kind == parsed.sections[3].kind == "Outro"
    assert suggestions == {0: [2], 1: [3]}
    assert "[End]\n中文尾聲|Dutch ending line\n" in output
    assert "[Outro]\nDutch ending line|中文尾聲\n" in output


def test_actual_mixed_language_lyric_under_end_is_still_rejected() -> None:
    source = "Verse\n中文歌詞\nEnd\n我要永遠愛祢\nbad\n"

    with pytest.raises(PairingError, match=r"\[End\] line 2 contains evidence of a second"):
        parse_lyrics(source)


@pytest.mark.parametrize("input_kind", ["Paste text", "Upload file"])
def test_chinese_end_song_works_through_both_dashboard_inputs(input_kind: str) -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    if input_kind == "Paste text":
        next(radio for radio in app.radio if radio.label == "Input source").set_value(
            "Paste text"
        ).run(timeout=20)
        next(area for area in app.text_area if area.label == "Paste song text").set_value(
            CHINESE_SOURCE
        )
        next(button for button in app.button if button.label == "PROCESS TEXT").click().run(
            timeout=20
        )
    else:
        app.file_uploader[0].set_value(
            ("chinese-end.txt", CHINESE_SOURCE.encode("utf-8"), "text/plain")
        ).run(timeout=20)

    assert not app.exception
    assert not app.error
    assert app.session_state["automatic_section_languages"] == {
        0: "zh",
        1: "zh",
        2: "zh",
    }
    output = app.session_state["generated_output"]
    assert "[End]\n我要永遠愛祢\n我願一生跟隨祢\n" in output
    assert app.session_state["saved_final_output"] == output
    assert next(
        area for area in app.text_area if area.label != "Paste song text"
    ).value == output
