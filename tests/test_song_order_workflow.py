from __future__ import annotations

import io
from pathlib import Path

import pytest
import streamlit as st
from docx import Document
from streamlit.testing.v1 import AppTest

from lyrics_dashboard.alignment import (
    build_exact_manual_plan,
    suggest_manual_line_groups,
    suggest_manual_selections,
)
from lyrics_dashboard.converter import (
    ConversionSettings,
    convert_lyrics,
    derive_output_filename,
)
from lyrics_dashboard.extractors import extract_text
from lyrics_dashboard.parser import parse_lyrics


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
CHINESE_SOURCE = (
    "135 人算什麼\n\n"
    "Order\n"
    "v c v c o\n\n"
    "Verse\n"
    "耶和華我們的主 你的名在地何美\n"
    "你將榮耀彰顯於天\n\n"
    "Chorus\n"
    "便說：人算什麼？你竟顧念他\n"
    "他比天使微小一點\n\n"
    "outro\n"
    "環顧宇宙萬物何廣闊 試問人算什麼？\n"
)
BILINGUAL_SOURCE = (
    "[Title]\n恩典 Genade\n"
    "[Order] v1 c1 v2 c2 c1\n\n"
    "Verse 1\n祢賜我平安\n\n"
    "Chorus 1\n我讚美祢\n\n"
    "Verse 2\nU geeft mij vrede\n\n"
    "Chorus 2\nIk prijs U\n"
)


def _docx_bytes(source: str) -> bytes:
    document = Document()
    for line in source.splitlines():
        document.add_paragraph(line)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


@pytest.fixture
def download_calls(monkeypatch):
    calls: list[dict[str, object]] = []
    original = st.download_button

    def record_download(*args, **kwargs):
        calls.append(dict(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(st, "download_button", record_download)
    return calls


def _button(app: AppTest, label: str):
    return next(button for button in app.button if button.label == label)


def _paste_source(app: AppTest, source: str) -> None:
    next(radio for radio in app.radio if radio.label == "Input source").set_value(
        "Paste text"
    ).run(timeout=20)
    next(area for area in app.text_area if area.label == "Paste song text").set_value(
        source
    )
    _button(app, "PROCESS TEXT").click().run(timeout=20)


@pytest.mark.parametrize("source_kind", ["paste", "txt", "docx"])
def test_chinese_song_order_reaches_preview_and_download_from_every_input(
    download_calls, source_kind: str
) -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    if source_kind == "paste":
        _paste_source(app, CHINESE_SOURCE)
    else:
        data = (
            _docx_bytes(CHINESE_SOURCE)
            if source_kind == "docx"
            else CHINESE_SOURCE.encode("utf-8")
        )
        mime = (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            if source_kind == "docx"
            else "text/plain"
        )
        app.file_uploader[0].set_value(
            (f"uploaded-song.{source_kind}", data, mime)
        ).run(timeout=20)

    assert not app.exception
    assert not app.error
    assert app.session_state["automatic_section_languages"] == {
        0: "zh",
        1: "zh",
        2: "zh",
    }
    output = app.session_state["generated_output"]
    assert output.startswith(
        "[Title]\n135 人算什麼\n[Order]\nV C V C O\n\n[Verse]\n"
    )
    assert output.count("[Order]") == 1
    assert output.count("[Verse]") == 1
    assert output.count("[Chorus]") == 1
    assert output.count("[Outro]") == 1
    assert "|" not in output
    assert next(
        area for area in app.text_area if area.label == "Converted lyrics"
    ).value == output
    assert app.session_state["saved_final_output"] == output
    assert download_calls[-1]["data"] == output.encode("utf-8")
    assert download_calls[-1]["file_name"] == "135 人算什麼.txt"


def test_docx_order_and_lyrics_match_the_pasted_source() -> None:
    pasted = parse_lyrics(CHINESE_SOURCE)
    uploaded = parse_lyrics(extract_text("song.docx", _docx_bytes(CHINESE_SOURCE)))

    assert uploaded == pasted
    assert uploaded.song_order == ("V", "C", "V", "C", "O")
    assert uploaded.raw_title == "135 人算什麼"


@pytest.mark.parametrize("include_title", [True, False])
def test_bilingual_conversion_preserves_order_and_original_section_sequence(
    include_title: bool,
) -> None:
    parsed = parse_lyrics(BILINGUAL_SOURCE)
    selections = suggest_manual_selections(parsed)
    groups = {
        source_index: suggest_manual_line_groups(parsed, source_index, selected)
        for source_index, selected in selections.items()
    }
    plan = build_exact_manual_plan(parsed, selections, groups)
    output = convert_lyrics(
        parsed,
        plan,
        ConversionSettings(include_title=include_title, switch_index=2),
    )

    expected_header = "[Title]\n恩典 Genade\n" if include_title else ""
    assert output.startswith(expected_header + "[Order]\nV1 C1 V2 C2 C1\n\n")
    assert output.index("[Verse 1]") < output.index("[Chorus 1]")
    assert output.index("[Chorus 1]") < output.index("[Verse 2]")
    assert output.index("[Verse 2]") < output.index("[Chorus 2]")
    assert output.count("[Chorus 1]") == 1
    assert "祢賜我平安|U geeft mij vrede" in output
    assert "U geeft mij vrede|祢賜我平安" in output


@pytest.mark.parametrize("with_title", [True, False])
def test_single_language_order_survives_without_an_exported_title(with_title: bool) -> None:
    source = CHINESE_SOURCE if with_title else CHINESE_SOURCE.split("\n\n", 1)[1]
    parsed = parse_lyrics(source)
    output = convert_lyrics(parsed, None, ConversionSettings(include_title=False))

    assert output.startswith("[Order]\nV C V C O\n\n[Verse]\n")
    assert "[Title]" not in output


def test_bilingual_dashboard_keeps_order_through_validation_and_saved_edits(
    download_calls,
) -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    _paste_source(app, BILINGUAL_SOURCE)
    assert not app.exception
    assert not app.error
    assert len(app.session_state["automatic_section_languages"]) == 4

    _button(app, "Validate these manual matches").click().run(timeout=20)
    assert not app.exception
    assert not app.error
    generated = app.session_state["generated_output"]
    assert generated.startswith("[Title]\n恩典 Genade\n[Order]\nV1 C1 V2 C2 C1\n\n")
    assert download_calls[-1]["file_name"] == "恩典 Genade.txt"
    assert download_calls[-1]["data"] == generated.encode("utf-8")

    edited = generated.replace("[Order]\nV1 C1 V2 C2 C1", "[Order]\nV1 C1 C1 V2 C2")
    next(area for area in app.text_area if area.label == "Converted lyrics").set_value(
        edited
    ).run(timeout=20)
    assert download_calls[-1]["data"] == generated.encode("utf-8")
    _button(app, "UPDATE").click().run(timeout=20)

    assert not app.exception
    assert not app.error
    assert app.session_state["generated_output"] == generated
    assert app.session_state["saved_final_output"] == edited
    assert download_calls[-1]["data"] == edited.encode("utf-8")
    assert download_calls[-1]["file_name"] == "恩典 Genade.txt"


@pytest.mark.parametrize("order_line", ["[Order] V C V C O", "[order]: v c v c o"])
def test_inline_order_metadata_never_becomes_part_of_download_filename(order_line: str) -> None:
    assert derive_output_filename(
        f"[Title]\n135 人算什麼\n{order_line}\n\n[Verse]\n歌詞\n",
        source_title="Original title",
        source_name="upload.docx",
    ) == "135 人算什麼.txt"
