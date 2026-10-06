from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from lyrics_dashboard.converter import derive_output_filename


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
SINGLE_SOURCE = "[Title]\nGrace\n\nVerse 1\nWe sing together for the Lord\n"
BILINGUAL_SOURCE = (
    "[Title]\n安靜 Still\n\n"
    "Verse 1\n藏我在翅膀下\n\n"
    "Verse 2\nHide me under Your wings\n"
)


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


def _submit_pasted_source(app: AppTest, source: str) -> None:
    next(area for area in app.text_area if area.label == "Paste song text").set_value(source)
    _button(app, "PROCESS TEXT").click().run(timeout=20)
    assert not app.exception


@pytest.mark.parametrize("input_kind", ["Upload file", "Paste text"])
@pytest.mark.parametrize(
    ("source", "expected_name", "bilingual"),
    [(SINGLE_SOURCE, "Grace.txt", False), (BILINGUAL_SOURCE, "安靜 Still.txt", True)],
    ids=["single-language", "bilingual-screenshot-title"],
)
def test_download_uses_song_title_for_both_input_and_conversion_modes(
    download_calls,
    input_kind: str,
    source: str,
    expected_name: str,
    bilingual: bool,
) -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    if input_kind == "Paste text":
        next(radio for radio in app.radio if radio.label == "Input source").set_value(
            input_kind
        ).run(timeout=20)
        _submit_pasted_source(app, source)
    else:
        app.file_uploader[0].set_value(
            ("unrelated upload name.txt", source.encode("utf-8"), "text/plain")
        ).run(timeout=20)
    if bilingual:
        _button(app, "Validate these manual matches").click().run(timeout=20)

    assert not app.exception
    assert not app.error
    assert download_calls[-1]["file_name"] == expected_name
    assert download_calls[-1]["data"] == app.session_state["saved_final_output"].encode("utf-8")


def test_filename_follows_saved_title_and_resets_for_new_pasted_song(download_calls) -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    next(radio for radio in app.radio if radio.label == "Input source").set_value(
        "Paste text"
    ).run(timeout=20)
    _submit_pasted_source(app, SINGLE_SOURCE)
    saved_output = app.session_state["saved_final_output"]
    changed_output = saved_output.replace("Grace", "Changed title", 1)

    next(area for area in app.text_area if area.label == "Converted lyrics").set_value(
        changed_output
    ).run(timeout=20)
    assert not app.exception
    assert download_calls[-1]["file_name"] == "Grace.txt"
    assert download_calls[-1]["data"] == saved_output.encode("utf-8")

    _button(app, "UPDATE").click().run(timeout=20)
    assert not app.exception
    assert download_calls[-1]["file_name"] == "Changed title.txt"
    assert download_calls[-1]["data"] == changed_output.encode("utf-8")

    _submit_pasted_source(app, SINGLE_SOURCE.replace("Grace", "Another song", 1))
    assert download_calls[-1]["file_name"] == "Another song.txt"
    assert b"Changed title" not in download_calls[-1]["data"]


@pytest.mark.parametrize(
    ("saved_output", "source_title", "source_name", "expected_name"),
    [
        ("[Title]\n安靜 Still\n\n[Verse 1]\nLyrics\n", "Original", "upload.docx", "安靜 Still.txt"),
        ("\ufeff  [tItLe]  \r\n安靜\r\nStill\r\n[Verse 1]\r\nLyrics", "", "", "安靜 Still.txt"),
        ("[Title]\n安靜|Still: Hope / Peace?*\n", "", "", "安靜 Still Hope Peace.txt"),
        ("[Title]\nI'll sing!\n", "", "", "I'll sing!.txt"),
        ("[Title]\n  Grace.  \n", "", "", "Grace.txt"),
        ("[Title]\n\n[Verse 1]\nLyrics", "Original song", "upload.docx", "Original song.txt"),
        ("[Verse 1]\nLyrics", "Original song", "upload.docx", "Original song.txt"),
        ("[Title]\n<>:/\\|?*\n", "Original song", "upload.docx", "Original song.txt"),
        ("[Verse 1]\nLyrics", "", "original document.docx", "original document.txt"),
        ("", "", "", "converted_lyrics.txt"),
        ("[Title]\nCON\n", "", "", "_CON.txt"),
        ("[Title]\nLpt1.song\n", "", "", "_Lpt1.song.txt"),
        ("[Title]\nPeace\x00\x1f\x7fHope\n", "", "", "Peace Hope.txt"),
    ],
)
def test_filename_preserves_title_and_has_safe_fallbacks(
    saved_output: str,
    source_title: str,
    source_name: str,
    expected_name: str,
) -> None:
    assert derive_output_filename(
        saved_output,
        source_title=source_title,
        source_name=source_name,
    ) == expected_name
