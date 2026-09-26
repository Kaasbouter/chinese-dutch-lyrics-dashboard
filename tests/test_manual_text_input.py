from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from lyrics_dashboard.converter import encode_utf8_txt
from lyrics_dashboard.extractors import SUPPORTED_EXTENSIONS
from lyrics_dashboard.parser import parse_lyrics


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
CHINESE = "Chinese"
LATIN = "Dutch/English or Latin script"

ENGLISH_SOURCE = (
    "[Title]\n"
    "Grace\n\n"
    "Verse 1\n"
    "We sing together for the Lord\n\n"
    "Chorus 1\n"
    "A short song\n"
)
BILINGUAL_SOURCE = (
    "[Title]\n"
    "中文歌名 | Dutch title\n\n"
    "Verse 1\n"
    "中文甲一\n"
    "中文甲二\n\n"
    "Chorus 1\n"
    "中文乙一\n\n"
    "Verse 2\n"
    "Dutch alpha one\n"
    "Dutch alpha two\n\n"
    "Chorus 2\n"
    "Dutch beta one\n"
)
NEW_BILINGUAL_SOURCE = (
    "[Title]\n"
    "新歌名 | New title\n\n"
    "Bridge 1\n"
    "新的中文歌詞\n\n"
    "Bridge 2\n"
    "A new Dutch line\n"
)


def _input_choice(app: AppTest):
    choices = [radio for radio in app.radio if radio.label == "Input source"]
    assert len(choices) == 1
    return choices[0]


def _source_area(app: AppTest):
    areas = [area for area in app.text_area if area.label == "Paste song text"]
    assert len(areas) == 1
    return areas[0]


def _final_area(app: AppTest):
    areas = [area for area in app.text_area if area.label != "Paste song text"]
    assert len(areas) == 1
    return areas[0]


def _button(app: AppTest, label: str):
    buttons = [button for button in app.button if button.label == label]
    assert len(buttons) == 1
    return buttons[0]


def _start_paste_mode() -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    _input_choice(app).set_value("Paste text").run(timeout=20)
    assert not app.exception
    return app


def _process(app: AppTest, source: str) -> AppTest:
    _source_area(app).set_value(source)
    _button(app, "PROCESS TEXT").click().run(timeout=20)
    assert not app.exception
    return app


def _language_delta(edits: dict[int, str]) -> dict[str, object]:
    return {
        "edited_rows": {
            row: {"Language": language} for row, language in edits.items()
        },
        "added_rows": [],
        "deleted_rows": [],
    }


def test_upload_is_default_and_existing_file_input_remains_available() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()

    assert not app.exception
    assert _input_choice(app).options == ["Upload file", "Paste text"]
    assert _input_choice(app).value == "Upload file"
    assert len(app.file_uploader) == 1
    assert not [area for area in app.text_area if area.label == "Paste song text"]
    assert not [button for button in app.button if button.label == "PROCESS TEXT"]
    assert any(message.value == "Upload a file to begin." for message in app.info)

    # Keep the existing upload control's supported extensions and conversion path.
    assert set(app.file_uploader[0].proto.type) == SUPPORTED_EXTENSIONS
    app.file_uploader[0].set_value(
        ("english-only.txt", ENGLISH_SOURCE.encode("utf-8"), "text/plain")
    ).run(timeout=20)
    assert not app.exception
    assert _final_area(app).value == app.session_state["generated_output"]
    assert "[Title]\nGrace" in _final_area(app).value
    assert "|" not in _final_area(app).value


def test_paste_mode_exposes_only_source_text_and_process_action_before_submission() -> None:
    app = _start_paste_mode()

    assert _input_choice(app).value == "Paste text"
    assert not app.file_uploader
    assert _source_area(app).value == ""
    form_children = [child.type for child in app.get("form")[0].children.values()]
    assert sorted(form_children) == ["button", "caption", "text_area"]
    assert form_children.index("text_area") < form_children.index("button")
    assert any(
        "Paste the song in the same basic format" in caption.value
        for caption in app.caption
    )
    assert len([button for button in app.button if button.label == "PROCESS TEXT"]) == 1
    assert not [button for button in app.button if button.label == "UPDATE"]
    assert not app.get("download_button")


@pytest.mark.parametrize(
    ("source", "expected_language", "expected_labels", "expected_title"),
    [
        (
            "1 中文歌名\n\nVerse 1\n這是中文歌詞\n\nChorus 1\n賜我平安\n",
            "zh",
            ["[Verse 1]", "[Chorus 1]"],
            "中文歌名",
        ),
        (
            "1 Genade\n\nVerse 1\nWij zingen samen voor de Heer\n",
            "nl",
            ["[Verse 1]"],
            "Genade",
        ),
        (ENGLISH_SOURCE, "nl", ["[Verse 1]", "[Chorus 1]"], "Grace"),
    ],
    ids=["chinese-only", "dutch-only", "english-only"],
)
def test_pasted_single_language_sources_use_existing_parser_and_converter(
    source: str,
    expected_language: str,
    expected_labels: list[str],
    expected_title: str,
) -> None:
    app = _process(_start_paste_mode(), source)
    parsed = parse_lyrics(source)

    assert not app.error
    assert parsed.mode == "single-language"
    assert parsed.single_language == expected_language
    assert [f"[{section.label}]" for section in parsed.sections] == expected_labels
    assert app.session_state["automatic_section_languages"] == {
        section.original_index: expected_language for section in parsed.sections
    }
    assert _final_area(app).value.startswith(f"[Title]\n{expected_title}\n")
    assert all(label in _final_area(app).value for label in expected_labels)
    assert "|" not in _final_area(app).value
    assert not app.multiselect
    assert not [button for button in app.button if button.label == "REFRESH"]
    assert not [
        item for item in app.selectbox if item.label == "Switch to Dutch first starting at"
    ]
    assert app.session_state["saved_final_output"] == _final_area(app).value


def test_pasted_bilingual_source_reaches_language_and_manual_matching_controls() -> None:
    app = _process(_start_paste_mode(), BILINGUAL_SOURCE)

    assert not app.error
    assert app.session_state["automatic_section_languages"] == {
        0: "zh",
        1: "zh",
        2: "nl",
        3: "nl",
    }
    editor = next(
        dataframe
        for dataframe in app.dataframe
        if dataframe.proto.id.endswith("-source_language_editor")
    )
    assert editor.value["Section"].tolist() == [
        "[Verse 1]",
        "[Chorus 1]",
        "[Verse 2]",
        "[Chorus 2]",
    ]
    assert editor.value["Language"].tolist() == [CHINESE, CHINESE, LATIN, LATIN]
    assert [item.key for item in app.multiselect] == ["manual_match_0", "manual_match_1"]
    assert app.session_state["manual_match_0"] == [2]
    assert app.session_state["manual_drag_layout_0"][0] == ((0,), (1,))
    cards = json.loads(app.get("component_instance")[0].proto.json_args)["items"]
    assert len(cards) == 3  # Card pool and two editable Chinese mapping rows.
    assert sum(len(box["items"]) for box in cards) == 2

    _button(app, "Validate these manual matches").click().run(timeout=20)
    assert not app.exception
    assert not app.error
    assert app.session_state["alignment_plan_matches_current_inputs"] is True
    assert "中文甲一|Dutch alpha one" in _final_area(app).value
    assert [item.label for item in app.selectbox if item.label == "Switch to Dutch first starting at"]
    assert app.get("download_button")[0].label == "Download final TXT"


def test_manual_section_and_line_matching_work_for_pasted_bilingual_text() -> None:
    app = _process(_start_paste_mode(), BILINGUAL_SOURCE)

    # A Chinese section can select more than one Dutch counterpart.
    next(
        item for item in app.multiselect if item.key == "manual_match_0"
    ).set_value([2, 3]).run(timeout=20)
    assert app.session_state["manual_match_0"] == [2, 3]
    assert {reference[0] for group in app.session_state["manual_drag_layout_0"][1] for reference in group} == {2, 3}
    _button(app, "Validate these manual matches").click().run(timeout=20)
    assert not app.error
    assert app.session_state["alignment_plan_matches_current_inputs"] is True
    assert app.session_state["alignment_plan"].alignments[0].counterpart_section_indices == (
        2,
        3,
    )

    # On a fresh song, merge two Chinese lines and assign both Dutch cards to
    # that one mapping row through the existing editable line-range control.
    app = _process(_start_paste_mode(), BILINGUAL_SOURCE)
    app.session_state["manual_lines_0"] = {
        "edited_rows": {0: {"Chinese line(s)": "1-2"}},
        "added_rows": [],
        "deleted_rows": [1],
    }
    _button(app, "Validate these manual matches").click().run(timeout=20)
    assert not app.error
    assert app.session_state["manual_drag_layout_0"] == (
        ((0, 1),),
        (((2, 0), (2, 1)),),
        (),
    )
    line_group = app.session_state["alignment_plan"].alignments[0].aligned_lines[0]
    assert line_group.source_line_indices == (0, 1)
    assert line_group.translation_references[0].line_indices == (0, 1)


def test_pasted_leading_links_use_existing_title_detection() -> None:
    source = (
        "https://example.invalid/forbidden-marker\n\n"
        "1 Song title\n\n"
        "Verse 1\n"
        "Lyrics stay in place\n"
    )
    app = _process(_start_paste_mode(), source)

    assert not app.error
    assert _final_area(app).value.startswith("[Title]\nSong title\n")
    assert "forbidden-marker" not in _final_area(app).value
    assert "Lyrics stay in place" in _final_area(app).value


def test_language_correction_and_refresh_use_existing_state_for_paste() -> None:
    app = _process(_start_paste_mode(), BILINGUAL_SOURCE)
    original = dict(app.session_state["effective_section_languages"])

    app.session_state["source_language_editor"] = _language_delta({2: CHINESE})
    app.run(timeout=20)
    assert app.session_state["pending_section_languages"][2] == "zh"
    assert app.session_state["effective_section_languages"] == original

    app.session_state["source_language_editor"] = _language_delta({2: CHINESE})
    _button(app, "REFRESH").click().run(timeout=20)
    assert not app.exception
    assert app.session_state["effective_section_languages"][2] == "zh"
    assert {item.key for item in app.multiselect} == {
        "manual_match_0",
        "manual_match_1",
        "manual_match_2",
    }


def test_source_draft_and_reruns_do_not_replace_processed_song() -> None:
    app = _process(_start_paste_mode(), ENGLISH_SOURCE)
    fingerprint = app.session_state["file_fingerprint"]
    generated = app.session_state["generated_output"]
    saved = app.session_state["saved_final_output"]
    replacement = ENGLISH_SOURCE.replace("Grace", "Changed title")

    _source_area(app).set_value(replacement).run(timeout=20)
    assert not app.exception
    assert app.session_state["file_fingerprint"] == fingerprint
    assert app.session_state["processed_source_text"] == ENGLISH_SOURCE
    assert app.session_state["generated_output"] == generated
    assert app.session_state["saved_final_output"] == saved
    assert _source_area(app).value == replacement
    assert _final_area(app).value == generated

    app.run(timeout=20)
    assert app.session_state["file_fingerprint"] == fingerprint
    assert app.session_state["processed_source_text"] == ENGLISH_SOURCE
    assert _final_area(app).value == generated


def test_changed_submission_gets_new_identity_and_resets_source_specific_state() -> None:
    app = _process(_start_paste_mode(), BILINGUAL_SOURCE)
    _button(app, "Validate these manual matches").click().run(timeout=20)
    first_fingerprint = app.session_state["file_fingerprint"]
    assert "alignment_plan" in app.session_state.filtered_state

    app.session_state["source_language_editor"] = _language_delta({2: CHINESE})
    _button(app, "REFRESH").click().run(timeout=20)
    assert app.session_state["effective_section_languages"][2] == "zh"
    assert "manual_match_2" in app.session_state.filtered_state

    _process(app, NEW_BILINGUAL_SOURCE)
    assert not app.error
    assert app.session_state["file_fingerprint"] != first_fingerprint
    expected_fingerprint = "paste:" + hashlib.sha256(
        NEW_BILINGUAL_SOURCE.encode("utf-8")
    ).hexdigest()
    assert app.session_state["source_fingerprint"] == expected_fingerprint
    assert app.session_state["processed_source_text"] == NEW_BILINGUAL_SOURCE
    assert app.session_state["automatic_section_languages"] == {0: "zh", 1: "nl"}
    assert app.session_state["pending_section_languages"] == {0: "zh", 1: "nl"}
    assert app.session_state["effective_section_languages"] == {0: "zh", 1: "nl"}
    assert app.session_state["source_language_editor"] == _language_delta({})
    assert "manual_match_2" not in app.session_state.filtered_state
    assert "alignment_plan" not in app.session_state.filtered_state
    assert "generated_output" not in app.session_state.filtered_state


def test_blank_process_attempt_preserves_active_song_and_output() -> None:
    app = _process(_start_paste_mode(), ENGLISH_SOURCE)
    prior = {
        key: app.session_state[key]
        for key in (
            "file_fingerprint",
            "processed_source_text",
            "generated_output",
            "editable_preview",
            "saved_final_output",
        )
    }

    _source_area(app).set_value(" \n\t ")
    _button(app, "PROCESS TEXT").click().run(timeout=20)

    assert not app.exception
    assert any(
        "Paste song text before processing." in message.value for message in app.error
    )
    assert {key: app.session_state[key] for key in prior} == prior
    assert _final_area(app).value == prior["editable_preview"]


def test_paste_source_is_separate_from_update_and_saved_txt(monkeypatch) -> None:
    downloads: list[bytes] = []
    real_download_button = st.download_button

    def record_download(*args, **kwargs):
        downloads.append(kwargs["data"])
        return real_download_button(*args, **kwargs)

    monkeypatch.setattr(st, "download_button", record_download)
    app = _process(_start_paste_mode(), ENGLISH_SOURCE)
    original_source = app.session_state["processed_source_text"]
    original_saved = app.session_state["saved_final_output"]
    manual_output = "  [A manual title!]\nUnstructured final text | kept//exactly  \n"

    _final_area(app).set_value(manual_output).run(timeout=20)
    assert app.session_state["processed_source_text"] == original_source
    assert app.session_state["saved_final_output"] == original_saved
    assert downloads[-1] == encode_utf8_txt(original_saved)

    _button(app, "UPDATE").click().run(timeout=20)
    assert not app.exception
    assert app.session_state["processed_source_text"] == original_source
    assert _source_area(app).value == ENGLISH_SOURCE
    assert _final_area(app).value == manual_output
    assert app.session_state["saved_final_output"] == manual_output
    assert downloads[-1] == encode_utf8_txt(manual_output)


def test_equivalent_uploaded_and_pasted_text_generate_same_result() -> None:
    uploaded = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    uploaded.file_uploader[0].set_value(
        ("equivalent.txt", ENGLISH_SOURCE.encode("utf-8"), "text/plain")
    ).run(timeout=20)
    pasted = _process(_start_paste_mode(), ENGLISH_SOURCE)

    assert not uploaded.exception
    assert not pasted.exception
    assert uploaded.session_state["automatic_section_languages"] == pasted.session_state[
        "automatic_section_languages"
    ]
    assert uploaded.session_state["generated_output"] == pasted.session_state[
        "generated_output"
    ]
    assert _final_area(uploaded).value == _final_area(pasted).value
    assert uploaded.session_state["saved_final_output"] == pasted.session_state[
        "saved_final_output"
    ]


def test_equivalent_bilingual_upload_and_paste_generate_same_manual_conversion() -> None:
    uploaded = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    uploaded.file_uploader[0].set_value(
        ("equivalent-bilingual.txt", BILINGUAL_SOURCE.encode("utf-8"), "text/plain")
    ).run(timeout=20)
    pasted = _process(_start_paste_mode(), BILINGUAL_SOURCE)

    for app in (uploaded, pasted):
        _button(app, "Validate these manual matches").click().run(timeout=20)
        assert not app.exception
        assert not app.error

    assert uploaded.session_state["automatic_section_languages"] == pasted.session_state[
        "automatic_section_languages"
    ]
    assert uploaded.session_state["alignment_plan"] == pasted.session_state[
        "alignment_plan"
    ]
    assert uploaded.session_state["generated_output"] == pasted.session_state[
        "generated_output"
    ]
    assert _final_area(uploaded).value == _final_area(pasted).value


def test_switching_input_modes_keeps_submitted_paste_available() -> None:
    app = _process(_start_paste_mode(), ENGLISH_SOURCE)
    fingerprint = app.session_state["file_fingerprint"]
    saved = app.session_state["saved_final_output"]

    _input_choice(app).set_value("Upload file").run(timeout=20)
    assert not app.exception
    assert len(app.file_uploader) == 1
    assert not [area for area in app.text_area if area.label == "Paste song text"]
    assert any(message.value == "Upload a file to begin." for message in app.info)

    _input_choice(app).set_value("Paste text").run(timeout=20)
    assert not app.exception
    assert not app.file_uploader
    assert _source_area(app).value == ENGLISH_SOURCE
    assert app.session_state["file_fingerprint"] == fingerprint
    assert app.session_state["saved_final_output"] == saved
    assert _final_area(app).value == saved


def test_upload_after_paste_becomes_active_until_new_paste_is_submitted() -> None:
    app = _process(_start_paste_mode(), ENGLISH_SOURCE)
    pasted_fingerprint = app.session_state["source_fingerprint"]
    pasted_preview = _final_area(app).value
    uploaded_source = (
        "[Title]\n"
        "A different upload\n\n"
        "Verse 1\n"
        "This uploaded lyric has changed\n"
    )

    _input_choice(app).set_value("Upload file").run(timeout=20)
    app.file_uploader[0].set_value(
        ("different-upload.txt", uploaded_source.encode("utf-8"), "text/plain")
    ).run(timeout=20)

    assert not app.exception
    assert not app.error
    assert app.session_state["processed_source_kind"] == "Upload file"
    assert app.session_state["source_fingerprint"] != pasted_fingerprint
    assert app.session_state["file_fingerprint"] == app.session_state[
        "source_fingerprint"
    ]
    assert app.session_state["processed_source_text"] == uploaded_source
    assert "A different upload" in _final_area(app).value
    assert _final_area(app).value != pasted_preview
    assert app.session_state["generated_output"] == _final_area(app).value
    assert app.session_state["saved_final_output"] == _final_area(app).value
    uploaded_fingerprint = app.session_state["source_fingerprint"]
    uploaded_preview = _final_area(app).value

    _input_choice(app).set_value("Paste text").run(timeout=20)

    assert not app.exception
    assert not app.error
    assert not app.file_uploader
    assert _source_area(app).value == ENGLISH_SOURCE
    assert app.session_state["processed_source_kind"] == "Upload file"
    assert app.session_state["source_fingerprint"] == uploaded_fingerprint
    assert app.session_state["processed_source_text"] == uploaded_source
    assert app.session_state["generated_output"] == uploaded_preview
    assert app.session_state["saved_final_output"] == uploaded_preview
    assert _final_area(app).value == uploaded_preview
