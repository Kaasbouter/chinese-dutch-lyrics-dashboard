from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import lyrics_dashboard.converter as converter_module
from lyrics_dashboard.converter import encode_utf8_txt
from lyrics_dashboard.parser import apply_section_languages, parse_lyrics


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"

CHINESE = "Chinese"
LATIN = "Dutch/English or Latin script"
AUTOMATIC_KEY = "automatic_section_languages"
PENDING_KEY = "pending_section_languages"
EFFECTIVE_KEY = "effective_section_languages"
EDITOR_KEY = "source_language_editor"

AUTOMATIC_UNEVEN_LANGUAGES = {
    0: "zh",
    1: "zh",
    2: "nl",
    3: "nl",
    4: "nl",
    5: "nl",
}

UNEVEN_SOURCE = (
    "[Title]\n"
    "原始歌名 | Original title?!\n\n"
    "Verse 1\n"
    "中文甲一，//保留\n"
    "中文甲二 | 保留\n\n"
    "Chorus 1\n"
    "中文乙一！\n"
    "中文乙二\n\n"
    "Bridge 1\n"
    "Dutch wrong one // keep\n"
    "Dutch wrong two | keep\n\n"
    "Verse 2\n"
    "Dutch alpha one\n"
    "Dutch alpha two\n\n"
    "Chorus 2\n"
    "Dutch beta one\n"
    "Dutch beta two\n\n"
    "Bridge 2\n"
    "Dutch gamma one\n"
    "Dutch gamma two\n"
)

NEW_BILINGUAL_SOURCE = (
    "[Title]\n"
    "新歌 | New song\n\n"
    "Verse 7\n"
    "全新中文歌詞\n\n"
    "Verse 8\n"
    "Completely new Dutch lyric\n"
)

SINGLE_LANGUAGE_SOURCE = (
    "[Title]\n"
    "A single language song\n\n"
    "Verse 1\n"
    "Only one Latin script lyric\n\n"
    "Chorus 1\n"
    "Still one language here\n"
)

SWAPPABLE_SOURCE = (
    "[Title]\n"
    "中文歌名 | Dutch title\n\n"
    "Verse 1\n"
    "中文甲\n\n"
    "Chorus 1\n"
    "中文乙\n\n"
    "Verse 2\n"
    "Dutch alpha\n\n"
    "Chorus 2\n"
    "Dutch beta\n"
)

THREE_PAIR_SOURCE = (
    "[Title]\n"
    "中文歌名 | Dutch title\n\n"
    "Verse 1\n"
    "中文甲\n\n"
    "Chorus 1\n"
    "中文乙\n\n"
    "Bridge 1\n"
    "中文丙\n\n"
    "Verse 2\n"
    "Dutch alpha\n\n"
    "Chorus 2\n"
    "Dutch beta\n\n"
    "Bridge 2\n"
    "Dutch gamma\n"
)


def _uploaded_app(
    source: str = UNEVEN_SOURCE,
    *,
    filename: str = "uneven-language-detection.txt",
) -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    app.file_uploader[0].set_value(
        (filename, source.encode("utf-8"), "text/plain")
    ).run(timeout=20)
    assert not app.exception
    return app


def _language_editor(app: AppTest):
    editors = [
        dataframe
        for dataframe in app.dataframe
        if dataframe.proto.id.endswith(f"-{EDITOR_KEY}")
    ]
    assert len(editors) == 1
    return editors[0]


def _language_delta(edits: dict[int, str]) -> dict[str, object]:
    """Return Streamlit 1.56's session-state payload for a data-editor edit."""
    return {
        "edited_rows": {
            row: {"Language": language} for row, language in edits.items()
        },
        "added_rows": [],
        "deleted_rows": [],
    }


def _refresh_with_edits(app: AppTest, edits: dict[int, str]) -> AppTest:
    app.session_state[EDITOR_KEY] = _language_delta(edits)
    next(button for button in app.button if button.label == "REFRESH").click().run(
        timeout=20
    )
    assert not app.exception
    return app


def _section_snapshot(parsed) -> tuple[object, ...]:
    return (
        parsed.raw_title,
        parsed.chinese_title,
        parsed.dutch_title,
        tuple(
            (
                section.original_index,
                section.label,
                section.kind,
                section.number,
                section.lines,
            )
            for section in parsed.sections
        ),
    )


def _record_download_payloads(monkeypatch: pytest.MonkeyPatch) -> list[bytes]:
    payloads: list[bytes] = []
    original_download_button = st.download_button

    def recording_download_button(*args, **kwargs):
        payloads.append(kwargs["data"])
        return original_download_button(*args, **kwargs)

    monkeypatch.setattr(st, "download_button", recording_download_button)
    return payloads


def test_language_editor_defaults_choices_read_only_columns_and_refresh_order() -> None:
    app = _uploaded_app()
    editor = _language_editor(app)
    column_config = json.loads(editor.proto.columns)

    assert list(editor.value.columns) == [
        "Index",
        "Section",
        "Language",
        "Lines",
        "Opening text",
    ]
    assert editor.value["Language"].tolist() == [
        CHINESE,
        CHINESE,
        LATIN,
        LATIN,
        LATIN,
        LATIN,
    ]
    assert app.session_state[AUTOMATIC_KEY] == AUTOMATIC_UNEVEN_LANGUAGES
    assert app.session_state[PENDING_KEY] == AUTOMATIC_UNEVEN_LANGUAGES
    assert app.session_state[EFFECTIVE_KEY] == AUTOMATIC_UNEVEN_LANGUAGES

    assert editor.proto.editing_mode == editor.proto.FIXED
    assert column_config["Language"]["type_config"] == {
        "type": "selectbox",
        "options": [CHINESE, LATIN],
    }
    assert [
        column
        for column in editor.value.columns
        if not column_config.get(column, {}).get("disabled", False)
    ] == ["Language"]
    for column in ("Index", "Section", "Lines", "Opening text"):
        assert column_config[column]["disabled"] is True

    detected = next(
        expander
        for expander in app.expander
        if expander.label == "Detected source sections"
    )
    detected_children = list(detected.children.values())
    assert [child.type for child in detected_children] == [
        "caption",
        "dataframe",
        "button",
    ]
    assert detected_children[0].value == (
        "Detected languages can be corrected manually below. "
        "Click REFRESH after making changes."
    )
    assert detected_children[1].proto.id == editor.proto.id
    assert detected_children[2].label == "REFRESH"
    assert detected_children[2].key == "refresh_section_languages"


@pytest.mark.parametrize(
    "edits, expected_pending",
    (
        ({2: CHINESE}, {**AUTOMATIC_UNEVEN_LANGUAGES, 2: "zh"}),
        (
            {2: CHINESE, 3: CHINESE},
            {**AUTOMATIC_UNEVEN_LANGUAGES, 2: "zh", 3: "zh"},
        ),
    ),
    ids=("one-pending-correction", "multiple-pending-corrections"),
)
def test_pending_edits_do_not_change_effective_groups_or_count_notice(
    edits: dict[int, str],
    expected_pending: dict[int, str],
) -> None:
    app = _uploaded_app()
    delta = _language_delta(edits)

    app.session_state[EDITOR_KEY] = delta
    app.run(timeout=20)

    assert not app.exception
    assert app.session_state[EDITOR_KEY] == delta
    assert app.session_state[AUTOMATIC_KEY] == AUTOMATIC_UNEVEN_LANGUAGES
    assert app.session_state[PENDING_KEY] == expected_pending
    assert app.session_state[EFFECTIVE_KEY] == AUTOMATIC_UNEVEN_LANGUAGES
    assert any(
        "2 Chinese sections and 4 Dutch sections" in warning.value
        for warning in app.warning
    )

    source_matches = {
        multiselect.key: multiselect for multiselect in app.multiselect
    }
    assert set(source_matches) == {"manual_match_0", "manual_match_1"}
    assert all(len(multiselect.options) == 4 for multiselect in source_matches.values())
    assert all(
        any("D1" in option and "[Bridge 1]" in option for option in multiselect.options)
        for multiselect in source_matches.values()
    )


def test_refresh_applies_three_by_three_effective_groups_without_mutating_source() -> None:
    source_bytes = UNEVEN_SOURCE.encode("utf-8")
    automatic_parsed = parse_lyrics(UNEVEN_SOURCE)
    automatic_snapshot = _section_snapshot(automatic_parsed)
    app = _uploaded_app()
    correction = _language_delta({2: CHINESE})

    _refresh_with_edits(app, {2: CHINESE})

    expected_effective = {**AUTOMATIC_UNEVEN_LANGUAGES, 2: "zh"}
    assert app.session_state[AUTOMATIC_KEY] == AUTOMATIC_UNEVEN_LANGUAGES
    assert app.session_state[PENDING_KEY] == expected_effective
    assert app.session_state[EFFECTIVE_KEY] == expected_effective
    assert app.session_state[EDITOR_KEY] == correction
    assert sum(
        language == "zh" for language in app.session_state[EFFECTIVE_KEY].values()
    ) == 3
    assert sum(
        language == "nl" for language in app.session_state[EFFECTIVE_KEY].values()
    ) == 3
    assert not any("Chinese sections and" in warning.value for warning in app.warning)
    assert any(
        success.value
        == "The document has 3 Chinese sections and 3 Dutch/English or Latin-script sections."
        for success in app.success
    )

    source_matches = {
        multiselect.key: multiselect for multiselect in app.multiselect
    }
    assert set(source_matches) == {
        "manual_match_0",
        "manual_match_1",
        "manual_match_2",
    }
    assert "[Bridge 1]" in source_matches["manual_match_2"].label
    assert "Dutch wrong one // keep" in source_matches["manual_match_2"].label
    assert all(len(multiselect.options) == 3 for multiselect in source_matches.values())
    assert all(
        all("[Bridge 1]" not in option for option in multiselect.options)
        for multiselect in source_matches.values()
    )
    assert app.file_uploader[0].value.getvalue() == source_bytes

    corrected_parsed = apply_section_languages(
        automatic_parsed,
        app.session_state[EFFECTIVE_KEY],
    )
    assert _section_snapshot(corrected_parsed) == automatic_snapshot
    assert [section.language for section in corrected_parsed.sections] == [
        "zh",
        "zh",
        "zh",
        "nl",
        "nl",
        "nl",
    ]
    all_source_lines = "\n".join(
        line for section in corrected_parsed.sections for line in section.lines
    )
    assert "//" in all_source_lines
    assert "|" in all_source_lines

    # A later rerun still builds grouping from the effective override. The original
    # detector map remains a separate baseline and cannot overwrite it.
    app.run(timeout=20)
    assert app.session_state[AUTOMATIC_KEY] == AUTOMATIC_UNEVEN_LANGUAGES
    assert app.session_state[EFFECTIVE_KEY] == expected_effective
    assert {multiselect.key for multiselect in app.multiselect} == {
        "manual_match_0",
        "manual_match_1",
        "manual_match_2",
    }


def test_refresh_selectively_filters_canonical_layout_and_renumbers_d_codes() -> None:
    app = _uploaded_app()
    unaffected_selection = list(app.session_state["manual_match_1"])
    unaffected_layout = deepcopy(app.session_state["manual_drag_layout_1"])

    # Use canonical section/line references, not display-card strings. Section 2
    # will become Chinese; section 3 remains a compatible Dutch counterpart.
    app.session_state["manual_match_0"] = [2, 3]
    app.session_state["manual_drag_layout_0"] = (
        ((0,), (1,)),
        (((2, 0), (3, 0)), ((2, 1), (3, 1))),
        (),
    )

    _refresh_with_edits(app, {2: CHINESE})

    assert any(
        info.value
        == "Language assignments updated. Some incompatible mappings were reset."
        for info in app.info
    )
    assert app.session_state["manual_match_0"] == [3]
    assert app.session_state["manual_drag_layout_0"] == (
        ((0,), (1,)),
        (((3, 0),), ((3, 1),)),
        (),
    )
    assert app.session_state["manual_match_1"] == unaffected_selection
    assert app.session_state["manual_drag_layout_1"] == unaffected_layout

    source_matches = {
        multiselect.key: multiselect for multiselect in app.multiselect
    }
    expected_options = [
        "D1 — [Verse 2] — Dutch alpha one",
        "D2 — [Chorus 2] — Dutch beta one",
        "D3 — [Bridge 2] — Dutch gamma one",
    ]
    assert source_matches["manual_match_0"].options == expected_options
    assert source_matches["manual_match_0"].value == [3]
    assert source_matches["manual_match_1"].value == [4]

    component_tokens = []
    for component in app.get("component_instance"):
        component_args = json.loads(component.proto.json_args)
        component_tokens.append(
            [
                item
                for container in component_args["items"]
                for item in container["items"]
            ]
        )
    assert component_tokens[0]
    assert component_tokens[1]
    assert all(token.startswith("D1:") for token in component_tokens[0])
    assert all(token.startswith("D2:") for token in component_tokens[1])


def test_refresh_preserves_a_fully_compatible_reciprocal_plan() -> None:
    app = _uploaded_app(NEW_BILINGUAL_SOURCE, filename="role-swap.txt")
    next(
        button
        for button in app.button
        if button.label == "Validate these manual matches"
    ).click().run(timeout=20)

    original_plan = app.session_state["alignment_plan"]
    original_saved_output = app.session_state["saved_final_output"]
    assert app.session_state["alignment_plan_matches_current_inputs"] is True

    _refresh_with_edits(app, {0: LATIN, 1: CHINESE})

    assert app.session_state[EFFECTIVE_KEY] == {0: "nl", 1: "zh"}
    assert app.session_state["alignment_plan"] == original_plan
    assert app.session_state["alignment_plan_matches_current_inputs"] is True
    assert app.session_state["manual_match_1"] == [0]
    assert app.session_state["manual_drag_layout_1"] == (
        ((0,),),
        (((0, 0),),),
        (),
    )
    assert not any(
        info.value
        == "Language assignments updated. Some incompatible mappings were reset."
        for info in app.info
    )
    assert app.session_state["saved_final_output"] == original_saved_output


def test_refresh_invalidates_plan_but_keeps_unaffected_validated_drafts() -> None:
    app = _uploaded_app(THREE_PAIR_SOURCE, filename="partial-reset.txt")
    next(
        button
        for button in app.button
        if button.label == "Validate these manual matches"
    ).click().run(timeout=20)

    preserved_layouts = {
        source_index: deepcopy(
            app.session_state[f"manual_drag_layout_{source_index}"]
        )
        for source_index in (1, 2)
    }
    saved_before = app.session_state["saved_final_output"]
    assert app.session_state["alignment_plan_matches_current_inputs"] is True

    _refresh_with_edits(app, {3: CHINESE})

    assert "alignment_plan" not in app.session_state.filtered_state
    assert any(
        info.value
        == "Language assignments updated. Some incompatible mappings were reset."
        for info in app.info
    )
    assert app.session_state["manual_match_0"] == []
    for source_index, preserved_layout in preserved_layouts.items():
        assert app.session_state[f"manual_drag_layout_{source_index}"] == (
            preserved_layout
        )
    assert app.session_state["manual_match_1"] == [4]
    assert app.session_state["manual_match_2"] == [5]
    assert app.session_state["saved_final_output"] == saved_before


def test_genuinely_new_upload_resets_all_language_maps_and_editor_delta() -> None:
    app = _uploaded_app()
    original_fingerprint = app.session_state["file_fingerprint"]
    _refresh_with_edits(app, {2: CHINESE})

    assert app.session_state[AUTOMATIC_KEY] != app.session_state[EFFECTIVE_KEY]
    assert app.session_state[EDITOR_KEY]["edited_rows"]

    app.file_uploader[0].set_value(
        (
            "genuinely-new.txt",
            NEW_BILINGUAL_SOURCE.encode("utf-8"),
            "text/plain",
        )
    ).run(timeout=20)

    expected_new_languages = {0: "zh", 1: "nl"}
    assert not app.exception
    assert app.session_state["file_fingerprint"] != original_fingerprint
    assert app.session_state[AUTOMATIC_KEY] == expected_new_languages
    assert app.session_state[PENDING_KEY] == expected_new_languages
    assert app.session_state[EFFECTIVE_KEY] == expected_new_languages
    assert app.session_state[EDITOR_KEY] == _language_delta({})
    assert _language_editor(app).value["Language"].tolist() == [CHINESE, LATIN]
    assert app.file_uploader[0].value.getvalue() == NEW_BILINGUAL_SOURCE.encode(
        "utf-8"
    )
    assert {multiselect.key for multiselect in app.multiselect} == {
        "manual_match_0"
    }
    assert not any(
        key in app.session_state.filtered_state
        for key in ("manual_match_1", "manual_match_2")
    )


def test_clean_single_language_mode_has_no_language_editor_or_refresh() -> None:
    app = _uploaded_app(
        SINGLE_LANGUAGE_SOURCE,
        filename="single-language.txt",
    )

    detected = next(
        expander
        for expander in app.expander
        if expander.label == "Detected source sections"
    )
    assert [child.type for child in detected.children.values()] == ["dataframe"]
    assert not [
        dataframe
        for dataframe in app.dataframe
        if dataframe.proto.id.endswith(f"-{EDITOR_KEY}")
    ]
    assert EDITOR_KEY not in app.session_state.filtered_state
    assert not any(button.label == "REFRESH" for button in app.button)
    assert app.session_state[AUTOMATIC_KEY] == {0: "nl", 1: "nl"}
    assert app.session_state[PENDING_KEY] == {0: "nl", 1: "nl"}
    assert app.session_state[EFFECTIVE_KEY] == {0: "nl", 1: "nl"}
    assert len(app.text_area) == 1
    assert any(button.label == "UPDATE" for button in app.button)
    assert [button.label for button in app.get("download_button")] == [
        "Download final TXT"
    ]


def test_refresh_preserves_output_state_and_update_download_still_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download_payloads = _record_download_payloads(monkeypatch)
    conversion_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    original_convert_lyrics = converter_module.convert_lyrics

    def recording_convert_lyrics(*args, **kwargs):
        conversion_calls.append((args, kwargs))
        return original_convert_lyrics(*args, **kwargs)

    monkeypatch.setattr(converter_module, "convert_lyrics", recording_convert_lyrics)
    app = _uploaded_app(SWAPPABLE_SOURCE, filename="swappable.txt")
    next(
        button
        for button in app.button
        if button.label == "Validate these manual matches"
    ).click().run(timeout=20)

    assert not app.exception
    assert len(conversion_calls) == 1
    generated_before = app.session_state["generated_output"]
    saved_before = app.session_state["saved_final_output"]
    exact_draft = "  [Manual title?!]\nManual // text | kept exactly  \n"
    app.text_area[0].set_value(exact_draft).run(timeout=20)

    assert app.session_state["generated_output"] == generated_before
    assert app.session_state["editable_preview"] == exact_draft
    assert app.session_state["edited_output"] == exact_draft
    assert app.session_state["saved_final_output"] == saved_before
    assert download_payloads[-1] == encode_utf8_txt(saved_before)
    download_calls_before_refresh = len(download_payloads)

    _refresh_with_edits(app, {1: LATIN, 2: CHINESE})

    corrected_languages = {0: "zh", 1: "nl", 2: "zh", 3: "nl"}
    assert app.session_state[EFFECTIVE_KEY] == corrected_languages
    assert {multiselect.key for multiselect in app.multiselect} == {
        "manual_match_0",
        "manual_match_2",
    }
    assert app.session_state["generated_output"] == generated_before
    assert app.session_state["editable_preview"] == exact_draft
    assert app.session_state["edited_output"] == exact_draft
    assert app.session_state["saved_final_output"] == saved_before
    assert len(conversion_calls) == 1
    assert len(download_payloads) == download_calls_before_refresh

    next(
        multiselect
        for multiselect in app.multiselect
        if multiselect.key == "manual_match_0"
    ).set_value([1]).run(timeout=20)
    next(
        multiselect
        for multiselect in app.multiselect
        if multiselect.key == "manual_match_2"
    ).set_value([3]).run(timeout=20)
    next(
        button
        for button in app.button
        if button.label == "Validate these manual matches"
    ).click().run(timeout=20)

    assert not app.exception
    assert not app.error
    assert app.session_state[EFFECTIVE_KEY] == corrected_languages
    assert app.session_state["alignment_plan_matches_current_inputs"] is True
    assert [
        alignment.source_section_index
        for alignment in app.session_state["alignment_plan"].alignments
    ] == [0, 1, 2, 3]
    switch = next(
        selectbox
        for selectbox in app.selectbox
        if selectbox.label == "Switch to Dutch first starting at"
    )
    assert "From [Chorus 1] (Dutch section)" in switch.options
    assert "From [Verse 2] (Chinese section)" in switch.options
    assert app.text_area[0].value == exact_draft
    assert app.session_state["generated_output"] == generated_before
    assert app.session_state["saved_final_output"] == saved_before
    assert download_payloads[-1] == encode_utf8_txt(saved_before)
    assert len(conversion_calls) == 1

    next(button for button in app.button if button.label == "UPDATE").click().run(
        timeout=20
    )

    assert app.text_area[0].value == exact_draft
    assert app.session_state["editable_preview"] == exact_draft
    assert app.session_state["saved_final_output"] == exact_draft
    assert download_payloads[-1] == encode_utf8_txt(exact_draft)
    assert any(success.value == "Changes saved" for success in app.success)
    assert len(conversion_calls) == 1

    next(
        button
        for button in app.button
        if button.label == "REGENERATE AUTOMATIC PREVIEW"
    ).click().run(timeout=20)

    assert len(conversion_calls) == 2
    converted_parsed = conversion_calls[-1][0][0]
    assert [section.language for section in converted_parsed.sections] == [
        "zh",
        "nl",
        "zh",
        "nl",
    ]
    assert app.session_state["saved_final_output"] == exact_draft
    assert download_payloads[-1] == encode_utf8_txt(exact_draft)
