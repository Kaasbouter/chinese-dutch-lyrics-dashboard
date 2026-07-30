from copy import deepcopy
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from lyrics_dashboard.alignment import (
    format_line_spec,
    suggest_manual_line_groups,
    suggest_manual_selections,
)
from lyrics_dashboard.converter import encode_utf8_txt
from lyrics_dashboard.extractors import extract_text
from lyrics_dashboard.parser import parse_lyrics


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
SAMPLE_PATH = (
    Path(__file__).resolve().parents[1] / "samples" / "basic_format_example.docx"
)


def _uploaded_sample_app() -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=10).run()
    app.file_uploader[0].set_value(
        (
            SAMPLE_PATH.name,
            SAMPLE_PATH.read_bytes(),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    ).run(timeout=10)
    return app


def test_dashboard_initial_render() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=10).run()

    assert not app.exception
    assert app.title[0].value == "Chinese\u2013Dutch Lyrics Converter"
    assert any(message.value == "Upload a file to begin." for message in app.info)


def test_dashboard_sample_drag_layout_validates_end_to_end() -> None:
    app = _uploaded_sample_app()

    assert not app.exception
    validate_buttons = [
        button
        for button in app.button
        if button.label == "Validate these manual matches"
    ]
    assert len(validate_buttons) == 1
    validate_buttons[0].click().run(timeout=10)

    assert not app.exception
    assert not app.error
    assert any(
        item.value
        == "Every detected section and line now has a reciprocal manual translation mapping."
        for item in app.success
    )
    download_buttons = app.get("download_button")
    assert len(download_buttons) == 1
    assert download_buttons[0].label == "Download final TXT"
    assert len(app.text_area) == 1
    preview_text = app.text_area[0].value
    assert preview_text == app.session_state["edited_output"]
    assert preview_text.startswith("[Title]\n")
    assert encode_utf8_txt(preview_text).decode("utf-8") == preview_text


def test_step_three_keeps_one_vertical_drop_box_per_suggested_mapping_entry() -> None:
    app = _uploaded_sample_app()
    parsed = parse_lyrics(extract_text(SAMPLE_PATH.name, SAMPLE_PATH.read_bytes()))
    chinese_sections = [section for section in parsed.sections if section.language == "zh"]
    suggested_selections = suggest_manual_selections(parsed)
    components = app.get("component_instance")
    reference_headings = [markdown.value for markdown in app.markdown]

    assert not app.exception
    assert len(components) == len(chinese_sections)
    assert reference_headings.count("**Chinese source lines**") == len(chinese_sections)
    assert reference_headings.count("**Selected Dutch lines**") == len(chinese_sections)
    assert reference_headings.count("**Mapping rows**") == len(chinese_sections)

    for component, source in zip(components, chinese_sections):
        component_args = json.loads(component.proto.json_args)
        selected = suggested_selections[source.original_index]
        expected_groups = suggest_manual_line_groups(
            parsed,
            source.original_index,
            selected,
        )
        containers = component_args["items"]
        pool, *target_boxes = containers
        range_editors = [
            dataframe
            for dataframe in app.dataframe
            if dataframe.proto.id.endswith(f"-manual_lines_{source.original_index}")
        ]

        assert component.proto.component_name == "streamlit_sortables.sortable_items"
        assert component_args["direction"] == "vertical"
        assert pool["header"] == f"Dutch card pool \u2014 [{source.label}]"
        assert len(target_boxes) == len(expected_groups)
        assert len({box["header"] for box in target_boxes}) == len(target_boxes)
        assert len(range_editors) == 1

        range_editor = range_editors[0]
        expected_specs = [
            format_line_spec(group.source_line_indices) for group in expected_groups
        ]
        assert range_editor.proto.editing_mode == range_editor.proto.DYNAMIC
        assert list(range_editor.value.columns) == ["Chinese line(s)"]
        assert range_editor.value["Chinese line(s)"].tolist() == expected_specs

        style = component_args["customStyle"]
        assert "grid-template-columns:" in style
        assert 'content: "Dutch reference(s) \u2014 drop card(s) here"' in style

        for position, (box, group) in enumerate(
            zip(target_boxes, expected_groups),
            start=1,
        ):
            expected_header = (
                f"Mapping row {position} \u2014 Chinese line(s) "
                f"{format_line_spec(group.source_line_indices)}\n"
                + "\n".join(
                    f"{line_index + 1}. {source.lines[line_index]}"
                    for line_index in group.source_line_indices
                )
            )
            assert box["header"] == expected_header
            assert isinstance(box["items"], list)


def test_seeded_drag_into_another_mapping_box_blocks_incomplete_layout() -> None:
    app = _uploaded_sample_app()
    source_index = 0
    revision = app.session_state[f"manual_drag_revision_{source_index}"]
    component_key = f"manual_drag_component_{source_index}_{revision}"
    layout = deepcopy(app.session_state[f"manual_drag_board_{source_index}"])

    moved_card = layout[1]["items"].pop()
    layout[2]["items"].insert(0, moved_card)
    app.session_state[component_key] = layout
    app.run(timeout=10)

    assert not app.exception
    assert app.session_state[f"manual_drag_board_{source_index}"] == layout
    assert any(
        warning.value == "These Chinese mapping rows still need a Dutch card: 1."
        for warning in app.warning
    )

    validate = next(
        button
        for button in app.button
        if button.label == "Validate these manual matches"
    )
    validate.click().run(timeout=10)

    assert not app.exception
    assert any(
        error.value == "[Verse 1] Chinese line group 1 needs at least one Dutch card."
        for error in app.error
    )
    assert not any(subheader.value.startswith("4.") for subheader in app.subheader)


def test_dashboard_shows_non_blocking_chinese_character_fallback_warning() -> None:
    source = (
        "[Title]\n"
        "中文歌曲 | Nederlandse titel\n\n"
        "Verse 1\n"
        "中华人民共和国\n\n"
        "Verse 2\n"
        "Nederlandse regel\n"
    )
    app = AppTest.from_file(str(APP_PATH), default_timeout=10).run()
    app.file_uploader[0].set_value(
        ("character-fallback.txt", source.encode("utf-8"), "text/plain")
    ).run(timeout=10)

    validate = next(
        button
        for button in app.button
        if button.label == "Validate these manual matches"
    )
    validate.click().run(timeout=10)
    chinese_maximum = next(
        number_input
        for number_input in app.number_input
        if number_input.label == "Maximum Chinese characters per segment"
    )
    chinese_maximum.set_value(4).run(timeout=10)

    assert not app.exception
    assert any(
        "character" in warning.value.lower() and "split" in warning.value.lower()
        for warning in app.warning
    )
    assert len(app.get("download_button")) == 1


def test_single_language_dashboard_skips_bilingual_controls_and_exports_preview() -> None:
    source = (
        "[Title]\n"
        "Grace\n\n"
        "Verse 1\n"
        "We sing together for the Lord\n\n"
        "Chorus 1\n"
        "A short song\n"
    )
    app = AppTest.from_file(str(APP_PATH), default_timeout=10).run()
    app.file_uploader[0].set_value(
        ("english-only.txt", source.encode("utf-8"), "text/plain")
    ).run(timeout=10)

    assert not app.exception
    assert not app.error
    assert any(
        message.value
        == "Single-language song detected \u2014 Dutch/English or Latin-script language."
        for message in app.success
    )
    assert not any(
        button.label == "Validate these manual matches" for button in app.button
    )
    assert not app.multiselect
    assert not app.get("component_instance")
    assert not any(
        selectbox.label == "Switch to Dutch first starting at"
        for selectbox in app.selectbox
    )
    assert not any(
        subheader.value.startswith(
            (
                "2. Match",
                "3. Confirm",
                "4. Choose",
            )
        )
        for subheader in app.subheader
    )

    assert len(app.text_area) == 1
    preview_text = app.text_area[0].value
    assert "|" not in preview_text
    assert preview_text == app.session_state["edited_output"]
    assert encode_utf8_txt(preview_text).decode("utf-8") == preview_text

    edited_preview = preview_text.replace("A short song", "A revised song")
    app.text_area[0].set_value(edited_preview).run(timeout=10)
    assert app.text_area[0].value == edited_preview
    assert app.session_state["edited_output"] == edited_preview
    assert encode_utf8_txt(app.text_area[0].value).decode("utf-8") == edited_preview

    download_buttons = app.get("download_button")
    assert len(download_buttons) == 1
    assert download_buttons[0].label == "Download final TXT"


def test_leading_link_never_reaches_manual_controls_preview_or_txt() -> None:
    forbidden_marker = "forbidden-ui-marker"
    source = (
        f"https://example.invalid/{forbidden_marker}\n\n"
        "中文歌名 | Nederlandse titel\n\n"
        "Verse 1\n"
        "中文歌词\n\n"
        "Verse 2\n"
        "Nederlandse liedregel\n"
    )
    app = AppTest.from_file(str(APP_PATH), default_timeout=10).run()
    app.file_uploader[0].set_value(
        ("leading-link.txt", source.encode("utf-8"), "text/plain")
    ).run(timeout=10)

    assert not app.exception
    assert not app.error

    manual_control_text = "\n".join(
        [
            *(multiselect.label for multiselect in app.multiselect),
            *(str(dataframe.value) for dataframe in app.dataframe),
            *(
                component.proto.json_args
                for component in app.get("component_instance")
            ),
        ]
    ).lower()
    status_text = "\n".join(
        str(item.value)
        for collection in (
            app.error,
            app.warning,
            app.info,
            app.success,
            app.caption,
            app.markdown,
        )
        for item in collection
    ).lower()

    assert forbidden_marker not in manual_control_text
    assert forbidden_marker not in status_text

    validate = next(
        button
        for button in app.button
        if button.label == "Validate these manual matches"
    )
    validate.click().run(timeout=10)

    assert not app.exception
    assert not app.error
    generated_preview = app.text_area[0].value
    assert generated_preview == app.session_state["edited_output"]
    assert forbidden_marker not in generated_preview.lower()
    assert encode_utf8_txt(generated_preview).decode("utf-8") == generated_preview

    edited_preview = generated_preview.replace(
        "Nederlandse titel",
        "Aangepaste titel",
    )
    app.text_area[0].set_value(edited_preview).run(timeout=10)

    assert app.text_area[0].value == edited_preview
    assert app.session_state["edited_output"] == edited_preview
    assert forbidden_marker not in edited_preview.lower()
    downloaded_txt = encode_utf8_txt(app.text_area[0].value)
    assert downloaded_txt.decode("utf-8") == edited_preview
    assert forbidden_marker.encode("utf-8") not in downloaded_txt.lower()
