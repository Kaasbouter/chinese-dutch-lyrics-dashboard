from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import lyrics_dashboard.converter as converter_module
from lyrics_dashboard.alignment import build_exact_manual_plan
from lyrics_dashboard.converter import ConversionSettings, convert_lyrics
from lyrics_dashboard.models import AlignedLine, TranslationReference
from lyrics_dashboard.parser import parse_lyrics


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
CHINESE_LINES = (
    "天地玄黃，祢是永遠君王萬歲",
    "甲乙,丙丁戊己，庚辛壬癸子丑",
    "天地玄黃，宇宙洪荒萬",
    "天地玄黃,宇宙洪荒",
)
DUTCH_LINES = (
    "Nederlandse regel",
    "Een tweede regel",
    "We zingen samen",
    "Een kort lied",
)
FIRST_SIDE_CHINESE = (
    "天地玄黃//祢是永遠君王萬歲",
    "甲乙丙丁戊己//庚辛壬癸子丑",
    "天地玄黃//宇宙洪荒萬",
    "天地玄黃宇宙洪荒",
)
SECOND_SIDE_CHINESE = (
    FIRST_SIDE_CHINESE[0],
    FIRST_SIDE_CHINESE[1],
    "天地玄黃宇宙洪荒萬",
    FIRST_SIDE_CHINESE[3],
)


def _source(bilingual: bool) -> str:
    title = "歌名 | Dutch title" if bilingual else "歌名"
    source = f"[Title]\n{title}\n\nVerse 1\n" + "\n".join(CHINESE_LINES)
    if bilingual:
        source += "\n\nVerse 2\n" + "\n".join(DUTCH_LINES)
    return source + "\n"


def _expected_output(bilingual: bool, switch_index: int | None = 1) -> str:
    if not bilingual:
        return "[Title]\n歌名\n\n[Verse 1]\n" + "\n".join(
            FIRST_SIDE_CHINESE
        ) + "\n"
    sections = []
    for section_index in range(2):
        dutch_first = switch_index is not None and section_index >= switch_index
        chinese_lines = SECOND_SIDE_CHINESE if dutch_first else FIRST_SIDE_CHINESE
        rows = [
            f"{dutch}|{chinese}" if dutch_first else f"{chinese}|{dutch}"
            for chinese, dutch in zip(chinese_lines, DUTCH_LINES, strict=True)
        ]
        sections.append(f"[Verse {section_index + 1}]\n" + "\n".join(rows))
    return "[Title]\n歌名 Dutch title\n\n" + "\n\n".join(sections) + "\n"


def _plan(parsed):
    return build_exact_manual_plan(
        parsed,
        selections={0: (1,)},
        line_groups_by_chinese={
            0: tuple(
                AlignedLine(
                    source_line_indices=(index,),
                    translation_references=(TranslationReference(1, (index,)),),
                )
                for index in range(len(CHINESE_LINES))
            ),
        },
    )


@pytest.mark.parametrize("switch_index", [None, 0, 1])
def test_comma_priority_uses_existing_positional_thresholds_in_both_sections(
    switch_index: int | None,
) -> None:
    parsed = parse_lyrics(_source(bilingual=True))
    output = convert_lyrics(
        parsed,
        _plan(parsed),
        ConversionSettings(switch_index=switch_index),
    )

    assert output == _expected_output(bilingual=True, switch_index=switch_index)
    assert "，" not in output and "," not in output
    for row in (line for line in output.splitlines() if "|" in line):
        assert all(side.count("//") <= 1 for side in row.split("|"))


def test_chinese_only_commas_preserve_first_side_threshold_and_content() -> None:
    parsed = parse_lyrics(_source(bilingual=False))
    warnings: list[str] = []
    output = convert_lyrics(parsed, None, ConversionSettings(), warnings=warnings)

    assert parsed.mode == "single-language"
    assert parsed.single_language == "zh"
    assert output == _expected_output(bilingual=False)
    assert not warnings
    assert "|" not in output
    assert "，" not in output and "," not in output


def _button(app: AppTest, label: str):
    return next(button for button in app.button if button.label == label)


def _preview(app: AppTest):
    return next(area for area in app.text_area if area.label == "Converted lyrics")


@pytest.fixture
def workflow_records(monkeypatch):
    downloads: list[dict[str, object]] = []
    conversions: list[object] = []
    original_download = st.download_button
    original_convert = converter_module.convert_lyrics

    def record_download(*args, **kwargs):
        downloads.append(dict(kwargs))
        return original_download(*args, **kwargs)

    def record_convert(*args, **kwargs):
        conversions.append(args[0])
        return original_convert(*args, **kwargs)

    monkeypatch.setattr(st, "download_button", record_download)
    monkeypatch.setattr(converter_module, "convert_lyrics", record_convert)
    return downloads, conversions


def _submit_source(input_kind: str, source: str) -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    if input_kind == "Paste text":
        next(radio for radio in app.radio if radio.label == "Input source").set_value(
            "Paste text"
        ).run(timeout=20)
        next(area for area in app.text_area if area.label == "Paste song text").set_value(
            source
        )
        _button(app, "PROCESS TEXT").click().run(timeout=20)
    else:
        app.file_uploader[0].set_value(
            ("chinese-comma.txt", source.encode("utf-8"), "text/plain")
        ).run(timeout=20)
    assert not app.exception
    assert not app.error
    return app


@pytest.mark.parametrize("input_kind", ["Upload file", "Paste text"])
@pytest.mark.parametrize("bilingual", [False, True], ids=["chinese-only", "bilingual"])
def test_comma_preview_download_and_exact_update_for_uploaded_and_processed_text(
    workflow_records,
    input_kind: str,
    bilingual: bool,
) -> None:
    downloads, conversions = workflow_records
    app = _submit_source(input_kind, _source(bilingual))
    if bilingual:
        _button(app, "Validate these manual matches").click().run(timeout=20)
    assert not app.exception
    assert not app.error

    generated = _expected_output(bilingual)
    assert _preview(app).value == generated
    assert app.session_state["generated_output"] == generated
    assert app.session_state["editable_preview"] == generated
    assert app.session_state["saved_final_output"] == generated
    assert downloads[-1]["data"] == generated.encode("utf-8")
    assert downloads[-1]["data"].decode("utf-8") == _preview(app).value
    assert downloads[-1]["mime"] == "text/plain; charset=utf-8"
    assert app.get("download_button")[0].proto.ignore_rerun is True
    assert "，" not in generated and "," not in generated

    if bilingual:
        switch = next(
            selectbox
            for selectbox in app.selectbox
            if selectbox.label == "Switch to Dutch first starting at"
        )
        switch.set_value(0).run(timeout=20)
        assert app.session_state["custom_switch_index"] == 0
        assert _preview(app).value == generated
        assert downloads[-1]["data"] == generated.encode("utf-8")
        _button(app, "REGENERATE AUTOMATIC PREVIEW").click().run(timeout=20)
        regenerated = _expected_output(bilingual=True, switch_index=0)
        assert _preview(app).value == regenerated
        assert app.session_state["generated_output"] == regenerated
        assert app.session_state["saved_final_output"] == generated
        assert downloads[-1]["data"] == generated.encode("utf-8")
    else:
        regenerated = generated

    # Exact UPDATE must preserve arbitrary manual edits without reconversion,
    # including punctuation deliberately added after automatic cleaning.
    exact_edit = regenerated + "\n  自訂， 手動, //  Eigen tekst?!  \n\n"
    conversion_count = len(conversions)
    _preview(app).set_value(exact_edit).run(timeout=20)
    assert _preview(app).value == exact_edit
    assert app.session_state["saved_final_output"] == generated
    assert downloads[-1]["data"] == generated.encode("utf-8")

    _button(app, "UPDATE").click().run(timeout=20)
    assert not app.exception
    assert not app.error
    assert _preview(app).value == exact_edit
    assert app.session_state["generated_output"] == regenerated
    assert app.session_state["editable_preview"] == exact_edit
    assert app.session_state["saved_final_output"] == exact_edit
    assert downloads[-1]["data"] == exact_edit.encode("utf-8")
    assert downloads[-1]["data"].decode("utf-8") == _preview(app).value
    assert len(conversions) == conversion_count
