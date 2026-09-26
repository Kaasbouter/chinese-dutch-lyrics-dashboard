from __future__ import annotations

import hashlib
import re
from collections.abc import MutableMapping
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_sortables import sort_items

from lyrics_dashboard.alignment import (
    build_exact_manual_plan,
    format_line_spec,
    parse_line_spec,
    suggest_manual_line_groups,
    suggest_manual_selections,
    validate_alignment_plan,
)
from lyrics_dashboard.converter import (
    ConversionSettings,
    convert_lyrics,
    derive_first_side_limit,
    encode_utf8_txt,
)
from lyrics_dashboard.drag_mapping import (
    build_drag_line_groups,
    decode_drag_containers,
    suggest_drag_line_assignments,
)
from lyrics_dashboard.errors import LyricsDashboardError, PairingError
from lyrics_dashboard.extractors import SUPPORTED_EXTENSIONS, extract_text
from lyrics_dashboard.models import AlignmentPlan, ParsedLyrics
from lyrics_dashboard.parser import apply_section_languages, parse_lyrics

CHINESE_MAX_KEY = "custom_chinese_max_length"
DUTCH_MAX_KEY = "custom_dutch_max_length"
SWITCH_INDEX_KEY = "custom_switch_index"
TITLE_SEPARATOR_KEY = "custom_title_separator"
GENERATED_OUTPUT_KEY = "generated_output"
EDITABLE_PREVIEW_KEY = "editable_preview"
EDITED_OUTPUT_WIDGET_KEY = "edited_output"
SAVED_FINAL_OUTPUT_KEY = "saved_final_output"
AUTOMATIC_LANGUAGES_KEY = "automatic_section_languages"
PENDING_LANGUAGES_KEY = "pending_section_languages"
EFFECTIVE_LANGUAGES_KEY = "effective_section_languages"
LANGUAGE_EDITOR_KEY = "source_language_editor"
PRESERVED_ALIGNMENT_KEY = "preserved_alignment_after_language_refresh"
ALIGNMENT_CURRENT_KEY = "alignment_plan_matches_current_inputs"
INPUT_SOURCE_KEY = "input_source"
PASTED_SOURCE_DRAFT_KEY = "pasted_source_draft"
PASTED_SOURCE_DRAFT_MEMORY_KEY = "pasted_source_draft_memory"
PROCESSED_SOURCE_TEXT_KEY = "processed_source_text"
PROCESSED_SOURCE_NAME_KEY = "processed_source_name"
PROCESSED_SOURCE_KIND_KEY = "processed_source_kind"
SOURCE_FINGERPRINT_KEY = "source_fingerprint"

LANGUAGE_LABEL_BY_CODE = {
    "zh": "Chinese",
    "nl": "Dutch/English or Latin script",
}
LANGUAGE_CODE_BY_LABEL = {
    label: code for code, label in LANGUAGE_LABEL_BY_CODE.items()
}
READ_ONLY_LANGUAGE_COLUMNS = ("Index", "Section", "Lines", "Opening text")

DragLayout = tuple[
    tuple[tuple[int, ...], ...],
    tuple[tuple[tuple[int, int], ...], ...],
    tuple[tuple[int, int], ...],
]


def _activate_source(fingerprint: str) -> None:
    """Reset per-song state when a submitted upload or paste changes."""
    if st.session_state.get(SOURCE_FINGERPRINT_KEY) == fingerprint:
        return

    for state_key in list(st.session_state):
        if state_key in {
            "alignment_plan",
            "alignment_fingerprint",
            "alignment_input_signature",
            ALIGNMENT_CURRENT_KEY,
            "control_signature",
            AUTOMATIC_LANGUAGES_KEY,
            PENDING_LANGUAGES_KEY,
            EFFECTIVE_LANGUAGES_KEY,
            LANGUAGE_EDITOR_KEY,
            PRESERVED_ALIGNMENT_KEY,
            "refresh_section_languages",
            GENERATED_OUTPUT_KEY,
            EDITABLE_PREVIEW_KEY,
            EDITED_OUTPUT_WIDGET_KEY,
            SAVED_FINAL_OUTPUT_KEY,
            "applied_conversion_warnings",
            "generation_attempt_signature",
            "generation_error",
            "generation_succeeded",
            "preview_save_succeeded",
            PROCESSED_SOURCE_TEXT_KEY,
            PROCESSED_SOURCE_NAME_KEY,
            PROCESSED_SOURCE_KIND_KEY,
        } or state_key.startswith(
            (
                "manual_match_",
                "manual_lines_",
                "manual_lines_signature_",
                "manual_drag_",
                "manual_join_",
            )
        ):
            st.session_state.pop(state_key, None)

    st.session_state[SOURCE_FINGERPRINT_KEY] = fingerprint
    # Keep the existing upload identity key for the established dashboard state.
    st.session_state["file_fingerprint"] = fingerprint


def _section_language_assignments(parsed: ParsedLyrics) -> dict[int, str]:
    """Return stable per-document language assignments keyed by source index."""
    return {
        section.original_index: section.language
        for section in parsed.sections
    }


def _alignment_draft(
    plan: AlignmentPlan,
    source_index: int,
) -> tuple[tuple[int, ...], DragLayout] | None:
    """Recover a directional drag-board draft from a reciprocal validated plan."""
    try:
        alignment = plan.for_section(source_index)
    except KeyError:
        return None

    assignments = tuple(
        tuple(
            (reference.section_index, line_index)
            for reference in group.translation_references
            for line_index in reference.line_indices
        )
        for group in alignment.aligned_lines
    )
    layout: DragLayout = (
        tuple(group.source_line_indices for group in alignment.aligned_lines),
        assignments,
        (),
    )
    return alignment.counterpart_section_indices, layout


def _clear_source_mapping_state(
    state: MutableMapping[str, object],
    source_index: int,
) -> bool:
    """Remove only the mapping widgets owned by one former Chinese source."""
    match_key = f"manual_match_{source_index}"
    layout_key = f"manual_drag_layout_{source_index}"
    had_mapping = bool(state.get(match_key)) or layout_key in state
    exact_keys = {
        match_key,
        f"manual_lines_{source_index}",
        f"manual_lines_base_{source_index}",
        f"manual_lines_signature_{source_index}",
        f"manual_drag_selection_{source_index}",
        f"manual_drag_board_{source_index}",
        f"manual_drag_board_signature_{source_index}",
        f"manual_drag_revision_{source_index}",
        f"manual_drag_reset_{source_index}",
        f"manual_drag_pool_{source_index}",
        layout_key,
    }
    component_prefix = f"manual_drag_component_{source_index}_"
    join_prefix = f"manual_join_{source_index}"
    for state_key in list(state):
        if (
            state_key in exact_keys
            or state_key.startswith(component_prefix)
            or state_key.startswith(join_prefix)
        ):
            state.pop(state_key, None)
    return had_mapping


def _normalize_drag_layout(value: object) -> DragLayout | None:
    """Normalize persisted canonical drag references without trusting display strings."""
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        return None
    raw_groups, raw_assignments, raw_unassigned = value
    try:
        groups = tuple(tuple(int(item) for item in group) for group in raw_groups)
        assignments = tuple(
            tuple((int(section), int(line)) for section, line in assignment)
            for assignment in raw_assignments
        )
        unassigned = tuple(
            (int(section), int(line)) for section, line in raw_unassigned
        )
    except (TypeError, ValueError):
        return None
    if len(groups) != len(assignments):
        return None
    return groups, assignments, unassigned


def _reconcile_language_mapping_state(
    state: MutableMapping[str, object],
    previous: ParsedLyrics,
    updated: ParsedLyrics,
    fingerprint: str,
) -> bool:
    """Preserve compatible mapping drafts and remove only cross-language conflicts."""
    previous_languages = _section_language_assignments(previous)
    updated_languages = _section_language_assignments(updated)
    if previous_languages == updated_languages:
        return False

    mappings_reset = False
    stored_plan = state.get("alignment_plan")
    previous_plan = (
        stored_plan
        if isinstance(stored_plan, AlignmentPlan)
        and bool(state.get(ALIGNMENT_CURRENT_KEY))
        else None
    )
    plan_is_valid = False
    if isinstance(previous_plan, AlignmentPlan):
        try:
            validate_alignment_plan(updated, previous_plan)
        except LyricsDashboardError:
            mappings_reset = True
            for state_key in (
                "alignment_plan",
                "alignment_fingerprint",
                "alignment_input_signature",
                PRESERVED_ALIGNMENT_KEY,
                ALIGNMENT_CURRENT_KEY,
            ):
                state.pop(state_key, None)
        else:
            plan_is_valid = True
            state[PRESERVED_ALIGNMENT_KEY] = True
    elif isinstance(stored_plan, AlignmentPlan):
        for state_key in (
            "alignment_plan",
            "alignment_fingerprint",
            "alignment_input_signature",
            PRESERVED_ALIGNMENT_KEY,
            ALIGNMENT_CURRENT_KEY,
        ):
            state.pop(state_key, None)

    previous_chinese = {
        index for index, language in previous_languages.items() if language == "zh"
    }
    updated_chinese = {
        index for index, language in updated_languages.items() if language == "zh"
    }
    for source_index in sorted(previous_chinese - updated_chinese):
        removed_directional_state = _clear_source_mapping_state(
            state,
            source_index,
        )
        if removed_directional_state and not plan_is_valid:
            mappings_reset = True

    for source_index in sorted(updated_chinese):
        match_key = f"manual_match_{source_index}"
        layout_key = f"manual_drag_layout_{source_index}"
        plan_draft = (
            _alignment_draft(previous_plan, source_index)
            if isinstance(previous_plan, AlignmentPlan)
            else None
        )

        has_preserved_selection = match_key in state or plan_draft is not None
        if match_key in state:
            raw_selected = state.get(match_key)
            selected = (
                tuple(
                    dict.fromkeys(
                        int(index)
                        for index in raw_selected
                        if isinstance(index, int)
                    )
                )
                if isinstance(raw_selected, (tuple, list))
                else ()
            )
        elif plan_draft is not None:
            selected = tuple(plan_draft[0])
        else:
            selected = ()

        valid_selected = tuple(
            index
            for index in selected
            if 0 <= index < len(updated.sections)
            and updated.sections[index].language == "nl"
        )
        if valid_selected != selected:
            mappings_reset = True
        if has_preserved_selection:
            state[match_key] = list(valid_selected)

        layout = _normalize_drag_layout(state.get(layout_key))
        if layout is None and plan_draft is not None:
            layout = plan_draft[1]

        if layout is not None:
            source_groups, assignments, unassigned = layout
            allowed_sections = set(valid_selected)

            def compatible(reference: tuple[int, int]) -> bool:
                section_index, line_index = reference
                return (
                    section_index in allowed_sections
                    and 0 <= line_index < len(updated.sections[section_index].lines)
                    and updated.sections[section_index].language == "nl"
                )

            filtered_assignments = tuple(
                tuple(reference for reference in assignment if compatible(reference))
                for assignment in assignments
            )
            filtered_unassigned = tuple(
                reference for reference in unassigned if compatible(reference)
            )
            old_references = tuple(
                reference
                for assignment in assignments
                for reference in assignment
            ) + tuple(unassigned)
            new_references = tuple(
                reference
                for assignment in filtered_assignments
                for reference in assignment
            ) + filtered_unassigned
            if new_references != old_references:
                mappings_reset = True

            preserved_layout: DragLayout = (
                source_groups,
                filtered_assignments,
                filtered_unassigned,
            )
            state[layout_key] = preserved_layout
            state[f"manual_lines_base_{source_index}"] = [
                {"Chinese line(s)": format_line_spec(group)}
                for group in source_groups
            ]
            state.pop(f"manual_lines_{source_index}", None)

            selection_signature = (
                fingerprint,
                source_index,
                valid_selected,
            )
            state[f"manual_lines_signature_{source_index}"] = selection_signature
            state[f"manual_drag_selection_{source_index}"] = selection_signature

        old_revision = int(
            state.get(f"manual_drag_revision_{source_index}", -1)
        )
        for state_key in list(state):
            if state_key.startswith(f"manual_drag_component_{source_index}_"):
                state.pop(state_key, None)
        state.pop(f"manual_drag_board_{source_index}", None)
        state.pop(f"manual_drag_board_signature_{source_index}", None)
        state[f"manual_drag_revision_{source_index}"] = old_revision + 1

    if plan_is_valid:
        state["alignment_plan"] = previous_plan
        state["alignment_fingerprint"] = fingerprint
    return mappings_reset


def _current_customization(
    parsed: ParsedLyrics,
    alignment_plan: AlignmentPlan | None,
    fingerprint: str,
) -> tuple[ConversionSettings, tuple[object, ...]]:
    """Build settings and their signature from the live widget state."""
    if parsed.mode == "single-language":
        chinese_max = (
            int(st.session_state[CHINESE_MAX_KEY])
            if parsed.single_language == "zh"
            else 10
        )
        dutch_max = (
            int(st.session_state[DUTCH_MAX_KEY])
            if parsed.single_language == "nl"
            else 40
        )
        settings = ConversionSettings(
            chinese_max_length=chinese_max,
            dutch_max_length=dutch_max,
        )
        signature = (
            fingerprint,
            parsed.mode,
            parsed.single_language,
            tuple(section.language for section in parsed.sections),
            chinese_max,
            dutch_max,
        )
        return settings, signature

    selected_switch = st.session_state[SWITCH_INDEX_KEY]
    chinese_max = int(st.session_state[CHINESE_MAX_KEY])
    dutch_max = int(st.session_state[DUTCH_MAX_KEY])
    title_separator_choice = str(st.session_state[TITLE_SEPARATOR_KEY])
    settings = ConversionSettings(
        switch_index=selected_switch,
        chinese_max_length=chinese_max,
        dutch_max_length=dutch_max,
        title_separator="|" if title_separator_choice.startswith("|") else " ",
    )
    signature = (
        fingerprint,
        repr(alignment_plan),
        tuple(section.language for section in parsed.sections),
        selected_switch,
        chinese_max,
        dutch_max,
        title_separator_choice,
    )
    return settings, signature


def _generate_conversion_output(
    parsed: ParsedLyrics,
    alignment_plan: AlignmentPlan | None,
    fingerprint: str,
    *,
    initialize_saved_output: bool,
    announce: bool = True,
) -> bool:
    """Generate output without conflating conversion with manual preview saves."""
    settings, control_signature = _current_customization(
        parsed,
        alignment_plan,
        fingerprint,
    )
    conversion_warnings: list[str] = []
    try:
        generated = convert_lyrics(
            parsed,
            alignment_plan,
            settings,
            warnings=conversion_warnings,
        )
    except (LyricsDashboardError, ValueError) as exc:
        st.session_state["generation_attempt_signature"] = control_signature
        st.session_state["generation_error"] = str(exc)
        st.session_state["generation_succeeded"] = False
        return False

    st.session_state["control_signature"] = control_signature
    st.session_state[GENERATED_OUTPUT_KEY] = generated
    st.session_state[EDITABLE_PREVIEW_KEY] = generated
    st.session_state[EDITED_OUTPUT_WIDGET_KEY] = generated
    if initialize_saved_output or SAVED_FINAL_OUTPUT_KEY not in st.session_state:
        st.session_state[SAVED_FINAL_OUTPUT_KEY] = generated
    st.session_state["applied_conversion_warnings"] = tuple(conversion_warnings)
    st.session_state["generation_attempt_signature"] = control_signature
    st.session_state["generation_error"] = None
    st.session_state["generation_succeeded"] = announce
    st.session_state["preview_save_succeeded"] = False
    return True


def _regenerate_conversion_output(
    parsed: ParsedLyrics,
    alignment_plan: AlignmentPlan | None,
    fingerprint: str,
) -> None:
    """Deliberately rebuild the editable preview while keeping the saved TXT stable."""
    _generate_conversion_output(
        parsed,
        alignment_plan,
        fingerprint,
        initialize_saved_output=False,
    )


def _capture_editable_preview() -> None:
    """Keep a non-widget copy of the current draft when the textarea changes."""
    st.session_state[EDITABLE_PREVIEW_KEY] = st.session_state[
        EDITED_OUTPUT_WIDGET_KEY
    ]
    st.session_state["generation_succeeded"] = False
    st.session_state["preview_save_succeeded"] = False


def _save_editable_preview() -> None:
    """Save the exact current textarea value without invoking conversion logic."""
    current_preview = st.session_state[EDITED_OUTPUT_WIDGET_KEY]
    st.session_state[EDITABLE_PREVIEW_KEY] = current_preview
    st.session_state[SAVED_FINAL_OUTPUT_KEY] = current_preview
    st.session_state["generation_succeeded"] = False
    st.session_state["preview_save_succeeded"] = True


def _render_regeneration_action(
    *,
    parsed: ParsedLyrics,
    alignment_plan: AlignmentPlan | None,
    fingerprint: str,
    control_signature: tuple[object, ...],
) -> None:
    """Offer an explicit conversion action separate from saving manual edits."""
    st.button(
        "REGENERATE AUTOMATIC PREVIEW",
        key="regenerate_output",
        width="stretch",
        help=(
            "Rebuild from the current mappings and splitting settings. This replaces the "
            "editable preview; click UPDATE afterward to save it for download."
        ),
        on_click=_regenerate_conversion_output,
        args=(parsed, alignment_plan, fingerprint),
    )

    if st.session_state.get("generation_attempt_signature") == control_signature:
        generation_error = st.session_state.get("generation_error")
        if generation_error:
            st.error(generation_error)
        elif st.session_state.get("generation_succeeded"):
            st.success("Preview regenerated; click UPDATE to save it")
    elif st.session_state.get("control_signature") != control_signature:
        st.info(
            "The conversion settings or mappings have changed. Regenerate the automatic "
            "preview when you want to replace the current editable draft."
        )


def _render_editable_preview(*, subheader: str, help_text: str) -> None:
    """Render the state-managed preview without resetting its current draft."""
    if EDITED_OUTPUT_WIDGET_KEY not in st.session_state:
        st.session_state[EDITED_OUTPUT_WIDGET_KEY] = st.session_state[
            EDITABLE_PREVIEW_KEY
        ]

    st.subheader(subheader)
    st.caption(help_text)
    st.text_area(
        "Converted lyrics",
        key=EDITED_OUTPUT_WIDGET_KEY,
        height=560,
        label_visibility="collapsed",
        on_change=_capture_editable_preview,
    )
    st.session_state[EDITABLE_PREVIEW_KEY] = st.session_state[
        EDITED_OUTPUT_WIDGET_KEY
    ]


def _render_final_actions(
    *,
    output_name: str,
) -> None:
    """Render the exact-save action directly above the saved TXT download."""
    download_data = encode_utf8_txt(st.session_state[SAVED_FINAL_OUTPUT_KEY])
    _, action_column = st.columns([2, 1], gap="small")
    with action_column:
        st.button(
            "UPDATE",
            key="update_output",
            type="primary",
            width="stretch",
            on_click=_save_editable_preview,
        )
        st.download_button(
            "Download final TXT",
            data=download_data,
            file_name=output_name,
            mime="text/plain; charset=utf-8",
            on_click="ignore",
            type="primary",
            width="stretch",
        )

        if st.session_state.get("preview_save_succeeded"):
            st.success("Changes saved")


DRAG_BOARD_STYLE = """
.sortable-component, .sortable-component * {
    box-sizing: border-box;
}
.sortable-component.vertical {
    display: flex;
    flex-direction: column;
    flex-wrap: nowrap;
    gap: 0.75rem;
    padding: 0.1rem;
    width: 100%;
}
.sortable-component.vertical > .sortable-container {
    align-items: stretch;
    background: #ffffff;
    border: 1px solid #94a3b8;
    border-radius: 0.65rem;
    display: grid;
    flex: 0 0 auto;
    grid-template-columns: minmax(15rem, 40%) minmax(0, 60%);
    margin: 0 !important;
    overflow: hidden;
    padding: 0;
    width: 100% !important;
}
.sortable-component.vertical > .sortable-container > .sortable-container-header {
    align-items: center;
    background: #f8fafc;
    border-radius: 0;
    border-right: 1px solid #cbd5e1;
    color: #0f172a;
    display: flex;
    font-weight: 700;
    line-height: 1.45;
    min-width: 0;
    overflow-wrap: anywhere;
    padding: 0.75rem 0.85rem;
    white-space: pre-line;
}
.sortable-component.vertical > .sortable-container > .sortable-container-body {
    background: #f8fbff;
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    min-height: 4.6rem;
    min-width: 0;
    padding: 0.55rem;
}
.sortable-component.vertical > .sortable-container:not(:first-of-type) >
.sortable-container-body:empty::before {
    color: #64748b;
    content: "Dutch reference(s) — drop card(s) here";
    font-style: italic;
    margin: auto 0;
}
.sortable-component.vertical > .sortable-container:first-of-type {
    background: #eff6ff;
    border: 2px solid #60a5fa;
    display: block;
}
.sortable-component.vertical > .sortable-container:first-of-type >
.sortable-container-header {
    background: #dbeafe;
    border-bottom: 1px solid #93c5fd;
    border-right: 0;
}
.sortable-component.vertical > .sortable-container:first-of-type >
.sortable-container-body {
    background: #eff6ff;
    display: grid;
    gap: 0.45rem;
    grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr));
    max-height: 14rem;
    min-height: 4.25rem;
    overflow-y: auto;
}
.sortable-component.vertical > .sortable-container:first-of-type >
.sortable-container-body:empty::before {
    color: #475569;
    content: "All Dutch cards are placed in mapping rows";
    font-style: italic;
    grid-column: 1 / -1;
    margin: auto 0;
}
.sortable-item, .sortable-item:hover {
    background: #dbeafe;
    border: 1px solid #60a5fa;
    border-radius: 0.45rem;
    color: #0f172a;
    cursor: grab;
    font-weight: 600;
    line-height: 1.35;
    margin: 0 !important;
    overflow-wrap: anywhere;
    padding: 0.6rem 0.7rem;
    white-space: pre-wrap;
    width: 100%;
}
.sortable-item:focus-visible {
    outline: 3px solid #f59e0b;
    outline-offset: 2px;
}
.sortable-item.dragging {
    cursor: grabbing;
}
@media (max-width: 720px) {
    .sortable-component.vertical > .sortable-container:not(:first-of-type) {
        grid-template-columns: 1fr;
    }
    .sortable-component.vertical > .sortable-container:not(:first-of-type) >
    .sortable-container-header {
        border-bottom: 1px solid #cbd5e1;
        border-right: 0;
    }
    .sortable-component.vertical > .sortable-container:first-of-type >
    .sortable-container-body {
        grid-template-columns: 1fr;
    }
}
"""

st.set_page_config(page_title="Chinese–Dutch Lyrics Converter", page_icon="🎵", layout="wide")

st.title("Chinese–Dutch Lyrics Converter")
st.caption(
    "A completely free local dashboard. Single-language songs convert directly; for bilingual "
    "songs, confirm the true Chinese–Dutch matches and language order before downloading UTF-8 TXT."
)

with st.expander("Expected input format", expanded=False):
    st.markdown(
        """
- A title at the top.
- Either one single-language set of lyric sections, or one complete Chinese block and one complete Dutch block.
- Section headings such as `Verse 1`, `Chorus 1`, `Bridge`, or `Refrein 1`.
- Bilingual blocks may have **different numbers of sections or lyric lines**.
- For bilingual songs, you manually confirm which sections and exact line ranges are translations of each other.
- No API key, subscription, cloud AI, or paid service is used.
        """
    )

input_source = st.radio(
    "Input source",
    ("Upload file", "Paste text"),
    horizontal=True,
    key=INPUT_SOURCE_KEY,
)

if input_source == "Upload file":
    if PASTED_SOURCE_DRAFT_KEY in st.session_state:
        st.session_state[PASTED_SOURCE_DRAFT_MEMORY_KEY] = st.session_state[
            PASTED_SOURCE_DRAFT_KEY
        ]
    uploaded_file = st.file_uploader(
        "1. Upload the basic-format lyrics file",
        type=[extension.lstrip(".") for extension in sorted(SUPPORTED_EXTENSIONS)],
        help="Supported: DOCX, PDF with selectable text, PPTX, XLSX, TXT, MD, CSV, JSON, XML and HTML.",
    )
    if not uploaded_file:
        st.info("Upload a file to begin.")
        st.stop()

    file_bytes = uploaded_file.getvalue()
    fingerprint = hashlib.sha256(file_bytes).hexdigest()
    _activate_source(fingerprint)
    try:
        source_text = extract_text(uploaded_file.name, file_bytes)
        automatic_parsed = parse_lyrics(source_text)
    except LyricsDashboardError as exc:
        st.error(str(exc))
        st.stop()
    source_name = uploaded_file.name
    st.session_state[PROCESSED_SOURCE_TEXT_KEY] = source_text
    st.session_state[PROCESSED_SOURCE_NAME_KEY] = source_name
    st.session_state[PROCESSED_SOURCE_KIND_KEY] = "Upload file"
else:
    if PASTED_SOURCE_DRAFT_KEY not in st.session_state:
        st.session_state[PASTED_SOURCE_DRAFT_KEY] = st.session_state.get(
            PASTED_SOURCE_DRAFT_MEMORY_KEY,
            st.session_state.get(PROCESSED_SOURCE_TEXT_KEY, "")
            if st.session_state.get(PROCESSED_SOURCE_KIND_KEY) == "Paste text"
            else "",
        )
    with st.form("paste_source_form"):
        st.caption("Paste the song in the same basic format you would normally upload.")
        st.text_area(
            "Paste song text",
            key=PASTED_SOURCE_DRAFT_KEY,
            height=320,
        )
        process_text = st.form_submit_button("PROCESS TEXT")

    submitted_parsed: ParsedLyrics | None = None
    if process_text:
        submitted_text = st.session_state[PASTED_SOURCE_DRAFT_KEY]
        st.session_state[PASTED_SOURCE_DRAFT_MEMORY_KEY] = submitted_text
        if not submitted_text.strip():
            st.error("Paste song text before processing.")
        else:
            try:
                submitted_parsed = parse_lyrics(submitted_text)
            except LyricsDashboardError as exc:
                st.error(str(exc))
            else:
                fingerprint = "paste:" + hashlib.sha256(
                    submitted_text.encode("utf-8")
                ).hexdigest()
                _activate_source(fingerprint)
                st.session_state[PROCESSED_SOURCE_TEXT_KEY] = submitted_text
                st.session_state[PROCESSED_SOURCE_NAME_KEY] = "pasted_song.txt"
                st.session_state[PROCESSED_SOURCE_KIND_KEY] = "Paste text"

    if PROCESSED_SOURCE_TEXT_KEY not in st.session_state:
        st.info("Paste song text and click PROCESS TEXT to begin.")
        st.stop()

    if st.session_state.get(PROCESSED_SOURCE_KIND_KEY) == "Upload file":
        st.info("The last uploaded song remains active until you click PROCESS TEXT.")

    source_text = st.session_state[PROCESSED_SOURCE_TEXT_KEY]
    source_name = st.session_state[PROCESSED_SOURCE_NAME_KEY]
    fingerprint = st.session_state[SOURCE_FINGERPRINT_KEY]
    try:
        automatic_parsed = submitted_parsed or parse_lyrics(source_text)
    except LyricsDashboardError as exc:
        st.error(str(exc))
        st.stop()

automatic_languages = _section_language_assignments(automatic_parsed)
st.session_state[AUTOMATIC_LANGUAGES_KEY] = dict(automatic_languages)
if PENDING_LANGUAGES_KEY not in st.session_state:
    st.session_state[PENDING_LANGUAGES_KEY] = dict(automatic_languages)
if EFFECTIVE_LANGUAGES_KEY not in st.session_state:
    st.session_state[EFFECTIVE_LANGUAGES_KEY] = dict(automatic_languages)

try:
    parsed = apply_section_languages(
        automatic_parsed,
        st.session_state[EFFECTIVE_LANGUAGES_KEY],
    )
except (LyricsDashboardError, ValueError) as exc:
    st.error(f"The saved language assignments are invalid: {exc}")
    st.stop()

section_rows = [
    {
        "Index": section.original_index,
        "Section": f"[{section.label}]",
        "Language": LANGUAGE_LABEL_BY_CODE[section.language],
        "Lines": len(section.lines),
        "Opening text": section.lines[0],
    }
    for section in automatic_parsed.sections
]
with st.expander("Detected source sections", expanded=False):
    if automatic_parsed.mode == "bilingual":
        st.caption(
            "Detected languages can be corrected manually below. Click REFRESH after making changes."
        )
        edited_section_rows = st.data_editor(
            pd.DataFrame(section_rows),
            key=LANGUAGE_EDITOR_KEY,
            num_rows="fixed",
            width="stretch",
            hide_index=True,
            disabled=list(READ_ONLY_LANGUAGE_COLUMNS),
            column_config={
                "Language": st.column_config.SelectboxColumn(
                    "Language",
                    options=list(LANGUAGE_CODE_BY_LABEL),
                    format_func=lambda language: f"{language}  ▾",
                    required=True,
                    help=(
                        "Chinese, or the existing Dutch/English/Latin-script language group."
                    ),
                ),
            },
        )
        pending_languages = {
            int(row["Index"]): LANGUAGE_CODE_BY_LABEL[str(row["Language"])]
            for row in edited_section_rows.to_dict("records")
        }
        st.session_state[PENDING_LANGUAGES_KEY] = pending_languages

        if st.button(
            "REFRESH",
            key="refresh_section_languages",
            width="stretch",
        ):
            try:
                updated_parsed = apply_section_languages(
                    automatic_parsed,
                    pending_languages,
                )
            except (LyricsDashboardError, ValueError) as exc:
                st.error(f"Language assignments were not applied: {exc}")
            else:
                mappings_reset = _reconcile_language_mapping_state(
                    st.session_state,
                    parsed,
                    updated_parsed,
                    fingerprint,
                )
                st.session_state[EFFECTIVE_LANGUAGES_KEY] = dict(
                    pending_languages
                )
                parsed = updated_parsed
                if mappings_reset:
                    st.info(
                        "Language assignments updated. Some incompatible mappings were reset."
                    )
    else:
        st.dataframe(
            pd.DataFrame(section_rows),
            width="stretch",
            hide_index=True,
        )

for warning in parsed.warnings:
    st.warning(warning)

chinese_sections = [section for section in parsed.sections if section.language == "zh"]
dutch_sections = [section for section in parsed.sections if section.language == "nl"]
single_language_mode = parsed.mode == "single-language"
if (
    not single_language_mode
    and len(chinese_sections) == len(dutch_sections)
):
    st.success(
        f"The document has {len(chinese_sections)} Chinese sections and "
        f"{len(dutch_sections)} Dutch/English or Latin-script sections."
    )
detected_language_label = (
    "Chinese"
    if parsed.single_language == "zh"
    else "Dutch/English or Latin-script language"
)
if single_language_mode:
    st.success(f"Single-language song detected — {detected_language_label}.")

section_to_code = {
    section.original_index: f"D{position}"
    for position, section in enumerate(dutch_sections, start=1)
}

if single_language_mode:
    st.subheader("2. Configure single-language splitting")
    with st.expander("Splitting rules", expanded=False):
        if parsed.single_language == "zh":
            chinese_max = st.number_input(
                "Maximum Chinese characters per segment",
                min_value=4,
                max_value=40,
                value=10,
                step=1,
                key=CHINESE_MAX_KEY,
            )
            dutch_max = 40
            normal_limit = int(chinese_max)
        else:
            chinese_max = 10
            dutch_max = st.number_input(
                "Maximum Latin-script characters per segment",
                min_value=10,
                max_value=100,
                value=40,
                step=1,
                key=DUTCH_MAX_KEY,
            )
            normal_limit = int(dutch_max)
        st.caption(
            "Punctuation is removed before measuring. The sole language is treated as the "
            "first output side and therefore uses the existing stricter 80% limit, even "
            "though no `|` is generated. Existing word-safe, grammatical-phrase protection, "
            "Chinese segmentation, balance, and minimum-fragment rules remain active. Current "
            f"normal/sole-language limits: {normal_limit}/"
            f"{derive_first_side_limit(normal_limit)}."
        )

    _, control_signature = _current_customization(
        parsed,
        None,
        fingerprint,
    )
    if GENERATED_OUTPUT_KEY not in st.session_state:
        if not _generate_conversion_output(
            parsed,
            None,
            fingerprint,
            initialize_saved_output=True,
            announce=False,
        ):
            st.error(st.session_state["generation_error"])
            st.stop()

    for warning in st.session_state.get("applied_conversion_warnings", ()):
        st.warning(warning)

    _render_regeneration_action(
        parsed=parsed,
        alignment_plan=None,
        fingerprint=fingerprint,
        control_signature=control_signature,
    )
    _render_editable_preview(
        subheader="3. Preview and download the TXT",
        help_text=(
            "The preview is editable. Check all `//` placements, then click UPDATE to save "
            "the exact text for download."
        ),
    )

    safe_stem = re.sub(
        r"[^\w\-]+",
        "_",
        Path(source_name).stem,
        flags=re.UNICODE,
    ).strip("_")
    output_name = f"{safe_stem or 'converted_lyrics'}_formatted.txt"
    _render_final_actions(
        output_name=output_name,
    )
    st.stop()

st.subheader("2. Match each Chinese section to its Dutch translation")
st.info(
    "The dashboard does not translate or judge meaning. The initial selections are only editable suggestions based on section type and order. You must choose the true translated counterpart(s)."
)

suggestions = suggest_manual_selections(parsed)
selections: dict[int, list[int]] = {}
for source in chinese_sections:
    candidate_indices = [candidate.original_index for candidate in dutch_sections]
    labels = {
        candidate.original_index: (
            f"{section_to_code[candidate.original_index]} — [{candidate.label}] — {candidate.lines[0]}"
        )
        for candidate in dutch_sections
    }
    default = [
        index for index in suggestions.get(source.original_index, []) if index in candidate_indices
    ]
    match_key = f"manual_match_{source.original_index}"
    selections[source.original_index] = st.multiselect(
        f"Dutch translation for [{source.label}] — {source.lines[0]}",
        options=candidate_indices,
        default=default if match_key not in st.session_state else None,
        format_func=lambda index, labels=labels: labels[index],
        key=match_key,
        help="Select more than one Dutch section when one Chinese section is translated across multiple sections.",
    )

covered_dutch = {
    section_index
    for selected_sections in selections.values()
    for section_index in selected_sections
}
missing_dutch = [section for section in dutch_sections if section.original_index not in covered_dutch]
if missing_dutch:
    st.warning(
        "These Dutch sections are not selected yet: "
        + ", ".join(f"{section_to_code[item.original_index]} [{item.label}]" for item in missing_dutch)
    )
else:
    st.success("Every Dutch section is included in at least one proposed section match.")

st.subheader("3. Confirm the exact translated line ranges")
st.markdown(
    "Use the same clear reference panels and vertical mapping rows as before. Edit only the "
    "**Chinese line(s)** ranges when a row needs to cover several consecutive lines, then drag "
    "the numbered Dutch cards into the **Dutch reference(s)** box beside the correct Chinese row. "
    "You never need to type a Dutch reference code."
)
st.caption(
    "The starting card placement is only a structural suggestion. You can move all cards back "
    "to the Dutch card pool and arrange them yourself. Final validation still requires every "
    "Chinese and Dutch line to be matched."
)

drag_layout_by_source: dict[
    int,
    tuple[
        tuple[tuple[int, ...], ...],
        tuple[tuple[tuple[int, int], ...], ...],
        tuple[tuple[int, int], ...],
    ],
] = {}
drag_errors: dict[int, LyricsDashboardError] = {}
chinese_range_specs_by_source: dict[int, tuple[str, ...]] = {}

for source in chinese_sections:
    source_index = source.original_index
    selected = tuple(selections.get(source_index, ()))
    selection_signature = (fingerprint, source_index, selected)
    selection_key = f"manual_drag_selection_{source_index}"
    editor_key = f"manual_lines_{source_index}"
    editor_base_key = f"manual_lines_base_{source_index}"
    editor_signature_key = f"manual_lines_signature_{source_index}"
    layout_state_key = f"manual_drag_layout_{source_index}"
    board_state_key = f"manual_drag_board_{source_index}"
    board_signature_key = f"manual_drag_board_signature_{source_index}"
    board_revision_key = f"manual_drag_revision_{source_index}"
    component_prefix = f"manual_drag_component_{source_index}_"

    if st.session_state.get(selection_key) != selection_signature:
        for state_key in list(st.session_state):
            if state_key.startswith(component_prefix):
                st.session_state.pop(state_key, None)
        for state_key in (
            editor_key,
            editor_base_key,
            editor_signature_key,
            layout_state_key,
            board_state_key,
            board_signature_key,
            board_revision_key,
        ):
            st.session_state.pop(state_key, None)
        st.session_state[selection_key] = selection_signature

    with st.expander(f"[{source.label}] exact line matching", expanded=True):
        if not selected:
            st.warning("Select at least one Dutch counterpart above before editing this section.")
            chinese_range_specs_by_source[source_index] = ()
            drag_layout_by_source[source_index] = ((), (), ())
            continue

        left, right = st.columns(2)
        with left:
            st.markdown("**Chinese source lines**")
            for line_number, line in enumerate(source.lines, start=1):
                st.text(f"{line_number}. {line}")
        with right:
            st.markdown("**Selected Dutch lines**")
            for section_index in selected:
                dutch = parsed.sections[section_index]
                code = section_to_code[section_index]
                st.markdown(f"**{code} — [{dutch.label}]**")
                for line_number, line in enumerate(dutch.lines, start=1):
                    st.text(f"{line_number}. {line}")

        suggested_groups = suggest_manual_line_groups(parsed, source_index, selected)
        default_range_rows = [
            {"Chinese line(s)": format_line_spec(group.source_line_indices)}
            for group in suggested_groups
        ]
        if st.session_state.get(editor_signature_key) != selection_signature:
            st.session_state.pop(editor_key, None)
            st.session_state.pop(editor_base_key, None)
            st.session_state[editor_signature_key] = selection_signature
        if editor_base_key not in st.session_state:
            st.session_state[editor_base_key] = default_range_rows

        st.markdown("**Mapping rows**")
        st.caption(
            "The Chinese ranges stay editable for one-to-many or many-to-one matches. "
            "Add or remove rows when needed; Dutch references are handled only by dragging below."
        )
        edited_ranges = st.data_editor(
            pd.DataFrame(
                st.session_state[editor_base_key],
                columns=["Chinese line(s)"],
            ),
            key=editor_key,
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "Chinese line(s)": st.column_config.TextColumn(
                    "Chinese line(s)",
                    help="One consecutive line or range, for example 1 or 2-3.",
                    required=True,
                ),
            },
        )
        range_specs = tuple(
            str(value).strip()
            for value in edited_ranges["Chinese line(s)"].fillna("").tolist()
            if str(value).strip()
        )
        chinese_range_specs_by_source[source_index] = range_specs

        try:
            source_groups = tuple(
                parse_line_spec(
                    spec,
                    len(source.lines),
                    f"[{source.label}] mapping row {row_number} Chinese lines",
                )
                for row_number, spec in enumerate(range_specs, start=1)
            )
            if not source_groups:
                raise PairingError(f"[{source.label}] has no Chinese mapping rows.")
            flattened_source_lines = tuple(
                line_index for group in source_groups for line_index in group
            )
            if flattened_source_lines != tuple(range(len(source.lines))):
                raise PairingError(
                    f"[{source.label}] Chinese mapping rows must cover every line once "
                    "and stay in order."
                )
        except LyricsDashboardError as exc:
            drag_errors[source_index] = exc
            drag_layout_by_source[source_index] = ((), (), ())
            st.error(str(exc))
            continue

        token_to_reference: dict[str, tuple[int, int]] = {}
        reference_to_token: dict[tuple[int, int], str] = {}
        for section_index in selected:
            dutch = parsed.sections[section_index]
            code = section_to_code[section_index]
            for line_index, line in enumerate(dutch.lines):
                token = f"{code}:{line_index + 1} — {line}"
                token_to_reference[token] = (section_index, line_index)
                reference_to_token[(section_index, line_index)] = token

        unassigned_header = f"Dutch card pool — [{source.label}]"
        target_headers = tuple(
            (
                f"Mapping row {position} — Chinese line(s) {format_line_spec(group)}\n"
                + "\n".join(
                    f"{line_index + 1}. {source.lines[line_index]}" for line_index in group
                )
            )
            for position, group in enumerate(source_groups, start=1)
        )
        suggested_assignments = suggest_drag_line_assignments(
            parsed,
            source_index,
            selected,
            source_groups,
        )
        default_containers: list[dict[str, object]] = [
            {"header": unassigned_header, "items": []}
        ]
        default_containers.extend(
            {
                "header": header,
                "items": [reference_to_token[reference] for reference in assignment],
            }
            for header, assignment in zip(target_headers, suggested_assignments)
        )
        initial_containers = default_containers
        preserved_layout = _normalize_drag_layout(
            st.session_state.get(layout_state_key)
        )
        if preserved_layout is not None:
            preserved_groups, preserved_assignments, preserved_unassigned = (
                preserved_layout
            )
            preserved_references = tuple(
                reference
                for assignment in preserved_assignments
                for reference in assignment
            ) + tuple(preserved_unassigned)
            if (
                preserved_groups == source_groups
                and len(preserved_assignments) == len(source_groups)
                and len(preserved_references) == len(set(preserved_references))
                and set(preserved_references) == set(reference_to_token)
            ):
                initial_containers = [
                    {
                        "header": unassigned_header,
                        "items": [
                            reference_to_token[reference]
                            for reference in preserved_unassigned
                        ],
                    }
                ]
                initial_containers.extend(
                    {
                        "header": header,
                        "items": [
                            reference_to_token[reference]
                            for reference in assignment
                        ],
                    }
                    for header, assignment in zip(
                        target_headers,
                        preserved_assignments,
                    )
                )
        pool_containers: list[dict[str, object]] = [
            {
                "header": unassigned_header,
                "items": list(reference_to_token.values()),
            }
        ]
        pool_containers.extend(
            {"header": header, "items": []}
            for header in target_headers
        )

        board_signature = (
            "vertical_mapping_rows_v3",
            selection_signature,
            source_groups,
            tuple(section_to_code.items()),
        )
        if st.session_state.get(board_signature_key) != board_signature:
            st.session_state[board_state_key] = initial_containers
            st.session_state[board_signature_key] = board_signature
            st.session_state[board_revision_key] = (
                int(st.session_state.get(board_revision_key, -1)) + 1
            )

        suggested_button, pool_button, drag_help = st.columns([1, 1, 2])
        with suggested_button:
            if st.button("Use suggested placement", key=f"manual_drag_reset_{source_index}"):
                st.session_state[board_state_key] = default_containers
                st.session_state[board_revision_key] = (
                    int(st.session_state.get(board_revision_key, 0)) + 1
                )
        with pool_button:
            if st.button("Move all cards to pool", key=f"manual_drag_pool_{source_index}"):
                st.session_state[board_state_key] = pool_containers
                st.session_state[board_revision_key] = (
                    int(st.session_state.get(board_revision_key, 0)) + 1
                )
        with drag_help:
            st.caption(
                "Drag cards from the pool or between the right-hand Dutch boxes. "
                "Keep each Dutch section in its original line order."
            )

        st.markdown("**Drag Dutch reference cards into the mapping rows**")
        component_key = (
            f"{component_prefix}{int(st.session_state.get(board_revision_key, 0))}"
        )
        sorted_containers = sort_items(
            st.session_state[board_state_key],
            multi_containers=True,
            direction="vertical",
            custom_style=DRAG_BOARD_STYLE,
            key=component_key,
        )
        st.session_state[board_state_key] = sorted_containers

        try:
            unassigned, assignments = decode_drag_containers(
                sorted_containers,
                unassigned_header=unassigned_header,
                target_headers=target_headers,
                token_to_reference=token_to_reference,
            )
        except LyricsDashboardError as exc:
            drag_errors[source_index] = exc
            unassigned = tuple(token_to_reference.values())
            assignments = tuple(() for _group in source_groups)

        drag_layout_by_source[source_index] = (
            source_groups,
            assignments,
            unassigned,
        )
        st.session_state[layout_state_key] = drag_layout_by_source[source_index]

        empty_targets = [
            format_line_spec(group)
            for group, assignment in zip(source_groups, assignments)
            if not assignment
        ]
        if empty_targets:
            st.warning(
                "These Chinese mapping rows still need a Dutch card: "
                + ", ".join(empty_targets)
                + "."
            )
        if unassigned:
            st.caption(
                f"{len(unassigned)} Dutch card(s) remain in the pool on this board. "
                "That is valid only when those lines are assigned on another Chinese board."
            )

alignment_input_signature = repr(
    (
        tuple((key, tuple(value)) for key, value in sorted(selections.items())),
        tuple(sorted(chinese_range_specs_by_source.items())),
        tuple(
            (
                source_index,
                source_groups,
                assignments,
            )
            for source_index, (source_groups, assignments, _unassigned) in sorted(
                drag_layout_by_source.items()
            )
        ),
    )
)

if st.button("Validate these manual matches", type="primary", width="stretch"):
    try:
        if drag_errors:
            raise next(iter(drag_errors.values()))
        line_groups_by_chinese = {
            source.original_index: build_drag_line_groups(
                parsed,
                source.original_index,
                selections.get(source.original_index, ()),
                drag_layout_by_source[source.original_index][0],
                drag_layout_by_source[source.original_index][1],
                drag_layout_by_source[source.original_index][2],
            )
            for source in chinese_sections
        }
        st.session_state["alignment_plan"] = build_exact_manual_plan(
            parsed,
            selections,
            line_groups_by_chinese,
        )
        st.session_state["alignment_fingerprint"] = fingerprint
        st.session_state["alignment_input_signature"] = alignment_input_signature
    except LyricsDashboardError as exc:
        st.error(str(exc))

alignment_plan = st.session_state.get("alignment_plan")
if (
    st.session_state.pop(PRESERVED_ALIGNMENT_KEY, False)
    and isinstance(alignment_plan, AlignmentPlan)
):
    st.session_state["alignment_fingerprint"] = fingerprint
    st.session_state["alignment_input_signature"] = alignment_input_signature

alignment_is_current = (
    isinstance(alignment_plan, AlignmentPlan)
    and st.session_state.get("alignment_fingerprint") == fingerprint
    and st.session_state.get("alignment_input_signature") == alignment_input_signature
)
st.session_state[ALIGNMENT_CURRENT_KEY] = alignment_is_current
if not alignment_is_current:
    alignment_plan = None

if alignment_plan is None:
    st.info("Arrange the Dutch cards and validate the manual line matches to continue.")
    st.stop()

for warning in alignment_plan.warnings:
    st.warning(warning)

alignment_rows = []
for alignment in alignment_plan.alignments:
    source = parsed.sections[alignment.source_section_index]
    counterparts = [parsed.sections[index] for index in alignment.counterpart_section_indices]
    alignment_rows.append(
        {
            "Output section": f"[{source.label}]",
            "Language": "Chinese" if source.language == "zh" else "Dutch",
            "Matched translation": ", ".join(f"[{item.label}]" for item in counterparts),
            "Output line groups": len(alignment.aligned_lines),
        }
    )

st.success("Every detected section and line now has a reciprocal manual translation mapping.")
with st.expander("Review all exact Chinese–Dutch pairs", expanded=True):
    st.dataframe(pd.DataFrame(alignment_rows), width="stretch", hide_index=True)

    exact_pair_rows = []
    for alignment in alignment_plan.alignments:
        source = parsed.sections[alignment.source_section_index]
        for group_number, line_group in enumerate(alignment.aligned_lines, start=1):
            source_separator = "" if source.language == "zh" else " "
            source_text_for_pair = source_separator.join(
                source.lines[index].strip() for index in line_group.source_line_indices
            )
            translated_pieces = []
            translated_language = None
            for reference in line_group.translation_references:
                translated_section = parsed.sections[reference.section_index]
                translated_language = translated_language or translated_section.language
                translated_separator = "" if translated_section.language == "zh" else " "
                translated_pieces.append(
                    translated_separator.join(
                        translated_section.lines[index].strip()
                        for index in reference.line_indices
                    )
                )
            outer_separator = "" if translated_language == "zh" else " "
            translated_text = outer_separator.join(translated_pieces)

            if source.language == "zh":
                chinese_text, dutch_text = source_text_for_pair, translated_text
            else:
                dutch_text, chinese_text = source_text_for_pair, translated_text

            exact_pair_rows.append(
                {
                    "Output section": f"[{source.label}]",
                    "Pair": group_number,
                    "Chinese": chinese_text,
                    "Dutch": dutch_text,
                }
            )

    st.dataframe(pd.DataFrame(exact_pair_rows), width="stretch", hide_index=True)

st.subheader("4. Choose the language-order switch")
option_values: list[int | None] = [None] + [section.original_index for section in parsed.sections]
option_labels: dict[int | None, str] = {None: "Never — Chinese first throughout"}
for section in parsed.sections:
    language_label = "Chinese section" if section.language == "zh" else "Dutch section"
    option_labels[section.original_index] = f"From [{section.label}] ({language_label})"

first_dutch_index = next(
    (section.original_index for section in parsed.sections if section.language == "nl"),
    None,
)
default_position = option_values.index(first_dutch_index) if first_dutch_index in option_values else 0
selected_switch = st.selectbox(
    "Switch to Dutch first starting at",
    options=option_values,
    index=default_position,
    format_func=lambda value: option_labels[value],
    help="The selected section and every section after it use Dutch|Chinese. Earlier sections use Chinese|Dutch.",
    key=SWITCH_INDEX_KEY,
)

with st.expander("Splitting rules", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        chinese_max = st.number_input(
            "Maximum Chinese characters per segment",
            min_value=4,
            max_value=40,
            value=10,
            step=1,
            key=CHINESE_MAX_KEY,
        )
    with c2:
        dutch_max = st.number_input(
            "Maximum Dutch characters per segment",
            min_value=10,
            max_value=100,
            value=40,
            step=1,
            key=DUTCH_MAX_KEY,
        )
    title_separator_choice = st.selectbox(
        "Title language separator",
        options=["Space — matches the example", "| — same as lyric lines"],
        index=0,
        key=TITLE_SEPARATOR_KEY,
    )
    st.caption(
        "Punctuation is removed before measuring. Long Chinese uses the nearest balanced local "
        "word-segmentation boundary; long Dutch uses the nearest whitespace boundary. Each "
        "language receives at most one `//` per output row. The final language before `|` uses "
        "80% of its normal limit, so it splits slightly sooner; the language after `|` keeps its "
        f"normal limit. Current before/after limits: Chinese "
        f"{derive_first_side_limit(int(chinese_max))}/{int(chinese_max)}, Dutch "
        f"{derive_first_side_limit(int(dutch_max))}/{int(dutch_max)}."
    )

_, control_signature = _current_customization(
    parsed,
    alignment_plan,
    fingerprint,
)
if GENERATED_OUTPUT_KEY not in st.session_state:
    if not _generate_conversion_output(
        parsed,
        alignment_plan,
        fingerprint,
        initialize_saved_output=True,
        announce=False,
    ):
        st.error(st.session_state["generation_error"])
        st.stop()

for warning in st.session_state.get("applied_conversion_warnings", ()):
    st.warning(warning)

_render_regeneration_action(
    parsed=parsed,
    alignment_plan=alignment_plan,
    fingerprint=fingerprint,
    control_signature=control_signature,
)
_render_editable_preview(
    subheader="5. Preview and download the TXT",
    help_text=(
        "The preview is editable. Check the translation pairs and all `//` placements, then "
        "click UPDATE to save the exact text for download."
    ),
)

safe_stem = re.sub(r"[^\w\-]+", "_", Path(source_name).stem, flags=re.UNICODE).strip("_")
output_name = f"{safe_stem or 'converted_lyrics'}_formatted.txt"

_render_final_actions(
    output_name=output_name,
)
