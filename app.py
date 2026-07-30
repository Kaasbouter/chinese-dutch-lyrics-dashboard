from __future__ import annotations

import hashlib
import re
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
from lyrics_dashboard.parser import parse_lyrics

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
if st.session_state.get("file_fingerprint") != fingerprint:
    for state_key in list(st.session_state):
        if state_key in {
            "alignment_plan",
            "alignment_fingerprint",
            "alignment_input_signature",
            "control_signature",
            "edited_output",
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
    st.session_state["file_fingerprint"] = fingerprint

try:
    source_text = extract_text(uploaded_file.name, file_bytes)
    parsed = parse_lyrics(source_text)
except LyricsDashboardError as exc:
    st.error(str(exc))
    st.stop()

for warning in parsed.warnings:
    st.warning(warning)

chinese_sections = [section for section in parsed.sections if section.language == "zh"]
dutch_sections = [section for section in parsed.sections if section.language == "nl"]
single_language_mode = parsed.mode == "single-language"
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

section_rows = [
    {
        "Index": section.original_index,
        "Section": f"[{section.label}]",
        "Language": (
            "Chinese"
            if section.language == "zh"
            else (
                "Dutch/English or Latin script"
                if single_language_mode
                else "Dutch"
            )
        ),
        "Lines": len(section.lines),
        "Opening text": section.lines[0],
    }
    for section in parsed.sections
]
with st.expander("Detected source sections", expanded=False):
    st.dataframe(pd.DataFrame(section_rows), width="stretch", hide_index=True)

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

    settings = ConversionSettings(
        chinese_max_length=int(chinese_max),
        dutch_max_length=int(dutch_max),
    )
    conversion_warnings: list[str] = []
    try:
        generated = convert_lyrics(
            parsed,
            None,
            settings,
            warnings=conversion_warnings,
        )
    except LyricsDashboardError as exc:
        st.error(str(exc))
        st.stop()

    for warning in conversion_warnings:
        st.warning(warning)

    control_signature = (
        fingerprint,
        parsed.mode,
        parsed.single_language,
        int(chinese_max),
        int(dutch_max),
    )
    if st.session_state.get("control_signature") != control_signature:
        st.session_state["control_signature"] = control_signature
        st.session_state["edited_output"] = generated

    st.subheader("3. Preview and download the TXT")
    st.caption(
        "The preview is editable. Check all `//` placements before downloading."
    )
    final_text = st.text_area(
        "Converted lyrics",
        key="edited_output",
        height=560,
        label_visibility="collapsed",
    )

    safe_stem = re.sub(
        r"[^\w\-]+",
        "_",
        Path(uploaded_file.name).stem,
        flags=re.UNICODE,
    ).strip("_")
    output_name = f"{safe_stem or 'converted_lyrics'}_formatted.txt"
    st.download_button(
        "Download final TXT",
        data=encode_utf8_txt(final_text),
        file_name=output_name,
        mime="text/plain; charset=utf-8",
        type="primary",
        width="stretch",
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
    selections[source.original_index] = st.multiselect(
        f"Dutch translation for [{source.label}] — {source.lines[0]}",
        options=candidate_indices,
        default=default,
        format_func=lambda index, labels=labels: labels[index],
        key=f"manual_match_{source.original_index}",
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
    editor_signature_key = f"manual_lines_signature_{source_index}"
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
            editor_signature_key,
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
            st.session_state[editor_signature_key] = selection_signature

        st.markdown("**Mapping rows**")
        st.caption(
            "The Chinese ranges stay editable for one-to-many or many-to-one matches. "
            "Add or remove rows when needed; Dutch references are handled only by dragging below."
        )
        edited_ranges = st.data_editor(
            pd.DataFrame(default_range_rows, columns=["Chinese line(s)"]),
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

        board_signature = ("vertical_mapping_rows_v2", selection_signature, source_groups)
        if st.session_state.get(board_signature_key) != board_signature:
            st.session_state[board_state_key] = default_containers
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
    st.session_state.get("alignment_fingerprint") != fingerprint
    or st.session_state.get("alignment_input_signature") != alignment_input_signature
):
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
        )
    with c2:
        dutch_max = st.number_input(
            "Maximum Dutch characters per segment",
            min_value=10,
            max_value=100,
            value=40,
            step=1,
        )
    title_separator_choice = st.selectbox(
        "Title language separator",
        options=["Space — matches the example", "| — same as lyric lines"],
        index=0,
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

settings = ConversionSettings(
    switch_index=selected_switch,
    chinese_max_length=int(chinese_max),
    dutch_max_length=int(dutch_max),
    title_separator="|" if title_separator_choice.startswith("|") else " ",
)

conversion_warnings: list[str] = []
try:
    generated = convert_lyrics(
        parsed,
        alignment_plan,
        settings,
        warnings=conversion_warnings,
    )
except LyricsDashboardError as exc:
    st.error(str(exc))
    st.stop()

for warning in conversion_warnings:
    st.warning(warning)

control_signature = (
    fingerprint,
    repr(alignment_plan),
    selected_switch,
    int(chinese_max),
    int(dutch_max),
    title_separator_choice,
)
if st.session_state.get("control_signature") != control_signature:
    st.session_state["control_signature"] = control_signature
    st.session_state["edited_output"] = generated

st.subheader("5. Preview and download the TXT")
st.caption(
    "The preview is editable. Check the translation pairs and all `//` placements before downloading."
)
final_text = st.text_area(
    "Converted lyrics",
    key="edited_output",
    height=560,
    label_visibility="collapsed",
)

safe_stem = re.sub(r"[^\w\-]+", "_", Path(uploaded_file.name).stem, flags=re.UNICODE).strip("_")
output_name = f"{safe_stem or 'converted_lyrics'}_formatted.txt"

st.download_button(
    "Download final TXT",
    data=encode_utf8_txt(final_text),
    file_name=output_name,
    mime="text/plain; charset=utf-8",
    type="primary",
    width="stretch",
)
