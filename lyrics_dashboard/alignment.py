from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

from .errors import PairingError
from .models import (
    AlignedLine,
    AlignmentPlan,
    ParsedLyrics,
    SectionAlignment,
    TranslationReference,
)

_LINE_RANGE_RE = re.compile(r"^(?P<start>\d+)(?:\s*-\s*(?P<end>\d+))?$")
_REFERENCE_RE = re.compile(r"^(?P<code>D\d+)\s*:\s*(?P<lines>.+)$", re.IGNORECASE)


def _validate_line_indices(indices: tuple[int, ...], line_count: int, context: str) -> None:
    if not indices:
        raise PairingError(f"{context} has no line indices.")
    if tuple(sorted(set(indices))) != indices:
        raise PairingError(f"{context} line indices must be unique and ascending.")
    if indices[0] < 0 or indices[-1] >= line_count:
        raise PairingError(f"{context} contains an out-of-range line index.")
    if indices != tuple(range(indices[0], indices[-1] + 1)):
        raise PairingError(f"{context} must reference consecutive lines.")


def parse_line_spec(spec: str, line_count: int, context: str) -> tuple[int, ...]:
    """Parse a one-based line range such as ``1`` or ``2-4``."""
    cleaned = str(spec).strip()
    match = _LINE_RANGE_RE.fullmatch(cleaned)
    if not match:
        raise PairingError(f"{context} must be a line number or range such as 1 or 2-4.")

    start = int(match.group("start"))
    end = int(match.group("end") or start)
    if start < 1 or end < start or end > line_count:
        raise PairingError(f"{context} is outside the available lines 1-{line_count}.")
    return tuple(range(start - 1, end))


def format_line_spec(indices: Sequence[int]) -> str:
    if not indices:
        return ""
    start = indices[0] + 1
    end = indices[-1] + 1
    return str(start) if start == end else f"{start}-{end}"


def parse_translation_spec(
    spec: str,
    *,
    parsed: ParsedLyrics,
    code_to_section: Mapping[str, int],
    allowed_sections: Iterable[int],
    context: str,
) -> tuple[TranslationReference, ...]:
    """Parse ``D1:1-2; D2:1`` into exact Dutch line references."""
    allowed = set(allowed_sections)
    parts = [part.strip() for part in str(spec).split(";") if part.strip()]
    if not parts:
        raise PairingError(f"{context} needs at least one Dutch line reference.")

    references: list[TranslationReference] = []
    seen_sections: set[int] = set()
    for part in parts:
        match = _REFERENCE_RE.fullmatch(part)
        if not match:
            raise PairingError(
                f"{context} has an invalid Dutch reference '{part}'. Use D1:1 or D1:1-2; D2:1."
            )
        code = match.group("code").upper()
        if code not in code_to_section:
            raise PairingError(f"{context} uses unknown Dutch section code {code}.")
        section_index = code_to_section[code]
        if section_index not in allowed:
            raise PairingError(f"{context} uses {code}, but that section was not selected as a translation.")
        if section_index in seen_sections:
            raise PairingError(f"{context} repeats {code}. Combine its lines into one consecutive range.")
        seen_sections.add(section_index)

        section = parsed.sections[section_index]
        if section.language != "nl":
            raise PairingError(f"{context} can only reference Dutch sections.")
        line_indices = parse_line_spec(
            match.group("lines"),
            len(section.lines),
            f"{context} {code}",
        )
        references.append(
            TranslationReference(section_index=section_index, line_indices=line_indices)
        )
    return tuple(references)


def parse_manual_rows(
    parsed: ParsedLyrics,
    source_section_index: int,
    selected_dutch_sections: Sequence[int],
    rows: Sequence[Mapping[str, object]],
    code_to_section: Mapping[str, int],
) -> tuple[AlignedLine, ...]:
    """Convert editable dashboard rows into exact Chinese-to-Dutch line groups."""
    source = parsed.sections[source_section_index]
    if source.language != "zh":
        raise PairingError("Exact manual rows must be entered for Chinese source sections.")

    groups: list[AlignedLine] = []
    for row_number, row in enumerate(rows, start=1):
        source_spec = str(row.get("Chinese line(s)", "")).strip()
        translation_spec = str(row.get("Dutch reference(s)", "")).strip()
        if not source_spec and not translation_spec:
            continue
        if not source_spec or not translation_spec:
            raise PairingError(
                f"[{source.label}] row {row_number} must contain both a Chinese range and Dutch reference."
            )
        source_indices = parse_line_spec(
            source_spec,
            len(source.lines),
            f"[{source.label}] row {row_number} Chinese lines",
        )
        references = parse_translation_spec(
            translation_spec,
            parsed=parsed,
            code_to_section=code_to_section,
            allowed_sections=selected_dutch_sections,
            context=f"[{source.label}] row {row_number}",
        )
        groups.append(
            AlignedLine(
                source_line_indices=source_indices,
                translation_references=references,
            )
        )

    if not groups:
        raise PairingError(f"[{source.label}] has no manual line matches.")
    return tuple(groups)


def validate_alignment_plan(parsed: ParsedLyrics, plan: AlignmentPlan) -> None:
    expected_sections = set(range(len(parsed.sections)))
    actual_sections = {alignment.source_section_index for alignment in plan.alignments}
    if len(actual_sections) != len(plan.alignments):
        raise PairingError("Alignment contains duplicate source-section entries.")
    if actual_sections != expected_sections:
        missing = sorted(expected_sections - actual_sections)
        extra = sorted(actual_sections - expected_sections)
        raise PairingError(f"Alignment must cover every section. Missing={missing}, extra={extra}.")

    for alignment in plan.alignments:
        source = parsed.sections[alignment.source_section_index]
        if not alignment.counterpart_section_indices:
            raise PairingError(f"[{source.label}] does not have a confirmed translation match.")
        if not alignment.aligned_lines:
            raise PairingError(f"[{source.label}] does not have line-level translation matches.")

        counterpart_set = set(alignment.counterpart_section_indices)
        for counterpart_index in counterpart_set:
            if counterpart_index < 0 or counterpart_index >= len(parsed.sections):
                raise PairingError(f"[{source.label}] references an invalid counterpart section.")
            counterpart = parsed.sections[counterpart_index]
            if counterpart.language == source.language:
                raise PairingError(
                    f"[{source.label}] is matched to [{counterpart.label}] in the same language."
                )

        covered_source_lines: list[int] = []
        previous_source_end = -1
        previous_translation_positions: dict[int, int] = defaultdict(lambda: -1)
        used_counterparts: set[int] = set()

        for group_index, line_group in enumerate(alignment.aligned_lines, start=1):
            source_indices = line_group.source_line_indices
            _validate_line_indices(
                source_indices,
                len(source.lines),
                f"[{source.label}] group {group_index} source",
            )
            if source_indices[0] <= previous_source_end:
                raise PairingError(f"[{source.label}] source line groups overlap or are out of order.")
            previous_source_end = source_indices[-1]
            covered_source_lines.extend(source_indices)

            if not line_group.translation_references:
                raise PairingError(f"[{source.label}] group {group_index} has no translated lyric reference.")

            for reference in line_group.translation_references:
                if reference.section_index not in counterpart_set:
                    raise PairingError(
                        f"[{source.label}] group {group_index} uses an undeclared counterpart section."
                    )
                translated = parsed.sections[reference.section_index]
                _validate_line_indices(
                    reference.line_indices,
                    len(translated.lines),
                    f"[{source.label}] group {group_index} translation",
                )
                if reference.line_indices[0] <= previous_translation_positions[reference.section_index]:
                    raise PairingError(
                        f"[{source.label}] translation lines overlap or are out of order within [{translated.label}]."
                    )
                previous_translation_positions[reference.section_index] = reference.line_indices[-1]
                used_counterparts.add(reference.section_index)

        if covered_source_lines != list(range(len(source.lines))):
            raise PairingError(
                f"[{source.label}] must cover every source line exactly once; got one-based lines "
                f"{[index + 1 for index in covered_source_lines]}."
            )
        if used_counterparts != counterpart_set:
            raise PairingError(
                f"[{source.label}] declares counterpart sections that are not used in its line matches."
            )

    alignment_by_source = {item.source_section_index: item for item in plan.alignments}
    for alignment in plan.alignments:
        for counterpart_index in alignment.counterpart_section_indices:
            reverse = alignment_by_source[counterpart_index]
            if alignment.source_section_index not in reverse.counterpart_section_indices:
                source = parsed.sections[alignment.source_section_index]
                counterpart = parsed.sections[counterpart_index]
                raise PairingError(
                    f"Section matching is not reciprocal: [{source.label}] uses [{counterpart.label}], "
                    f"but the reverse match is missing."
                )

    directed_edges: set[tuple[int, int, int, int]] = set()
    for alignment in plan.alignments:
        for line_group in alignment.aligned_lines:
            for source_line_index in line_group.source_line_indices:
                for reference in line_group.translation_references:
                    for translation_line_index in reference.line_indices:
                        directed_edges.add(
                            (
                                alignment.source_section_index,
                                source_line_index,
                                reference.section_index,
                                translation_line_index,
                            )
                        )
    missing_reverse = [
        edge
        for edge in directed_edges
        if (edge[2], edge[3], edge[0], edge[1]) not in directed_edges
    ]
    if missing_reverse:
        source_section, source_line, target_section, target_line = missing_reverse[0]
        raise PairingError(
            "Line matching is not reciprocal for "
            f"[{parsed.sections[source_section].label}] line {source_line + 1} and "
            f"[{parsed.sections[target_section].label}] line {target_line + 1}."
        )


def _partition_indices(count: int, groups: int) -> list[tuple[int, ...]]:
    if groups <= 0 or groups > count:
        raise ValueError("groups must be between 1 and count")
    boundaries = [round(index * count / groups) for index in range(groups + 1)]
    partitions: list[tuple[int, ...]] = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            end = start + 1
        partitions.append(tuple(range(start, min(end, count))))
    return partitions


def suggest_manual_selections(parsed: ParsedLyrics) -> dict[int, list[int]]:
    """Suggest editable Dutch counterparts for each Chinese section by kind and order."""
    suggestions: dict[int, list[int]] = {}
    chinese_sections = [section for section in parsed.sections if section.language == "zh"]
    dutch_sections = [section for section in parsed.sections if section.language == "nl"]

    for source in chinese_sections:
        same_kind = [section for section in dutch_sections if section.kind == source.kind]
        pool = same_kind or dutch_sections
        same_kind_sources = [section for section in chinese_sections if section.kind == source.kind]
        source_rank = same_kind_sources.index(source) if source in same_kind_sources else 0
        target_position = 0.0 if len(same_kind_sources) <= 1 else source_rank / (len(same_kind_sources) - 1)
        candidate_rank = 0 if len(pool) <= 1 else round(target_position * (len(pool) - 1))
        suggestions[source.original_index] = [pool[candidate_rank].original_index]
    return suggestions


def suggest_manual_line_groups(
    parsed: ParsedLyrics,
    source_section_index: int,
    selected_dutch_sections: Sequence[int],
) -> tuple[AlignedLine, ...]:
    """Create a non-semantic sequential starting suggestion for the editable line table."""
    source = parsed.sections[source_section_index]
    if source.language != "zh":
        raise PairingError("Line suggestions are generated from Chinese source sections.")
    if not selected_dutch_sections:
        return ()

    translation_refs_flat: list[tuple[int, int]] = []
    for counterpart_index in selected_dutch_sections:
        counterpart = parsed.sections[counterpart_index]
        if counterpart.language != "nl":
            raise PairingError(f"[{source.label}] can only be matched to Dutch sections.")
        translation_refs_flat.extend(
            (counterpart_index, line_index) for line_index in range(len(counterpart.lines))
        )

    group_count = min(len(source.lines), len(translation_refs_flat))
    source_groups = _partition_indices(len(source.lines), group_count)
    translation_groups = _partition_indices(len(translation_refs_flat), group_count)

    groups: list[AlignedLine] = []
    for source_indices, translation_flat_indices in zip(source_groups, translation_groups):
        references_by_section: dict[int, list[int]] = {}
        order: list[int] = []
        for flat_index in translation_flat_indices:
            section_index, line_index = translation_refs_flat[flat_index]
            if section_index not in references_by_section:
                references_by_section[section_index] = []
                order.append(section_index)
            references_by_section[section_index].append(line_index)
        groups.append(
            AlignedLine(
                source_line_indices=source_indices,
                translation_references=tuple(
                    TranslationReference(
                        section_index=section_index,
                        line_indices=tuple(references_by_section[section_index]),
                    )
                    for section_index in order
                ),
            )
        )
    return tuple(groups)


def manual_groups_to_rows(
    groups: Sequence[AlignedLine],
    section_to_code: Mapping[int, str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for group in groups:
        references = "; ".join(
            f"{section_to_code[reference.section_index]}:{format_line_spec(reference.line_indices)}"
            for reference in group.translation_references
        )
        rows.append(
            {
                "Chinese line(s)": format_line_spec(group.source_line_indices),
                "Dutch reference(s)": references,
            }
        )
    return rows


def _build_reverse_alignments(
    parsed: ParsedLyrics,
    chinese_alignments: Sequence[SectionAlignment],
) -> list[SectionAlignment]:
    dutch_sections = [section for section in parsed.sections if section.language == "nl"]
    reverse_alignments: list[SectionAlignment] = []

    # For every Dutch line, collect the exact Chinese section/line indices that point to it.
    edges_by_dutch_line: dict[tuple[int, int], dict[int, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for alignment in chinese_alignments:
        for group in alignment.aligned_lines:
            for reference in group.translation_references:
                for dutch_line_index in reference.line_indices:
                    edges_by_dutch_line[(reference.section_index, dutch_line_index)][
                        alignment.source_section_index
                    ].update(group.source_line_indices)

    for dutch in dutch_sections:
        signatures: list[tuple[tuple[int, tuple[int, ...]], ...]] = []
        for line_index in range(len(dutch.lines)):
            connected = edges_by_dutch_line.get((dutch.original_index, line_index), {})
            if not connected:
                raise PairingError(
                    f"Dutch [{dutch.label}] line {line_index + 1} is not used by any Chinese translation match."
                )
            signature = tuple(
                (section_index, tuple(sorted(line_indices)))
                for section_index, line_indices in sorted(connected.items())
            )
            for section_index, line_indices in signature:
                _validate_line_indices(
                    line_indices,
                    len(parsed.sections[section_index].lines),
                    f"Reverse match for [{dutch.label}] line {line_index + 1}",
                )
            signatures.append(signature)

        groups: list[AlignedLine] = []
        start = 0
        while start < len(signatures):
            end = start + 1
            while end < len(signatures) and signatures[end] == signatures[start]:
                end += 1
            signature = signatures[start]
            groups.append(
                AlignedLine(
                    source_line_indices=tuple(range(start, end)),
                    translation_references=tuple(
                        TranslationReference(section_index=section_index, line_indices=line_indices)
                        for section_index, line_indices in signature
                    ),
                )
            )
            start = end

        counterpart_indices = tuple(
            dict.fromkeys(
                reference.section_index
                for group in groups
                for reference in group.translation_references
            )
        )
        reverse_alignments.append(
            SectionAlignment(
                source_section_index=dutch.original_index,
                counterpart_section_indices=counterpart_indices,
                aligned_lines=tuple(groups),
                confidence="manual",
                note="Automatically derived reciprocal mapping from the confirmed Chinese line matches.",
            )
        )
    return reverse_alignments


def build_exact_manual_plan(
    parsed: ParsedLyrics,
    selections: Mapping[int, Sequence[int]],
    line_groups_by_chinese: Mapping[int, Sequence[AlignedLine]],
) -> AlignmentPlan:
    """Build a fully manual, reciprocal alignment without any paid API or AI model."""
    chinese_sections = [section for section in parsed.sections if section.language == "zh"]
    dutch_sections = [section for section in parsed.sections if section.language == "nl"]

    chinese_alignments: list[SectionAlignment] = []
    selected_dutch_coverage: set[int] = set()
    for source in chinese_sections:
        selected = tuple(dict.fromkeys(int(index) for index in selections.get(source.original_index, ())))
        if not selected:
            raise PairingError(f"Choose at least one Dutch translated section for [{source.label}].")
        for counterpart_index in selected:
            if counterpart_index < 0 or counterpart_index >= len(parsed.sections):
                raise PairingError(f"[{source.label}] has an invalid selected counterpart.")
            counterpart = parsed.sections[counterpart_index]
            if counterpart.language != "nl":
                raise PairingError(f"[{source.label}] can only be matched to Dutch sections.")
        selected_dutch_coverage.update(selected)

        groups = tuple(line_groups_by_chinese.get(source.original_index, ()))
        if not groups:
            raise PairingError(f"[{source.label}] has no confirmed line matches.")
        chinese_alignments.append(
            SectionAlignment(
                source_section_index=source.original_index,
                counterpart_section_indices=selected,
                aligned_lines=groups,
                confidence="manual",
                note="User-confirmed Chinese and Dutch section and line ranges.",
            )
        )

    missing_dutch_sections = [
        section.label for section in dutch_sections if section.original_index not in selected_dutch_coverage
    ]
    if missing_dutch_sections:
        raise PairingError(
            "Every Dutch section must be selected by at least one Chinese section. Missing: "
            + ", ".join(f"[{label}]" for label in missing_dutch_sections)
        )

    reverse_alignments = _build_reverse_alignments(parsed, chinese_alignments)
    all_alignments = sorted(
        [*chinese_alignments, *reverse_alignments],
        key=lambda alignment: alignment.source_section_index,
    )
    plan = AlignmentPlan(
        alignments=tuple(all_alignments),
        method="manual",
        warnings=(
            "All translation matches are manual. The dashboard does not translate or check meaning automatically; review every exact pair before downloading.",
        ),
    )
    validate_alignment_plan(parsed, plan)
    return plan
