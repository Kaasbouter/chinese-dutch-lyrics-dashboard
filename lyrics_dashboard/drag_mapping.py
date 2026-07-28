from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

from .errors import PairingError
from .models import AlignedLine, ParsedLyrics, TranslationReference

DutchLineToken = tuple[int, int]


def make_chinese_line_groups(
    line_count: int,
    join_with_previous: Sequence[bool],
) -> tuple[tuple[int, ...], ...]:
    """Build consecutive Chinese line groups from join/split choices."""
    if line_count < 1:
        raise ValueError("A Chinese section must contain at least one line.")
    if len(join_with_previous) != line_count - 1:
        raise ValueError("One join choice is required for every Chinese line after the first.")

    groups: list[list[int]] = [[0]]
    for line_index, join in enumerate(join_with_previous, start=1):
        if join:
            groups[-1].append(line_index)
        else:
            groups.append([line_index])
    return tuple(tuple(group) for group in groups)


def _validate_source_groups(
    parsed: ParsedLyrics,
    source_section_index: int,
    source_line_groups: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    if source_section_index < 0 or source_section_index >= len(parsed.sections):
        raise PairingError("The drag board references an invalid Chinese section.")
    source = parsed.sections[source_section_index]
    if source.language != "zh":
        raise PairingError("Drag boards can only be built for Chinese source sections.")

    groups = tuple(tuple(int(index) for index in group) for group in source_line_groups)
    if not groups:
        raise PairingError(f"[{source.label}] has no Chinese line groups.")

    flattened: list[int] = []
    for group_number, group in enumerate(groups, start=1):
        if not group:
            raise PairingError(f"[{source.label}] Chinese group {group_number} is empty.")
        if group != tuple(range(group[0], group[-1] + 1)):
            raise PairingError(
                f"[{source.label}] Chinese group {group_number} must contain consecutive lines."
            )
        flattened.extend(group)

    if flattened != list(range(len(source.lines))):
        raise PairingError(
            f"[{source.label}] Chinese groups must cover every line once and stay in order."
        )
    return groups


def _selected_dutch_tokens(
    parsed: ParsedLyrics,
    source_section_index: int,
    selected_dutch_sections: Sequence[int],
) -> tuple[tuple[int, ...], tuple[DutchLineToken, ...]]:
    source = parsed.sections[source_section_index]
    selected = tuple(dict.fromkeys(int(index) for index in selected_dutch_sections))
    if not selected:
        raise PairingError(f"[{source.label}] needs at least one selected Dutch section.")

    tokens: list[DutchLineToken] = []
    for section_index in selected:
        if section_index < 0 or section_index >= len(parsed.sections):
            raise PairingError(f"[{source.label}] selects an invalid Dutch section.")
        section = parsed.sections[section_index]
        if section.language != "nl":
            raise PairingError(f"[{source.label}] can only be matched to Dutch sections.")
        tokens.extend((section_index, line_index) for line_index in range(len(section.lines)))
    return selected, tuple(tokens)


def suggest_drag_line_assignments(
    parsed: ParsedLyrics,
    source_section_index: int,
    selected_dutch_sections: Sequence[int],
    source_line_groups: Sequence[Sequence[int]],
) -> tuple[tuple[DutchLineToken, ...], ...]:
    """Create a sequential, non-semantic starting layout for a drag board."""
    groups = _validate_source_groups(parsed, source_section_index, source_line_groups)
    _selected, tokens = _selected_dutch_tokens(
        parsed,
        source_section_index,
        selected_dutch_sections,
    )

    assignments: list[list[DutchLineToken]] = [[] for _group in groups]
    if len(groups) <= len(tokens):
        boundaries = [round(index * len(tokens) / len(groups)) for index in range(len(groups) + 1)]
        for group_index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            assignments[group_index].extend(tokens[start:end])
    else:
        for token_index, token in enumerate(tokens):
            if len(tokens) == 1:
                group_index = 0
            else:
                group_index = round(token_index * (len(groups) - 1) / (len(tokens) - 1))
            assignments[group_index].append(token)
    return tuple(tuple(assignment) for assignment in assignments)


def decode_drag_containers(
    containers: Sequence[Mapping[str, object]],
    *,
    unassigned_header: str,
    target_headers: Sequence[str],
    token_to_reference: Mapping[str, DutchLineToken],
) -> tuple[tuple[DutchLineToken, ...], tuple[tuple[DutchLineToken, ...], ...]]:
    """Validate a sortable component result and decode its opaque display cards."""
    expected_headers = (unassigned_header, *target_headers)
    if len(containers) != len(expected_headers):
        raise PairingError("The drag board returned an unexpected number of containers.")

    by_header: dict[str, tuple[str, ...]] = {}
    all_items: list[str] = []
    for container in containers:
        if not isinstance(container, Mapping):
            raise PairingError("The drag board returned malformed container data.")
        header = container.get("header")
        items = container.get("items")
        if not isinstance(header, str) or not isinstance(items, list):
            raise PairingError("The drag board returned malformed container data.")
        if header in by_header:
            raise PairingError("The drag board returned duplicate container headers.")
        if not all(isinstance(item, str) for item in items):
            raise PairingError("The drag board returned a malformed Dutch line card.")
        by_header[header] = tuple(items)
        all_items.extend(items)

    if set(by_header) != set(expected_headers):
        raise PairingError("The drag board returned stale or unknown containers.")
    if len(all_items) != len(set(all_items)):
        raise PairingError("A Dutch line card appears more than once on the drag board.")

    expected_items = set(token_to_reference)
    actual_items = set(all_items)
    if actual_items != expected_items:
        missing = len(expected_items - actual_items)
        extra = len(actual_items - expected_items)
        raise PairingError(
            f"The drag board lost or added Dutch cards (missing={missing}, unknown={extra})."
        )

    def decode(items: Sequence[str]) -> tuple[DutchLineToken, ...]:
        return tuple(token_to_reference[item] for item in items)

    return (
        decode(by_header[unassigned_header]),
        tuple(decode(by_header[header]) for header in target_headers),
    )


def build_drag_line_groups(
    parsed: ParsedLyrics,
    source_section_index: int,
    selected_dutch_sections: Sequence[int],
    source_line_groups: Sequence[Sequence[int]],
    assignments: Sequence[Sequence[DutchLineToken]],
    unassigned: Sequence[DutchLineToken],
) -> tuple[AlignedLine, ...]:
    """Convert a complete drag board into exact, validated Chinese line groups."""
    groups = _validate_source_groups(parsed, source_section_index, source_line_groups)
    source = parsed.sections[source_section_index]
    selected, expected_tokens = _selected_dutch_tokens(
        parsed,
        source_section_index,
        selected_dutch_sections,
    )
    if len(assignments) != len(groups):
        raise PairingError(f"[{source.label}] has a stale drag-board layout.")

    normalized_assignments = tuple(
        tuple((int(section_index), int(line_index)) for section_index, line_index in assignment)
        for assignment in assignments
    )
    normalized_unassigned = tuple(
        (int(section_index), int(line_index)) for section_index, line_index in unassigned
    )
    all_tokens = [
        token
        for assignment in normalized_assignments
        for token in assignment
    ] + list(normalized_unassigned)

    if len(all_tokens) != len(set(all_tokens)):
        raise PairingError(f"[{source.label}] contains a duplicated Dutch line card.")
    if set(all_tokens) != set(expected_tokens):
        missing = len(set(expected_tokens) - set(all_tokens))
        extra = len(set(all_tokens) - set(expected_tokens))
        raise PairingError(
            f"[{source.label}] has stale Dutch cards (missing={missing}, unknown={extra})."
        )

    selected_set = set(selected)
    previous_line_by_section: dict[int, int] = defaultdict(lambda: -1)
    used_sections: set[int] = set()
    aligned_groups: list[AlignedLine] = []

    for group_number, (source_indices, assigned_tokens) in enumerate(
        zip(groups, normalized_assignments),
        start=1,
    ):
        if not assigned_tokens:
            first_line = source_indices[0] + 1
            last_line = source_indices[-1] + 1
            line_label = str(first_line) if first_line == last_line else f"{first_line}-{last_line}"
            raise PairingError(
                f"[{source.label}] Chinese line group {line_label} needs at least one Dutch card."
            )

        references: list[TranslationReference] = []
        run_section: int | None = None
        run_lines: list[int] = []

        def flush_run() -> None:
            nonlocal run_section, run_lines
            if run_section is None:
                return
            references.append(
                TranslationReference(
                    section_index=run_section,
                    line_indices=tuple(run_lines),
                )
            )
            run_section = None
            run_lines = []

        for section_index, line_index in assigned_tokens:
            if section_index not in selected_set:
                raise PairingError(
                    f"[{source.label}] group {group_number} uses an unselected Dutch section."
                )
            section = parsed.sections[section_index]
            if line_index < 0 or line_index >= len(section.lines):
                raise PairingError(
                    f"[{source.label}] group {group_number} uses an invalid Dutch line."
                )
            if line_index <= previous_line_by_section[section_index]:
                raise PairingError(
                    f"[{source.label}] Dutch cards must stay in line order within [{section.label}]."
                )
            previous_line_by_section[section_index] = line_index
            used_sections.add(section_index)

            if run_section == section_index and line_index == run_lines[-1] + 1:
                run_lines.append(line_index)
            else:
                flush_run()
                run_section = section_index
                run_lines = [line_index]
        flush_run()

        aligned_groups.append(
            AlignedLine(
                source_line_indices=source_indices,
                translation_references=tuple(references),
            )
        )

    unused_sections = [
        parsed.sections[index].label for index in selected if index not in used_sections
    ]
    if unused_sections:
        raise PairingError(
            f"[{source.label}] has no dragged lines from selected section(s): "
            + ", ".join(f"[{label}]" for label in unused_sections)
            + "."
        )
    return tuple(aligned_groups)
