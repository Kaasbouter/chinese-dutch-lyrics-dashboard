from __future__ import annotations

from dataclasses import dataclass

from .alignment import validate_alignment_plan
from .models import AlignmentPlan, ParsedLyrics, Section
from .splitter import MINIMUM_SPLIT_LIMIT, split_lyric_result
from .text_processing import clean_content_text

FIRST_SIDE_LIMIT_RATIO = 0.80
FIRST_SIDE_MINIMUM_FRAGMENT_RATIO = 0.25
MINIMUM_FIRST_SIDE_FRAGMENT_LENGTH = 2


@dataclass(frozen=True)
class ConversionSettings:
    switch_index: int | None
    chinese_max_length: int = 10
    dutch_max_length: int = 40
    include_title: bool = True
    title_separator: str = " "


def derive_first_side_limit(normal_limit: int) -> int:
    """Return the stricter positional limit used before the output ``|``."""
    if normal_limit < MINIMUM_SPLIT_LIMIT:
        raise ValueError("Normal maximum length must be at least 4.")
    return max(
        MINIMUM_SPLIT_LIMIT,
        int(normal_limit * FIRST_SIDE_LIMIT_RATIO),
    )


def encode_utf8_txt(text: str) -> bytes:
    """Encode the editable preview exactly as the downloaded UTF-8 TXT body."""
    return text.encode("utf-8")


def _format_title(parsed: ParsedLyrics, separator: str) -> list[str]:
    if not parsed.raw_title and not parsed.chinese_title and not parsed.dutch_title:
        return []
    if parsed.chinese_title and parsed.dutch_title:
        chinese_title = clean_content_text(parsed.chinese_title, "zh")
        dutch_title = clean_content_text(parsed.dutch_title, "nl")
        title = separator.join(part for part in (chinese_title, dutch_title) if part)
    elif parsed.chinese_title:
        title = clean_content_text(parsed.chinese_title, "zh")
    elif parsed.dutch_title:
        title = clean_content_text(parsed.dutch_title, "nl")
    else:
        title = clean_content_text(parsed.raw_title, "nl")
    if not title:
        return []
    return ["[Title]", title, ""]


def _join_source_lines(section: Section, indices: tuple[int, ...]) -> str:
    separator = "" if section.language == "zh" else " "
    return separator.join(section.lines[index].strip() for index in indices).strip()


def _join_translation_lines(parsed: ParsedLyrics, references) -> str:
    pieces: list[str] = []
    language = None
    for reference in references:
        section = parsed.sections[reference.section_index]
        language = language or section.language
        separator = "" if section.language == "zh" else " "
        pieces.append(separator.join(section.lines[index].strip() for index in reference.line_indices))
    outer_separator = "" if language == "zh" else " "
    return outer_separator.join(piece.strip() for piece in pieces if piece.strip()).strip()


def convert_lyrics(
    parsed: ParsedLyrics,
    alignment_plan: AlignmentPlan,
    settings: ConversionSettings,
    *,
    warnings: list[str] | None = None,
) -> str:
    """Create the final UTF-8 TXT body from validated semantic/manual matches."""
    if settings.switch_index is not None and not 0 <= settings.switch_index < len(parsed.sections):
        raise ValueError("switch_index is outside the detected section range.")
    validate_alignment_plan(parsed, alignment_plan)

    output: list[str] = []
    if settings.include_title:
        output.extend(_format_title(parsed, settings.title_separator))

    for section in parsed.sections:
        alignment = alignment_plan.for_section(section.original_index)
        output.append(f"[{section.label}]")
        dutch_first = settings.switch_index is not None and section.original_index >= settings.switch_index
        chinese_limit = settings.chinese_max_length
        dutch_limit = settings.dutch_max_length
        chinese_minimum_fragment = 1
        dutch_minimum_fragment = 1

        if dutch_first:
            dutch_limit = derive_first_side_limit(settings.dutch_max_length)
            dutch_minimum_fragment = MINIMUM_FIRST_SIDE_FRAGMENT_LENGTH
        else:
            chinese_limit = derive_first_side_limit(settings.chinese_max_length)
            chinese_minimum_fragment = MINIMUM_FIRST_SIDE_FRAGMENT_LENGTH

        for row_number, line_group in enumerate(alignment.aligned_lines, start=1):
            source_text = _join_source_lines(section, line_group.source_line_indices)
            translation_text = _join_translation_lines(parsed, line_group.translation_references)

            if section.language == "zh":
                chinese_text, dutch_text = source_text, translation_text
            else:
                dutch_text, chinese_text = source_text, translation_text

            chinese_result = split_lyric_result(
                chinese_text,
                "zh",
                chinese_limit,
                minimum_fragment_length=chinese_minimum_fragment,
                minimum_fragment_ratio=(
                    FIRST_SIDE_MINIMUM_FRAGMENT_RATIO if not dutch_first else 0.0
                ),
            )
            dutch_result = split_lyric_result(
                dutch_text,
                "nl",
                dutch_limit,
                minimum_fragment_length=dutch_minimum_fragment,
                minimum_fragment_ratio=(
                    FIRST_SIDE_MINIMUM_FRAGMENT_RATIO if dutch_first else 0.0
                ),
            )
            if chinese_result.used_character_fallback and warnings is not None:
                warning = (
                    f"[{section.label}] output row {row_number}: the local Chinese word "
                    "segmenter found no safe word or whitespace boundary, so a last-resort "
                    "character split was used. Review this `//` in the editable preview."
                )
                if warning not in warnings:
                    warnings.append(warning)

            chinese = chinese_result.text
            dutch = dutch_result.text
            output.append(f"{dutch}|{chinese}" if dutch_first else f"{chinese}|{dutch}")
        output.append("")

    return "\n".join(output).rstrip() + "\n"
