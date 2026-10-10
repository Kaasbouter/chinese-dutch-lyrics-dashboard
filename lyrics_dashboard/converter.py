from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .alignment import validate_alignment_plan
from .errors import PairingError
from .models import AlignmentPlan, ParsedLyrics, Section
from .splitter import MINIMUM_SPLIT_LIMIT, split_lyric_result
from .text_processing import clean_content_text

FIRST_SIDE_LIMIT_RATIO = 0.80
FIRST_SIDE_MINIMUM_FRAGMENT_RATIO = 0.25
MINIMUM_FIRST_SIDE_FRAGMENT_LENGTH = 2


@dataclass(frozen=True)
class ConversionSettings:
    switch_index: int | None = None
    chinese_max_length: int = 10
    dutch_max_length: int = 40
    include_title: bool = True
    title_separator: str = " "


def derive_first_side_limit(normal_limit: int) -> int:
    """Return the stricter limit used for the first or sole output side."""
    if normal_limit < MINIMUM_SPLIT_LIMIT:
        raise ValueError("Normal maximum length must be at least 4.")
    return max(
        MINIMUM_SPLIT_LIMIT,
        int(normal_limit * FIRST_SIDE_LIMIT_RATIO),
    )


def encode_utf8_txt(text: str) -> bytes:
    """Encode the editable preview exactly as the downloaded UTF-8 TXT body."""
    return text.encode("utf-8")


def derive_output_filename(
    saved_output: str,
    *,
    source_title: str,
    source_name: str,
) -> str:
    """Name the download after its saved title, with the source as a fallback."""
    title_lines: list[str] = []
    collecting_title = False
    for line in saved_output.splitlines():
        line = line.strip().lstrip("\ufeff").strip()
        if line.casefold() == "[title]":
            collecting_title = True
            continue
        if line.startswith("[") and line.endswith("]"):
            break
        if collecting_title:
            if not line and title_lines:
                break
            if line:
                title_lines.append(line)

    for candidate in (
        " ".join(title_lines),
        source_title,
        Path(source_name).stem,
        "converted_lyrics",
    ):
        # Keep Unicode and spaces, replacing characters forbidden in filenames.
        stem = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f]+', " ", candidate)
        stem = " ".join(stem.split()).strip(" .")
        if stem:
            break
    if re.fullmatch(
        r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?",
        stem,
        flags=re.IGNORECASE,
    ):
        stem = f"_{stem}"
    return f"{stem}.txt"


def _format_title(parsed: ParsedLyrics, separator: str) -> list[str]:
    if not parsed.raw_title and not parsed.chinese_title and not parsed.dutch_title:
        return []
    if parsed.mode == "single-language":
        language = parsed.single_language
        if language not in ("zh", "nl"):
            raise PairingError("Single-language conversion needs one detected lyric language.")
        title_source = (
            parsed.chinese_title if language == "zh" else parsed.dutch_title
        ) or parsed.raw_title
        title = clean_content_text(title_source, language)
        return ["[Title]", title, ""] if title else []
    if parsed.chinese_title and parsed.dutch_title:
        chinese_source = parsed.chinese_title
        dutch_source = parsed.dutch_title
        # Keep the source song number first even when a Latin-first title is
        # rendered in the existing Chinese-first order. Clean its punctuation
        # together with the title rather than adding another copy of the digits.
        number_prefix = re.match(r"^\d+", parsed.raw_title)
        if number_prefix and parsed.raw_title.startswith(dutch_source):
            prefix_end = number_prefix.end()
            while prefix_end < len(dutch_source):
                character = dutch_source[prefix_end]
                if not (
                    character.isspace()
                    or unicodedata.category(character).startswith("P")
                ):
                    break
                prefix_end += 1
            prefix = dutch_source[:prefix_end]
            dutch_source = dutch_source[len(prefix):]
            chinese_source = prefix + chinese_source
        chinese_title = clean_content_text(chinese_source, "zh")
        dutch_title = clean_content_text(dutch_source, "nl")
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


def _convert_single_language(
    parsed: ParsedLyrics,
    settings: ConversionSettings,
    warnings: list[str] | None,
) -> str:
    language = parsed.single_language
    if language not in ("zh", "nl"):
        raise PairingError("Single-language conversion needs one detected lyric language.")

    normal_limit = (
        settings.chinese_max_length if language == "zh" else settings.dutch_max_length
    )
    first_side_limit = derive_first_side_limit(normal_limit)
    output: list[str] = []
    if settings.include_title:
        output.extend(_format_title(parsed, settings.title_separator))

    for section in parsed.sections:
        if section.language != language:
            raise PairingError(
                "Single-language conversion cannot contain sections in another language."
            )
        output.append(f"[{section.label}]")
        for row_number, source_text in enumerate(section.lines, start=1):
            result = split_lyric_result(
                source_text,
                language,
                first_side_limit,
                minimum_fragment_length=MINIMUM_FIRST_SIDE_FRAGMENT_LENGTH,
                minimum_fragment_ratio=FIRST_SIDE_MINIMUM_FRAGMENT_RATIO,
            )
            if result.used_character_fallback and warnings is not None:
                warning = (
                    f"[{section.label}] output row {row_number}: the local Chinese word "
                    "segmenter found no safe word or whitespace boundary, so a last-resort "
                    "character split was used. Review this `//` in the editable preview."
                )
                if warning not in warnings:
                    warnings.append(warning)
            output.append(result.text)
        output.append("")

    return "\n".join(output).rstrip() + "\n"


def convert_lyrics(
    parsed: ParsedLyrics,
    alignment_plan: AlignmentPlan | None,
    settings: ConversionSettings,
    *,
    warnings: list[str] | None = None,
) -> str:
    """Create TXT from parser-approved single lyrics or validated bilingual matches."""
    if parsed.mode == "single-language":
        return _convert_single_language(parsed, settings, warnings)
    if alignment_plan is None:
        raise PairingError("Bilingual conversion requires a validated alignment plan.")
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
