from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Literal

LanguageCode = Literal["zh", "nl"]


@dataclass(frozen=True)
class CleanContentResult:
    text: str
    original_whitespace_boundaries: tuple[int, ...] = ()
    punctuation_separator_boundaries: tuple[int, ...] = ()


def _is_punctuation(character: str) -> bool:
    return unicodedata.category(character).startswith("P")


def _is_cjk(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2FA1F
    )


def _next_content_character(text: str, start: int) -> str:
    for character in text[start:]:
        if character.isspace() or _is_punctuation(character):
            continue
        return character
    return ""


def _previous_content_character(characters: list[tuple[str, bool]]) -> str:
    for character, _from_original_whitespace in reversed(characters):
        if not character.isspace():
            return character
    return ""


def _punctuation_needs_separator(previous: str, following: str) -> bool:
    if not previous or not following:
        return False
    if not previous.isalnum() or not following.isalnum():
        return False
    return not (_is_cjk(previous) and _is_cjk(following))


def _normalize_whitespace(
    characters: list[tuple[str, bool]],
) -> CleanContentResult:
    normalized: list[str] = []
    original_boundaries: list[int] = []
    punctuation_boundaries: list[int] = []
    index = 0

    while index < len(characters):
        character, from_original_whitespace = characters[index]
        if not character.isspace():
            normalized.append(character)
            index += 1
            continue

        whitespace_end = index + 1
        has_original_whitespace = from_original_whitespace
        while (
            whitespace_end < len(characters)
            and characters[whitespace_end][0].isspace()
        ):
            has_original_whitespace = (
                has_original_whitespace or characters[whitespace_end][1]
            )
            whitespace_end += 1

        if normalized and whitespace_end < len(characters):
            separator_start = len(normalized)
            normalized.append(" ")
            separator_end = len(normalized)
            if has_original_whitespace:
                original_boundaries.append(separator_end)
            else:
                punctuation_boundaries.extend((separator_start, separator_end))
        index = whitespace_end

    return CleanContentResult(
        text="".join(normalized),
        original_whitespace_boundaries=tuple(original_boundaries),
        punctuation_separator_boundaries=tuple(punctuation_boundaries),
    )


def clean_content_result(
    text: str,
    language: LanguageCode,
) -> CleanContentResult:
    """Clean content while retaining the origin of normalized separators.

    Structural output markers are intentionally not passed through this function;
    the converter adds ``[Section]``, ``|``, and ``//`` after content cleaning.
    """
    cleaned: list[tuple[str, bool]] = []
    index = 0
    while index < len(text):
        character = text[index]
        if not _is_punctuation(character):
            cleaned.append((character, character.isspace()))
            index += 1
            continue

        punctuation_end = index + 1
        while punctuation_end < len(text) and _is_punctuation(text[punctuation_end]):
            punctuation_end += 1

        previous = _previous_content_character(cleaned)
        following = _next_content_character(text, punctuation_end)
        if language == "nl" or _punctuation_needs_separator(previous, following):
            cleaned.append((" ", False))
        index = punctuation_end

    return _normalize_whitespace(cleaned)


def clean_content_text(text: str, language: LanguageCode) -> str:
    """Remove Unicode punctuation and normalize whitespace in content."""
    return clean_content_result(text, language).text
