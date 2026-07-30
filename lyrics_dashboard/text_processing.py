from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection
from dataclasses import dataclass
from typing import Literal

LanguageCode = Literal["zh", "nl"]

_SCHEME_URL_RE = re.compile(r"^(?:https?|ftp)://[^\s<>]+$", re.IGNORECASE)
_WWW_URL_RE = re.compile(r"^www\.[^\s<>]+$", re.IGNORECASE)
_DOMAIN_URL_RE = re.compile(
    r"^(?P<host>(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"(?P<tld>[a-z]{2,63}))(?::\d{1,5})?(?P<suffix>[/?#][^\s<>]*)?$",
    re.IGNORECASE,
)
_MARKDOWN_LINK_RE = re.compile(
    r"^\[[^\]\r\n]*\]\((?P<target>[^\s()]+)\)$",
    re.IGNORECASE,
)
_TITLE_HEADING_RE = re.compile(
    r"^\s*\[?\s*title(?:\s*\d+)?\s*\]?\s*:?\s*$",
    re.IGNORECASE,
)
_WRAPPED_URL_FRAGMENT_RE = re.compile(
    r"^[A-Za-z0-9%._~:/?#\[\]@!$&'()*+,;=-]+$"
)
_COMMON_BARE_DOMAIN_TLDS = {
    "ai",
    "app",
    "be",
    "biz",
    "church",
    "co",
    "com",
    "de",
    "dev",
    "edu",
    "eu",
    "gov",
    "info",
    "int",
    "io",
    "link",
    "live",
    "ly",
    "me",
    "mil",
    "music",
    "net",
    "nl",
    "online",
    "org",
    "site",
    "tv",
    "uk",
    "xyz",
}


@dataclass(frozen=True)
class CleanContentResult:
    text: str
    original_whitespace_boundaries: tuple[int, ...] = ()
    punctuation_separator_boundaries: tuple[int, ...] = ()


def is_standalone_url_line(line: str) -> bool:
    """Return whether a complete line is clearly a URL or link-only markup."""
    candidate = line.strip()
    if len(candidate) >= 2 and candidate.startswith("<") and candidate.endswith(">"):
        candidate = candidate[1:-1].strip()

    markdown_match = _MARKDOWN_LINK_RE.fullmatch(candidate)
    if markdown_match:
        candidate = markdown_match.group("target")

    if _SCHEME_URL_RE.fullmatch(candidate) or _WWW_URL_RE.fullmatch(candidate):
        return True

    domain_match = _DOMAIN_URL_RE.fullmatch(candidate)
    if not domain_match:
        return False
    return bool(domain_match.group("suffix")) or (
        domain_match.group("tld").lower() in _COMMON_BARE_DOMAIN_TLDS
    )


def _is_wrapped_url_continuation(previous_line: str, line: str) -> bool:
    candidate = line.strip()
    if (
        not candidate
        or any(character.isspace() for character in candidate)
        or is_standalone_url_line(candidate)
        or not _WRAPPED_URL_FRAGMENT_RE.fullmatch(candidate)
    ):
        return False

    previous = previous_line.rstrip()
    if previous.endswith(("-", "_")):
        return any(
            marker in candidate
            for marker in ("-", "_", "/", "?", "&", "=", "#", "%")
        )
    if previous.endswith(("=", "&", "?", "#", "%")):
        return True
    if candidate.startswith(("-", "_", "/", "?", "&", "=", "#", "%")):
        return True
    return False


def remove_leading_links(
    text: str,
    hyperlink_line_indices: Collection[int] = (),
) -> str:
    """Remove only link blocks before the first real title-content line.

    ``hyperlink_line_indices`` carries format-specific metadata, such as DOCX
    paragraphs whose visible text does not itself resemble a URL.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    hyperlink_lines = set(hyperlink_line_indices)
    preserved_title_headings: list[str] = []
    removed_link = False
    index = 0

    while index < len(lines):
        candidate = lines[index].strip()
        if not candidate:
            index += 1
            continue

        if _TITLE_HEADING_RE.fullmatch(candidate):
            preserved_title_headings.append(lines[index])
            index += 1
            continue

        plain_url = is_standalone_url_line(candidate)
        if index in hyperlink_lines or plain_url:
            removed_link = True
            previous = candidate
            index += 1

            if plain_url:
                while index < len(lines):
                    continuation = lines[index].strip()
                    if not _is_wrapped_url_continuation(previous, continuation):
                        break
                    previous = continuation
                    index += 1
            continue

        break

    if not removed_link:
        return text
    return "\n".join((*preserved_title_headings, *lines[index:]))


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
