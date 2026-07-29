from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable

from .errors import PairingError, ParseError
from .models import Language, ParsedLyrics, Section


def _is_cjk_character(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2FA1F
    )


def _is_latin_letter(character: str) -> bool:
    return (
        unicodedata.category(character).startswith("L")
        and unicodedata.name(character, "").startswith("LATIN ")
    )


def _script_counts(text: str) -> tuple[int, int]:
    return (
        sum(_is_cjk_character(character) for character in text),
        sum(_is_latin_letter(character) for character in text),
    )


def _first_character_index(
    text: str,
    predicate: Callable[[str], bool],
) -> int | None:
    return next(
        (index for index, character in enumerate(text) if predicate(character)),
        None,
    )


def _contains_unexpected_letter(text: str, language: Language) -> bool:
    for character in text:
        if not unicodedata.category(character).startswith("L"):
            continue
        if language == "zh" and not _is_cjk_character(character):
            return True
        if language == "nl" and not _is_latin_letter(character):
            return True
    return False


_KIND_ALIASES = {
    "title": "Title",
    "verse": "Verse",
    "couplet": "Verse",
    "chorus": "Chorus",
    "refrein": "Chorus",
    "refrain": "Chorus",
    "bridge": "Bridge",
    "pre-chorus": "Pre-Chorus",
    "pre chorus": "Pre-Chorus",
    "intro": "Intro",
    "outro": "Outro",
    "interlude": "Interlude",
    "tag": "Tag",
}

_HEADING_RE = re.compile(
    r"^\s*\[?\s*(?P<kind>title|verse|couplet|chorus|refrein|refrain|bridge|"
    r"pre[- ]chorus|intro|outro|interlude|tag)\s*(?P<number>\d+)?\s*\]?\s*:?\s*$",
    re.IGNORECASE,
)


def _clean_line(line: str) -> str:
    return (
        line.replace("\ufeff", "")
        .replace("\u00a0", " ")
        .replace("\u200b", "")
        .strip()
    )


def _match_heading(line: str) -> tuple[str, int | None] | None:
    match = _HEADING_RE.match(line)
    if not match:
        return None
    raw_kind = match.group("kind").lower().replace("-", " ")
    normalized_key = raw_kind if raw_kind in _KIND_ALIASES else raw_kind.replace(" ", "-")
    if normalized_key not in _KIND_ALIASES:
        normalized_key = raw_kind
    kind = _KIND_ALIASES[normalized_key]
    number = int(match.group("number")) if match.group("number") else None
    return kind, number


def _display_label(kind: str, number: int | None) -> str:
    return f"{kind} {number}" if number is not None else kind


def detect_language(text: str) -> Language:
    cjk, latin = _script_counts(text)
    if cjk == 0 and latin == 0:
        return "unknown"
    if cjk >= max(2, latin * 1.2):
        return "zh"
    if latin >= max(2, cjk * 1.2):
        return "nl"
    return "zh" if cjk > latin else "nl"


def _remove_leading_track_number(title: str) -> str:
    return re.sub(r"^\s*\d+\s*[-._):]?\s*", "", title).strip()


def _split_title(title_lines: list[str]) -> tuple[str, str, str]:
    cleaned_lines = [_remove_leading_track_number(line) for line in title_lines if line.strip()]
    raw_title = " ".join(cleaned_lines).strip()
    if not raw_title:
        return "", "", ""

    if "|" in raw_title:
        left, right = (part.strip() for part in raw_title.split("|", 1))
        if detect_language(left) == "zh":
            return raw_title, left, right
        if detect_language(right) == "zh":
            return raw_title, right, left
        return raw_title, left, right

    if len(cleaned_lines) >= 2:
        chinese_line = next((line for line in cleaned_lines if detect_language(line) == "zh"), "")
        dutch_line = next((line for line in cleaned_lines if detect_language(line) == "nl"), "")
        if chinese_line and dutch_line:
            return raw_title, chinese_line, dutch_line

    first_latin = _first_character_index(raw_title, _is_latin_letter)
    first_cjk = _first_character_index(raw_title, _is_cjk_character)
    if first_latin is not None and first_cjk is not None:
        if first_cjk < first_latin:
            chinese = raw_title[:first_latin].strip(" -–—|,;")
            dutch = raw_title[first_latin:].strip(" -–—|,;")
        else:
            dutch = raw_title[:first_cjk].strip(" -–—|,;")
            chinese = raw_title[first_cjk:].strip(" -–—|,;")
        return raw_title, chinese, dutch

    language = detect_language(raw_title)
    if language == "zh":
        return raw_title, raw_title, ""
    return raw_title, "", raw_title


def _validate_language_blocks(sections: list[Section]) -> list[str]:
    warnings: list[str] = []
    known = [section.language for section in sections if section.language != "unknown"]
    transitions = sum(left != right for left, right in zip(known, known[1:]))
    if transitions > 1:
        warnings.append(
            "The language blocks are interleaved. The preferred basic format is one complete Chinese block followed by one complete Dutch block (or the reverse)."
        )

    chinese_count = sum(section.language == "zh" for section in sections)
    dutch_count = sum(section.language == "nl" for section in sections)
    if chinese_count != dutch_count:
        warnings.append(
            f"The document has {chinese_count} Chinese sections and {dutch_count} Dutch sections. This is allowed; semantic alignment must decide which sections correspond."
        )
    return warnings


def _validate_single_language_candidate(
    sections: list[Section],
    raw_title: str,
    language: Language,
) -> None:
    """Reject bilingual evidence hidden inside an apparent one-language block."""
    for section in sections:
        for line_number, line in enumerate(section.lines, start=1):
            if "|" in line:
                raise PairingError(
                    f"[{section.label}] line {line_number} contains `|`, which signals "
                    "bilingual structure and is not valid in single-language mode."
                )
            if _contains_unexpected_letter(line, language):
                raise PairingError(
                    f"[{section.label}] line {line_number} contains evidence of a second "
                    "or unsupported language, but only one language block was detected. "
                    "Separate bilingual lyrics into headed sections instead of using "
                    "single-language mode."
                )

    if "|" in raw_title:
        raise PairingError(
            "The title contains bilingual evidence (`|`), but only one lyric language block "
            "was detected. Check that the second language has recognizable section headings."
        )
    if _contains_unexpected_letter(raw_title, language):
        raise PairingError(
            "The title contains bilingual evidence, but only one lyric language block was "
            "detected. Check that the second language has recognizable section headings."
        )


def parse_lyrics(text: str) -> ParsedLyrics:
    """Parse a basic-format bilingual or single-language lyric document."""
    lines = [_clean_line(line) for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    if not any(lines):
        raise ParseError("The source text is empty.")

    title_lines: list[str] = []
    raw_sections: list[tuple[str, int | None, list[str]]] = []
    current_kind: str | None = None
    current_number: int | None = None
    current_lines: list[str] = []
    seen_lyric_heading = False
    collecting_title_heading = False

    def flush_section() -> None:
        nonlocal current_kind, current_number, current_lines
        if current_kind is None:
            return
        lyric_lines = [line for line in current_lines if line]
        if not lyric_lines:
            raise ParseError(f"[{_display_label(current_kind, current_number)}] does not contain lyric lines.")
        raw_sections.append((current_kind, current_number, lyric_lines))
        current_kind = None
        current_number = None
        current_lines = []

    for line in lines:
        heading = _match_heading(line) if line else None
        if heading:
            kind, number = heading
            if kind == "Title":
                if seen_lyric_heading:
                    raise ParseError("[Title] must appear before lyric sections.")
                collecting_title_heading = True
                continue

            collecting_title_heading = False
            if current_kind is not None:
                flush_section()
            seen_lyric_heading = True
            current_kind = kind
            current_number = number
            current_lines = []
            continue

        if not seen_lyric_heading:
            if line:
                title_lines.append(line)
            continue

        if collecting_title_heading:
            if line:
                title_lines.append(line)
            continue

        if current_kind is not None and line:
            current_lines.append(line)

    if current_kind is not None:
        flush_section()

    if not raw_sections:
        raise ParseError(
            "No lyric sections were found. Use headings such as 'Verse 1', 'Chorus 1', or 'Bridge'."
        )

    sections: list[Section] = []
    for index, (kind, number, lyric_lines) in enumerate(raw_sections):
        language = detect_language(" ".join(lyric_lines))
        sections.append(
            Section(
                label=_display_label(kind, number),
                kind=kind,
                number=number,
                lines=tuple(lyric_lines),
                language=language,
                original_index=index,
            )
        )

    unknown = [section.label for section in sections if section.language == "unknown"]
    if unknown:
        raise PairingError(f"Could not detect a language for: {', '.join(unknown)}.")

    raw_title, chinese_title, dutch_title = _split_title(title_lines)
    detected_languages = {section.language for section in sections}
    if len(detected_languages) == 1:
        single_language = next(iter(detected_languages))
        _validate_single_language_candidate(sections, raw_title, single_language)
        mode = "single-language"
        warnings: list[str] = []
    else:
        mode = "bilingual"
        warnings = _validate_language_blocks(sections)

    return ParsedLyrics(
        raw_title=raw_title,
        chinese_title=chinese_title,
        dutch_title=dutch_title,
        sections=tuple(sections),
        warnings=tuple(warnings),
        mode=mode,
    )
