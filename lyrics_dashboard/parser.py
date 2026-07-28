from __future__ import annotations

import re

from .errors import PairingError, ParseError
from .models import Language, ParsedLyrics, Section

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
LATIN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")

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
    cjk = len(CJK_RE.findall(text))
    latin = len(LATIN_RE.findall(text))
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

    first_latin = LATIN_RE.search(raw_title)
    first_cjk = CJK_RE.search(raw_title)
    if first_latin and first_cjk:
        if first_cjk.start() < first_latin.start():
            chinese = raw_title[: first_latin.start()].strip(" -–—|,;")
            dutch = raw_title[first_latin.start() :].strip(" -–—|,;")
        else:
            dutch = raw_title[: first_cjk.start()].strip(" -–—|,;")
            chinese = raw_title[first_cjk.start() :].strip(" -–—|,;")
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


def parse_lyrics(text: str) -> ParsedLyrics:
    """Parse a basic-format bilingual lyric document without assuming equal counts."""
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
    if not any(section.language == "zh" for section in sections):
        raise PairingError("The file does not contain a detected Chinese lyric section.")
    if not any(section.language == "nl" for section in sections):
        raise PairingError("The file does not contain a detected Dutch lyric section.")

    raw_title, chinese_title, dutch_title = _split_title(title_lines)
    warnings = _validate_language_blocks(sections)

    return ParsedLyrics(
        raw_title=raw_title,
        chinese_title=chinese_title,
        dutch_title=dutch_title,
        sections=tuple(sections),
        warnings=tuple(warnings),
    )
