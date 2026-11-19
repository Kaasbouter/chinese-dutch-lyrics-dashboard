from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection
from dataclasses import dataclass
from typing import Literal

LanguageCode = Literal["zh", "nl"]
LatinLyricLanguage = Literal["auto", "nl", "other"]

_DUTCH_LYRIC_WORDS = frozenset(
    "als alleen bent blijft daarmee dank doorheen een eeuwig gebed geboren geef geen "
    "gegaan genade gezongen geloof groot hallo hebt heeft heilig heerlijk heer hemel het "
    "hoe houd ik jezus jou jouw jullie koning kracht kribbe laat leven liefde liefheb "
    "liefhebt liggen liggend luister maakt maar mij mijn neem niet nooit ontferming "
    "onze ons opdat samen schenkt schenk stilte steeds taal toch trouw tegen uit uw "
    "vader vandaag vindt voor vrede waarom waar wacht waren wereld werd "
    "wij wijs wordt zal zegen zijn zij zing zingen zonder zullen".split()
)
_ENGLISH_LYRIC_WORDS = frozenset(
    "all alone always am and are be because bless child come could day dear down each ever "
    "every faithful father for forever from glory grace hallelujah has have heaven here "
    "him hold holy hope how i into just know let life light listen lord love me mercy more my "
    "never not now oh only our peace praise pray rise shall sing still take than that the their "
    "them then there these they this those thou through together too unto upon very voice wait "
    "walk was were what when where who why will with without world worthy would yes you your".split()
)

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
    original_comma_boundaries: tuple[int, ...] = ()
    retained_comma_offsets: tuple[int, ...] = ()


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


def guess_latin_lyric_language(text: str) -> Literal["nl", "other"] | None:
    """Use distinct Dutch and English words as hints within the Latin group."""
    words = set(re.findall(r"[^\W\d_]+", text.casefold()))
    dutch = len(words & _DUTCH_LYRIC_WORDS)
    english = len(words & _ENGLISH_LYRIC_WORDS)
    if dutch > english:
        return "nl"
    if english:
        return "other"
    return None


def dutch_lyric_comma_indices(
    text: str,
    language: LatinLyricLanguage,
    *,
    dutch_context: bool,
) -> frozenset[int]:
    """Find commas with lyric content on both sides of the same source line."""
    if language == "other" or "," not in text:
        return frozenset()

    indices: set[int] = set()
    next_offset = 0
    for line in text.split("\n"):
        offset = next_offset
        next_offset += len(line) + 1
        if "," not in line:
            continue
        if language == "auto":
            hint = guess_latin_lyric_language(line)
            if hint == "other" or (hint is None and not dutch_context):
                continue
        content = [
            index for index, character in enumerate(line)
            if not (character.isspace() or _is_punctuation(character))
        ]
        if content:
            indices.update(
                offset + index
                for index in range(content[0] + 1, content[-1])
                if line[index] == ","
            )
    return frozenset(indices)


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
    original_comma_positions: Collection[int],
    retained_comma_counts: dict[int, int],
) -> CleanContentResult:
    normalized: list[str] = []
    original_boundaries: list[int] = []
    punctuation_boundaries: list[int] = []
    comma_positions = set(original_comma_positions)
    comma_boundaries: list[int] = []
    retained_comma_offsets: list[int] = []
    index = 0

    while index < len(characters):
        character, from_original_whitespace = characters[index]
        if not character.isspace():
            if index in comma_positions:
                comma_boundaries.append(len(normalized))
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
            if retained_comma_counts:
                comma_count = sum(
                    retained_comma_counts.get(position, 0)
                    for position in range(index, whitespace_end)
                )
                retained_comma_offsets.extend([separator_start] * comma_count)
            normalized.append(" ")
            separator_end = len(normalized)
            if has_original_whitespace:
                original_boundaries.append(separator_end)
            else:
                punctuation_boundaries.extend((separator_start, separator_end))
            # A removed comma touching whitespace shares its normalized edge,
            # including when the comma came immediately after the whitespace.
            if any(
                position in comma_positions
                for position in range(index, whitespace_end + 1)
            ):
                comma_boundaries.append(separator_end)
        index = whitespace_end

    return CleanContentResult(
        text="".join(normalized),
        original_whitespace_boundaries=tuple(original_boundaries),
        punctuation_separator_boundaries=tuple(punctuation_boundaries),
        original_comma_boundaries=tuple(
            sorted({
                boundary
                for boundary in comma_boundaries
                if 0 < boundary < len(normalized)
            })
        ),
        retained_comma_offsets=tuple(retained_comma_offsets),
    )


def clean_content_result(
    text: str,
    language: LanguageCode,
    *,
    retained_comma_indices: frozenset[int] = frozenset(),
) -> CleanContentResult:
    """Clean content while tracking source whitespace and relevant comma positions.

    Structural output markers are intentionally not passed through this function;
    the converter adds ``[Section]``, ``|``, and ``//`` after content cleaning.
    """
    cleaned: list[tuple[str, bool]] = []
    original_comma_positions: list[int] = []
    retained_comma_counts: dict[int, int] = {}
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

        if language == "zh" and any(
            comma in text[index:punctuation_end] for comma in (",", "，")
        ):
            original_comma_positions.append(len(cleaned))
        if language == "nl" and retained_comma_indices:
            retained_count = sum(
                position in retained_comma_indices
                for position in range(index, punctuation_end)
            )
            if retained_count:
                retained_comma_counts[len(cleaned)] = retained_count
        previous = _previous_content_character(cleaned)
        following = _next_content_character(text, punctuation_end)
        if language == "nl" or _punctuation_needs_separator(previous, following):
            cleaned.append((" ", False))
        index = punctuation_end

    return _normalize_whitespace(cleaned, original_comma_positions, retained_comma_counts)


def clean_content_text(text: str, language: LanguageCode) -> str:
    """Remove Unicode punctuation and normalize whitespace in content."""
    return clean_content_result(text, language).text
