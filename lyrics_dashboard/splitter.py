from __future__ import annotations

import logging
import math
import unicodedata
from dataclasses import dataclass
from typing import Iterable

import jieba

from .text_processing import LanguageCode, clean_content_result

jieba.setLogLevel(logging.WARNING)
_CHINESE_TOKENIZER = jieba.Tokenizer()
MINIMUM_SPLIT_LIMIT = 4
_PROTECTED_ARTICLES = frozenset({"de", "het", "een", "uw", "a", "an", "the"})


@dataclass(frozen=True)
class SplitResult:
    text: str
    used_character_fallback: bool = False


def _split_parts(text: str, boundary: int) -> tuple[str, str] | None:
    left = text[:boundary].rstrip()
    right = text[boundary:].lstrip()
    if not left or not right:
        return None
    return left, right


def _choose_balanced_boundary(
    text: str,
    candidates: Iterable[int],
    max_length: int,
    minimum_fragment_length: int,
    *,
    preferred_boundary: int | None = None,
) -> int | None:
    choices: list[tuple[int, str, str]] = []
    for boundary in sorted(set(int(candidate) for candidate in candidates)):
        if not 0 < boundary < len(text):
            continue
        parts = _split_parts(text, boundary)
        if parts is None:
            continue
        left, right = parts
        if min(len(left), len(right)) < minimum_fragment_length:
            continue
        choices.append((boundary, left, right))

    if not choices:
        return None

    midpoint = len(text) / 2

    def score(
        choice: tuple[int, str, str],
    ) -> tuple[int, int, int, float, int]:
        boundary, left, right = choice
        both_within_limit = len(left) <= max_length and len(right) <= max_length
        imbalance = abs(len(left) - len(right))
        return (
            0 if both_within_limit else 1,
            (
                abs(boundary - preferred_boundary)
                if preferred_boundary is not None
                else imbalance
            ),
            imbalance if preferred_boundary is not None else 0,
            abs(boundary - midpoint),
            boundary,
        )

    return min(choices, key=score)[0]


def _article_safe_boundaries(
    text: str,
    candidates: Iterable[int],
) -> tuple[int, ...]:
    """Exclude boundaries after protected middle-sentence words."""
    safe: list[int] = []
    for boundary in candidates:
        left_words = text[:boundary].rstrip().split()
        right_words = text[boundary:].lstrip().split()
        if (
            len(left_words) >= 2
            and right_words
            and left_words[-1].casefold() in _PROTECTED_ARTICLES
        ):
            continue
        safe.append(boundary)
    return tuple(safe)


def _chinese_word_boundaries(text: str) -> tuple[int, ...]:
    boundaries: set[int] = set()
    for _word, _start, end in _CHINESE_TOKENIZER.tokenize(
        text,
        mode="default",
        HMM=True,
    ):
        if 0 < end < len(text):
            boundaries.add(end)
    return tuple(sorted(boundaries))


def _is_chinese_character(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2FA1F
    )


def _middle_two_character_word_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Return exact middle-of-text two-character Chinese Jieba tokens."""
    return tuple(
        (start, end)
        for word, start, end in _CHINESE_TOKENIZER.tokenize(
            text,
            mode="default",
            HMM=True,
        )
        if (
            len(word) == 2
            and end - start == 2
            and all(_is_chinese_character(character) for character in word)
            and 0 < start
            and end < len(text)
        )
    )


def _character_boundaries(text: str) -> tuple[int, ...]:
    return tuple(
        boundary
        for boundary in range(1, len(text))
        if not unicodedata.combining(text[boundary])
    )


def _adjust_boundary_around_two_character_word(
    text: str,
    preferred: int,
    candidates: Iterable[int],
    max_length: int,
    minimum_fragment_length: int,
) -> int | None:
    """Move a planned split only when it cuts a middle two-character word."""
    candidate_boundaries = tuple(candidates)
    word_spans = _middle_two_character_word_spans(text)
    protected_span = next(
        (
            (start, end)
            for start, end in word_spans
            if start < preferred < end
        ),
        None,
    )
    if protected_span is None:
        return preferred

    adjacent_boundary = _choose_balanced_boundary(
        text,
        protected_span,
        max_length,
        minimum_fragment_length,
        preferred_boundary=preferred,
    )
    if adjacent_boundary is not None:
        return adjacent_boundary

    safe_candidates = tuple(
        boundary
        for boundary in candidate_boundaries
        if not any(start < boundary < end for start, end in word_spans)
    )
    return _choose_balanced_boundary(
        text,
        safe_candidates,
        max_length,
        minimum_fragment_length,
        preferred_boundary=preferred,
    )


def split_lyric_result(
    text: str,
    language: LanguageCode,
    max_length: int,
    *,
    minimum_fragment_length: int = 1,
    minimum_fragment_ratio: float = 0.0,
) -> SplitResult:
    """Clean lyric content and add at most one balanced ``//``."""
    if max_length < MINIMUM_SPLIT_LIMIT:
        raise ValueError("Maximum segment length must be at least 4.")
    if minimum_fragment_length < 1:
        raise ValueError("Minimum fragment length must be at least 1.")
    if not 0.0 <= minimum_fragment_ratio < 0.5:
        raise ValueError("Minimum fragment ratio must be between 0 and 0.5.")

    cleaning = clean_content_result(text, language)
    cleaned = cleaning.text
    if not cleaned or len(cleaned) <= max_length:
        return SplitResult(cleaned)
    required_fragment_length = max(
        minimum_fragment_length,
        math.ceil(len(cleaned) * minimum_fragment_ratio),
    )

    if language == "nl":
        whitespace_candidates = cleaning.original_whitespace_boundaries
        boundary = _choose_balanced_boundary(
            cleaned,
            whitespace_candidates,
            max_length,
            required_fragment_length,
        )
        if boundary is None:
            return SplitResult(cleaned)
        safe_boundaries = _article_safe_boundaries(
            cleaned,
            whitespace_candidates,
        )
        if boundary not in safe_boundaries:
            boundary = _choose_balanced_boundary(
                cleaned,
                safe_boundaries,
                max_length,
                required_fragment_length,
                preferred_boundary=boundary,
            )
        if boundary is None:
            return SplitResult(cleaned)
        left, right = _split_parts(cleaned, boundary) or (cleaned, "")
        return SplitResult(f"{left}//{right}" if right else left)

    word_candidates = tuple(
        boundary
        for boundary in _chinese_word_boundaries(cleaned)
        if boundary not in cleaning.punctuation_separator_boundaries
    )
    word_boundary = _choose_balanced_boundary(
        cleaned,
        word_candidates,
        max_length,
        required_fragment_length,
    )
    if word_boundary is not None:
        left, right = _split_parts(cleaned, word_boundary) or (cleaned, "")
        return SplitResult(f"{left}//{right}" if right else left)
    whitespace_candidates = cleaning.original_whitespace_boundaries
    whitespace_boundary = _choose_balanced_boundary(
        cleaned,
        whitespace_candidates,
        max_length,
        required_fragment_length,
    )
    if whitespace_boundary is not None:
        left, right = _split_parts(cleaned, whitespace_boundary) or (cleaned, "")
        return SplitResult(f"{left}//{right}" if right else left)
    character_candidates = _character_boundaries(cleaned)
    character_boundary = _choose_balanced_boundary(
        cleaned,
        character_candidates,
        max_length,
        required_fragment_length,
    )
    if character_boundary is None:
        return SplitResult(cleaned)
    character_boundary = _adjust_boundary_around_two_character_word(
        cleaned,
        character_boundary,
        character_candidates,
        max_length,
        required_fragment_length,
    )
    if character_boundary is None:
        return SplitResult(cleaned)
    left, right = _split_parts(cleaned, character_boundary) or (cleaned, "")
    return SplitResult(
        f"{left}//{right}" if right else left,
        used_character_fallback=bool(right),
    )


def split_lyric(text: str, language: LanguageCode, max_length: int) -> str:
    """Return cleaned lyric content with zero or one balanced ``//``."""
    return split_lyric_result(text, language, max_length).text
