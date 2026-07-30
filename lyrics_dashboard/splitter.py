from __future__ import annotations

import logging
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

import jieba

from .text_processing import LanguageCode, clean_content_result

jieba.setLogLevel(logging.WARNING)
_CHINESE_TOKENIZER = jieba.Tokenizer()
MINIMUM_SPLIT_LIMIT = 4
_DUTCH_ARTICLES_AND_DETERMINERS = frozenset({"de", "het", "een", "uw"})
_DUTCH_PERSONAL_PRONOUNS = frozenset(
    "ik mij me jij je jou u hij hem zij ze haar wij we ons jullie hen hun".split()
)
_DUTCH_PREPOSITIONS = frozenset(
    (
        "aan achter bij binnen boven buiten door in langs met na naar naast om "
        "onder op over rond tegen tot tussen uit van voor zonder vanaf vanuit "
        "tijdens sinds volgens ondanks dankzij wegens behalve"
    ).split()
)
_ENGLISH_ARTICLES_AND_DETERMINERS = frozenset({"a", "an", "the"})
_CONTEXTUAL_ENGLISH_POSSESSIVE_DETERMINERS = frozenset({"your"})
_CONTEXTUAL_LATIN_MODIFIERS = frozenset({"new", "nieuw"})
_ENGLISH_PERSONAL_PRONOUNS = frozenset(
    "i me you he him she her it we us they them".split()
)
_ENGLISH_PREPOSITIONS = frozenset(
    (
        "about above across after against along among around at before behind "
        "below beneath beside between beyond by despite down during for from "
        "in inside into near of off on onto out outside over past through "
        "throughout to toward towards under underneath until up upon with "
        "within without"
    ).split()
)
_PROTECTED_LATIN_LEAD_WORDS = frozenset().union(
    _DUTCH_ARTICLES_AND_DETERMINERS,
    _DUTCH_PERSONAL_PRONOUNS,
    _DUTCH_PREPOSITIONS,
    _ENGLISH_ARTICLES_AND_DETERMINERS,
    _ENGLISH_PERSONAL_PRONOUNS,
    _ENGLISH_PREPOSITIONS,
)
_CHINESE_PERSONAL_PRONOUNS = frozenset(
    "我 我們 我们 你 你們 你们 您 祢 祂 他 她 它 他們 他们 她們 她们 它們 它们".split()
)
_CHINESE_PREPOSITION_LIKE_TOKENS = frozenset(
    (
        "在 從 从 向 對 对 給 给 為 为 被 把 跟 與 与 到 自 由 關於 关于 "
        "為了 为了 因 因為 因为 靠 朝 往"
    ).split()
)
_PROTECTED_CHINESE_LEAD_TOKENS = (
    _CHINESE_PERSONAL_PRONOUNS | _CHINESE_PREPOSITION_LIKE_TOKENS
)
_PROTECTED_CHINESE_LEADS_LONGEST_FIRST = tuple(
    sorted(
        _PROTECTED_CHINESE_LEAD_TOKENS,
        key=lambda token: (-len(token), token),
    )
)


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


def _grammatical_token_spans(
    text: str,
    language: LanguageCode,
) -> tuple[tuple[str, int, int], ...]:
    if language == "nl":
        return tuple(
            (match.group(), match.start(), match.end())
            for match in re.finditer(r"\S+", text)
        )
    raw_tokens = tuple(
        (word, start, end)
        for word, start, end in _CHINESE_TOKENIZER.tokenize(
            text,
            mode="default",
            HMM=True,
        )
        if word.strip()
    )
    tokens: list[tuple[str, int, int]] = []
    index = 0
    while index < len(raw_tokens):
        start = raw_tokens[index][1]
        protected_match: tuple[str, int, int, int] | None = None
        for protected_token in _PROTECTED_CHINESE_LEADS_LONGEST_FIRST:
            end = start + len(protected_token)
            if not text.startswith(protected_token, start):
                continue
            end_index = index
            while (
                end_index < len(raw_tokens)
                and raw_tokens[end_index][2] < end
            ):
                end_index += 1
            if (
                end_index < len(raw_tokens)
                and raw_tokens[end_index][2] == end
            ):
                protected_match = (
                    protected_token,
                    start,
                    end,
                    end_index + 1,
                )
                break

        if protected_match is None:
            tokens.append(raw_tokens[index])
            index += 1
            continue
        token, start, end, index = protected_match
        tokens.append((token, start, end))
    return tuple(tokens)


def _is_protected_grammatical_lead(
    token: str,
    language: LanguageCode,
) -> bool:
    if language == "nl":
        return token.casefold() in _PROTECTED_LATIN_LEAD_WORDS
    return token in _PROTECTED_CHINESE_LEAD_TOKENS


def _extend_local_latin_chain(
    tokens: tuple[tuple[str, int, int], ...],
    chain_start_index: int,
    index: int,
) -> int:
    """Extend only the explicit short determiner/modifier chain patterns."""
    lead_words = {
        token.casefold()
        for token, _start, _end in tokens[chain_start_index:index]
    }
    has_preposition = bool(
        lead_words & (_DUTCH_PREPOSITIONS | _ENGLISH_PREPOSITIONS)
    )
    has_determiner = bool(
        lead_words
        & (
            _DUTCH_ARTICLES_AND_DETERMINERS
            | _ENGLISH_ARTICLES_AND_DETERMINERS
        )
    )

    if (
        has_preposition
        and not lead_words & _CONTEXTUAL_ENGLISH_POSSESSIVE_DETERMINERS
        and index < len(tokens)
        and tokens[index][0].casefold()
        in _CONTEXTUAL_ENGLISH_POSSESSIVE_DETERMINERS
    ):
        index += 1
        has_determiner = True
    if (
        has_determiner
        and not lead_words & _CONTEXTUAL_LATIN_MODIFIERS
        and index < len(tokens)
        and tokens[index][0].casefold() in _CONTEXTUAL_LATIN_MODIFIERS
    ):
        index += 1
    return index


def _protected_grammatical_chain_spans(
    text: str,
    language: LanguageCode,
) -> tuple[tuple[int, int], ...]:
    """Keep pronoun, determiner, and preposition runs with the next token."""
    tokens = _grammatical_token_spans(text, language)
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(tokens):
        if not _is_protected_grammatical_lead(tokens[index][0], language):
            index += 1
            continue

        chain_start_index = index
        while True:
            while (
                index < len(tokens)
                and _is_protected_grammatical_lead(
                    tokens[index][0],
                    language,
                )
            ):
                index += 1
            if language != "nl":
                break
            extended_index = _extend_local_latin_chain(
                tokens,
                chain_start_index,
                index,
            )
            if extended_index == index:
                break
            index = extended_index

        if index < len(tokens):
            spans.append((tokens[chain_start_index][1], tokens[index][2]))
            index += 1
        elif index - chain_start_index > 1:
            spans.append((tokens[chain_start_index][1], tokens[index - 1][2]))
    return tuple(spans)


def _boundary_is_outside_spans(
    boundary: int,
    spans: Iterable[tuple[int, int]],
) -> bool:
    return not any(start < boundary < end for start, end in spans)


def _adjust_boundary_around_protected_spans(
    text: str,
    preferred: int,
    candidates: Iterable[int],
    max_length: int,
    minimum_fragment_length: int,
    grammatical_spans: Iterable[tuple[int, int]],
    *,
    two_character_word_spans: Iterable[tuple[int, int]] = (),
) -> int | None:
    """Relocate an already-required unsafe split using unchanged safeguards."""
    candidate_boundaries = tuple(candidates)
    grammar_spans = tuple(grammatical_spans)
    word_spans = tuple(two_character_word_spans)
    protected_spans = grammar_spans + word_spans
    grammar_span = next(
        (
            (start, end)
            for start, end in grammar_spans
            if start < preferred < end
        ),
        None,
    )
    word_span = next(
        (
            (start, end)
            for start, end in word_spans
            if start < preferred < end
        ),
        None,
    )
    if grammar_span is None and word_span is None:
        return preferred

    safe_candidates = tuple(
        boundary
        for boundary in candidate_boundaries
        if _boundary_is_outside_spans(boundary, protected_spans)
    )
    safe_boundary = _choose_balanced_boundary(
        text,
        safe_candidates,
        max_length,
        minimum_fragment_length,
        preferred_boundary=preferred,
    )

    if grammar_span is not None:
        chain_start = grammar_span[0]
        chain_start_boundary = _choose_balanced_boundary(
            text,
            (
                chain_start,
            )
            if (
                chain_start in candidate_boundaries
                and _boundary_is_outside_spans(
                    chain_start,
                    protected_spans,
                )
            )
            else (),
            max_length,
            minimum_fragment_length,
            preferred_boundary=preferred,
        )
        if chain_start_boundary is not None:
            if safe_boundary is None:
                return chain_start_boundary
            chain_start_parts = _split_parts(text, chain_start_boundary)
            safe_parts = _split_parts(text, safe_boundary)
            if chain_start_parts is not None and safe_parts is not None:
                chain_start_within_limit = all(
                    len(part) <= max_length for part in chain_start_parts
                )
                safe_within_limit = all(
                    len(part) <= max_length for part in safe_parts
                )
                chain_start_imbalance = abs(
                    len(chain_start_parts[0]) - len(chain_start_parts[1])
                )
                safe_imbalance = abs(len(safe_parts[0]) - len(safe_parts[1]))
                if (
                    chain_start_within_limit
                    and not safe_within_limit
                ) or (
                    chain_start_within_limit == safe_within_limit
                    and chain_start_imbalance <= safe_imbalance
                ):
                    return chain_start_boundary

    # Preserve the existing targeted two-character behavior: when grammar did
    # not also reject the boundary, first try the two adjacent token edges.
    if grammar_span is None and word_span is not None:
        adjacent_boundary = _choose_balanced_boundary(
            text,
            (
                boundary
                for boundary in word_span
                if (
                    boundary in candidate_boundaries
                    and _boundary_is_outside_spans(boundary, protected_spans)
                )
            ),
            max_length,
            minimum_fragment_length,
            preferred_boundary=preferred,
        )
        if adjacent_boundary is not None:
            return adjacent_boundary

    return safe_boundary


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
    grammatical_spans = _protected_grammatical_chain_spans(cleaned, language)

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
        boundary = _adjust_boundary_around_protected_spans(
            cleaned,
            boundary,
            whitespace_candidates,
            max_length,
            required_fragment_length,
            grammatical_spans,
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
        word_boundary = _adjust_boundary_around_protected_spans(
            cleaned,
            word_boundary,
            word_candidates,
            max_length,
            required_fragment_length,
            grammatical_spans,
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
        whitespace_boundary = _adjust_boundary_around_protected_spans(
            cleaned,
            whitespace_boundary,
            whitespace_candidates,
            max_length,
            required_fragment_length,
            grammatical_spans,
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
    character_boundary = _adjust_boundary_around_protected_spans(
        cleaned,
        character_boundary,
        character_candidates,
        max_length,
        required_fragment_length,
        grammatical_spans,
        two_character_word_spans=_middle_two_character_word_spans(cleaned),
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
