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
_CHINESE_SPACE_MINIMUM_FRAGMENT_RATIO = 0.25
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
_DUTCH_UW_FOLLOWING_VERBS = frozenset(
    "is zijn bent ben was waren wordt worden blijft blijven".split()
)
_MAX_DUTCH_UW_NOUN_PHRASE_TOKENS = 3
_DUTCH_INDIRECT_OBJECT_PRONOUNS = frozenset(
    "mij me jou je u hem haar ons jullie hen hun".split()
)
_DUTCH_POSSESSIVE_DETERMINERS = frozenset(
    "mijn jouw uw zijn haar ons onze jullie hun".split()
)
_ENGLISH_ARTICLES_AND_DETERMINERS = frozenset({"a", "an", "the"})
_CONTEXTUAL_ENGLISH_POSSESSIVE_DETERMINERS = frozenset({"your"})
_CONTEXTUAL_LATIN_MODIFIERS = frozenset({"new", "nieuw"})
_ENGLISH_PERSONAL_PRONOUNS = frozenset(
    "i me you he him she her it we us they them".split()
)
_ENGLISH_INDIRECT_OBJECT_PRONOUNS = frozenset(
    "me you him her us them".split()
)
_ENGLISH_POSSESSIVE_DETERMINERS = frozenset(
    "my your his her its our their".split()
)
_INDIRECT_OBJECT_SUBJECT_PRONOUNS = frozenset(
    "he hij i ik it jij jullie she u we wij you ze zij".split()
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
_INDIRECT_OBJECT_PREPOSITIONS = frozenset({"aan", "voor", "to", "for"})
_INDIRECT_OBJECT_PRONOUNS = (
    _DUTCH_INDIRECT_OBJECT_PRONOUNS
    | _ENGLISH_INDIRECT_OBJECT_PRONOUNS
)
_INDIRECT_OBJECT_POSSESSIVE_DETERMINERS = (
    _DUTCH_POSSESSIVE_DETERMINERS
    | _ENGLISH_POSSESSIVE_DETERMINERS
)
_INDIRECT_OBJECT_DETERMINERS = frozenset().union(
    _DUTCH_ARTICLES_AND_DETERMINERS,
    _ENGLISH_ARTICLES_AND_DETERMINERS,
    _INDIRECT_OBJECT_POSSESSIVE_DETERMINERS,
)
_INDIRECT_OBJECT_MODIFIERS = frozenset(
    (
        "best beste dear dierbaar dierbare faithful good goed goede great "
        "groot grote heilig heilige holy lief lieve new nieuw nieuwe trouw "
        "trouwe very zeer"
    ).split()
)
_TRANSFER_AND_COMMUNICATION_VERBS = frozenset(
    (
        "breng brengen brengt bracht brachten gebracht geef geven geeft gaf "
        "gaven gegeven schenk schenken schenkt schonk schonken geschonken "
        "vertel vertellen vertelt vertelde vertelden verteld bring bringing "
        "brings brought gave give given gives giving grant granted granting "
        "grants tell telling tells told"
    ).split()
)
_DIRECT_OBJECT_VERBS = _TRANSFER_AND_COMMUNICATION_VERBS | frozenset(
    (
        "teach teaches taught teaching love loves loved loving know knows knew "
        "known knowing see sees saw seen seeing hear hears heard hearing call "
        "calls called calling open opens opened opening make makes made making "
        "keep keeps kept keeping forgive forgives forgave forgiven forgiving "
        "bless blesses blessed blessing receive receives received receiving "
        "follow follows followed following praise praises praised praising "
        "show shows showed shown showing lead leads led leading guide guides "
        "guided guiding save saves saved saving geven gaf geeft gegeven "
        "brengen brengt bracht gebracht schenken schenkt schonk geschonken "
        "vertellen vertelt vertelde verteld leren leert leerde geleerd leiden "
        "leidt leidde geleid gidsen gidst gidste gegidst redden redt redde "
        "gered kennen kent kende gekend zien ziet zag gezien horen hoort "
        "hoorde gehoord roepen roept riep geroepen openen opent opende "
        "geopend maken maakt maakte gemaakt bewaren bewaart bewaarde bewaard "
        "vergeven vergeeft vergaf vergeven zegenen zegent zegende gezegend "
        "ontvangen ontvangt ontving ontvangen volgen volgt volgde gevolgd "
        "prijzen prijst prees geprezen tonen toont toonde getoond"
    ).split()
)
_DIRECT_OBJECT_PRONOUNS = _INDIRECT_OBJECT_PRONOUNS | frozenset({"it", "ons"})
_SHORT_NOUN_PHRASE_DETERMINERS = frozenset().union(
    _DUTCH_ARTICLES_AND_DETERMINERS,
    _ENGLISH_ARTICLES_AND_DETERMINERS,
    _CONTEXTUAL_ENGLISH_POSSESSIVE_DETERMINERS,
)
_DIRECT_OBJECT_NOUN_PHRASE_DETERMINERS = _INDIRECT_OBJECT_DETERMINERS
_NOUN_VERB_HEADS = frozenset({"love", "praise"})
_BARE_DIRECT_OBJECT_HEADS = frozenset(
    (
        "strength hope life love grace peace light name song praise thanks "
        "promise gate kracht hoop leven liefde genade vrede licht naam lied "
        "lof dank belofte"
    ).split()
)
_SHORT_NOUN_PHRASE_HEADS = _BARE_DIRECT_OBJECT_HEADS | frozenset(
    (
        "friend friends people child children smile voice river morning heart "
        "vriend vrienden mensen kind kinderen glimlach stem rivier ochtend hart"
    ).split()
)
_NOUN_ADJUNCT_HINTS = frozenset(
    "life river garden morning heart heaven kingdom water fire soul song gospel leven rivier tuin ochtend hart hemel koninkrijk water vuur ziel lied".split()
)
_LOGICAL_CLAUSE_MARKERS = frozenset(
    "that which who when where because although while dat die wie wanneer waar omdat hoewel terwijl".split()
)
_LOGICAL_CONJUNCTIONS = frozenset("and but or yet so en maar of doch dus".split())
_INDIRECT_OBJECT_CLAUSE_BOUNDARIES = frozenset(
    (
        "als although and as because but dat doordat en hoewel if maar of "
        "omdat or since that though terwijl want wanneer when while"
    ).split()
)
_INDIRECT_OBJECT_STOP_WORDS = frozenset().union(
    _TRANSFER_AND_COMMUNICATION_VERBS,
    _INDIRECT_OBJECT_CLAUSE_BOUNDARIES,
    _INDIRECT_OBJECT_PREPOSITIONS,
    _DUTCH_UW_FOLLOWING_VERBS,
    "am are be been being is was were".split(),
)
_LOCAL_NOUN_PHRASE_BOUNDARIES = frozenset().union(
    _INDIRECT_OBJECT_STOP_WORDS,
    _DIRECT_OBJECT_VERBS,
    _DUTCH_PREPOSITIONS,
    _ENGLISH_PREPOSITIONS,
    _DUTCH_PERSONAL_PRONOUNS,
    _ENGLISH_PERSONAL_PRONOUNS,
    _DIRECT_OBJECT_NOUN_PHRASE_DETERMINERS,
    _LOGICAL_CLAUSE_MARKERS,
    _LOGICAL_CONJUNCTIONS,
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


def _without_equivalent_split(
    text: str,
    candidates: Iterable[int],
    rejected_boundary: int,
) -> tuple[int, ...]:
    """Exclude both whitespace edges when they render one rejected split."""
    rejected_parts = _split_parts(text, rejected_boundary)
    return tuple(
        boundary
        for boundary in candidates
        if _split_parts(text, boundary) != rejected_parts
    )


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


def _protected_uw_noun_verb_spans(
    tokens: tuple[tuple[str, int, int], ...],
) -> tuple[tuple[int, int], ...]:
    """Protect a mid-line ``uw`` noun phrase through its following verb."""
    spans: list[tuple[int, int]] = []
    for uw_index, (token, start, _end) in enumerate(tokens):
        if uw_index == 0 or token.casefold() != "uw":
            continue
        if (
            uw_index + 1 >= len(tokens)
            or tokens[uw_index + 1][0].casefold()
            in _DUTCH_UW_FOLLOWING_VERBS
        ):
            continue

        first_verb_index = uw_index + 2
        verb_search_end = min(
            uw_index + _MAX_DUTCH_UW_NOUN_PHRASE_TOKENS + 2,
            len(tokens),
        )
        for verb_index in range(first_verb_index, verb_search_end):
            if (
                tokens[verb_index][0].casefold()
                not in _DUTCH_UW_FOLLOWING_VERBS
            ):
                continue

            # Include the first token after the verb so the whitespace
            # boundary immediately following the verb is also unsafe.
            construction_end_index = min(verb_index + 1, len(tokens) - 1)
            spans.append((start, tokens[construction_end_index][2]))
            break
    return tuple(spans)


def _indirect_object_head_end(
    tokens: tuple[tuple[str, int, int], ...],
    start_index: int,
) -> int | None:
    """Return the exclusive end of up to two modifiers and one head."""
    index = start_index
    modifier_count = 0
    while (
        index < len(tokens)
        and modifier_count < 2
        and tokens[index][0].casefold() in _INDIRECT_OBJECT_MODIFIERS
    ):
        modifier_count += 1
        index += 1
    if (
        index >= len(tokens)
        or tokens[index][0].casefold() in _INDIRECT_OBJECT_STOP_WORDS
    ):
        return None
    return index + 1


def _indirect_object_noun_phrase_end(
    tokens: tuple[tuple[str, int, int], ...],
    start_index: int,
) -> int | None:
    """Return the exclusive end of one conservative recipient noun phrase."""
    if start_index >= len(tokens):
        return None
    word = tokens[start_index][0].casefold()
    if (
        word in _INDIRECT_OBJECT_POSSESSIVE_DETERMINERS
        and start_index + 1 < len(tokens)
        and tokens[start_index + 1][0].casefold()
        in _INDIRECT_OBJECT_MODIFIERS
    ):
        return _indirect_object_head_end(tokens, start_index + 1)
    if word in _INDIRECT_OBJECT_PRONOUNS:
        return start_index + 1
    if word in _INDIRECT_OBJECT_DETERMINERS:
        return _indirect_object_head_end(tokens, start_index + 1)
    if word in _INDIRECT_OBJECT_STOP_WORDS:
        return None
    return start_index + 1


def _indirect_object_complement_end(
    tokens: tuple[tuple[str, int, int], ...],
    start_index: int,
) -> int | None:
    """Return the bounded local content attached to an object pronoun."""
    if (
        start_index < len(tokens)
        and tokens[start_index][0].casefold()
        in _INDIRECT_OBJECT_DETERMINERS
    ):
        return _indirect_object_noun_phrase_end(tokens, start_index)
    return _indirect_object_head_end(tokens, start_index)


def _local_noun_content(word: str, *, allow_ambiguous_head: bool = False) -> bool:
    """Accept one Latin content token without crossing a clear local boundary."""
    return word.isalpha() and (
        word not in _LOCAL_NOUN_PHRASE_BOUNDARIES
        or (allow_ambiguous_head and word in _NOUN_VERB_HEADS)
    )


def _short_latin_noun_phrase_end(
    tokens: tuple[tuple[str, int, int], ...],
    start_index: int,
    *,
    allow_possessive_determiner: bool = False,
) -> int | None:
    """Return the end of a determiner plus at most two modifiers and a head."""
    determiners = (
        _DIRECT_OBJECT_NOUN_PHRASE_DETERMINERS
        if allow_possessive_determiner
        else _SHORT_NOUN_PHRASE_DETERMINERS
    )
    if (
        start_index + 1 >= len(tokens)
        or tokens[start_index][0].casefold() not in determiners
    ):
        return None

    index = start_index + 1
    modifier_count = 0
    while (
        index < len(tokens)
        and modifier_count < 2
        and tokens[index][0].casefold() in _INDIRECT_OBJECT_MODIFIERS
    ):
        modifier_count += 1
        index += 1

    if modifier_count:
        if (
            index < len(tokens)
            and tokens[index][0].casefold() not in _INDIRECT_OBJECT_MODIFIERS
            and _local_noun_content(
                tokens[index][0].casefold(),
                allow_ambiguous_head=True,
            )
        ):
            return index + 1
        return None

    if not _local_noun_content(
        tokens[index][0].casefold(), allow_ambiguous_head=True
    ):
        return None
    # A small head vocabulary permits noun+noun forms such as "the life gate"
    # without treating arbitrary following words as part of the noun phrase.
    if (
        index + 1 < len(tokens)
        and tokens[index + 1][0].casefold() not in _INDIRECT_OBJECT_MODIFIERS
        and (
            tokens[index + 1][0].casefold() in _SHORT_NOUN_PHRASE_HEADS
            or tokens[index][0].casefold() in _NOUN_ADJUNCT_HINTS
        )
        and _local_noun_content(tokens[index + 1][0].casefold())
    ):
        return index + 2
    return index + 1


def _is_transfer_recipient_with_complement(
    tokens: tuple[tuple[str, int, int], ...],
    pronoun_index: int,
) -> bool:
    """Keep the existing give/bring/grant recipient handling intact."""
    return (
        pronoun_index > 0
        and tokens[pronoun_index - 1][0].casefold()
        in _TRANSFER_AND_COMMUNICATION_VERBS
        and _indirect_object_complement_end(tokens, pronoun_index + 1)
        is not None
    )


def _is_direct_object_pronoun_after_verb(
    tokens: tuple[tuple[str, int, int], ...],
    pronoun_index: int,
) -> bool:
    if pronoun_index == 0:
        return False
    return (
        tokens[pronoun_index][0].casefold() in _DIRECT_OBJECT_PRONOUNS
        and tokens[pronoun_index - 1][0].casefold() in _DIRECT_OBJECT_VERBS
        and not _is_transfer_recipient_with_complement(tokens, pronoun_index)
    )


def _protected_short_latin_noun_phrase_spans(
    tokens: tuple[tuple[str, int, int], ...],
) -> tuple[tuple[int, int], ...]:
    spans = []
    for index, (_word, start, _end) in enumerate(tokens):
        phrase_end = _short_latin_noun_phrase_end(tokens, index)
        if phrase_end is not None:
            spans.append((start, tokens[phrase_end - 1][2]))
    return tuple(spans)


def _protected_latin_direct_object_spans(
    tokens: tuple[tuple[str, int, int], ...],
) -> tuple[tuple[int, int], ...]:
    """Protect only an adjacent recognised verb and one bounded object."""
    spans = []
    for index, (verb, start, _end) in enumerate(tokens[:-1]):
        if verb.casefold() not in _DIRECT_OBJECT_VERBS:
            continue
        object_index = index + 1
        object_word = tokens[object_index][0].casefold()
        if object_word in _DIRECT_OBJECT_PRONOUNS:
            if _is_transfer_recipient_with_complement(tokens, object_index):
                object_end = None
            else:
                possessive_end = _short_latin_noun_phrase_end(
                    tokens,
                    object_index,
                    allow_possessive_determiner=True,
                )
                if (
                    possessive_end is not None
                    and (
                        possessive_end > object_index + 2
                        or tokens[possessive_end - 1][0].casefold()
                        in _SHORT_NOUN_PHRASE_HEADS
                    )
                ):
                    object_end = possessive_end
                else:
                    object_end = object_index + 1
        elif object_word in _DIRECT_OBJECT_NOUN_PHRASE_DETERMINERS:
            object_end = _short_latin_noun_phrase_end(
                tokens,
                object_index,
                allow_possessive_determiner=True,
            )
            if (
                object_end is not None
                and verb.casefold() in _TRANSFER_AND_COMMUNICATION_VERBS
                and object_word in _INDIRECT_OBJECT_POSSESSIVE_DETERMINERS
            ):
                object_end = None
        elif object_word in _BARE_DIRECT_OBJECT_HEADS:
            object_end = object_index + 1
        else:
            object_end = None
        if object_end is not None:
            spans.append((start, tokens[object_end - 1][2]))
    return tuple(spans)


def _has_nearby_transfer_verb(
    tokens: tuple[tuple[str, int, int], ...],
    phrase_start_index: int,
) -> bool:
    """Recognise only a transfer verb in the preceding four local tokens."""
    search_start = max(0, phrase_start_index - 4)
    for index in range(phrase_start_index - 1, search_start - 1, -1):
        word = tokens[index][0].casefold()
        if word in _INDIRECT_OBJECT_CLAUSE_BOUNDARIES:
            return False
        if word in _TRANSFER_AND_COMMUNICATION_VERBS:
            return True
        if word in _INDIRECT_OBJECT_STOP_WORDS:
            return False
    return False


def _has_direct_transfer_context(
    tokens: tuple[tuple[str, int, int], ...],
    phrase_start_index: int,
) -> bool:
    """Recognise an immediate transfer verb or one local inversion shape."""
    if phrase_start_index == 0:
        return False
    previous_word = tokens[phrase_start_index - 1][0].casefold()
    if previous_word in _TRANSFER_AND_COMMUNICATION_VERBS:
        return True
    return (
        phrase_start_index >= 2
        and previous_word in _INDIRECT_OBJECT_SUBJECT_PRONOUNS
        and tokens[phrase_start_index - 2][0].casefold()
        in _TRANSFER_AND_COMMUNICATION_VERBS
    )


def _protected_latin_indirect_object_spans(
    tokens: tuple[tuple[str, int, int], ...],
) -> tuple[tuple[int, int], ...]:
    """Return bounded, exact-token recipient phrases near transfer verbs."""
    spans: set[tuple[int, int]] = set()
    for index, (token, start, _end) in enumerate(tokens):
        word = token.casefold()
        follows_indirect_preposition = (
            index > 0
            and tokens[index - 1][0].casefold()
            in _INDIRECT_OBJECT_PREPOSITIONS
        )
        has_transfer_context = _has_nearby_transfer_verb(tokens, index)
        has_direct_transfer_context = _has_direct_transfer_context(
            tokens,
            index,
        )

        phrase_end_index: int | None = None
        if (
            word in _INDIRECT_OBJECT_PREPOSITIONS
            and (index == 0 or has_transfer_context)
        ):
            phrase_end_index = _indirect_object_noun_phrase_end(
                tokens,
                index + 1,
            )
        elif (
            word in _INDIRECT_OBJECT_PRONOUNS
            and not follows_indirect_preposition
            and (index == 0 or has_direct_transfer_context)
        ):
            phrase_end_index = _indirect_object_complement_end(
                tokens,
                index + 1,
            )
        elif (
            word in _INDIRECT_OBJECT_DETERMINERS
            and not follows_indirect_preposition
            and has_direct_transfer_context
        ):
            phrase_end_index = _indirect_object_head_end(
                tokens,
                index + 1,
            )

        if (
            phrase_end_index is not None
            and phrase_end_index > index + 1
            and phrase_end_index < len(tokens)
        ):
            spans.add((start, tokens[phrase_end_index - 1][2]))
    return tuple(sorted(spans))


def _protected_grammatical_chain_spans(
    text: str,
    language: LanguageCode,
) -> tuple[tuple[int, int], ...]:
    """Keep pronoun, determiner, and preposition runs with the next token."""
    tokens = _grammatical_token_spans(text, language)
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(tokens):
        if (
            language == "nl"
            and _is_direct_object_pronoun_after_verb(tokens, index)
        ):
            # The direct object completes the preceding verb phrase; the
            # generic pronoun lead must not absorb the next clause or repeat.
            index += 1
            continue
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
    if language == "nl":
        spans.extend(_protected_latin_indirect_object_spans(tokens))
        spans.extend(_protected_uw_noun_verb_spans(tokens))
        spans.extend(_protected_short_latin_noun_phrase_spans(tokens))
        spans.extend(_protected_latin_direct_object_spans(tokens))
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


def _logical_latin_boundary_value(
    text: str,
    boundary: int,
    tokens: tuple[tuple[str, int, int], ...],
    phrase_spans: tuple[tuple[int, int], ...],
) -> int:
    """Recognise a few local clause and complete-phrase edges."""
    right_index = next(
        (index for index, (_word, start, _end) in enumerate(tokens) if start >= boundary),
        None,
    )
    if right_index is None or right_index == 0:
        return 0
    right_word = tokens[right_index][0].casefold()
    left_word = tokens[right_index - 1][0].casefold()
    value = 0
    if right_word in _LOGICAL_CLAUSE_MARKERS:
        value = 2
    elif (
        right_word in _LOGICAL_CONJUNCTIONS
        or left_word in _LOGICAL_CONJUNCTIONS
    ):
        value = 1

    if any(
        end <= boundary and text[end:boundary].isspace()
        for _start, end in phrase_spans
    ):
        value = max(value, 1)

    if right_index + 1 < len(tokens):
        next_word = tokens[right_index + 1][0].casefold()
        if (
            _local_noun_content(right_word)
            and _local_noun_content(next_word)
            and any(
                tokens[index][0].casefold() == right_word
                and tokens[index + 1][0].casefold() == next_word
                for index in range(right_index - 1)
            )
        ):
            value = max(value, 2)
    return value


def _prefer_logical_latin_boundary(
    text: str,
    selected: int,
    preferred: int,
    candidates: Iterable[int],
    max_length: int,
    minimum_fragment_length: int,
    protected_spans: tuple[tuple[int, int], ...],
) -> int:
    """Break a close balance tie without relaxing any existing split guard."""
    selected_parts = _split_parts(text, selected)
    if selected_parts is None:
        return selected
    selected_imbalance = abs(len(selected_parts[0]) - len(selected_parts[1]))
    selected_within_limit = all(len(part) <= max_length for part in selected_parts)
    tokens = _grammatical_token_spans(text, "nl")
    phrase_spans = (
        _protected_short_latin_noun_phrase_spans(tokens)
        + _protected_latin_direct_object_spans(tokens)
    )
    selected_value = _logical_latin_boundary_value(
        text, selected, tokens, phrase_spans
    )
    alternatives: list[tuple[int, int, int, int]] = []
    for boundary in set(candidates):
        if (
            boundary == selected
            or abs(boundary - selected) > 6
            or abs(boundary - preferred) > abs(selected - preferred) + 5
            or not _boundary_is_outside_spans(boundary, protected_spans)
        ):
            continue
        parts = _split_parts(text, boundary)
        if parts is None:
            continue
        if min(map(len, parts)) < max(
            minimum_fragment_length,
            math.ceil(len(text) * 0.25),
        ):
            continue
        if all(len(part) <= max_length for part in parts) != selected_within_limit:
            continue
        imbalance = abs(len(parts[0]) - len(parts[1]))
        if imbalance > selected_imbalance + 2:
            continue
        value = _logical_latin_boundary_value(
            text, boundary, tokens, phrase_spans
        )
        if value > selected_value:
            alternatives.append(
                (-value, imbalance, abs(boundary - preferred), boundary)
            )
    return min(alternatives)[3] if alternatives else selected


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
    # Keep the existing split trigger's single-space allowance separate from
    # the priority given to all source-space candidates below.
    single_space_boundary: int | None = None
    if (
        language == "zh"
        and cleaned.count(" ") == 1
        and len(cleaning.original_whitespace_boundaries) == 1
    ):
        candidate = cleaning.original_whitespace_boundaries[0]
        if _split_parts(cleaned, candidate) is not None:
            single_space_boundary = candidate

    # The separator can place an already-required split, but cannot require it.
    threshold_length = len(cleaned) - (
        1 if single_space_boundary is not None else 0
    )
    if not cleaned or threshold_length <= max_length:
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
        preferred_boundary = boundary
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
        boundary = _prefer_logical_latin_boundary(
            cleaned,
            boundary,
            preferred_boundary,
            whitespace_candidates,
            max_length,
            required_fragment_length,
            grammatical_spans,
        )
        left, right = _split_parts(cleaned, boundary) or (cleaned, "")
        return SplitResult(f"{left}//{right}" if right else left)

    word_candidates = tuple(
        boundary
        for boundary in _chinese_word_boundaries(cleaned)
        if boundary not in cleaning.punctuation_separator_boundaries
    )
    space_candidates = cleaning.original_whitespace_boundaries
    if space_candidates:
        preferred_fragment_length = max(
            required_fragment_length,
            math.ceil(
                len(cleaned) * _CHINESE_SPACE_MINIMUM_FRAGMENT_RATIO
            ),
        )
        safe_space_candidates = tuple(
            boundary
            for boundary in space_candidates
            if _boundary_is_outside_spans(boundary, grammatical_spans)
        )
        preferred_boundary = _choose_balanced_boundary(
            cleaned,
            safe_space_candidates,
            max_length,
            preferred_fragment_length,
        )
        if preferred_boundary is not None:
            ranked_boundary = _choose_balanced_boundary(
                cleaned,
                (*word_candidates, *safe_space_candidates),
                max_length,
                required_fragment_length,
                preferred_boundary=preferred_boundary,
            )
            if ranked_boundary == preferred_boundary:
                left, right = _split_parts(
                    cleaned,
                    preferred_boundary,
                ) or (cleaned, "")
                return SplitResult(f"{left}//{right}" if right else left)
        for space_boundary in space_candidates:
            word_candidates = _without_equivalent_split(
                cleaned,
                word_candidates,
                space_boundary,
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
    for space_boundary in space_candidates:
        whitespace_candidates = _without_equivalent_split(
            cleaned,
            whitespace_candidates,
            space_boundary,
        )
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
    for space_boundary in space_candidates:
        character_candidates = _without_equivalent_split(
            cleaned,
            character_candidates,
            space_boundary,
        )
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
