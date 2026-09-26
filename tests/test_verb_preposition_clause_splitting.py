from __future__ import annotations

import math

import pytest

from lyrics_dashboard import splitter


def _first_side(source: str, maximum: int) -> str:
    return splitter.split_lyric_result(
        source,
        "nl",
        maximum,
        minimum_fragment_length=2,
        minimum_fragment_ratio=0.25,
    ).text


def _assert_preserved_and_balanced(source: str, result: str) -> None:
    assert result.count("//") <= 1
    assert result.replace("//", " ").split() == source.split()
    if "//" in result:
        left, right = result.split("//")
        assert min(len(left), len(right)) >= max(2, math.ceil(len(source) * 0.25))


@pytest.mark.parametrize("maximum", (32, 40))
def test_reported_verlangt_naar_water_ends_before_zo(maximum: int) -> None:
    source = "Als een hert dat verlangt naar water zo verlangt mijn ziel naar U"
    result = _first_side(source, maximum)

    assert result == "Als een hert dat verlangt naar water//zo verlangt mijn ziel naar U"
    assert "verlangt//naar water" not in result
    assert "verlangt naar//water" not in result
    _assert_preserved_and_balanced(source, result)


def test_one_local_span_rejects_both_internal_verb_preposition_edges() -> None:
    source = "Als een hert dat verlangt naar water zo verlangt mijn ziel naar U"
    tokens = splitter._grammatical_token_spans(source, "nl")
    spans = splitter._protected_latin_verb_preposition_spans(tokens)
    verb_start = source.index("verlangt")
    after_verb = source.index(" naar")
    after_preposition = source.index(" water")
    phrase_end = source.index(" water") + len(" water")

    assert (verb_start, phrase_end) in spans
    assert not splitter._boundary_is_outside_spans(after_verb, spans)
    assert not splitter._boundary_is_outside_spans(after_preposition, spans)
    assert phrase_end < source.index("zo")


@pytest.mark.parametrize(
    ("source", "maximum", "phrase"),
    (
        ("Wij verlangen naar water en leven verder", 20, "verlangen naar water"),
        ("Wij geloven in de Heer en zingen opnieuw", 20, "geloven in de Heer"),
        ("Wij vertrouwen op Uw woord en zingen vandaag", 20, "vertrouwen op Uw woord"),
        ("Wij bidden om vrede en zingen vandaag", 20, "bidden om vrede"),
        ("Wij roepen tot de Heer en zingen vandaag", 20, "roepen tot de Heer"),
    ),
)
def test_dutch_verb_preposition_complement_stays_together(
    source: str, maximum: int, phrase: str
) -> None:
    result = _first_side(source, maximum)

    assert "//" in result
    assert phrase in result
    _assert_preserved_and_balanced(source, result)


@pytest.mark.parametrize(
    ("source", "phrase"),
    (
        ("We long for living water so we sing again", "long for living water"),
        ("We listen to Your word and follow You", "listen to Your word"),
        ("We believe in the Lord and sing again", "believe in the Lord"),
        ("We trust in Your grace and sing again", "trust in Your grace"),
    ),
)
def test_english_verb_preposition_complement_stays_together(
    source: str, phrase: str
) -> None:
    result = _first_side(source, 20)

    assert "//" in result
    assert phrase in result
    _assert_preserved_and_balanced(source, result)


def test_zo_wins_only_a_nearby_balance_tie() -> None:
    source = "Hij leeft zo wij zingen"
    candidates = tuple(index for index, char in enumerate(source) if char == " ")
    minimum = max(2, math.ceil(len(source) * 0.25))
    initial = splitter._choose_balanced_boundary(source, candidates, 10, minimum)

    assert initial == source.index(" wij")
    assert _first_side(source, 10) == "Hij leeft//zo wij zingen"


def test_zo_does_not_create_a_split_or_override_balance() -> None:
    assert _first_side("Hij leeft zo wij zingen", 40) == "Hij leeft zo wij zingen"

    source = "Hi zo alle mensen zingen vandaag in vrede"
    result = _first_side(source, 12)
    assert "Hi//zo" not in result
    _assert_preserved_and_balanced(source, result)


@pytest.mark.parametrize(
    "source",
    ("Wij verlangen naar zo zingen wij", "Wij verlangen naar de zo zingen wij"),
)
def test_clause_start_stops_a_recognised_complement(source: str) -> None:
    tokens = splitter._grammatical_token_spans(source, "nl")

    assert splitter._protected_latin_verb_preposition_spans(tokens) == ()


def test_existing_latin_chain_and_threshold_rules_still_apply() -> None:
    assert _first_side("genade trouw in het licht", 10) == (
        "genade trouw//in het licht"
    )
    assert _first_side("Wij verlangen naar water", 40) == (
        "Wij verlangen naar water"
    )
