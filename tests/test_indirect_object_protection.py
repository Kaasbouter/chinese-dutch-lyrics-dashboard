from __future__ import annotations

import math
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from lyrics_dashboard.alignment import build_exact_manual_plan
from lyrics_dashboard.converter import (
    ConversionSettings,
    convert_lyrics,
    encode_utf8_txt,
)
from lyrics_dashboard.models import (
    AlignedLine,
    AlignmentPlan,
    ParsedLyrics,
    TranslationReference,
)
from lyrics_dashboard.parser import parse_lyrics
from lyrics_dashboard.splitter import split_lyric, split_lyric_result


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _first_side_result(source: str, maximum: int) -> str:
    return split_lyric_result(
        source,
        "nl",
        maximum,
        minimum_fragment_length=2,
        minimum_fragment_ratio=0.25,
    ).text


def _assert_preserved(source: str, result: str) -> None:
    assert result.count("//") <= 1
    assert result.replace("//", " ").split() == source.split()


@pytest.mark.parametrize(
    ("source", "expected", "forbidden"),
    (
        (
            "Hij geeft mij nieuwe hoop",
            "Hij geeft//mij nieuwe hoop",
            ("mij//nieuwe hoop",),
        ),
        (
            "God schenkt ons nieuwe kracht",
            "God schenkt//ons nieuwe kracht",
            ("ons//nieuwe kracht",),
        ),
        (
            "Hij geeft mij nieuwe hoop voor iedere dag",
            "Hij geeft mij nieuwe hoop//voor iedere dag",
            ("mij//nieuwe hoop", "mij nieuwe//hoop"),
        ),
        (
            "God schenkt ons nieuwe kracht voor iedere dag",
            "God schenkt ons nieuwe kracht//voor iedere dag",
            ("ons//nieuwe kracht", "ons nieuwe//kracht"),
        ),
        (
            "Hij geeft haar een belofte voor altijd",
            "Hij geeft haar een belofte//voor altijd",
            ("haar//een belofte", "haar een//belofte"),
        ),
        (
            "Hij geeft hun goed nieuws voor vandaag",
            "Hij geeft hun goed nieuws//voor vandaag",
            ("hun//goed nieuws", "hun goed//nieuws"),
        ),
    ),
)
def test_dutch_object_pronoun_phrase_stays_complete(
    source: str,
    expected: str,
    forbidden: tuple[str, ...],
) -> None:
    result = split_lyric(source, "nl", 4)

    assert result == expected
    assert all(boundary not in result for boundary in forbidden)
    _assert_preserved(source, result)


@pytest.mark.parametrize(
    ("source", "expected", "forbidden"),
    (
        (
            "Wij brengen aan de mensen goed nieuws",
            "Wij brengen//aan de mensen goed nieuws",
            ("aan//de mensen", "aan de//mensen"),
        ),
        (
            "Hij vertelt aan mijn vriend het verhaal",
            "Hij vertelt//aan mijn vriend het verhaal",
            ("aan mijn//vriend",),
        ),
        (
            "Hij vertelt aan mijn goede vriend het verhaal vandaag",
            "Hij vertelt aan mijn goede vriend//het verhaal vandaag",
            ("aan mijn//goede vriend", "aan mijn goede//vriend"),
        ),
        (
            "Hij vertelt aan haar goede vriend een verhaal vandaag",
            "Hij vertelt aan haar goede vriend//een verhaal vandaag",
            ("aan haar//goede vriend", "aan haar goede//vriend"),
        ),
        (
            "Wij brengen voor Uw volk goed nieuws",
            "Wij brengen//voor Uw volk goed nieuws",
            ("voor//Uw volk", "voor Uw//volk"),
        ),
        (
            "God schenkt mijn goede vriend nieuwe hoop",
            "God schenkt//mijn goede vriend nieuwe hoop",
            ("mijn//goede vriend", "mijn goede//vriend"),
        ),
        (
            "God schenkt onze kinderen nieuwe hoop vandaag",
            "God schenkt onze kinderen//nieuwe hoop vandaag",
            ("onze//kinderen",),
        ),
        (
            "Hij vertelt aan mijn zeer goede vriend een verhaal",
            "Hij vertelt//aan mijn zeer goede vriend een verhaal",
            (
                "aan mijn zeer//goede vriend",
                "aan mijn zeer goede//vriend",
            ),
        ),
    ),
)
def test_dutch_prepositional_and_determiner_recipient_stays_complete(
    source: str,
    expected: str,
    forbidden: tuple[str, ...],
) -> None:
    result = split_lyric(source, "nl", 4)

    assert result == expected
    assert all(boundary not in result for boundary in forbidden)
    _assert_preserved(source, result)


@pytest.mark.parametrize(
    ("source", "expected", "forbidden"),
    (
        (
            "He gives me new hope",
            "He gives//me new hope",
            ("me//new hope",),
        ),
        (
            "God grants us new strength",
            "God grants//us new strength",
            ("us//new strength",),
        ),
        (
            "He gives me new hope for every day",
            "He gives me new hope//for every day",
            ("me//new hope", "me new//hope"),
        ),
        (
            "God grants us new strength for every day",
            "God grants//us new strength for every day",
            ("us//new strength", "us new//strength"),
        ),
        (
            "God gives her a promise for every day",
            "God gives//her a promise for every day",
            ("her//a promise", "her a//promise"),
        ),
        (
            "God gives them good news for today",
            "God gives//them good news for today",
            ("them//good news", "them good//news"),
        ),
    ),
)
def test_english_object_pronoun_phrase_stays_complete(
    source: str,
    expected: str,
    forbidden: tuple[str, ...],
) -> None:
    result = split_lyric(source, "nl", 4)

    assert result == expected
    assert all(boundary not in result for boundary in forbidden)
    _assert_preserved(source, result)


@pytest.mark.parametrize(
    ("source", "expected", "forbidden"),
    (
        (
            "We bring to the people good news",
            "We bring to the people//good news",
            ("to//the people", "to the//people"),
        ),
        (
            "He tells to my friend the story",
            "He tells//to my friend the story",
            ("to my//friend",),
        ),
        (
            "He tells to my good friend the story today",
            "He tells to my good friend//the story today",
            ("to my//good friend", "to my good//friend"),
        ),
        (
            "He tells to her good friend a story today",
            "He tells to her good friend//a story today",
            ("to her//good friend", "to her good//friend"),
        ),
        (
            "We bring for Your people good news",
            "We bring for Your people//good news",
            ("for//Your people", "for Your//people"),
        ),
        (
            "God gives my good friend new hope",
            "God gives//my good friend new hope",
            ("my//good friend", "my good//friend"),
        ),
        (
            "God grants our children new hope today",
            "God grants//our children new hope today",
            ("our//children",),
        ),
        (
            "He tells to my very good friend a story",
            "He tells//to my very good friend a story",
            (
                "to my very//good friend",
                "to my very good//friend",
            ),
        ),
    ),
)
def test_english_prepositional_and_determiner_recipient_stays_complete(
    source: str,
    expected: str,
    forbidden: tuple[str, ...],
) -> None:
    result = split_lyric(source, "nl", 4)

    assert result == expected
    assert all(boundary not in result for boundary in forbidden)
    _assert_preserved(source, result)


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (
            "We bring to my good friend a story",
            "We bring//to my good friend a story",
        ),
        (
            "He tells to my good friend a story",
            "He tells//to my good friend a story",
        ),
        (
            "We bring to the people news",
            "We bring//to the people news",
        ),
        (
            "Wij brengen voor Uw volk hoop",
            "Wij brengen//voor Uw volk hoop",
        ),
    ),
)
def test_safe_boundary_before_complete_recipient_is_preferred(
    source: str,
    expected: str,
) -> None:
    result = split_lyric(source, "nl", 20)

    assert result == expected
    _assert_preserved(source, result)


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (
            "Hij geeft mij nieuwe hoop voor iedere dag",
            "Hij geeft mij nieuwe hoop//voor iedere dag",
        ),
        (
            "Wij brengen aan de mensen het goede nieuws",
            "Wij brengen aan de mensen//het goede nieuws",
        ),
        (
            "He gives me new hope for every day",
            "He gives me new hope//for every day",
        ),
        (
            "We bring to the people the good news",
            "We bring to the people//the good news",
        ),
    ),
)
def test_safe_boundary_after_complete_recipient_remains_allowed(
    source: str,
    expected: str,
) -> None:
    result = split_lyric(source, "nl", 20)

    assert result == expected
    _assert_preserved(source, result)


@pytest.mark.parametrize(
    "source",
    (
        "Hij geeft mij nieuwe hoop",
        "Wij brengen aan de mensen goed nieuws",
        "He gives me new hope",
        "We bring to the people good news",
    ),
)
def test_indirect_object_never_creates_a_new_split(source: str) -> None:
    assert split_lyric(source, "nl", 80) == source


@pytest.mark.parametrize(
    ("source", "maximum", "expected", "forbidden"),
    (
        (
            "mij nieuwe hoop vandaag",
            20,
            "mij nieuwe hoop//vandaag",
            ("mij//nieuwe", "mij nieuwe//hoop"),
        ),
        (
            "aan mijn goede vriend vandaag",
            20,
            "aan mijn goede vriend//vandaag",
            (
                "aan//mijn goede vriend",
                "aan mijn//goede vriend",
                "aan mijn goede//vriend",
            ),
        ),
        (
            "me new hope today",
            16,
            "me new hope//today",
            ("me//new", "me new//hope"),
        ),
        (
            "to my good friend today",
            20,
            "to my good friend//today",
            (
                "to//my good friend",
                "to my//good friend",
                "to my good//friend",
            ),
        ),
    ),
)
def test_phrase_at_line_start_never_creates_a_leading_split(
    source: str,
    maximum: int,
    expected: str,
    forbidden: tuple[str, ...],
) -> None:
    result = split_lyric(source, "nl", maximum)

    assert result == expected
    assert not result.startswith("//")
    assert all(boundary not in result for boundary in forbidden)
    _assert_preserved(source, result)


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (
            "God geeft mij nieuwe hoop en vandaag blijven wij vol vertrouwen "
            "verder zingen",
            "God geeft mij nieuwe hoop en vandaag//"
            "blijven wij vol vertrouwen verder zingen",
        ),
        (
            "God gives me new hope and today we keep singing together with "
            "confidence",
            "God gives me new hope and today//"
            "we keep singing together with confidence",
        ),
    ),
)
def test_indirect_object_elsewhere_does_not_move_safe_planned_boundary(
    source: str,
    expected: str,
) -> None:
    result = _first_side_result(source, 24)

    assert result == expected
    _assert_preserved(source, result)


def test_existing_balance_uses_after_edge_when_before_edge_is_worse() -> None:
    source = "Hij geeft mij nieuwe hoop voor iedere dag"
    result = _first_side_result(source, 4)

    assert result == "Hij geeft mij nieuwe hoop//voor iedere dag"
    assert min(map(len, result.split("//"))) >= math.ceil(len(source) * 0.25)


def test_invalid_adjacent_edges_use_closest_other_safe_boundary() -> None:
    source = "Vandaag geeft Hij mij nieuwe hoop voor morgen"
    result = _first_side_result(source, 4)

    assert result == "Vandaag geeft//Hij mij nieuwe hoop voor morgen"
    assert "mij//nieuwe hoop" not in result
    assert "mij nieuwe//hoop" not in result
    _assert_preserved(source, result)


@pytest.mark.parametrize(
    "source",
    (
        "x gaf mij nieuwe hoop y",
        "gave me new hope y",
        "aa gaf mij superlangwoord z",
    ),
)
def test_line_remains_unsplit_when_no_balanced_safe_candidate_exists(
    source: str,
) -> None:
    result = _first_side_result(source, 4)

    assert result == source
    assert "//" not in result


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (
            "Wij zien haar mooie glimlach iedere dag",
            "Wij zien haar mooie//glimlach iedere dag",
        ),
        (
            "We see her bright smile every day",
            "We see her bright//smile every day",
        ),
        (
            "He forgives me new hope for every day",
            "He forgives me new//hope for every day",
        ),
        (
            "He gives thanks while her good voice sings softly through",
            "He gives thanks while her good//voice sings softly through",
        ),
    ),
)
def test_unrecognised_verbs_do_not_create_indirect_object_spans(
    source: str,
    expected: str,
) -> None:
    assert split_lyric(source, "nl", 20) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (
            "God geeft hoop aan mij vandaag",
            "God geeft hoop//aan mij vandaag",
        ),
        (
            "God geeft hoop voor ons vandaag",
            "God geeft hoop//voor ons vandaag",
        ),
        (
            "God gives hope to me today",
            "God gives hope//to me today",
        ),
        (
            "God gives hope for us today",
            "God gives hope//for us today",
        ),
    ),
)
def test_preposition_plus_pronoun_recipient_stays_complete(
    source: str,
    expected: str,
) -> None:
    result = split_lyric(source, "nl", 20)

    assert result == expected
    _assert_preserved(source, result)


def test_middle_only_extension_does_not_expand_at_line_ending() -> None:
    source = "God schenkt mijn goede vriend"

    assert split_lyric(source, "nl", 20) == (
        "God schenkt mijn//goede vriend"
    )


def test_existing_latin_and_chinese_protections_remain_unchanged() -> None:
    assert split_lyric("dit is uw belofte", "nl", 10) == (
        "dit is//uw belofte"
    )
    assert split_lyric("genade trouw in het licht", "nl", 13) == (
        "genade trouw//in het licht"
    )
    assert split_lyric("we sing the holy song", "nl", 10) == (
        "we sing//the holy song"
    )
    assert (
        split_lyric_result(
            "天地玄黃 祢是永遠君王萬歲",
            "zh",
            8,
            minimum_fragment_length=2,
            minimum_fragment_ratio=0.25,
        ).text
        == "天地玄黃//祢是永遠君王萬歲"
    )
    assert split_lyric("甲乙在祢裡面丙丁", "zh", 4) == (
        "甲乙//在祢裡面丙丁"
    )
    assert split_lyric("賜我永平安直到永遠", "zh", 4) == (
        "賜我永平安//直到永遠"
    )


def _bilingual_pair(dutch_line: str) -> tuple[ParsedLyrics, AlignmentPlan]:
    parsed = parse_lyrics(
        "中文歌名 | Nederlandse titel\n\n"
        "Verse 1\n"
        "短歌\n\n"
        "Verse 2\n"
        f"{dutch_line}\n"
    )
    plan = build_exact_manual_plan(
        parsed,
        {0: (1,)},
        {
            0: (
                AlignedLine(
                    source_line_indices=(0,),
                    translation_references=(
                        TranslationReference(1, (0,)),
                    ),
                ),
            ),
        },
    )
    return parsed, plan


def test_indirect_object_rule_works_on_both_pipe_sides_after_switch() -> None:
    dutch = "Hij geeft mij nieuwe hoop voor iedere dag"
    protected = "Hij geeft mij nieuwe hoop//voor iedere dag"
    parsed, plan = _bilingual_pair(dutch)

    output = convert_lyrics(
        parsed,
        plan,
        ConversionSettings(
            switch_index=1,
            chinese_max_length=40,
            dutch_max_length=40,
        ),
    )
    lyric_rows = [line for line in output.splitlines() if "|" in line]

    assert lyric_rows == [
        f"短歌|{protected}",
        f"{protected}|短歌",
    ]
    assert all(
        side.count("//") <= 1
        for row in lyric_rows
        for side in row.split("|")
    )
    for row in lyric_rows:
        dutch_side = next(
            side for side in row.split("|") if side != "短歌"
        )
        assert dutch_side.replace("//", " ").split() == dutch.split()


def test_single_language_preview_and_utf8_download_are_identical() -> None:
    source = (
        "[Title]\n"
        "Grace\n\n"
        "Verse 1\n"
        "He gives me new hope for every day\n"
    )
    expected = (
        "[Title]\n"
        "Grace\n\n"
        "[Verse 1]\n"
        "He gives me new hope//for every day\n"
    )
    app = AppTest.from_file(str(APP_PATH), default_timeout=10).run()
    app.file_uploader[0].set_value(
        ("indirect-object.txt", source.encode("utf-8"), "text/plain")
    ).run(timeout=10)

    assert not app.exception
    assert not app.error
    assert len(app.text_area) == 1
    preview = app.text_area[0].value
    assert preview == expected
    assert preview == app.session_state["edited_output"]
    assert encode_utf8_txt(preview).decode("utf-8") == preview

    edited_preview = preview.replace("Grace", "New Grace")
    app.text_area[0].set_value(edited_preview).run(timeout=10)

    assert app.text_area[0].value == edited_preview
    assert app.session_state["edited_output"] == edited_preview
    assert encode_utf8_txt(app.text_area[0].value).decode("utf-8") == (
        edited_preview
    )
    download_buttons = app.get("download_button")
    assert len(download_buttons) == 1
    assert download_buttons[0].label == "Download final TXT"
