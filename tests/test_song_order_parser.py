from __future__ import annotations

import pytest

from lyrics_dashboard.errors import ParseError
from lyrics_dashboard.parser import apply_section_languages, parse_lyrics


CHINESE_SONG = (
    "135 人算什麼\n\n"
    "Order\n"
    "v c v c o\n\n"
    "Verse\n"
    "耶和華我們的主 你的名在地何美\n"
    "你將榮耀彰顯於天\n\n"
    "Chorus\n"
    "人算什麼 你竟顧念他\n\n"
    "outro\n"
    "環顧宇宙萬物何廣闊\n"
)


def test_screenshot_order_is_metadata_in_a_chinese_only_song() -> None:
    parsed = parse_lyrics(CHINESE_SONG)

    assert parsed.raw_title == parsed.chinese_title == "135 人算什麼"
    assert parsed.dutch_title == ""
    assert parsed.song_order == ("V", "C", "V", "C", "O")
    assert parsed.mode == "single-language"
    assert parsed.single_language == "zh"
    assert [section.label for section in parsed.sections] == ["Verse", "Chorus", "Outro"]
    assert parsed.sections[0].lines == (
        "耶和華我們的主 你的名在地何美",
        "你將榮耀彰顯於天",
    )
    assert not parsed.warnings


@pytest.mark.parametrize(
    "order_block",
    [
        "[Order] V1 C V2 C B C",
        "Order V1 C V2 C B C",
        "Order: v1 c v2 c b c",
        "[oRdEr]: v1 C v2 c B C",
        "Order\nV1 C V2 C B C",
        "[Order]\n\nV1 C\nV2 C\nB C",
        "Order:\r\n\r\nv1\tc\tv2\tc\tb\tc",
        "[ Order ]\u00a0v1\u00a0c\u00a0v2 c b c",
    ],
)
def test_order_forms_normalize_codes_without_changing_title_or_sections(order_block: str) -> None:
    parsed = parse_lyrics(f"[Title]\nGrace\n{order_block}\nVerse 1\nA lyric line\n")

    assert parsed.song_order == ("V1", "C", "V2", "C", "B", "C")
    assert parsed.raw_title == "Grace"
    assert [section.label for section in parsed.sections] == ["Verse 1"]
    assert parsed.sections[0].lines == ("A lyric line",)


def test_order_accepts_all_section_codes_and_numbered_variants() -> None:
    codes = "i v1 pc c c2 b2 t o il e"
    parsed = parse_lyrics(f"[Order] {codes}\nVerse\nA lyric line\n")

    assert parsed.song_order == ("I", "V1", "PC", "C", "C2", "B2", "T", "O", "IL", "E")
    assert parsed.raw_title == ""


@pytest.mark.parametrize(
    ("heading", "label", "kind", "number"),
    [
        ("[v1]", "Verse 1", "Verse", 1),
        ("[C]", "Chorus", "Chorus", None),
        ("[c2]", "Chorus 2", "Chorus", 2),
        ("[PC]", "Pre-Chorus", "Pre-Chorus", None),
        ("[B2]", "Bridge 2", "Bridge", 2),
        ("[I]", "Intro", "Intro", None),
        ("[O]", "Outro", "Outro", None),
        ("[T]", "Tag", "Tag", None),
        ("[IL]", "Interlude", "Interlude", None),
        ("[E]", "End", "Outro", None),
        ("[ PC 2 ]:", "Pre-Chorus 2", "Pre-Chorus", 2),
        ("prechorus", "Pre-Chorus", "Pre-Chorus", None),
        ("Pre Chorus 2", "Pre-Chorus 2", "Pre-Chorus", 2),
        ("[Pre-Chorus 1]", "Pre-Chorus 1", "Pre-Chorus", 1),
    ],
)
def test_section_heading_codes_and_prechorus_spellings(
    heading: str, label: str, kind: str, number: int | None
) -> None:
    parsed = parse_lyrics(f"歌名\n{heading}\n中文歌詞\n")

    assert parsed.sections[0].label == label
    assert parsed.sections[0].kind == kind
    assert parsed.sections[0].number == number
    assert parsed.sections[0].lines == ("中文歌詞",)


def test_full_section_names_cover_every_order_code() -> None:
    parsed = parse_lyrics(
        "中文歌名\n[Order] I V1 PC C C2 B T O IL E\n"
        "Intro\n引言歌詞\nVerse 1\n主歌歌詞\nPre-Chorus\n預副歌歌詞\n"
        "Chorus\n副歌歌詞\nChorus 2\n另一副歌\nBridge\n過渡歌詞\n"
        "Tag\n重複歌詞\noutro\n結尾歌詞\nInterlude\n間奏歌詞\nEnd\n末段歌詞\n"
    )

    assert len(parsed.sections) == 10
    assert all(section.language == "zh" for section in parsed.sections)
    assert parsed.mode == "single-language"


def test_order_between_sections_never_becomes_a_lyric_line() -> None:
    parsed = parse_lyrics(
        "中文歌名\nVerse\n中文主歌\nOrder\nv c v c o\n"
        "Chorus\n中文副歌\nOutro\n中文尾聲\n"
    )

    assert parsed.song_order == ("V", "C", "V", "C", "O")
    assert [section.lines for section in parsed.sections] == [
        ("中文主歌",), ("中文副歌",), ("中文尾聲",)
    ]


def test_order_at_end_does_not_change_bilingual_detection_or_language_overrides() -> None:
    parsed = parse_lyrics(
        "中文歌名 | Dutch title\nVerse 1\n中文主歌\nVerse 2\nDutch verse\n"
        "[Order] V1 V2 V1\n"
    )
    updated = apply_section_languages(parsed, {0: "zh", 1: "nl"})

    assert parsed.song_order == updated.song_order == ("V1", "V2", "V1")
    assert updated.mode == "bilingual"
    assert [section.lines for section in updated.sections] == [("中文主歌",), ("Dutch verse",)]


def test_no_order_keeps_ordinary_lyrics_and_written_section_order() -> None:
    parsed = parse_lyrics(
        "Order my steps\nVerse\nI\nOrder my steps\nOrder my steps: I follow\n"
        "Chorus\nA song of hope\n"
    )

    assert parsed.song_order == ()
    assert parsed.raw_title == "Order my steps"
    assert parsed.sections[0].lines == ("I", "Order my steps", "Order my steps: I follow")
    assert [section.label for section in parsed.sections] == ["Verse", "Chorus"]


def test_identical_order_metadata_can_be_repeated_between_language_blocks() -> None:
    parsed = parse_lyrics(
        "[Order] V1 C\nVerse 1\n中文主歌\nOrder\nv1 c\nVerse 2\nDutch verse\n"
    )

    assert parsed.song_order == ("V1", "C")
    assert len(parsed.sections) == 2


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("[Order]\nVerse\nA lyric line\n", "needs section codes"),
        ("Verse\nA lyric line\nOrder\n", "needs section codes"),
        ("[Order] V1 UNKNOWN\nVerse\nA lyric line\n", "unrecognized code: 'UNKNOWN'"),
        ("Order:\nV1 C\nThese are not codes\nVerse\nA lyric line\n", "unrecognized code"),
        ("[Order] V1 C\nVerse\nA lyric line\n[Order] C V1\n", "different \\[Order\\] lists"),
    ],
)
def test_malformed_order_has_an_actionable_error(source: str, message: str) -> None:
    with pytest.raises(ParseError, match=message):
        parse_lyrics(source)
