from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Language = Literal["zh", "nl", "unknown"]
Confidence = Literal["manual"]


@dataclass(frozen=True)
class Section:
    label: str
    kind: str
    number: int | None
    lines: tuple[str, ...]
    language: Language
    original_index: int


@dataclass(frozen=True)
class ParsedLyrics:
    raw_title: str
    chinese_title: str
    dutch_title: str
    sections: tuple[Section, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class TranslationReference:
    section_index: int
    line_indices: tuple[int, ...]


@dataclass(frozen=True)
class AlignedLine:
    source_line_indices: tuple[int, ...]
    translation_references: tuple[TranslationReference, ...]


@dataclass(frozen=True)
class SectionAlignment:
    source_section_index: int
    counterpart_section_indices: tuple[int, ...]
    aligned_lines: tuple[AlignedLine, ...]
    confidence: Confidence
    note: str = ""


@dataclass(frozen=True)
class AlignmentPlan:
    alignments: tuple[SectionAlignment, ...]
    method: Literal["manual"]
    warnings: tuple[str, ...] = ()

    def for_section(self, section_index: int) -> SectionAlignment:
        for alignment in self.alignments:
            if alignment.source_section_index == section_index:
                return alignment
        raise KeyError(section_index)
