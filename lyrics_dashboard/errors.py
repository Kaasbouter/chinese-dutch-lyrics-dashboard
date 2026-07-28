class LyricsDashboardError(Exception):
    """Base exception for user-facing conversion errors."""


class ExtractionError(LyricsDashboardError):
    """Raised when text cannot be extracted from an uploaded file."""


class ParseError(LyricsDashboardError):
    """Raised when the source text does not match the expected lyric format."""


class PairingError(LyricsDashboardError):
    """Raised when Chinese and Dutch sections cannot be paired safely."""
