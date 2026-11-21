"""Deterministic self-containment, hygiene, and deduplication filters for v2."""

from __future__ import annotations

import re
from typing import Optional, Set, Tuple

# Regex patterns for subtitle and transcript artifacts
HTML_TAG_PATTERN = re.compile(r"<[^>]+>", re.IGNORECASE)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
SDH_BRACKETS_PATTERN = re.compile(r"\[.*?\]|\(.*?\)")
SPEAKER_PREFIX_PATTERN = re.compile(r"^[A-Za-z0-9_\-\s]{2,15}:\s*")
LEADING_DASH_PATTERN = re.compile(r"^[-–—]+\s*")
MULTIPLE_ELLIPSES_PATTERN = re.compile(r"\.{3,}|…")

# Unresolved fragment openers
FRAGMENT_OPENER_PATTERN = re.compile(
    r"^(and\s+then|because\s+of\s+that|so\s+he|so\s+she|well,|yeah,|no,\s+wait|but\s+it)\b",
    re.IGNORECASE,
)

# Common subject/pronoun + verb indicator in English to detect complete clauses
CLAUSE_INDICATOR_PATTERN = re.compile(
    r"\b(i|you|he|she|it|we|they|who|what|this|that|there)\s+(am|is|are|was|were|have|has|had|do|does|did|can|could|will|would|should|must|go|went|said|thought|felt|knew|wanted|see|saw)\b",
    re.IGNORECASE,
)

# Script ranges
HEBREW_CHAR_PATTERN = re.compile(r"[\u0590-\u05FF]")
GUJARATI_CHAR_PATTERN = re.compile(r"[\u0A80-\u0AFF]")
VIETNAMESE_CHAR_PATTERN = re.compile(
    r"[a-zA-Z\sàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ.,!?;:'\"–—\-]"
)


def normalize_source_en(text: str) -> str:
    """Normalize English source string by cleaning basic punctuation artifacts."""
    clean = text.strip()
    clean = LEADING_DASH_PATTERN.sub("", clean)
    clean = SPEAKER_PREFIX_PATTERN.sub("", clean)
    clean = HTML_TAG_PATTERN.sub("", clean)
    clean = SDH_BRACKETS_PATTERN.sub("", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def normalize_dedup_key(text: str) -> str:
    """Normalize English source to alphanumeric lowercase for deduplication."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def check_word_count(text: str, min_words: int = 6, max_words: int = 22) -> bool:
    """Check if English sentence has between min_words and max_words."""
    words = [w for w in text.split() if any(c.isalnum() for c in w)]
    return min_words <= len(words) <= max_words


def has_subtitle_artifacts(text: str) -> bool:
    """Detect subtitle/SDH artifacts like brackets, speaker prefixes, HTML, URLs, or ellipses."""
    if text.startswith("- ") or text.startswith("-- ") or text.startswith("— "):
        return True
    if SPEAKER_PREFIX_PATTERN.search(text):
        return True
    if SDH_BRACKETS_PATTERN.search(text):
        return True
    if HTML_TAG_PATTERN.search(text):
        return True
    if URL_PATTERN.search(text):
        return True
    if MULTIPLE_ELLIPSES_PATTERN.search(text):
        return True
    return False


def is_unresolved_fragment(text: str) -> bool:
    """Check if sentence starts with fragment opener without a complete clause."""
    match = FRAGMENT_OPENER_PATTERN.search(text)
    if not match:
        return False
    # If opener is present, verify whether the remainder contains a subject+verb clause
    remainder = text[match.end():]
    if not CLAUSE_INDICATOR_PATTERN.search(remainder):
        return True
    return False


def check_script_hygiene(text: str, lang: str, threshold: float = 0.70) -> bool:
    """Ensure target string contains at least threshold proportion of target-script characters."""
    if not text.strip():
        return False
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False

    if lang == "he":
        # Target Hebrew letters or standard punctuation
        valid = [c for c in chars if HEBREW_CHAR_PATTERN.match(c) or c in ".,!?;:'\"-–—()"]
    elif lang == "gu":
        # Target Gujarati letters, digits or punctuation
        valid = [c for c in chars if GUJARATI_CHAR_PATTERN.match(c) or c in ".,!?;:'\"-–—()|॥"]
    elif lang == "vi":
        # Target Vietnamese characters
        valid = [c for c in chars if VIETNAMESE_CHAR_PATTERN.match(c)]
    else:
        return True

    ratio = len(valid) / len(chars)
    return ratio >= threshold


class Deduplicator:
    """Tracks normalized English sources to prevent cross-split and cross-corpus duplicates."""

    def __init__(self) -> None:
        self._seen: Set[str] = set()

    def is_duplicate(self, text: str) -> bool:
        key = normalize_dedup_key(text)
        if not key or key in self._seen:
            return True
        return False

    def add(self, text: str) -> None:
        key = normalize_dedup_key(text)
        if key:
            self._seen.add(key)


def validate_candidate(
    source_en: str,
    reference_target: str,
    lang: str,
    deduplicator: Optional[Deduplicator] = None,
) -> Tuple[bool, Optional[str]]:
    """Validate a candidate pair against all hygiene and self-containment rules."""
    clean_en = normalize_source_en(source_en)
    clean_target = reference_target.strip()

    if not clean_en or not clean_target:
        return False, "empty_text"

    if has_subtitle_artifacts(source_en):
        return False, "subtitle_artifacts"

    if has_subtitle_artifacts(clean_target):
        return False, "target_subtitle_artifacts"

    if not check_word_count(clean_en, 6, 22):
        return False, "word_count_out_of_bounds"

    if is_unresolved_fragment(clean_en):
        return False, "unresolved_fragment_opener"

    if not check_script_hygiene(clean_target, lang, 0.70):
        return False, "script_hygiene_failed"

    if deduplicator is not None and deduplicator.is_duplicate(clean_en):
        return False, "duplicate_source_en"

    return True, None
