"""Unit tests for self-containment, hygiene, and deduplication filters."""

import pytest
from provenance.v2.filters import (
    Deduplicator,
    check_script_hygiene,
    check_word_count,
    has_subtitle_artifacts,
    is_unresolved_fragment,
    normalize_dedup_key,
    normalize_source_en,
    validate_candidate,
)


def test_word_count() -> None:
    assert check_word_count("This is a simple sentence here.", 6, 22) is True
    assert check_word_count("Too short.", 6, 22) is False
    assert check_word_count(" ".join(["word"] * 25), 6, 22) is False


def test_subtitle_artifacts() -> None:
    assert has_subtitle_artifacts("- Did you see that?") is True
    assert has_subtitle_artifacts("JOHN: Where are you going?") is True
    assert has_subtitle_artifacts("She said hello [laughter].") is True
    assert has_subtitle_artifacts("<i>Subtitle text</i>") is True
    assert has_subtitle_artifacts("Check out https://example.com now") is True
    assert has_subtitle_artifacts("Wait... what happened?") is True
    assert has_subtitle_artifacts("Could you please close the window?") is False


def test_unresolved_fragment_openers() -> None:
    assert is_unresolved_fragment("And then yesterday afternoon.") is True
    assert is_unresolved_fragment("Because of that last night.") is True
    assert is_unresolved_fragment("Well, I am thinking we should go now.") is False
    assert is_unresolved_fragment("Could you pass the salt please?") is False


def test_script_hygiene() -> None:
    # Hebrew
    assert check_script_hygiene("שלום, מה שלומך הבוקר?", "he", 0.70) is True
    assert check_script_hygiene("Hello world how are you?", "he", 0.70) is False

    # Gujarati
    assert check_script_hygiene("નમસ્તે, આપ કેમ છો?", "gu", 0.70) is True
    assert check_script_hygiene("Hello world how are you?", "gu", 0.70) is False

    # Vietnamese
    assert check_script_hygiene("Chào buổi sáng, bạn khỏe không?", "vi", 0.70) is True


def test_deduplicator() -> None:
    dedup = Deduplicator()
    assert dedup.is_duplicate("Hello, how are you?") is False
    dedup.add("Hello, how are you?")
    assert dedup.is_duplicate("hello how are you") is True
    assert dedup.is_duplicate("Hello! How are you??") is True
    assert dedup.is_duplicate("Different sentence entirely.") is False
