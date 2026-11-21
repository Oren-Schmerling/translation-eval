"""Base candidate row data model for source loaders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class CandidateRow:
    """Raw candidate sentence pair harvested from an upstream source."""

    source_en: Optional[str]
    reference_target: str
    source_corpus: str
    source_split: str
    origin_direction: Literal["source_first", "target_first"]
    metadata: dict = field(default_factory=dict)
