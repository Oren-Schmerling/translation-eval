"""Strict Pydantic schema for pragmatic MT dataset v2."""

from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class PragmaticProfile(BaseModel):
    """Pragmatic axes harvested deterministically backward from the target."""

    listener_gender: Optional[Literal["male", "female", "neutral"]] = Field(
        default=None, description="Gender of the addressed listener"
    )
    speaker_gender: Optional[Literal["male", "female"]] = Field(
        default=None, description="Gender of the speaker"
    )
    register_tier: Optional[Literal["intimate", "polite", "honorific"]] = Field(
        default=None, description="Formality / politeness register tier"
    )
    regional_dialect: Optional[Literal["vi_north", "vi_south"]] = Field(
        default=None, description="Regional dialect variant for Vietnamese"
    )
    social_relationship: Optional[str] = Field(
        default=None, description="Kinship or interpersonal relationship tag"
    )


class V2Item(BaseModel):
    """Single benchmark item in the v2 dataset."""

    item_id: str = Field(description="Unique benchmark item identifier, e.g. v2_vi_cas_0001")
    dataset_version: Literal["v2"] = Field(default="v2", description="Dataset schema version")
    track: Literal[
        "casual_conversational", "register_contrastive", "diagnostic_morphology"
    ] = Field(description="Quarantined evaluation track")
    lang: Literal["he", "gu", "vi"] = Field(description="Target language code")
    source_en: str = Field(description="English source text")
    reference_target: str = Field(description="Native human-written target reference")
    source_corpus: str = Field(description="Upstream source dataset identifier")
    source_split: str = Field(description="Upstream split name (e.g. train, test, dev)")
    origin_direction: Literal["source_first", "target_first"] = Field(
        description="Whether source was back-translated from native target or parallel"
    )
    pragmatic_profile: PragmaticProfile = Field(
        default_factory=PragmaticProfile, description="Harvested pragmatic dimensions"
    )
    required_anchors: List[str] = Field(
        default_factory=list,
        description="Tokens guaranteed to exist in reference_target",
    )
    forbidden_markers: List[str] = Field(
        default_factory=list,
        description="Tokens guaranteed to NOT exist in reference_target",
    )
    harvest_confidence: Literal[
        "deterministic_exact", "metadata_verified", "baseline_unlabeled"
    ] = Field(
        default="deterministic_exact",
        description="Methodology confidence for the pragmatic label extraction",
    )
