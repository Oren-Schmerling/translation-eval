"""Strict quality assertions for Pragmatic MT Dataset v2."""

import json
from pathlib import Path
from typing import List
import pytest
from provenance.v2.filters import normalize_dedup_key
from provenance.v2.schema import V2Item

REPO_ROOT = Path(__file__).resolve().parents[3]
JSONL_PATH = REPO_ROOT / "provenance" / "v2" / "output" / "dataset_v2.jsonl"
if not JSONL_PATH.exists():
    JSONL_PATH = REPO_ROOT / "data" / "compiled" / "dataset_v2.jsonl"
MANIFEST_PATH = REPO_ROOT / "provenance" / "v2" / "manifest.json"


@pytest.fixture(scope="module")
def dataset_items() -> List[V2Item]:
    assert JSONL_PATH.exists(), f"dataset_v2.jsonl not found at {JSONL_PATH}"
    items: List[V2Item] = []
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            # Pydantic validation on 100% of rows
            item = V2Item.model_validate(data)
            items.append(item)
    return items


def test_zero_iwslt_in_casual(dataset_items: List[V2Item]) -> None:
    """Assertion 1: Zero IWSLT rows in casual_conversational."""
    casual = [it for it in dataset_items if it.track == "casual_conversational"]
    iwslt_rows = [it for it in casual if "iwslt" in it.source_corpus.lower()]
    assert len(iwslt_rows) == 0, f"Found {len(iwslt_rows)} IWSLT rows in casual_conversational track!"


def test_zero_duplicate_source_en_in_casual(dataset_items: List[V2Item]) -> None:
    """Assertion 2: 0 duplicate source_en strings in casual track."""
    casual = [it for it in dataset_items if it.track == "casual_conversational"]
    seen = set()
    duplicates = []
    for it in casual:
        key = normalize_dedup_key(it.source_en)
        if key in seen:
            duplicates.append((it.item_id, it.source_en))
        seen.add(key)
    assert len(duplicates) == 0, f"Found {len(duplicates)} duplicate source_en strings in casual: {duplicates[:5]}"


def test_target_first_ratio_he_vi(dataset_items: List[V2Item]) -> None:
    """Assertion 3: Target-first rows comprise >= 30% of casual_conversational across he and vi."""
    casual = [it for it in dataset_items if it.track == "casual_conversational"]

    he_casual = [it for it in casual if it.lang == "he"]
    vi_casual = [it for it in casual if it.lang == "vi"]

    he_tf = [it for it in he_casual if it.origin_direction == "target_first"]
    vi_tf = [it for it in vi_casual if it.origin_direction == "target_first"]

    he_ratio = len(he_tf) / len(he_casual) if he_casual else 0
    vi_ratio = len(vi_tf) / len(vi_casual) if vi_casual else 0

    assert he_ratio >= 0.30, f"Hebrew target-first ratio {he_ratio:.2%} < 30% (count: {len(he_tf)}/{len(he_casual)})"
    assert vi_ratio >= 0.30, f"Vietnamese target-first ratio {vi_ratio:.2%} < 30% (count: {len(vi_tf)}/{len(vi_casual)})"


def test_gujarati_casual_registers(dataset_items: List[V2Item]) -> None:
    """Assertion 4: Gujarati casual has >= 25% intimate (તું) and >= 15% honorific (આપ)."""
    gu_casual = [it for it in dataset_items if it.track == "casual_conversational" and it.lang == "gu"]
    assert len(gu_casual) > 0, "No Gujarati casual items found"
    if len(gu_casual) < 50:
        pytest.skip("Skipping register balance assertions on small dry-run slice")

    l1_intimate = [it for it in gu_casual if it.pragmatic_profile.register_tier == "intimate"]
    l3_honorific = [it for it in gu_casual if it.pragmatic_profile.register_tier == "honorific"]
    l2_polite = [it for it in gu_casual if it.pragmatic_profile.register_tier == "polite"]

    l1_ratio = len(l1_intimate) / len(gu_casual)
    l3_ratio = len(l3_honorific) / len(gu_casual)
    l2_ratio = len(l2_polite) / len(gu_casual)

    assert l1_ratio >= 0.25, f"Gujarati L1 intimate ratio {l1_ratio:.2%} < 25% ({len(l1_intimate)}/{len(gu_casual)})"
    assert l3_ratio >= 0.15, f"Gujarati L3 honorific ratio {l3_ratio:.2%} < 15% ({len(l3_honorific)}/{len(gu_casual)})"
    assert l2_ratio <= 0.60, f"Gujarati L2 polite ratio {l2_ratio:.2%} > 60% ({len(l2_polite)}/{len(gu_casual)})"


def test_vietnamese_dialect_balance(dataset_items: List[V2Item]) -> None:
    """Assertion 5: Vietnamese dialect-tagged items are 55%–65% vi_south."""
    vi_casual = [it for it in dataset_items if it.track == "casual_conversational" and it.lang == "vi"]
    vi_dialect_tagged = [it for it in vi_casual if it.pragmatic_profile.regional_dialect in ("vi_south", "vi_north")]

    assert len(vi_dialect_tagged) > 0, "No Vietnamese dialect-tagged items found in casual track"
    if len(vi_dialect_tagged) < 50:
        pytest.skip("Skipping dialect balance assertions on small dry-run slice")

    south_count = sum(1 for it in vi_dialect_tagged if it.pragmatic_profile.regional_dialect == "vi_south")
    south_ratio = south_count / len(vi_dialect_tagged)

    assert 0.55 <= south_ratio <= 0.65, (
        f"Vietnamese south dialect ratio {south_ratio:.2%} outside 55%-65% range "
        f"({south_count}/{len(vi_dialect_tagged)})"
    )


def test_required_and_forbidden_anchors(dataset_items: List[V2Item]) -> None:
    """Assertion 6: Every required_anchor token exists in target, and zero forbidden_markers exist in target."""
    for it in dataset_items:
        if it.reference_target is None:
            continue
        # Check required anchors
        for anchor in it.required_anchors:
            assert anchor in it.reference_target, (
                f"Item {it.item_id}: required anchor '{anchor}' missing from target '{it.reference_target}'"
            )
        # Check forbidden markers
        for marker in it.forbidden_markers:
            assert marker not in it.reference_target, (
                f"Item {it.item_id}: forbidden marker '{marker}' found in target '{it.reference_target}'"
            )


def test_winomt_diagnostic_structure(dataset_items: List[V2Item]) -> None:
    """Assertion 7: WinoMT items have null reference_target and valid entity gender."""
    wino_items = [it for it in dataset_items if "mt_gender" in it.source_corpus.lower() or "wino" in it.source_corpus.lower()]
    if wino_items:
        for it in wino_items:
            assert it.track == "diagnostic_morphology"
            assert it.reference_target is None
            assert it.pragmatic_profile.listener_gender in ("male", "female")


def test_contrastive_pairs_integrity(dataset_items: List[V2Item]) -> None:
    """Assertion 8: Register contrastive track contains formal and informal variants."""
    reg_items = [it for it in dataset_items if it.track == "register_contrastive"]
    if reg_items:
        for it in reg_items:
            assert it.pragmatic_profile.register_tier in ("honorific", "polite")
            assert len(it.required_anchors) > 0
            assert isinstance(it.forbidden_markers, list)
