"""Deterministic backward label harvesting from native target text."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple
from provenance.v2.schema import PragmaticProfile

# ---------------------------------------------------------------------------
# Hebrew Lexicons & Patterns
# ---------------------------------------------------------------------------

HE_MALE_PRONOUNS = {"אתה"}
HE_FEMALE_PRONOUNS = {"את"}

HE_FEMALE_2P_VERBS = {
    "תעשי", "תבואי", "תגידי", "תלכי", "תראי", "תקחי", "תשמעי",
    "תתקשרי", "תחזרי", "תשבי", "תדברי", "תסבירי", "תפסיקי", "תמשיכי",
    "הולכת", "יודעת", "אוכלת", "עייפה", "בטוחה", "מוכנה", "רוצה",
    "יכולה", "צריכה", "שומעת", "מבינה", "זוכרת",
}

HE_MALE_2P_VERBS = {
    "תעשה", "תבוא", "תגיד", "תלך", "תראה", "תקח", "תשמע",
    "תתקשר", "תחזור", "תשב", "תדבר", "תסביר", "תפסיק", "תמשיך",
    "עייף", "בטוח", "מוכן",
}

HE_AMBIGUOUS_WITHOUT_NIQQUD = {"עשית", "אמרת", "הלכת", "ראית", "לקחת", "שמעת"}

# ---------------------------------------------------------------------------
# Gujarati Lexicons & Patterns
# ---------------------------------------------------------------------------

GU_L1_INTIMATE = ["તું", "તને", "તારું", "તારી", "તારા", "તારાથી", "તારામાં"]
GU_L2_POLITE = ["તમે", "તમને", "તમારું", "તમારી", "તમારા", "તમારાથી", "તમારામાં"]
GU_L3_HONORIFIC = ["આપ", "આપને", "આપનું", "આપની", "આપના", "આપનાથી", "આપનામાં"]

# ---------------------------------------------------------------------------
# Vietnamese Kinship & Dialect Lexicons
# ---------------------------------------------------------------------------

VI_KINSHIP_PAIRS: List[Tuple[str, List[str], List[str]]] = [
    ("anh_em", ["anh"], ["em"]),
    ("chi_em", ["chị"], ["em"]),
    ("chu_chau", ["chú"], ["cháu"]),
    ("may_tao", ["mày"], ["tao"]),
    ("ban_minh", ["bạn"], ["mình"]),
    ("con_bo", ["con"], ["bố", "ba"]),
    ("con_me", ["con"], ["mẹ", "má"]),
]

VI_SOUTH_ANCHORS = [
    "muỗng", "vô", "hổng", "nè", "nghen", "má", "tui",
    "bông", "nhiều chuyện", "chạy xe", "bển", "trên trển", "dữ dằn",
]

VI_NORTH_ANCHORS = [
    "thìa", "vào", "không", "nhé", "nhỉ", "bố", "tôi", "hoa", "vâng",
]

VI_DIALECT_OPPOSITES: Dict[str, str] = {
    "muỗng": "thìa",
    "thìa": "muỗng",
    "vô": "vào",
    "vào": "vô",
    "hổng": "không",
    "má": "mẹ",
    "tui": "tôi",
    "tôi": "tui",
    "bông": "hoa",
    "hoa": "bông",
    "bố": "ba",
}


def _tokenize_words(text: str) -> List[str]:
    """Tokenize text into lowercase words/tokens stripping punctuation."""
    clean = re.sub(r"[^\w\s\u0590-\u05FF\u0A80-\u0AFF]", " ", text)
    return [w.strip() for w in clean.split() if w.strip()]


def harvest_hebrew_labels(
    target_text: str, metadata: Optional[dict] = None
) -> Tuple[PragmaticProfile, List[str], List[str]]:
    """Deterministically harvest Hebrew pragmatic dimensions from unvocalized target gold."""
    tokens = set(_tokenize_words(target_text))
    profile = PragmaticProfile()
    required: List[str] = []
    forbidden: List[str] = []

    has_male_pronoun = bool(tokens.intersection(HE_MALE_PRONOUNS))
    has_female_pronoun = bool(tokens.intersection(HE_FEMALE_PRONOUNS))

    # Check for direct object marker ambiguity: 'את ה...' before a definite direct object noun.
    # If followed by a feminine 2nd person verb/participle/adjective (e.g. הולכת, מוכנה), it is the 2nd person pronoun.
    matched_fem_verbs = [v for v in HE_FEMALE_2P_VERBS if v in tokens]
    matched_male_verbs = [v for v in HE_MALE_2P_VERBS if v in tokens]

    # If 'את' appears before a recognized feminine verb (like 'את הולכת'), it is unambiguously 'you' (fem)
    is_subject_fem = False
    if has_female_pronoun:
        if matched_fem_verbs or any(re.search(rf"\bאת\s+{v}\b", target_text) for v in HE_FEMALE_2P_VERBS):
            is_subject_fem = True
        elif re.search(r"\b(איפה|מי|למה|לאן|מה)\s+את\b", target_text):
            is_subject_fem = True
        elif re.search(r"\bאת\s+ה[א-ת]+", target_text):
            # Accusative marker preceding noun with definite article
            has_female_pronoun = False

    if is_subject_fem:
        has_female_pronoun = True

    # Only set listener_gender if orthographically unambiguous
    if has_male_pronoun and not has_female_pronoun:
        profile.listener_gender = "male"
        required.append("אתה")
        forbidden.append("את")
    elif has_female_pronoun and not has_male_pronoun:
        profile.listener_gender = "female"
        required.append("את")
        forbidden.append("אתה")
    elif matched_fem_verbs and not matched_male_verbs:
        # Feminine imperative or participle
        v = matched_fem_verbs[0]
        profile.listener_gender = "female"
        required.append(v)
        forbidden.append("אתה")
    elif matched_male_verbs and not matched_fem_verbs:
        v = matched_male_verbs[0]
        profile.listener_gender = "male"
        required.append(v)
        forbidden.append("את")
    else:
        profile.listener_gender = None

    # Check 1st person speaker gender cues (e.g. אני עייפה / אני עייף)
    if re.search(r"\bאני\s+(עייפה|בטוחה|מוכנה|הולכת|יודעת)\b", target_text):
        profile.speaker_gender = "female"
        cue = re.search(r"\b(עייפה|בטוחה|מוכנה|הולכת|יודעת)\b", target_text).group(1)
        if cue not in required:
            required.append(cue)
    elif re.search(r"\bאני\s+(עייף|בטוח|מוכן|הולך|יודע)\b", target_text):
        profile.speaker_gender = "male"
        cue = re.search(r"\b(עייף|בטוח|מוכן|הולך|יודע)\b", target_text).group(1)
        if cue not in required:
            required.append(cue)

    # If sentence only has unvocalized ambiguous verbs without pronoun, ensure null
    has_ambig = bool(tokens.intersection(HE_AMBIGUOUS_WITHOUT_NIQQUD))
    if has_ambig and not profile.listener_gender and not profile.speaker_gender:
        profile.listener_gender = None
        profile.speaker_gender = None

    return profile, required, forbidden


def harvest_gujarati_labels(
    target_text: str, metadata: Optional[dict] = None
) -> Tuple[PragmaticProfile, List[str], List[str]]:
    """Deterministically harvest Gujarati register tier and gender from target gold."""
    profile = PragmaticProfile()
    required: List[str] = []
    forbidden: List[str] = []

    matched_l1 = [w for w in GU_L1_INTIMATE if w in target_text]
    matched_l2 = [w for w in GU_L2_POLITE if w in target_text]
    matched_l3 = [w for w in GU_L3_HONORIFIC if w in target_text]

    if matched_l1 and not matched_l2 and not matched_l3:
        profile.register_tier = "intimate"
        required.append(matched_l1[0])
        forbidden.extend(["તમે", "તમને", "આપ", "આપને"])
    elif matched_l2 and not matched_l1 and not matched_l3:
        profile.register_tier = "polite"
        required.append(matched_l2[0])
        forbidden.extend(["તું", "તને", "આપ", "આપને"])
    elif matched_l3 and not matched_l1 and not matched_l2:
        profile.register_tier = "honorific"
        required.append(matched_l3[0])
        forbidden.extend(["તું", "તને", "તમે", "તમને"])
    else:
        profile.register_tier = None

    # Gender suffixes or verified metadata
    if metadata and "gender" in metadata:
        g = str(metadata["gender"]).lower()
        if g in ("male", "m", "1"):
            profile.speaker_gender = "male"
        elif g in ("female", "f", "0", "2"):
            profile.speaker_gender = "female"

    return profile, required, forbidden


def harvest_vietnamese_labels(
    target_text: str, metadata: Optional[dict] = None
) -> Tuple[PragmaticProfile, List[str], List[str]]:
    """Deterministically harvest Vietnamese kinship, dialect, and register from target gold."""
    tokens = [w.lower() for w in _tokenize_words(target_text)]
    token_set = set(tokens)
    text_lower = target_text.lower()
    profile = PragmaticProfile()
    required: List[str] = []
    forbidden: List[str] = []

    # 1. Kinship pair: extract ONLY when BOTH pronouns appear
    for label, side_a_options, side_b_options in VI_KINSHIP_PAIRS:
        has_a = [opt for opt in side_a_options if opt in token_set]
        has_b = [opt for opt in side_b_options if opt in token_set]
        if has_a and has_b:
            profile.social_relationship = label
            anchor_a = has_a[0]
            anchor_b = has_b[0]
            if anchor_a not in required:
                required.append(anchor_a)
            if anchor_b not in required:
                required.append(anchor_b)
            if label == "may_tao":
                profile.register_tier = "intimate"
            break

    # 2. Regional dialect from ViDia2Std/ViMD source region tags AND deterministic lexical anchors
    meta_region = (metadata.get("region") or "").lower() if metadata else ""
    matched_south = [w for w in VI_SOUTH_ANCHORS if w in text_lower]
    # For Northern lexical detection on unlabelled sources, require distinctive markers (not generic 'không' or 'tôi')
    matched_north_distinctive = [w for w in ["thìa", "nhé", "nhỉ", "vâng"] if w in text_lower]

    if "south" in meta_region:
        profile.regional_dialect = "vi_south"
        for w in matched_south:
            if w in text_lower and w not in required:
                required.append(w)
                opp = VI_DIALECT_OPPOSITES.get(w)
                if opp and opp not in forbidden and opp not in text_lower:
                    forbidden.append(opp)
    elif "north" in meta_region:
        profile.regional_dialect = "vi_north"
        matched_north = [w for w in VI_NORTH_ANCHORS if w in text_lower]
        for w in matched_north:
            if w in text_lower and w not in required:
                required.append(w)
                opp = VI_DIALECT_OPPOSITES.get(w)
                if opp and opp not in forbidden and opp not in text_lower:
                    forbidden.append(opp)
    elif matched_south and not matched_north_distinctive:
        profile.regional_dialect = "vi_south"
        for w in matched_south:
            if w in text_lower and w not in required:
                required.append(w)
                opp = VI_DIALECT_OPPOSITES.get(w)
                if opp and opp not in forbidden and opp not in text_lower:
                    forbidden.append(opp)
    elif matched_north_distinctive and not matched_south:
        profile.regional_dialect = "vi_north"
        matched_north = [w for w in VI_NORTH_ANCHORS if w in text_lower]
        for w in matched_north:
            if w in text_lower and w not in required:
                required.append(w)
                opp = VI_DIALECT_OPPOSITES.get(w)
                if opp and opp not in forbidden and opp not in text_lower:
                    forbidden.append(opp)

    # 3. Clean anchors that actually exist in target_text
    required = [r for r in required if r in target_text or r.lower() in text_lower]
    forbidden = [f for f in forbidden if f not in target_text and f.lower() not in text_lower]

    return profile, required, forbidden


def harvest_pragmatic_labels(
    target_text: Optional[str], lang: str, metadata: Optional[dict] = None
) -> Tuple[PragmaticProfile, List[str], List[str]]:
    """Main dispatch function for deterministic backward label harvesting."""
    if target_text is None:
        profile = PragmaticProfile()
        if metadata and "gender" in metadata:
            g = str(metadata["gender"]).lower()
            if g in ("male", "m"):
                profile.listener_gender = "male"
            elif g in ("female", "f"):
                profile.listener_gender = "female"
        return profile, [], []

    # Check if contrastive metadata provides ground truth anchors
    if metadata and "contrastive_anchors" in metadata:
        profile = PragmaticProfile()
        formality = str(metadata.get("formality") or "").lower()
        if formality == "formal":
            profile.register_tier = "honorific"
        elif formality == "informal":
            profile.register_tier = "polite"
        clean_req = [r for r in metadata.get("contrastive_anchors", []) if r in target_text]
        clean_forb = [f for f in metadata.get("contrastive_forbidden", []) if f not in target_text]
        return profile, clean_req, clean_forb

    if lang == "he":
        profile, req, forb = harvest_hebrew_labels(target_text, metadata)
    elif lang == "gu":
        profile, req, forb = harvest_gujarati_labels(target_text, metadata)
    elif lang == "vi":
        profile, req, forb = harvest_vietnamese_labels(target_text, metadata)
    else:
        profile, req, forb = PragmaticProfile(), [], []

    # Final safety assertions: required must be in text, forbidden must NOT be in text
    clean_req = [r for r in req if r in target_text]
    clean_forb = [f for f in forb if f not in target_text]

    return profile, clean_req, clean_forb
