//! Layer 1: deterministic lexical anchor compliance.
//!
//! Anchor checks are boundary-aware: a token only counts when it appears as a
//! whole word, so Hebrew feminine forms such as `את` and `מוכן` do not match
//! inside masculine forms such as `אתה` and `מוכנה`.

use std::collections::HashMap;

use aho_corasick::AhoCorasick;
use unicode_normalization::UnicodeNormalization;

use crate::schema::{AnchorResult, BiasProfile, ComplianceAnchors, TargetVariant};

/// Normalizes text to Unicode NFC so canonically equivalent spellings compare
/// equal.
pub fn normalize_nfc(input: &str) -> String {
    input.nfc().collect()
}

fn boundary_before(haystack: &str, start: usize) -> bool {
    haystack[..start]
        .chars()
        .next_back()
        .map_or(true, |c| !c.is_alphabetic())
}

fn boundary_after(haystack: &str, end: usize) -> bool {
    haystack[end..]
        .chars()
        .next()
        .map_or(true, |c| !c.is_alphabetic())
}

/// Returns true when `needle` occurs in `haystack_nfc` as a boundary-delimited
/// token.
///
/// The haystack is expected to already be NFC-normalized; the needle is
/// normalized defensively so callers can pass raw dataset values.
pub fn contains_token(haystack_nfc: &str, needle: &str) -> bool {
    let needle = normalize_nfc(needle);
    if needle.is_empty() {
        return false;
    }

    let automaton = match AhoCorasick::new([needle.as_str()]) {
        Ok(automaton) => automaton,
        Err(_) => return false,
    };

    automaton
        .find_overlapping_iter(haystack_nfc)
        .any(|m| boundary_before(haystack_nfc, m.start()) && boundary_after(haystack_nfc, m.end()))
}

/// Checks a translation against all lexical anchors.
///
/// Forbidden tokens are checked directly. A required token is satisfied by a
/// direct match, then by a synset equivalent, then by a global acceptable
/// synonym. All matched forms are recorded for reporting.
pub fn evaluate_anchors(translation: &str, anchors: &ComplianceAnchors) -> AnchorResult {
    let haystack = normalize_nfc(translation);
    let mut result = AnchorResult::default();

    for token in &anchors.must_not_contain {
        if !token.is_empty() && contains_token(&haystack, token) {
            result.forbidden_hits.push(token.clone());
        }
    }

    for token in &anchors.must_contain_any {
        if token.is_empty() {
            continue;
        }

        if contains_token(&haystack, token) {
            result.matched_must_contain.push(token.clone());
            continue;
        }

        let mut satisfied = false;

        if let Some(equivalents) = anchors.synset_equivalents.get(token) {
            for equivalent in equivalents {
                if !equivalent.is_empty() && contains_token(&haystack, equivalent) {
                    result
                        .matched_equivalents
                        .push(format!("{token}=>{equivalent}"));
                    result.matched_must_contain.push(token.clone());
                    satisfied = true;
                    break;
                }
            }
        }

        if satisfied {
            continue;
        }

        for synonym in &anchors.acceptable_synonyms {
            if !synonym.is_empty() && contains_token(&haystack, synonym) {
                result
                    .matched_equivalents
                    .push(format!("{token}=>{synonym}"));
                result.matched_must_contain.push(token.clone());
                break;
            }
        }
    }

    result.passed = result.forbidden_hits.is_empty()
        && (anchors.must_contain_any.is_empty() || !result.matched_must_contain.is_empty());

    result
}

fn push_non_empty(values: &mut Vec<String>, value: Option<&str>) {
    if let Some(value) = value {
        if !value.is_empty() {
            values.push(value.to_owned());
        }
    }
}

/// Most frequent value, with deterministic lexical tie-breaking.
fn mode(values: &[String]) -> Option<String> {
    if values.is_empty() {
        return None;
    }

    let mut counts: HashMap<&str, usize> = HashMap::new();
    for value in values {
        *counts.entry(value.as_str()).or_insert(0) += 1;
    }

    counts
        .into_iter()
        .max_by(|(a_value, a_count), (b_value, b_count)| {
            a_count.cmp(b_count).then_with(|| b_value.cmp(a_value))
        })
        .map(|(value, _)| value.to_owned())
}

/// Infers the pragmatic cell most consistent with a default translation by
/// evaluating it against sibling variants that carry lexical anchors.
///
/// Only siblings with non-empty anchors are considered. Confidence is the
/// fraction of those siblings whose anchors pass.
pub fn classify_default_bias(
    default_translation: &str,
    siblings: &[TargetVariant],
) -> BiasProfile {
    let mut considered = 0usize;
    let mut matched: Vec<&TargetVariant> = Vec::new();

    for sibling in siblings {
        if sibling.anchors.is_empty() {
            continue;
        }
        considered += 1;
        if evaluate_anchors(default_translation, &sibling.anchors).passed {
            matched.push(sibling);
        }
    }

    let mut formality_values = Vec::new();
    let mut speaker_values = Vec::new();
    let mut listener_values = Vec::new();
    let mut dialect_values = Vec::new();

    for variant in &matched {
        if !variant.coarse.formality.is_empty() {
            formality_values.push(variant.coarse.formality.clone());
        }
        push_non_empty(
            &mut speaker_values,
            variant.coarse.speaker_gender.as_deref(),
        );
        push_non_empty(
            &mut listener_values,
            variant.coarse.listener_gender.as_deref(),
        );
        push_non_empty(&mut dialect_values, variant.coarse.dialect.as_deref());
    }

    let mut matched_variant_ids: Vec<String> = matched
        .iter()
        .map(|variant| variant.variant_id.clone())
        .collect();
    matched_variant_ids.sort();

    BiasProfile {
        formality: mode(&formality_values),
        speaker_gender: mode(&speaker_values),
        listener_gender: mode(&listener_values),
        dialect: mode(&dialect_values),
        matched_variant_ids,
        confidence: if considered == 0 {
            0.0
        } else {
            matched.len() as f64 / considered as f64
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{CoarseDimensions, Language};

    fn anchors(must_contain: &[&str], must_not_contain: &[&str]) -> ComplianceAnchors {
        ComplianceAnchors {
            must_contain_any: must_contain.iter().map(|s| (*s).to_owned()).collect(),
            must_not_contain: must_not_contain
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
            ..ComplianceAnchors::default()
        }
    }

    fn coarse(formality: &str, speaker_gender: Option<&str>) -> CoarseDimensions {
        CoarseDimensions {
            formality: formality.to_owned(),
            speaker_gender: speaker_gender.map(str::to_owned),
            listener_gender: None,
            dialect: None,
        }
    }

    fn variant(id: &str, coarse: CoarseDimensions, anchors: ComplianceAnchors) -> TargetVariant {
        TargetVariant {
            variant_id: id.to_owned(),
            item_id: "item-1".to_owned(),
            source_en: "I am ready to help you".to_owned(),
            lang: Language::He,
            track: "test".to_owned(),
            bucket: "test".to_owned(),
            is_cued_source: false,
            coarse,
            granular: Default::default(),
            conflict_override: None,
            gold_reference: String::new(),
            anchors,
        }
    }

    #[test]
    fn hebrew_default_bias_selects_male_variant() {
        let male = variant(
            "male",
            coarse("default", Some("male")),
            anchors(&["מוכן"], &["מוכנה"]),
        );
        let female = variant(
            "female",
            coarse("default", Some("female")),
            anchors(&["מוכנה"], &["מוכן"]),
        );

        let profile = classify_default_bias("אני מוכן לעזור לך", &[male, female]);

        assert_eq!(profile.speaker_gender.as_deref(), Some("male"));
        assert_eq!(profile.matched_variant_ids, vec!["male".to_owned()]);
        assert!((profile.confidence - 0.5).abs() < 1e-9);
    }

    #[test]
    fn southern_vietnamese_forbidden_marker_fails() {
        let anchors = anchors(&["muỗng"], &["nhé", "thìa", "bát"]);
        let good = "Em đã bỏ cái muỗng vô cái chén nhỏ nha.";
        let result = evaluate_anchors(good, &anchors);

        assert!(result.passed);
        assert!(result.forbidden_hits.is_empty());
        assert_eq!(result.matched_must_contain, vec!["muỗng".to_owned()]);

        let bad = good.replace("nha", "nhé");
        let result = evaluate_anchors(&bad, &anchors);

        assert!(!result.passed);
        assert_eq!(result.forbidden_hits, vec!["nhé".to_owned()]);
        assert!(result.matched_must_contain.contains(&"muỗng".to_owned()));
    }

    #[test]
    fn forbidden_tokens_respect_word_boundaries() {
        let forbidden = anchors(&[], &["את", "מוכן"]);
        let result = evaluate_anchors("אתה מוכן", &forbidden);

        assert!(!result.passed);
        assert!(!result.forbidden_hits.contains(&"את".to_owned()));
        assert!(result.forbidden_hits.contains(&"מוכן".to_owned()));

        let required = anchors(&["את"], &[]);
        let result = evaluate_anchors("אתה מוכן", &required);
        assert!(!result.passed);

        let result = evaluate_anchors("את מוכן", &required);
        assert!(result.passed);
    }

    #[test]
    fn empty_anchors_pass() {
        let result = evaluate_anchors("anything at all", &ComplianceAnchors::default());

        assert!(result.passed);
        assert!(result.matched_must_contain.is_empty());
        assert!(result.forbidden_hits.is_empty());
        assert!(result.matched_equivalents.is_empty());
    }

    #[test]
    fn synset_equivalents_and_synonyms_satisfy_required_tokens() {
        let mut anchors = anchors(&["muỗng"], &[]);
        anchors
            .synset_equivalents
            .insert("muỗng".to_owned(), vec!["thìa".to_owned()]);

        let result = evaluate_anchors("cái thìa nhỏ", &anchors);
        assert!(result.passed);
        assert_eq!(
            result.matched_equivalents,
            vec!["muỗng=>thìa".to_owned()]
        );

        let anchors = ComplianceAnchors {
            must_contain_any: vec!["muỗng".to_owned()],
            acceptable_synonyms: vec!["cái thìa".to_owned()],
            ..ComplianceAnchors::default()
        };
        let result = evaluate_anchors("cái thìa nhỏ", &anchors);
        assert!(result.passed);
        assert_eq!(
            result.matched_equivalents,
            vec!["muỗng=>cái thìa".to_owned()]
        );
    }
}
