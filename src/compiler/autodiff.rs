//! Resource-backed text normalization and anchor inference.
//!
//! The anchor inference implemented here is intentionally conservative: it
//! compares sibling gold variants and only derives lexical constraints that are
//! explicitly supported by the closed-class resources or by the morphology of
//! the two variants.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;
use similar::{DiffTag, TextDiff};
use unicode_normalization::UnicodeNormalization;

use crate::schema::{ComplianceAnchors, Language, TargetVariant};

/// Closed-class and synset resources for a single language.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct LangResources {
    #[serde(default)]
    pub level_1_intimate: Vec<String>,
    #[serde(default)]
    pub level_2_polite: Vec<String>,
    #[serde(default)]
    pub level_3_honorific: Vec<String>,
    #[serde(default)]
    pub masculine_markers: Vec<String>,
    #[serde(default)]
    pub feminine_markers: Vec<String>,
    #[serde(default)]
    pub masculine_tokens: Vec<String>,
    #[serde(default)]
    pub feminine_tokens: Vec<String>,
    #[serde(default)]
    pub pronouns_and_kinship: Vec<String>,
    #[serde(default)]
    pub northern_markers: Vec<String>,
    #[serde(default)]
    pub southern_markers: Vec<String>,
    #[serde(default)]
    pub synsets: HashMap<String, String>,
}

impl LangResources {
    fn closed_class_iter(&self) -> impl Iterator<Item = &String> + '_ {
        self.level_1_intimate
            .iter()
            .chain(self.level_2_polite.iter())
            .chain(self.level_3_honorific.iter())
            .chain(self.masculine_markers.iter())
            .chain(self.feminine_markers.iter())
            .chain(self.masculine_tokens.iter())
            .chain(self.feminine_tokens.iter())
            .chain(self.pronouns_and_kinship.iter())
            .chain(self.northern_markers.iter())
            .chain(self.southern_markers.iter())
    }
}

/// Resource bundle for all languages supported by the evaluation suite.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct ClosedClassResources {
    #[serde(default)]
    pub gu: LangResources,
    #[serde(default)]
    pub he: LangResources,
    #[serde(default)]
    pub vi: LangResources,
}

impl ClosedClassResources {
    /// Loads the resource bundle from a JSON file.
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read closed-class resources {}", path.display()))?;
        serde_json::from_str(&raw).with_context(|| {
            format!(
                "failed to parse closed-class resources {}",
                path.display()
            )
        })
    }

    /// Returns the resource entry for a language.
    pub fn for_language(&self, lang: Language) -> &LangResources {
        match lang {
            Language::Gu => &self.gu,
            Language::He => &self.he,
            Language::Vi => &self.vi,
        }
    }

    /// Union of every closed-class token list, excluding synset keys.
    pub fn closed_class_tokens(&self, lang: Language) -> Vec<String> {
        let mut seen = HashSet::new();
        let mut tokens = Vec::new();
        for token in self.for_language(lang).closed_class_iter() {
            if seen.insert(token.clone()) {
                tokens.push(token.clone());
            }
        }
        tokens
    }

    /// Returns whether a token belongs to any closed-class list.
    pub fn is_closed_class(&self, lang: Language, token: &str) -> bool {
        let normalized = normalize_nfc(token);
        self.for_language(lang)
            .closed_class_iter()
            .any(|candidate| {
                candidate.as_str() == token || normalize_nfc(candidate.as_str()) == normalized
            })
    }

    /// Looks up the synset identifier assigned to a token.
    pub fn synset(&self, lang: Language, token: &str) -> Option<&str> {
        let resources = self.for_language(lang);
        if let Some(synset) = resources.synsets.get(token) {
            return Some(synset.as_str());
        }
        let normalized = normalize_nfc(token);
        resources
            .synsets
            .iter()
            .find(|(key, _)| normalize_nfc(key.as_str()) == normalized)
            .map(|(_, synset)| synset.as_str())
    }

    /// True when both tokens are assigned the same synset identifier.
    pub fn same_synset(&self, lang: Language, a: &str, b: &str) -> bool {
        matches!(
            (self.synset(lang, a), self.synset(lang, b)),
            (Some(left), Some(right)) if left == right
        )
    }

    /// All other resource keys belonging to the same synset as `token`.
    pub fn synset_equivalents(&self, lang: Language, token: &str) -> Vec<String> {
        let Some(target) = self.synset(lang, token) else {
            return Vec::new();
        };
        let normalized = normalize_nfc(token);
        let mut equivalents: Vec<String> = self
            .for_language(lang)
            .synsets
            .iter()
            .filter(|(key, synset)| {
                synset.as_str() == target && normalize_nfc(key.as_str()) != normalized
            })
            .map(|(key, _)| key.clone())
            .collect();
        equivalents.sort();
        equivalents.dedup();
        equivalents
    }
}

/// Normalizes a string to Unicode NFC.
pub fn normalize_nfc(input: &str) -> String {
    input.nfc().collect()
}

fn is_trimmed_punctuation(c: char) -> bool {
    matches!(
        c,
        '?' | '!'
            | '.'
            | ','
            | '।'
            | '״'
            | '"'
            | '\''
            | ';'
            | ':'
            | '('
            | ')'
            | '['
            | ']'
            | '{'
            | '}'
            | '«'
            | '»'
            | '–'
            | '—'
    )
}

/// Removes leading/trailing punctuation from each whitespace-separated word.
pub fn strip_punctuation(input: &str) -> String {
    input
        .split_whitespace()
        .map(|word| word.trim_matches(is_trimmed_punctuation))
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Normalizes, strips edge punctuation, and whitespace-tokenizes a string.
pub fn tokenize(input: &str) -> Vec<String> {
    let normalized = normalize_nfc(input);
    let stripped = strip_punctuation(&normalized);
    stripped
        .split_whitespace()
        .filter(|token| !token.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Approximate morphological relatedness based on a shared leading character stem.
pub fn morphological_shared_stem(a: &str, b: &str) -> bool {
    if a == b {
        return false;
    }

    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let max_len = a_chars.len().max(b_chars.len());
    if max_len == 0 {
        return false;
    }

    let common = a_chars
        .iter()
        .zip(b_chars.iter())
        .take_while(|(left, right)| left == right)
        .count();

    // Compare scalar values rather than grapheme clusters: Indic vowel signs
    // are separate scalars, so grapheme-based prefixes under-count shared
    // stems such as "નાની"/"નાના". A strictly greater-than-half prefix keeps
    // genuine inflections while rejecting weaker lexical near-misses.
    common >= 2 && (common as f64) / (max_len as f64) > 0.5
}

fn push_unique(target: &mut Vec<String>, token: &str) {
    if !target.iter().any(|existing| existing.as_str() == token) {
        target.push(token.to_owned());
    }
}

fn is_whitespace_only(value: &str) -> bool {
    value.chars().all(char::is_whitespace)
}

type WordDiff = (Vec<String>, Vec<String>, Vec<(String, String)>);

fn word_diff(tokens_old: &[String], tokens_new: &[String]) -> WordDiff {
    let old_joined = tokens_old.join(" ");
    let new_joined = tokens_new.join(" ");
    let diff = TextDiff::from_words(old_joined.as_str(), new_joined.as_str());

    let mut unique_old = Vec::new();
    let mut unique_new = Vec::new();
    let mut aligned = Vec::new();

    for op in diff.ops() {
        if op.tag() == DiffTag::Equal {
            continue;
        }

        let mut old_tokens = Vec::new();
        let mut new_tokens = Vec::new();
        for change in diff.iter_changes(op) {
            let value = change.value();
            if is_whitespace_only(value) {
                continue;
            }
            if change.old_index().is_some() {
                old_tokens.push(value.to_owned());
            }
            if change.new_index().is_some() {
                new_tokens.push(value.to_owned());
            }
        }

        match op.tag() {
            DiffTag::Equal => {}
            DiffTag::Delete => unique_old.extend(old_tokens),
            DiffTag::Insert => unique_new.extend(new_tokens),
            DiffTag::Replace => {
                if old_tokens.len() == new_tokens.len() {
                    aligned.extend(old_tokens.iter().cloned().zip(new_tokens.iter().cloned()));
                }
                unique_old.extend(old_tokens);
                unique_new.extend(new_tokens);
            }
        }
    }

    (unique_old, unique_new, aligned)
}

fn retained_tokens(
    unique: &[String],
    other_unique: &[String],
    aligned: &[(String, String)],
    lang: Language,
    resources: &ClosedClassResources,
    old_side: bool,
    dialect_differs: bool,
) -> Vec<String> {
    let mut retained = Vec::new();

    for token in unique {
        let closed_class = resources.is_closed_class(lang, token);
        let keep = if dialect_differs {
            let synset_shared = other_unique
                .iter()
                .any(|other| resources.same_synset(lang, token, other));
            let aligned_participant = aligned.iter().any(|(old, new)| {
                if old_side {
                    old == token
                } else {
                    new == token
                }
            });
            closed_class || synset_shared || aligned_participant
        } else if closed_class {
            true
        } else {
            // Open-class synonyms are noise for same-dialect siblings even when
            // they happen to share a synset; genuine morphology is retained.
            let synset_shared = other_unique
                .iter()
                .any(|other| resources.same_synset(lang, token, other));
            if synset_shared {
                false
            } else {
                other_unique
                    .iter()
                    .any(|other| morphological_shared_stem(token, other))
            }
        };

        if keep {
            push_unique(&mut retained, token);
        }
    }

    retained
}

/// Populates compliance anchors for all variants from sibling gold variants.
///
/// Existing anchors are cleared first so repeated compilation is deterministic.
pub fn populate_anchors(variants: &mut [TargetVariant], resources: &ClosedClassResources) {
    for variant in variants.iter_mut() {
        variant.anchors = ComplianceAnchors::default();
    }

    let mut groups: HashMap<(String, Language), Vec<usize>> = HashMap::new();
    for (index, variant) in variants.iter().enumerate() {
        groups
            .entry((variant.item_id.clone(), variant.lang))
            .or_default()
            .push(index);
    }

    for indices in groups.values() {
        let tokens: Vec<Vec<String>> = indices
            .iter()
            .map(|&index| tokenize(&variants[index].gold_reference))
            .collect();
        let langs: Vec<Language> = indices
            .iter()
            .map(|&index| variants[index].lang)
            .collect();
        let dialects: Vec<Option<String>> = indices
            .iter()
            .map(|&index| variants[index].coarse.dialect.clone())
            .collect();

        for left in 0..indices.len() {
            for right in 0..indices.len() {
                if left == right {
                    continue;
                }

                let i = indices[left];
                let j = indices[right];
                let (unique_i, unique_j, aligned) = word_diff(&tokens[left], &tokens[right]);
                let dialect_differs = match (&dialects[left], &dialects[right]) {
                    (Some(a), Some(b)) => a != b,
                    _ => false,
                };
                let lang = langs[left];

                let retained_i = retained_tokens(
                    &unique_i,
                    &unique_j,
                    &aligned,
                    lang,
                    resources,
                    true,
                    dialect_differs,
                );
                let retained_j = retained_tokens(
                    &unique_j,
                    &unique_i,
                    &aligned,
                    lang,
                    resources,
                    false,
                    dialect_differs,
                );

                for token in &retained_i {
                    push_unique(&mut variants[i].anchors.must_contain_any, token);
                    push_unique(&mut variants[j].anchors.must_not_contain, token);
                }
                for token in &retained_j {
                    push_unique(&mut variants[j].anchors.must_contain_any, token);
                    push_unique(&mut variants[i].anchors.must_not_contain, token);
                }
            }
        }
    }

    for variant in variants.iter_mut() {
        let forbidden: HashSet<String> = variant.anchors.must_not_contain.iter().cloned().collect();
        variant
            .anchors
            .must_contain_any
            .retain(|token| !forbidden.contains(token));

        let mut equivalents_by_token = HashMap::new();
        for token in &variant.anchors.must_contain_any {
            if resources.synset(variant.lang, token).is_none() {
                continue;
            }
            let mut equivalents: Vec<String> = resources
                .synset_equivalents(variant.lang, token)
                .into_iter()
                .map(|equivalent| normalize_nfc(&equivalent))
                .filter(|equivalent| !forbidden.contains(equivalent) && equivalent != token)
                .collect();
            equivalents.sort();
            equivalents.dedup();
            equivalents_by_token.insert(token.clone(), equivalents);
        }
        variant.anchors.synset_equivalents = equivalents_by_token;
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use unicode_normalization::UnicodeNormalization;

    use super::*;
    use crate::schema::{CoarseDimensions, GranularDimensions};

    fn test_resources() -> ClosedClassResources {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("data")
            .join("resources")
            .join("closed_class.json");
        ClosedClassResources::load(&path).expect("test resources should load")
    }

    fn make_variant(
        variant_id: &str,
        gold: &str,
        formality: &str,
        dialect: Option<&str>,
        lang: Language,
    ) -> TargetVariant {
        TargetVariant {
            variant_id: variant_id.to_owned(),
            item_id: "item-1".to_owned(),
            source_en: "source sentence".to_owned(),
            lang,
            track: "core".to_owned(),
            bucket: "test".to_owned(),
            is_cued_source: false,
            coarse: CoarseDimensions {
                formality: formality.to_owned(),
                speaker_gender: Some("male".to_owned()),
                listener_gender: None,
                dialect: dialect.map(str::to_owned),
            },
            granular: GranularDimensions::default(),
            conflict_override: None,
            gold_reference: gold.to_owned(),
            anchors: ComplianceAnchors::default(),
        }
    }

    fn has(tokens: &[String], needle: &str) -> bool {
        tokens.iter().any(|token| token.as_str() == needle)
    }

    #[test]
    fn gujarati_three_tier_formality_diff() {
        let resources = test_resources();
        let mut variants = vec![
            make_variant(
                "gu-intimate",
                "તેં નાના વાટકામાં ચમચી મૂકી?",
                "level_1_intimate",
                None,
                Language::Gu,
            ),
            make_variant(
                "gu-polite",
                "તમે નાના વાટકામાં ચમચી મૂકી?",
                "level_2_polite",
                None,
                Language::Gu,
            ),
            make_variant(
                "gu-honorific",
                "આપે નાના વાટકામાં ચમચી મૂકી?",
                "level_3_honorific",
                None,
                Language::Gu,
            ),
        ];

        populate_anchors(&mut variants, &resources);

        let intimate = &variants[0].anchors;
        assert!(has(&intimate.must_contain_any, "તેં"));
        assert!(has(&intimate.must_not_contain, "તમે"));
        assert!(has(&intimate.must_not_contain, "આપે"));

        let honorific = &variants[2].anchors;
        assert!(has(&honorific.must_contain_any, "આપે"));
        assert!(has(&honorific.must_not_contain, "તેં"));
        assert!(has(&honorific.must_not_contain, "તમે"));
    }

    #[test]
    fn same_dialect_filters_synset_noise_but_keeps_morphology() {
        let resources = test_resources();
        let mut variants = vec![
            make_variant(
                "gu-a",
                "તેં નાની વાટકીમાં ચમચી મૂકી?",
                "level_1_intimate",
                None,
                Language::Gu,
            ),
            make_variant(
                "gu-b",
                "તમે નાના વાટકોમાં ચમચી મૂકી?",
                "level_2_polite",
                None,
                Language::Gu,
            ),
        ];

        populate_anchors(&mut variants, &resources);

        let first = &variants[0].anchors;
        assert!(has(&first.must_contain_any, "તેં"));
        assert!(!has(&first.must_contain_any, "વાટકી"));
        assert!(has(&first.must_contain_any, "નાની"));
    }

    #[test]
    fn vietnamese_dialect_allolex() {
        let resources = test_resources();
        let mut variants = vec![
            make_variant(
                "vi-north",
                "Em đã bỏ cái thìa vào cái bát nhỏ chưa?",
                "default",
                Some("northern"),
                Language::Vi,
            ),
            make_variant(
                "vi-south",
                "Em bỏ cái muỗng vô cái chén nhỏ chưa?",
                "default",
                Some("southern"),
                Language::Vi,
            ),
        ];

        populate_anchors(&mut variants, &resources);

        let northern = &variants[0].anchors;
        let southern = &variants[1].anchors;

        for token in ["muỗng", "vô", "chén"] {
            assert!(has(&southern.must_contain_any, token), "missing {token}");
        }
        for token in ["thìa", "vào", "bát"] {
            assert!(
                has(&southern.must_not_contain, token),
                "expected forbidden {token}"
            );
        }

        for token in ["thìa", "vào", "bát"] {
            assert!(has(&northern.must_contain_any, token), "missing {token}");
        }
        for token in ["muỗng", "vô", "chén"] {
            assert!(
                has(&northern.must_not_contain, token),
                "expected forbidden {token}"
            );
        }
    }

    #[test]
    fn nfd_input_produces_nfc_anchors() {
        let resources = test_resources();
        let northern_nfc = "Em đã bỏ cái thìa vào cái bát nhỏ chưa?";
        let southern_nfc = "Em bỏ cái muỗng vô cái chén nhỏ chưa?";
        let northern_nfd: String = northern_nfc.nfd().collect();
        let southern_nfd: String = southern_nfc.nfd().collect();

        assert_eq!(tokenize(&northern_nfd), tokenize(northern_nfc));
        assert_eq!(tokenize(&southern_nfd), tokenize(southern_nfc));

        let mut variants = vec![
            make_variant(
                "vi-north",
                &northern_nfd,
                "default",
                Some("northern"),
                Language::Vi,
            ),
            make_variant(
                "vi-south",
                &southern_nfd,
                "default",
                Some("southern"),
                Language::Vi,
            ),
        ];

        populate_anchors(&mut variants, &resources);

        for variant in &variants {
            for token in variant
                .anchors
                .must_contain_any
                .iter()
                .chain(variant.anchors.must_not_contain.iter())
            {
                assert_eq!(token, &normalize_nfc(token), "anchor is not NFC: {token:?}");
            }
            for (key, values) in &variant.anchors.synset_equivalents {
                assert_eq!(key, &normalize_nfc(key));
                for value in values {
                    assert_eq!(value, &normalize_nfc(value));
                }
            }
        }

        let southern = variants
            .iter()
            .find(|variant| variant.variant_id == "vi-south")
            .expect("southern variant");
        assert!(has(&southern.anchors.must_contain_any, "muỗng"));
    }
}
