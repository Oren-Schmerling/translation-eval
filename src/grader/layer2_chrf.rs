//! Layer 2: pure-Rust chrF++ scoring.
//!
//! Character n-grams are computed over characters with whitespace removed,
//! while word n-grams are computed over whitespace tokens. Both streams are
//! NFC-normalized and lowercased before counting. Scores are macro-averaged
//! over n-gram orders that exist in the reference and returned as a percentage.

use std::collections::HashMap;

use unicode_normalization::UnicodeNormalization;

const BETA: f64 = 2.0;
const CHAR_MAX_N: usize = 6;
const WORD_MAX_N: usize = 2;

/// NFC-normalizes and lowercases text before n-gram extraction.
pub fn normalize_text(input: &str) -> String {
    input.nfc().flat_map(char::to_lowercase).collect()
}

fn is_surrounding_punctuation(c: char) -> bool {
    c.is_ascii_punctuation()
        || matches!(
            c,
            '“' | '”'
                | '‘'
                | '’'
                | '«'
                | '»'
                | '…'
                | '—'
                | '–'
                | '׳'
                | '״'
                | '،'
                | '؟'
                | '।'
                | '॥'
        )
}

fn word_tokens(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter_map(|token| {
            let trimmed = token.trim_matches(is_surrounding_punctuation);
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_owned())
            }
        })
        .collect()
}

fn char_tokens(text: &str) -> Vec<String> {
    text.chars()
        .filter(|c| !c.is_whitespace())
        .map(|c| c.to_string())
        .collect()
}

fn count_ngrams(tokens: &[String], n: usize) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    if n == 0 || tokens.len() < n {
        return counts;
    }

    for window in tokens.windows(n) {
        let mut key = String::new();
        for (index, token) in window.iter().enumerate() {
            if index > 0 {
                key.push('\u{1f}');
            }
            key.push_str(token);
        }
        *counts.entry(key).or_insert(0) += 1;
    }

    counts
}

/// Computes the F-beta score for one n-gram order.
///
/// Returns `None` when the reference has no n-grams of this order (the order is
/// skipped). Returns `Some(0.0)` when the hypothesis has no such n-grams.
fn order_f_score(hypothesis: &[String], reference: &[String], n: usize) -> Option<f64> {
    let reference_total = match reference.len().checked_sub(n) {
        Some(extra) => extra + 1,
        None => return None,
    };
    let hypothesis_total = match hypothesis.len().checked_sub(n) {
        Some(extra) => extra + 1,
        None => return Some(0.0),
    };

    let hypothesis_counts = count_ngrams(hypothesis, n);
    let reference_counts = count_ngrams(reference, n);

    let matched: usize = hypothesis_counts
        .iter()
        .map(|(ngram, count)| {
            let reference_count = reference_counts.get(ngram).copied().unwrap_or(0);
            (*count).min(reference_count)
        })
        .sum();

    let precision = matched as f64 / hypothesis_total as f64;
    let recall = matched as f64 / reference_total as f64;

    if precision + recall == 0.0 {
        return Some(0.0);
    }

    let beta_squared = BETA * BETA;
    Some(
        (1.0 + beta_squared) * precision * recall
            / (beta_squared * precision + recall),
    )
}

/// chrF++ score in the range `0.0..=100.0`.
pub fn chrf(hypothesis: &str, reference: &str) -> f64 {
    let hypothesis = normalize_text(hypothesis);
    let reference = normalize_text(reference);

    let hypothesis_chars = char_tokens(&hypothesis);
    let reference_chars = char_tokens(&reference);
    let hypothesis_words = word_tokens(&hypothesis);
    let reference_words = word_tokens(&reference);

    let mut score_sum = 0.0;
    let mut orders = 0usize;

    for n in 1..=CHAR_MAX_N {
        if let Some(score) = order_f_score(&hypothesis_chars, &reference_chars, n) {
            score_sum += score;
            orders += 1;
        }
    }

    for n in 1..=WORD_MAX_N {
        if let Some(score) = order_f_score(&hypothesis_words, &reference_words, n) {
            score_sum += score;
            orders += 1;
        }
    }

    if orders == 0 {
        return 0.0;
    }

    (score_sum / orders as f64) * 100.0
}

/// Alias for [`chrf`] with the full metric name.
pub fn chrf_plus_plus(hypothesis: &str, reference: &str) -> f64 {
    chrf(hypothesis, reference)
}

/// Best chrF++ score over the reference and any acceptable synonyms.
pub fn best_chrf(
    hypothesis: &str,
    reference: &str,
    acceptable_synonyms: &[String],
) -> f64 {
    let mut best = chrf_plus_plus(hypothesis, reference);

    for synonym in acceptable_synonyms {
        let score = chrf_plus_plus(hypothesis, synonym);
        if score > best {
            best = score;
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match_scores_100() {
        let score = chrf("שלום עולם", "שלום עולם");
        assert!((score - 100.0).abs() < 1e-3, "score was {score}");
    }

    #[test]
    fn fully_disjoint_scores_zero() {
        assert_eq!(chrf("abc", "xyz"), 0.0);
    }

    #[test]
    fn morphological_inflection_scores_partially() {
        let score = chrf("אני מוכן לעזור לך", "אני מוכנה לעזור לך");
        assert!(
            score > 50.0 && score < 95.0,
            "expected a partial score, got {score}"
        );
    }

    #[test]
    fn empty_hypothesis_scores_zero() {
        assert_eq!(chrf("", "abc"), 0.0);
        assert_eq!(chrf("", ""), 0.0);
    }

    #[test]
    fn best_chrf_prefers_acceptable_synonyms() {
        let synonyms = vec!["שלום".to_owned()];
        let score = best_chrf("שלום", "להתראות", &synonyms);
        assert!((score - 100.0).abs() < 1e-3, "score was {score}");
    }
}
