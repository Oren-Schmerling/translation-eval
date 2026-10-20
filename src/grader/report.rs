//! Markdown and CSV reporting for evaluation runs.
//!
//! `render_markdown` produces a five-table summary suitable for a pull request
//! or release note. `write_csv` emits the flat record stream for downstream
//! analysis. All aggregation helpers are intentionally private.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;

use anyhow::{Context, Result};

use crate::schema::{BiasProfile, CapabilityStatus, CoarseDimensions, EvalTier, RunRecord};

/// Renders a complete Markdown report from run records.
pub fn render_markdown(records: &[RunRecord]) -> String {
    let mut out = String::new();
    out.push_str("# Pragmatic MT Evaluation Report\n\n");

    render_tier1(&mut out, records);
    render_tier2(&mut out, records);
    render_tier2c(&mut out, records);
    render_tier3(&mut out, records);
    render_telemetry(&mut out, records);

    out
}

/// Writes a Markdown report, creating parent directories as needed.
pub fn write_markdown(records: &[RunRecord], path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create report directory {}", parent.display()))?;
        }
    }
    std::fs::write(path, render_markdown(records))
        .with_context(|| format!("failed to write markdown report {}", path.display()))
}

/// Writes one CSV row per run record, creating parent directories as needed.
pub fn write_csv(records: &[RunRecord], path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create report directory {}", parent.display()))?;
        }
    }

    let mut writer = csv::Writer::from_path(path)
        .with_context(|| format!("failed to create CSV report {}", path.display()))?;

    writer.write_record([
        "provider_id",
        "tier",
        "item_id",
        "variant_id",
        "lang",
        "track",
        "bucket",
        "capability",
        "layer1_passed",
        "chrf",
        "judge_overall",
        "high_disagreement",
        "ttft_ms",
        "total_latency_ms",
        "uncached_cost_usd",
        "cached_cost_usd",
    ])?;

    for record in records {
        let layer1_passed = record
            .layer1
            .as_ref()
            .map(|anchor| anchor.passed.to_string())
            .unwrap_or_default();
        let chrf = record.chrf.map(|value| value.to_string()).unwrap_or_default();
        let judge_overall = record
            .ensemble
            .as_ref()
            .map(|ensemble| ensemble.mean.overall().to_string())
            .unwrap_or_default();
        let high_disagreement = record.high_disagreement_flag.to_string();
        let ttft = record.ttft_ms.map(|value| value.to_string()).unwrap_or_default();
        let total_latency = record.total_latency_ms.to_string();
        let uncached_cost = record.uncached_cost_usd.to_string();
        let cached_cost = record.cached_cost_usd.to_string();

        writer.write_record([
            record.provider_id.as_str(),
            record.tier.label(),
            record.item_id.as_str(),
            record.variant_id.as_str(),
            record.lang.code(),
            record.track.as_str(),
            record.bucket.as_str(),
            record.capability.label(),
            layer1_passed.as_str(),
            chrf.as_str(),
            judge_overall.as_str(),
            high_disagreement.as_str(),
            ttft.as_str(),
            total_latency.as_str(),
            uncached_cost.as_str(),
            cached_cost.as_str(),
        ])?;
    }

    writer
        .flush()
        .with_context(|| format!("failed to flush CSV report {}", path.display()))?;
    Ok(())
}

/// Returns the requested percentile using linear interpolation.
///
/// Values are sorted in place. `p` is clamped to `0..=100`; an empty slice
/// returns `None`. For example, the median of `[10, 20, 30, 40]` is `25`.
pub fn percentile(values: &mut [f64], p: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let clamped = p.clamp(0.0, 100.0);
    let rank = clamped / 100.0 * (values.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let weight = rank - lower as f64;

    Some(values[lower] * (1.0 - weight) + values[upper] * weight)
}

fn render_tier1(out: &mut String, records: &[RunRecord]) {
    out.push_str("## Table 1: Tier 1 Default Quality & Bias Distribution\n\n");
    out.push_str(
        "| Provider | Lang | n | Mean chrF | Mean Judge | % Male | % Female | \
         % L1 Intimate | % L2 Polite | % L3 Honorific | % Northern | % Southern |\n",
    );
    out.push_str(
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    );

    let mut groups: BTreeMap<(String, String), Vec<&RunRecord>> = BTreeMap::new();
    for record in records
        .iter()
        .filter(|record| record.tier == EvalTier::Tier1DefaultZeroShot)
    {
        groups
            .entry((record.provider_id.clone(), record.lang.code().to_owned()))
            .or_default()
            .push(record);
    }

    if groups.is_empty() {
        out.push_str(&empty_row(12));
    }

    for ((provider, lang), group) in groups {
        let n = group.len();
        let mean_chrf_value = mean_chrf(&group);
        let mean_judge_value = mean_judge(&group);

        let profiles: Vec<&BiasProfile> = group
            .iter()
            .filter_map(|record| record.bias_profile.as_ref())
            .collect();

        let speaker_values: Vec<&str> = profiles
            .iter()
            .filter_map(|profile| profile.speaker_gender.as_deref())
            .filter(|value| !value.is_empty())
            .collect();
        let male_count = speaker_values
            .iter()
            .filter(|value| value_matches(value, "male", "m"))
            .count();
        let female_count = speaker_values
            .iter()
            .filter(|value| value_matches(value, "female", "f"))
            .count();

        let (gu_l1, gu_l2, gu_l3) = if lang == "gu" {
            let formality_values: Vec<&str> = profiles
                .iter()
                .filter_map(|profile| profile.formality.as_deref())
                .filter(|value| !value.is_empty())
                .collect();
            let total = formality_values.len();
            (
                fmt_pct(
                    formality_values
                        .iter()
                        .filter(|value| value.eq_ignore_ascii_case("level_1_intimate"))
                        .count(),
                    total,
                ),
                fmt_pct(
                    formality_values
                        .iter()
                        .filter(|value| value.eq_ignore_ascii_case("level_2_polite"))
                        .count(),
                    total,
                ),
                fmt_pct(
                    formality_values
                        .iter()
                        .filter(|value| value.eq_ignore_ascii_case("level_3_honorific"))
                        .count(),
                    total,
                ),
            )
        } else {
            (na(), na(), na())
        };

        let (vi_north, vi_south) = if lang == "vi" {
            let dialect_values: Vec<&str> = profiles
                .iter()
                .filter_map(|profile| profile.dialect.as_deref())
                .filter(|value| !value.is_empty())
                .collect();
            let total = dialect_values.len();
            (
                fmt_pct(
                    dialect_values
                        .iter()
                        .filter(|value| value.eq_ignore_ascii_case("northern"))
                        .count(),
                    total,
                ),
                fmt_pct(
                    dialect_values
                        .iter()
                        .filter(|value| value.eq_ignore_ascii_case("southern"))
                        .count(),
                    total,
                ),
            )
        } else {
            (na(), na())
        };

        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            escape_cell(&provider),
            lang,
            n,
            fmt_float(mean_chrf_value, 3),
            fmt_float(mean_judge_value, 2),
            fmt_pct(male_count, speaker_values.len()),
            fmt_pct(female_count, speaker_values.len()),
            gu_l1,
            gu_l2,
            gu_l3,
            vi_north,
            vi_south,
        );
    }
    out.push('\n');
}

fn render_tier2(out: &mut String, records: &[RunRecord]) {
    out.push_str("## Table 2: Tier 2 Parameter Compliance (ACR %)\n\n");
    out.push_str(
        "| Provider | Tier | n | Compliance % | UnsupportedByProvider | \
         PartiallySupported missing axes |\n",
    );
    out.push_str("|---|---|---:|---:|---|---|\n");

    let mut groups: BTreeMap<(String, String), Vec<&RunRecord>> = BTreeMap::new();
    for record in records.iter().filter(|record| {
        matches!(
            record.tier,
            EvalTier::Tier2ACoarseParams | EvalTier::Tier2BGranularParams
        )
    }) {
        groups
            .entry((record.provider_id.clone(), record.tier.label().to_owned()))
            .or_default()
            .push(record);
    }

    if groups.is_empty() {
        out.push_str(&empty_row(6));
    }

    for ((provider, tier), group) in groups {
        let n = group.len();
        let passed = group
            .iter()
            .filter(|record| record.layer1.as_ref().is_some_and(|anchor| anchor.passed))
            .count();

        let mut unsupported_count = 0usize;
        let mut unsupported_reasons = BTreeSet::new();
        let mut partial_count = 0usize;
        let mut partial_axes = BTreeSet::new();

        for record in &group {
            match &record.capability {
                CapabilityStatus::UnsupportedByProvider { reason } => {
                    unsupported_count += 1;
                    if !reason.is_empty() {
                        unsupported_reasons.insert(escape_cell(reason));
                    }
                }
                CapabilityStatus::PartiallySupported { missing_axes, .. } => {
                    partial_count += 1;
                    for axis in missing_axes {
                        if !axis.is_empty() {
                            partial_axes.insert(escape_cell(axis));
                        }
                    }
                }
                CapabilityStatus::Supported => {}
            }
        }

        let unsupported = count_with_list(unsupported_count, unsupported_reasons.into_iter());
        let partial = count_with_list(partial_count, partial_axes.into_iter());

        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {} | {} |",
            escape_cell(&provider),
            escape_cell(&tier),
            n,
            fmt_pct(passed, n),
            unsupported,
            partial,
        );
    }
    out.push('\n');
}

fn render_tier2c(out: &mut String, records: &[RunRecord]) {
    out.push_str("## Table 3: Tier 2C Adversarial Parameter Override Rate (POR %)\n\n");
    out.push_str("| Provider | Conflict Requests (n) | Override Hits | POR % |\n");
    out.push_str("|---|---:|---:|---:|\n");

    let mut groups: BTreeMap<String, Vec<&RunRecord>> = BTreeMap::new();
    for record in records.iter().filter(|record| {
        record.tier == EvalTier::Tier2CAdversarialConflict
            && record.conflict_override.is_some()
    }) {
        groups
            .entry(record.provider_id.clone())
            .or_default()
            .push(record);
    }

    if groups.is_empty() {
        out.push_str(&empty_row(4));
    }

    for (provider, group) in groups {
        let n = group.len();
        let hits = group
            .iter()
            .filter(|record| override_hit(record))
            .count();
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} |",
            escape_cell(&provider),
            n,
            hits,
            fmt_pct(hits, n),
        );
    }
    out.push('\n');
}

fn render_tier3(out: &mut String, records: &[RunRecord]) {
    out.push_str("## Table 4: Tier 3 Few-Shot Adaptation Delta\n\n");
    out.push_str(
        "| Provider | Tier1 chrF | Tier1 Judge | Tier3A chrF | Tier3A Judge | \
         Tier3B chrF | Tier3B Judge | Tier3C chrF | Tier3C Judge | \
         Δ chrF 1→3A | Δ Judge 1→3A | Δ chrF 3A→3B | Δ Judge 3A→3B | \
         Δ chrF 3A→3C | Δ Judge 3A→3C |\n",
    );
    out.push_str(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    );

    let mut groups: BTreeMap<String, Vec<&RunRecord>> = BTreeMap::new();
    for record in records.iter().filter(|record| {
        matches!(
            record.tier,
            EvalTier::Tier1DefaultZeroShot
                | EvalTier::Tier3AFewShotK5Pos
                | EvalTier::Tier3BFewShotK20Pos
                | EvalTier::Tier3CFewShotK5Contrastive
        )
    }) {
        groups
            .entry(record.provider_id.clone())
            .or_default()
            .push(record);
    }

    if groups.is_empty() {
        out.push_str(&empty_row(15));
    }

    for (provider, group) in groups {
        let t1 = tier_records(&group, EvalTier::Tier1DefaultZeroShot);
        let t3a = tier_records(&group, EvalTier::Tier3AFewShotK5Pos);
        let t3b = tier_records(&group, EvalTier::Tier3BFewShotK20Pos);
        let t3c = tier_records(&group, EvalTier::Tier3CFewShotK5Contrastive);

        let t1_chrf = mean_chrf(&t1);
        let t1_judge = mean_judge(&t1);
        let t3a_chrf = mean_chrf(&t3a);
        let t3a_judge = mean_judge(&t3a);
        let t3b_chrf = mean_chrf(&t3b);
        let t3b_judge = mean_judge(&t3b);
        let t3c_chrf = mean_chrf(&t3c);
        let t3c_judge = mean_judge(&t3c);

        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            escape_cell(&provider),
            fmt_float(t1_chrf, 3),
            fmt_float(t1_judge, 2),
            fmt_float(t3a_chrf, 3),
            fmt_float(t3a_judge, 2),
            fmt_float(t3b_chrf, 3),
            fmt_float(t3b_judge, 2),
            fmt_float(t3c_chrf, 3),
            fmt_float(t3c_judge, 2),
            fmt_signed(delta(t3a_chrf, t1_chrf)),
            fmt_signed(delta(t3a_judge, t1_judge)),
            fmt_signed(delta(t3b_chrf, t3a_chrf)),
            fmt_signed(delta(t3b_judge, t3a_judge)),
            fmt_signed(delta(t3c_chrf, t3a_chrf)),
            fmt_signed(delta(t3c_judge, t3a_judge)),
        );
    }
    out.push('\n');
}

fn render_telemetry(out: &mut String, records: &[RunRecord]) {
    out.push_str("## Table 5: Operational Telemetry\n\n");
    out.push_str(
        "| Provider | n | p50 TTFT (ms) | p95 TTFT (ms) | p50 Total (ms) | \
         p95 Total (ms) | Mean Uncached Cost | Mean Cached Cost | \
         Uncached / 1k | Cached / 1k |\n",
    );
    out.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");

    let mut groups: BTreeMap<String, Vec<&RunRecord>> = BTreeMap::new();
    for record in records {
        groups
            .entry(record.provider_id.clone())
            .or_default()
            .push(record);
    }

    if groups.is_empty() {
        out.push_str(&empty_row(10));
    }

    for (provider, group) in groups {
        let n = group.len();
        let mut ttfts: Vec<f64> = group.iter().filter_map(|record| record.ttft_ms).collect();
        let mut totals: Vec<f64> = group.iter().map(|record| record.total_latency_ms).collect();

        let p50_ttft = percentile(&mut ttfts, 50.0);
        let p95_ttft = percentile(&mut ttfts, 95.0);
        let p50_total = percentile(&mut totals, 50.0);
        let p95_total = percentile(&mut totals, 95.0);

        let mean_uncached = mean_of(group.iter().map(|record| record.uncached_cost_usd));
        let mean_cached = mean_of(group.iter().map(|record| record.cached_cost_usd));

        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            escape_cell(&provider),
            n,
            fmt_float(p50_ttft, 2),
            fmt_float(p95_ttft, 2),
            fmt_float(p50_total, 2),
            fmt_float(p95_total, 2),
            fmt_money(mean_uncached),
            fmt_money(mean_cached),
            fmt_projected(mean_uncached),
            fmt_projected(mean_cached),
        );
    }
    out.push('\n');
}

fn mean_of<I>(values: I) -> Option<f64>
where
    I: IntoIterator<Item = f64>,
{
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        sum += value;
        count += 1;
    }
    if count == 0 {
        None
    } else {
        Some(sum / count as f64)
    }
}

fn mean_chrf(records: &[&RunRecord]) -> Option<f64> {
    mean_of(records.iter().filter_map(|record| record.chrf))
}

fn mean_judge(records: &[&RunRecord]) -> Option<f64> {
    mean_of(
        records
            .iter()
            .filter_map(|record| record.ensemble.as_ref())
            .map(|ensemble| ensemble.mean.overall() as f64),
    )
}

fn tier_records<'a>(records: &[&'a RunRecord], tier: EvalTier) -> Vec<&'a RunRecord> {
    records
        .iter()
        .copied()
        .filter(|record| record.tier == tier)
        .collect()
}

fn delta(new: Option<f64>, base: Option<f64>) -> Option<f64> {
    match (new, base) {
        (Some(new), Some(base)) => Some(new - base),
        _ => None,
    }
}

fn override_hit(record: &RunRecord) -> bool {
    let Some(override_coarse) = record.conflict_override.as_ref() else {
        return false;
    };
    match record.bias_profile.as_ref() {
        Some(profile) => override_matches_profile(override_coarse, profile),
        None => record.layer1.as_ref().is_some_and(|anchor| anchor.passed),
    }
}

fn override_matches_profile(override_coarse: &CoarseDimensions, profile: &BiasProfile) -> bool {
    let mut matched = false;

    if !override_coarse.formality.is_empty() && override_coarse.formality != "default" {
        matched |= profile.formality.as_deref() == Some(override_coarse.formality.as_str());
    }
    if let Some(value) = override_coarse
        .speaker_gender
        .as_deref()
        .filter(|value| !value.is_empty())
    {
        matched |= profile.speaker_gender.as_deref() == Some(value);
    }
    if let Some(value) = override_coarse
        .listener_gender
        .as_deref()
        .filter(|value| !value.is_empty())
    {
        matched |= profile.listener_gender.as_deref() == Some(value);
    }
    if let Some(value) = override_coarse
        .dialect
        .as_deref()
        .filter(|value| !value.is_empty())
    {
        matched |= profile.dialect.as_deref() == Some(value);
    }

    matched
}

fn value_matches(value: &str, long: &str, short: &str) -> bool {
    value.eq_ignore_ascii_case(long) || value.eq_ignore_ascii_case(short)
}

fn fmt_float(value: Option<f64>, precision: usize) -> String {
    match value {
        Some(value) => format!("{:.*}", precision, value),
        None => na(),
    }
}

fn fmt_signed(value: Option<f64>) -> String {
    match value {
        Some(value) => format!("{value:+.2}"),
        None => na(),
    }
}

fn fmt_money(value: Option<f64>) -> String {
    match value {
        Some(value) => format!("${value:.6}"),
        None => na(),
    }
}

fn fmt_projected(value: Option<f64>) -> String {
    match value {
        Some(value) => format!("${:.6}", value * 1000.0),
        None => na(),
    }
}

fn fmt_pct(count: usize, total: usize) -> String {
    if total == 0 {
        na()
    } else {
        format!("{:.1}%", count as f64 / total as f64 * 100.0)
    }
}

fn count_with_list<I>(count: usize, values: I) -> String
where
    I: IntoIterator<Item = String>,
{
    if count == 0 {
        return "0".to_owned();
    }
    let values: Vec<String> = values.into_iter().collect();
    if values.is_empty() {
        count.to_string()
    } else {
        format!("{} ({})", count, values.join("; "))
    }
}

fn escape_cell(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', " ")
}

fn na() -> String {
    "n/a".to_owned()
}

fn empty_row(columns: usize) -> String {
    let mut row = String::from("|");
    for _ in 0..columns {
        row.push_str(" n/a |");
    }
    row.push('\n');
    row
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{
        AnchorResult, ComplianceAnchors, EnsembleResult, GranularDimensions, JudgeScores, Language,
    };

    fn record(
        provider: &str,
        lang: Language,
        tier: EvalTier,
        chrf: f64,
        judge: f32,
    ) -> RunRecord {
        let scores = JudgeScores {
            formality_compliance: judge,
            gender_agreement: judge,
            dialect_authenticity: judge,
            naturalness_fluency: judge,
        };
        let bias_profile = Some(BiasProfile {
            formality: Some(if lang == Language::Gu {
                "level_2_polite".to_owned()
            } else {
                "default".to_owned()
            }),
            speaker_gender: Some("male".to_owned()),
            listener_gender: Some("female".to_owned()),
            dialect: Some(if lang == Language::Vi {
                "northern".to_owned()
            } else {
                "standard".to_owned()
            }),
            matched_variant_ids: Vec::new(),
            confidence: 0.9,
        });
        let conflict_override = if tier == EvalTier::Tier2CAdversarialConflict {
            Some(CoarseDimensions {
                formality: "level_2_polite".to_owned(),
                ..CoarseDimensions::default()
            })
        } else {
            None
        };

        RunRecord {
            provider_id: provider.to_owned(),
            tier,
            request_id: format!("req-{provider}-{}", tier.label()),
            item_id: "item-1".to_owned(),
            variant_id: "variant-1".to_owned(),
            source_en: "Hello, how are you?".to_owned(),
            lang,
            coarse: CoarseDimensions::default(),
            granular: GranularDimensions::default(),
            conflict_override,
            track: "pragmatics".to_owned(),
            bucket: "smoke".to_owned(),
            is_cued_source: false,
            gold_reference: "gold".to_owned(),
            translation: "translation".to_owned(),
            anchors: ComplianceAnchors::default(),
            capability: CapabilityStatus::Supported,
            ttft_ms: Some(100.0),
            total_latency_ms: 200.0,
            prompt_tokens: Some(10),
            completion_tokens: Some(10),
            char_count: 11,
            uncached_cost_usd: 0.001,
            cached_cost_usd: 0.0005,
            layer1: Some(AnchorResult {
                passed: true,
                ..AnchorResult::default()
            }),
            chrf: Some(chrf),
            ensemble: Some(EnsembleResult {
                judge_a: scores,
                judge_b: scores,
                mean: scores,
                high_disagreement_flag: false,
                disagreement_axes: Vec::new(),
            }),
            bias_profile,
            high_disagreement_flag: false,
        }
    }

    fn headings() -> [&'static str; 5] {
        [
            "## Table 1: Tier 1 Default Quality & Bias Distribution",
            "## Table 2: Tier 2 Parameter Compliance (ACR %)",
            "## Table 3: Tier 2C Adversarial Parameter Override Rate (POR %)",
            "## Table 4: Tier 3 Few-Shot Adaptation Delta",
            "## Table 5: Operational Telemetry",
        ]
    }

    #[test]
    fn markdown_contains_all_tables_and_provider() {
        let records = vec![
            record(
                "prov-x",
                Language::He,
                EvalTier::Tier1DefaultZeroShot,
                0.50,
                3.0,
            ),
            record(
                "prov-x",
                Language::Gu,
                EvalTier::Tier1DefaultZeroShot,
                0.70,
                4.0,
            ),
            record(
                "prov-x",
                Language::Vi,
                EvalTier::Tier1DefaultZeroShot,
                0.60,
                3.5,
            ),
            record(
                "prov-x",
                Language::He,
                EvalTier::Tier2ACoarseParams,
                0.40,
                2.0,
            ),
            record(
                "prov-x",
                Language::He,
                EvalTier::Tier2BGranularParams,
                0.45,
                2.5,
            ),
            record(
                "prov-x",
                Language::Gu,
                EvalTier::Tier2CAdversarialConflict,
                0.40,
                2.0,
            ),
            record(
                "prov-x",
                Language::He,
                EvalTier::Tier3AFewShotK5Pos,
                0.55,
                3.2,
            ),
            record(
                "prov-x",
                Language::He,
                EvalTier::Tier3BFewShotK20Pos,
                0.60,
                3.4,
            ),
            record(
                "prov-x",
                Language::He,
                EvalTier::Tier3CFewShotK5Contrastive,
                0.58,
                3.3,
            ),
        ];

        let markdown = render_markdown(&records);
        for heading in headings() {
            assert!(markdown.contains(heading), "missing heading: {heading}");
        }
        assert!(markdown.contains("prov-x"));
        assert!(markdown.contains("# Pragmatic MT Evaluation Report"));
    }

    #[test]
    fn empty_records_still_render_headings_and_na() {
        let markdown = render_markdown(&[]);
        for heading in headings() {
            assert!(markdown.contains(heading), "missing heading: {heading}");
        }
        assert!(markdown.contains("n/a"));
    }

    #[test]
    fn percentile_uses_linear_interpolation() {
        let mut values = [10.0, 20.0, 30.0, 40.0];
        assert_eq!(percentile(&mut values, 0.0), Some(10.0));
        assert_eq!(percentile(&mut values, 50.0), Some(25.0));
        assert_eq!(percentile(&mut values, 100.0), Some(40.0));
        assert_eq!(percentile(&mut [], 50.0), None);
    }
}
