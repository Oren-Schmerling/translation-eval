//! Strongly typed data model shared by the pragmatic MT evaluation pipeline.
//!
//! The types in this module deliberately contain no I/O beyond dataset
//! loading/saving. Keeping the schema self-contained makes it usable from the
//! library, the CLI, tests, and downstream tooling without pulling in provider
//! or scoring implementations.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::Path;
use std::str::FromStr;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

/// Languages covered by the evaluation suite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    He,
    Gu,
    Vi,
}

impl Language {
    /// Returns the ISO-style language code used in data files.
    pub fn code(self) -> &'static str {
        match self {
            Self::He => "he",
            Self::Gu => "gu",
            Self::Vi => "vi",
        }
    }

    /// Returns the English display name for the language.
    pub fn english_name(self) -> &'static str {
        match self {
            Self::He => "Hebrew",
            Self::Gu => "Gujarati",
            Self::Vi => "Vietnamese",
        }
    }

    /// Parses a language from a two-letter code, case-insensitively.
    pub fn from_code(code: &str) -> Option<Self> {
        match code.trim() {
            c if c.eq_ignore_ascii_case("he") => Some(Self::He),
            c if c.eq_ignore_ascii_case("gu") => Some(Self::Gu),
            c if c.eq_ignore_ascii_case("vi") => Some(Self::Vi),
            _ => None,
        }
    }
}

impl fmt::Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

impl FromStr for Language {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some(lang) = Self::from_code(s) {
            return Ok(lang);
        }
        if s.trim().eq_ignore_ascii_case("hebrew") {
            Ok(Self::He)
        } else if s.trim().eq_ignore_ascii_case("gujarati") {
            Ok(Self::Gu)
        } else if s.trim().eq_ignore_ascii_case("vietnamese") {
            Ok(Self::Vi)
        } else {
            bail!("unknown language: {s:?}")
        }
    }
}

/// Evaluation tiers ordered from least to most demanding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvalTier {
    Tier1DefaultZeroShot,
    Tier2ACoarseParams,
    Tier2BGranularParams,
    Tier2CAdversarialConflict,
    Tier3AFewShotK5Pos,
    Tier3BFewShotK20Pos,
    Tier3CFewShotK5Contrastive,
}

impl EvalTier {
    /// Stable Rust-style label used in reports and run records.
    pub fn label(self) -> &'static str {
        match self {
            Self::Tier1DefaultZeroShot => "Tier1DefaultZeroShot",
            Self::Tier2ACoarseParams => "Tier2ACoarseParams",
            Self::Tier2BGranularParams => "Tier2BGranularParams",
            Self::Tier2CAdversarialConflict => "Tier2CAdversarialConflict",
            Self::Tier3AFewShotK5Pos => "Tier3AFewShotK5Pos",
            Self::Tier3BFewShotK20Pos => "Tier3BFewShotK20Pos",
            Self::Tier3CFewShotK5Contrastive => "Tier3CFewShotK5Contrastive",
        }
    }

    /// Whether the tier supplies few-shot examples in the prompt.
    pub fn is_few_shot(self) -> bool {
        matches!(
            self,
            Self::Tier3AFewShotK5Pos
                | Self::Tier3BFewShotK20Pos
                | Self::Tier3CFewShotK5Contrastive
        )
    }

    /// Number of positive examples used by this tier.
    pub fn few_shot_k(self) -> usize {
        match self {
            Self::Tier1DefaultZeroShot
            | Self::Tier2ACoarseParams
            | Self::Tier2BGranularParams
            | Self::Tier2CAdversarialConflict => 0,
            Self::Tier3AFewShotK5Pos => 5,
            Self::Tier3BFewShotK20Pos => 20,
            Self::Tier3CFewShotK5Contrastive => 5,
        }
    }

    /// Whether the tier uses explicitly contrastive (positive/negative) examples.
    pub fn is_contrastive(self) -> bool {
        matches!(self, Self::Tier3CFewShotK5Contrastive)
    }
}

impl fmt::Display for EvalTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

impl FromStr for EvalTier {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Accept Rust variant names, snake_case, kebab-case, and short aliases.
        let normalized: String = s
            .chars()
            .filter(|c| !matches!(*c, '_' | '-' | ' '))
            .flat_map(char::to_lowercase)
            .collect();

        match normalized.as_str() {
            "tier1defaultzeroshot" | "1" | "tier1" => Ok(Self::Tier1DefaultZeroShot),
            "tier2acoarseparams" | "tier2a" | "2a" => Ok(Self::Tier2ACoarseParams),
            "tier2bgranularparams" | "tier2b" | "2b" => Ok(Self::Tier2BGranularParams),
            "tier2cadversarialconflict" | "tier2c" | "2c" => {
                Ok(Self::Tier2CAdversarialConflict)
            }
            "tier3afewshotk5pos" | "tier3a" | "3a" => Ok(Self::Tier3AFewShotK5Pos),
            "tier3bfewshotk20pos" | "tier3b" | "3b" => Ok(Self::Tier3BFewShotK20Pos),
            "tier3cfewshotk5contrastive" | "tier3c" | "3c" => {
                Ok(Self::Tier3CFewShotK5Contrastive)
            }
            _ => bail!("unknown evaluation tier: {s:?}"),
        }
    }
}

/// Coarse pragmatic dimensions controlling formality and participant gender.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoarseDimensions {
    pub formality: String,
    pub speaker_gender: Option<String>,
    pub listener_gender: Option<String>,
    pub dialect: Option<String>,
}

impl Default for CoarseDimensions {
    fn default() -> Self {
        Self {
            formality: "default".to_owned(),
            speaker_gender: None,
            listener_gender: None,
            dialect: None,
        }
    }
}

impl CoarseDimensions {
    /// True when no non-default dimension is populated.
    pub fn is_empty(&self) -> bool {
        self.populated_axes().is_empty()
    }

    /// Names of dimensions that carry information beyond the default cell.
    pub fn populated_axes(&self) -> Vec<String> {
        let mut axes = Vec::new();
        if !self.formality.is_empty() && self.formality != "default" {
            axes.push("formality".to_owned());
        }
        if self.speaker_gender.as_deref().is_some_and(|v| !v.is_empty()) {
            axes.push("speaker_gender".to_owned());
        }
        if self.listener_gender.as_deref().is_some_and(|v| !v.is_empty()) {
            axes.push("listener_gender".to_owned());
        }
        if self.dialect.as_deref().is_some_and(|v| !v.is_empty()) {
            axes.push("dialect".to_owned());
        }
        axes
    }

    /// Returns a copy where non-empty overrides replace this cell's values.
    pub fn with_override(&self, o: &CoarseDimensions) -> Self {
        let mut merged = self.clone();
        if !o.formality.is_empty() {
            merged.formality = o.formality.clone();
        }
        if o.speaker_gender.is_some() {
            merged.speaker_gender = o.speaker_gender.clone();
        }
        if o.listener_gender.is_some() {
            merged.listener_gender = o.listener_gender.clone();
        }
        if o.dialect.is_some() {
            merged.dialect = o.dialect.clone();
        }
        merged
    }
}

/// Fine-grained dimensions such as pronoun tier and kinship relationships.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct GranularDimensions {
    pub pronoun_tier: Option<String>,
    pub kinship_pair: Option<String>,
    pub relationship: Option<String>,
}

impl GranularDimensions {
    /// True when every granular dimension is absent or empty.
    pub fn is_empty(&self) -> bool {
        self.pronoun_tier.as_deref().map_or(true, str::is_empty)
            && self.kinship_pair.as_deref().map_or(true, str::is_empty)
            && self.relationship.as_deref().map_or(true, str::is_empty)
    }
}

/// Lexical constraints used by automatic compliance checks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ComplianceAnchors {
    pub must_contain_any: Vec<String>,
    pub must_not_contain: Vec<String>,
    pub synset_equivalents: HashMap<String, Vec<String>>,
    pub acceptable_synonyms: Vec<String>,
}

impl ComplianceAnchors {
    /// True when no anchor or synonym constraints are present.
    pub fn is_empty(&self) -> bool {
        self.must_contain_any.is_empty()
            && self.must_not_contain.is_empty()
            && self.acceptable_synonyms.is_empty()
            && self
                .synset_equivalents
                .values()
                .all(|equivalents| equivalents.is_empty())
    }
}

/// A single target-language variant within a source item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetVariant {
    pub variant_id: String,
    pub item_id: String,
    pub source_en: String,
    pub lang: Language,
    pub track: String,
    pub bucket: String,
    pub is_cued_source: bool,
    pub coarse: CoarseDimensions,
    pub granular: GranularDimensions,
    pub conflict_override: Option<CoarseDimensions>,
    #[serde(default)]
    pub gold_reference: String,
    #[serde(default)]
    pub anchors: ComplianceAnchors,
}

/// A single prompt example, optionally carrying a contrastive negative.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FewShotExample {
    pub source_en: String,
    pub target: String,
    pub negative: Option<String>,
}

/// Fully resolved translation request handed to a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationRequest {
    pub request_id: String,
    pub tier: EvalTier,
    pub item_id: String,
    pub variant_id: String,
    pub source_en: String,
    pub lang: Language,
    pub coarse: CoarseDimensions,
    pub granular: GranularDimensions,
    pub conflict_override: Option<CoarseDimensions>,
    pub examples: Vec<FewShotExample>,
    pub gold_reference: String,
    pub anchors: ComplianceAnchors,
}

impl TranslationRequest {
    /// The coarse cell that should actually be used for this request.
    pub fn effective_coarse(&self) -> &CoarseDimensions {
        self.conflict_override.as_ref().unwrap_or(&self.coarse)
    }
}

/// Provider capability declaration for a requested pragmatic cell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum CapabilityStatus {
    Supported,
    PartiallySupported {
        supported_axes: Vec<String>,
        missing_axes: Vec<String>,
    },
    UnsupportedByProvider {
        reason: String,
    },
}

impl CapabilityStatus {
    /// True only when the provider fully supports the request.
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::Supported)
    }

    /// True when the provider can answer, fully or partially.
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Supported | Self::PartiallySupported { .. })
    }

    /// Stable human-readable capability label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Supported => "Supported",
            Self::PartiallySupported { .. } => "PartiallySupported",
            Self::UnsupportedByProvider { .. } => "UnsupportedByProvider",
        }
    }

    /// Axes that a partially supporting provider cannot honor.
    pub fn missing_axes(&self) -> Vec<String> {
        match self {
            Self::PartiallySupported { missing_axes, .. } => missing_axes.clone(),
            Self::Supported | Self::UnsupportedByProvider { .. } => Vec::new(),
        }
    }
}

impl fmt::Display for CapabilityStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Telemetry captured around a provider call.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TelemetryEnvelope<T> {
    pub payload: T,
    pub ttft_ms: Option<f64>,
    pub total_latency_ms: f64,
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub char_count: usize,
    pub uncached_cost_usd: f64,
    pub cached_cost_usd: f64,
}

impl<T> TelemetryEnvelope<T> {
    /// Constructs a telemetry envelope from already-measured values.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        payload: T,
        ttft_ms: Option<f64>,
        total_latency_ms: f64,
        prompt_tokens: Option<u32>,
        completion_tokens: Option<u32>,
        char_count: usize,
        uncached_cost_usd: f64,
        cached_cost_usd: f64,
    ) -> Self {
        Self {
            payload,
            ttft_ms,
            total_latency_ms,
            prompt_tokens,
            completion_tokens,
            char_count,
            uncached_cost_usd,
            cached_cost_usd,
        }
    }
}

/// Result of checking a translation against lexical anchors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct AnchorResult {
    pub passed: bool,
    pub matched_must_contain: Vec<String>,
    pub forbidden_hits: Vec<String>,
    pub matched_equivalents: Vec<String>,
}

/// Inferred pragmatic profile of a translation or source item.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BiasProfile {
    pub formality: Option<String>,
    pub speaker_gender: Option<String>,
    pub listener_gender: Option<String>,
    pub dialect: Option<String>,
    pub matched_variant_ids: Vec<String>,
    pub confidence: f64,
}

/// Scores produced by a single judge along the four pragmatic axes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct JudgeScores {
    pub formality_compliance: f32,
    pub gender_agreement: f32,
    pub dialect_authenticity: f32,
    pub naturalness_fluency: f32,
}

impl Default for JudgeScores {
    fn default() -> Self {
        Self {
            formality_compliance: 0.0,
            gender_agreement: 0.0,
            dialect_authenticity: 0.0,
            naturalness_fluency: 0.0,
        }
    }
}

impl JudgeScores {
    /// Arithmetic mean of the four axis scores.
    pub fn overall(&self) -> f32 {
        (self.formality_compliance
            + self.gender_agreement
            + self.dialect_authenticity
            + self.naturalness_fluency)
            / 4.0
    }

    /// Axis names paired with their scores, in stable report order.
    pub fn axes(&self) -> [(&'static str, f32); 4] {
        [
            ("formality_compliance", self.formality_compliance),
            ("gender_agreement", self.gender_agreement),
            ("dialect_authenticity", self.dialect_authenticity),
            ("naturalness_fluency", self.naturalness_fluency),
        ]
    }
}

/// Aggregated result from two judges, including disagreement diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleResult {
    pub judge_a: JudgeScores,
    pub judge_b: JudgeScores,
    pub mean: JudgeScores,
    pub high_disagreement_flag: bool,
    pub disagreement_axes: Vec<String>,
}

/// Flat, serializable record for one provider call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRecord {
    pub provider_id: String,
    pub tier: EvalTier,
    pub request_id: String,
    pub item_id: String,
    pub variant_id: String,
    pub source_en: String,
    pub lang: Language,
    pub coarse: CoarseDimensions,
    pub granular: GranularDimensions,
    pub conflict_override: Option<CoarseDimensions>,
    pub track: String,
    pub bucket: String,
    pub is_cued_source: bool,
    pub gold_reference: String,
    pub translation: String,
    pub anchors: ComplianceAnchors,
    pub capability: CapabilityStatus,
    pub ttft_ms: Option<f64>,
    pub total_latency_ms: f64,
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub char_count: usize,
    pub uncached_cost_usd: f64,
    pub cached_cost_usd: f64,
    #[serde(default)]
    pub layer1: Option<AnchorResult>,
    #[serde(default)]
    pub chrf: Option<f64>,
    #[serde(default)]
    pub ensemble: Option<EnsembleResult>,
    #[serde(default)]
    pub bias_profile: Option<BiasProfile>,
    #[serde(default)]
    pub high_disagreement_flag: bool,
}

impl RunRecord {
    /// Builds a run record from a request and the provider response envelope.
    ///
    /// Request-level fields are copied verbatim. Dataset-only metadata
    /// (`track`, `bucket`, `is_cued_source`) is left at its neutral default
    /// because it is not part of [`TranslationRequest`].
    pub fn from_translation(
        req: &TranslationRequest,
        provider_id: impl Into<String>,
        envelope: &TelemetryEnvelope<String>,
        capability: CapabilityStatus,
    ) -> Self {
        Self {
            provider_id: provider_id.into(),
            tier: req.tier,
            request_id: req.request_id.clone(),
            item_id: req.item_id.clone(),
            variant_id: req.variant_id.clone(),
            source_en: req.source_en.clone(),
            lang: req.lang,
            coarse: req.coarse.clone(),
            granular: req.granular.clone(),
            conflict_override: req.conflict_override.clone(),
            track: String::new(),
            bucket: String::new(),
            is_cued_source: false,
            gold_reference: req.gold_reference.clone(),
            translation: envelope.payload.clone(),
            anchors: req.anchors.clone(),
            capability,
            ttft_ms: envelope.ttft_ms,
            total_latency_ms: envelope.total_latency_ms,
            prompt_tokens: envelope.prompt_tokens,
            completion_tokens: envelope.completion_tokens,
            char_count: envelope.char_count,
            uncached_cost_usd: envelope.uncached_cost_usd,
            cached_cost_usd: envelope.cached_cost_usd,
            layer1: None,
            chrf: None,
            ensemble: None,
            bias_profile: None,
            high_disagreement_flag: false,
        }
    }
}

/// Collection of target variants with dataset-aware lookup helpers.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EvalDataset {
    pub variants: Vec<TargetVariant>,
}

impl EvalDataset {
    /// Loads a dataset from JSON.
    ///
    /// Both `{"variants": [...]}` and a bare JSON array of variants are
    /// accepted so hand-authored fixtures remain convenient.
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read dataset {}", path.display()))?;
        let value: serde_json::Value = serde_json::from_str(&raw)
            .with_context(|| format!("failed to parse dataset JSON {}", path.display()))?;

        let dataset = match value {
            serde_json::Value::Array(items) => {
                let variants: Vec<TargetVariant> = serde_json::from_value(
                    serde_json::Value::Array(items),
                )
                .with_context(|| {
                    format!(
                        "failed to decode bare variant array in {}",
                        path.display()
                    )
                })?;
                Self { variants }
            }
            other @ serde_json::Value::Object(_) => serde_json::from_value(other).with_context(|| {
                format!(
                    "failed to decode dataset object in {}",
                    path.display()
                )
            })?,
            _ => bail!(
                "dataset {} must be a JSON object or array",
                path.display()
            ),
        };

        Ok(dataset)
    }

    /// Writes the dataset as pretty JSON, creating parent directories.
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).with_context(|| {
                    format!("failed to create dataset directory {}", parent.display())
                })?;
            }
        }
        let mut encoded = serde_json::to_string_pretty(self)
            .context("failed to encode evaluation dataset")?;
        encoded.push('\n');
        std::fs::write(path, encoded)
            .with_context(|| format!("failed to write dataset {}", path.display()))?;
        Ok(())
    }

    /// All variants belonging to a source item.
    pub fn for_item(&self, item_id: &str) -> Vec<&TargetVariant> {
        self.variants
            .iter()
            .filter(|v| v.item_id == item_id)
            .collect()
    }

    /// Variants for the same item, excluding the supplied variant id.
    pub fn siblings(&self, item_id: &str, variant_id: &str) -> Vec<&TargetVariant> {
        self.variants
            .iter()
            .filter(|v| v.item_id == item_id && v.variant_id != variant_id)
            .collect()
    }

    /// Positive-only few-shot examples ordered by pragmatic closeness.
    pub fn few_shot_examples(
        &self,
        target: &TargetVariant,
        k: usize,
    ) -> Vec<FewShotExample> {
        if k == 0 {
            return Vec::new();
        }

        let mut candidates: Vec<&TargetVariant> = self
            .variants
            .iter()
            .filter(|v| v.lang == target.lang && v.item_id != target.item_id)
            .collect();

        // Stable sort keeps selection deterministic while preferring the exact
        // coarse cell, then the same formality, then any in-language variant.
        candidates.sort_by_key(|v| {
            if coarse_cell_eq(&v.coarse, &target.coarse) {
                0u8
            } else if v.coarse.formality == target.coarse.formality {
                1
            } else {
                2
            }
        });

        let mut seen = HashSet::new();
        let mut examples = Vec::new();
        for variant in candidates {
            if examples.len() >= k {
                break;
            }
            if seen.insert(variant.variant_id.clone()) {
                examples.push(FewShotExample {
                    source_en: variant.source_en.clone(),
                    target: variant.gold_reference.clone(),
                    negative: None,
                });
            }
        }
        examples
    }

    /// Contrastive few-shot examples for the target's coarse cell.
    pub fn contrastive_examples(
        &self,
        target: &TargetVariant,
        k: usize,
    ) -> Vec<FewShotExample> {
        if k == 0 {
            return Vec::new();
        }

        let pool: Vec<&TargetVariant> = self
            .variants
            .iter()
            .filter(|v| v.lang == target.lang && v.item_id != target.item_id)
            .collect();

        let mut positives: Vec<&TargetVariant> = pool
            .iter()
            .copied()
            .filter(|v| coarse_cell_eq(&v.coarse, &target.coarse))
            .collect();

        let mut negatives: Vec<&TargetVariant> = pool
            .iter()
            .copied()
            .filter(|v| coarse_differs_on_populated_axis(&target.coarse, &v.coarse))
            .collect();

        // Prefer clearly wrong/default variants as contrastive negatives.
        negatives.sort_by_key(|v| {
            let wrong_formality = v.coarse.formality != target.coarse.formality;
            let is_default = v.coarse.formality == "default";
            match (wrong_formality, is_default) {
                (true, _) => 0u8,
                (false, true) => 1,
                (false, false) => 2,
            }
        });
        positives.sort_by_key(|v| v.variant_id.clone());

        let mut examples = Vec::new();

        if positives.is_empty() {
            let fallback_negatives: Vec<&TargetVariant> = if negatives.is_empty() {
                pool.clone()
            } else {
                negatives.clone()
            };

            for negative in fallback_negatives {
                if examples.len() >= k {
                    break;
                }
                examples.push(FewShotExample {
                    source_en: target.source_en.clone(),
                    target: target.gold_reference.clone(),
                    negative: Some(negative.gold_reference.clone()),
                });
            }
            return examples;
        }

        for positive in positives {
            let available_negatives: Vec<&TargetVariant> = if negatives.is_empty() {
                pool.iter()
                    .copied()
                    .filter(|v| v.variant_id != positive.variant_id)
                    .collect()
            } else {
                negatives.clone()
            };

            for negative in available_negatives {
                if examples.len() >= k {
                    break;
                }
                examples.push(FewShotExample {
                    source_en: positive.source_en.clone(),
                    target: positive.gold_reference.clone(),
                    negative: Some(negative.gold_reference.clone()),
                });
            }

            if examples.len() >= k {
                break;
            }
        }

        examples
    }
}

fn coarse_cell_eq(a: &CoarseDimensions, b: &CoarseDimensions) -> bool {
    a.formality == b.formality
        && a.speaker_gender == b.speaker_gender
        && a.listener_gender == b.listener_gender
        && a.dialect == b.dialect
}

fn coarse_differs_on_populated_axis(
    target: &CoarseDimensions,
    candidate: &CoarseDimensions,
) -> bool {
    if !target.formality.is_empty()
        && target.formality != "default"
        && target.formality != candidate.formality
    {
        return true;
    }

    if let Some(value) = target.speaker_gender.as_deref() {
        if !value.is_empty() && candidate.speaker_gender.as_deref() != Some(value) {
            return true;
        }
    }
    if let Some(value) = target.listener_gender.as_deref() {
        if !value.is_empty() && candidate.listener_gender.as_deref() != Some(value) {
            return true;
        }
    }
    if let Some(value) = target.dialect.as_deref() {
        if !value.is_empty() && candidate.dialect.as_deref() != Some(value) {
            return true;
        }
    }

    false
}
