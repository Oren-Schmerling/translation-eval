//! Translation provider abstractions and concrete provider adapters.
//!
//! Each adapter implements [`TranslationProvider`], which combines a cheap,
//! synchronous capability check with an asynchronous translation call that is
//! wrapped in a [`TelemetryEnvelope`] carrying latency, token, and cost data.

pub mod amazon;
pub mod azure;
pub mod deepl;
pub mod gcloud;
pub mod llm_router;

pub use amazon::AmazonTranslateProvider;
pub use azure::{AzureGenAiTranslatorProvider, AzureStandardNmtProvider};
pub use deepl::DeepLProvider;
pub use gcloud::{GoogleAdaptiveLlmProvider, GoogleBasicNmtProvider};
pub use llm_router::{LlmFamily, StreamingLlmConfig, StreamingLlmProvider};

use crate::schema::{CapabilityStatus, Language, TelemetryEnvelope, TranslationRequest};

/// Common interface implemented by every translation backend.
#[async_trait::async_trait]
pub trait TranslationProvider: Send + Sync {
    /// Stable identifier used in run records and reports.
    fn provider_id(&self) -> &str;

    /// Declares whether this provider can honor the request's pragmatic cell.
    fn check_capability(&self, req: &TranslationRequest) -> CapabilityStatus;

    /// Performs the translation and returns payload plus telemetry.
    async fn translate(
        &self,
        req: &TranslationRequest,
    ) -> anyhow::Result<TelemetryEnvelope<String>>;
}

/// Per-million-token prices used to estimate request costs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostRates {
    pub input_per_million_usd: f64,
    pub output_per_million_usd: f64,
}

impl CostRates {
    /// Constructs a rate pair from per-million-token prices.
    pub const fn new(input_per_million_usd: f64, output_per_million_usd: f64) -> Self {
        Self {
            input_per_million_usd,
            output_per_million_usd,
        }
    }

    /// A zero-cost rate pair for providers with no per-token billing.
    pub const fn zero() -> Self {
        Self::new(0.0, 0.0)
    }
}

impl Default for CostRates {
    fn default() -> Self {
        Self::zero()
    }
}

/// Coarse token estimate: four characters per token, rounded up, min one.
pub fn estimate_tokens(text: &str) -> u32 {
    let chars = text.chars().count();
    if chars == 0 {
        0
    } else {
        ((chars + 3) / 4).max(1) as u32
    }
}

/// Computes uncached and cached-context costs in USD.
///
/// The cached variant bills the shared context prefix at a quarter of the
/// input rate. The cached context is clamped to the prompt size, and the
/// returned cached cost is never allowed to exceed the uncached cost.
pub fn compute_costs(
    prompt_tokens: u32,
    completion_tokens: u32,
    cached_context_tokens: u32,
    rates: CostRates,
) -> (f64, f64) {
    let prompt = f64::from(prompt_tokens);
    let completion = f64::from(completion_tokens);
    let cached = f64::from(cached_context_tokens.min(prompt_tokens));

    let uncached = (prompt * rates.input_per_million_usd
        + completion * rates.output_per_million_usd)
        / 1_000_000.0;

    let cached_cost = ((prompt - cached) * rates.input_per_million_usd
        + cached * rates.input_per_million_usd * 0.25
        + completion * rates.output_per_million_usd)
        / 1_000_000.0;

    (uncached, cached_cost.min(uncached))
}

/// Minimal wall-clock stopwatch for measuring provider latency.
pub struct LatencyStopwatch {
    start: std::time::Instant,
}

impl LatencyStopwatch {
    /// Starts the stopwatch immediately.
    pub fn new() -> Self {
        Self {
            start: std::time::Instant::now(),
        }
    }

    /// Milliseconds elapsed since construction.
    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }
}

impl Default for LatencyStopwatch {
    fn default() -> Self {
        Self::new()
    }
}

/// Trims surrounding whitespace and any trailing slashes from a base URL.
pub fn normalize_base_url(base: &str) -> String {
    base.trim().trim_end_matches('/').to_owned()
}

/// Returns the target language code used by the evaluation providers.
pub fn target_language_code(lang: Language) -> &'static str {
    match lang {
        Language::He => "he",
        Language::Gu => "gu",
        Language::Vi => "vi",
    }
}

/// Builds a telemetry envelope for a provider that returns plain text.
///
/// `cached_context` is expressed in tokens and clamped during cost
/// computation. Costs are zero when either token count is missing.
pub fn bare_response_envelope(
    payload: String,
    total_ms: f64,
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    rates: CostRates,
    cached_context: f64,
) -> TelemetryEnvelope<String> {
    let char_count = payload.chars().count();
    let (uncached_cost_usd, cached_cost_usd) = match (prompt_tokens, completion_tokens) {
        (Some(prompt), Some(completion)) => {
            let cached_tokens = cached_context.max(0.0).round() as u32;
            compute_costs(prompt, completion, cached_tokens, rates)
        }
        _ => (0.0, 0.0),
    };

    TelemetryEnvelope::new(
        payload,
        None,
        total_ms,
        prompt_tokens,
        completion_tokens,
        char_count,
        uncached_cost_usd,
        cached_cost_usd,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_tokens_rounds_up_and_handles_empty() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("a"), 1);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcde"), 2);
        assert_eq!(estimate_tokens("שלום"), 1);
        assert_eq!(estimate_tokens("שלום עולם"), 3);
    }

    #[test]
    fn compute_costs_applies_cached_discount() {
        let rates = CostRates::new(1_000_000.0, 2_000_000.0);
        let (uncached, cached) = compute_costs(1_000, 500, 1_000, rates);
        // uncached = 1000*1 + 500*2 = 2000
        // cached   = 1000*1*0.25 + 500*2 = 1250
        assert!((uncached - 2_000.0).abs() < 1e-9);
        assert!((cached - 1_250.0).abs() < 1e-9);
        assert!(cached < uncached);
    }

    #[test]
    fn compute_costs_clamps_cached_context_and_never_exceeds_uncached() {
        let rates = CostRates::new(3_000_000.0, 1_000_000.0);
        let (uncached, cached) = compute_costs(100, 100, 10_000, rates);
        assert!(cached <= uncached);
        assert!((uncached - 400.0).abs() < 1e-9);
        assert!((cached - 175.0).abs() < 1e-9);
    }

    #[test]
    fn normalizes_base_urls() {
        assert_eq!(
            normalize_base_url("  https://translation.googleapis.com/  "),
            "https://translation.googleapis.com"
        );
        assert_eq!(normalize_base_url("https://x.example///"), "https://x.example");
        assert_eq!(normalize_base_url(""), "");
    }
}
