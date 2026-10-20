//! Streaming LLM translation provider.
//!
//! A single adapter covers OpenAI-compatible, Anthropic, Vertex, and Cohere
//! streaming APIs. The prompt builder encodes the pragmatic tier description
//! so the same request schema can be evaluated across hosted model families.

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::providers::{
    compute_costs, estimate_tokens, normalize_base_url, CostRates, LatencyStopwatch,
    TranslationProvider,
};
use crate::schema::{
    CapabilityStatus, CoarseDimensions, EvalTier, TelemetryEnvelope, TranslationRequest,
};

/// Supported hosted LLM wire formats.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum LlmFamily {
    #[serde(rename = "openai")]
    OpenAi,
    #[serde(rename = "anthropic")]
    Anthropic,
    #[serde(rename = "vertex")]
    Vertex,
    #[serde(rename = "cohere")]
    Cohere,
}

impl Default for LlmFamily {
    fn default() -> Self {
        Self::OpenAi
    }
}

/// Connection and pricing configuration for a streaming LLM provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingLlmConfig {
    pub endpoint_base_url: String,
    pub api_key: String,
    pub model: String,
    pub family: LlmFamily,
    pub input_cost_per_million_usd: f64,
    pub output_cost_per_million_usd: f64,
}

impl StreamingLlmConfig {
    /// Creates an OpenAI-compatible configuration with default pricing.
    pub fn new(
        endpoint_base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            endpoint_base_url: endpoint_base_url.into(),
            api_key: api_key.into(),
            model: model.into(),
            family: LlmFamily::OpenAi,
            input_cost_per_million_usd: 0.15,
            output_cost_per_million_usd: 0.60,
        }
    }

    /// Selects the wire format used by the endpoint.
    pub fn with_family(mut self, family: LlmFamily) -> Self {
        self.family = family;
        self
    }

    /// Overrides per-million-token pricing.
    pub fn with_rates(
        mut self,
        input_cost_per_million_usd: f64,
        output_cost_per_million_usd: f64,
    ) -> Self {
        self.input_cost_per_million_usd = input_cost_per_million_usd;
        self.output_cost_per_million_usd = output_cost_per_million_usd;
        self
    }
}

/// Streaming LLM provider.
pub struct StreamingLlmProvider {
    config: StreamingLlmConfig,
    client: reqwest::Client,
}

impl StreamingLlmProvider {
    /// Creates a provider from its serializable configuration.
    pub fn new(config: StreamingLlmConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    /// Returns the provider configuration.
    pub fn config(&self) -> &StreamingLlmConfig {
        &self.config
    }

    /// Builds the tier-aware translation prompt for a request.
    pub fn build_prompt(&self, req: &TranslationRequest) -> String {
        build_prompt(req)
    }

    fn endpoint(&self) -> String {
        let base = normalize_base_url(&self.config.endpoint_base_url);
        match &self.config.family {
            LlmFamily::OpenAi | LlmFamily::Cohere => format!("{base}/chat/completions"),
            LlmFamily::Anthropic => format!("{base}/v1/messages"),
            LlmFamily::Vertex => format!(
                "{base}/v1beta/models/{}:streamGenerateContent?alt=sse",
                self.config.model
            ),
        }
    }

    fn request_body(&self, prompt: &str) -> Value {
        match &self.config.family {
            LlmFamily::OpenAi => json!({
                "model": self.config.model,
                "stream": true,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            }),
            LlmFamily::Cohere => json!({
                "model": self.config.model,
                "stream": true,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            }),
            LlmFamily::Anthropic => json!({
                "model": self.config.model,
                "max_tokens": 1024,
                "stream": true,
                "messages": [{"role": "user", "content": prompt}],
            }),
            LlmFamily::Vertex => json!({
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            }),
        }
    }
}

#[async_trait]
impl TranslationProvider for StreamingLlmProvider {
    fn provider_id(&self) -> &str {
        "streaming-llm"
    }

    fn check_capability(&self, _req: &TranslationRequest) -> CapabilityStatus {
        CapabilityStatus::Supported
    }

    async fn translate(&self, req: &TranslationRequest) -> Result<TelemetryEnvelope<String>> {
        let prompt = build_prompt(req);
        let stopwatch = LatencyStopwatch::new();
        let url = self.endpoint();
        let body = self.request_body(&prompt);

        let mut builder = self
            .client
            .post(&url)
            .header("content-type", "application/json");

        match &self.config.family {
            LlmFamily::OpenAi | LlmFamily::Cohere => {
                builder = builder.header(
                    "Authorization",
                    format!("Bearer {}", self.config.api_key),
                );
            }
            LlmFamily::Anthropic => {
                builder = builder
                    .header("x-api-key", &self.config.api_key)
                    .header("anthropic-version", "2023-06-01");
            }
            LlmFamily::Vertex => {
                builder = builder.header("x-goog-api-key", &self.config.api_key);
            }
        }

        let response = builder
            .json(&body)
            .send()
            .await
            .context("failed to call streaming LLM endpoint")?;

        let status = response.status();
        if !status.is_success() {
            let detail = response.text().await.unwrap_or_default();
            bail!("streaming LLM request failed with {status}: {detail}");
        }

        let mut stream = response.bytes_stream();
        let mut buffer: Vec<u8> = Vec::new();
        let mut translation = String::new();
        let mut ttft_ms: Option<f64> = None;
        let mut prompt_tokens: Option<u32> = None;
        let mut completion_tokens: Option<u32> = None;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("failed to read streaming LLM response chunk")?;
            buffer.extend_from_slice(&chunk);

            while let Some(newline) = buffer.iter().position(|byte| *byte == b'\n') {
                let line = String::from_utf8_lossy(&buffer[..newline]).into_owned();
                buffer.drain(..=newline);

                if let Some(parsed) = parse_sse_line(line.trim(), &self.config.family) {
                    if let Some(content) = parsed.content {
                        if ttft_ms.is_none() && !content.is_empty() {
                            ttft_ms = Some(stopwatch.elapsed_ms());
                        }
                        translation.push_str(&content);
                    }
                    if parsed.prompt_tokens.is_some() {
                        prompt_tokens = parsed.prompt_tokens;
                    }
                    if parsed.completion_tokens.is_some() {
                        completion_tokens = parsed.completion_tokens;
                    }
                }
            }
        }

        // A stream may end without a trailing newline.
        let trailing = String::from_utf8_lossy(&buffer);
        if !trailing.trim().is_empty() {
            if let Some(parsed) = parse_sse_line(trailing.trim(), &self.config.family) {
                if let Some(content) = parsed.content {
                    if ttft_ms.is_none() && !content.is_empty() {
                        ttft_ms = Some(stopwatch.elapsed_ms());
                    }
                    translation.push_str(&content);
                }
                if parsed.prompt_tokens.is_some() {
                    prompt_tokens = parsed.prompt_tokens;
                }
                if parsed.completion_tokens.is_some() {
                    completion_tokens = parsed.completion_tokens;
                }
            }
        }

        let total_latency_ms = stopwatch.elapsed_ms();
        let prompt_tokens = prompt_tokens.unwrap_or_else(|| estimate_tokens(&prompt));
        let completion_tokens =
            completion_tokens.unwrap_or_else(|| estimate_tokens(&translation));
        let cached_context_tokens = if req.tier.is_few_shot() {
            (prompt_tokens as f64 * 0.6) as u32
        } else {
            0
        };

        let rates = CostRates::new(
            self.config.input_cost_per_million_usd,
            self.config.output_cost_per_million_usd,
        );
        let (uncached_cost_usd, cached_cost_usd) = compute_costs(
            prompt_tokens,
            completion_tokens,
            cached_context_tokens,
            rates,
        );

        let char_count = translation.chars().count();

        Ok(TelemetryEnvelope::new(
            translation,
            ttft_ms,
            total_latency_ms,
            Some(prompt_tokens),
            Some(completion_tokens),
            char_count,
            uncached_cost_usd,
            cached_cost_usd,
        ))
    }
}

/// Builds a tier-aware translation prompt from a request.
pub fn build_prompt(req: &TranslationRequest) -> String {
    let mut prompt = String::new();

    prompt.push_str(&format!(
        "You are a professional translator. Translate the following English source text into {}.\n",
        req.lang.english_name()
    ));
    prompt.push_str(
        "Preserve pragmatic nuances such as formality, speaker and listener gender, and dialect.\n\n",
    );

    match req.tier {
        EvalTier::Tier2ACoarseParams
        | EvalTier::Tier2BGranularParams
        | EvalTier::Tier2CAdversarialConflict => {
            prompt.push_str(&format!(
                "COARSE PARAMETERS: {}\n\n",
                coarse_json(&req.coarse)
            ));

            if req.tier == EvalTier::Tier2BGranularParams {
                prompt.push_str(&format!(
                    "GRANULAR PARAMETERS: {}\n\n",
                    json!({
                        "pronoun_tier": req.granular.pronoun_tier,
                        "kinship_pair": req.granular.kinship_pair,
                        "relationship": req.granular.relationship,
                    })
                ));
            }

            if req.tier == EvalTier::Tier2CAdversarialConflict {
                let override_value = req
                    .conflict_override
                    .as_ref()
                    .map(coarse_json)
                    .unwrap_or_else(|| json!({}));
                prompt.push_str(&format!(
                    "ADVERSARIAL OVERRIDE: {override_value}\n"
                ));
                prompt.push_str(
                    "These API parameters override any conflicting cues in the source text.\n\n",
                );
            }
        }
        EvalTier::Tier1DefaultZeroShot
        | EvalTier::Tier3AFewShotK5Pos
        | EvalTier::Tier3BFewShotK20Pos
        | EvalTier::Tier3CFewShotK5Contrastive => {}
    }

    if !req.examples.is_empty() {
        prompt.push_str("Examples:\n\n");
        for (index, example) in req.examples.iter().enumerate() {
            let number = index + 1;
            if let Some(negative) = example.negative.as_deref() {
                prompt.push_str(&format!(
                    "Contrastive Example {number}:\nSource: {}\nDO NOT USE: {negative}\nUSE: {}\n\n",
                    example.source_en, example.target
                ));
            } else {
                prompt.push_str(&format!(
                    "Positive Example {number}:\nSource: {}\nTarget: {}\n\n",
                    example.source_en, example.target
                ));
            }
        }
    }

    prompt.push_str(&format!(
        "Source: {}\n\nTranslation:",
        req.source_en
    ));

    prompt
}

fn coarse_json(coarse: &CoarseDimensions) -> Value {
    json!({
        "formality": coarse.formality,
        "speaker_gender": coarse.speaker_gender,
        "listener_gender": coarse.listener_gender,
        "dialect": coarse.dialect,
    })
}

struct ParsedSse {
    content: Option<String>,
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
}

fn parse_sse_line(line: &str, family: &LlmFamily) -> Option<ParsedSse> {
    let line = line.trim();
    let data = line.strip_prefix("data:")?.trim();
    if data.is_empty() || data == "[DONE]" {
        return None;
    }

    let value: Value = serde_json::from_str(data).ok()?;

    let content = match family {
        LlmFamily::OpenAi => value
            .pointer("/choices/0/delta/content")
            .and_then(Value::as_str)
            .or_else(|| {
                value
                    .pointer("/choices/0/message/content")
                    .and_then(Value::as_str)
            }),
        LlmFamily::Anthropic => value
            .pointer("/delta/text")
            .and_then(Value::as_str)
            .or_else(|| {
                value
                    .pointer("/content_block/delta/text")
                    .and_then(Value::as_str)
            }),
        LlmFamily::Vertex => value
            .pointer("/candidates/0/content/parts/0/text")
            .and_then(Value::as_str),
        LlmFamily::Cohere => value
            .pointer("/delta/message/content")
            .and_then(Value::as_str)
            .or_else(|| {
                value
                    .pointer("/delta/message/content/text")
                    .and_then(Value::as_str)
            })
            .or_else(|| {
                value
                    .pointer("/delta/content/text")
                    .and_then(Value::as_str)
            }),
    }
    .filter(|text| !text.is_empty())
    .map(str::to_owned);

    let (prompt_tokens, completion_tokens) = match family {
        LlmFamily::OpenAi => (
            value
                .pointer("/usage/prompt_tokens")
                .and_then(Value::as_u64)
                .map(|v| v as u32),
            value
                .pointer("/usage/completion_tokens")
                .and_then(Value::as_u64)
                .map(|v| v as u32),
        ),
        LlmFamily::Anthropic => (
            value
                .pointer("/message/usage/input_tokens")
                .and_then(Value::as_u64)
                .or_else(|| {
                    value
                        .pointer("/usage/input_tokens")
                        .and_then(Value::as_u64)
                })
                .map(|v| v as u32),
            value
                .pointer("/message/usage/output_tokens")
                .and_then(Value::as_u64)
                .or_else(|| {
                    value
                        .pointer("/message_delta/usage/output_tokens")
                        .and_then(Value::as_u64)
                })
                .or_else(|| {
                    value
                        .pointer("/usage/output_tokens")
                        .and_then(Value::as_u64)
                })
                .map(|v| v as u32),
        ),
        LlmFamily::Vertex => (
            value
                .pointer("/usageMetadata/promptTokenCount")
                .and_then(Value::as_u64)
                .map(|v| v as u32),
            value
                .pointer("/usageMetadata/candidatesTokenCount")
                .and_then(Value::as_u64)
                .map(|v| v as u32),
        ),
        LlmFamily::Cohere => (None, None),
    };

    Some(ParsedSse {
        content,
        prompt_tokens,
        completion_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{
        ComplianceAnchors, FewShotExample, GranularDimensions, Language,
    };
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn base_request(tier: EvalTier) -> TranslationRequest {
        TranslationRequest {
            request_id: "req-llm".to_owned(),
            tier,
            item_id: "item-llm".to_owned(),
            variant_id: "variant-llm".to_owned(),
            source_en: "Good morning, my friend.".to_owned(),
            lang: Language::He,
            coarse: CoarseDimensions {
                formality: "level_2_polite".to_owned(),
                speaker_gender: Some("male".to_owned()),
                listener_gender: Some("female".to_owned()),
                dialect: Some("levantine".to_owned()),
            },
            granular: GranularDimensions {
                pronoun_tier: Some("respectful".to_owned()),
                kinship_pair: Some("friend".to_owned()),
                relationship: Some("acquaintance".to_owned()),
            },
            conflict_override: Some(CoarseDimensions {
                formality: "level_1_intimate".to_owned(),
                speaker_gender: Some("female".to_owned()),
                listener_gender: Some("male".to_owned()),
                dialect: None,
            }),
            examples: Vec::new(),
            gold_reference: String::new(),
            anchors: ComplianceAnchors::default(),
        }
    }

    #[test]
    fn tier2b_prompt_contains_granular_pronoun_tier() {
        let req = base_request(EvalTier::Tier2BGranularParams);
        let prompt = build_prompt(&req);
        assert!(prompt.contains("GRANULAR PARAMETERS:"));
        assert!(prompt.contains("respectful"));
    }

    #[test]
    fn tier2c_prompt_contains_adversarial_override() {
        let req = base_request(EvalTier::Tier2CAdversarialConflict);
        let prompt = build_prompt(&req);
        assert!(prompt.contains("ADVERSARIAL OVERRIDE:"));
        assert!(prompt.contains("override any conflicting cues"));
    }

    #[tokio::test]
    async fn streams_openai_content_and_telemetry() {
        let server = MockServer::start().await;
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"שלום \"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"עולם\"}}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":10}}\n\n",
            "data: [DONE]\n\n",
        );
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&server)
            .await;

        let config =
            StreamingLlmConfig::new(server.uri(), "api-key", "test-model");
        let provider = StreamingLlmProvider::new(config);

        let mut req = base_request(EvalTier::Tier3CFewShotK5Contrastive);
        req.examples = (1..=5)
            .map(|n| FewShotExample {
                source_en: format!("Source {n}"),
                target: format!("Target {n}"),
                negative: Some(format!("Negative {n}")),
            })
            .collect();

        let envelope = provider.translate(&req).await.expect("streamed translation");

        assert_eq!(envelope.payload, "שלום עולם");
        assert_eq!(envelope.prompt_tokens, Some(100));
        assert_eq!(envelope.completion_tokens, Some(10));
        let ttft = envelope.ttft_ms.expect("ttft recorded");
        assert!(ttft <= envelope.total_latency_ms);
        assert!(envelope.uncached_cost_usd > envelope.cached_cost_usd);
        assert!(provider.check_capability(&req).is_supported());

        let requests = server.received_requests().await.expect("captured requests");
        let request = requests.first().expect("captured request");
        let body_text = String::from_utf8_lossy(&request.body);
        assert!(body_text.contains("DO NOT USE:"));
        assert!(body_text.contains("USE:"));
        assert!(body_text.contains("Target 5"));
    }
}
