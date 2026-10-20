//! DeepL translation provider.
//!
//! DeepL exposes a compact REST API with first-class support for formality in
//! several target languages. This adapter only maps the coarse formality axis
//! and reports the remaining pragmatic axes as missing capabilities.

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::{json, Value};

use crate::providers::{
    bare_response_envelope, estimate_tokens, normalize_base_url, CostRates, LatencyStopwatch,
    TranslationProvider,
};
use crate::schema::{
    CapabilityStatus, EvalTier, Language, TelemetryEnvelope, TranslationRequest,
};

/// Default DeepL API endpoint used for free-tier keys.
pub const DEFAULT_DEEPL_BASE_URL: &str = "https://api-free.deepl.com";

/// DeepL provider backed by the REST `/v2/translate` endpoint.
pub struct DeepLProvider {
    base_url: String,
    api_key: String,
    client: reqwest::Client,
}

impl DeepLProvider {
    /// Creates a provider. An empty base URL selects the public DeepL
    /// endpoint, which is convenient for tests and configuration defaults.
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        let supplied = base_url.into();
        let base_url = if supplied.trim().is_empty() {
            DEFAULT_DEEPL_BASE_URL.to_owned()
        } else {
            normalize_base_url(&supplied).to_string()
        };
        Self {
            base_url,
            api_key: api_key.into(),
            client: reqwest::Client::new(),
        }
    }

    fn target_lang(lang: Language) -> &'static str {
        match lang {
            Language::He => "HE",
            Language::Gu => "GU",
            Language::Vi => "VI",
        }
    }

    fn formality_value(req: &TranslationRequest) -> Option<&'static str> {
        if req.tier != EvalTier::Tier2ACoarseParams {
            return None;
        }
        match req.coarse.formality.as_str() {
            "level_1_intimate" => Some("less"),
            "level_2_polite" | "level_3_honorific" => Some("more"),
            _ => None,
        }
    }
}

#[async_trait]
impl TranslationProvider for DeepLProvider {
    fn provider_id(&self) -> &str {
        "deepl"
    }

    fn check_capability(&self, req: &TranslationRequest) -> CapabilityStatus {
        match req.tier {
            EvalTier::Tier1DefaultZeroShot => CapabilityStatus::Supported,
            EvalTier::Tier2ACoarseParams => {
                let mut missing_axes = Vec::new();
                if req
                    .coarse
                    .speaker_gender
                    .as_deref()
                    .is_some_and(|v| !v.is_empty())
                {
                    missing_axes.push("speaker_gender".to_owned());
                }
                if req
                    .coarse
                    .listener_gender
                    .as_deref()
                    .is_some_and(|v| !v.is_empty())
                {
                    missing_axes.push("listener_gender".to_owned());
                }
                if req.coarse.dialect.as_deref().is_some_and(|v| !v.is_empty()) {
                    missing_axes.push("dialect".to_owned());
                }

                if missing_axes.is_empty() {
                    CapabilityStatus::Supported
                } else {
                    CapabilityStatus::PartiallySupported {
                        supported_axes: vec!["formality".to_owned()],
                        missing_axes,
                    }
                }
            }
            EvalTier::Tier2BGranularParams
            | EvalTier::Tier2CAdversarialConflict
            | EvalTier::Tier3AFewShotK5Pos
            | EvalTier::Tier3BFewShotK20Pos
            | EvalTier::Tier3CFewShotK5Contrastive => CapabilityStatus::UnsupportedByProvider {
                reason: format!("DeepL does not support {}", req.tier.label()),
            },
        }
    }

    async fn translate(&self, req: &TranslationRequest) -> Result<TelemetryEnvelope<String>> {
        let stopwatch = LatencyStopwatch::new();
        let url = format!("{}/v2/translate", self.base_url);

        let mut body = json!({
            "text": [req.source_en],
            "target_lang": Self::target_lang(req.lang),
            "source_lang": "EN",
        });

        if let Some(formality) = Self::formality_value(req) {
            body["formality"] = Value::String(formality.to_owned());
        }

        let mut builder = self
            .client
            .post(&url)
            .header("content-type", "application/json");

        if !self.api_key.trim().is_empty() {
            builder = builder.header(
                "Authorization",
                format!("DeepL-Auth-Key {}", self.api_key),
            );
        }

        let response = builder
            .json(&body)
            .send()
            .await
            .context("failed to call DeepL translate endpoint")?;

        let status = response.status();
        if !status.is_success() {
            let detail = response.text().await.unwrap_or_default();
            anyhow::bail!("DeepL translate failed with {status}: {detail}");
        }

        let value: Value = response
            .json()
            .await
            .context("failed to decode DeepL translate response")?;

        let translation = value
            .get("translations")
            .and_then(Value::as_array)
            .and_then(|translations| translations.first())
            .and_then(|translation| translation.get("text"))
            .and_then(Value::as_str)
            .context("DeepL response did not contain translations[0].text")?
            .to_owned();

        let total_latency_ms = stopwatch.elapsed_ms();
        let prompt_tokens = Some(estimate_tokens(&req.source_en));
        let completion_tokens = Some(estimate_tokens(&translation));

        Ok(bare_response_envelope(
            translation,
            total_latency_ms,
            prompt_tokens,
            completion_tokens,
            CostRates::zero(),
            0.0,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{CoarseDimensions, ComplianceAnchors, GranularDimensions};
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn request(tier: EvalTier, formality: &str) -> TranslationRequest {
        TranslationRequest {
            request_id: "req-1".to_owned(),
            tier,
            item_id: "item-1".to_owned(),
            variant_id: "variant-1".to_owned(),
            source_en: "Hello".to_owned(),
            lang: Language::He,
            coarse: CoarseDimensions {
                formality: formality.to_owned(),
                speaker_gender: None,
                listener_gender: None,
                dialect: None,
            },
            granular: GranularDimensions::default(),
            conflict_override: None,
            examples: Vec::new(),
            gold_reference: String::new(),
            anchors: ComplianceAnchors::default(),
        }
    }

    #[tokio::test]
    async fn serializes_formality_less_and_more() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v2/translate"))
            .and(body_partial_json(json!({"formality": "less"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "translations": [{"text": "שלום"}]
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v2/translate"))
            .and(body_partial_json(json!({"formality": "more"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "translations": [{"text": "שלום כבודו"}]
            })))
            .mount(&server)
            .await;

        let provider = DeepLProvider::new(server.uri(), "key");
        let intimate = provider
            .translate(&request(EvalTier::Tier2ACoarseParams, "level_1_intimate"))
            .await
            .expect("intimate translation");
        let polite = provider
            .translate(&request(EvalTier::Tier2ACoarseParams, "level_2_polite"))
            .await
            .expect("polite translation");

        assert_eq!(intimate.payload, "שלום");
        assert_eq!(polite.payload, "שלום כבודו");
    }

    #[test]
    fn reports_missing_gender_and_dialect_axes() {
        let provider = DeepLProvider::new("", "key");
        let mut req = request(EvalTier::Tier2ACoarseParams, "level_2_polite");
        req.coarse.speaker_gender = Some("male".to_owned());
        req.coarse.listener_gender = Some("female".to_owned());
        req.coarse.dialect = Some("levantine".to_owned());

        let capability = provider.check_capability(&req);
        match capability {
            CapabilityStatus::PartiallySupported {
                supported_axes,
                missing_axes,
            } => {
                assert_eq!(supported_axes, vec!["formality".to_owned()]);
                assert!(missing_axes.contains(&"speaker_gender".to_owned()));
                assert!(missing_axes.contains(&"listener_gender".to_owned()));
                assert!(missing_axes.contains(&"dialect".to_owned()));
            }
            other => panic!("unexpected capability: {other:?}"),
        }
    }

    #[tokio::test]
    async fn parses_successful_translation() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v2/translate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "translations": [{"text": "שלום עולם"}]
            })))
            .mount(&server)
            .await;

        let provider = DeepLProvider::new(server.uri(), "key");
        let envelope = provider
            .translate(&request(EvalTier::Tier1DefaultZeroShot, "default"))
            .await
            .expect("translation");

        assert_eq!(envelope.payload, "שלום עולם");
        assert_eq!(envelope.char_count, "שלום עולם".chars().count());
        assert!(envelope.ttft_ms.is_none());
    }
}
