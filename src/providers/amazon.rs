//! Amazon Translate provider.
//!
//! The adapter targets AWS-compatible JSON endpoints. Request signing is
//! deliberately out of scope: the `Authorization` header carries the supplied
//! credential so a local proxy or test double can validate routing, while a
//! production deployment is expected to provide an already-signed transport.

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::{json, Value};

use crate::providers::{
    bare_response_envelope, estimate_tokens, normalize_base_url, CostRates, LatencyStopwatch,
    TranslationProvider,
};
use crate::schema::{CapabilityStatus, EvalTier, Language, TelemetryEnvelope, TranslationRequest};

/// Default regional Amazon Translate endpoint.
pub const DEFAULT_AMAZON_BASE_URL: &str = "https://translate.us-east-1.amazonaws.com";

/// Amazon Translate provider using the `TranslateText` JSON operation.
pub struct AmazonTranslateProvider {
    base_url: String,
    access_key: String,
    client: reqwest::Client,
}

impl AmazonTranslateProvider {
    /// Creates a provider. An empty base URL selects the public US East
    /// endpoint.
    pub fn new(base_url: impl Into<String>, access_key: impl Into<String>) -> Self {
        let supplied = base_url.into();
        let base_url = if supplied.trim().is_empty() {
            DEFAULT_AMAZON_BASE_URL.to_owned()
        } else {
            normalize_base_url(&supplied).to_string()
        };
        Self {
            base_url,
            access_key: access_key.into(),
            client: reqwest::Client::new(),
        }
    }

    fn target_lang(lang: Language) -> &'static str {
        match lang {
            Language::He => "he",
            Language::Gu => "gu",
            Language::Vi => "vi",
        }
    }

    fn formality_value(req: &TranslationRequest) -> Option<&'static str> {
        if req.tier != EvalTier::Tier2ACoarseParams {
            return None;
        }
        match req.coarse.formality.trim() {
            "level_1_intimate" => Some("INFORMAL"),
            "" | "default" => None,
            _ => Some("FORMAL"),
        }
    }
}

#[async_trait]
impl TranslationProvider for AmazonTranslateProvider {
    fn provider_id(&self) -> &str {
        "amazon-translate"
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
            EvalTier::Tier3AFewShotK5Pos => CapabilityStatus::Supported,
            EvalTier::Tier2BGranularParams
            | EvalTier::Tier2CAdversarialConflict
            | EvalTier::Tier3BFewShotK20Pos
            | EvalTier::Tier3CFewShotK5Contrastive => CapabilityStatus::UnsupportedByProvider {
                reason: format!("Amazon Translate does not support {}", req.tier.label()),
            },
        }
    }

    async fn translate(&self, req: &TranslationRequest) -> Result<TelemetryEnvelope<String>> {
        let stopwatch = LatencyStopwatch::new();
        let url = format!("{}/", self.base_url);

        let mut body = json!({
            "Text": req.source_en,
            "SourceLanguageCode": "en",
            "TargetLanguageCode": Self::target_lang(req.lang),
        });

        if let Some(formality) = Self::formality_value(req) {
            body["Settings"] = json!({ "Formality": formality });
        }

        if req.tier == EvalTier::Tier3AFewShotK5Pos && !req.examples.is_empty() {
            body["ParallelDataNames"] = json!(["benchmark-parallel-data"]);
        }

        let mut builder = self
            .client
            .post(&url)
            .header("content-type", "application/json")
            .header(
                "X-Amz-Target",
                "AWSShineFrontendService_20170701.TranslateText",
            );

        if !self.access_key.trim().is_empty() {
            builder = builder.header(
                "Authorization",
                format!("AWS4-HMAC-SHA256 Credential={}", self.access_key),
            );
        }

        let response = builder
            .json(&body)
            .send()
            .await
            .context("failed to call Amazon Translate endpoint")?;

        let status = response.status();
        if !status.is_success() {
            let detail = response.text().await.unwrap_or_default();
            anyhow::bail!("Amazon Translate failed with {status}: {detail}");
        }

        let value: Value = response
            .json()
            .await
            .context("failed to decode Amazon Translate response")?;

        let translation = value
            .get("TranslatedText")
            .and_then(Value::as_str)
            .context("Amazon Translate response did not contain TranslatedText")?
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
    async fn sends_formality_settings_and_parses_text() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/"))
            .and(body_partial_json(json!({
                "Settings": {"Formality": "FORMAL"},
                "TargetLanguageCode": "he"
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "TranslatedText": "שלום"
            })))
            .mount(&server)
            .await;

        let provider = AmazonTranslateProvider::new(server.uri(), "access");
        let envelope = provider
            .translate(&request(EvalTier::Tier2ACoarseParams, "level_2_polite"))
            .await
            .expect("translation");

        assert_eq!(envelope.payload, "שלום");
    }

    #[test]
    fn tier2b_is_unsupported() {
        let provider = AmazonTranslateProvider::new("", "access");
        let capability =
            provider.check_capability(&request(EvalTier::Tier2BGranularParams, "default"));
        assert!(matches!(
            capability,
            CapabilityStatus::UnsupportedByProvider { .. }
        ));
    }
}
