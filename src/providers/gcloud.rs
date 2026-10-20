//! Google Cloud Translation adapters.
//!
//! Two providers are exposed: a basic Neural Machine Translation client that
//! only handles default zero-shot requests, and an adaptive translation-LLM
//! client that additionally supports coarse few-shot prompting.

use anyhow::{bail, Context, Result};
use reqwest::Client;
use serde_json::{json, Value};

use super::{
    bare_response_envelope, estimate_tokens, normalize_base_url, target_language_code,
    CostRates, LatencyStopwatch, TranslationProvider,
};
use crate::schema::{CapabilityStatus, EvalTier, TelemetryEnvelope, TranslationRequest};

/// Production Google Cloud Translation endpoint.
pub const DEFAULT_BASE_URL: &str = "https://translation.googleapis.com";

/// Google Cloud basic NMT provider.
#[derive(Debug, Clone)]
pub struct GoogleBasicNmtProvider {
    base_url: String,
    project_id: String,
    api_key: String,
    client: Client,
}

impl GoogleBasicNmtProvider {
    /// Constructs a basic NMT provider from an endpoint, project, and key.
    pub fn new(
        base_url: impl Into<String>,
        project_id: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        let base_url: String = base_url.into();
        Self {
            base_url: normalize_base_url(&base_url),
            project_id: project_id.into(),
            api_key: api_key.into(),
            client: Client::new(),
        }
    }

    fn endpoint(&self) -> String {
        format!(
            "{}/v3/projects/{}/locations/global:translateText",
            self.base_url, self.project_id
        )
    }
}

#[async_trait::async_trait]
impl TranslationProvider for GoogleBasicNmtProvider {
    fn provider_id(&self) -> &str {
        "google-basic-nmt"
    }

    fn check_capability(&self, req: &TranslationRequest) -> CapabilityStatus {
        match req.tier {
            EvalTier::Tier1DefaultZeroShot => CapabilityStatus::Supported,
            other => CapabilityStatus::UnsupportedByProvider {
                reason: format!(
                    "Google basic NMT supports only Tier1DefaultZeroShot, not {}",
                    other.label()
                ),
            },
        }
    }

    async fn translate(
        &self,
        req: &TranslationRequest,
    ) -> Result<TelemetryEnvelope<String>> {
        if !self.check_capability(req).is_available() {
            bail!(
                "Google basic NMT does not support evaluation tier {}",
                req.tier.label()
            );
        }

        let body = json!({
            "contents": [req.source_en],
            "targetLanguageCode": target_language_code(req.lang),
            "sourceLanguageCode": "en",
            "mimeType": "text/plain",
        });

        let stopwatch = LatencyStopwatch::new();
        let response = self
            .client
            .post(self.endpoint())
            .header("x-goog-api-key", &self.api_key)
            .json(&body)
            .send()
            .await
            .context("Google basic NMT request failed")?;

        let payload = parse_google_response(response).await?;
        let total_ms = stopwatch.elapsed_ms();

        let prompt_tokens = Some(estimate_tokens(&req.source_en));
        let completion_tokens = Some(estimate_tokens(&payload));

        Ok(bare_response_envelope(
            payload,
            total_ms,
            prompt_tokens,
            completion_tokens,
            CostRates::zero(),
            0.0,
        ))
    }
}

/// Google Cloud adaptive translation-LLM provider.
#[derive(Debug, Clone)]
pub struct GoogleAdaptiveLlmProvider {
    base_url: String,
    project_id: String,
    api_key: String,
    client: Client,
}

impl GoogleAdaptiveLlmProvider {
    /// Constructs an adaptive LLM provider from an endpoint, project, and key.
    pub fn new(
        base_url: impl Into<String>,
        project_id: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        let base_url: String = base_url.into();
        Self {
            base_url: normalize_base_url(&base_url),
            project_id: project_id.into(),
            api_key: api_key.into(),
            client: Client::new(),
        }
    }

    fn endpoint(&self) -> String {
        format!(
            "{}/v3/projects/{}/locations/global:translateText",
            self.base_url, self.project_id
        )
    }
}

#[async_trait::async_trait]
impl TranslationProvider for GoogleAdaptiveLlmProvider {
    fn provider_id(&self) -> &str {
        "google-adaptive-llm"
    }

    fn check_capability(&self, req: &TranslationRequest) -> CapabilityStatus {
        match req.tier {
            EvalTier::Tier1DefaultZeroShot | EvalTier::Tier3AFewShotK5Pos => {
                CapabilityStatus::Supported
            }
            other => CapabilityStatus::UnsupportedByProvider {
                reason: format!(
                    "Google adaptive LLM supports only Tier1DefaultZeroShot and \
                     Tier3AFewShotK5Pos, not {}",
                    other.label()
                ),
            },
        }
    }

    async fn translate(
        &self,
        req: &TranslationRequest,
    ) -> Result<TelemetryEnvelope<String>> {
        if !self.check_capability(req).is_available() {
            bail!(
                "Google adaptive LLM does not support evaluation tier {}",
                req.tier.label()
            );
        }

        let mut body = json!({
            "contents": [req.source_en],
            "targetLanguageCode": target_language_code(req.lang),
            "sourceLanguageCode": "en",
            "mimeType": "text/plain",
            "model": "general/translation-llm",
        });

        if req.tier == EvalTier::Tier3AFewShotK5Pos {
            let pairs: Vec<Value> = req
                .examples
                .iter()
                .take(5)
                .map(|example| {
                    json!({
                        "sourceSentence": example.source_en,
                        "targetSentence": example.target,
                    })
                })
                .collect();

            if !pairs.is_empty() {
                if let Some(object) = body.as_object_mut() {
                    object.insert(
                        "referenceSentenceConfig".to_owned(),
                        json!({ "referenceSentencePairs": pairs }),
                    );
                }
            }
        }

        let stopwatch = LatencyStopwatch::new();
        let response = self
            .client
            .post(self.endpoint())
            .header("x-goog-api-key", &self.api_key)
            .json(&body)
            .send()
            .await
            .context("Google adaptive LLM request failed")?;

        let payload = parse_google_response(response).await?;
        let total_ms = stopwatch.elapsed_ms();

        let prompt_tokens = Some(estimate_tokens(&req.source_en));
        let completion_tokens = Some(estimate_tokens(&payload));

        Ok(bare_response_envelope(
            payload,
            total_ms,
            prompt_tokens,
            completion_tokens,
            CostRates::zero(),
            0.0,
        ))
    }
}

async fn parse_google_response(response: reqwest::Response) -> Result<String> {
    let status = response.status();
    let value: Value = response
        .json()
        .await
        .context("failed to decode Google translation response")?;

    if !status.is_success() {
        bail!("Google translation API returned {status}: {value}");
    }

    value
        .get("translations")
        .and_then(Value::as_array)
        .and_then(|translations| translations.first())
        .and_then(|translation| translation.get("translatedText"))
        .and_then(Value::as_str)
        .map(str::to_owned)
        .context("Google translation response missing translations[0].translatedText")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{
        CoarseDimensions, ComplianceAnchors, FewShotExample, GranularDimensions, Language,
    };
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn request(tier: EvalTier, lang: Language) -> TranslationRequest {
        TranslationRequest {
            request_id: "req-1".to_owned(),
            tier,
            item_id: "item-1".to_owned(),
            variant_id: "variant-1".to_owned(),
            source_en: "Hello world".to_owned(),
            lang,
            coarse: CoarseDimensions::default(),
            granular: GranularDimensions::default(),
            conflict_override: None,
            examples: Vec::new(),
            gold_reference: String::new(),
            anchors: ComplianceAnchors::default(),
        }
    }

    #[test]
    fn basic_rejects_tier2a() {
        let provider = GoogleBasicNmtProvider::new(DEFAULT_BASE_URL, "project", "key");
        let capability = provider.check_capability(&request(EvalTier::Tier2ACoarseParams, Language::He));
        match capability {
            CapabilityStatus::UnsupportedByProvider { reason } => {
                assert!(reason.contains("Tier2ACoarseParams"));
            }
            other => panic!("expected unsupported, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn basic_translates_and_reports_provider_id() {
        let server = MockServer::start().await;
        let expected_path = "/v3/projects/test-project/locations/global:translateText";

        Mock::given(method("POST"))
            .and(path(expected_path))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(json!({"translations": [{"translatedText": "שלום עולם"}]})),
            )
            .mount(&server)
            .await;

        let provider = GoogleBasicNmtProvider::new(server.uri(), "test-project", "secret");
        assert_eq!(provider.provider_id(), "google-basic-nmt");

        let envelope = provider
            .translate(&request(EvalTier::Tier1DefaultZeroShot, Language::He))
            .await
            .expect("translation should succeed");

        assert_eq!(envelope.payload, "שלום עולם");
        assert!(envelope.total_latency_ms >= 0.0);
        assert!(envelope.prompt_tokens.unwrap_or(0) > 0);
        assert_eq!(envelope.char_count, "שלום עולם".chars().count());

        let requests = server
            .received_requests()
            .await
            .expect("wiremock should record requests");
        assert_eq!(requests.len(), 1);
        assert_eq!(
            requests[0]
                .headers
                .get("x-goog-api-key")
                .and_then(|value| value.to_str().ok()),
            Some("secret")
        );
    }

    #[tokio::test]
    async fn adaptive_tier3a_serializes_five_reference_pairs() {
        let server = MockServer::start().await;
        let expected_path = "/v3/projects/test-project/locations/global:translateText";

        Mock::given(method("POST"))
            .and(path(expected_path))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(json!({"translations": [{"translatedText": "תרגום"}]})),
            )
            .mount(&server)
            .await;

        let provider = GoogleAdaptiveLlmProvider::new(server.uri(), "test-project", "secret");

        let mut req = request(EvalTier::Tier3AFewShotK5Pos, Language::He);
        req.examples = (0..6)
            .map(|index| FewShotExample {
                source_en: format!("source {index}"),
                target: format!("target {index}"),
                negative: None,
            })
            .collect();

        let envelope = provider
            .translate(&req)
            .await
            .expect("adaptive translation should succeed");
        assert_eq!(envelope.payload, "תרגום");

        let requests = server
            .received_requests()
            .await
            .expect("wiremock should record requests");
        assert_eq!(requests.len(), 1);
        let body: Value = serde_json::from_slice(&requests[0].body)
            .expect("request body should be valid JSON");

        assert_eq!(body["model"], "general/translation-llm");
        let pairs = body["referenceSentenceConfig"]["referenceSentencePairs"]
            .as_array()
            .expect("reference pairs should be an array");
        assert_eq!(pairs.len(), 5);
        assert_eq!(pairs[0]["sourceSentence"], "source 0");
        assert_eq!(pairs[0]["targetSentence"], "target 0");
        assert_eq!(pairs[4]["sourceSentence"], "source 4");
    }
}
