//! Azure AI Translator adapters.
//!
//! The standard NMT provider handles only default zero-shot translation. The
//! generative provider additionally maps coarse formality and gender controls
//! where Azure exposes them, and passes inline reference pairs for 3A.

use anyhow::{bail, Context, Result};
use reqwest::Client;
use serde_json::{json, Value};

use super::{
    bare_response_envelope, estimate_tokens, normalize_base_url, target_language_code,
    CostRates, LatencyStopwatch, TranslationProvider,
};
use crate::schema::{
    CapabilityStatus, CoarseDimensions, EvalTier, TelemetryEnvelope, TranslationRequest,
};

/// Azure AI Translator standard NMT provider.
#[derive(Debug, Clone)]
pub struct AzureStandardNmtProvider {
    base_url: String,
    api_key: String,
    region: String,
    client: Client,
}

impl AzureStandardNmtProvider {
    /// Constructs a standard NMT provider from endpoint, key, and region.
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        region: impl Into<String>,
    ) -> Self {
        let base_url: String = base_url.into();
        Self {
            base_url: normalize_base_url(&base_url),
            api_key: api_key.into(),
            region: region.into(),
            client: Client::new(),
        }
    }

    fn endpoint(&self) -> String {
        format!("{}/translate", self.base_url)
    }
}

#[async_trait::async_trait]
impl TranslationProvider for AzureStandardNmtProvider {
    fn provider_id(&self) -> &str {
        "azure-standard-nmt"
    }

    fn check_capability(&self, req: &TranslationRequest) -> CapabilityStatus {
        match req.tier {
            EvalTier::Tier1DefaultZeroShot => CapabilityStatus::Supported,
            other => CapabilityStatus::UnsupportedByProvider {
                reason: format!(
                    "Azure standard NMT supports only Tier1DefaultZeroShot, not {}",
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
                "Azure standard NMT does not support evaluation tier {}",
                req.tier.label()
            );
        }

        let body = json!([{ "text": req.source_en }]);

        let stopwatch = LatencyStopwatch::new();
        let response = self
            .send_translate(&body, req)
            .await
            .context("Azure standard NMT request failed")?;

        let payload = parse_azure_response(response).await?;
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

impl AzureStandardNmtProvider {
    async fn send_translate(
        &self,
        body: &Value,
        req: &TranslationRequest,
    ) -> Result<reqwest::Response> {
        let mut request = self
            .client
            .post(self.endpoint())
            .query(&[
                ("api-version", "3.0"),
                ("to", target_language_code(req.lang)),
                ("from", "en"),
            ])
            .header("Ocp-Apim-Subscription-Key", &self.api_key)
            .json(body);

        if !self.region.trim().is_empty() {
            request = request.header("Ocp-Apim-Subscription-Region", &self.region);
        }

        request.send().await.context("Azure translate POST failed")
    }
}

/// Azure generative translation provider with coarse controls and references.
#[derive(Debug, Clone)]
pub struct AzureGenAiTranslatorProvider {
    base_url: String,
    api_key: String,
    region: String,
    client: Client,
}

impl AzureGenAiTranslatorProvider {
    /// Constructs a generative provider from endpoint, key, and region.
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        region: impl Into<String>,
    ) -> Self {
        let base_url: String = base_url.into();
        Self {
            base_url: normalize_base_url(&base_url),
            api_key: api_key.into(),
            region: region.into(),
            client: Client::new(),
        }
    }

    fn endpoint(&self) -> String {
        format!("{}/translate", self.base_url)
    }

    fn build_body(req: &TranslationRequest) -> Value {
        let mut object = serde_json::Map::new();
        object.insert("text".to_owned(), json!(req.source_en));

        if req.tier == EvalTier::Tier2ACoarseParams {
            let coarse = req.effective_coarse();
            if let Some(tone) = tone_for_formality(&coarse.formality) {
                object.insert("tone".to_owned(), json!(tone));
            }
            if let Some(gender) = gender_for_coarse(coarse) {
                object.insert("gender".to_owned(), json!(gender));
            }
        }

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
                object.insert(
                    "referenceSentenceConfig".to_owned(),
                    json!({ "referenceSentencePairs": pairs }),
                );
            }
        }

        Value::Array(vec![Value::Object(object)])
    }

    async fn send_translate(
        &self,
        body: &Value,
        req: &TranslationRequest,
    ) -> Result<reqwest::Response> {
        let mut request = self
            .client
            .post(self.endpoint())
            .query(&[
                ("api-version", "3.0"),
                ("to", target_language_code(req.lang)),
                ("from", "en"),
            ])
            .header("Ocp-Apim-Subscription-Key", &self.api_key)
            .json(body);

        if !self.region.trim().is_empty() {
            request = request.header("Ocp-Apim-Subscription-Region", &self.region);
        }

        request.send().await.context("Azure translate POST failed")
    }
}

#[async_trait::async_trait]
impl TranslationProvider for AzureGenAiTranslatorProvider {
    fn provider_id(&self) -> &str {
        "azure-genai-translator"
    }

    fn check_capability(&self, req: &TranslationRequest) -> CapabilityStatus {
        match req.tier {
            EvalTier::Tier1DefaultZeroShot => CapabilityStatus::Supported,
            EvalTier::Tier2ACoarseParams => {
                let coarse = req.effective_coarse();
                if coarse
                    .dialect
                    .as_deref()
                    .is_some_and(|dialect| !dialect.trim().is_empty())
                {
                    CapabilityStatus::PartiallySupported {
                        supported_axes: vec![
                            "formality".to_owned(),
                            "speaker_gender".to_owned(),
                            "listener_gender".to_owned(),
                        ],
                        missing_axes: vec!["dialect".to_owned()],
                    }
                } else {
                    CapabilityStatus::Supported
                }
            }
            EvalTier::Tier3AFewShotK5Pos => CapabilityStatus::Supported,
            other => CapabilityStatus::UnsupportedByProvider {
                reason: format!(
                    "Azure generative translator supports only Tier1DefaultZeroShot, \
                     Tier2ACoarseParams, and Tier3AFewShotK5Pos, not {}",
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
                "Azure generative translator does not support evaluation tier {}",
                req.tier.label()
            );
        }

        let body = Self::build_body(req);

        let stopwatch = LatencyStopwatch::new();
        let response = self
            .send_translate(&body, req)
            .await
            .context("Azure generative translator request failed")?;

        let payload = parse_azure_response(response).await?;
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

fn tone_for_formality(formality: &str) -> Option<&'static str> {
    match formality.trim() {
        "level_1_intimate" => Some("informal"),
        "level_2_polite" | "level_3_honorific" => Some("formal"),
        _ => None,
    }
}

fn gender_for_coarse(coarse: &CoarseDimensions) -> Option<&'static str> {
    let raw = coarse
        .listener_gender
        .as_deref()
        .or(coarse.speaker_gender.as_deref())?
        .trim();

    if raw.eq_ignore_ascii_case("male") {
        Some("male")
    } else if raw.eq_ignore_ascii_case("female") {
        Some("female")
    } else {
        None
    }
}

async fn parse_azure_response(response: reqwest::Response) -> Result<String> {
    let status = response.status();
    let value: Value = response
        .json()
        .await
        .context("failed to decode Azure translation response")?;

    if !status.is_success() {
        bail!("Azure translation API returned {status}: {value}");
    }

    value
        .as_array()
        .and_then(|items| items.first())
        .and_then(|item| item.get("translations"))
        .and_then(Value::as_array)
        .and_then(|translations| translations.first())
        .and_then(|translation| translation.get("text"))
        .and_then(Value::as_str)
        .map(str::to_owned)
        .context("Azure translation response missing [0].translations[0].text")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{
        ComplianceAnchors, FewShotExample, GranularDimensions, Language,
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

    fn body_of(requests: &[wiremock::Request]) -> Value {
        serde_json::from_slice(&requests[0].body).expect("request body should be valid JSON")
    }

    #[tokio::test]
    async fn genai_tier2a_sends_tone_and_gender() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/translate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([
                {"translations": [{"text": "שלום"}]}
            ])))
            .mount(&server)
            .await;

        let provider = AzureGenAiTranslatorProvider::new(server.uri(), "key", "");
        let mut req = request(EvalTier::Tier2ACoarseParams, Language::He);
        req.coarse.formality = "level_1_intimate".to_owned();
        req.coarse.speaker_gender = Some("male".to_owned());
        req.coarse.listener_gender = Some("female".to_owned());

        let envelope = provider
            .translate(&req)
            .await
            .expect("translation should succeed");
        assert_eq!(envelope.payload, "שלום");

        let requests = server
            .received_requests()
            .await
            .expect("wiremock should record requests");
        assert_eq!(requests.len(), 1);

        let body = body_of(&requests);
        assert_eq!(body[0]["tone"], "informal");
        assert_eq!(body[0]["gender"], "female");
        assert_eq!(body[0]["text"], "Hello world");

        let query = requests[0].url.query().unwrap_or_default();
        assert!(query.contains("to=he"));
        assert!(query.contains("from=en"));
        assert!(query.contains("api-version=3.0"));
    }

    #[test]
    fn genai_tier2a_dialect_is_partially_supported() {
        let provider = AzureGenAiTranslatorProvider::new("https://example.test", "key", "");
        let mut req = request(EvalTier::Tier2ACoarseParams, Language::Vi);
        req.coarse.dialect = Some("southern".to_owned());

        match provider.check_capability(&req) {
            CapabilityStatus::PartiallySupported {
                supported_axes,
                missing_axes,
            } => {
                assert_eq!(missing_axes, vec!["dialect".to_owned()]);
                assert!(supported_axes.contains(&"formality".to_owned()));
                assert!(supported_axes.contains(&"speaker_gender".to_owned()));
                assert!(supported_axes.contains(&"listener_gender".to_owned()));
            }
            other => panic!("expected partial support, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn genai_tier3a_sends_five_reference_pairs() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/translate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([
                {"translations": [{"text": "תרגום"}]}
            ])))
            .mount(&server)
            .await;

        let provider = AzureGenAiTranslatorProvider::new(server.uri(), "key", "eastus");
        let mut req = request(EvalTier::Tier3AFewShotK5Pos, Language::He);
        req.examples = (0..7)
            .map(|index| FewShotExample {
                source_en: format!("source {index}"),
                target: format!("target {index}"),
                negative: None,
            })
            .collect();

        let envelope = provider
            .translate(&req)
            .await
            .expect("translation should succeed");
        assert_eq!(envelope.payload, "תרגום");

        let requests = server
            .received_requests()
            .await
            .expect("wiremock should record requests");
        assert_eq!(requests.len(), 1);

        let body = body_of(&requests);
        let pairs = body[0]["referenceSentenceConfig"]["referenceSentencePairs"]
            .as_array()
            .expect("reference pairs should be an array");
        assert_eq!(pairs.len(), 5);
        assert_eq!(pairs[0]["sourceSentence"], "source 0");
        assert_eq!(pairs[4]["targetSentence"], "target 4");
        assert_eq!(
            requests[0]
                .headers
                .get("Ocp-Apim-Subscription-Region")
                .and_then(|value| value.to_str().ok()),
            Some("eastus")
        );
    }

    #[test]
    fn standard_nmt_rejects_tier2a() {
        let provider = AzureStandardNmtProvider::new("https://example.test", "key", "");
        let capability =
            provider.check_capability(&request(EvalTier::Tier2ACoarseParams, Language::He));
        match capability {
            CapabilityStatus::UnsupportedByProvider { reason } => {
                assert!(reason.contains("Tier2ACoarseParams"));
            }
            other => panic!("expected unsupported, got {other:?}"),
        }
    }
}
