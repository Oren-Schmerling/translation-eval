//! Layer 3: two-judge ensemble scoring for pragmatic translation quality.
//!
//! A [`JudgeClient`] is any model endpoint that can return the four
//! [`JudgeScores`] axes. [`evaluate_ensemble`] queries two judges concurrently
//! and reports both their mean and any axes where they disagree strongly.

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use serde_json::json;

use crate::schema::{
    CoarseDimensions, EnsembleResult, GranularDimensions, JudgeScores, Language,
};

/// Client capable of scoring a translation on the four pragmatic axes.
#[async_trait]
pub trait JudgeClient: Send + Sync {
    /// Stable identifier used in diagnostics.
    fn judge_id(&self) -> &str;

    /// Scores a request, returning a value in `0..=5` for every axis.
    async fn score(&self, request: &JudgeRequest) -> anyhow::Result<JudgeScores>;
}

/// The material handed to a judge model.
#[derive(Debug, Clone)]
pub struct JudgeRequest {
    pub source_en: String,
    pub translation: String,
    pub lang: Language,
    pub coarse: CoarseDimensions,
    pub granular: GranularDimensions,
}

impl JudgeRequest {
    /// Builds a judge request from its constituent parts.
    pub fn new(
        source_en: impl Into<String>,
        translation: impl Into<String>,
        lang: Language,
        coarse: CoarseDimensions,
        granular: GranularDimensions,
    ) -> Self {
        Self {
            source_en: source_en.into(),
            translation: translation.into(),
            lang,
            coarse,
            granular,
        }
    }
}

const JUDGE_RUBRIC: &str = r#"You are an expert bilingual pragmatic translation evaluator.
Evaluate the supplied English source and its translation on exactly four axes:

- formality_compliance: Does the translation use the requested level of formality/register?
- gender_agreement: Does the translation agree with the requested speaker/listener gender?
- dialect_authenticity: Does the translation use the requested regional dialect naturally?
- naturalness_fluency: Is the translation idiomatic and fluent for a native speaker?

Respond ONLY with a JSON object containing exactly these four keys. Each value
MUST be an integer from 1 (completely violates the requirement) to 5 (fully
complies). Do not add commentary, markdown, or extra keys."#;

/// Judge backed by an OpenAI-compatible chat-completions HTTP endpoint.
pub struct HttpJudgeClient {
    base_url: String,
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl HttpJudgeClient {
    /// Creates a client, trimming any trailing slash from `base_url`.
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        let base_url: String = base_url.into();
        Self {
            base_url: base_url.trim_end_matches('/').to_owned(),
            api_key: api_key.into(),
            model: model.into(),
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl JudgeClient for HttpJudgeClient {
    fn judge_id(&self) -> &str {
        &self.model
    }

    async fn score(&self, request: &JudgeRequest) -> Result<JudgeScores> {
        let url = format!("{}/chat/completions", self.base_url);

        let coarse_json =
            serde_json::to_string(&request.coarse).unwrap_or_else(|_| "{}".to_owned());
        let granular_json =
            serde_json::to_string(&request.granular).unwrap_or_else(|_| "{}".to_owned());
        let user_content = format!(
            "Evaluate the following translation.\n\n\
             Source (English):\n{}\n\n\
             Translation ({} / {}):\n{}\n\n\
             Coarse dimensions (requested):\n{}\n\n\
             Granular dimensions (requested):\n{}\n\n\
             Return the four-key JSON object now.",
            request.source_en,
            request.lang.english_name(),
            request.lang.code(),
            request.translation,
            coarse_json,
            granular_json,
        );

        let body = json!({
            "model": self.model,
            "temperature": 0,
            "response_format": { "type": "json_object" },
            "messages": [
                { "role": "system", "content": JUDGE_RUBRIC },
                { "role": "user", "content": user_content }
            ]
        });

        let mut builder = self.client.post(&url).json(&body);
        if !self.api_key.is_empty() {
            builder = builder.bearer_auth(&self.api_key);
        }

        let response = builder
            .send()
            .await
            .with_context(|| format!("judge request to {url} failed"))?;
        let status = response.status();
        let text = response
            .text()
            .await
            .context("failed to read judge response body")?;

        if !status.is_success() {
            return Err(anyhow!(
                "judge request failed with status {status}: {text}"
            ));
        }

        let value: serde_json::Value = serde_json::from_str(&text)
            .with_context(|| format!("judge response was not valid JSON: {text}"))?;
        let content = value
            .get("choices")
            .and_then(|choices| choices.get(0))
            .and_then(|choice| choice.get("message"))
            .and_then(|message| message.get("content"))
            .and_then(|content| content.as_str())
            .ok_or_else(|| anyhow!("judge response missing choices[0].message.content"))?;

        let stripped = strip_code_fences(content);
        let mut scores: JudgeScores = serde_json::from_str(stripped)
            .with_context(|| format!("judge content was not valid JudgeScores: {stripped}"))?;

        scores.formality_compliance = scores.formality_compliance.clamp(0.0, 5.0);
        scores.gender_agreement = scores.gender_agreement.clamp(0.0, 5.0);
        scores.dialect_authenticity = scores.dialect_authenticity.clamp(0.0, 5.0);
        scores.naturalness_fluency = scores.naturalness_fluency.clamp(0.0, 5.0);

        Ok(scores)
    }
}

fn strip_code_fences(content: &str) -> &str {
    let trimmed = content.trim();
    if let Some(rest) = trimmed.strip_prefix("```") {
        let rest = rest.trim_start();
        let rest = rest.strip_prefix("json").unwrap_or(rest).trim_start();
        rest.strip_suffix("```").unwrap_or(rest).trim()
    } else {
        trimmed
    }
}

/// Runs two judges concurrently and aggregates their scores.
///
/// The mean is computed per axis. An axis is flagged when the two judges
/// differ by at least `2.0`; additionally, a translation that failed Layer 1
/// but still receives a high overall judge mean is treated as suspicious.
pub async fn evaluate_ensemble(
    judge_a: &dyn JudgeClient,
    judge_b: &dyn JudgeClient,
    request: &JudgeRequest,
    layer1_passed: bool,
) -> Result<EnsembleResult> {
    let (result_a, result_b) = tokio::join!(judge_a.score(request), judge_b.score(request));
    let judge_a_scores = result_a.context("judge A failed")?;
    let judge_b_scores = result_b.context("judge B failed")?;

    let mean = JudgeScores {
        formality_compliance: (judge_a_scores.formality_compliance
            + judge_b_scores.formality_compliance)
            / 2.0,
        gender_agreement: (judge_a_scores.gender_agreement + judge_b_scores.gender_agreement) / 2.0,
        dialect_authenticity: (judge_a_scores.dialect_authenticity
            + judge_b_scores.dialect_authenticity)
            / 2.0,
        naturalness_fluency: (judge_a_scores.naturalness_fluency
            + judge_b_scores.naturalness_fluency)
            / 2.0,
    };

    let axes = [
        (
            "formality_compliance",
            judge_a_scores.formality_compliance,
            judge_b_scores.formality_compliance,
        ),
        (
            "gender_agreement",
            judge_a_scores.gender_agreement,
            judge_b_scores.gender_agreement,
        ),
        (
            "dialect_authenticity",
            judge_a_scores.dialect_authenticity,
            judge_b_scores.dialect_authenticity,
        ),
        (
            "naturalness_fluency",
            judge_a_scores.naturalness_fluency,
            judge_b_scores.naturalness_fluency,
        ),
    ];

    let disagreement_axes: Vec<String> = axes
        .iter()
        .filter(|(_, a, b)| (a - b).abs() >= 2.0)
        .map(|(name, _, _)| (*name).to_owned())
        .collect();

    let high_disagreement_flag =
        !disagreement_axes.is_empty() || (!layer1_passed && mean.overall() >= 4.0);

    Ok(EnsembleResult {
        judge_a: judge_a_scores,
        judge_b: judge_b_scores,
        mean,
        high_disagreement_flag,
        disagreement_axes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockJudge {
        id: String,
        scores: JudgeScores,
    }

    impl MockJudge {
        fn new(id: &str, scores: [f32; 4]) -> Self {
            Self {
                id: id.to_owned(),
                scores: JudgeScores {
                    formality_compliance: scores[0],
                    gender_agreement: scores[1],
                    dialect_authenticity: scores[2],
                    naturalness_fluency: scores[3],
                },
            }
        }
    }

    #[async_trait]
    impl JudgeClient for MockJudge {
        fn judge_id(&self) -> &str {
            &self.id
        }

        async fn score(&self, _request: &JudgeRequest) -> Result<JudgeScores> {
            Ok(self.scores)
        }
    }

    fn request() -> JudgeRequest {
        JudgeRequest::new(
            "Hello, how are you?",
            "שלום, מה שלומך?",
            Language::He,
            CoarseDimensions::default(),
            GranularDimensions::default(),
        )
    }

    #[tokio::test]
    async fn disagreement_is_flagged_and_means_are_computed() {
        let judge_a = MockJudge::new("a", [5.0, 5.0, 5.0, 5.0]);
        let judge_b = MockJudge::new("b", [2.0, 5.0, 4.0, 4.0]);

        let result = evaluate_ensemble(&judge_a, &judge_b, &request(), true)
            .await
            .expect("ensemble should succeed");

        assert_eq!(result.mean.formality_compliance, 3.5);
        assert_eq!(result.mean.gender_agreement, 5.0);
        assert!(result.high_disagreement_flag);
        assert!(result
            .disagreement_axes
            .iter()
            .any(|axis| axis == "formality_compliance"));
    }

    #[tokio::test]
    async fn matching_judges_do_not_flag_disagreement() {
        let judge_a = MockJudge::new("a", [4.0, 4.0, 4.0, 4.0]);
        let judge_b = MockJudge::new("b", [4.0, 4.0, 4.0, 4.0]);

        let result = evaluate_ensemble(&judge_a, &judge_b, &request(), true)
            .await
            .expect("ensemble should succeed");

        assert!(!result.high_disagreement_flag);
        assert!(result.disagreement_axes.is_empty());
    }
}
