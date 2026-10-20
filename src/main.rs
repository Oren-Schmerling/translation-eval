//! Command-line interface for the pragmatic MT evaluation harness.
//!
//! Subcommands mirror the four pipeline stages: `compile`, `run`, `grade`,
//! and `report`. Every command is safe to invoke with missing optional inputs;
//! required files produce a contextual error rather than a panic.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};

use pragmatic_mt_eval::compiler::compile_dataset;
use pragmatic_mt_eval::grader::{
    best_chrf, classify_default_bias, evaluate_anchors, evaluate_ensemble, write_csv,
    write_markdown, HttpJudgeClient, JudgeRequest,
};
use pragmatic_mt_eval::providers::amazon::{AmazonTranslateProvider, DEFAULT_AMAZON_BASE_URL};
use pragmatic_mt_eval::providers::deepl::{DeepLProvider, DEFAULT_DEEPL_BASE_URL};
use pragmatic_mt_eval::providers::llm_router::{
    LlmFamily, StreamingLlmConfig, StreamingLlmProvider,
};
use pragmatic_mt_eval::providers::{
    AzureGenAiTranslatorProvider, AzureStandardNmtProvider, GoogleAdaptiveLlmProvider,
    GoogleBasicNmtProvider, TranslationProvider,
};
use pragmatic_mt_eval::schema::{
    EvalDataset, EvalTier, Language, RunRecord, TelemetryEnvelope, TranslationRequest,
};

const DEFAULT_GOOGLE_BASE_URL: &str = "https://translation.googleapis.com";
const DEFAULT_AZURE_BASE_URL: &str = "https://api.cognitive.microsofttranslator.com";
const DEFAULT_LLM_BASE_URL: &str = "https://api.openai.com/v1";
const ALL_TIERS: [EvalTier; 7] = [
    EvalTier::Tier1DefaultZeroShot,
    EvalTier::Tier2ACoarseParams,
    EvalTier::Tier2BGranularParams,
    EvalTier::Tier2CAdversarialConflict,
    EvalTier::Tier3AFewShotK5Pos,
    EvalTier::Tier3BFewShotK20Pos,
    EvalTier::Tier3CFewShotK5Contrastive,
];

/// Pragmatic MT evaluation harness.
#[derive(Debug, Parser)]
#[command(
    name = "pragmatic-mt-eval",
    version,
    about = "Compile, run, grade, and report pragmatic machine translation evaluations."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Compile curation CSV files into a single evaluation dataset.
    Compile {
        /// Directory containing `*_curation.csv` files.
        #[arg(long, default_value = "data/templates")]
        input_dir: PathBuf,
        /// Destination JSON dataset.
        #[arg(long, default_value = "data/compiled/eval_dataset.json")]
        output: PathBuf,
        /// Closed-class and synset resource bundle.
        #[arg(long, default_value = "data/resources/closed_class.json")]
        resources: PathBuf,
    },
    /// Call a translation provider for each selected variant and tier.
    Run {
        /// Compiled evaluation dataset.
        #[arg(long, default_value = "data/compiled/eval_dataset.json")]
        dataset: PathBuf,
        /// Translation provider to exercise.
        #[arg(long, value_enum)]
        provider: ProviderKind,
        /// Comma-separated tiers (`all`, `1`, `2a`, `3b`, ...).
        #[arg(long, default_value = "all")]
        tiers: String,
        /// Destination JSONL file.
        #[arg(long, default_value = "results/results.jsonl")]
        output: PathBuf,
        /// Override the provider endpoint; production defaults are used when omitted.
        #[arg(long)]
        base_url: Option<String>,
        /// Provider API key. Falls back to the provider's standard environment variable.
        #[arg(long)]
        api_key: Option<String>,
        /// Google Cloud project id (Google providers).
        #[arg(long)]
        project_id: Option<String>,
        /// Azure region (Azure providers).
        #[arg(long)]
        region: Option<String>,
        /// Model name (LLM providers).
        #[arg(long)]
        model: Option<String>,
        /// Wire format family for `--provider llm`.
        #[arg(long, value_enum)]
        model_family: Option<ModelFamily>,
        /// Comma-separated language filter (`he`, `gu`, `vi`).
        #[arg(long)]
        language: Option<String>,
        /// Maximum number of variants per language and tier.
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Grade a run JSONL with anchors, chrF++, bias classification, and judges.
    Grade {
        /// Run records produced by `run`.
        #[arg(long, default_value = "results/results.jsonl")]
        input: PathBuf,
        /// Destination graded JSONL file.
        #[arg(long, default_value = "results/graded.jsonl")]
        output: PathBuf,
        /// Compiled evaluation dataset used for sibling lookup.
        #[arg(long, default_value = "data/compiled/eval_dataset.json")]
        dataset: PathBuf,
        /// Optional JSON file configuring two OpenAI-compatible judges.
        #[arg(long)]
        judges_config: Option<PathBuf>,
    },
    /// Render Markdown and CSV reports from graded records.
    Report {
        /// Graded JSONL file.
        #[arg(long, default_value = "results/graded.jsonl")]
        input: PathBuf,
        /// Destination Markdown report.
        #[arg(long, default_value = "results/report.md")]
        markdown: PathBuf,
        /// Destination CSV summary.
        #[arg(long, default_value = "results/summary.csv")]
        csv: PathBuf,
    },
}

/// Supported translation providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ProviderKind {
    GoogleNmt,
    GoogleAdaptive,
    AzureNmt,
    AzureGenai,
    Deepl,
    Amazon,
    Llm,
}

/// Supported LLM wire formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ModelFamily {
    Openai,
    Anthropic,
    Vertex,
    Cohere,
}

impl ModelFamily {
    fn into_llm_family(self) -> LlmFamily {
        match self {
            Self::Openai => LlmFamily::OpenAi,
            Self::Anthropic => LlmFamily::Anthropic,
            Self::Vertex => LlmFamily::Vertex,
            Self::Cohere => LlmFamily::Cohere,
        }
    }

    fn default_model(self) -> &'static str {
        match self {
            Self::Openai => "gpt-4o-mini",
            Self::Anthropic => "claude-3-5-sonnet-latest",
            Self::Vertex => "gemini-1.5-flash",
            Self::Cohere => "command-r-plus",
        }
    }

    fn api_key_env(self) -> &'static str {
        match self {
            Self::Openai => "OPENAI_API_KEY",
            Self::Anthropic => "ANTHROPIC_API_KEY",
            Self::Vertex => "GOOGLE_API_KEY",
            Self::Cohere => "COHERE_API_KEY",
        }
    }
}

impl ProviderKind {
    fn api_key_env(self) -> &'static str {
        match self {
            Self::GoogleNmt | Self::GoogleAdaptive => "GOOGLE_API_KEY",
            Self::AzureNmt | Self::AzureGenai => "AZURE_TRANSLATOR_KEY",
            Self::Deepl => "DEEPL_API_KEY",
            Self::Amazon => "AWS_ACCESS_KEY_ID",
            Self::Llm => "OPENAI_API_KEY",
        }
    }
}

fn env_value(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn resolve_api_key(
    provider: ProviderKind,
    explicit: Option<String>,
    family: ModelFamily,
) -> String {
    if let Some(key) = explicit {
        return key;
    }
    let env_name = if provider == ProviderKind::Llm {
        family.api_key_env()
    } else {
        provider.api_key_env()
    };
    env_value(env_name).unwrap_or_default()
}

fn resolve_base_url(provider: ProviderKind, explicit: Option<String>) -> String {
    if let Some(url) = explicit {
        return url;
    }
    match provider {
        ProviderKind::GoogleNmt | ProviderKind::GoogleAdaptive => {
            DEFAULT_GOOGLE_BASE_URL.to_owned()
        }
        ProviderKind::AzureNmt | ProviderKind::AzureGenai => DEFAULT_AZURE_BASE_URL.to_owned(),
        ProviderKind::Deepl => DEFAULT_DEEPL_BASE_URL.to_owned(),
        ProviderKind::Amazon => DEFAULT_AMAZON_BASE_URL.to_owned(),
        ProviderKind::Llm => DEFAULT_LLM_BASE_URL.to_owned(),
    }
}

fn build_provider(options: &RunOptions, family: ModelFamily) -> Box<dyn TranslationProvider> {
    let base_url = resolve_base_url(options.provider, options.base_url.clone());
    let api_key = resolve_api_key(options.provider, options.api_key.clone(), family);

    match options.provider {
        ProviderKind::GoogleNmt => {
            let project_id = options
                .project_id
                .clone()
                .or_else(|| env_value("GCP_PROJECT_ID"))
                .unwrap_or_default();
            Box::new(GoogleBasicNmtProvider::new(
                base_url, project_id, api_key,
            ))
        }
        ProviderKind::GoogleAdaptive => {
            let project_id = options
                .project_id
                .clone()
                .or_else(|| env_value("GCP_PROJECT_ID"))
                .unwrap_or_default();
            Box::new(GoogleAdaptiveLlmProvider::new(
                base_url, project_id, api_key,
            ))
        }
        ProviderKind::AzureNmt => {
            let region = options
                .region
                .clone()
                .or_else(|| env_value("AZURE_TRANSLATOR_REGION"))
                .unwrap_or_default();
            Box::new(AzureStandardNmtProvider::new(
                base_url, api_key, region,
            ))
        }
        ProviderKind::AzureGenai => {
            let region = options
                .region
                .clone()
                .or_else(|| env_value("AZURE_TRANSLATOR_REGION"))
                .unwrap_or_default();
            Box::new(AzureGenAiTranslatorProvider::new(
                base_url, api_key, region,
            ))
        }
        ProviderKind::Deepl => Box::new(DeepLProvider::new(base_url, api_key)),
        ProviderKind::Amazon => Box::new(AmazonTranslateProvider::new(base_url, api_key)),
        ProviderKind::Llm => {
            let model = options
                .model
                .clone()
                .unwrap_or_else(|| family.default_model().to_owned());
            let config = StreamingLlmConfig::new(base_url, api_key, model)
                .with_family(family.into_llm_family());
            Box::new(StreamingLlmProvider::new(config))
        }
    }
}

/// Options shared by the provider construction path.
struct RunOptions {
    provider: ProviderKind,
    base_url: Option<String>,
    api_key: Option<String>,
    project_id: Option<String>,
    region: Option<String>,
    model: Option<String>,
}

fn parse_tiers(raw: &str) -> Result<Vec<EvalTier>> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("all") {
        return Ok(ALL_TIERS.to_vec());
    }

    let mut tiers = Vec::new();
    for part in trimmed.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let tier: EvalTier = part
            .parse()
            .with_context(|| format!("invalid tier {part:?} in --tiers"))?;
        if !tiers.contains(&tier) {
            tiers.push(tier);
        }
    }

    if tiers.is_empty() {
        bail!("--tiers did not contain any tier");
    }
    Ok(tiers)
}

fn parse_languages(raw: &str) -> Result<Vec<Language>> {
    let mut languages = Vec::new();
    for part in raw.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let language: Language = part
            .parse()
            .with_context(|| format!("invalid language {part:?} in --language"))?;
        if !languages.contains(&language) {
            languages.push(language);
        }
    }
    if languages.is_empty() {
        bail!("--language did not contain any language");
    }
    Ok(languages)
}

fn examples_for(dataset: &EvalDataset, tier: EvalTier, variant: &pragmatic_mt_eval::schema::TargetVariant) -> Vec<pragmatic_mt_eval::schema::FewShotExample> {
    match tier {
        EvalTier::Tier3AFewShotK5Pos => dataset.few_shot_examples(variant, 5),
        EvalTier::Tier3BFewShotK20Pos => dataset.few_shot_examples(variant, 20),
        EvalTier::Tier3CFewShotK5Contrastive => dataset.contrastive_examples(variant, 5),
        _ => Vec::new(),
    }
}

fn build_request(
    dataset: &EvalDataset,
    provider_id: &str,
    tier: EvalTier,
    variant: &pragmatic_mt_eval::schema::TargetVariant,
) -> TranslationRequest {
    let conflict_override = if tier == EvalTier::Tier2CAdversarialConflict {
        variant.conflict_override.clone()
    } else {
        None
    };

    TranslationRequest {
        request_id: format!("{provider_id}-{}-{}", tier.label(), variant.variant_id),
        tier,
        item_id: variant.item_id.clone(),
        variant_id: variant.variant_id.clone(),
        source_en: variant.source_en.clone(),
        lang: variant.lang,
        coarse: variant.coarse.clone(),
        granular: variant.granular.clone(),
        conflict_override,
        examples: examples_for(dataset, tier, variant),
        gold_reference: variant.gold_reference.clone(),
        anchors: variant.anchors.clone(),
    }
}

fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create output directory {}", parent.display())
            })?;
        }
    }
    Ok(())
}

fn open_jsonl_append(path: &Path) -> Result<BufWriter<File>> {
    ensure_parent(path)?;
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open JSONL output {}", path.display()))?;
    Ok(BufWriter::new(file))
}

fn create_jsonl(path: &Path) -> Result<BufWriter<File>> {
    ensure_parent(path)?;
    let file = File::create(path)
        .with_context(|| format!("failed to create JSONL output {}", path.display()))?;
    Ok(BufWriter::new(file))
}

fn write_record(writer: &mut BufWriter<File>, record: &RunRecord) -> Result<()> {
    let line = serde_json::to_string(record).context("failed to encode run record")?;
    writeln!(writer, "{line}").context("failed to write run record")
}

fn read_records(path: &Path) -> Result<Vec<RunRecord>> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read records from {}", path.display()))?;

    let mut records = Vec::new();
    for (index, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let record: RunRecord = serde_json::from_str(line).with_context(|| {
            format!(
                "failed to parse record on line {} of {}",
                index + 1,
                path.display()
            )
        })?;
        records.push(record);
    }
    Ok(records)
}

fn run_compile(input_dir: &Path, output: &Path, resources: &Path) -> Result<()> {
    let dataset = compile_dataset(input_dir, resources)?;
    dataset.save(output)?;

    let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    for variant in &dataset.variants {
        *counts.entry(variant.lang.code()).or_insert(0) += 1;
    }

    println!(
        "Compiled {} variants into {}",
        dataset.variants.len(),
        output.display()
    );
    for (language, count) in counts {
        println!("  {language}: {count}");
    }
    Ok(())
}

async fn run_provider(options: RunOptions, args: &RunArgs) -> Result<()> {
    let dataset = EvalDataset::load(&args.dataset)
        .with_context(|| format!("while loading {}", args.dataset.display()))?;
    let tiers = parse_tiers(&args.tiers)?;
    let language_filter = args
        .language
        .as_deref()
        .map(parse_languages)
        .transpose()?;

    let family = args.model_family.unwrap_or(ModelFamily::Openai);
    let provider = build_provider(&options, family);
    let provider_id = provider.provider_id().to_owned();

    let mut languages: Vec<Language> = Vec::new();
    for variant in &dataset.variants {
        if !languages.contains(&variant.lang) {
            languages.push(variant.lang);
        }
    }
    if let Some(filter) = language_filter.as_ref() {
        languages.retain(|language| filter.contains(language));
    }

    let mut writer = open_jsonl_append(&args.output)?;

    let mut written = 0usize;
    let mut errors = 0usize;
    let mut unsupported = 0usize;
    let mut per_tier: BTreeMap<&'static str, usize> = BTreeMap::new();

    for language in languages {
        for &tier in &tiers {
            let variants: Vec<&pragmatic_mt_eval::schema::TargetVariant> = dataset
                .variants
                .iter()
                .filter(|variant| variant.lang == language)
                .take(args.limit.unwrap_or(usize::MAX))
                .collect();

            for variant in variants {
                let request = build_request(&dataset, &provider_id, tier, variant);
                let capability = provider.check_capability(&request);

                if !capability.is_available() {
                    let envelope = TelemetryEnvelope::new(
                        String::new(),
                        None,
                        0.0,
                        None,
                        None,
                        0,
                        0.0,
                        0.0,
                    );
                    let mut record = RunRecord::from_translation(
                        &request,
                        provider_id.clone(),
                        &envelope,
                        capability,
                    );
                    record.track = variant.track.clone();
                    record.bucket = variant.bucket.clone();
                    record.is_cued_source = variant.is_cued_source;
                    write_record(&mut writer, &record)?;
                    written += 1;
                    unsupported += 1;
                    *per_tier.entry(tier.label()).or_insert(0) += 1;
                    continue;
                }

                match provider.translate(&request).await {
                    Ok(envelope) => {
                        let mut record = RunRecord::from_translation(
                            &request,
                            provider_id.clone(),
                            &envelope,
                            capability,
                        );
                        record.track = variant.track.clone();
                        record.bucket = variant.bucket.clone();
                        record.is_cued_source = variant.is_cued_source;
                        write_record(&mut writer, &record)?;
                        written += 1;
                        *per_tier.entry(tier.label()).or_insert(0) += 1;
                    }
                    Err(error) => {
                        eprintln!(
                            "warning: request {} failed: {error:#}",
                            request.request_id
                        );
                        errors += 1;
                    }
                }
            }
        }
    }

    writer
        .flush()
        .with_context(|| format!("failed to flush {}", args.output.display()))?;

    println!(
        "Wrote {written} records to {} ({errors} request errors, {unsupported} capability-only)",
        args.output.display()
    );
    for (tier, count) in per_tier {
        println!("  {tier}: {count}");
    }
    Ok(())
}

#[derive(Debug, serde::Deserialize)]
struct JudgeConfigFile {
    judge_a: JudgeConfig,
    judge_b: JudgeConfig,
}

#[derive(Debug, serde::Deserialize)]
struct JudgeConfig {
    base_url: String,
    #[serde(default)]
    api_key: String,
    model: String,
}

fn load_judges(path: &Path) -> Result<(HttpJudgeClient, HttpJudgeClient)> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read judges config {}", path.display()))?;
    let config: JudgeConfigFile = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse judges config {}", path.display()))?;

    let judge_a = HttpJudgeClient::new(
        config.judge_a.base_url,
        config.judge_a.api_key,
        config.judge_a.model,
    );
    let judge_b = HttpJudgeClient::new(
        config.judge_b.base_url,
        config.judge_b.api_key,
        config.judge_b.model,
    );
    Ok((judge_a, judge_b))
}

async fn run_grade(args: &GradeArgs) -> Result<()> {
    let dataset = EvalDataset::load(&args.dataset)
        .with_context(|| format!("while loading {}", args.dataset.display()))?;
    let mut records = read_records(&args.input)?;

    let judges = args
        .judges_config
        .as_deref()
        .map(load_judges)
        .transpose()?;

    let mut judge_errors = 0usize;
    for record in records.iter_mut() {
        let anchors = evaluate_anchors(&record.translation, &record.anchors);
        record.layer1 = Some(anchors.clone());
        record.chrf = Some(best_chrf(
            &record.translation,
            &record.gold_reference,
            &record.anchors.acceptable_synonyms,
        ));

        let siblings: Vec<pragmatic_mt_eval::schema::TargetVariant> = dataset
            .siblings(&record.item_id, &record.variant_id)
            .into_iter()
            .filter(|variant| variant.lang == record.lang)
            .cloned()
            .collect();
        record.bias_profile = Some(classify_default_bias(&record.translation, &siblings));

        record.ensemble = None;
        record.high_disagreement_flag = false;

        if let Some((judge_a, judge_b)) = judges.as_ref() {
            let request = JudgeRequest::new(
                record.source_en.clone(),
                record.translation.clone(),
                record.lang,
                record.coarse.clone(),
                record.granular.clone(),
            );
            match evaluate_ensemble(judge_a, judge_b, &request, anchors.passed).await {
                Ok(ensemble) => {
                    record.high_disagreement_flag = ensemble.high_disagreement_flag;
                    record.ensemble = Some(ensemble);
                }
                Err(error) => {
                    eprintln!(
                        "warning: judge ensemble failed for {}: {error:#}",
                        record.request_id
                    );
                    judge_errors += 1;
                }
            }
        }
    }

    let mut writer = create_jsonl(&args.output)?;
    for record in &records {
        write_record(&mut writer, record)?;
    }
    writer
        .flush()
        .with_context(|| format!("failed to flush {}", args.output.display()))?;

    println!(
        "Graded {} records into {} ({judge_errors} judge errors)",
        records.len(),
        args.output.display()
    );
    Ok(())
}

fn run_report(args: &ReportArgs) -> Result<()> {
    let records = read_records(&args.input)?;
    write_markdown(&records, &args.markdown)?;
    write_csv(&records, &args.csv)?;

    println!("Wrote Markdown report to {}", args.markdown.display());
    println!("Wrote CSV summary to {}", args.csv.display());
    Ok(())
}

#[derive(Debug, Parser)]
struct RunArgs {
    #[arg(long, default_value = "data/compiled/eval_dataset.json")]
    dataset: PathBuf,
    #[arg(long, value_enum)]
    provider: ProviderKind,
    #[arg(long, default_value = "all")]
    tiers: String,
    #[arg(long, default_value = "results/results.jsonl")]
    output: PathBuf,
    #[arg(long)]
    base_url: Option<String>,
    #[arg(long)]
    api_key: Option<String>,
    #[arg(long)]
    project_id: Option<String>,
    #[arg(long)]
    region: Option<String>,
    #[arg(long)]
    model: Option<String>,
    #[arg(long, value_enum)]
    model_family: Option<ModelFamily>,
    #[arg(long)]
    language: Option<String>,
    #[arg(long)]
    limit: Option<usize>,
}

#[derive(Debug, Parser)]
struct GradeArgs {
    #[arg(long, default_value = "results/results.jsonl")]
    input: PathBuf,
    #[arg(long, default_value = "results/graded.jsonl")]
    output: PathBuf,
    #[arg(long, default_value = "data/compiled/eval_dataset.json")]
    dataset: PathBuf,
    #[arg(long)]
    judges_config: Option<PathBuf>,
}

#[derive(Debug, Parser)]
struct ReportArgs {
    #[arg(long, default_value = "results/graded.jsonl")]
    input: PathBuf,
    #[arg(long, default_value = "results/report.md")]
    markdown: PathBuf,
    #[arg(long, default_value = "results/summary.csv")]
    csv: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Compile {
            input_dir,
            output,
            resources,
        } => run_compile(&input_dir, &output, &resources),
        Command::Run {
            dataset,
            provider,
            tiers,
            output,
            base_url,
            api_key,
            project_id,
            region,
            model,
            model_family,
            language,
            limit,
        } => {
            let args = RunArgs {
                dataset,
                provider,
                tiers,
                output,
                base_url,
                api_key,
                project_id,
                region,
                model,
                model_family,
                language,
                limit,
            };
            let options = RunOptions {
                provider: args.provider,
                base_url: args.base_url.clone(),
                api_key: args.api_key.clone(),
                project_id: args.project_id.clone(),
                region: args.region.clone(),
                model: args.model.clone(),
            };
            run_provider(options, &args).await
        }
        Command::Grade {
            input,
            output,
            dataset,
            judges_config,
        } => {
            let args = GradeArgs {
                input,
                output,
                dataset,
                judges_config,
            };
            run_grade(&args).await
        }
        Command::Report {
            input,
            markdown,
            csv,
        } => {
            let args = ReportArgs {
                input,
                markdown,
                csv,
            };
            run_report(&args)
        }
    }
}
