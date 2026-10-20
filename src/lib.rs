//! Pragmatic machine translation evaluation harness.
//!
//! `pragmatic_mt_eval` measures whether machine translation systems honor
//! pragmatic constraints (formality, speaker/listener gender, dialect, and
//! fine-grained pronoun/kinship choices) in addition to producing fluent,
//! accurate output.
//!
//! The crate is organised as a pipeline:
//!
//! * [`schema`] — shared, strongly typed data model for datasets, requests,
//!   provider responses, and graded run records.
//! * [`compiler`] — turns curation CSV files into a compiled [`schema::EvalDataset`],
//!   inferring lexical compliance anchors from sibling gold variants.
//! * [`providers`] — adapters for Google Cloud, Azure, DeepL, Amazon Translate,
//!   and streaming LLM endpoints.
//! * [`grader`] — deterministic anchor checks (Layer 1), chrF++ scoring
//!   (Layer 2), a two-judge ensemble (Layer 3), and Markdown/CSV reporting.
//!
//! The crate performs no network I/O at import or construction time, which
//! keeps the test suite fully offline.

pub mod compiler;
pub mod grader;
pub mod providers;
pub mod schema;

pub use compiler::{
    compile_dataset, compile_dataset_with_resources, ClosedClassResources, LangResources,
};
pub use grader::{
    best_chrf, chrf, chrf_plus_plus, classify_default_bias, evaluate_anchors, evaluate_ensemble,
    render_markdown, write_csv, write_markdown, HttpJudgeClient, JudgeClient, JudgeRequest,
};
pub use providers::{
    AmazonTranslateProvider, AzureGenAiTranslatorProvider, AzureStandardNmtProvider,
    DeepLProvider, GoogleAdaptiveLlmProvider, GoogleBasicNmtProvider, LlmFamily,
    StreamingLlmConfig, StreamingLlmProvider, TranslationProvider,
};
pub use schema::*;
