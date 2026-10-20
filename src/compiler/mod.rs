//! Dataset compilation from curation CSV files into the evaluation schema.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::schema::{
    ComplianceAnchors, CoarseDimensions, EvalDataset, GranularDimensions, Language, TargetVariant,
};

pub mod autodiff;
pub use autodiff::{
    morphological_shared_stem, normalize_nfc, populate_anchors, strip_punctuation, tokenize,
    ClosedClassResources, LangResources,
};

/// One row of a `*_curation.csv` file.
///
/// The base columns are required. Pragmatic and conflict columns that may be
/// absent from older curation files are defaulted to empty strings.
#[derive(Debug, Deserialize)]
struct CurationRow {
    item_id: String,
    variant_id: String,
    source_en: String,
    lang: String,
    track: String,
    bucket: String,
    is_cued_source: String,
    formality: String,
    #[serde(default)]
    speaker_gender: String,
    #[serde(default)]
    listener_gender: String,
    #[serde(default)]
    dialect: String,
    #[serde(default)]
    pronoun_tier: String,
    #[serde(default)]
    kinship_pair: String,
    #[serde(default)]
    relationship: String,
    #[serde(default)]
    conflict_formality: String,
    #[serde(default)]
    conflict_speaker_gender: String,
    #[serde(default)]
    conflict_listener_gender: String,
    #[serde(default)]
    conflict_dialect: String,
    gold_reference: String,
}

fn parse_bool(value: &str) -> Result<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" => Ok(true),
        "false" | "0" | "no" => Ok(false),
        _ => bail!("invalid boolean value {value:?}"),
    }
}

fn empty_to_none(value: String) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

fn build_variant(row: CurationRow) -> Result<TargetVariant> {
    let CurationRow {
        item_id,
        variant_id,
        source_en,
        lang,
        track,
        bucket,
        is_cued_source,
        formality,
        speaker_gender,
        listener_gender,
        dialect,
        pronoun_tier,
        kinship_pair,
        relationship,
        conflict_formality,
        conflict_speaker_gender,
        conflict_listener_gender,
        conflict_dialect,
        gold_reference,
    } = row;

    let lang: Language = lang
        .parse()
        .with_context(|| format!("invalid language {lang:?}"))?;
    let is_cued_source = parse_bool(&is_cued_source)
        .with_context(|| format!("invalid is_cued_source {is_cued_source:?}"))?;

    let coarse = CoarseDimensions {
        formality: formality.trim().to_owned(),
        speaker_gender: empty_to_none(speaker_gender),
        listener_gender: empty_to_none(listener_gender),
        dialect: empty_to_none(dialect),
    };
    let granular = GranularDimensions {
        pronoun_tier: empty_to_none(pronoun_tier),
        kinship_pair: empty_to_none(kinship_pair),
        relationship: empty_to_none(relationship),
    };

    let conflict_formality = conflict_formality.trim().to_owned();
    let conflict_speaker_gender = empty_to_none(conflict_speaker_gender);
    let conflict_listener_gender = empty_to_none(conflict_listener_gender);
    let conflict_dialect = empty_to_none(conflict_dialect);
    let has_conflict = !conflict_formality.is_empty()
        || conflict_speaker_gender.is_some()
        || conflict_listener_gender.is_some()
        || conflict_dialect.is_some();
    let conflict_override = has_conflict.then(|| CoarseDimensions {
        formality: conflict_formality,
        speaker_gender: conflict_speaker_gender,
        listener_gender: conflict_listener_gender,
        dialect: conflict_dialect,
    });

    Ok(TargetVariant {
        variant_id: variant_id.trim().to_owned(),
        item_id: item_id.trim().to_owned(),
        source_en: source_en.trim().to_owned(),
        lang,
        track: track.trim().to_owned(),
        bucket: bucket.trim().to_owned(),
        is_cued_source,
        coarse,
        granular,
        conflict_override,
        gold_reference: gold_reference.trim().to_owned(),
        anchors: ComplianceAnchors::default(),
    })
}

fn parse_curation_file(path: &Path) -> Result<Vec<TargetVariant>> {
    let mut reader = csv::ReaderBuilder::new()
        .trim(csv::Trim::All)
        .flexible(true)
        .from_path(path)
        .with_context(|| format!("failed to open curation CSV {}", path.display()))?;

    let mut variants = Vec::new();
    for (row_index, record) in reader.deserialize::<CurationRow>().enumerate() {
        let row = record.with_context(|| {
            format!(
                "failed to parse row {} of {}",
                row_index + 2,
                path.display()
            )
        })?;
        let variant = build_variant(row).with_context(|| {
            format!(
                "failed to build variant from row {} of {}",
                row_index + 2,
                path.display()
            )
        })?;
        variants.push(variant);
    }

    Ok(variants)
}

/// Compiles every `*_curation.csv` in `input_dir` using resources from disk.
pub fn compile_dataset(input_dir: &Path, resources_path: &Path) -> Result<EvalDataset> {
    let resources = ClosedClassResources::load(resources_path)?;
    compile_dataset_with_resources(input_dir, &resources)
}

/// Compiles every `*_curation.csv` in `input_dir` using already-loaded resources.
pub fn compile_dataset_with_resources(
    input_dir: &Path,
    resources: &ClosedClassResources,
) -> Result<EvalDataset> {
    let entries = std::fs::read_dir(input_dir).with_context(|| {
        format!(
            "failed to read curation input directory {}",
            input_dir.display()
        )
    })?;

    let mut paths: Vec<PathBuf> = Vec::new();
    for entry in entries {
        let entry = entry.with_context(|| {
            format!(
                "failed to read an entry in curation input directory {}",
                input_dir.display()
            )
        })?;
        let file_type = entry.file_type().with_context(|| {
            format!(
                "failed to inspect curation entry {}",
                entry.path().display()
            )
        })?;
        if !file_type.is_file() {
            continue;
        }
        let file_name = entry.file_name();
        let Some(file_name) = file_name.to_str() else {
            continue;
        };
        if file_name.ends_with("_curation.csv") {
            paths.push(entry.path());
        }
    }
    paths.sort();

    let mut variants = Vec::new();
    let mut seen_variant_ids = HashSet::new();
    for path in paths {
        for variant in parse_curation_file(&path)? {
            if !seen_variant_ids.insert(variant.variant_id.clone()) {
                bail!(
                    "duplicate variant_id {:?} while compiling {}",
                    variant.variant_id,
                    path.display()
                );
            }
            variants.push(variant);
        }
    }

    populate_anchors(&mut variants, resources);
    variants.sort_by(|left, right| {
        (left.lang.code(), &left.item_id, &left.variant_id).cmp(&(
            right.lang.code(),
            &right.item_id,
            &right.variant_id,
        ))
    });

    Ok(EvalDataset { variants })
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use super::*;

    const HEADER: &str = "item_id,variant_id,source_en,lang,track,bucket,is_cued_source,formality,speaker_gender,listener_gender,dialect,pronoun_tier,kinship_pair,relationship,conflict_formality,conflict_speaker_gender,conflict_listener_gender,conflict_dialect,gold_reference";

    fn resources_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("data")
            .join("resources")
            .join("closed_class.json")
    }

    fn csv_row(variant_id: &str, formality: &str, gold: &str) -> String {
        [
            "gu-1",
            variant_id,
            "Put the spoon in the small bowl",
            "gu",
            "core",
            "test",
            "false",
            formality,
            "male",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            gold,
        ]
        .join(",")
    }

    #[test]
    fn compiles_small_gujarati_dataset() {
        let dir = tempfile::tempdir().expect("tempdir");
        let contents = format!(
            "{HEADER}\n{}\n{}\n",
            csv_row("gu-1-a", "level_1_intimate", "તેં નાના વાટકામાં ચમચી મૂકી?"),
            csv_row("gu-1-b", "level_2_polite", "તમે નાના વાટકામાં ચમચી મૂકી?"),
        );
        fs::write(dir.path().join("sample_curation.csv"), contents).expect("write CSV");

        let dataset = compile_dataset(dir.path(), &resources_path()).expect("compile dataset");
        assert_eq!(dataset.variants.len(), 2);
        assert!(dataset
            .variants
            .iter()
            .any(|variant| !variant.anchors.must_contain_any.is_empty()));
    }

    #[test]
    fn rejects_duplicate_variant_ids() {
        let dir = tempfile::tempdir().expect("tempdir");
        let contents = format!(
            "{HEADER}\n{}\n{}\n",
            csv_row("gu-1-a", "level_1_intimate", "તેં નાના વાટકામાં ચમચી મૂકી?"),
            csv_row("gu-1-a", "level_2_polite", "તમે નાના વાટકામાં ચમચી મૂકી?"),
        );
        fs::write(dir.path().join("duplicate_curation.csv"), contents).expect("write CSV");

        let result = compile_dataset(dir.path(), &resources_path());
        assert!(result.is_err());
    }

    #[test]
    fn rejects_unknown_language() {
        let dir = tempfile::tempdir().expect("tempdir");
        let row = csv_row("gu-1-a", "level_1_intimate", "તેં નાના વાટકામાં ચમચી મૂકી?")
            .replace(",gu,", ",zz,");
        let contents = format!("{HEADER}\n{row}\n");
        fs::write(dir.path().join("bad_language_curation.csv"), contents).expect("write CSV");

        let result = compile_dataset(dir.path(), &resources_path());
        assert!(result.is_err());
    }
}
