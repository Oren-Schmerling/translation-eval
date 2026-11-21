import * as fs from 'fs';
import * as path from 'path';
import {
  CoarseDimensions,
  EvalTier,
  FewShotExample,
  Language,
  SuiteType,
  TargetVariant,
  TranslationRequest,
} from './types';

export interface Dataset {
  variants: TargetVariant[];
}

export interface LoadDatasetOptions {
  datasetVersion?: 'v1' | 'v2';
  track?: 'casual_conversational' | 'register_contrastive' | 'diagnostic_morphology' | 'all' | string;
  datasetPath?: string;
}

export function loadDataset(
  datasetPathOrOptions?: string | LoadDatasetOptions,
  options?: LoadDatasetOptions
): Dataset {
  let opts: LoadDatasetOptions = {};
  if (typeof datasetPathOrOptions === 'string') {
    opts = { ...(options || {}), datasetPath: datasetPathOrOptions };
  } else if (datasetPathOrOptions && typeof datasetPathOrOptions === 'object') {
    opts = datasetPathOrOptions;
  } else if (options && typeof options === 'object') {
    opts = options;
  }

  const isV2 = opts.datasetVersion === 'v2';
  const defaultPath = isV2
    ? path.join(process.cwd(), 'data', 'compiled', 'dataset_v2.jsonl')
    : path.join(process.cwd(), 'data', 'compiled', 'eval_dataset.json');

  const p = opts.datasetPath || defaultPath;
  const raw = fs.readFileSync(p, 'utf8');

  let variants: TargetVariant[] = [];
  if (p.endsWith('.jsonl')) {
    const lines = raw.split('\n');
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        const item = JSON.parse(trimmed);
        const profile = item.pragmatic_profile || {};
        variants.push({
          variant_id: `${item.item_id}_v1`,
          item_id: item.item_id,
          source_en: item.source_en,
          lang: item.lang,
          track: item.track || 'casual_conversational',
          bucket: 'A',
          is_cued_source: false,
          coarse: {
            formality: profile.register_tier || 'default',
            speaker_gender: profile.speaker_gender || null,
            listener_gender: profile.listener_gender || null,
            dialect: profile.regional_dialect || null,
          },
          granular: {
            pronoun_tier: null,
            kinship_pair: null,
            relationship: profile.social_relationship || null,
          },
          conflict_override: null,
          gold_reference: item.reference_target || '',
          anchors: {
            must_contain_any: item.required_anchors || [],
            must_not_contain: item.forbidden_markers || [],
            synset_equivalents: {},
            acceptable_synonyms: [],
          },
        });
      } catch {
        // ignore parse error
      }
    }
  } else {
    const parsed = JSON.parse(raw);
    variants = parsed.variants || [];
  }

  if (opts.track && opts.track !== 'all') {
    variants = variants.filter((v) => v.track === opts.track);
  }

  return { variants };
}

function coarseCellEq(a: CoarseDimensions, b: CoarseDimensions): boolean {
  return (
    a.formality === b.formality &&
    a.speaker_gender === b.speaker_gender &&
    a.listener_gender === b.listener_gender &&
    a.dialect === b.dialect
  );
}

function coarseDiffersOnPopulatedAxis(target: CoarseDimensions, other: CoarseDimensions): boolean {
  if (target.speaker_gender && other.speaker_gender && target.speaker_gender !== other.speaker_gender) return true;
  if (target.listener_gender && other.listener_gender && target.listener_gender !== other.listener_gender) return true;
  if (target.dialect && other.dialect && target.dialect !== other.dialect) return true;
  if (target.formality && other.formality && target.formality !== other.formality) return true;
  return false;
}

export function getFewShotExamples(
  dataset: Dataset,
  target: TargetVariant,
  k: number
): FewShotExample[] {
  if (k <= 0) return [];
  const candidates = dataset.variants.filter(
    (v) => v.lang === target.lang && v.item_id !== target.item_id
  );

  candidates.sort((a, b) => {
    const scoreA = coarseCellEq(a.coarse, target.coarse) ? 0 : a.coarse.formality === target.coarse.formality ? 1 : 2;
    const scoreB = coarseCellEq(b.coarse, target.coarse) ? 0 : b.coarse.formality === target.coarse.formality ? 1 : 2;
    return scoreA - scoreB;
  });

  const examples: FewShotExample[] = [];
  const seen = new Set<string>();
  for (const v of candidates) {
    if (examples.length >= k) break;
    if (!seen.has(v.variant_id)) {
      seen.add(v.variant_id);
      examples.push({
        source_en: v.source_en,
        target: v.gold_reference,
      });
    }
  }
  return examples;
}

export function getContrastiveExamples(
  dataset: Dataset,
  target: TargetVariant,
  k: number
): FewShotExample[] {
  if (k <= 0) return [];
  const pool = dataset.variants.filter(
    (v) => v.lang === target.lang && v.item_id !== target.item_id
  );

  const positives = pool.filter((v) => coarseCellEq(v.coarse, target.coarse));
  const negatives = pool.filter((v) => coarseDiffersOnPopulatedAxis(target.coarse, v.coarse));

  const examples: FewShotExample[] = [];
  for (let i = 0; i < Math.min(k, positives.length, negatives.length); i++) {
    examples.push({
      source_en: positives[i].source_en,
      target: positives[i].gold_reference,
      negative: negatives[i].gold_reference,
    });
  }
  return examples;
}

export function buildRequestsForSuite(
  dataset: Dataset,
  providerId: string,
  suite: SuiteType | string
): TranslationRequest[] {
  const primaryTiers: EvalTier[] = [
    'tier1_default_zero_shot',
    'tier2_a_coarse_params',
    'tier3_a_few_shot_k5_pos',
  ];

  let selectedVariants: TargetVariant[] = [];
  let tiersToUse: EvalTier[] = primaryTiers;

  const isV2 = dataset.variants.some((v) => v.item_id?.startsWith('v2_'));

  if (suite === 'smoke' && isV2) {
    // 30 items total: 10 per lang (gu, he, vi) across casual_conversational (5) + diagnostic_morphology (5)
    const langs: Language[] = ['gu', 'he', 'vi'];
    for (const l of langs) {
      const casual = dataset.variants.filter((v) => v.lang === l && v.track === 'casual_conversational').slice(0, 5);
      const diag = dataset.variants.filter((v) => v.lang === l && v.track === 'diagnostic_morphology').slice(0, 5);
      selectedVariants.push(...casual, ...diag);
    }
    tiersToUse = ['tier1_default_zero_shot'];
  } else if (suite === 'smoke') {
    // Legacy v1 smoke: 9 items across 3 primary tiers
    const langs: Language[] = ['gu', 'he', 'vi'];
    for (const l of langs) {
      const match = dataset.variants.find((v) => v.lang === l && v.item_id === 'cul_a_01')
        || dataset.variants.find((v) => v.lang === l);
      if (match) selectedVariants.push(match);
    }
    tiersToUse = primaryTiers;
  } else {
    selectedVariants = dataset.variants;
    tiersToUse = primaryTiers;
  }

  const requests: TranslationRequest[] = [];
  for (const tier of tiersToUse) {
    for (const variant of selectedVariants) {
      requests.push(makeRequest(dataset, providerId, tier, variant));
    }
  }
  return requests;
}

function makeRequest(
  dataset: Dataset,
  providerId: string,
  tier: EvalTier,
  variant: TargetVariant
): TranslationRequest {
  let examples: FewShotExample[] = [];
  if (tier === 'tier3_a_few_shot_k5_pos') {
    examples = getFewShotExamples(dataset, variant, 5);
  } else if (tier === 'tier3_b_few_shot_k20_pos') {
    examples = getFewShotExamples(dataset, variant, 20);
  } else if (tier === 'tier3_c_few_shot_k5_contrastive') {
    examples = getContrastiveExamples(dataset, variant, 5);
  }

  return {
    request_id: `${providerId}-${tier}-${variant.variant_id}`,
    tier,
    item_id: variant.item_id,
    variant_id: variant.variant_id,
    source_en: variant.source_en,
    lang: variant.lang,
    coarse: variant.coarse,
    granular: variant.granular,
    conflict_override: tier === 'tier2_c_adversarial_conflict' ? variant.conflict_override : null,
    examples,
    gold_reference: variant.gold_reference,
    anchors: variant.anchors,
    track: variant.track,
  };
}
