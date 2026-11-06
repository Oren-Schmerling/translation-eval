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

export function loadDataset(datasetPath?: string): Dataset {
  const p = datasetPath || path.join(process.cwd(), 'data', 'compiled', 'eval_dataset.json');
  const raw = fs.readFileSync(p, 'utf8');
  return JSON.parse(raw);
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
  suite: SuiteType
): TranslationRequest[] {
  const primaryTiers: EvalTier[] = [
    'tier1_default_zero_shot',
    'tier2_a_coarse_params',
    'tier3_a_few_shot_k5_pos',
  ];

  const allTiers: EvalTier[] = [
    'tier1_default_zero_shot',
    'tier2_a_coarse_params',
    'tier2_b_granular_params',
    'tier2_c_adversarial_conflict',
    'tier3_a_few_shot_k5_pos',
    'tier3_b_few_shot_k20_pos',
    'tier3_c_few_shot_k5_contrastive',
  ];

  let selectedVariants: TargetVariant[] = [];
  let tiersToUse: EvalTier[] = primaryTiers;

  if (suite === 'smoke') {
    // 9 items/model: 1 variant per lang (gu, he, vi) across 3 primary tiers (3 * 3 = 9)
    const langs: Language[] = ['gu', 'he', 'vi'];
    for (const l of langs) {
      const match = dataset.variants.find((v) => v.lang === l && v.item_id === 'cul_a_01')
        || dataset.variants.find((v) => v.lang === l);
      if (match) selectedVariants.push(match);
    }
    tiersToUse = primaryTiers;
  } else if (suite === 'stratified') {
    // 180 items/model across all tiers and gu/he/vi
    // Balanced selection: 60 per language, spread across tiers
    const byLang: Record<Language, TargetVariant[]> = { gu: [], he: [], vi: [] };
    for (const v of dataset.variants) byLang[v.lang].push(v);

    // Collect requests across all tiers up to 180 total
    const requests: TranslationRequest[] = [];
    const targetPerLang = 60;
    for (const l of ['gu', 'he', 'vi'] as Language[]) {
      const list = byLang[l];
      let langCount = 0;
      for (const tier of allTiers) {
        for (const variant of list) {
          if (langCount >= targetPerLang) break;
          requests.push(makeRequest(dataset, providerId, tier, variant));
          langCount++;
        }
        if (langCount >= targetPerLang) break;
      }
    }
    return requests;
  } else {
    // full: 720 variants across 3 primary tiers = 2,160 items/model
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
  };
}
