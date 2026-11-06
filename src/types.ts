export type Language = 'he' | 'gu' | 'vi';

export type EvalTier =
  | 'tier1_default_zero_shot'
  | 'tier2_a_coarse_params'
  | 'tier2_b_granular_params'
  | 'tier2_c_adversarial_conflict'
  | 'tier3_a_few_shot_k5_pos'
  | 'tier3_b_few_shot_k20_pos'
  | 'tier3_c_few_shot_k5_contrastive';

export interface CoarseDimensions {
  formality: string;
  speaker_gender: string | null;
  listener_gender: string | null;
  dialect: string | null;
}

export interface GranularDimensions {
  pronoun_tier: string | null;
  kinship_pair: string | null;
  relationship: string | null;
}

export interface ComplianceAnchors {
  must_contain_any: string[];
  must_not_contain: string[];
  synset_equivalents: Record<string, string[]>;
  acceptable_synonyms: string[];
}

export interface TargetVariant {
  variant_id: string;
  item_id: string;
  source_en: string;
  lang: Language;
  track: string;
  bucket: string;
  is_cued_source: boolean;
  coarse: CoarseDimensions;
  granular: GranularDimensions;
  conflict_override?: CoarseDimensions | null;
  gold_reference: string;
  anchors: ComplianceAnchors;
}

export interface FewShotExample {
  source_en: string;
  target: string;
  negative?: string | null;
}

export interface TranslationRequest {
  request_id: string;
  tier: EvalTier;
  item_id: string;
  variant_id: string;
  source_en: string;
  lang: Language;
  coarse: CoarseDimensions;
  granular: GranularDimensions;
  conflict_override?: CoarseDimensions | null;
  examples: FewShotExample[];
  gold_reference: string;
  anchors: ComplianceAnchors;
}

export interface TelemetryEnvelope {
  translation: string;
  ttft_ms: number | null;
  total_latency_ms: number;
  prompt_tokens: number;
  completion_tokens: number;
  char_count: number;
  uncached_cost_usd: number;
  cached_cost_usd: number;
}

export interface RunRecord {
  provider_id: string;
  tier: EvalTier;
  request_id: string;
  item_id: string;
  variant_id: string;
  source_en: string;
  lang: Language;
  coarse: CoarseDimensions;
  granular: GranularDimensions;
  conflict_override: CoarseDimensions | null;
  track: string;
  bucket: string;
  is_cued_source: boolean;
  gold_reference: string;
  translation: string;
  anchors: ComplianceAnchors;
  capability: { status: 'supported' | 'unsupported_by_provider' | 'partially_supported' };
  ttft_ms: number | null;
  total_latency_ms: number;
  prompt_tokens: number;
  completion_tokens: number;
  char_count: number;
  uncached_cost_usd: number;
  cached_cost_usd: number;
  layer1: null;
  chrf: null;
  ensemble: null;
  bias_profile: null;
  high_disagreement_flag: boolean;
}

export interface ModelPricing {
  inputPerM: number;
  outputPerM: number;
  cachedReadPerM?: number;
}

export interface ProbedModel {
  id: string;
  modelSlug: string;
  provider: 'google-ai-studio' | 'openrouter';
  mode: 'stream' | 'batch' | 'free_queue';
  bucket: 'ai_studio' | 'fallback' | 'ceiling' | 'contender' | 'mt_specialist' | 'free';
  pricing: ModelPricing;
  isExpensive?: boolean;
}

export interface ProbedRoster {
  timestamp: string;
  ai_studio: {
    unlocked_stream: string[];
    unlocked_batch: string[];
    failed: { model: string; error: string }[];
  };
  openrouter: {
    fallbacks: string[];
    ceiling_model: string | null;
    contenders: string[];
    mt_specialists?: string[];
    free_models: string[];
  };
  active_roster: ProbedModel[];
}

export type SuiteType = 'smoke' | 'stratified' | 'full';

export const ZERO_PREAMBLE_SYSTEM_PROMPT =
  'Output ONLY the translated target text. Do not include explanations, transliterations, notes, or multiple gender/formality variants.';
