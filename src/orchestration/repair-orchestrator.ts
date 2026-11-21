import { matchesScript } from '../discovery/model-prober';
import { OpenRouterRunner } from '../providers/openrouter';
import { getModelFallbackChain } from '../routing/model-router';
import { Language, ModelPricing, TelemetryEnvelope, TranslationRequest } from '../types';

export interface Layer1ValidationResult {
  passed: boolean;
  errors: string[];
}

export function validateLayer1(
  lang: Language | string,
  translation: string,
  requiredAnchors: string[] = [],
  forbiddenMarkers: string[] = []
): Layer1ValidationResult {
  const errors: string[] = [];
  const normalizedText = translation.trim();

  if (!matchesScript(lang as 'gu' | 'he' | 'vi', normalizedText)) {
    errors.push(`invalid script purity for language '${lang}'`);
  }

  const missingAnchors: string[] = [];
  for (const anchor of requiredAnchors) {
    if (!anchor) continue;
    if (!normalizedText.includes(anchor)) {
      missingAnchors.push(anchor);
    }
  }
  if (missingAnchors.length > 0) {
    errors.push(`missing required anchors [${missingAnchors.join(', ')}]`);
  }

  const detectedForbidden: string[] = [];
  for (const marker of forbiddenMarkers) {
    if (!marker) continue;
    if (normalizedText.includes(marker)) {
      detectedForbidden.push(marker);
    }
  }
  if (detectedForbidden.length > 0) {
    errors.push(`detected forbidden markers [${detectedForbidden.join(', ')}]`);
  }

  return {
    passed: errors.length === 0,
    errors,
  };
}

export interface OrchestratedExecutionResult {
  envelope: TelemetryEnvelope;
  first_pass_translation: string;
  final_translation: string;
  layer1_passed: boolean;
  repair_triggered: boolean;
  repair_succeeded: boolean;
  first_pass_acr: number;
  post_repair_acr: number;
  total_latency_ms: number;
  total_cost_usd: number;
  repair_model_used?: string;
}

export async function runOrchestratedTranslation(
  runner: OpenRouterRunner,
  primaryModel: string,
  req: TranslationRequest,
  pricing: ModelPricing,
  options: {
    enableRepair?: boolean;
    requiredAnchors?: string[];
    forbiddenMarkers?: string[];
  } = {}
): Promise<OrchestratedExecutionResult> {
  const required = options.requiredAnchors
    ?? req.anchors?.must_contain_any
    ?? [];
  const forbidden = options.forbiddenMarkers
    ?? req.anchors?.must_not_contain
    ?? [];

  // Stage 2: Primary Call
  const primaryEnv = await runner.translate(primaryModel, req, pricing);
  const firstPassTranslation = primaryEnv.translation;

  // Stage 3: Synchronous layer1_passed Gate (<1ms)
  const validation = validateLayer1(req.lang, firstPassTranslation, required, forbidden);

  if (validation.passed || !options.enableRepair) {
    return {
      envelope: primaryEnv,
      first_pass_translation: firstPassTranslation,
      final_translation: firstPassTranslation,
      layer1_passed: validation.passed,
      repair_triggered: false,
      repair_succeeded: false,
      first_pass_acr: validation.passed ? 1 : 0,
      post_repair_acr: validation.passed ? 1 : 0,
      total_latency_ms: primaryEnv.total_latency_ms,
      total_cost_usd: primaryEnv.uncached_cost_usd,
    };
  }

  // Stage 4: 1-Hop Contrastive Repair (ONLY when layer1_passed === false and enableRepair is set)
  const fallbackChain = getModelFallbackChain({
    lang: req.lang,
    listener_gender: req.coarse?.listener_gender,
    speaker_gender: req.coarse?.speaker_gender,
    register_tier: req.coarse?.formality,
    regional_dialect: req.coarse?.dialect,
    social_relationship: req.granular?.relationship,
  });

  // Pick next fast instruction-following model
  const repairModel = fallbackChain.find((m) => m !== primaryModel && !m.includes('tencent/hy-mt2-'))
    || fallbackChain[0]
    || 'deepseek/deepseek-v4.1-flash';

  const repairRequest: TranslationRequest = {
    ...req,
    source_en: `${req.source_en}\n\n[Previous Draft]: ${firstPassTranslation}\n[Correction Requirement]: ${validation.errors.join('; ')}. Output ONLY the grammatically consistent corrected translation.`,
  };

  try {
    const repairEnv = await runner.translate(repairModel, repairRequest, pricing);
    const postValidation = validateLayer1(req.lang, repairEnv.translation, required, forbidden);

    return {
      envelope: repairEnv,
      first_pass_translation: firstPassTranslation,
      final_translation: repairEnv.translation,
      layer1_passed: postValidation.passed,
      repair_triggered: true,
      repair_succeeded: postValidation.passed,
      first_pass_acr: 0,
      post_repair_acr: postValidation.passed ? 1 : 0,
      total_latency_ms: primaryEnv.total_latency_ms + repairEnv.total_latency_ms,
      total_cost_usd: primaryEnv.uncached_cost_usd + repairEnv.uncached_cost_usd,
      repair_model_used: repairModel,
    };
  } catch (err) {
    // If repair call fails, fallback to first pass
    return {
      envelope: primaryEnv,
      first_pass_translation: firstPassTranslation,
      final_translation: firstPassTranslation,
      layer1_passed: false,
      repair_triggered: true,
      repair_succeeded: false,
      first_pass_acr: 0,
      post_repair_acr: 0,
      total_latency_ms: primaryEnv.total_latency_ms,
      total_cost_usd: primaryEnv.uncached_cost_usd,
    };
  }
}
