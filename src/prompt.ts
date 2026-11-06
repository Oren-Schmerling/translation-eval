import {
  EvalTier,
  Language,
  TranslationRequest,
  ZERO_PREAMBLE_SYSTEM_PROMPT,
} from './types';

export function languageEnglishName(lang: Language): string {
  switch (lang) {
    case 'he':
      return 'Hebrew';
    case 'gu':
      return 'Gujarati';
    case 'vi':
      return 'Vietnamese';
  }
}

export function buildSystemPrompt(tier: EvalTier): string {
  // Constraint 5: Tier 1 Zero-Preamble Contract
  if (tier === 'tier1_default_zero_shot') {
    return ZERO_PREAMBLE_SYSTEM_PROMPT;
  }
  return (
    'You are a professional translator. ' +
    ZERO_PREAMBLE_SYSTEM_PROMPT
  );
}

export function buildPromptText(req: TranslationRequest): string {
  const langName = languageEnglishName(req.lang);
  let prompt = `You are a professional translator. Translate the following English source text into ${langName}.\n`;
  prompt += 'Preserve pragmatic nuances such as formality, speaker and listener gender, and dialect.\n\n';

  switch (req.tier) {
    case 'tier2_a_coarse_params':
    case 'tier2_b_granular_params':
    case 'tier2_c_adversarial_conflict': {
      prompt += `COARSE PARAMETERS: ${JSON.stringify({
        formality: req.coarse.formality,
        speaker_gender: req.coarse.speaker_gender,
        listener_gender: req.coarse.listener_gender,
        dialect: req.coarse.dialect,
      })}\n\n`;

      if (req.tier === 'tier2_b_granular_params') {
        prompt += `GRANULAR PARAMETERS: ${JSON.stringify({
          pronoun_tier: req.granular.pronoun_tier,
          kinship_pair: req.granular.kinship_pair,
          relationship: req.granular.relationship,
        })}\n\n`;
      }

      if (req.tier === 'tier2_c_adversarial_conflict') {
        const override = req.conflict_override ? {
          formality: req.conflict_override.formality,
          speaker_gender: req.conflict_override.speaker_gender,
          listener_gender: req.conflict_override.listener_gender,
          dialect: req.conflict_override.dialect,
        } : {};
        prompt += `ADVERSARIAL OVERRIDE: ${JSON.stringify(override)}\n`;
        prompt += 'These API parameters override any conflicting cues in the source text.\n\n';
      }
      break;
    }
    case 'tier1_default_zero_shot':
    case 'tier3_a_few_shot_k5_pos':
    case 'tier3_b_few_shot_k20_pos':
    case 'tier3_c_few_shot_k5_contrastive':
      break;
  }

  if (req.examples && req.examples.length > 0) {
    prompt += 'Examples:\n\n';
    req.examples.forEach((ex, idx) => {
      const num = idx + 1;
      if (ex.negative) {
        prompt += `Contrastive Example ${num}:\nSource: ${ex.source_en}\nDO NOT USE: ${ex.negative}\nUSE: ${ex.target}\n\n`;
      } else {
        prompt += `Positive Example ${num}:\nSource: ${ex.source_en}\nTarget: ${ex.target}\n\n`;
      }
    });
  }

  prompt += `Source: ${req.source_en}\n\nTranslation:`;
  return prompt;
}

export function cleanTranslationResponse(raw: string): string {
  let cleaned = raw.trim();
  // Strip code block fences if returned
  if (cleaned.startsWith('```') && cleaned.endsWith('```')) {
    cleaned = cleaned.replace(/^```[^\n]*\n/, '').replace(/\n```$/, '').trim();
  }
  return cleaned;
}
