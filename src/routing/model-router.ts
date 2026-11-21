import { Language } from '../types';

export class BlockedModelForLanguageError extends Error {
  constructor(public readonly lang: string, public readonly model: string) {
    super(`BlockedModelForLanguageError: Model '${model}' is strictly blocked for language '${lang}' due to hallucination risks.`);
    this.name = 'BlockedModelForLanguageError';
  }
}

export const BLOCKED_MODELS_BY_LANG: Record<string, string[]> = {
  he: ['tencent/hy-mt2-1.8b'],
  gu: ['z-ai/glm-5.3-flash', 'tencent/hy-mt2-1.8b'],
  vi: [
    'tencent/hy-mt2-1.8b',
    'tencent/hy-mt2-7b',
    'tencent/hy-mt2-30b-a3b',
  ],
};

export const DEFAULT_ACTIVE_MODELS = [
  'deepseek/deepseek-v4.1-flash',
  'anthropic/claude-haiku-5.5',
  'tencent/hy-mt2-7b',
  'tencent/hy-mt2-30b-a3b',
  'z-ai/glm-5.3-flash',
  'qwen/qwen3.8-flash',
];

export const PURGED_DEFAULT_MODELS = [
  'anthropic/claude-sonnet-5.5',
  'qwen/qwen3.8-27b',
  'minimax/minimax-m3.1-flash',
  'tencent/hy-mt2-1.8b',
];

export function cleanModelSlug(model: string): string {
  return model.replace(/^openrouter:/, '').replace(/:batch$/, '');
}

export function isModelBlockedForLanguage(lang: string, model: string): boolean {
  const clean = cleanModelSlug(model).toLowerCase();
  // Any language blocks 1.8b
  if (clean === 'tencent/hy-mt2-1.8b' || clean.includes('hy-mt2-1.8b')) {
    return true;
  }
  const blockedList = BLOCKED_MODELS_BY_LANG[lang.toLowerCase()] || [];
  return blockedList.some((blocked) => {
    const cleanBlocked = cleanModelSlug(blocked).toLowerCase();
    return clean === cleanBlocked || clean.includes(cleanBlocked);
  });
}

export function assertModelAllowedForLanguage(lang: string, model: string): void {
  if (isModelBlockedForLanguage(lang, model)) {
    throw new BlockedModelForLanguageError(lang, model);
  }
}

export interface PragmaticRoutingParams {
  lang: Language | string;
  listener_gender?: string | null;
  speaker_gender?: string | null;
  register_tier?: string | null;
  regional_dialect?: string | null;
  social_relationship?: string | null;
  preferFast?: boolean;
}

export function requiresPragmaticSteering(params: PragmaticRoutingParams): boolean {
  if (params.listener_gender === 'female' || params.speaker_gender === 'female') return true;
  if (params.register_tier === 'intimate' || params.register_tier === 'honorific') return true;
  if (params.regional_dialect === 'vi_south') return true;
  if (params.social_relationship !== null && params.social_relationship !== undefined && params.social_relationship !== '') return true;
  return false;
}

export function getModelFallbackChain(params: PragmaticRoutingParams): string[] {
  const lang = params.lang.toLowerCase();
  let rawChain: string[];

  if (params.preferFast) {
    rawChain = ['anthropic/claude-haiku-5.5', 'deepseek/deepseek-v4.1-flash'];
  } else if (lang === 'he') {
    if (requiresPragmaticSteering(params)) {
      rawChain = [
        'deepseek/deepseek-v4.1-flash',
        'anthropic/claude-haiku-5.5',
        'tencent/hy-mt2-7b',
      ];
    } else {
      rawChain = [
        'deepseek/deepseek-v4.1-flash',
        'tencent/hy-mt2-7b',
        'anthropic/claude-haiku-5.5',
      ];
    }
  } else if (lang === 'gu') {
    if (requiresPragmaticSteering(params)) {
      rawChain = [
        'deepseek/deepseek-v4.1-flash',
        'anthropic/claude-haiku-5.5',
        'tencent/hy-mt2-7b',
        'tencent/hy-mt2-30b-a3b',
      ];
    } else {
      rawChain = [
        'tencent/hy-mt2-7b',
        'deepseek/deepseek-v4.1-flash',
        'anthropic/claude-haiku-5.5',
        'tencent/hy-mt2-30b-a3b',
      ];
    }
  } else if (lang === 'vi') {
    if (params.regional_dialect === 'vi_south') {
      rawChain = [
        'z-ai/glm-5.3-flash',
        'deepseek/deepseek-v4.1-flash',
        'anthropic/claude-haiku-5.5',
        'qwen/qwen3.8-flash',
      ];
    } else {
      rawChain = [
        'deepseek/deepseek-v4.1-flash',
        'anthropic/claude-haiku-5.5',
        'z-ai/glm-5.3-flash',
        'qwen/qwen3.8-flash',
      ];
    }
  } else {
    rawChain = [
      'deepseek/deepseek-v4.1-flash',
      'anthropic/claude-haiku-5.5',
      'qwen/qwen3.8-flash',
    ];
  }

  // Automatically filter out blocked models for this language
  return rawChain.filter((m) => !isModelBlockedForLanguage(lang, m));
}

export function normalizeMessagesForModel(
  model: string,
  systemPrompt: string | undefined,
  userPrompt: string,
  isFewShotTier = false
): Array<{ role: string; content: string; cache_control?: { type: string } }> {
  const isTencent = cleanModelSlug(model).startsWith('tencent/hy-mt2-');
  if (isTencent) {
    const combinedContent = systemPrompt ? `${systemPrompt}\n\n${userPrompt}` : userPrompt;
    return [{ role: 'user', content: combinedContent }];
  }

  const messages: Array<{ role: string; content: string; cache_control?: { type: string } }> = [];
  if (systemPrompt) {
    messages.push({
      role: 'system',
      content: systemPrompt,
      ...(isFewShotTier ? { cache_control: { type: 'ephemeral' } } : {}),
    });
  }
  messages.push({ role: 'user', content: userPrompt });
  return messages;
}

export function buildDeterministicSessionId(params: {
  lang: string;
  register_tier?: string | null;
  regional_dialect?: string | null;
  social_relationship?: string | null;
}): string {
  const reg = params.register_tier || 'none';
  const dial = params.regional_dialect || 'none';
  const soc = params.social_relationship || 'none';
  return `mt-${params.lang}-${reg}-${dial}-${soc}`;
}
