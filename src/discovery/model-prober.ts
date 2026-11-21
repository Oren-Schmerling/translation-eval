import { ProbedModel, ProbedRoster } from '../types';

export interface ProberOptions {
  openrouterApiKey?: string;
  ceilingModelOverride?: string;
  includeCeiling?: boolean;
  includeFree?: boolean;
  preferBatch?: boolean;
  skipMicroProbe?: boolean;
}

export function isExpensiveModel(inputPerM: number, outputPerM: number): boolean {
  return inputPerM > 0.50 || outputPerM > 2.50;
}

export function resolveBatchSlug(
  slug: string,
  orModels: any[],
  preferBatch: boolean
): { modelSlug: string; isBatch: boolean } {
  if (!preferBatch) return { modelSlug: slug, isBatch: false };
  const batchCandidate = slug.endsWith(':batch') ? slug : `${slug}:batch`;
  const exists = orModels.some((m) => m.id === batchCandidate);
  return exists ? { modelSlug: batchCandidate, isBatch: true } : { modelSlug: slug, isBatch: false };
}

export function matchesScript(lang: 'gu' | 'he' | 'vi', text: string): boolean {
  if (!text || text.trim().length === 0) return false;
  switch (lang) {
    case 'gu':
      return /[\u0A80-\u0AFF]/.test(text);
    case 'he':
      return /[\u0590-\u05FF]/.test(text);
    case 'vi':
      return /[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]/.test(text)
        || text.toLowerCase().includes('chào')
        || text.toLowerCase().includes('buổi sáng');
  }
}

export async function probeAndResolveRoster(opts: ProberOptions): Promise<ProbedRoster> {
  const orKey = opts.openrouterApiKey || process.env.OPENROUTER_API_KEY || '';
  const preferBatch = opts.preferBatch ?? false;

  const roster: ProbedRoster = {
    timestamp: new Date().toISOString(),
    openrouter: { fallbacks: [], ceiling_model: null, contenders: [], free_models: [] },
    active_roster: [],
  };

  let orModels: any[] = [];
  if (orKey) {
    try {
      const res = await fetch('https://openrouter.ai/api/v1/models', {
        headers: { Authorization: `Bearer ${orKey}` },
        signal: AbortSignal.timeout(15_000),
      });
      if (res.ok) {
        const d: any = await res.json();
        orModels = d.data || [];
      }
    } catch {
      // ignore
    }
  }

  // Ceiling model selection (Bucket 2 - only if requested or explicit override)
  if (opts.includeCeiling || opts.ceilingModelOverride) {
    const baseCeiling = opts.ceilingModelOverride || 'anthropic/claude-sonnet-5.5';
    const { modelSlug: ceilingSlug, isBatch: isCeilingBatch } = resolveBatchSlug(baseCeiling, orModels, preferBatch);

    const ceilingModelMeta = orModels.find((m) => m.id === ceilingSlug) || orModels.find((m) => m.id === baseCeiling);
    if (ceilingModelMeta) {
      const inCost = (parseFloat(ceilingModelMeta.pricing?.prompt || '0') * 1_000_000) || 1.0;
      const outCost = (parseFloat(ceilingModelMeta.pricing?.completion || '0') * 1_000_000) || 5.0;
      roster.openrouter.ceiling_model = ceilingSlug;
      roster.active_roster.push({
        id: `openrouter:${ceilingSlug}`,
        modelSlug: ceilingSlug,
        provider: 'openrouter',
        mode: isCeilingBatch ? 'batch' : 'stream',
        bucket: 'ceiling',
        pricing: { inputPerM: inCost, outputPerM: outCost, cachedReadPerM: 0.10 },
        isExpensive: true,
      });
    }
  }

  // Bucket 3: High-Promise Low-Cost Contenders (Purged Roster: deepseek, haiku, glm-5.3, qwen3.8-flash)
  const contenderSlugs = [
    'deepseek/deepseek-v4.1-flash',
    'anthropic/claude-haiku-5.5',
    'z-ai/glm-5.3-flash',
    'qwen/qwen3.8-flash',
  ];

  for (const baseSlug of contenderSlugs) {
    const { modelSlug, isBatch } = resolveBatchSlug(baseSlug, orModels, preferBatch);
    const m = orModels.find((item) => item.id === modelSlug) || orModels.find((item) => item.id === baseSlug);
    if (m && !roster.active_roster.some((ar) => ar.modelSlug.replace(/:batch$/, '') === baseSlug.replace(/:batch$/, ''))) {
      const inCost = (parseFloat(m.pricing?.prompt || '0') * 1_000_000) || 0;
      const outCost = (parseFloat(m.pricing?.completion || '0') * 1_000_000) || 0;
      if (isExpensiveModel(inCost, outCost)) continue;

      roster.openrouter.contenders.push(modelSlug);
      roster.active_roster.push({
        id: `openrouter:${modelSlug}`,
        modelSlug: modelSlug,
        provider: 'openrouter',
        mode: isBatch ? 'batch' : 'stream',
        bucket: 'contender',
        pricing: { inputPerM: inCost, outputPerM: outCost },
      });
    }
  }

  // Bucket 3B: Dedicated Machine Translation Specialists (Purged: 7b and 30b-a3b only, 1.8b purged)
  roster.openrouter.mt_specialists = [];
  const mtSpecialistSlugs = [
    'tencent/hy-mt2-7b',
    'tencent/hy-mt2-30b-a3b',
  ];

  for (const baseSlug of mtSpecialistSlugs) {
    const { modelSlug, isBatch } = resolveBatchSlug(baseSlug, orModels, preferBatch);
    const m = orModels.find((item) => item.id === modelSlug) || orModels.find((item) => item.id === baseSlug);
    if (m && !roster.active_roster.some((ar) => ar.modelSlug === modelSlug)) {
      const inCost = (parseFloat(m.pricing?.prompt || '0') * 1_000_000) || 0.074;
      const outCost = (parseFloat(m.pricing?.completion || '0') * 1_000_000) || 0.295;
      roster.openrouter.mt_specialists.push(modelSlug);
      roster.active_roster.push({
        id: `openrouter:${modelSlug}`,
        modelSlug: modelSlug,
        provider: 'openrouter',
        mode: isBatch ? 'batch' : 'stream',
        bucket: 'mt_specialist',
        pricing: { inputPerM: inCost, outputPerM: outCost },
      });
    }
  }

  return roster;
}
