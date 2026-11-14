import { ProbedModel, ProbedRoster } from '../types';

export interface ProberOptions {
  openrouterApiKey?: string;
  ceilingModelOverride?: string;
  preferBatch?: boolean;
  skipMicroProbe?: boolean;
}

export function isExpensiveModel(inputPerM: number, outputPerM: number): boolean {
  return inputPerM > 0.50 || outputPerM > 2.50;
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

  let expensiveCeilingAdded = false;

  // Ceiling model selection (Bucket 2)
  const ceilingSlug = opts.ceilingModelOverride
    || (preferBatch && orModels.some((m) => m.id === 'anthropic/claude-sonnet-5.5:batch')
      ? 'anthropic/claude-sonnet-5.5:batch'
      : 'anthropic/claude-sonnet-5.5');

  const ceilingModelMeta = orModels.find((m) => m.id === ceilingSlug);
  if (ceilingModelMeta) {
    const inCost = (parseFloat(ceilingModelMeta.pricing?.prompt || '0') * 1_000_000) || 1.0;
    const outCost = (parseFloat(ceilingModelMeta.pricing?.completion || '0') * 1_000_000) || 5.0;
    roster.openrouter.ceiling_model = ceilingSlug;
    roster.active_roster.push({
      id: `openrouter:${ceilingSlug}`,
      modelSlug: ceilingSlug,
      provider: 'openrouter',
      mode: ceilingSlug.endsWith(':batch') ? 'batch' : 'stream',
      bucket: 'ceiling',
      pricing: { inputPerM: inCost, outputPerM: outCost, cachedReadPerM: 0.10 },
      isExpensive: true,
    });
    expensiveCeilingAdded = true;
  }

  // Bucket 3: High-Promise Low-Cost Contenders
  const contenderSlugs = [
    preferBatch ? 'anthropic/claude-haiku-5.5:batch' : 'anthropic/claude-haiku-5.5',
    preferBatch ? 'deepseek/deepseek-v4.1-flash:batch' : 'deepseek/deepseek-v4.1-flash-1017',
    'qwen/qwen3.8-flash',
    'qwen/qwen3.8-27b',
    preferBatch ? 'z-ai/glm-5.3-flash:batch' : 'z-ai/glm-5.3-flash',
    'minimax/minimax-m3.1-flash',
  ];

  for (const slug of contenderSlugs) {
    const m = orModels.find((item) => item.id === slug) || orModels.find((item) => item.id.replace(/:batch$/, '') === slug.replace(/:batch$/, ''));
    if (m && !roster.active_roster.some((ar) => ar.modelSlug === m.id)) {
      const inCost = (parseFloat(m.pricing?.prompt || '0') * 1_000_000) || 0;
      const outCost = (parseFloat(m.pricing?.completion || '0') * 1_000_000) || 0;
      if (isExpensiveModel(inCost, outCost)) continue;

      roster.openrouter.contenders.push(m.id);
      roster.active_roster.push({
        id: `openrouter:${m.id}`,
        modelSlug: m.id,
        provider: 'openrouter',
        mode: m.id.endsWith(':batch') ? 'batch' : 'stream',
        bucket: 'contender',
        pricing: { inputPerM: inCost, outputPerM: outCost },
      });
    }
  }

  // Bucket 3B: Dedicated Machine Translation Specialists (tencent/hy-mt2-*)
  roster.openrouter.mt_specialists = [];
  const mtSpecialistSlugs = [
    'tencent/hy-mt2-30b-a3b',
    'tencent/hy-mt2-7b',
    'tencent/hy-mt2-1.8b',
  ];

  for (const slug of mtSpecialistSlugs) {
    const m = orModels.find((item) => item.id === slug);
    if (m && !roster.active_roster.some((ar) => ar.modelSlug === m.id)) {
      const inCost = (parseFloat(m.pricing?.prompt || '0') * 1_000_000) || 0.074;
      const outCost = (parseFloat(m.pricing?.completion || '0') * 1_000_000) || 0.295;
      roster.openrouter.mt_specialists.push(m.id);
      roster.active_roster.push({
        id: `openrouter:${m.id}`,
        modelSlug: m.id,
        provider: 'openrouter',
        mode: 'stream',
        bucket: 'mt_specialist',
        pricing: { inputPerM: inCost, outputPerM: outCost },
      });
    }
  }

  return roster;
}
