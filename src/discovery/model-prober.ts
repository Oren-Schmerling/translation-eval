import { ModelPricing, ProbedModel, ProbedRoster } from '../types';

export interface ProberOptions {
  geminiApiKey?: string;
  openrouterApiKey?: string;
  ceilingModelOverride?: string;
  preferBatch?: boolean;
  skipMicroProbe?: boolean;
}

export interface AiStudioProbeResult {
  model: string;
  status: 'AI_STUDIO_BATCH' | 'AI_STUDIO_STREAM' | 'FALLBACK_TO_OPENROUTER';
  error?: string;
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

export async function probeGoogleAiStudioModel(
  model: string,
  apiKey: string
): Promise<AiStudioProbeResult> {
  const cleanId = model.replace(/^models\//, '');
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${cleanId}:generateContent?key=${apiKey}`;

  try {
    let res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{ parts: [{ text: 'Translate to Spanish: hi' }] }],
        generationConfig: { maxOutputTokens: 5, thinkingConfig: { thinkingBudget: 0 } },
      }),
      signal: AbortSignal.timeout(10_000),
    });

    if (!res.ok && res.status === 400) {
      res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{ parts: [{ text: 'Translate to Spanish: hi' }] }],
          generationConfig: { maxOutputTokens: 5 },
        }),
        signal: AbortSignal.timeout(10_000),
      });
    }

    if (!res.ok) {
      const errText = await res.text();
      return {
        model: cleanId,
        status: 'FALLBACK_TO_OPENROUTER',
        error: `HTTP ${res.status}: ${errText.slice(0, 100)}`,
      };
    }

    // Model is unlocked! Check if batch is permitted
    const batchUrl = `https://generativelanguage.googleapis.com/v1beta/models/${cleanId}:batchGenerateContent?key=${apiKey}`;
    try {
      const batchRes = await fetch(batchUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          batch: {
            displayName: 'probe',
            inputConfig: { requests: { requests: [{ request: { contents: [{ parts: [{ text: 'hi' }] }] } }] } },
          },
        }),
        signal: AbortSignal.timeout(10_000),
      });
      if (batchRes.status === 200 || batchRes.status === 202) {
        return { model: cleanId, status: 'AI_STUDIO_BATCH' };
      }
    } catch {
      // ignore
    }

    return { model: cleanId, status: 'AI_STUDIO_STREAM' };
  } catch (err: any) {
    return {
      model: cleanId,
      status: 'FALLBACK_TO_OPENROUTER',
      error: err?.message || 'Timeout/network error',
    };
  }
}



export async function probeAndResolveRoster(opts: ProberOptions): Promise<ProbedRoster> {
  const geminiKey = opts.geminiApiKey || process.env.GEMINI_API_KEY || process.env.GOOGLE_AI_STUDIO_API_KEY || '';
  const orKey = opts.openrouterApiKey || process.env.OPENROUTER_API_KEY || '';
  const preferBatch = opts.preferBatch ?? false;

  const roster: ProbedRoster = {
    timestamp: new Date().toISOString(),
    ai_studio: { unlocked_stream: [], unlocked_batch: [], failed: [] },
    openrouter: { fallbacks: [], ceiling_model: null, contenders: [], free_models: [] },
    active_roster: [],
  };

  const targetCandidates = [
    'gemini-4-argon',
    'gemini-3.8-pro',
    'gemini-3.8-flash',
    'gemini-3.8-flash-lite',
    'gemma-4-31b-it',
    'gemma-4-26b-a4b-it',
  ];

  if (geminiKey) {
    try {
      const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models?pageSize=100&key=${geminiKey}`, {
        signal: AbortSignal.timeout(10_000),
      });
      if (res.ok) {
        const data: any = await res.json();
        const discovered = (data.models || [])
          .map((m: any) => m.name.replace(/^models\//, ''))
          .filter((id: string) =>
            (/^gemini-3\.[7-9]/.test(id) || /^gemini-[4-9]/.test(id)) &&
            !id.includes('-tts') &&
            !id.includes('-image') &&
            !id.includes('-live') &&
            !id.includes('-audio') &&
            !id.includes('-embedding')
          );
        for (const id of discovered) {
          if (!targetCandidates.includes(id)) targetCandidates.push(id);
        }
      }
    } catch {
      // fallback to static list
    }

    console.log(`[Phase 1A] Probing ${targetCandidates.length} Google AI Studio candidates...`);
    for (const cand of targetCandidates) {
      const probeRes = await probeGoogleAiStudioModel(cand, geminiKey);
      console.log(`  - ${cand}: ${probeRes.status}${probeRes.error ? ` (${probeRes.error.slice(0, 45)})` : ''}`);

      if (probeRes.status === 'AI_STUDIO_BATCH') {
        roster.ai_studio.unlocked_batch.push(probeRes.model);
        roster.active_roster.push({
          id: `google-ai-studio:${probeRes.model}`,
          modelSlug: probeRes.model,
          provider: 'google-ai-studio',
          mode: preferBatch ? 'batch' : 'stream',
          bucket: 'ai_studio',
          pricing: { inputPerM: 0.075, outputPerM: 0.30 },
        });
      } else if (probeRes.status === 'AI_STUDIO_STREAM') {
        roster.ai_studio.unlocked_stream.push(probeRes.model);
        roster.active_roster.push({
          id: `google-ai-studio:${probeRes.model}`,
          modelSlug: probeRes.model,
          provider: 'google-ai-studio',
          mode: 'stream',
          bucket: 'ai_studio',
          pricing: { inputPerM: 0.15, outputPerM: 0.60 },
        });
      } else {
        roster.ai_studio.failed.push({ model: probeRes.model, error: probeRes.error || 'Probe failed' });
      }
    }
  }

  // Step 1B: Fetch OpenRouter models and assemble buckets
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

  const lockedAiStudioSlugs = new Set([
    ...roster.ai_studio.unlocked_stream,
    ...roster.ai_studio.unlocked_batch,
  ]);

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

  // Bucket 1: Fallback Google models
  for (const failed of roster.ai_studio.failed) {
    const raw = failed.model;
    const potentialSlugs = [
      preferBatch ? `google/${raw}:batch` : null,
      `google/${raw}`,
    ].filter(Boolean) as string[];

    const matched = orModels.find((m) => potentialSlugs.includes(m.id));
    if (matched && !lockedAiStudioSlugs.has(raw)) {
      const inCost = (parseFloat(matched.pricing?.prompt || '0') * 1_000_000) || 0;
      const outCost = (parseFloat(matched.pricing?.completion || '0') * 1_000_000) || 0;
      const expensive = isExpensiveModel(inCost, outCost);

      if (expensive) {
        if (opts.ceilingModelOverride === matched.id && !expensiveCeilingAdded) {
          expensiveCeilingAdded = true;
        } else {
          continue; // Blocked by single expensive model budget cap
        }
      }

      roster.openrouter.fallbacks.push(matched.id);
      roster.active_roster.push({
        id: `openrouter:${matched.id}`,
        modelSlug: matched.id,
        provider: 'openrouter',
        mode: matched.id.endsWith(':batch') ? 'batch' : 'stream',
        bucket: 'fallback',
        pricing: { inputPerM: inCost, outputPerM: outCost },
        isExpensive: expensive,
      });
    }
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
