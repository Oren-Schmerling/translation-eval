import { buildPromptText, buildSystemPrompt, cleanTranslationResponse } from '../prompt';
import {
  assertModelAllowedForLanguage,
  buildDeterministicSessionId,
  normalizeMessagesForModel,
} from '../routing/model-router';
import { ModelPricing, TelemetryEnvelope, TranslationRequest } from '../types';

export interface OpenRouterRunnerOptions {
  apiKey: string;
}

export class OpenRouterRunner {
  private apiKey: string;

  constructor(opts: OpenRouterRunnerOptions) {
    this.apiKey = opts.apiKey;
  }

  async translate(
    model: string,
    req: TranslationRequest,
    pricing: ModelPricing,
    isStreamMode = true
  ): Promise<TelemetryEnvelope> {
    assertModelAllowedForLanguage(req.lang, model);

    const startTime = performance.now();
    let ttftMs: number | null = null;

    const systemPrompt = buildSystemPrompt(req.tier);
    const userPrompt = buildPromptText(req);
    const isFewShotTier = req.tier.startsWith('tier3');

    const messages = normalizeMessagesForModel(model, systemPrompt, userPrompt, isFewShotTier);
    const sessionId = buildDeterministicSessionId({
      lang: req.lang,
      register_tier: req.coarse?.formality,
      regional_dialect: req.coarse?.dialect,
      social_relationship: req.granular?.relationship,
    });

    const providerConfig: any = { allow_fallbacks: true };
    if (isStreamMode) {
      providerConfig.sort = 'latency';
      providerConfig.preferred_max_latency = { p90: 1.5 };
    }

    const makeBody = (withReasoning: boolean) => ({
      model,
      messages,
      max_tokens: 1024,
      stream: true,
      session_id: sessionId,
      provider: providerConfig,
      ...(withReasoning ? { reasoning: { effort: 'none' } } : {}),
    });

    let resp = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(makeBody(true)),
    });

    if (!resp.ok && resp.status === 400) {
      resp = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(makeBody(false)),
      });
    }

    if (!resp.ok) {
      const errText = await resp.text();
      throw new Error(`OpenRouter HTTP ${resp.status}: ${errText}`);
    }

    const reader = resp.body?.getReader();
    const decoder = new TextDecoder();
    let translation = '';
    let promptTokens = 0;
    let completionTokens = 0;
    let buffer = '';

    if (reader) {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data:')) continue;
          const jsonStr = trimmed.slice(5).trim();
          if (!jsonStr || jsonStr === '[DONE]') continue;

          try {
            const parsed = JSON.parse(jsonStr);
            const delta = parsed?.choices?.[0]?.delta?.content;
            if (delta) {
              if (ttftMs === null) {
                ttftMs = performance.now() - startTime;
              }
              translation += delta;
            }
            if (parsed?.usage) {
              promptTokens = parsed.usage.prompt_tokens || promptTokens;
              completionTokens = parsed.usage.completion_tokens || completionTokens;
            }
          } catch {
            // ignore partial chunk parse
          }
        }
      }
    }

    const totalLatencyMs = performance.now() - startTime;
    const finalTranslation = cleanTranslationResponse(translation);

    if (promptTokens === 0) promptTokens = Math.ceil((systemPrompt.length + userPrompt.length) / 4);
    if (completionTokens === 0) completionTokens = Math.ceil(finalTranslation.length / 4);

    const inputRate = pricing.inputPerM;
    const outputRate = pricing.outputPerM;
    const uncachedCost = (promptTokens * inputRate + completionTokens * outputRate) / 1_000_000;

    const cachedTokens = isFewShotTier ? Math.floor(promptTokens * 0.6) : 0;
    const cachedReadRate = pricing.cachedReadPerM ?? inputRate * 0.10;
    const cachedCost = (
      (promptTokens - cachedTokens) * inputRate +
      cachedTokens * cachedReadRate +
      completionTokens * outputRate
    ) / 1_000_000;

    return {
      translation: finalTranslation,
      ttft_ms: ttftMs ?? totalLatencyMs,
      total_latency_ms: totalLatencyMs,
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      char_count: finalTranslation.length,
      uncached_cost_usd: uncachedCost,
      cached_cost_usd: cachedCost,
    };
  }

  async runBatch(
    batchModelSlug: string,
    requests: TranslationRequest[],
    pricing: ModelPricing
  ): Promise<Map<string, TelemetryEnvelope>> {
    const startTime = performance.now();
    const results = new Map<string, TelemetryEnvelope>();

    const batchRequests = requests.map((req) => {
      assertModelAllowedForLanguage(req.lang, batchModelSlug);
      const systemPrompt = buildSystemPrompt(req.tier);
      const userPrompt = buildPromptText(req);
      const isFewShotTier = req.tier.startsWith('tier3');
      const messages = normalizeMessagesForModel(batchModelSlug, systemPrompt, userPrompt, isFewShotTier);
      const sessionId = buildDeterministicSessionId({
        lang: req.lang,
        register_tier: req.coarse?.formality,
        regional_dialect: req.coarse?.dialect,
        social_relationship: req.granular?.relationship,
      });

      return {
        custom_id: req.request_id,
        method: 'POST',
        url: '/v1/chat/completions',
        body: {
          messages,
          max_tokens: 1024,
          session_id: sessionId,
          reasoning: { effort: 'none' },
        },
      };
    });

    const submitRes = await fetch('https://openrouter.ai/api/v1/batches', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        endpoint: '/v1/chat/completions',
        model: batchModelSlug,
        requests: batchRequests,
      }),
    });

    if (!submitRes.ok) {
      throw new Error(`OpenRouter Batch submit failed: HTTP ${submitRes.status}`);
    }

    const batchData: any = await submitRes.json();
    const batchId = batchData.id;

    const maxWaitMs = 600_000;
    const pollStart = Date.now();
    let status = batchData.status;

    while (status !== 'completed') {
      if (status === 'failed' || status === 'cancelled') {
        throw new Error(`OpenRouter Batch ${batchId} failed with status ${status}`);
      }
      if (Date.now() - pollStart > maxWaitMs) {
        const baseSlug = batchModelSlug.replace(/:batch$/, '');
        let currentIndex = 0;
        const poolSize = Math.min(10, requests.length);
        const workers = Array.from({ length: poolSize }, async () => {
          while (currentIndex < requests.length) {
            const req = requests[currentIndex++];
            try {
              const env = await this.translate(baseSlug, req, pricing, false);
              results.set(req.request_id, env);
            } catch (err: any) {
              console.error(`Batch fallback error on ${req.request_id}:`, err?.message);
            }
          }
        });
        await Promise.all(workers);
        return results;
      }
      await new Promise((r) => setTimeout(r, 4000));

      const pollRes = await fetch(`https://openrouter.ai/api/v1/batches/${batchId}`, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      if (pollRes.ok) {
        const polled: any = await pollRes.json();
        status = polled.status;
        if (polled.results && Array.isArray(polled.results)) {
          const totalLatency = performance.now() - startTime;
          polled.results.forEach((item: any) => {
            const reqId = item.custom_id;
            // Strict deduplication by custom_id
            if (results.has(reqId)) return;
            const text = cleanTranslationResponse(
              item?.response?.body?.choices?.[0]?.message?.content || ''
            );
            const pTokens = item?.response?.body?.usage?.prompt_tokens || 60;
            const cTokens = item?.response?.body?.usage?.completion_tokens || 20;
            const cost = (pTokens * pricing.inputPerM + cTokens * pricing.outputPerM) / 1_000_000;

            results.set(reqId, {
              translation: text,
              ttft_ms: null,
              total_latency_ms: totalLatency / requests.length,
              prompt_tokens: pTokens,
              completion_tokens: cTokens,
              char_count: text.length,
              uncached_cost_usd: cost,
              cached_cost_usd: cost,
            });
          });
          return results;
        }
      }
    }

    return results;
  }
}
