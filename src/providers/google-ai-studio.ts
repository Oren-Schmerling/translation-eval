import { buildPromptText, buildSystemPrompt, cleanTranslationResponse } from '../prompt';
import { TelemetryEnvelope, TranslationRequest } from '../types';

export interface GoogleAiStudioClientOptions {
  apiKey: string;
  rpmLimit?: number;
  onDailyQuotaExceeded?: (model: string, completedItems: string[]) => Promise<void>;
}

export class GoogleAiStudioRunner {
  private apiKey: string;
  private rpmLimit: number;
  private minIntervalMs: number;
  private lastRequestTime = 0;

  constructor(opts: GoogleAiStudioClientOptions) {
    this.apiKey = opts.apiKey;
    this.rpmLimit = opts.rpmLimit ?? 15;
    this.minIntervalMs = Math.ceil(60_000 / this.rpmLimit);
  }

  private async pace(): Promise<void> {
    const now = Date.now();
    const elapsed = now - this.lastRequestTime;
    if (elapsed < this.minIntervalMs) {
      await new Promise((r) => setTimeout(r, this.minIntervalMs - elapsed));
    }
    this.lastRequestTime = Date.now();
  }

  async translate(model: string, req: TranslationRequest): Promise<TelemetryEnvelope> {
    await this.pace();
    const cleanModel = model.replace(/^models\//, '');
    const startTime = performance.now();
    let ttftMs: number | null = null;

    const systemInstruction = buildSystemPrompt(req.tier);
    const userPrompt = buildPromptText(req);

    const makeBody = (withThinking: boolean) => ({
      systemInstruction: { parts: [{ text: systemInstruction }] },
      contents: [{ parts: [{ text: userPrompt }] }],
      generationConfig: {
        maxOutputTokens: 1024,
        ...(withThinking ? { thinkingConfig: { thinkingBudget: 0 } } : {}),
      },
    });

    // Use streaming endpoint
    const streamUrl = `https://generativelanguage.googleapis.com/v1beta/models/${cleanModel}:streamGenerateContent?alt=sse&key=${this.apiKey}`;
    let resp = await fetch(streamUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(makeBody(true)),
    });

    if (!resp.ok && resp.status === 400) {
      const errText = await resp.clone().text();
      // Retry without thinkingConfig for non-thinking models, or prepend system instruction
      const noThinking = makeBody(false);
      if (/developer instruction/i.test(errText)) {
        resp = await fetch(streamUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            contents: [{ parts: [{ text: `${systemInstruction}\n\n${userPrompt}` }] }],
            generationConfig: { maxOutputTokens: 1024 },
          }),
        });
      } else {
        resp = await fetch(streamUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(noThinking),
        });
      }
    }

    if (!resp.ok) {
      const errText = await resp.text();
      const retryAfterHeader = resp.headers.get('retry-after');
      if (retryAfterHeader) {
        const secs = parseInt(retryAfterHeader, 10);
        if (!isNaN(secs) && secs > 0) {
          this.minIntervalMs = Math.max(this.minIntervalMs, (secs + 1) * 1000);
        }
      }

      if (resp.status === 429 && /quota exceeded.*per day/i.test(errText)) {
        throw new Error(`DAILY_QUOTA_EXHAUSTED: ${errText}`);
      }
      throw new Error(`Google AI Studio HTTP ${resp.status}: ${errText}`);
    }

    // Process SSE stream
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
            const text = parsed?.candidates?.[0]?.content?.parts?.[0]?.text;
            if (text) {
              if (ttftMs === null) {
                ttftMs = performance.now() - startTime;
              }
              translation += text;
            }
            if (parsed?.usageMetadata) {
              promptTokens = parsed.usageMetadata.promptTokenCount || promptTokens;
              completionTokens = parsed.usageMetadata.candidatesTokenCount || completionTokens;
            }
          } catch {
            // ignore partial chunk parse
          }
        }
      }
    }

    const totalLatencyMs = performance.now() - startTime;
    const finalTranslation = cleanTranslationResponse(translation);

    // Default token estimates if not provided by stream
    if (promptTokens === 0) promptTokens = Math.ceil((systemInstruction.length + userPrompt.length) / 4);
    if (completionTokens === 0) completionTokens = Math.ceil(finalTranslation.length / 4);

    const inputRate = 0.15; // default $0.15/M
    const outputRate = 0.60; // default $0.60/M
    const uncachedCost = (promptTokens * inputRate + completionTokens * outputRate) / 1_000_000;
    const cachedTokens = req.tier.startsWith('tier3') ? Math.floor(promptTokens * 0.6) : 0;
    const cachedCost = ((promptTokens - cachedTokens) * inputRate + cachedTokens * (inputRate * 0.25) + completionTokens * outputRate) / 1_000_000;

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
}
