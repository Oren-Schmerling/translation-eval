import { buildPromptText, buildSystemPrompt, cleanTranslationResponse } from '../prompt';
import { TelemetryEnvelope, TranslationRequest } from '../types';

export interface GoogleBatchRunnerOptions {
  apiKey: string;
}

export class GoogleBatchRunner {
  private apiKey: string;

  constructor(opts: GoogleBatchRunnerOptions) {
    this.apiKey = opts.apiKey;
  }

  async runBatch(
    model: string,
    requests: TranslationRequest[]
  ): Promise<Map<string, TelemetryEnvelope>> {
    const cleanModel = model.replace(/^models\//, '');
    const startTime = performance.now();
    const results = new Map<string, TelemetryEnvelope>();

    const inlinedRequests = requests.map((req) => {
      const systemInstruction = buildSystemPrompt(req.tier);
      const userPrompt = buildPromptText(req);
      return {
        request: {
          systemInstruction: { parts: [{ text: systemInstruction }] },
          contents: [{ parts: [{ text: userPrompt }] }],
          generationConfig: { maxOutputTokens: 1024, thinkingConfig: { thinkingBudget: 0 } },
        },
      };
    });

    const url = `https://generativelanguage.googleapis.com/v1beta/models/${cleanModel}:batchGenerateContent?key=${this.apiKey}`;
    const submitResp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        batch: {
          displayName: `eval-${Date.now()}`,
          inputConfig: { requests: { requests: inlinedRequests } },
        },
      }),
    });

    if (!submitResp.ok) {
      throw new Error(`Google Batch submission failed: HTTP ${submitResp.status}`);
    }

    const batchData: any = await submitResp.json();
    const batchName = batchData.name; // batches/{id}

    // Poll until completed
    const maxPollTimeMs = 600_000; // 10 minutes
    const pollStart = Date.now();
    let batchState = batchData.state;

    while (batchState !== 'COMPLETED' && batchState !== 'SUCCEEDED') {
      if (batchState === 'FAILED' || batchState === 'CANCELLED') {
        throw new Error(`Google Batch failed with state ${batchState}`);
      }
      if (Date.now() - pollStart > maxPollTimeMs) {
        throw new Error(`Google Batch timed out after ${maxPollTimeMs}ms`);
      }
      await new Promise((r) => setTimeout(r, 5000));

      const pollRes = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/${batchName}?key=${this.apiKey}`
      );
      if (pollRes.ok) {
        const polled: any = await pollRes.json();
        batchState = polled.state;
        if (polled.output?.inlinedResponses) {
          const totalLatency = performance.now() - startTime;
          polled.output.inlinedResponses.forEach((item: any, idx: number) => {
            const req = requests[idx];
            if (!req) return;
            const text = cleanTranslationResponse(
              item?.response?.candidates?.[0]?.content?.parts?.[0]?.text || ''
            );
            const promptTokens = item?.response?.usageMetadata?.promptTokenCount || 60;
            const completionTokens = item?.response?.usageMetadata?.candidatesTokenCount || 20;

            // Batch discount: 50% off standard rates
            const inputRate = 0.075;
            const outputRate = 0.15;
            const uncachedCost = (promptTokens * inputRate + completionTokens * outputRate) / 1_000_000;

            results.set(req.request_id, {
              translation: text,
              ttft_ms: null,
              total_latency_ms: totalLatency / requests.length,
              prompt_tokens: promptTokens,
              completion_tokens: completionTokens,
              char_count: text.length,
              uncached_cost_usd: uncachedCost,
              cached_cost_usd: uncachedCost,
            });
          });
          return results;
        }
      }
    }

    return results;
  }
}
