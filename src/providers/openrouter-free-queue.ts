import * as fs from 'fs';
import * as path from 'path';
import { buildPromptText, buildSystemPrompt, cleanTranslationResponse } from '../prompt';
import { RunRecord, TelemetryEnvelope, TranslationRequest } from '../types';

export interface FreeQueueOptions {
  apiKey: string;
  suite: string;
  minIntervalMs?: number; // default 3150ms
  checkpointDir?: string; // default results/checkpoints
}

export class OpenRouterFreeQueue {
  private apiKey: string;
  private suite: string;
  private minIntervalMs: number;
  private checkpointDir: string;
  private lastRequestTime = 0;
  private dailyRequestCount = 0;
  private readonly maxRpd = 1000;

  constructor(opts: FreeQueueOptions) {
    this.apiKey = opts.apiKey;
    this.suite = opts.suite;
    this.minIntervalMs = opts.minIntervalMs ?? 3150;
    this.checkpointDir = opts.checkpointDir ?? path.join(process.cwd(), 'results', 'checkpoints');

    if (!fs.existsSync(this.checkpointDir)) {
      fs.mkdirSync(this.checkpointDir, { recursive: true });
    }
  }

  getCheckpointPath(providerId: string, model: string): string {
    const cleanProvider = providerId.replace(/[^a-zA-Z0-9_-]/g, '_');
    const cleanModel = model.replace(/[^a-zA-Z0-9_-]/g, '_');
    return path.join(this.checkpointDir, `${cleanProvider}_${cleanModel}_${this.suite}.jsonl`);
  }

  loadExistingCheckpoints(providerId: string, model: string): Map<string, RunRecord> {
    const filePath = this.getCheckpointPath(providerId, model);
    const existing = new Map<string, RunRecord>();
    if (!fs.existsSync(filePath)) return existing;

    const lines = fs.readFileSync(filePath, 'utf8').split('\n');
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        const record = JSON.parse(trimmed) as RunRecord;
        const key = `${record.provider_id}::${record.tier}::${record.variant_id}`;
        existing.set(key, record);
      } catch {
        // ignore partial line
      }
    }
    return existing;
  }

  appendCheckpoint(providerId: string, model: string, record: RunRecord): void {
    const filePath = this.getCheckpointPath(providerId, model);
    fs.appendFileSync(filePath, JSON.stringify(record) + '\n', 'utf8');
  }

  private async pace(): Promise<void> {
    if (this.dailyRequestCount >= this.maxRpd) {
      throw new Error(`DAILY_RPD_LIMIT_REACHED: Exceeded ${this.maxRpd} free requests per day`);
    }

    const now = Date.now();
    const elapsed = now - this.lastRequestTime;
    if (elapsed < this.minIntervalMs) {
      await new Promise((r) => setTimeout(r, this.minIntervalMs - elapsed));
    }
    this.lastRequestTime = Date.now();
    this.dailyRequestCount++;
  }

  async translate(model: string, req: TranslationRequest): Promise<TelemetryEnvelope> {
    let backoffMs = 3500;
    const maxRetries = 3;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      await this.pace();
      const startTime = performance.now();
      const systemPrompt = buildSystemPrompt(req.tier);
      const userPrompt = buildPromptText(req);

      const makeBody = (withReasoning: boolean) => ({
        model,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt },
        ],
        max_tokens: 1024,
        ...(withReasoning ? { reasoning: { effort: 'none' } } : {}),
      });

      try {
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

        if (resp.status === 404) {
          const errText = await resp.text();
          if (errText.includes('use this slug instead:')) {
            const redirectSlug = errText.split('use this slug instead:')[1]?.trim()?.split(/[\s,"']/)[0];
            if (redirectSlug) {
              resp = await fetch('https://openrouter.ai/api/v1/chat/completions', {
                method: 'POST',
                headers: { Authorization: `Bearer ${this.apiKey}`, 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  model: redirectSlug,
                  messages: [
                    { role: 'system', content: systemPrompt },
                    { role: 'user', content: userPrompt },
                  ],
                  max_tokens: 1024,
                  reasoning: { effort: 'none' },
                }),
              });
            }
          }
        }

        if (resp.status === 429 || resp.status === 503) {
          if (attempt === maxRetries) {
            const errText = await resp.text();
            throw new Error(`OpenRouter Free Queue rate limit exhausted: ${errText}`);
          }
          await new Promise((r) => setTimeout(r, backoffMs));
          backoffMs *= 2;
          continue;
        }

        if (!resp.ok) {
          const errText = await resp.text();
          throw new Error(`OpenRouter Free Queue HTTP ${resp.status}: ${errText}`);
        }

        const data: any = await resp.json();
        const totalLatencyMs = performance.now() - startTime;
        const rawContent = data?.choices?.[0]?.message?.content || '';
        const translation = cleanTranslationResponse(rawContent);

        const promptTokens = data?.usage?.prompt_tokens
          || Math.ceil((systemPrompt.length + userPrompt.length) / 4);
        const completionTokens = data?.usage?.completion_tokens
          || Math.ceil(translation.length / 4);

        return {
          translation,
          ttft_ms: totalLatencyMs,
          total_latency_ms: totalLatencyMs,
          prompt_tokens: promptTokens,
          completion_tokens: completionTokens,
          char_count: translation.length,
          uncached_cost_usd: 0,
          cached_cost_usd: 0,
        };
      } catch (err: any) {
        if (attempt === maxRetries) throw err;
        await new Promise((r) => setTimeout(r, backoffMs));
        backoffMs *= 2;
      }
    }

    throw new Error('Failed after retries');
  }
}
