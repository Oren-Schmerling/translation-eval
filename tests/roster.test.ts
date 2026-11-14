import { describe, expect, it, beforeEach, afterEach } from 'bun:test';
import * as fs from 'fs';
import * as path from 'path';
import { isExpensiveModel, matchesScript, probeAndResolveRoster } from '../src/discovery/model-prober';
import { OpenRouterFreeQueue } from '../src/providers/openrouter-free-queue';
import { RunRecord, TranslationRequest } from '../src/types';

describe('Pragmatic MT Prober & Roster Resolver', () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  // Test 1: Probe and resolve models exclusively from OpenRouter
  it('resolves models exclusively from OpenRouter without Google AI Studio', async () => {
    let googleEndpointCalled = false;
    globalThis.fetch = async (url: string | URL | Request) => {
      const u = url.toString();
      if (u.includes('googleapis.com')) {
        googleEndpointCalled = true;
        return new Response('{}', { status: 200 });
      }
      if (u.includes('openrouter.ai/api/v1/models')) {
        return new Response(JSON.stringify({
          data: [
            { id: 'anthropic/claude-sonnet-5.5', pricing: { prompt: '0.000002', completion: '0.00001' } },
            { id: 'qwen/qwen3.8-flash', pricing: { prompt: '0.00000006', completion: '0.00000024' } },
            { id: 'minimax/minimax-m3.1-flash', pricing: { prompt: '0.0000001', completion: '0.0000004' } },
          ],
        }), { status: 200 });
      }
      return new Response('Not Found', { status: 404 });
    };

    const roster = await probeAndResolveRoster({
      openrouterApiKey: 'test-or-key',
      skipMicroProbe: true,
    });

    expect(googleEndpointCalled).toBe(false);
    expect(roster.ai_studio).toBeUndefined();
    expect(roster.active_roster.length).toBeGreaterThan(0);
    expect(roster.active_roster.every((m) => m.provider === 'openrouter')).toBe(true);
    expect(roster.openrouter.ceiling_model).toBe('anthropic/claude-sonnet-5.5');
    expect(roster.openrouter.contenders).toContain('qwen/qwen3.8-flash');
  });

  // Test 2: Selects batch mode endpoints when preferBatch is requested
  it('selects batch mode endpoints when preferBatch is requested', async () => {
    globalThis.fetch = async (url: string | URL | Request) => {
      const u = url.toString();
      if (u.includes('openrouter.ai/api/v1/models')) {
        return new Response(JSON.stringify({
          data: [
            { id: 'anthropic/claude-sonnet-5.5:batch', pricing: { prompt: '0.000001', completion: '0.000005' } },
            { id: 'anthropic/claude-haiku-5.5:batch', pricing: { prompt: '0.00000025', completion: '0.00000125' } },
          ],
        }), { status: 200 });
      }
      return new Response('Not Found', { status: 404 });
    };

    const roster = await probeAndResolveRoster({
      openrouterApiKey: 'test-or-key',
      preferBatch: true,
      skipMicroProbe: true,
    });

    expect(roster.openrouter.ceiling_model).toBe('anthropic/claude-sonnet-5.5:batch');
    const ceiling = roster.active_roster.find((m) => m.modelSlug === 'anthropic/claude-sonnet-5.5:batch');
    expect(ceiling?.mode).toBe('batch');
    const haiku = roster.active_roster.find((m) => m.modelSlug === 'anthropic/claude-haiku-5.5:batch');
    expect(haiku?.mode).toBe('batch');
  });

  // Test 3: Single Expensive Model Budget Cap blocks more than 1 flagship paid model
  it('enforces single expensive model budget cap (max 1 flagship paid model)', async () => {
    globalThis.fetch = async (url: string | URL | Request) => {
      const u = url.toString();
      if (u.includes('openrouter.ai/api/v1/models')) {
        return new Response(JSON.stringify({
          data: [
            // Both are expensive: in > 0.50, out > 2.50
            { id: 'anthropic/claude-sonnet-5.5', pricing: { prompt: '0.000002', completion: '0.00001' } },
            { id: 'deepseek/deepseek-v4.1-flash-1017', pricing: { prompt: '0.000002', completion: '0.00001' } },
          ],
        }), { status: 200 });
      }
      return new Response('Not Found', { status: 404 });
    };

    const roster = await probeAndResolveRoster({
      openrouterApiKey: 'test-or-key',
      skipMicroProbe: true,
    });

    // Default ceiling model is claude-sonnet-5.5
    expect(roster.openrouter.ceiling_model).toBe('anthropic/claude-sonnet-5.5');

    // Count expensive models in active roster
    const expensiveModels = roster.active_roster.filter((m) => m.isExpensive);
    expect(expensiveModels.length).toBe(1);
    expect(expensiveModels[0].modelSlug).toBe('anthropic/claude-sonnet-5.5');

    // Expensive contender is skipped due to budget cap
    expect(roster.openrouter.contenders).not.toContain('deepseek/deepseek-v4.1-flash-1017');
  });

  // Test 4: :free models obey pacing and checkpoint after each item
  it('enforces free queue pacing interval and immediate checkpoint persistence', async () => {
    const testDir = path.join(process.cwd(), 'results', 'test_checkpoints');
    if (fs.existsSync(testDir)) fs.rmSync(testDir, { recursive: true, force: true });

    let callCount = 0;
    const callTimes: number[] = [];

    globalThis.fetch = async (url: string | URL | Request) => {
      callCount++;
      callTimes.push(Date.now());
      return new Response(JSON.stringify({
        choices: [{ message: { content: 'સવાર સારી' } }],
        usage: { prompt_tokens: 10, completion_tokens: 5 },
      }), { status: 200 });
    };

    // Use a smaller test pacing of 100ms for fast unit test execution
    const queue = new OpenRouterFreeQueue({
      apiKey: 'test-key',
      suite: 'test_suite',
      minIntervalMs: 100,
      checkpointDir: testDir,
    });

    const mockReq: TranslationRequest = {
      request_id: 'test-req-1',
      tier: 'tier1_default_zero_shot',
      item_id: 'test_item_01',
      variant_id: 'test_variant_01',
      source_en: 'Good morning',
      lang: 'gu',
      coarse: { formality: 'default', speaker_gender: null, listener_gender: null, dialect: null },
      granular: { pronoun_tier: null, kinship_pair: null, relationship: null },
      conflict_override: null,
      examples: [],
      gold_reference: 'સવાર સારી',
      anchors: { must_contain_any: [], must_not_contain: [], synset_equivalents: {}, acceptable_synonyms: [] },
    };

    const env = await queue.translate('test/model:free', mockReq);
    expect(env.translation).toBe('સવાર સારી');

    // Verify checkpoint was written
    const rec: RunRecord = {
      provider_id: 'openrouter:test/model:free',
      tier: mockReq.tier,
      request_id: mockReq.request_id,
      item_id: mockReq.item_id,
      variant_id: mockReq.variant_id,
      source_en: mockReq.source_en,
      lang: mockReq.lang,
      coarse: mockReq.coarse,
      granular: mockReq.granular,
      conflict_override: null,
      track: 'test',
      bucket: 'A',
      is_cued_source: false,
      gold_reference: mockReq.gold_reference,
      translation: env.translation,
      anchors: mockReq.anchors,
      capability: { status: 'supported' },
      ttft_ms: env.ttft_ms,
      total_latency_ms: env.total_latency_ms,
      prompt_tokens: env.prompt_tokens,
      completion_tokens: env.completion_tokens,
      char_count: env.char_count,
      uncached_cost_usd: 0,
      cached_cost_usd: 0,
      layer1: null,
      chrf: null,
      ensemble: null,
      bias_profile: null,
      high_disagreement_flag: false,
    };

    queue.appendCheckpoint('openrouter', 'test/model:free', rec);

    // Append a second sibling variant with the same item_id
    const rec2: RunRecord = {
      ...rec,
      request_id: 'test-req-2',
      variant_id: 'test_variant_02',
      lang: 'vi',
      translation: 'Chào buổi sáng',
    };
    queue.appendCheckpoint('openrouter', 'test/model:free', rec2);

    // Verify checkpoint loading preserves both sibling variants
    const loaded = queue.loadExistingCheckpoints('openrouter', 'test/model:free');
    const key1 = `openrouter:test/model:free::${mockReq.tier}::${mockReq.variant_id}`;
    const key2 = `openrouter:test/model:free::${mockReq.tier}::test_variant_02`;
    expect(loaded.has(key1)).toBe(true);
    expect(loaded.has(key2)).toBe(true);
    expect(loaded.get(key1)?.translation).toBe('સવાર સારી');
    expect(loaded.get(key2)?.translation).toBe('Chào buổi sáng');

    // Clean up
    fs.rmSync(testDir, { recursive: true, force: true });
  });

  it('matches target language scripts correctly', () => {
    expect(matchesScript('gu', 'સવાર સારી')).toBe(true);
    expect(matchesScript('gu', 'Good morning')).toBe(false);

    expect(matchesScript('he', 'בוקר טוב')).toBe(true);
    expect(matchesScript('he', 'Good morning')).toBe(false);

    expect(matchesScript('vi', 'Chào buổi sáng')).toBe(true);
    expect(matchesScript('vi', 'Good morning')).toBe(false);
  });

  // Test 6: Bucket 3B Dedicated MT Specialists (tencent/hy-mt2-*) and paid models only
  it('discovers Bucket 3B MT specialists and excludes all free models', async () => {
    globalThis.fetch = async (url: string | URL | Request) => {
      const u = url.toString();
      if (u.includes('generativelanguage.googleapis.com/v1beta/models?')) {
        return new Response(JSON.stringify({ models: [] }), { status: 200 });
      }
      if (u.includes('openrouter.ai/api/v1/models')) {
        return new Response(JSON.stringify({
          data: [
            { id: 'anthropic/claude-sonnet-5.5', pricing: { prompt: '0.000002', completion: '0.00001' } },
            { id: 'tencent/hy-mt2-30b-a3b', pricing: { prompt: '0.000000074', completion: '0.000000295' } },
            { id: 'tencent/hy-mt2-7b', pricing: { prompt: '0.000000074', completion: '0.000000295' } },
            { id: 'tencent/hy-mt2-1.8b', pricing: { prompt: '0.000000044', completion: '0.000000177' } },
            { id: 'tencent/hy3:free', pricing: { prompt: '0', completion: '0' } },
            { id: 'mock/no-paid:free', pricing: { prompt: '0', completion: '0' } },
          ],
        }), { status: 200 });
      }
      return new Response('Not Found', { status: 404 });
    };

    const roster = await probeAndResolveRoster({
      openrouterApiKey: 'test-or-key',
      skipMicroProbe: true,
    });

    expect(roster.openrouter.mt_specialists).toEqual([
      'tencent/hy-mt2-30b-a3b',
      'tencent/hy-mt2-7b',
      'tencent/hy-mt2-1.8b',
    ]);

    const mtModels = roster.active_roster.filter((m) => m.bucket === 'mt_specialist');
    expect(mtModels.length).toBe(3);
    expect(mtModels.map((m) => m.modelSlug)).toEqual([
      'tencent/hy-mt2-30b-a3b',
      'tencent/hy-mt2-7b',
      'tencent/hy-mt2-1.8b',
    ]);

    // Free models are completely excluded; all active models must have non-zero pricing
    expect(roster.active_roster.some((m) => m.modelSlug.endsWith(':free'))).toBe(false);
    expect(roster.active_roster.some((m) => m.bucket === 'free')).toBe(false);
    expect(roster.active_roster.every((m) => m.pricing.inputPerM > 0 && m.pricing.outputPerM > 0)).toBe(true);
    expect(roster.openrouter.free_models).toEqual([]);
  });
});

