import { describe, expect, it } from 'bun:test';
import * as fs from 'fs';
import * as path from 'path';
import {
  assertModelAllowedForLanguage,
  BlockedModelForLanguageError,
  getModelFallbackChain,
  normalizeMessagesForModel,
  requiresPragmaticSteering,
} from '../src/routing/model-router';
import { deduplicateRecords, makeCompositeKey } from '../src/reporting/report-generator';
import { makeCheckpointKey, OpenRouterFreeQueue } from '../src/providers/openrouter-free-queue';
import { RunRecord } from '../src/types';

describe('Checkpoint 2.1 Verification Suite', () => {
  // Checkpoint 2.1.1: 3x duplicated :batch mock outputs yield exactly 1 persisted record
  it('deduplicates 3x identical batch mock records in writer and aggregator', () => {
    const testDir = path.join(process.cwd(), 'results', 'test_dedup_checkpoints');
    if (fs.existsSync(testDir)) fs.rmSync(testDir, { recursive: true, force: true });

    const queue = new OpenRouterFreeQueue({
      apiKey: 'test-key',
      suite: 'test_suite',
      checkpointDir: testDir,
    });

    const mockRecord: RunRecord = {
      provider_id: 'openrouter:deepseek/deepseek-v4.1-flash:batch',
      tier: 'tier1_default_zero_shot',
      request_id: 'deepseek-v4.1-flash-batch-tier1-v2_gu_cas_0001',
      item_id: 'v2_gu_cas_0001',
      variant_id: 'v2_gu_cas_0001_v1',
      source_en: 'Where are you going?',
      lang: 'gu',
      coarse: { formality: 'default', speaker_gender: null, listener_gender: null, dialect: null },
      granular: { pronoun_tier: null, kinship_pair: null, relationship: null },
      conflict_override: null,
      track: 'casual_conversational',
      bucket: 'A',
      is_cued_source: false,
      gold_reference: 'તમે ક્યાં જઈ રહ્યા છો?',
      translation: 'તમે ક્યાં જઈ રહ્યા છો?',
      anchors: { must_contain_any: [], must_not_contain: [], synset_equivalents: {}, acceptable_synonyms: [] },
      capability: { status: 'supported' },
      ttft_ms: null,
      total_latency_ms: 120,
      prompt_tokens: 30,
      completion_tokens: 10,
      char_count: 22,
      uncached_cost_usd: 0.0001,
      cached_cost_usd: 0.0001,
      layer1: null,
      chrf: null,
      ensemble: null,
      bias_profile: null,
      high_disagreement_flag: false,
    };

    // Simulate 3x duplicated persistence calls
    const res1 = queue.appendCheckpoint('openrouter', 'deepseek/deepseek-v4.1-flash:batch', mockRecord);
    const res2 = queue.appendCheckpoint('openrouter', 'deepseek/deepseek-v4.1-flash:batch', mockRecord);
    const res3 = queue.appendCheckpoint('openrouter', 'deepseek/deepseek-v4.1-flash:batch', mockRecord);

    expect(res1).toBe(true);
    expect(res2).toBe(false); // Skipped as duplicate
    expect(res3).toBe(false); // Skipped as duplicate

    // Verify loaded checkpoints has exactly 1 record
    const loaded = queue.loadExistingCheckpoints('openrouter', 'deepseek/deepseek-v4.1-flash:batch');
    expect(loaded.size).toBe(1);

    // Verify aggregator deduplication on a 3x duplicated legacy array
    const legacyTriplicated = [mockRecord, { ...mockRecord }, { ...mockRecord }];
    const { unique, duplicatesCount } = deduplicateRecords(legacyTriplicated);
    expect(unique.length).toBe(1);
    expect(duplicatesCount).toBe(2);

    fs.rmSync(testDir, { recursive: true, force: true });
  });

  // Checkpoint 2.1.2: BLOCKED_MODELS_BY_LANG throws on gu + glm, vi + hy-mt2-7b, and * + hy-mt2-1.8b
  it('throws BlockedModelForLanguageError on forbidden cross-language hallucination pairs', () => {
    // gu + glm-5.3-flash
    expect(() => assertModelAllowedForLanguage('gu', 'z-ai/glm-5.3-flash')).toThrow(BlockedModelForLanguageError);
    expect(() => assertModelAllowedForLanguage('gu', 'openrouter:z-ai/glm-5.3-flash:batch')).toThrow(BlockedModelForLanguageError);

    // vi + hy-mt2-7b (and all tencent/hy-mt2-* on vi)
    expect(() => assertModelAllowedForLanguage('vi', 'tencent/hy-mt2-7b')).toThrow(BlockedModelForLanguageError);
    expect(() => assertModelAllowedForLanguage('vi', 'tencent/hy-mt2-30b-a3b')).toThrow(BlockedModelForLanguageError);

    // * + hy-mt2-1.8b (all languages block 1.8b)
    expect(() => assertModelAllowedForLanguage('he', 'tencent/hy-mt2-1.8b')).toThrow(BlockedModelForLanguageError);
    expect(() => assertModelAllowedForLanguage('gu', 'tencent/hy-mt2-1.8b')).toThrow(BlockedModelForLanguageError);
    expect(() => assertModelAllowedForLanguage('vi', 'tencent/hy-mt2-1.8b')).toThrow(BlockedModelForLanguageError);

    // Allowed pairs must NOT throw
    expect(() => assertModelAllowedForLanguage('he', 'deepseek/deepseek-v4.1-flash')).not.toThrow();
    expect(() => assertModelAllowedForLanguage('gu', 'tencent/hy-mt2-7b')).not.toThrow();
    expect(() => assertModelAllowedForLanguage('vi', 'deepseek/deepseek-v4.1-flash')).not.toThrow();
  });

  // Checkpoint 2.1.3: Fallback routing for cued vs default Gujarati
  it('routes cued Gujarati to deepseek first and default Gujarati to hy-mt2-7b first', () => {
    // Cued Gujarati: listener_gender female or social relationship
    const cuedParams = {
      lang: 'gu',
      listener_gender: 'female',
    };
    expect(requiresPragmaticSteering(cuedParams)).toBe(true);
    const cuedChain = getModelFallbackChain(cuedParams);
    expect(cuedChain[0]).toBe('deepseek/deepseek-v4.1-flash');
    expect(cuedChain).toContain('tencent/hy-mt2-7b');
    // Ensure blocked glm is NOT present
    expect(cuedChain).not.toContain('z-ai/glm-5.3-flash');
    expect(cuedChain).not.toContain('tencent/hy-mt2-1.8b');

    // Default Gujarati (no pragmatic cues)
    const defaultParams = {
      lang: 'gu',
      listener_gender: null,
      speaker_gender: null,
      register_tier: 'default',
    };
    expect(requiresPragmaticSteering(defaultParams)).toBe(false);
    const defaultChain = getModelFallbackChain(defaultParams);
    expect(defaultChain[0]).toBe('tencent/hy-mt2-7b');
    expect(defaultChain[1]).toBe('deepseek/deepseek-v4.1-flash');
  });

  // Checkpoint 2.1.4: tencent/hy-mt2-* request normalizer folds systemPrompt into user message
  it('normalizes tencent/hy-mt2-* requests folding system prompt into user turn', () => {
    const sysPrompt = 'Output ONLY the translation.';
    const userPrompt = 'Good morning';

    const tencentMessages = normalizeMessagesForModel('tencent/hy-mt2-7b', sysPrompt, userPrompt);
    expect(tencentMessages.length).toBe(1);
    expect(tencentMessages[0].role).toBe('user');
    expect(tencentMessages[0].content).toBe(`${sysPrompt}\n\n${userPrompt}`);
    expect(tencentMessages.some((m) => m.role === 'system')).toBe(false);

    // Standard models retain system role
    const standardMessages = normalizeMessagesForModel('deepseek/deepseek-v4.1-flash', sysPrompt, userPrompt);
    expect(standardMessages.length).toBe(2);
    expect(standardMessages[0].role).toBe('system');
    expect(standardMessages[1].role).toBe('user');
  });

  // Pairwise judge presentation order & Bradley-Terry leaderboard
  it('correctly handles deterministic pairwise candidate presentation and unswapping', () => {
    const { isPresentationSwapped, parseJudgeDecision, computePairwiseLeaderboard } = require('../src/reporting/pairwise-judge');
    
    // Deterministic presentation swap check
    const swapped = isPresentationSwapped('item_01');
    expect(typeof swapped).toBe('boolean');

    // Winner unswap logic
    expect(parseJudgeDecision('{"winner": "A"}', false)).toBe('A');
    expect(parseJudgeDecision('{"winner": "A"}', true)).toBe('B');
    expect(parseJudgeDecision('{"winner": "B"}', false)).toBe('B');
    expect(parseJudgeDecision('{"winner": "B"}', true)).toBe('A');
    expect(parseJudgeDecision('{"winner": "tie"}', true)).toBe('tie');

    // Leaderboard calculation
    const leaderboard = computePairwiseLeaderboard([
      { item_id: '1', source_en: '', reference_target: '', model_a: 'M1', model_b: 'M2', translation_a: '', translation_b: '', presentation_order: 'standard', winner: 'A' },
      { item_id: '2', source_en: '', reference_target: '', model_a: 'M1', model_b: 'M2', translation_a: '', translation_b: '', presentation_order: 'standard', winner: 'A' },
      { item_id: '3', source_en: '', reference_target: '', model_a: 'M1', model_b: 'M2', translation_a: '', translation_b: '', presentation_order: 'standard', winner: 'B' },
    ]);
    expect(leaderboard[0].model).toBe('M1');
    expect(leaderboard[0].wins).toBe(2);
    expect(leaderboard[1].model).toBe('M2');
    expect(leaderboard[1].wins).toBe(1);
  });

  // Checkpoint 2.1.5: Verify all modified/new source files are < 500 lines
  it('verifies all modified and new source files stay strictly under 500 lines', () => {
    const filesToCheck = [
      'src/cli.ts',
      'src/routing/model-router.ts',
      'src/orchestration/repair-orchestrator.ts',
      'src/providers/openrouter.ts',
      'src/providers/openrouter-free-queue.ts',
      'src/reporting/report-generator.ts',
      'src/reporting/pairwise-judge.ts',
      'src/discovery/model-prober.ts',
      'src/suite.ts',
    ];

    for (const f of filesToCheck) {
      const p = path.join(process.cwd(), f);
      const lines = fs.readFileSync(p, 'utf8').split('\n').length;
      expect(lines).toBeLessThan(500);
    }
  });
});
