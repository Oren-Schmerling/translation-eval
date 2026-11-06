import * as fs from 'fs';
import * as path from 'path';
import { spawnSync } from 'child_process';
import { probeAndResolveRoster } from './discovery/model-prober';
import { GoogleAiStudioRunner } from './providers/google-ai-studio';
import { GoogleBatchRunner } from './providers/google-batch';
import { OpenRouterRunner } from './providers/openrouter';
import { OpenRouterFreeQueue } from './providers/openrouter-free-queue';
import { buildRequestsForSuite, loadDataset } from './suite';
import { ProbedModel, ProbedRoster, RunRecord, SuiteType, TelemetryEnvelope } from './types';

function parseArgs() {
  const args = process.argv.slice(2);
  let probeOnly = false;
  let preferBatch = false;
  let withJudges = false;
  let judgesConfig: string | undefined;
  let ceilingModel: string | undefined;
  let suite: SuiteType = 'smoke';

  for (const arg of args) {
    if (arg === '--probe-only') probeOnly = true;
    else if (arg === '--prefer-batch') preferBatch = true;
    else if (arg === '--with-judges' || arg === '--judges') withJudges = true;
    else if (arg.startsWith('--judges-config=')) judgesConfig = arg.slice('--judges-config='.length);
    else if (arg.startsWith('--ceiling-model=')) ceilingModel = arg.slice('--ceiling-model='.length);
    else if (arg.startsWith('--suite=')) suite = arg.slice('--suite='.length) as SuiteType;
  }
  return { probeOnly, preferBatch, withJudges, judgesConfig, ceilingModel, suite };
}

function printRoutingTable(roster: ProbedRoster) {
  console.log('\n========================================================================================');
  console.log('                          PRAGMATIC MT BENCHMARK ROUTING TABLE                          ');
  console.log('========================================================================================');

  console.log('\n[A] Google AI Studio Models (Direct Execution)');
  if (roster.ai_studio.unlocked_batch.length === 0 && roster.ai_studio.unlocked_stream.length === 0) {
    console.log('  (none unlocked on current plan)');
  }
  for (const m of roster.ai_studio.unlocked_batch) console.log(`  - ${m} [AI_STUDIO_BATCH] (50% discount)`);
  for (const m of roster.ai_studio.unlocked_stream) console.log(`  - ${m} [AI_STUDIO_STREAM] (15 RPM rate-limited)`);

  console.log('\n[B] Google Models Falling Back to OpenRouter');
  if (roster.openrouter.fallbacks.length === 0) {
    console.log('  (none routed)');
  }
  for (const m of roster.openrouter.fallbacks) console.log(`  - ${m}`);

  console.log('\n[C] Single Expensive Ceiling Model (Max 1 Budget Cap)');
  if (roster.openrouter.ceiling_model) {
    console.log(`  - ${roster.openrouter.ceiling_model}`);
  } else {
    console.log('  (none selected)');
  }

  console.log('\n[D1] High-Promise Low-Cost Contenders (Prefer :batch)');
  if (roster.openrouter.contenders.length === 0) {
    console.log('  (none)');
  }
  for (const m of roster.openrouter.contenders) console.log(`  - ${m}`);

  console.log('\n[D2] Bucket 3B — Dedicated MT Specialists (tencent/hy-mt2-*)');
  if (!roster.openrouter.mt_specialists || roster.openrouter.mt_specialists.length === 0) {
    console.log('  (none)');
  } else {
    for (const m of roster.openrouter.mt_specialists) console.log(`  - ${m}`);
  }

  console.log('\n[E] Verified OpenRouter :free Models (3.15s Pacer Queue)');
  if (roster.openrouter.free_models.length === 0) {
    console.log('  (none passing)');
  }
  for (const m of roster.openrouter.free_models) console.log(`  - ${m}`);

  console.log('\n========================================================================================');
  console.log(`Total Active Models: ${roster.active_roster.length}`);
  console.log('========================================================================================\n');
}

function createRunRecord(
  model: ProbedModel,
  req: any,
  envelope: TelemetryEnvelope
): RunRecord {
  return {
    provider_id: model.id,
    tier: req.tier,
    request_id: req.request_id,
    item_id: req.item_id,
    variant_id: req.variant_id,
    source_en: req.source_en,
    lang: req.lang,
    coarse: req.coarse,
    granular: req.granular,
    conflict_override: req.conflict_override,
    track: 'culture_and_daily_life',
    bucket: 'A',
    is_cued_source: false,
    gold_reference: req.gold_reference,
    translation: envelope.translation,
    anchors: req.anchors,
    capability: { status: 'supported' },
    ttft_ms: envelope.ttft_ms,
    total_latency_ms: envelope.total_latency_ms,
    prompt_tokens: envelope.prompt_tokens,
    completion_tokens: envelope.completion_tokens,
    char_count: envelope.char_count,
    uncached_cost_usd: envelope.uncached_cost_usd,
    cached_cost_usd: envelope.cached_cost_usd,
    layer1: null,
    chrf: null,
    ensemble: null,
    bias_profile: null,
    high_disagreement_flag: false,
  };
}

async function main() {
  const { probeOnly, preferBatch, withJudges, judgesConfig, ceilingModel, suite } = parseArgs();
  console.log(`Running translation-eval [suite=${suite}, preferBatch=${preferBatch}]...`);

  const geminiKey = process.env.GEMINI_API_KEY || process.env.GOOGLE_AI_STUDIO_API_KEY || '';
  const orKey = process.env.OPENROUTER_API_KEY || '';

  const roster = await probeAndResolveRoster({
    geminiApiKey: geminiKey,
    openrouterApiKey: orKey,
    ceilingModelOverride: ceilingModel,
    preferBatch,
  });

  // Write probed-roster.json
  const rosterPath = path.join(process.cwd(), 'probed-roster.json');
  fs.writeFileSync(rosterPath, JSON.stringify(roster, null, 2), 'utf8');
  console.log(`Wrote probed roster to ${rosterPath}`);

  printRoutingTable(roster);
  if (probeOnly) return;

  const dataset = loadDataset();
  const resultsDir = path.join(process.cwd(), 'results');
  if (!fs.existsSync(resultsDir)) fs.mkdirSync(resultsDir, { recursive: true });

  const sweepOutputFile = path.join(resultsDir, `sweep_${suite}.jsonl`);
  fs.writeFileSync(sweepOutputFile, '', 'utf8');
  const freeQueue = new OpenRouterFreeQueue({ apiKey: orKey, suite });
  const orRunner = new OpenRouterRunner({ apiKey: orKey });
  const aiStudioRunner = new GoogleAiStudioRunner({ apiKey: geminiKey });
  const aiStudioBatchRunner = new GoogleBatchRunner({ apiKey: geminiKey });

  let totalItemsCompleted = 0;

  for (const model of roster.active_roster) {
    console.log(`\n>>> Executing model: ${model.id} [mode=${model.mode}]`);
    const requests = buildRequestsForSuite(dataset, model.id, suite);
    const existingCheckpoints = freeQueue.loadExistingCheckpoints(model.provider, model.modelSlug);

    for (const r of requests) {
      const key = `${model.id}::${r.tier}::${r.variant_id}`;
      const rec = existingCheckpoints.get(key);
      if (rec) {
        fs.appendFileSync(sweepOutputFile, JSON.stringify(rec) + '\n', 'utf8');
        totalItemsCompleted++;
      }
    }

    const pendingRequests = requests.filter((r) => {
      const key = `${model.id}::${r.tier}::${r.variant_id}`;
      return !existingCheckpoints.has(key);
    });

    if (pendingRequests.length === 0) {
      console.log(`  All ${requests.length} items already completed in checkpoints.`);
      continue;
    }

    console.log(`  Processing ${pendingRequests.length}/${requests.length} items...`);

    if (model.mode === 'batch' && model.provider === 'openrouter') {
      try {
        const batchMap = await orRunner.runBatch(model.modelSlug, pendingRequests, model.pricing);
        for (const req of pendingRequests) {
          const env = batchMap.get(req.request_id);
          if (env) {
            const rec = createRunRecord(model, req, env);
            freeQueue.appendCheckpoint(model.provider, model.modelSlug, rec);
            fs.appendFileSync(sweepOutputFile, JSON.stringify(rec) + '\n', 'utf8');
            totalItemsCompleted++;
          }
        }
      } catch (err: any) {
        console.error(`  Batch failed for ${model.id}: ${err.message}. Falling back to standard.`);
        for (const req of pendingRequests) {
          const env = await orRunner.translate(model.modelSlug.replace(/:batch$/, ''), req, model.pricing);
          const rec = createRunRecord(model, req, env);
          freeQueue.appendCheckpoint(model.provider, model.modelSlug, rec);
          fs.appendFileSync(sweepOutputFile, JSON.stringify(rec) + '\n', 'utf8');
          totalItemsCompleted++;
        }
      }
    } else if (model.mode === 'batch' && model.provider === 'google-ai-studio') {
      try {
        const batchMap = await aiStudioBatchRunner.runBatch(model.modelSlug, pendingRequests);
        for (const req of pendingRequests) {
          const env = batchMap.get(req.request_id);
          if (env) {
            const rec = createRunRecord(model, req, env);
            freeQueue.appendCheckpoint(model.provider, model.modelSlug, rec);
            fs.appendFileSync(sweepOutputFile, JSON.stringify(rec) + '\n', 'utf8');
            totalItemsCompleted++;
          }
        }
      } catch (err: any) {
        console.error(`  AI Studio batch failed: ${err.message}. Falling back to streaming.`);
        for (const req of pendingRequests) {
          const env = await aiStudioRunner.translate(model.modelSlug, req);
          const rec = createRunRecord(model, req, env);
          freeQueue.appendCheckpoint(model.provider, model.modelSlug, rec);
          fs.appendFileSync(sweepOutputFile, JSON.stringify(rec) + '\n', 'utf8');
          totalItemsCompleted++;
        }
      }
    } else {
      // Item by item execution (Stream or Free Queue)
      for (const req of pendingRequests) {
        try {
          let env: TelemetryEnvelope;
          if (model.provider === 'google-ai-studio') {
            try {
              env = await aiStudioRunner.translate(model.modelSlug, req);
            } catch (err: any) {
              if (err.message.includes('DAILY_QUOTA_EXHAUSTED')) {
                console.warn(`  AI Studio daily quota reached for ${model.modelSlug}. Falling back to OpenRouter.`);
                const orSlug = `google/${model.modelSlug}`;
                env = await orRunner.translate(orSlug, req, { inputPerM: 0.15, outputPerM: 0.60 });
              } else {
                throw err;
              }
            }
          } else if (model.mode === 'free_queue') {
            env = await freeQueue.translate(model.modelSlug, req);
          } else {
            env = await orRunner.translate(model.modelSlug, req, model.pricing);
          }

          const rec = createRunRecord(model, req, env);
          freeQueue.appendCheckpoint(model.provider, model.modelSlug, rec);
          fs.appendFileSync(sweepOutputFile, JSON.stringify(rec) + '\n', 'utf8');
          totalItemsCompleted++;
          process.stdout.write('.');
        } catch (itemErr: any) {
          console.error(`\n  Error on ${req.request_id}: ${itemErr.message}`);
        }
      }
      console.log(' done.');
    }
  }

  console.log(`\nCompleted sweep! Wrote ${totalItemsCompleted} records to ${sweepOutputFile}`);

  // Trigger grading & report via Rust
  const gradedFile = path.join(resultsDir, `sweep_${suite}_graded.jsonl`);
  const reportFile = path.join(resultsDir, `sweep_${suite}_report.md`);
  const summaryFile = path.join(resultsDir, `sweep_${suite}_summary.csv`);

  console.log('\nRunning grader and report generator...');
  const gradeArgs = [
    'run', '--', 'grade',
    '--input', sweepOutputFile,
    '--output', gradedFile,
  ];

  if (withJudges || judgesConfig) {
    let configPath = judgesConfig;
    if (!configPath) {
      configPath = path.join(resultsDir, `judges_${suite}.json`);
      const judgePayload = {
        judge_a: {
          base_url: 'https://openrouter.ai/api/v1',
          api_key: orKey,
          model: 'anthropic/claude-haiku-5.5',
        },
        judge_b: {
          base_url: 'https://openrouter.ai/api/v1',
          api_key: orKey,
          model: 'deepseek/deepseek-v4.1-flash',
        },
      };
      fs.writeFileSync(configPath, JSON.stringify(judgePayload, null, 2), 'utf8');
    }
    gradeArgs.push('--judges-config', configPath);
  }

  const gradeRes = spawnSync('cargo', gradeArgs, { stdio: 'inherit' });

  if (gradeRes.status === 0) {
    spawnSync('cargo', [
      'run', '--', 'report',
      '--input', gradedFile,
      '--markdown', reportFile,
      '--csv', summaryFile,
    ], { stdio: 'inherit' });
    console.log(`\nMarkdown report generated at: ${reportFile}`);
    console.log(`CSV summary generated at: ${summaryFile}`);
  }
}

if (import.meta.main || process.argv[1]?.endsWith('cli.ts')) {
  main().catch((e) => {
    console.error('Fatal CLI error:', e);
    process.exit(1);
  });
}
