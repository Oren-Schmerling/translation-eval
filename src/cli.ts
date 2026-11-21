import * as fs from 'fs';
import * as path from 'path';
import { spawnSync } from 'child_process';
import { probeAndResolveRoster } from './discovery/model-prober';
import { runOrchestratedTranslation } from './orchestration/repair-orchestrator';
import { OpenRouterRunner } from './providers/openrouter';
import { makeCheckpointKey, OpenRouterFreeQueue } from './providers/openrouter-free-queue';
import { loadAndDeduplicateJSONL, writeReportFiles } from './reporting/report-generator';
import { isModelBlockedForLanguage } from './routing/model-router';
import { buildRequestsForSuite, loadDataset } from './suite';
import { ProbedModel, ProbedRoster, RunRecord, SuiteType, TelemetryEnvelope } from './types';

export async function runConcurrent<T>(
  items: T[],
  concurrency: number,
  worker: (item: T, index: number) => Promise<void>
): Promise<void> {
  let currentIndex = 0;
  const workerCount = Math.min(concurrency, items.length);
  const workers = Array.from({ length: workerCount }, async () => {
    while (currentIndex < items.length) {
      const index = currentIndex++;
      await worker(items[index], index);
    }
  });
  await Promise.all(workers);
}

function resolveJudgeModelSlug(name: string): string {
  const n = name.trim().toLowerCase();
  if (n === 'sonnet 5.5' || n === 'sonnet-5.5' || n === 'claude-sonnet-5.5' || n.includes('claude-sonnet-5.5')) return 'anthropic/claude-sonnet-5.5';
  if (n === 'haiku 5.5' || n === 'haiku-5.5' || n === 'haiku' || n === 'claude-haiku-5.5' || n.includes('claude-haiku-5.5')) return 'anthropic/claude-haiku-5.5';
  if (n === 'qwen3.8 flash' || n === 'qwen3.8-flash' || n.includes('qwen3.8-flash')) return 'qwen/qwen3.8-flash';
  if (n === 'glm 5.3 flash' || n === 'glm-5.3-flash' || n === 'glm' || n.includes('glm-5.3-flash')) return 'z-ai/glm-5.3-flash';
  if (n === 'deepseek 4.1 flash' || n === 'deepseek flash' || n === 'deepseek-4.1-flash' || n === 'deepseek-v4.1-flash' || n.includes('deepseek-v4.1-flash')) return 'deepseek/deepseek-v4.1-flash';
  if (n === 'tencent hy3' || n === 'tencent-hy3' || n === 'hy3' || n.includes('tencent/hy3')) return 'tencent/hy3';
  if (n === 'tencent hy-mt2-30b-a3b' || n.includes('hy-mt2-30b-a3b')) return 'tencent/hy-mt2-30b-a3b';
  return name.trim();
}

function parseArgs() {
  const args = process.argv.slice(2);
  let probeOnly = false;
  let preferBatch = false;
  let withJudges = false;
  let enableRepair = false;
  let includeCeiling = false;
  let mockScorer = false;
  let judgesConfig: string | undefined;
  let judgeModels: string[] | undefined;
  let modelFilter: string[] | undefined;
  let ceilingModel: string | undefined;
  let suite: SuiteType = 'smoke';
  let concurrency = 10;
  let datasetVersion: 'v1' | 'v2' = 'v2';
  let track = 'all';

  for (const arg of args) {
    if (arg === '--probe-only') probeOnly = true;
    else if (arg === '--prefer-batch') preferBatch = true;
    else if (arg === '--with-judges' || arg === '--judges') withJudges = true;
    else if (arg === '--enable-repair') enableRepair = true;
    else if (arg === '--include-ceiling') includeCeiling = true;
    else if (arg === '--mock-scorer' || arg === '--score-metricx') mockScorer = true;
    else if (arg.startsWith('--dataset-version=')) datasetVersion = arg.slice('--dataset-version='.length) as 'v1' | 'v2';
    else if (arg.startsWith('--track=')) track = arg.slice('--track='.length);
    else if (arg.startsWith('--judges=')) {
      withJudges = true;
      judgeModels = arg.slice('--judges='.length).split(',').map((s) => resolveJudgeModelSlug(s.trim())).filter(Boolean);
    }
    else if (arg.startsWith('--models=')) {
      modelFilter = arg.slice('--models='.length).split(',').map((s) => s.trim().toLowerCase()).filter(Boolean);
    }
    else if (arg.startsWith('--concurrency=')) {
      const parsed = parseInt(arg.slice('--concurrency='.length), 10);
      if (!isNaN(parsed) && parsed > 0) concurrency = parsed;
    }
    else if (arg.startsWith('--judges-config=')) judgesConfig = arg.slice('--judges-config='.length);
    else if (arg.startsWith('--ceiling-model=')) ceilingModel = arg.slice('--ceiling-model='.length);
    else if (arg.startsWith('--suite=')) suite = arg.slice('--suite='.length) as SuiteType;
  }
  return {
    probeOnly,
    preferBatch,
    withJudges,
    enableRepair,
    includeCeiling,
    mockScorer,
    judgesConfig,
    judgeModels,
    modelFilter,
    ceilingModel,
    suite,
    concurrency,
    datasetVersion,
    track,
  };
}

function printRoutingTable(roster: ProbedRoster) {
  console.log('\n========================================================================================');
  console.log('                 PRAGMATIC MT BENCHMARK ROUTING TABLE (OPENROUTER ONLY)                 ');
  console.log('========================================================================================');

  console.log('\n[A] Single Expensive Ceiling Model (Max 1 Budget Cap)');
  if (roster.openrouter.ceiling_model) {
    console.log(`  - ${roster.openrouter.ceiling_model}`);
  } else {
    console.log('  (none selected)');
  }

  console.log('\n[B] High-Promise Low-Cost Contenders (Prefer :batch)');
  if (roster.openrouter.contenders.length === 0) {
    console.log('  (none)');
  }
  for (const m of roster.openrouter.contenders) console.log(`  - ${m}`);

  console.log('\n[C] Bucket 3B — Dedicated MT Specialists (tencent/hy-mt2-*)');
  if (!roster.openrouter.mt_specialists || roster.openrouter.mt_specialists.length === 0) {
    console.log('  (none)');
  } else {
    for (const m of roster.openrouter.mt_specialists) console.log(`  - ${m}`);
  }

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
    track: req.track || 'casual_conversational',
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
  const {
    probeOnly,
    preferBatch,
    withJudges,
    enableRepair,
    includeCeiling,
    mockScorer,
    judgesConfig,
    judgeModels,
    modelFilter,
    ceilingModel,
    suite,
    concurrency,
    datasetVersion,
    track,
  } = parseArgs();
  console.log(`Running translation-eval [suite=${suite}, preferBatch=${preferBatch}, repair=${enableRepair}, concurrency=${concurrency}]...`);

  const orKey = process.env.OPENROUTER_API_KEY || '';

  const roster = await probeAndResolveRoster({
    openrouterApiKey: orKey,
    ceilingModelOverride: ceilingModel,
    includeCeiling,
    preferBatch,
  });

  const rosterPath = path.join(process.cwd(), 'probed-roster.json');
  fs.writeFileSync(rosterPath, JSON.stringify(roster, null, 2), 'utf8');
  console.log(`Wrote probed roster to ${rosterPath}`);

  printRoutingTable(roster);
  if (probeOnly) return;

  const dataset = loadDataset(undefined, { datasetVersion, track });
  const resultsDir = path.join(process.cwd(), 'results');
  if (!fs.existsSync(resultsDir)) fs.mkdirSync(resultsDir, { recursive: true });

  const sweepOutputFile = path.join(resultsDir, `sweep_${suite}.jsonl`);
  fs.writeFileSync(sweepOutputFile, '', 'utf8');
  const freeQueue = new OpenRouterFreeQueue({ apiKey: orKey, suite });
  const orRunner = new OpenRouterRunner({ apiKey: orKey });

  let totalItemsCompleted = 0;
  const writtenSweepKeys = new Set<string>();

  const appendSweepRecord = (rec: RunRecord): boolean => {
    const key = makeCheckpointKey(rec.provider_id, rec.tier, rec.item_id, rec.variant_id);
    if (writtenSweepKeys.has(key)) return false;
    writtenSweepKeys.add(key);
    fs.appendFileSync(sweepOutputFile, JSON.stringify(rec) + '\n', 'utf8');
    totalItemsCompleted++;
    return true;
  };

  let activeModels = roster.active_roster;
  if (modelFilter && modelFilter.length > 0) {
    activeModels = roster.active_roster.filter((m) => {
      return modelFilter.some((pat) => {
        return m.id.toLowerCase().includes(pat) || m.modelSlug.toLowerCase().includes(pat);
      });
    });
    console.log(`Filtered to ${activeModels.length} models matching [${modelFilter.join(', ')}]:`);
    for (const m of activeModels) console.log(`  - ${m.id}`);
  }

  for (const model of activeModels) {
    console.log(`\n>>> Executing model: ${model.id} [mode=${model.mode}]`);
    const allRequests = buildRequestsForSuite(dataset, model.id, suite);
    const requests = allRequests.filter((r) => !isModelBlockedForLanguage(r.lang, model.modelSlug));
    const existingCheckpoints = freeQueue.loadExistingCheckpoints(model.provider, model.modelSlug);

    for (const r of requests) {
      const key = makeCheckpointKey(model.id, r.tier, r.item_id, r.variant_id);
      const rec = existingCheckpoints.get(key);
      if (rec) {
        appendSweepRecord(rec);
      }
    }

    const pendingRequests = requests.filter((r) => {
      const key = makeCheckpointKey(model.id, r.tier, r.item_id, r.variant_id);
      return !existingCheckpoints.has(key) && !writtenSweepKeys.has(key);
    });

    if (pendingRequests.length === 0) {
      console.log(`  All ${requests.length} items already completed in checkpoints.`);
      continue;
    }

    console.log(`  Processing ${pendingRequests.length}/${requests.length} items...`);

    if (model.mode === 'batch') {
      try {
        const batchMap = await orRunner.runBatch(model.modelSlug, pendingRequests, model.pricing);
        for (const req of pendingRequests) {
          const env = batchMap.get(req.request_id);
          if (env) {
            const rec = createRunRecord(model, req, env);
            freeQueue.appendCheckpoint(model.provider, model.modelSlug, rec);
            appendSweepRecord(rec);
          }
        }
      } catch (err: any) {
        console.error(`  Batch failed for ${model.id}: ${err.message}. Falling back to standard.`);
        const fallbackSlug = model.modelSlug.replace(/:batch$/, '');
        await runConcurrent(pendingRequests, concurrency, async (req) => {
          try {
            if (enableRepair) {
              const orch = await runOrchestratedTranslation(orRunner, fallbackSlug, req, model.pricing, { enableRepair: true });
              const rec = createRunRecord(model, req, orch.envelope);
              rec.translation = orch.final_translation;
              (rec as any).repair_triggered = orch.repair_triggered;
              (rec as any).repair_succeeded = orch.repair_succeeded;
              (rec as any).first_pass_acr = orch.first_pass_acr;
              (rec as any).post_repair_acr = orch.post_repair_acr;
              (rec as any).layer1_passed = orch.layer1_passed;
              rec.total_latency_ms = orch.total_latency_ms;
              rec.uncached_cost_usd = orch.total_cost_usd;
              freeQueue.appendCheckpoint(model.provider, model.modelSlug, rec);
              appendSweepRecord(rec);
            } else {
              const env = await orRunner.translate(fallbackSlug, req, model.pricing);
              const rec = createRunRecord(model, req, env);
              freeQueue.appendCheckpoint(model.provider, model.modelSlug, rec);
              appendSweepRecord(rec);
            }
            process.stdout.write('.');
          } catch (itemErr: any) {
            console.error(`\n  Error on ${req.request_id}: ${itemErr.message}`);
          }
        });
        console.log(' done.');
      }
    } else {
      await runConcurrent(pendingRequests, concurrency, async (req) => {
        try {
          if (enableRepair) {
            const orch = await runOrchestratedTranslation(orRunner, model.modelSlug, req, model.pricing, { enableRepair: true });
            const rec = createRunRecord(model, req, orch.envelope);
            rec.translation = orch.final_translation;
            (rec as any).repair_triggered = orch.repair_triggered;
            (rec as any).repair_succeeded = orch.repair_succeeded;
            (rec as any).first_pass_acr = orch.first_pass_acr;
            (rec as any).post_repair_acr = orch.post_repair_acr;
            (rec as any).layer1_passed = orch.layer1_passed;
            rec.total_latency_ms = orch.total_latency_ms;
            rec.uncached_cost_usd = orch.total_cost_usd;
            freeQueue.appendCheckpoint(model.provider, model.modelSlug, rec);
            appendSweepRecord(rec);
          } else {
            const env = await orRunner.translate(model.modelSlug, req, model.pricing);
            const rec = createRunRecord(model, req, env);
            freeQueue.appendCheckpoint(model.provider, model.modelSlug, rec);
            appendSweepRecord(rec);
          }
          process.stdout.write('.');
        } catch (itemErr: any) {
          console.error(`\n  Error on ${req.request_id}: ${itemErr.message}`);
        }
      });
      console.log(' done.');
    }
  }

  console.log(`\nCompleted sweep! Wrote ${totalItemsCompleted} records to ${sweepOutputFile}`);

  // Run MetricX scoring if requested
  let fileToGrade = sweepOutputFile;
  if (mockScorer) {
    const scoredFile = path.join(resultsDir, `sweep_${suite}_scored.jsonl`);
    console.log(`\nScoring with MetricX-24 mock scorer: ${scoredFile}...`);
    const pyBin = fs.existsSync(path.join(process.cwd(), '.venv-metricx', 'bin', 'python'))
      ? path.join(process.cwd(), '.venv-metricx', 'bin', 'python')
      : 'python3';
    const scoreProc = spawnSync(pyBin, [
      path.join(process.cwd(), 'scoring', 'metricx', 'score_metricx.py'),
      '--input', sweepOutputFile,
      '--output', scoredFile,
      '--mode', 'both',
      '--mock-scorer',
    ], { stdio: 'inherit' });
    if (scoreProc.status === 0) {
      fileToGrade = scoredFile;
    }
  }

  const gradedFile = path.join(resultsDir, `sweep_${suite}_graded.jsonl`);
  const reportFile = path.join(resultsDir, `sweep_${suite}_report.md`);
  const summaryFile = path.join(resultsDir, `sweep_${suite}_summary.csv`);

  console.log('\nRunning grader and report generator...');
  const gradeArgs = [
    'run', '--', 'grade',
    '--input', fileToGrade,
    '--output', gradedFile,
    '--concurrency', String(concurrency),
  ];

  if (withJudges || judgesConfig) {
    let configPath = judgesConfig;
    if (!configPath) {
      configPath = path.join(resultsDir, `judges_${suite}.json`);
      const defaultJudges = [
        'anthropic/claude-haiku-5.5',
        'deepseek/deepseek-v4.1-flash',
      ];
      const models = judgeModels && judgeModels.length > 0 ? judgeModels : defaultJudges;
      const judgePayload = {
        judges: models.map((m) => ({
          base_url: 'https://openrouter.ai/api/v1',
          api_key: orKey,
          model: m,
        })),
        judge_a: {
          base_url: 'https://openrouter.ai/api/v1',
          api_key: orKey,
          model: models[0],
        },
        judge_b: {
          base_url: 'https://openrouter.ai/api/v1',
          api_key: orKey,
          model: models[1] || models[0],
        },
      };
      fs.writeFileSync(configPath, JSON.stringify(judgePayload, null, 2), 'utf8');
      console.log(`Configured ${models.length} judges: ${models.join(', ')}`);
    }
    gradeArgs.push('--judges-config', configPath);
  }

  const gradeRes = spawnSync('cargo', gradeArgs, { stdio: 'inherit' });

  // Generate track-separated Phase 2 markdown report and CSV summary
  const targetRecordsFile = (gradeRes.status === 0 && fs.existsSync(gradedFile)) ? gradedFile : fileToGrade;
  const records = loadAndDeduplicateJSONL(targetRecordsFile);
  writeReportFiles(records, reportFile, summaryFile);
  console.log(`\nPhase 2 Markdown report generated at: ${reportFile}`);
  console.log(`CSV summary generated at: ${summaryFile}`);
}

if (import.meta.main || process.argv[1]?.endsWith('cli.ts')) {
  main().catch((e) => {
    console.error('Fatal CLI error:', e);
    process.exit(1);
  });
}
