import * as fs from 'fs';
import * as path from 'path';

export function makeCompositeKey(r: {
  provider_id: string;
  tier: string;
  item_id: string;
  variant_id: string;
}): string {
  return `${r.provider_id}::${r.tier}::${r.item_id}::${r.variant_id}`;
}

export function deduplicateRecords(records: any[]): { unique: any[]; duplicatesCount: number } {
  const seen = new Set<string>();
  const unique: any[] = [];
  let duplicatesCount = 0;

  for (const r of records) {
    const key = makeCompositeKey({
      provider_id: r.provider_id || '',
      tier: r.tier || '',
      item_id: r.item_id || '',
      variant_id: r.variant_id || '',
    });
    if (seen.has(key)) {
      duplicatesCount++;
    } else {
      seen.add(key);
      unique.push(r);
    }
  }

  if (duplicatesCount > 0) {
    console.warn(`[Report Aggregator] Warning: deduplicated ${duplicatesCount} duplicate records with key (provider_id, tier, item_id, variant_id).`);
  }

  return { unique, duplicatesCount };
}

export function loadAndDeduplicateJSONL(filePath: string): any[] {
  if (!fs.existsSync(filePath)) return [];
  const raw = fs.readFileSync(filePath, 'utf8');
  const records: any[] = [];
  for (const line of raw.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      records.push(JSON.parse(trimmed));
    } catch {
      // ignore
    }
  }
  return deduplicateRecords(records).unique;
}

function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(Math.floor((p / 100) * sorted.length), sorted.length - 1);
  return sorted[idx];
}

interface ModelMetrics {
  provider_id: string;
  count: number;
  metricx_ref_avg: number | null;
  metricx_qe_avg: number | null;
  acr_pct: number;
  post_repair_acr_pct: number | null;
  por_pct: number | null;
  chrf_avg: number | null;
  catastrophe_pct: number;
  p50_ttft_ms: number;
  p50_total_ms: number;
  uncached_per_1k: number;
  cached_per_1k: number;
}

function computeMetricsForGroup(records: any[]): ModelMetrics[] {
  const byModel = new Map<string, any[]>();
  for (const r of records) {
    const pid = r.provider_id || 'unknown';
    if (!byModel.has(pid)) byModel.set(pid, []);
    byModel.get(pid)!.push(r);
  }

  const result: ModelMetrics[] = [];
  for (const [provider_id, list] of byModel.entries()) {
    const refs = list.map((r) => r.metricx_ref).filter((v) => typeof v === 'number');
    const qes = list.map((r) => r.metricx_qe).filter((v) => typeof v === 'number');
    const chrfs = list.map((r) => r.chrf).filter((v) => typeof v === 'number');

    const acrHits = list.filter((r) => r.layer1_passed === true || r.layer1?.passed === true || r.first_pass_acr === 1).length;
    const hasRepairData = list.some((r) => r.repair_triggered !== undefined || r.post_repair_acr !== undefined);
    const postRepairHits = hasRepairData
      ? list.filter((r) => (r.post_repair_acr === 1 || r.repair_succeeded === true || r.layer1_passed === true || r.layer1?.passed === true)).length
      : null;

    const tier2cList = list.filter((r) => r.tier === 'tier2_c_adversarial_conflict');
    const porHits = tier2cList.filter((r) => r.layer1_passed === true || r.layer1?.passed === true).length;

    const catastropheCount = list.filter((r) => {
      if (r.chrf_catastrophe_flag) return true;
      if (typeof r.chrf === 'number' && r.chrf < 25.0) return true;
      return false;
    }).length;

    const ttfts = list.map((r) => r.ttft_ms).filter((v) => typeof v === 'number');
    const totals = list.map((r) => r.total_latency_ms).filter((v) => typeof v === 'number');
    const uncachedCosts = list.map((r) => r.uncached_cost_usd || 0);
    const cachedCosts = list.map((r) => r.cached_cost_usd || 0);

    const avg = (arr: number[]) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null);
    const avgCost = (costs: number[]) => (costs.length ? (costs.reduce((a, b) => a + b, 0) / costs.length) * 1000 : 0);

    result.push({
      provider_id,
      count: list.length,
      metricx_ref_avg: avg(refs),
      metricx_qe_avg: avg(qes),
      acr_pct: (acrHits / list.length) * 100,
      post_repair_acr_pct: postRepairHits !== null ? (postRepairHits / list.length) * 100 : null,
      por_pct: tier2cList.length ? (porHits / tier2cList.length) * 100 : null,
      chrf_avg: avg(chrfs),
      catastrophe_pct: (catastropheCount / list.length) * 100,
      p50_ttft_ms: percentile(ttfts, 50),
      p50_total_ms: percentile(totals, 50),
      uncached_per_1k: avgCost(uncachedCosts),
      cached_per_1k: avgCost(cachedCosts),
    });
  }

  // Sort ascending by MetricX-24 Ref (0-25 lower is better)
  result.sort((a, b) => {
    if (a.metricx_ref_avg !== null && b.metricx_ref_avg !== null) {
      return a.metricx_ref_avg - b.metricx_ref_avg;
    }
    return b.acr_pct - a.acr_pct;
  });

  return result;
}

function renderTable(metrics: ModelMetrics[], showRepair: boolean): string {
  let s = '| Model / Provider | MetricX-24 Ref (0–25 ↓) | MetricX-24 QE (0–25 ↓) | ACR % |';
  if (showRepair) s += ' Post-Repair ACR % |';
  s += ' chrF (Floor) | Catastrophe Flag % | p50 TTFT (ms) | p50 Total (ms) | Cost / 1k (Uncached / Cached) |\n';

  s += '|---|---|---|---|';
  if (showRepair) s += '---|';
  s += '---|---|---|---|---|\n';

  for (const m of metrics) {
    const refStr = m.metricx_ref_avg !== null ? m.metricx_ref_avg.toFixed(2) : 'N/A';
    const qeStr = m.metricx_qe_avg !== null ? m.metricx_qe_avg.toFixed(2) : 'N/A';
    const chrfStr = m.chrf_avg !== null ? m.chrf_avg.toFixed(1) : 'N/A';
    const costStr = `$${m.uncached_per_1k.toFixed(3)} / $${m.cached_per_1k.toFixed(3)}`;

    s += `| \`${m.provider_id}\` | **${refStr}** | ${qeStr} | ${m.acr_pct.toFixed(1)}% |`;
    if (showRepair) {
      s += ` ${m.post_repair_acr_pct !== null ? m.post_repair_acr_pct.toFixed(1) + '%' : 'N/A'} |`;
    }
    s += ` ${chrfStr} | ${m.catastrophe_pct.toFixed(1)}% | ${m.p50_ttft_ms.toFixed(0)} | ${m.p50_total_ms.toFixed(0)} | ${costStr} |\n`;
  }
  return s;
}

export function generateReportMarkdown(records: any[]): string {
  const { unique } = deduplicateRecords(records);
  const isMock = unique.some((r) => r.metricx_is_mock === true);
  const showRepair = unique.some((r) => r.repair_triggered !== undefined || r.post_repair_acr !== undefined);

  let md = '# Pragmatic MT Evaluation Report (Phase 2)\n\n';

  if (isMock) {
    md += '> [!WARNING]\n';
    md += '> **[MOCK METRICX - DO NOT PUBLISH]** Neural scoring was generated via deterministic CPU mock harness.\n';
    md += '> Re-score on RTX 3060 Ti host for authoritative metrics.\n\n';
  }

  md += '## Primary Metric Hierarchy\n';
  md += '- **Authoritative Pragmatics**: `ACR %` (Layer-1 Anchor Compliance), `Post-Repair ACR %` (1-Hop Contrastive Repair).\n';
  md += '- **Authoritative Fluency & Adequacy**: `MetricX-24 Ref (0–25 ↓)` and `MetricX-24 QE (0–25 ↓)` (lower is better).\n';
  md += '- **Catastrophe Guardrail**: `chrF < 25.0` or script-purity failure triggers catastrophe flag.\n\n';

  const tracks = ['casual_conversational', 'diagnostic_morphology', 'register_contrastive'];

  for (const track of tracks) {
    const trackRecords = unique.filter((r) => (r.track || 'casual_conversational') === track);
    if (trackRecords.length === 0) continue;

    const title = track === 'casual_conversational'
      ? 'Track 1: Casual Conversational (Headline Benchmark)'
      : track === 'diagnostic_morphology'
      ? 'Track 2: Diagnostic Morphology'
      : 'Track 3: Register Contrastive';

    md += `### ${title}\n\n`;
    md += `*Evaluated on ${trackRecords.length} records. Sorted strictly by \`MetricX-24 Ref (0–25 ↓)\` ascending.*\n\n`;

    const metrics = computeMetricsForGroup(trackRecords);
    md += renderTable(metrics, showRepair);
    md += '\n';
  }

  return md;
}

export function writeReportFiles(records: any[], reportMdPath: string, summaryCsvPath?: string): void {
  const md = generateReportMarkdown(records);
  const dir = path.dirname(reportMdPath);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(reportMdPath, md, 'utf8');

  if (summaryCsvPath) {
    const { unique } = deduplicateRecords(records);
    const metrics = computeMetricsForGroup(unique);
    let csv = 'provider_id,count,metricx_ref,metricx_qe,acr_pct,post_repair_acr_pct,chrf,catastrophe_pct,p50_ttft_ms,p50_total_ms\n';
    for (const m of metrics) {
      csv += `"${m.provider_id}",${m.count},${m.metricx_ref_avg ?? ''},${m.metricx_qe_avg ?? ''},${m.acr_pct.toFixed(2)},${m.post_repair_acr_pct?.toFixed(2) ?? ''},${m.chrf_avg ?? ''},${m.catastrophe_pct.toFixed(2)},${m.p50_ttft_ms},${m.p50_total_ms}\n`;
    }
    fs.writeFileSync(summaryCsvPath, csv, 'utf8');
  }
}

function parseCliArgs() {
  const args = process.argv.slice(2);
  let input = '';
  let output = '';
  let csv = '';

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--input' && args[i + 1]) input = args[++i];
    else if (args[i].startsWith('--input=')) input = args[i].slice('--input='.length);
    else if (args[i] === '--output' && args[i + 1]) output = args[++i];
    else if (args[i].startsWith('--output=')) output = args[i].slice('--output='.length);
    else if (args[i] === '--csv' && args[i + 1]) csv = args[++i];
    else if (args[i].startsWith('--csv=')) csv = args[i].slice('--csv='.length);
  }
  return { input, output, csv };
}

if (import.meta.main || process.argv[1]?.endsWith('report-generator.ts')) {
  const { input, output, csv } = parseCliArgs();
  if (!input || !output) {
    console.error('Usage: bun run src/reporting/report-generator.ts --input <file.jsonl> --output <report.md> [--csv <summary.csv>]');
    process.exit(1);
  }
  const records = loadAndDeduplicateJSONL(input);
  writeReportFiles(records, output, csv || undefined);
  console.log(`Report generated at: ${output}`);
}
