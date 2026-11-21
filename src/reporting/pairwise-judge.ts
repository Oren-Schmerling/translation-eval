import { OpenRouterRunner } from '../providers/openrouter';
import { ModelPricing, RunRecord } from '../types';

export interface PairwiseComparison {
  item_id: string;
  source_en: string;
  reference_target: string;
  model_a: string;
  model_b: string;
  translation_a: string;
  translation_b: string;
  presentation_order: 'standard' | 'swapped';
  winner: 'A' | 'B' | 'tie';
  reason?: string;
}

export interface PairwiseLeaderboardEntry {
  model: string;
  wins: number;
  losses: number;
  ties: number;
  total_matches: number;
  win_rate_pct: number;
}

export function isPresentationSwapped(itemId: string): boolean {
  let hash = 0;
  for (let i = 0; i < itemId.length; i++) {
    hash = (hash * 31 + itemId.charCodeAt(i)) | 0;
  }
  return Math.abs(hash) % 2 === 1;
}

export function buildPairwiseJudgePrompt(
  source: string,
  reference: string,
  candidateA: string,
  candidateB: string
): string {
  return [
    'You are an expert bilingual translation judge.',
    'Compare Candidate A and Candidate B against the English source and the authoritative native reference.',
    'Focus on pragmatic correctness, tone, register, and grammatical accuracy.',
    'Do NOT output a new translation.',
    '',
    `[Source (English)]: ${source}`,
    `[Human Native Reference]: ${reference}`,
    `[Candidate A]: ${candidateA}`,
    `[Candidate B]: ${candidateB}`,
    '',
    'Output STRICTLY valid JSON with no markdown formatting:',
    '{"winner": "A" | "B" | "tie", "reason": "brief explanation"}',
  ].join('\n');
}

export function parseJudgeDecision(raw: string, swapped: boolean): 'A' | 'B' | 'tie' {
  try {
    const cleaned = raw.replace(/```json/g, '').replace(/```/g, '').trim();
    const parsed = JSON.parse(cleaned);
    const win = parsed.winner?.toUpperCase();
    if (win === 'A') return swapped ? 'B' : 'A';
    if (win === 'B') return swapped ? 'A' : 'B';
    return 'tie';
  } catch {
    if (raw.includes('"winner": "A"') || raw.includes('"winner":"A"')) return swapped ? 'B' : 'A';
    if (raw.includes('"winner": "B"') || raw.includes('"winner":"B"')) return swapped ? 'A' : 'B';
    return 'tie';
  }
}

export function computePairwiseLeaderboard(
  comparisons: PairwiseComparison[]
): PairwiseLeaderboardEntry[] {
  const stats = new Map<string, { wins: number; losses: number; ties: number }>();

  const initModel = (m: string) => {
    if (!stats.has(m)) stats.set(m, { wins: 0, losses: 0, ties: 0 });
  };

  for (const c of comparisons) {
    initModel(c.model_a);
    initModel(c.model_b);

    const sA = stats.get(c.model_a)!;
    const sB = stats.get(c.model_b)!;

    if (c.winner === 'A') {
      sA.wins++;
      sB.losses++;
    } else if (c.winner === 'B') {
      sB.wins++;
      sA.losses++;
    } else {
      sA.ties++;
      sB.ties++;
    }
  }

  const entries: PairwiseLeaderboardEntry[] = [];
  for (const [model, st] of stats.entries()) {
    const total = st.wins + st.losses + st.ties;
    const winRate = total > 0 ? ((st.wins + 0.5 * st.ties) / total) * 100 : 0;
    entries.push({
      model,
      wins: st.wins,
      losses: st.losses,
      ties: st.ties,
      total_matches: total,
      win_rate_pct: winRate,
    });
  }

  entries.sort((a, b) => b.win_rate_pct - a.win_rate_pct);
  return entries;
}
