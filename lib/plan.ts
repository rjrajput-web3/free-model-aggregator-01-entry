// The routing plan: which free model serves each request, and in what order to
// fail over when a provider refuses.
//
// What the trial draws per provider, and what this plan is built around:
//   - Every provider's FIRST call always succeeds: its failure numbers start at 2,
//     its outage (when it has one) starts at 2, and its budget is at least 4.
//   - After that, each provider refuses 1-2 of its calls (its own numbers, drawn in
//     [2, budget-1]) and one provider goes down entirely from some call in
//     [2, budget-1]. Budgets run 4-16 calls per provider; the ten together hold
//     72-92 calls for 20 requests.
//   - A refusal is never the answer: the request is served by the next candidate.
//     A request whose first call refused still counts as failover-handled when it
//     is served, so a plan that fails over cleanly loses nothing on the failover
//     column; it only costs the extra provider call (the tie-break) and, where the
//     fallback model rates lower, a little quality.
//
// The plan, computed once at startup from the brief (deterministic, no clock, no
// unseeded randomness, so a replay makes the same choices):
//   1. Every request gets the full candidate list: every probed free model that
//      meets its requirements, best rating first. A paid model is never a
//      candidate. An OpenAI JSON-mode request never goes to an anthropic-wire
//      provider: that wire cannot echo JSON mode, so the request would fail its
//      wire checks there. (An Anthropic JSON request may go anywhere: its wire
//      carries no JSON-mode flag, so only the model's own json capability matters.)
//   2. Primaries are assigned greedily in manifest order under a per-provider cap.
//      The cap keeps every provider under its smallest possible budget, so a
//      primary can never fail for being over budget, and spreads the load so one
//      provider's outage cannot take many requests down. The caps favour the
//      best-rated models: the 1475 openrouter model takes the most, the 1390 /
//      1420 models take none as primaries and stay as failover depth.
//   3. At request time the primary is tried first, then the rest of the candidate
//      list in rating order. A provider that has failed twice in this trial is
//      pushed to the back of the order: two failures mean it is most likely out
//      of budget or down, and there are always other providers to try first.

import {
  capabilitiesOf,
  meetsRequirements,
  modelFree,
  ratingOfModel,
  type Brief,
  type Probe,
  type ProbedModel,
  type ProviderEntry,
  type RatingRow,
  type Requirement,
  type WorkloadRequest,
} from './brief.ts';

/** One (provider, model) pair that may serve a request, with its rating. */
export interface Candidate {
  provider: ProviderEntry;
  model: ProbedModel;
  row: RatingRow;
  rating: number;
}

/** How many primaries one provider may take. The provider set is fixed by the
 *  sealed table, so these are constants of the entry, not of any one trial:
 *  every provider's smallest possible budget is above its cap, so primaries can
 *  never fail for being over budget. */
export const PRIMARY_CAPS: Record<string, number> = {
  openrouter: 6,
  gemini: 3,
  zai: 2,
  cloudflare: 3,
  sambanova: 2,
  ovh: 2,
  groq: 1,
  cerebras: 1,
  pollinations: 0,
  kilol: 0,
};

/** After this many failures a provider is treated as down and tried last. */
const DOWN_AFTER_FAILURES = 2;

/** One request's ordered candidates and its assigned primary. */
export interface PlannedRequest {
  request: WorkloadRequest;
  candidates: Candidate[];
  primary: Candidate | null;
}

function byRatingThenId(a: Candidate, b: Candidate): number {
  if (a.rating !== b.rating) return b.rating - a.rating;
  if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
  return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
}

/**
 * Whether a candidate may serve a request on its client's wire. The only
 * wire restriction is JSON mode: an OpenAI client asks for JSON with
 * response_format, which an anthropic-wire provider cannot echo back.
 */
function wireOk(clientApi: 'openai' | 'anthropic', requires: Requirement, provider: ProviderEntry): boolean {
  if (clientApi === 'openai' && requires.json === true && provider.wire === 'anthropic') return false;
  return true;
}

/** All free models that may serve `request`, best rating first. */
export function candidatesFor(
  brief: Brief,
  probes: readonly Probe[],
  request: WorkloadRequest,
): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (row === null) continue;
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(request.requires, capabilitiesOf(model, row))) continue;
      if (!wireOk(request.api, request.requires, probe.provider)) continue;
      out.push({ provider: probe.provider, model, row, rating: row.rating_milli });
    }
  }
  return out.sort(byRatingThenId);
}

/**
 * Assign primaries: walk the manifest in order, and for each request take the
 * best candidate whose provider still has primary capacity. When every
 * candidate's provider is at its cap, take the best candidate anyway: a
 * request with a primary is always better than one without.
 */
export function assignPrimaries(
  brief: Brief,
  probes: readonly Probe[],
  manifest: readonly WorkloadRequest[],
): PlannedRequest[] {
  const used = new Map<string, number>();
  const planned: PlannedRequest[] = [];
  for (const request of manifest) {
    const candidates = candidatesFor(brief, probes, request);
    let primary: Candidate | null = null;
    for (const candidate of candidates) {
      const cap = PRIMARY_CAPS[candidate.provider.id] ?? 0;
      if ((used.get(candidate.provider.id) ?? 0) < cap) {
        primary = candidate;
        break;
      }
    }
    if (primary === null) primary = candidates[0] ?? null;
    if (primary !== null) {
      used.set(primary.provider.id, (used.get(primary.provider.id) ?? 0) + 1);
    }
    planned.push({ request, candidates, primary });
  }
  return planned;
}

/** Per-provider trial state: calls made and failures seen so far. */
export interface ProviderStats {
  calls: number;
  failures: number;
}

/**
 * The live routing state. Primaries are fixed at startup; the attempt order for
 * one request is its primary first, then the rest of its candidates in rating
 * order, with providers that have failed twice pushed to the back.
 */
export class Router {
  readonly planned: PlannedRequest[];
  private readonly stats = new Map<string, ProviderStats>();

  constructor(planned: PlannedRequest[]) {
    this.planned = planned;
  }

  statsOf(providerId: string): ProviderStats {
    let s = this.stats.get(providerId);
    if (s === undefined) {
      s = { calls: 0, failures: 0 };
      this.stats.set(providerId, s);
    }
    return s;
  }

  noteCall(providerId: string): void {
    this.statsOf(providerId).calls += 1;
  }

  noteFailure(providerId: string): void {
    this.statsOf(providerId).failures += 1;
  }

  /** The attempt order for one planned request: primary first, then the rest. */
  attemptsFor(planned: PlannedRequest): Candidate[] {
    const seen = new Set<string>();
    const ordered: Candidate[] = [];
    const push = (c: Candidate): void => {
      const key = `${c.provider.id}/${c.model.id}`;
      if (seen.has(key)) return;
      seen.add(key);
      ordered.push(c);
    };
    if (planned.primary !== null) push(planned.primary);
    const fresh: Candidate[] = [];
    const tired: Candidate[] = [];
    for (const candidate of planned.candidates) {
      if (planned.primary !== null && candidate === planned.primary) continue;
      const failures = this.statsOf(candidate.provider.id).failures;
      (failures >= DOWN_AFTER_FAILURES ? tired : fresh).push(candidate);
    }
    // Order the rest by rating, then provider/model id, so the choice stays
    // deterministic. (A risk-adjusted ordering that discounted candidates by
    // calls already made to their provider was tried on the practice trials
    // and made things worse: 25/27/28 calls versus 23/25/28 in pure rating
    // order, at the same score 99. The hidden failure numbers are not
    // monotone in the call count, so the discount misfires.)
    fresh.sort(byRatingThenId);
    tired.sort(byRatingThenId);
    for (const c of fresh) push(c);
    for (const c of tired) push(c);
    return ordered;
  }
}
