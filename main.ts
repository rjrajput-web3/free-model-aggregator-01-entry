// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The server reads the trial brief, probes each provider's catalog route,
// builds one catalog, registers its port and catalog with the env, and serves
// the twenty requests the env's client sends: POST /v1/chat/completions as an
// OpenAI-compatible server and POST /v1/messages as an Anthropic-compatible
// one, in JSON and in server-sent events. It polls GET /work and exits 0 the
// moment the env says the workload is done.
//
// Routing (see lib/plan.ts): every request is served by the best-rated free
// model that meets its requirements, under a per-provider primary cap that
// keeps every provider below its smallest possible budget, with failover to
// the next-best candidate when a provider refuses. The first call to any
// provider always succeeds; later calls may hit its failure numbers or its
// outage, and the plan is built so those cost one extra call and a little
// quality at most, never the request.
//
// `node main.ts --list` prints the catalog document and exits 0 before any
// server starts; the register body carries that same document, byte for byte.
// Every choice is a function of the brief, the requests and the providers'
// answers: no wall clock and no unseeded randomness, so a replay leaves the
// same state export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  isRecord,
  joinBase,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type Brief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
  type WorkloadRequest,
} from './lib/brief.ts';
import {
  assignPrimaries,
  candidatesFor,
  Router,
  type Candidate,
  type PlannedRequest,
} from './lib/plan.ts';
import {
  anthropicError,
  anthropicFramesToOpenAI,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  answerToAnthropic,
  answerToOpenAI,
  bufferedToAnthropic,
  geminiFramesToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  openAIToGemini,
  readSseFrames,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream provider call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let planned: PlannedRequest[] = [];
let router: Router = new Router([]);
/** How many client API requests have arrived: the manifest is in arrival order. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider, build the one catalog, and plan the twenty requests. */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(
    brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)),
  );
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes, brief.ratings);
  const manifest = brief.workload?.manifest ?? [];
  planned = assignPrimaries(brief, probes, manifest);
  router = new Router(planned);
  for (const p of planned) {
    const primary = p.primary === null ? 'none' : `${p.primary.provider.id}/${p.primary.model.id}`;
    log(`plan ${p.request.id} (${p.request.api}${p.request.stream ? ', stream' : ''}): ${primary}`);
  }
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** The planned request now arriving, by arrival order, else a body-derived plan. */
function plannedFor(api: 'openai' | 'anthropic', body: Json): PlannedRequest {
  const entry = planned[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry;
  const request: WorkloadRequest = {
    id: `extra-${manifestIndex}`,
    api,
    stream: body.stream === true,
    weight: 1,
    requires: requirementsFromBody(body),
  };
  return { request, candidates: candidatesFor(brief, probes, request), primary: null };
}

/** One client request body, or null when it is not one JSON object or too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  status: number,
  type: string,
  message: string,
): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** Build one upstream provider call for a candidate, from the client's body. */
async function callUpstream(
  candidate: Candidate,
  clientApi: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
): Promise<Response> {
  const provider = candidate.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (stream) headers.accept = 'text/event-stream';
  let url: string;
  let payload: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    payload =
      clientApi === 'openai'
        ? { ...body, model: candidate.model.id, ...(stream ? { stream_options: { include_usage: true } } : {}) }
        : anthropicToOpenAI(body, candidate.model.id);
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    payload =
      clientApi === 'anthropic'
        ? { ...body, model: candidate.model.id }
        : openAIToAnthropic(body, candidate.model.id);
  } else {
    const encoded = encodeURIComponent(candidate.model.id);
    url = stream
      ? `${base}/v1beta/models/${encoded}:streamGenerateContent?alt=sse`
      : `${base}/v1beta/models/${encoded}:generateContent`;
    payload = clientApi === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
  }
  return fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
  });
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/**
 * Serve one client request: try the planned candidates in order until one
 * answers, failing over on any refusal. Returns true when answered.
 */
async function serve(
  clientApi: 'openai' | 'anthropic',
  body: Json,
  res: ServerResponse,
): Promise<boolean> {
  const plan = plannedFor(clientApi, body);
  const stream = body.stream === true;
  const attempts = router.attemptsFor(plan);
  const requestId = plan.request.id;
  let tried = 0;
  for (const candidate of attempts) {
    tried += 1;
    router.noteCall(candidate.provider.id);
    let upstream: Response;
    try {
      upstream = await callUpstream(candidate, clientApi, body, stream);
    } catch (err) {
      router.noteFailure(candidate.provider.id);
      log(`${requestId}: ${candidate.provider.id}/${candidate.model.id} unreachable: ${(err as Error).message}`);
      continue;
    }
    if (!upstream.ok) {
      router.noteFailure(candidate.provider.id);
      const text = await upstream.text().catch(() => '');
      log(`${requestId}: ${candidate.provider.id}/${candidate.model.id} refused HTTP ${upstream.status}: ${text.slice(0, 200)}`);
      continue;
    }
    if (upstream.body === null) {
      router.noteFailure(candidate.provider.id);
      log(`${requestId}: ${candidate.provider.id}/${candidate.model.id} answered with no body`);
      continue;
    }
    log(
      `${requestId}: served by ${candidate.provider.id}/${candidate.model.id} ` +
        `(rating ${candidate.rating}, attempt ${tried})`,
    );
    const wire = candidate.provider.wire;
    if (stream) {
      res.writeHead(200, {
        'content-type': 'text/event-stream',
        'cache-control': 'no-cache',
        connection: 'keep-alive',
      });
      if (clientApi === 'openai' && wire === 'openai') {
        await pipeStream(upstream.body, res);
      } else if (clientApi === 'openai' && wire === 'anthropic') {
        const frames = await readSseFrames(upstream.body);
        for (const event of anthropicFramesToOpenAI(frames, candidate.model.id)) res.write(event);
        res.end();
      } else if (clientApi === 'openai' && wire === 'gemini') {
        const frames = await readSseFrames(upstream.body);
        for (const event of geminiFramesToOpenAI(frames, candidate.model.id)) res.write(event);
        res.end();
      } else if (clientApi === 'anthropic' && wire === 'anthropic') {
        await pipeStream(upstream.body, res);
      } else {
        // Anthropic streams need their input tokens in message_start, before
        // any content, and an OpenAI or Gemini provider only reports usage
        // at the end: buffer the stream, then emit the Anthropic sequence.
        const frames = await readSseFrames(upstream.body);
        const events =
          wire === 'openai'
            ? bufferedToAnthropic('openai', frames, candidate.model.id)
            : bufferedToAnthropic('gemini', frames, candidate.model.id);
        for (const event of events) res.write(event);
        res.end();
      }
      return true;
    }
    let answer: unknown;
    try {
      answer = await upstream.json();
    } catch {
      router.noteFailure(candidate.provider.id);
      log(`${requestId}: ${candidate.provider.id}/${candidate.model.id} answered non-JSON`);
      continue;
    }
    if (!isRecord(answer)) {
      router.noteFailure(candidate.provider.id);
      log(`${requestId}: ${candidate.provider.id}/${candidate.model.id} answered a non-object`);
      continue;
    }
    if (clientApi === 'openai') sendJson(res, 200, answerToOpenAI(wire, answer, candidate.model.id));
    else sendJson(res, 200, answerToAnthropic(wire, answer, candidate.model.id));
    return true;
  }
  log(`${requestId}: no candidate answered after ${tried} attempts`);
  return false;
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) {
      return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    }
    const ok = await serve('openai', body, res);
    if (!ok) sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory answered this request');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) {
      return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    }
    const ok = await serve('anthropic', body, res);
    if (!ok) sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory answered this request');
    return;
  }
  sendError(
    res,
    url.pathname === '/v1/messages' ? 'anthropic' : 'openai',
    404,
    'not_found',
    `no route ${req.method ?? ''} ${url.pathname}`,
  );
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) {
    throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  }
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') {
    log(`registered: the workload holds ${reply.requests} requests`);
  }
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the
    // one built object and stop before any server starts. A process whose
    // stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(
          res,
          req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai',
          500,
          'internal_error',
          'the server failed to answer',
        );
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
