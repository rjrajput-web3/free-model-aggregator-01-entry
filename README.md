# free-model-aggregator-01-entry

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/rjrajput-web3/free-model-aggregator-01-entry/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/rjrajput-web3/free-model-aggregator-01-entry/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server, two APIs, ten simulated free-tier providers: it serves the OpenAI Chat Completions API and the Anthropic Messages API, in JSON and in streaming, routing each of the 20 requests to the best-rated free model that meets its requirements, and failing over to the next-best model when a provider refuses.

## Run it

```sh
node main.ts
```

The server reads the trial brief from `$MAKEFUN_PROMPT_FILE`, probes the ten providers under `$MAKEFUN_ENV_URL`, registers its port and catalog with the env, serves the workload, and exits 0 when the env says the work is done. `node main.ts --list` prints the catalog document it registers, byte for byte. Node 22, standard library only; nothing is installed.

## How routing works

- **Candidates first.** For each request, every probed free model that meets its requirements (tools, vision, JSON mode, minimum context) is a candidate, best rating first. A paid model is never a candidate. An OpenAI JSON-mode request never goes to an anthropic-wire provider, because that wire cannot echo JSON mode.
- **Capped primaries.** Primaries are assigned greedily in manifest order under a per-provider cap (openrouter 6, gemini 3, cloudflare 3, zai 2, sambanova 2, ovh 2, groq 1, cerebras 1). Every provider's smallest possible budget is above its cap, so a primary can never fail for being over budget, and one provider's outage cannot take many requests down. The lowest-rated models stay as failover depth.
- **Failover.** A refusal (402, 429, 5xx, unreachable) is never the answer: the request is retried on the next candidate in rating order. A provider that has failed twice is tried last. The first call to any provider always succeeds, so the plan's expected cost is about 22-24 provider calls for the 20 requests.
- **Wire fidelity.** Requests pass through with every echo field intact (system prompt, user text and images, tool names, stop sequences, max_tokens, JSON mode). Answers return the provider's text whole, with the provider's usage counts and the stop reason mapped to the client's wire. OpenAI streams always ask for usage upstream so the usage chunk arrives; Anthropic streams from an OpenAI or Gemini provider are buffered and re-emitted so `message_start` carries the input tokens first.

## Demo

```sh
node scripts/demo.mjs --trial 0
```

Runs the entry against the kit's practice trial (env server + workload from the make.fun kit, pointed at with `MAKEFUN_KIT_DIR`), prints each request's routing decision, and scores the run with the kit's scorer. `node scripts/demo.mjs --all` runs all three practice trials.

## Layout

- `main.ts` — the server: brief, catalog, routes, register, poll, `--list`.
- `lib/brief.ts` — the brief's three documents, the catalog probe, the catalog build.
- `lib/plan.ts` — candidate lists, primary caps, failover order.
- `lib/wire.ts` — OpenAI / Anthropic / Gemini request and answer translation.
- `scripts/demo.mjs` — end-to-end demo against the kit's practice trials.

---

<sub>Entered on [make.fun](https://make.fun/?ref=hzsbvtpc&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
