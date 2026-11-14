// Demo: run this entry against one of the kit's practice trials and print
// every routing decision plus the score.
//
//   node scripts/demo.mjs --trial 0      one practice trial (default 0)
//   node scripts/demo.mjs --all           all three practice trials
//
// The kit is found at $MAKEFUN_KIT_DIR, or at the sibling directory
// ../free-model-aggregator-01-kit next to this repository. The demo starts
// the kit's own env server (the same one the grader runs) with the trial's
// providers and workload, starts this entry against it, waits for the entry
// to exit, stops the env so it writes its state export, then prints each
// request's routing decision and scores the run with the kit's scorer.
//
// Nothing here is a placeholder: the env, the workload and the scorer are
// the kit's own, and the decisions printed come from the real state export.

import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const KIT =
  process.env.MAKEFUN_KIT_DIR ??
  resolve(REPO, '..', 'free-model-aggregator-01-kit');

const args = process.argv.slice(2);
const all = args.includes('--all');
const trialArg = args[args.indexOf('--trial') + 1];
const trials = all ? [0, 1, 2] : [trialArg === undefined ? 0 : Number(trialArg)];

/** Await a promise with a timeout that is cleared when it settles, so the
 *  timer itself never holds the process open after the work is done. */
async function withTimeout(promise, ms) {
  let timer;
  const timeout = new Promise((resolvePromise) => {
    timer = setTimeout(() => resolvePromise({ timedOut: true }), ms);
  });
  const result = await Promise.race([
    promise.then((value) => ({ ...value, timedOut: false })),
    timeout,
  ]);
  clearTimeout(timer);
  return result;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitFor(path, timeoutMs, what) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    if (existsSync(path)) return true;
    await sleep(100);
  }
  throw new Error(`timed out waiting for ${what} at ${path}`);
}

/** Run a short-lived command to completion and capture its output. */
function run(command, argv, env) {
  return new Promise((resolvePromise, rejectPromise) => {
    const child = spawn(command, argv, {
      env: { ...process.env, ...env },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    if (child.stdout !== null) child.stdout.on('data', (d) => (stdout += d.toString()));
    if (child.stderr !== null) child.stderr.on('data', (d) => (stderr += d.toString()));
    child.on('error', rejectPromise);
    child.on('exit', (code, signal) => resolvePromise({ code, signal, stdout, stderr }));
  });
}

/** Spawn a process and resolve once it is running; its exit is watched separately. */
function spawnTracked(command, argv, env, { inheritStderr = false } = {}) {
  const child = spawn(command, argv, {
    env: { ...process.env, ...env },
    stdio: ['ignore', 'pipe', inheritStderr ? 'inherit' : 'pipe'],
  });
  let stdout = '';
  let stderr = '';
  if (child.stdout !== null) child.stdout.on('data', (d) => (stdout += d.toString()));
  if (child.stderr !== null && !inheritStderr) child.stderr.on('data', (d) => (stderr += d.toString()));
  const exited = new Promise((resolvePromise) => {
    child.on('exit', (code, signal) => resolvePromise({ code, signal }));
  });
  return { child, exited, get stdout() { return stdout; }, get stderr() { return stderr; } };
}

async function runTrial(n) {
  const trialDir = join(KIT, 'practice', `trial-${n}`);
  if (!existsSync(join(trialDir, 'input', 'env', 'providers.json'))) {
    throw new Error(`practice trial ${n} not found under ${trialDir}`);
  }
  const port = 18100 + n;
  const stateDir = join(tmpdir(), `fma-demo-trial-${n}`);
  rmSync(stateDir, { recursive: true, force: true });
  mkdirSync(stateDir, { recursive: true });
  const envUrl = `http://127.0.0.1:${port}`;

  console.log(`\n=== trial ${n} ===`);
  console.log(`env:    ${envUrl} (data ${trialDir}/input/env)`);
  console.log(`entry:  ${join(REPO, 'main.ts')}`);

  const env = spawnTracked('node', [join(KIT, 'program', 'env', 'server.mjs')], {
    MAKEFUN_ENV_HOST: '127.0.0.1',
    MAKEFUN_ENV_PORT: String(port),
    MAKEFUN_ENV_DATA_DIR: join(trialDir, 'input', 'env'),
    MAKEFUN_STATE_DIR: stateDir,
  });
  await waitFor(join(stateDir, 'ready'), 15000, 'the env ready file');
  console.log('env ready');

  const entry = spawnTracked('node', [join(REPO, 'main.ts')], {
    MAKEFUN_ENV_URL: envUrl,
    MAKEFUN_PROMPT_FILE: join(trialDir, 'input', 'prompt.txt'),
    MAKEFUN_TRIAL_SEED: `demo-trial-${n}`,
  }, { inheritStderr: true });

  const entryExit = await withTimeout(entry.exited, 90000);
  if (entryExit.timedOut) {
    entry.child.kill('SIGKILL');
    env.child.kill('SIGTERM');
    throw new Error(`trial ${n}: the entry did not exit within 90s`);
  }
  console.log(`entry exited with code ${entryExit.code}`);

  env.child.kill('SIGTERM');
  await waitFor(join(stateDir, 'export', 'state.json'), 15000, 'the state export');

  const state = JSON.parse(readFileSync(join(stateDir, 'export', 'state.json'), 'utf8'));
  console.log('\nrouting decisions:');
  for (const r of state.requests) {
    const failures = r.failures.map((f) => `${f.provider} HTTP ${f.status}`).join(', ');
    const failedChecks = r.checks.filter((c) => !c.pass).map((c) => c.name).join(', ');
    console.log(
      `  ${r.id}  ${String(r.provider ?? '-').padEnd(11)} ${(r.model ?? '-').padEnd(30)} ` +
        `rating ${String(r.rating_milli).padStart(4)}  valid=${r.valid ? 'yes' : 'NO '}  ` +
        `calls=${r.upstream_calls}${failures !== '' ? `  failures: ${failures}` : ''}` +
        `${failedChecks !== '' ? `  FAILED CHECKS: ${failedChecks}` : ''}`,
    );
  }

  const scoresDir = join(stateDir, 'scores');
  mkdirSync(scoresDir, { recursive: true });
  const scored = await run('node', [join(KIT, 'program', 'scorer', 'score.mjs')], {
    MAKEFUN_OUTPUT_DIR: join(stateDir, 'export'),
    MAKEFUN_REFERENCE_DIR: join(trialDir, 'reference'),
    MAKEFUN_SCORES_DIR: scoresDir,
  });
  const scores = JSON.parse(readFileSync(join(scoresDir, 'scores.json'), 'utf8'));
  console.log(`\nscore: ${scores.score} (quality ${scores.quality}, served ${scores.served}, wire ${scores.wire}, failover ${scores.failover}, catalog ${scores.catalog})`);
  console.log(`upstream calls: ${scores.upstream_calls}, working: ${scores.working}`);
  if (scored.stdout.trim() !== '') console.log(scored.stdout.trim());
  if (scored.stderr.trim() !== '') console.log(`scorer stderr: ${scored.stderr.trim()}`);
  return { trial: n, entryCode: entryExit.code, scores };
}

async function main() {
  if (!existsSync(join(KIT, 'program', 'env', 'server.mjs'))) {
    throw new Error(
      `kit not found at ${KIT}: set MAKEFUN_KIT_DIR to the unpacked free-model-aggregator-01-kit directory`,
    );
  }
  const results = [];
  for (const n of trials) results.push(await runTrial(n));
  console.log('\n=== summary ===');
  for (const r of results) {
    console.log(
      `trial ${r.trial}: score ${r.scores.score} ` +
        `(quality ${r.scores.quality}, served ${r.scores.served}, wire ${r.scores.wire}, ` +
        `failover ${r.scores.failover}, catalog ${r.scores.catalog}), ${r.scores.upstream_calls} upstream calls`,
    );
  }
  const bad = results.filter((r) => r.entryCode !== 0 || r.scores.score < 90);
  if (bad.length > 0) {
    console.log(`\n${bad.length} trial(s) below bar (entry exit != 0 or score < 90)`);
    process.exitCode = 1;
  }
}

try {
  await main();
} catch (err) {
  console.error(`demo failed: ${(err).stack ?? String(err)}`);
  process.exit(1);
}
