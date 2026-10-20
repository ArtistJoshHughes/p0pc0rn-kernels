// SPDX-License-Identifier: MPL-2.0
// Copyright 2026 Joshua Hughes
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// Headless test runner, no dependencies (Node 22+): serves the repository, opens test/index.html in Chrome or Edge through the
// DevTools protocol, prints each result and exits non-zero on any failure.
//   node test/run.mjs [--browser <path>] [--low-power] [--page bench/index.html] [--timeout 900]
// --low-power picks the integrated GPU on machines with two (Chrome's --use-webgpu-power-preference=force-low-power).
import http from 'node:http';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const PAGE = arg('--page', 'test/index.html'), TIMEOUT = +arg('--timeout', 900) * 1000;
const CANDIDATES = {
  win32: [process.env.PROGRAMFILES, process.env['PROGRAMFILES(X86)'], process.env.LOCALAPPDATA].filter(Boolean)
    .flatMap((d) => [path.join(d, 'Google/Chrome/Application/chrome.exe'), path.join(d, 'Microsoft/Edge/Application/msedge.exe')]),
  darwin: ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge'],
  linux: ['/usr/bin/google-chrome', '/usr/bin/google-chrome-stable', '/usr/bin/chromium', '/usr/bin/chromium-browser', '/usr/bin/microsoft-edge'],
};
const BROWSER = arg('--browser', process.env.CHROME_PATH || (CANDIDATES[process.platform] || []).find((p) => fs.existsSync(p)));
if (!BROWSER) { console.error('No Chrome or Edge found: pass --browser <path> or set CHROME_PATH.'); process.exit(2); }

const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.css': 'text/css' };
const server = http.createServer((req, res) => {
  const p = path.join(ROOT, decodeURIComponent(new URL(req.url, 'http://x').pathname));
  if (!p.startsWith(ROOT + path.sep) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'content-type': TYPES[path.extname(p)] || 'application/octet-stream', 'cache-control': 'no-store' });
  fs.createReadStream(p).pipe(res);
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const port = server.address().port;
const freePort = () => new Promise((r) => { const s = http.createServer(); s.listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => r(p)); }); });
const dbg = await freePort(), profile = fs.mkdtempSync(path.join(os.tmpdir(), 'kernels-test-'));
const flags = ['--headless=new', `--remote-debugging-port=${dbg}`, `--user-data-dir=${profile}`, '--no-first-run', '--no-default-browser-check',
  '--enable-unsafe-webgpu', ...(process.platform === 'linux' ? ['--enable-features=Vulkan'] : []),
  ...(process.argv.includes('--low-power') ? ['--use-webgpu-power-preference=force-low-power'] : []), 'about:blank'];
const browser = spawn(BROWSER, flags, { stdio: 'ignore' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let code = 1;
try {
  let target = null;
  for (let i = 0; i < 120 && !target; i++) {
    try { target = (await (await fetch(`http://127.0.0.1:${dbg}/json/list`)).json()).find((t) => t.type === 'page'); } catch (e) { /* not up yet */ }
    if (!target) await sleep(250);
  }
  if (!target) throw new Error('the browser did not start');
  const ws = new WebSocket(target.webSocketDebuggerUrl); let id = 0; const wait = new Map();
  ws.onmessage = (m) => { const d = JSON.parse(m.data); if (d.id && wait.has(d.id)) { wait.get(d.id)(d); wait.delete(d.id); } };
  await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
  const call = (method, params = {}) => new Promise((r) => { const i = ++id; wait.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
  const js = async (expr) => { const d = await call('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true }); return d.result && d.result.result ? d.result.result.value : undefined; };
  await call('Page.enable');
  await call('Page.navigate', { url: `http://127.0.0.1:${port}/${PAGE}` });
  const t0 = Date.now(); let shown = 0, res = null;
  while (Date.now() - t0 < TIMEOUT) {
    await sleep(1000);
    res = await js('JSON.stringify(window.__results || null)').then((s) => s && JSON.parse(s)).catch(() => null);
    if (!res) continue;
    for (const r of (res.results || []).slice(shown)) console.log(`${r.ok === false ? 'FAIL' : r.ok === null ? 'skip' : 'pass'}  ${r.name}${r.cos != null ? `  cos ${r.cos.toFixed(6)}` : ''}${r.maxErr != null ? `  max ${r.maxErr.toExponential(2)}` : ''}${r.error || r.note ? '  ' + (r.error || r.note) : ''}${r.gflops != null ? `  ${r.gflops} GFLOP/s` : ''}`);
    shown = (res.results || []).length;
    if (res.done) break;
  }
  if (!res || !res.done) throw new Error('timed out');
  if (res.error) throw new Error(res.error);
  if (res.device) console.log('device', JSON.stringify(res.device));
  const bad = (res.results || []).filter((r) => r.ok === false).length;
  console.log(bad ? `${bad} of ${res.results.length} failed` : `all ${res.results.length} passed`);
  if (process.argv.includes('--json')) fs.writeFileSync(arg('--json'), JSON.stringify(res, null, 1));
  code = bad ? 1 : 0;
  ws.close();
} catch (e) { console.error('error:', e.message); code = 1; }
finally { browser.kill(); server.close(); setTimeout(() => { try { fs.rmSync(profile, { recursive: true, force: true }); } catch (e) { /* in use */ } process.exit(code); }, 1500); }
