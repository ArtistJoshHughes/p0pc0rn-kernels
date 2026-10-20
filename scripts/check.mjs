// SPDX-License-Identifier: MPL-2.0
// Copyright 2026 Joshua Hughes
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// Checks that need no GPU (run in CI): every script parses, every source file carries the licence header, the package's exports load,
// and the shader generators produce WGSL for their main configurations. The GPU tests are test/run.mjs.
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
let bad = 0;
const fail = (m) => { console.log('FAIL ', m); bad++; };
const walk = (d) => fs.readdirSync(d, { withFileTypes: true }).flatMap((e) => e.isDirectory() ? walk(path.join(d, e.name)) : [path.join(d, e.name)]);
const scripts = ['src', 'test', 'bench', 'scripts', 'tools'].flatMap((d) => walk(path.join(ROOT, d))).filter((f) => /\.m?js$/.test(f));
for (const f of scripts) {
  const rel = path.relative(ROOT, f);
  try { execFileSync(process.execPath, ['--check', f], { stdio: 'pipe' }); } catch (e) { fail(`${rel} does not parse: ${String(e.stderr).trim().split('\n')[0]}`); }
  if (!fs.readFileSync(f, 'utf8').replace(/^#!.*\n/, '').startsWith('// SPDX-License-Identifier: MPL-2.0\n')) fail(`${rel} has no SPDX licence header`);
}
// type declarations and pages carry the same header (in an HTML comment right after the doctype)
const others = ['types', 'test', 'bench', 'site'].flatMap((d) => walk(path.join(ROOT, d))).filter((f) => /\.(ts|html)$/.test(f));
for (const f of others) {
  const t = fs.readFileSync(f, 'utf8');
  if (!(f.endsWith('.ts') ? t.startsWith('// SPDX-License-Identifier: MPL-2.0\n') : /^<!doctype html>\n<!-- SPDX-License-Identifier: MPL-2.0\n/i.test(t))) fail(`${path.relative(ROOT, f)} has no SPDX licence header`);
}
const m = await import(pathToFileURL(path.join(ROOT, 'src/index.js')).href);
for (const name of ['createGpu', 'calibrate', 'defaultTune', 'gemm', 'attention', 'attn256', 'attnF']) if (typeof m[name] !== 'function') fail(`export ${name} is missing`);
const shaders = [
  ['gemm q4 f16', () => m.gemmCode({ w: 'q4', x: 'f16', epi: 'f16', BM: 64, BN: 128 })],
  ['gemm q8 resadd', () => m.gemmCode({ w: 'q8', x: 'rms', epi: 'resadd', BM: 64, BN: 128, gate: true })],
  ['gemm f16 conv3', () => m.gemmCode({ w: 'f16', x: 'conv3', epi: 'f16', BM: 64, BN: 128 })],
  ['attention BQ64', () => m.attnCode({ BQ: 64, pv16: true })],
  ['attnF head 128', () => m.attnFCode(Object.assign({ D: 128, flavour: 'p' }, m.ATTNF_P))],
  ['attnF head 256', () => m.attnFCode(Object.assign({ D: 256, flavour: 'v' }, m.ATTNF_V))],
];
for (const [name, gen] of shaders) {
  try { const r = gen(), code = typeof r === 'string' ? r : r.code; if (!/@compute/.test(code)) fail(`${name}: no compute entry point`); }
  catch (e) { fail(`${name}: ${e.message}`); }
}
console.log(bad ? `${bad} problem(s)` : `ok: ${scripts.length} scripts, ${others.length} declaration and page files, ${shaders.length} shader configurations`);
process.exit(bad ? 1 : 0);
