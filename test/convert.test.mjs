// SPDX-License-Identifier: MPL-2.0
// Copyright 2026 Joshua Hughes
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// Converter tests (Node, no GPU): a synthetic safetensors file with every supported dtype and tensor kind goes through the in-memory
// converter and the command line; the encodings, offsets, shards and bytes are checked against the encoders and the source values.
//   node test/convert.test.mjs
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { convertSafetensors, writeSafetensors, packQ4, packQ8, toFloat32, f2h, h2f, MANIFEST_FORMAT } from '../src/index.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
let bad = 0, n = 0;
const check = (ok, what) => { n++; if (!ok) { bad++; console.log('FAIL ', what); } };
function rng(seed) { let s = seed >>> 0 || 1; return () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return s / 4294967296 - 0.5; }; }
const rand = (k, seed, scale = 0.1) => { const r = rng(seed), o = new Float32Array(k); for (let i = 0; i < k; i++) o[i] = r() * scale; return o; };
const bytesOf = (a) => new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
const bf16 = (W) => { const u = new Uint32Array(Float32Array.from(W).buffer), h = new Uint16Array(W.length); for (let i = 0; i < W.length; i++) h[i] = (u[i] + 0x7fff + ((u[i] >>> 16) & 1)) >>> 16; return h; };
const f16 = (W) => Uint16Array.from(W, (v) => f2h(v));
const eq = (a, b) => a.byteLength === b.byteLength && a.every((v, i) => v === b[i]);

// ---- the source file: every dtype, matrices, convolutions, vectors, an embedding, an odd shape, an integer tensor
const S = {
  qkv: rand(192 * 256, 1), mlp: rand(128 * 128, 2), conv: rand(64 * 32 * 9, 3), conv3: rand(64 * 32 * 27, 4), norm: rand(256, 5, 2), emb: rand(100 * 64, 6), odd: rand(100 * 30, 7),
};
const fp8 = new Uint8Array(64 * 64); for (let i = 0; i < fp8.length; i++) fp8[i] = (i * 37) & 0x77;   // finite E4M3 codes only
const T = {
  'blocks.0.attn.qkv.weight': { dtype: 'BF16', shape: [192, 256], data: bytesOf(bf16(S.qkv)) },
  'blocks.0.mlp.weight': { dtype: 'F32', shape: [128, 128], data: bytesOf(S.mlp) },
  'decoder.conv.weight': { dtype: 'F16', shape: [64, 32, 3, 3], data: bytesOf(f16(S.conv)) },
  'decoder.conv3.weight': { dtype: 'F32', shape: [64, 32, 3, 3, 3], data: bytesOf(S.conv3) },
  'blocks.0.norm.weight': { dtype: 'F32', shape: [256], data: bytesOf(S.norm) },
  'embed_tokens.weight': { dtype: 'BF16', shape: [100, 64], data: bytesOf(bf16(S.emb)) },
  'odd.weight': { dtype: 'F32', shape: [100, 30], data: bytesOf(S.odd) },
  'fp8.weight': { dtype: 'F8_E4M3', shape: [64, 64], data: fp8 },
  'position_ids': { dtype: 'I64', shape: [8], data: new Uint8Array(64) },
};
const file = writeSafetensors(T, { format: 'pt' });

// ---- dtype decoding
check(toFloat32('F8_E4M3', new Uint8Array([0x38]))[0] === 1 && toFloat32('F8_E4M3', new Uint8Array([0xc0]))[0] === -2, 'F8_E4M3 decodes 0x38 = 1, 0xc0 = -2');
check(toFloat32('F8_E5M2', new Uint8Array([0x3c]))[0] === 1, 'F8_E5M2 decodes 0x3c = 1');
check(toFloat32('BF16', bytesOf(Uint16Array.of(0x3f80, 0xc000))).join() === '1,-2', 'BF16 decodes');
check(toFloat32('F16', bytesOf(Uint16Array.of(0x3c00, 0xc000))).join() === '1,-2', 'F16 decodes');

// ---- in memory, small shards so that shards roll over
const { manifest: m, shards } = await convertSafetensors(file, { bits: 4, keep: /embed/, shardBytes: 64 * 1024 });
check(m.format === MANIFEST_FORMAT, 'manifest format');
const want = { 'blocks.0.attn.qkv.weight': 'q4g32a.t64', 'blocks.0.mlp.weight': 'q4g32a.t64', 'decoder.conv.weight': 'q4g32a.t64', 'decoder.conv3.weight': 'q4g32a.t64',
  'blocks.0.norm.weight': 'f32', 'embed_tokens.weight': 'f16', 'odd.weight': 'f16', 'fp8.weight': 'q4g32a.t64' };
for (const [k, enc] of Object.entries(want)) check(m.tensors[k] && m.tensors[k].enc === enc, `${k} -> ${enc} (got ${m.tensors[k] && m.tensors[k].enc})`);
check(m.skipped.position_ids === 'I64' && !m.tensors.position_ids, 'integer tensors are skipped and listed');
check(m.shards.length > 1, `shards roll over at the size limit (${m.shards.length} shards)`);
for (const s of m.shards) check(shards[s.file] && shards[s.file].byteLength === s.bytes, `${s.file} is ${s.bytes} bytes`);
const slice = (name) => { const t = m.tensors[name]; return shards[t.file].subarray(t.offset, t.offset + t.bytes); };
for (const [k, t] of Object.entries(m.tensors)) check(t.offset % 256 === 0 && t.offset + t.bytes <= shards[t.file].byteLength, `${k} is 256-byte aligned inside its shard`);

// ---- bytes: each tensor equals the encoder's output for the decoded source values
const bf = (h) => toFloat32('BF16', bytesOf(h));
check(eq(slice('blocks.0.attn.qkv.weight'), packQ4(bf(bf16(S.qkv)), 192, 256)), 'BF16 matrix -> q4 bytes');
check(eq(slice('blocks.0.mlp.weight'), packQ4(S.mlp, 128, 128)), 'F32 matrix -> q4 bytes');
check(eq(slice('fp8.weight'), packQ4(toFloat32('F8_E4M3', fp8), 64, 64)), 'F8_E4M3 matrix -> q4 bytes');
check(eq(slice('blocks.0.norm.weight'), bytesOf(S.norm)), 'vector -> f32, bit-exact');
check(eq(slice('embed_tokens.weight'), bytesOf(Uint16Array.from(bf(bf16(S.emb)), (v) => f2h(v)))), 'embedding kept fp16 (--keep)');
check(eq(slice('odd.weight'), bytesOf(f16(S.odd))), 'odd shape -> fp16');

// ---- convolution layout: OIHW / OITHW -> OHWI / OTHWI rows (fp16, so every value is checkable)
const { manifest: m16, shards: s16 } = await convertSafetensors(file, { bits: 16 });
const view16 = (name) => { const t = m16.tensors[name], b = s16[t.file].slice(t.offset, t.offset + t.bytes); return new Uint16Array(b.buffer); };
for (const [name, src, taps] of [['decoder.conv.weight', S.conv, 9], ['decoder.conv3.weight', S.conv3, 27]]) {
  const got = view16(name), I = 32; let ok = m16.tensors[name].layout === (taps === 9 ? 'OHWI' : 'OTHWI');
  for (let o = 0; o < 64 && ok; o++) for (let i = 0; i < I; i++) for (let t = 0; t < taps; t++) if (got[o * I * taps + t * I + i] !== f2h(h2f(f2h(src[(o * I + i) * taps + t])))) { ok = false; break; }
  check(ok, `${name}: every value lands at (o, tap, i)`);
}
// ---- 8-bit
const { manifest: m8, shards: s8 } = await convertSafetensors(file, { bits: 8 });
const t8 = m8.tensors['blocks.0.mlp.weight'];
check(t8.enc === 'q8g64s.t64' && eq(s8[t8.file].subarray(t8.offset, t8.offset + t8.bytes), packQ8(S.mlp, 128, 128)), '--bits 8 -> q8 bytes');

// ---- the command line gives the same bytes
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kernels-convert-'));
try {
  fs.writeFileSync(path.join(dir, 'model.safetensors'), file);
  execFileSync(process.execPath, [path.join(ROOT, 'tools/convert.mjs'), path.join(dir, 'model.safetensors'), '--out', path.join(dir, 'out'), '--shard-mb', '0.0625'], { stdio: 'pipe' });
  const cm = JSON.parse(fs.readFileSync(path.join(dir, 'out', 'manifest.json'), 'utf8'));
  check(JSON.stringify(cm.tensors) === JSON.stringify(m.tensors), 'command line: same manifest');
  for (const s of m.shards) check(eq(new Uint8Array(fs.readFileSync(path.join(dir, 'out', s.file))), shards[s.file]), `command line: ${s.file} identical`);
} finally { fs.rmSync(dir, { recursive: true, force: true }); }

console.log(bad ? `${bad} of ${n} converter checks failed` : `converter: all ${n} checks passed`);
process.exit(bad ? 1 : 0);
