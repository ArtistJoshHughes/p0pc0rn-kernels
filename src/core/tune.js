// SPDX-License-Identifier: MPL-2.0
// Copyright 2026 Joshua Hughes
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// Device set-up the kernels expect: measured throughput (dispatch sizing) and the default kernel choices (tune).
//   calibrate(gpu)   measures GEMM (fp32 and fp16-partial accumulation) and attention throughput, so every dispatch stays under the
//                    ~0.2 s watchdog guard (gpu.opts.maxDispatchSeconds); sets gpu.rate {mm, mm32, mm16, attn, mem} and gpu.calib.
//   defaultTune(gpu) the choices the p0pc0rn engines make: fp16 partial sums per 32-k step in the GEMM, the phone-tuned flash attention
//                    (attnF) for head size 128 and 256 where the device has 32 KB of workgroup memory.
'use strict';
import { gemm } from '../ops/gemm.js';
import { attention, ATTNF_P, ATTNF_V } from '../ops/attention.js';

/** Measured best settings per GPU family, from shared benchmark results (bench/index.html, "Tune for this device"). An entry applies
 *  when every field of `match` occurs (case-insensitive) in the adapter's info; its `tune` fields replace the defaults:
 *  {acc, gemm: {q4|q8|f16|conv|conv3: {BM, BN}}, attnF, attnF256}. The defaults themselves were tuned on an iPhone 16 Pro Max
 *  (Safari 26.6.1). Each entry names the sweep it came from. */
export const TUNE_TABLE = [
  { match: { vendor: 'amd', architecture: 'rdna-2' }, tune: { gemm: { conv3: { BM: 128, BN: 64 } } },
    source: 'AMD Radeon integrated graphics (RDNA 2), Edge, Oct 2026: 3x3x3 convolution with 128x64 tiles x1.93 at 512 and 256 channels, ' +
      'x1.16 at 128 (bit-identical output); GEMM and attention defaults were already fastest' },
];

/** the entry of TUNE_TABLE for this adapter, or null */
export function tuneEntry(g, table = TUNE_TABLE) {
  const info = g.info || {};
  return table.find((e) => Object.entries(e.match || {}).every(([k, v]) => String(info[k] || '').toLowerCase().includes(String(v).toLowerCase()))) || null;
}

export function defaultTune(g, o = {}) {
  const t = (tuneEntry(g, o.table) || {}).tune || {};
  g.tune = { acc: o.acc || t.acc || 'f16', vaeAcc: o.vaeAcc || 'f16', attn: o.attn || { BQ: 64, sg: g.subgroups && g.sgMin >= 8, pv16: true },
    attnF: o.attnF !== undefined ? o.attnF : t.attnF || ATTNF_P, attnF256: o.attnF256 !== undefined ? o.attnF256 : t.attnF256 || ATTNF_V };
  if (o.gemm || t.gemm) g.tune.gemm = Object.assign({}, t.gemm, o.gemm);
  return g.tune;
}

/** measure GEMM and attention throughput so every dispatch stays under the ~0.2 s watchdog guard (starts conservative) */
export async function calibrate(g) {
  g.rate = { mm: 0.08e12, attn: 0.03e12, mem: 10e9 };
  const M = 1024, N = 3072, K = 3072, recs = (N / 64) * (K / 32), u = new Uint32Array(recs * 320);
  let s = 12345; for (let i = 0; i < u.length; i++) { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; u[i] = s >>> 0; }
  for (let r = 0; r < recs; r++) for (let j = 0; j < 64; j++) u[r * 320 + 256 + j] = 0x2000 | ((0xA800 | (j & 0x3f)) << 16);
  const h16 = (n) => { const h = new Uint16Array(n); for (let i = 0; i < n; i++) { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; h[i] = (s & 0x83ff) | 0x3400; } return h; };
  const W = g.upload(u), X = g.upload(h16(M * K)), Y = g.buffer(M * N * 2), L = 1024, H = 8;
  const Q = g.upload(h16(H * L * 128)), O = g.buffer(L * H * 256);
  const time = async (fn, reps) => { fn(); await g.done(); const t0 = performance.now(); for (let i = 0; i < reps; i++) fn(); await g.done(); return (performance.now() - t0) / 1000 / reps; };
  try {
    const mm = (acc) => time(() => gemm(g, { w: { buf: W, off: 0, bytes: W._pf, enc: 'q4g32a.t64', rows: N, cols: K }, x: { buf: X }, M, N, K, epi: 'f16', y: { buf: Y }, acc }), 2);
    const t32 = await mm('f32'); g.rate.mm32 = Math.max(0.03e12, 0.8 * (2 * M * N * K) / t32);
    const t16 = await mm('f16'); g.rate.mm16 = Math.max(0.03e12, 0.8 * (2 * M * N * K) / t16);
    g.rate.mm = Math.min(g.rate.mm32, g.rate.mm16);
    const ta = await time(() => attention(g, { q: { buf: Q }, k: { buf: Q }, v: { buf: Q }, o: { buf: O }, Lq: L, Lk: L, Hq: H, Hkv: H, mode: 0 }), 2);
    g.rate.attn = Math.max(0.015e12, 0.8 * (4 * L * L * 128 * H) / ta);
    g.calib = { gemmF32Gflops: Math.round(2 * M * N * K / t32 / 1e9), gemmF16Gflops: Math.round(2 * M * N * K / t16 / 1e9), attnGflops: Math.round(4 * L * L * 128 * H / ta / 1e9) };
  } finally { [W, X, Y, Q, O].forEach((b) => g.free(b)); }
  return g.calib;
}
