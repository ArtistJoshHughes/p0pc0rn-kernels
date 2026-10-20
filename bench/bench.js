// SPDX-License-Identifier: MPL-2.0
// Copyright 2026 Joshua Hughes
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// Benchmark: the kernels at the sizes the p0pc0rn picture and clip models run (FLUX.2 [klein] 4B at 1280x720 = 4112 tokens; SANA-Video
// anchor attention; the LTX-2.3 decoder's 512-channel 3D convolutions), each new kernel next to the kernel it replaced.
// Method (what the phone work taught): cases are INTERLEAVED (round 1 runs every case once, then round 2, ...), so heat affects them
// alike; each sample repeats a case until it takes >= 60 ms; the result is the median over rounds; a fixed GEMM 'canary' runs before and
// after, so a hot device shows up as a lower canary rate. Phones slow down by 40-50% when hot: compare kernels within one run.
import { gemm, attention, attnF, attnFFits, attn256, ATTNF_P, ATTNF_V } from '../src/index.js';

function lcg(seed) { let s = seed >>> 0 || 1; return () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return s >>> 0; }; }
/** fp16 values of magnitude 0.25..0.5 with random signs (safe for any product chain) */
function h16(n, seed) { const r = lcg(seed), h = new Uint16Array(n); for (let i = 0; i < n; i++) h[i] = (r() & 0x83ff) | 0x3400; return h; }
/** q4g32a.t64 records with random codes and small fixed scales / minimums */
function q4(N, K, seed) {
  const recs = (N / 64) * (K / 32), u = new Uint32Array(recs * 320), r = lcg(seed);
  for (let i = 0; i < u.length; i++) u[i] = r();
  for (let k = 0; k < recs; k++) for (let j = 0; j < 64; j++) u[k * 320 + 256 + j] = 0x2000 | ((0xA800 | (j & 0x3f)) << 16);
  return u;
}

/** the cases; inputs are shared where kernels compare like for like (each attention pair reads the same q / k / v) */
export function cases(gpu) {
  const C = [], add = (group, name, flops, run) => C.push({ group, name, flops, run });
  const bufs = new Map(), get = (key, make) => { if (!bufs.has(key)) bufs.set(key, make()); return bufs.get(key); };
  // ---- GEMM: Klein 4B projections at 1280x720 (4112 tokens), 4-bit and fp16 weights
  const M = 4112, X = () => get('x', () => gpu.upload(h16(M * 3072, 5))), Y = () => get('y', () => gpu.buffer(M * 9216 * 2));
  for (const [label, N, K] of [['projection 4112 x 3072 x 3072', 3072, 3072], ['MLP 4112 x 9216 x 3072', 9216, 3072]]) {
    for (const enc of ['q4g32a.t64', 'f16']) {
      const W = get(`w${N}${enc}`, () => enc === 'f16' ? gpu.upload(h16(N * K, 3)) : gpu.upload(q4(N, K, 3)));
      add('Matrix multiply', `${label}, ${enc === 'f16' ? 'fp16' : '4-bit'} weights`, 2 * M * N * K,
        () => gemm(gpu, { w: { enc, buf: W, off: 0, bytes: W._pf, rows: N, cols: K }, x: { buf: X() }, y: { buf: Y() }, M, N, K }));
    }
  }
  // ---- attention, head 128: picture attention at 1280x720 (8 of the 24 heads keep a sample short)
  {
    const L = 4112, H = 8, D = 128, f = 4 * L * L * D * H, b = ['q', 'k', 'v'].map((n, i) => get('a128' + n, () => gpu.upload(h16(H * L * D, 7 + i))));
    const o = get('a128o', () => gpu.buffer(L * H * D * 2));
    const args = () => ({ q: { buf: b[0] }, k: { buf: b[1] }, v: { buf: b[2] }, o: { buf: o, ld: H * D }, Lq: L, Lk: L, Hq: H, Hkv: H, mode: 0 });
    add('Attention, head 128', '4112 tokens x 8 heads, standard kernel', f, () => attention(gpu, Object.assign(args(), { cfgF: null })));
    const cfg = Object.assign({ D, flavour: 'p', rate: gpu.rate.attn }, ATTNF_P);
    if (attnFFits(gpu, cfg)) add('Attention, head 128', '4112 tokens x 8 heads, phone-tuned (attnF)', f, () => attnF(gpu, args(), cfg));
  }
  // ---- attention, head 256: SANA-Video anchor attention (10 heads of 256) over a 480p clip's 4096 tokens
  {
    const L = 4096, H = 10, D = 256, f = 4 * L * L * D * H, b = ['q', 'k', 'v'].map((n, i) => get('a256' + n, () => gpu.upload(h16(H * L * D, 17 + i))));
    const o = get('a256o', () => gpu.buffer(L * H * D * 2));
    const args = () => ({ q: { buf: b[0] }, k: { buf: b[1] }, v: { buf: b[2] }, o: { buf: o, ld: H * D }, Lq: L, Lk: L, Hq: H, Hkv: H, mode: 0 });
    add('Attention, head 256', '4096 tokens x 10 heads, standard kernel', f, () => {
      const t = gpu.tune || (gpu.tune = {}), keep = t.attnF256; t.attnF256 = null;
      try { attn256(gpu, args()); } finally { t.attnF256 = keep; }
    });
    const cfg = Object.assign({ D, flavour: 'v', rate: gpu.rate.attn }, ATTNF_V);
    if (attnFFits(gpu, cfg)) add('Attention, head 256', '4096 tokens x 10 heads, phone-tuned (attnF)', f, () => attnF(gpu, args(), cfg));
  }
  // ---- 3x3x3 convolution: an LTX-2.3 decoder res-block conv (512 -> 512 channels, 4 frames of 32 x 32)
  {
    const T = 4, H = 32, W = 32, Ci = 512, N = 512, K = 27 * Ci, Mc = T * H * W;
    const Wt = get('cw', () => gpu.upload(h16(N * K, 23))), Xc = get('cx', () => gpu.upload(h16(Mc * Ci, 29))), Yc = get('cy', () => gpu.buffer(Mc * N * 2));
    add('3D convolution', '3x3x3, 512 -> 512 channels, 4 x 32 x 32 voxels', 2 * Mc * N * K,
      () => gemm(gpu, { w: { enc: 'f16', buf: Wt, off: 0, bytes: Wt._pf, rows: N, cols: K }, x: { buf: Xc }, y: { buf: Yc }, M: Mc, N, K,
        conv: { kind: 'conv3', Hi: H, Wi: W, Ci, Ho: H, Wo: W, stride: T, padT: 0, padL: 0, ks: 1 } }));
  }
  X(); Y();
  return { cases: C, free: () => { for (const b of bufs.values()) gpu.free(b); } };
}

async function sample(gpu, run, minMs = 60) {
  run(); await gpu.done();   // warm-up: pipelines compiled, caches hot
  let reps = 1;
  for (;;) {
    const t0 = performance.now(); for (let i = 0; i < reps; i++) run(); await gpu.done();
    const ms = performance.now() - t0;
    if (ms >= minMs || reps >= 64) return ms / reps;
    reps = Math.min(64, Math.ceil(reps * Math.max(2, (minMs * 1.2) / Math.max(ms, 1))));
  }
}
const median = (a) => { const s = [...a].sort((x, y) => x - y), n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };

/** interleaved rounds over every case; returns {canary: {startGflops, endGflops}, results: [{group, name, ms, gflops}]} */
export async function runBench(gpu, { rounds = 5, onStep = () => {} } = {}) {
  const { cases: C, free } = cases(gpu), times = C.map(() => []);
  const canary = async () => Math.round(C[0].flops / (await sample(gpu, C[0].run)) / 1e6);
  try {
    onStep('canary'); const start = await canary();
    for (let r = 0; r < rounds; r++) for (let i = 0; i < C.length; i++) { onStep(`round ${r + 1} of ${rounds}: ${C[i].name}`); times[i].push(await sample(gpu, C[i].run)); }
    onStep('canary'); const end = await canary();
    const results = C.map((c, i) => { const ms = median(times[i]); return { group: c.group, name: c.name, ms: +ms.toFixed(2), gflops: Math.round(c.flops / ms / 1e6), ok: true }; });
    return { canary: { startGflops: start, endGflops: end }, results };
  } finally { free(); }
}
