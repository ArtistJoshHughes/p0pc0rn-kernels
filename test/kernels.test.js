// SPDX-License-Identifier: MPL-2.0
// Copyright 2026 Joshua Hughes
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// Correctness tests: each kernel against a CPU reference on fixed-seed random data. run(gpu) is an async generator of
// {name, ok, cos, maxErr, note}. A pass needs cosine similarity > 0.9995 with the CPU result, or bit-identical output where the test
// says so (the time-chunked convolutions must equal the whole-clip convolution exactly).
import { toF16, fromF16, h2f, f2h, gemm, attention, attn256, attnF, attnFFits, ATTNF_P, ATTNF_V, packQ4, packQ8,
  convertSafetensors, writeSafetensors, uploadWeights } from '../src/index.js';

// ---- helpers
function rng(seed) { let s = seed >>> 0 || 1; return () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; }; }
function randn(n, seed, scale = 1) { const r = rng(seed), o = new Float32Array(n); for (let i = 0; i < n; i += 2) { const u = Math.max(1e-9, r()), v = r(), m = Math.sqrt(-2 * Math.log(u)) * scale; o[i] = m * Math.cos(2 * Math.PI * v); if (i + 1 < n) o[i + 1] = m * Math.sin(2 * Math.PI * v); } return o; }
const rf16 = (x) => h2f(f2h(x));   // round to fp16
const roundF16 = (a) => { const o = new Float32Array(a.length); for (let i = 0; i < a.length; i++) o[i] = rf16(a[i]); return o; };
export function compare(got, want) {
  let dot = 0, ga = 0, wa = 0, me = 0;
  for (let i = 0; i < want.length; i++) { const g = got[i], w = want[i]; dot += g * w; ga += g * g; wa += w * w; const e = Math.abs(g - w); if (!(e <= me)) me = Number.isFinite(e) ? Math.max(me, e) : Infinity; }
  const cos = dot / Math.sqrt(Math.max(1e-30, ga * wa));
  return { cos: Number.isFinite(cos) ? cos : 0, maxErr: me };
}
const same = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

// ---- the weight encodings (docs/FORMAT.md): the converter's own encoders, plus the dequantised weights for the CPU reference
const q4 = (W, N, K) => { const deq = new Float32Array(N * K); return { bytes: packQ4(W, N, K, deq), deq }; };
const q8 = (W, N, K) => { const deq = new Float32Array(N * K); return { bytes: packQ8(W, N, K, deq), deq }; };

// ---- CPU references
function cpuGemm(X, W, M, N, K) { const Y = new Float32Array(M * N); for (let m = 0; m < M; m++) for (let n = 0; n < N; n++) { let a = 0; const xo = m * K, wo = n * K; for (let k = 0; k < K; k++) a += X[xo + k] * W[wo + k]; Y[m * N + n] = a; } return Y; }
/** Q [Hq][Lq][D], K/V [Hkv][Lk][D] -> O [Lq][Hq*D]; mode 0 none, 1 causal + valid length (key j visible iff j <= min(i, nvalid-1)),
 *  2 valid length (j < nvalid); GQA: query head h reads kv head h / (Hq/Hkv); cap > 0: scores = cap * tanh(scores / cap) */
function cpuAttn(Q, Kt, V, { Hq, Hkv = Hq, Lq, Lk = Lq, D, mode = 0, nvalid = Lk, scale = 1 / Math.sqrt(D), cap = 0 }) {
  const O = new Float32Array(Lq * Hq * D), s = new Float32Array(Lk), grp = Hq / Hkv;
  for (let h = 0; h < Hq; h++) {
    const kh = Math.floor(h / grp);
    for (let i = 0; i < Lq; i++) {
      let mx = -Infinity;
      for (let j = 0; j < Lk; j++) {
        const vis = mode === 1 ? j <= Math.min(i, nvalid - 1) : mode === 2 ? j < nvalid : true;
        if (!vis) { s[j] = -Infinity; continue; }
        let a = 0; for (let d = 0; d < D; d++) a += Q[(h * Lq + i) * D + d] * Kt[(kh * Lk + j) * D + d];
        a *= scale; if (cap > 0) a = cap * Math.tanh(a / cap); s[j] = a; if (a > mx) mx = a;
      }
      let sum = 0; for (let j = 0; j < Lk; j++) { s[j] = s[j] === -Infinity ? 0 : Math.exp(s[j] - mx); sum += s[j]; }
      for (let d = 0; d < D; d++) { let a = 0; for (let j = 0; j < Lk; j++) a += s[j] * V[(kh * Lk + j) * D + d]; O[i * Hq * D + h * D + d] = a / sum; }
    }
  }
  return O;
}
/** 3x3x3 convolution over a T x H x W x C (THWC) clip, stride 1: time padding replicates the edge frames, space pads with zeros.
 *  W [N][27 * C] with k = (kt * 9 + ky * 3 + kx) * C + c  ->  Y [T * H * W][N] */
function cpuConv3(X, Wt, T, H, W, C, N) {
  const Y = new Float32Array(T * H * W * N), K = 27 * C;
  for (let t = 0; t < T; t++) for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) for (let n = 0; n < N; n++) {
    let a = 0;
    for (let kt = 0; kt < 3; kt++) for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) {
      const it = Math.min(T - 1, Math.max(0, t + kt - 1)), iy = y + ky - 1, ix = x + kx - 1;
      if (iy < 0 || ix < 0 || iy >= H || ix >= W) continue;
      const xo = ((it * H + iy) * W + ix) * C, wo = n * K + (kt * 9 + ky * 3 + kx) * C;
      for (let c = 0; c < C; c++) a += X[xo + c] * Wt[wo + c];
    }
    Y[((t * H + y) * W + x) * N + n] = a;
  }
  return Y;
}

const cases = [];
const test = (name, fn) => cases.push({ name, fn });
const read16 = async (gpu, b, n) => fromF16(new Uint16Array(await gpu.read(b, 0, n * 2)));

// ---- GEMM: three weight encodings x two accumulate modes (fp32 sums; fp16 partial sums per 32-k step, flushed to fp32)
for (const enc of ['f16', 'q4g32a.t64', 'q8g64s.t64']) for (const acc of ['f32', 'f16']) {
  test(`GEMM, ${enc} weights, ${acc === 'f32' ? 'fp32 sums' : 'fp16 partial sums'}`, async (gpu) => {
    const M = 32, N = 128, K = 256, X = roundF16(randn(M * K, 7)), W0 = randn(N * K, 11, 0.05);
    let wb, Wd;
    if (enc === 'f16') { Wd = roundF16(W0); wb = gpu.upload(toF16(Wd)); }
    else { const p = enc.startsWith('q4') ? q4(W0, N, K) : q8(W0, N, K); Wd = p.deq; wb = gpu.upload(p.bytes); }
    const xb = gpu.upload(toF16(X)), yb = gpu.buffer(M * N * 2);
    try {
      gemm(gpu, { w: { enc, buf: wb, off: 0, bytes: wb._pf, rows: N, cols: K }, x: { buf: xb }, y: { buf: yb }, M, N, K, acc });
      return compare(await read16(gpu, yb, M * N), cpuGemm(X, Wd, M, N, K));
    } finally { [wb, xb, yb].forEach((b) => gpu.free(b)); }
  });
}
test('GEMM, partial tiles (M 33, N 192)', async (gpu) => {
  const M = 33, N = 192, K = 256, X = roundF16(randn(M * K, 17)), Wd = roundF16(randn(N * K, 19, 0.05));
  const wb = gpu.upload(toF16(Wd)), xb = gpu.upload(toF16(X)), yb = gpu.buffer(M * N * 2);
  try {
    gemm(gpu, { w: { enc: 'f16', buf: wb, off: 0, bytes: wb._pf, rows: N, cols: K }, x: { buf: xb }, y: { buf: yb }, M, N, K });
    return compare(await read16(gpu, yb, M * N), cpuGemm(X, Wd, M, N, K));
  } finally { [wb, xb, yb].forEach((b) => gpu.free(b)); }
});

// ---- 3x3x3 convolution (implicit GEMM over THWC) and its time-chunked forms
const C3 = { T: 5, H: 6, W: 6, C: 32, N: 128 };
function conv3Setup(gpu) {
  const { T, H, W, C, N } = C3, X = roundF16(randn(T * H * W * C, 31)), Wt = roundF16(randn(N * 27 * C, 33, 0.03));
  return { X, Wt, xb: gpu.upload(toF16(X)), wb: gpu.upload(toF16(Wt)) };
}
/** output frames [t0, t0 + nf) of the clip; the input buffer starts at frame padL, or is a ring of ks frames (ks > 1) */
function conv3(gpu, s, x, y, t0, nf, padL = 0, ks = 1) {
  const { T, H, W, C, N } = C3;
  gemm(gpu, { w: { enc: 'f16', buf: s.wb, off: 0, bytes: s.wb._pf, rows: N, cols: 27 * C }, x, y, M: nf * H * W, N, K: 27 * C,
    conv: { kind: 'conv3', Hi: H, Wi: W, Ci: C, Ho: H, Wo: W, stride: T, padT: t0, padL, ks } });
}
test('3x3x3 convolution (THWC clip; edge frames repeat, zero padding in space)', async (gpu) => {
  const { T, H, W, C, N } = C3, s = conv3Setup(gpu), yb = gpu.buffer(T * H * W * N * 2);
  try {
    conv3(gpu, s, { buf: s.xb }, { buf: yb }, 0, T);
    return compare(await read16(gpu, yb, T * H * W * N), cpuConv3(s.X, s.Wt, T, H, W, C, N));
  } finally { [s.xb, s.wb, yb].forEach((b) => gpu.free(b)); }
});
for (const tile of [{ BM: 128, BN: 128 }, { BM: 64, BN: 64 }, { BM: 32, BN: 128 }, { BM: 128, BN: 64 }]) {
  test(`3x3x3 convolution, tile ${tile.BM}x${tile.BN} (what the tuning table can pick)`, async (gpu) => {
    const { T, H, W, C, N } = C3, s = conv3Setup(gpu), yb = gpu.buffer(T * H * W * N * 2);
    try {
      gemm(gpu, { w: { enc: 'f16', buf: s.wb, off: 0, bytes: s.wb._pf, rows: N, cols: 27 * C }, x: { buf: s.xb }, y: { buf: yb }, M: T * H * W, N, K: 27 * C, tile,
        conv: { kind: 'conv3', Hi: H, Wi: W, Ci: C, Ho: H, Wo: W, stride: T, padT: 0, padL: 0, ks: 1 } });
      return compare(await read16(gpu, yb, T * H * W * N), cpuConv3(s.X, s.Wt, T, H, W, C, N));
    } finally { [s.xb, s.wb, yb].forEach((b) => gpu.free(b)); }
  });
}
test('3x3x3 convolution in time chunks of 2 frames = the whole clip (bit-identical)', async (gpu) => {
  const { T, H, W, C, N } = C3, s = conv3Setup(gpu), n = T * H * W * N, ya = gpu.buffer(n * 2), yb = gpu.buffer(n * 2), fx = H * W * C * 2, fy = H * W * N * 2;
  try {
    conv3(gpu, s, { buf: s.xb }, { buf: ya }, 0, T);
    for (let t0 = 0; t0 < T; t0 += 2) { const padL = Math.max(0, t0 - 1); conv3(gpu, s, { buf: s.xb, off: padL * fx }, { buf: yb, off: t0 * fy }, t0, Math.min(2, T - t0), padL); }
    const a = new Uint16Array(await gpu.read(ya, 0, n * 2)), b = new Uint16Array(await gpu.read(yb, 0, n * 2));
    return { ok: same(a, b), ...compare(fromF16(b), fromF16(a)), note: same(a, b) ? 'bit-identical' : 'DIFFERS' };
  } finally { [s.xb, s.wb, ya, yb].forEach((b) => gpu.free(b)); }
});
test('3x3x3 convolution from a ring of 3 frames = the whole clip (bit-identical)', async (gpu) => {
  const { T, H, W, C, N } = C3, s = conv3Setup(gpu), n = T * H * W * N, ya = gpu.buffer(n * 2), yb = gpu.buffer(n * 2), fx = H * W * C, fy = H * W * N * 2;
  const ring = gpu.buffer(3 * fx * 2), X16 = toF16(s.X);
  try {
    conv3(gpu, s, { buf: s.xb }, { buf: ya }, 0, T);
    for (let t = 0; t < T; t++) {
      for (const g of new Set([Math.max(0, t - 1), t, Math.min(T - 1, t + 1)])) gpu.device.queue.writeBuffer(ring, (g % 3) * fx * 2, X16, g * fx, fx);
      conv3(gpu, s, { buf: ring }, { buf: yb, off: t * fy }, t, 1, 0, 3);
      await gpu.done();   // the ring is rewritten for the next frame
    }
    const a = new Uint16Array(await gpu.read(ya, 0, n * 2)), b = new Uint16Array(await gpu.read(yb, 0, n * 2));
    return { ok: same(a, b), ...compare(fromF16(b), fromF16(a)), note: same(a, b) ? 'bit-identical' : 'DIFFERS' };
  } finally { [s.xb, s.wb, ya, yb, ring].forEach((b) => gpu.free(b)); }
});

// ---- attention: the standard kernel (any device) and the phone-tuned kernels (32 KB of workgroup memory), every mask mode, GQA
const ATT = [
  { label: 'no mask', Hq: 2, Hkv: 2, Lq: 96, Lk: 96, mode: 0 },
  { label: 'causal + valid length 80, GQA 4:2', Hq: 4, Hkv: 2, Lq: 96, Lk: 96, mode: 1, nvalid: 80 },
  { label: 'valid length 77 of 100 keys, 64 queries', Hq: 2, Hkv: 2, Lq: 64, Lk: 100, mode: 2, nvalid: 77 },
];
function attnSetup(gpu, c, D, seed, qs = 0.5) {
  const Q = roundF16(randn(c.Hq * c.Lq * D, seed, qs)), Kt = roundF16(randn(c.Hkv * c.Lk * D, seed + 1, qs)), V = roundF16(randn(c.Hkv * c.Lk * D, seed + 2));
  return { Q, Kt, V, qb: gpu.upload(toF16(Q)), kb: gpu.upload(toF16(Kt)), vb: gpu.upload(toF16(V)), ob: gpu.buffer(c.Lq * c.Hq * D * 2) };
}
const attnArgs = (s, c, D) => ({ q: { buf: s.qb }, k: { buf: s.kb }, v: { buf: s.vb }, o: { buf: s.ob, ld: c.Hq * D }, Lq: c.Lq, Lk: c.Lk, Hq: c.Hq, Hkv: c.Hkv, mode: c.mode, nvalid: c.nvalid });
for (const c of ATT) {
  test(`attention, head 128, standard kernel, ${c.label}`, async (gpu) => {
    const s = attnSetup(gpu, c, 128, 21);
    try {
      attention(gpu, Object.assign(attnArgs(s, c, 128), { cfgF: null }));
      return compare(await read16(gpu, s.ob, c.Lq * c.Hq * 128), cpuAttn(s.Q, s.Kt, s.V, Object.assign({ D: 128 }, c)));
    } finally { [s.qb, s.kb, s.vb, s.ob].forEach((b) => gpu.free(b)); }
  });
  test(`attention, head 256, standard kernel, ${c.label}`, async (gpu) => {
    const s = attnSetup(gpu, c, 256, 41), t = gpu.tune || (gpu.tune = {}), keep = t.attnF256;
    try {
      t.attnF256 = null; attn256(gpu, attnArgs(s, c, 256));
      return compare(await read16(gpu, s.ob, c.Lq * c.Hq * 256), cpuAttn(s.Q, s.Kt, s.V, Object.assign({ D: 256 }, c)));
    } finally { t.attnF256 = keep; [s.qb, s.kb, s.vb, s.ob].forEach((b) => gpu.free(b)); }
  });
  for (const [D, F] of [[128, ATTNF_P], [256, ATTNF_V]]) {
    test(`attention, head ${D}, phone-tuned kernel (attnF), ${c.label}`, async (gpu) => {
      const cfg = Object.assign({ D, flavour: D === 128 ? 'p' : 'v' }, F);
      if (!attnFFits(gpu, cfg)) return { ok: true, note: 'skipped: needs 32 KB of workgroup memory' };
      const s = attnSetup(gpu, c, D, 61);
      try {
        attnF(gpu, attnArgs(s, c, D), cfg);
        return compare(await read16(gpu, s.ob, c.Lq * c.Hq * D), cpuAttn(s.Q, s.Kt, s.V, Object.assign({ D }, c)));
      } finally { [s.qb, s.kb, s.vb, s.ob].forEach((b) => gpu.free(b)); }
    });
  }
}
test('attention, head 256, phone-tuned kernel, soft-capped scores (cap 1)', async (gpu) => {
  const c = { Hq: 4, Hkv: 2, Lq: 96, Lk: 96, mode: 1, nvalid: 96 }, D = 256, cfg = Object.assign({ D, flavour: 'v' }, ATTNF_V);
  if (!attnFFits(gpu, cfg)) return { ok: true, note: 'skipped: needs 32 KB of workgroup memory' };
  const s = attnSetup(gpu, c, D, 71, 1.0);
  try {
    attnF(gpu, Object.assign(attnArgs(s, c, D), { cap: 1 }), cfg);
    return compare(await read16(gpu, s.ob, c.Lq * c.Hq * D), cpuAttn(s.Q, s.Kt, s.V, Object.assign({ D, cap: 1 }, c)));
  } finally { [s.qb, s.kb, s.vb, s.ob].forEach((b) => gpu.free(b)); }
});

// ---- converted model files: safetensors (as PyTorch writes them) -> the converter -> the GPU, end to end
const bf16 = (W) => { const u = new Uint32Array(Float32Array.from(W).buffer), h = new Uint16Array(W.length); for (let i = 0; i < W.length; i++) h[i] = (u[i] + 0x7fff + ((u[i] >>> 16) & 1)) >>> 16; return h; };
const bf16f = (h) => { const u = new Uint32Array(h.length); for (let i = 0; i < h.length; i++) u[i] = h[i] << 16; return new Float32Array(u.buffer); };
test('converted model: a BF16 linear layer, converted to 4-bit, through the GEMM', async (gpu) => {
  const M = 32, N = 192, K = 512, Wb = bf16(randn(N * K, 81, 0.05)), X = roundF16(randn(M * K, 83));
  const file = writeSafetensors({ 'layers.0.mlp.up_proj.weight': { dtype: 'BF16', shape: [N, K], data: new Uint8Array(Wb.buffer) } });
  const { manifest, shards } = await convertSafetensors(file, { bits: 4 }), model = uploadWeights(gpu, manifest, shards);
  const w = model.tensor('layers.0.mlp.up_proj.weight'), deq = new Float32Array(N * K); packQ4(bf16f(Wb), N, K, deq);
  const xb = gpu.upload(toF16(X)), yb = gpu.buffer(M * N * 2);
  try {
    if (w.enc !== 'q4g32a.t64') return { ok: false, note: 'encoded as ' + w.enc };
    gemm(gpu, { w, x: { buf: xb }, y: { buf: yb }, M, N, K });
    return compare(await read16(gpu, yb, M * N), cpuGemm(X, deq, M, N, K));
  } finally { model.free(); [xb, yb].forEach((b) => gpu.free(b)); }
});
test('converted model: a 3x3x3 conv weight in PyTorch layout [O, I, kt, kh, kw], through the 3D convolution', async (gpu) => {
  const { T, H, W, C, N } = C3, src = roundF16(randn(N * C * 27, 87, 0.03)), X = roundF16(randn(T * H * W * C, 89));
  const file = writeSafetensors({ 'decoder.conv.weight': { dtype: 'F16', shape: [N, C, 3, 3, 3], data: new Uint8Array(toF16(src).buffer) } });
  const { manifest, shards } = await convertSafetensors(file, { bits: 16 }), model = uploadWeights(gpu, manifest, shards), w = model.tensor('decoder.conv.weight');
  const ref = new Float32Array(N * 27 * C);   // the reference reads the PyTorch layout directly: W[n][c][kt][ky][kx]
  for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) for (let tap = 0; tap < 27; tap++) ref[n * 27 * C + tap * C + c] = src[(n * C + c) * 27 + tap];
  const xb = gpu.upload(toF16(X)), yb = gpu.buffer(T * H * W * N * 2);
  try {
    gemm(gpu, { w, x: { buf: xb }, y: { buf: yb }, M: T * H * W, N, K: 27 * C, conv: { kind: 'conv3', Hi: H, Wi: W, Ci: C, Ho: H, Wo: W, stride: T, padT: 0, padL: 0, ks: 1 } });
    return compare(await read16(gpu, yb, T * H * W * N), cpuConv3(X, ref, T, H, W, C, N));
  } finally { model.free(); [xb, yb].forEach((b) => gpu.free(b)); }
});

/** every test in order; a result without an explicit ok passes at cosine similarity > 0.9995 */
export async function* run(gpu) {
  for (const t of cases) {
    try {
      const r = await t.fn(gpu);
      yield Object.assign({ name: t.name }, r, { ok: r.ok !== undefined ? r.ok : r.cos > 0.9995 });
    } catch (e) { yield { name: t.name, ok: false, error: String((e && e.message) || e) }; }
  }
}
export const count = () => cases.length;
