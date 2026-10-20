// SPDX-License-Identifier: MPL-2.0
// Copyright 2026 Joshua Hughes
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// Tuning sweep: on this device, time several settings of each kernel (GEMM / 3D convolution tiles and accumulation, attention tile
// shapes and precision), each checked against the default setting's output first. The fastest correct setting per kernel is what this
// device would want in its tuning table (src/core/tune.js); shared results build those tables for each GPU family.
import { gemm, attnF, attnFCfg, attnFFits, ATTNF_P, ATTNF_V } from '../src/index.js';

function lcg(seed) { let s = seed >>> 0 || 1; return () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return s >>> 0; }; }
function h16(n, seed) { const r = lcg(seed), h = new Uint16Array(n); for (let i = 0; i < n; i++) h[i] = (r() & 0x83ff) | 0x3400; return h; }
function q4(N, K, seed) {
  const recs = (N / 64) * (K / 32), u = new Uint32Array(recs * 320), r = lcg(seed);
  for (let i = 0; i < u.length; i++) u[i] = r();
  for (let k = 0; k < recs; k++) for (let j = 0; j < 64; j++) u[k * 320 + 256 + j] = 0x2000 | ((0xA800 | (j & 0x3f)) << 16);
  return u;
}
const h2f = (h) => { const s = h & 0x8000 ? -1 : 1, e = (h >>> 10) & 31, m = h & 1023; return e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15); };
/** relative L2 difference of two fp16 buffers' contents */
function rel(a, b) { let d = 0, n = 0; for (let i = 0; i < a.length; i++) { const x = h2f(a[i]), y = h2f(b[i]); d += (x - y) ** 2; n += y * y; } return Number.isFinite(d) ? Math.sqrt(d / Math.max(n, 1e-30)) : Infinity; }

const TILES = [{ BM: 64, BN: 128 }, { BM: 128, BN: 128 }, { BM: 64, BN: 64 }, { BM: 32, BN: 128 }, { BM: 128, BN: 64 }];
const ACCS = ['f16', 'f32'];
const tileName = (t) => `${t.BM}x${t.BN}`;

/** the kernels and their settings; the first setting of each is the default (what defaultTune picks). Each family's buffers are made
 *  by its make() only when the sweep reaches it: {run, out, n, bufs} */
export function families(gpu) {
  const F = [];
  {   // GEMM, 4-bit weights, a Klein 4B projection at 1280x720
    const M = 4112, N = 3072, K = 3072;
    const settings = TILES.flatMap((t) => ACCS.map((acc) => ({ label: `tile ${tileName(t)}, ${acc === 'f16' ? 'fp16 partial sums' : 'fp32 sums'}`, cfg: { tile: t, acc } })));
    F.push({ name: 'Matrix multiply, 4-bit weights, 4112 x 3072 x 3072', key: 'gemm.q4', flops: 2 * M * N * K, tol: 4e-3, settings, make: () => {
      const io = { w: gpu.upload(q4(N, K, 3)), x: gpu.upload(h16(M * K, 5)), y: gpu.buffer(M * N * 2) };
      return { out: io.y, n: M * N, bufs: Object.values(io),
        run: (c) => gemm(gpu, { w: { enc: 'q4g32a.t64', buf: io.w, off: 0, bytes: io.w._pf, rows: N, cols: K }, x: { buf: io.x }, y: { buf: io.y }, M, N, K, tile: c.tile, acc: c.acc }) };
    } });
  }
  // 3x3x3 convolutions at the LTX-2.3 decoder's three channel widths (~58 GFLOP each), fp16 partial sums (the library default)
  for (const [C, S] of [[512, 32], [256, 64], [128, 128]]) {
    const T = 4, H = S, W = S, K = 27 * C, M = T * H * W;
    const settings = TILES.map((t) => ({ label: `tile ${tileName(t)}`, cfg: { tile: t, acc: 'f16' } }));
    F.push({ name: `3D convolution, ${C} -> ${C} channels, 4 x ${S} x ${S}`, key: `conv3.${C}`, flops: 2 * M * C * K, tol: 4e-3, settings, make: () => {
      const io = { w: gpu.upload(h16(C * K, 23)), x: gpu.upload(h16(M * C, 29)), y: gpu.buffer(M * C * 2) };
      return { out: io.y, n: M * C, bufs: Object.values(io),
        run: (c) => gemm(gpu, { w: { enc: 'f16', buf: io.w, off: 0, bytes: io.w._pf, rows: C, cols: K }, x: { buf: io.x }, y: { buf: io.y }, M, N: C, K, tile: c.tile, acc: c.acc,
          conv: { kind: 'conv3', Hi: H, Wi: W, Ci: C, Ho: H, Wo: W, stride: T, padT: 0, padL: 0, ks: 1 } }) };
    } });
  }
  for (const [D, base, L, H] of [[128, ATTNF_P, 4112, 4], [256, ATTNF_V, 4096, 4]]) {   // the phone-tuned attention
    const fl = D === 128 ? 'p' : 'v', mk = (o) => Object.assign({ D, flavour: fl, rate: gpu.rate.attn }, base, o);
    const shapes = D === 128 ? [{}, { BQ: 32, BK: 128, TK: 8, PQ: 8, PD: 4 }, { BQ: 64, BK: 32, TK: 4, PQ: 4, PD: 16 }] : [{}, { BQ: 32, BK: 64, TK: 4, PQ: 8, PD: 8 }];
    const prec = [{ label: 'fp16 partial sums', o: {} }, { label: 'fp32 scores', o: { s16: false } }, { label: 'fp32 P x V', o: { pv16: false } }, { label: 'no interleave', o: { il: false } }];
    const settings = [];
    for (const s of shapes) for (const p of prec) {
      const cfg = mk(Object.assign({}, s, p.o));
      try { attnFCfg(cfg); } catch (e) { continue; }   // a shape that does not fit 32 KB or has mismatched thread counts
      if (!attnFFits(gpu, cfg)) continue;
      settings.push({ label: `${cfg.BQ} queries x ${cfg.BK} keys, ${p.label}`, cfg });
    }
    if (!settings.length) continue;
    F.push({ name: `Attention, head ${D}, ${L} tokens x ${H} heads (attnF)`, key: `attnF.${D}`, flops: 4 * L * L * D * H, tol: 6e-3, settings, make: () => {
      const io = { q: gpu.upload(h16(H * L * D, 7)), k: gpu.upload(h16(H * L * D, 8)), v: gpu.upload(h16(H * L * D, 9)), o: gpu.buffer(L * H * D * 2) };
      return { out: io.o, n: L * H * D, bufs: Object.values(io),
        run: (c) => attnF(gpu, { q: { buf: io.q }, k: { buf: io.k }, v: { buf: io.v }, o: { buf: io.o, ld: H * D }, Lq: L, Lk: L, Hq: H, Hkv: H, mode: 0 }, c) };
    } });
  }
  return F;
}

async function sample(gpu, run, minMs = 60) {
  run(); await gpu.done();
  let reps = 1;
  for (;;) {
    const t0 = performance.now(); for (let i = 0; i < reps; i++) run(); await gpu.done();
    const ms = performance.now() - t0;
    if (ms >= minMs || reps >= 32) return ms / reps;
    reps = Math.min(32, Math.ceil(reps * Math.max(2, (minMs * 1.2) / Math.max(ms, 1))));
  }
}
const median = (a) => { const s = [...a].sort((x, y) => x - y), n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };

/** only: sweep just the kernels whose key starts with it ('conv3', 'attnF.256', ...).
 *  -> [{key, name, settings: [{label, cfg, ok, rel, ms, gflops}], best: {label, cfg, speedup}}] */
export async function runSweep(gpu, { rounds = 3, onStep = () => {}, only = '' } = {}) {
  const out = [];
  for (const fam of families(gpu).filter((x) => x.key.startsWith(only))) {
    onStep(`${fam.name}: preparing`);
    const f = Object.assign({}, fam, fam.make());
    try {
      // correctness first: every setting's output against the default's. The output is filled with NaN before each run, so a setting
      // whose shader failed to compile (a no-op dispatch) cannot pass on the previous setting's result.
      const poison = new Uint16Array(f.n).fill(0xffff);
      f.run(f.settings[0].cfg); const ref = new Uint16Array(await gpu.read(f.out, 0, f.n * 2));
      if (!ref.some((v) => v !== 0 && (v & 0x7c00) !== 0x7c00)) throw new Error('the default setting produced no output');
      const live = [];
      for (const s of f.settings) {
        onStep(`${f.name}: checking ${s.label}`);
        try {
          gpu.device.queue.writeBuffer(f.out, 0, poison); f.run(s.cfg);
          const got = new Uint16Array(await gpu.read(f.out, 0, f.n * 2)); s.rel = +rel(got, ref).toExponential(2); s.ok = s.rel <= f.tol;
        }
        catch (e) { s.ok = false; s.error = String(e.message || e); }
        if (s.ok) live.push(s);
      }
      // then speed, interleaved across the correct settings
      const times = live.map(() => []);
      for (let r = 0; r < rounds; r++) for (let i = 0; i < live.length; i++) { onStep(`${f.name}: round ${r + 1} of ${rounds}, ${live[i].label}`); times[i].push(await sample(gpu, () => f.run(live[i].cfg))); }
      live.forEach((s, i) => { s.ms = +median(times[i]).toFixed(2); s.gflops = Math.round(f.flops / s.ms / 1e6); });
      const def = f.settings[0], best = live.reduce((a, s) => (!a || s.ms < a.ms ? s : a), null);
      out.push({ key: f.key, name: f.name, settings: f.settings.map(({ label, cfg, ok, rel: r, ms, gflops, error }) => ({ label, cfg: strip(cfg), ok, rel: r, ms, gflops, error })),
        best: best && { label: best.label, cfg: strip(best.cfg), speedup: def.ms ? +(def.ms / best.ms).toFixed(3) : null } });
    } finally { for (const b of f.bufs) gpu.free(b); }
  }
  return out;
}
const strip = (c) => { const o = Object.assign({}, c); delete o.rate; return o; };
