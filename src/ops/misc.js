// SPDX-License-Identifier: MPL-2.0
// Copyright 2026 Joshua Hughes
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// Small kernels: row statistics (LayerNorm / RMSNorm, fp32), GroupNorm statistics (exact two-pass over the WHOLE tensor, so a
// row-banded VAE decode has no seams), elementwise ops (fp32 residual add, casts, SiLU, GELU, SwiGLU, adaLN modulate, Euler axpy),
// row softmax, f16 transpose, direct conv for tiny channel counts (VAE conv_in 3->128, conv_out 128->3).
'use strict';
import { P } from '../core/gpu.js';

const WG = 256;
const dummy = (g, i) => { g._dummies = g._dummies || []; return g._dummies[i] || (g._dummies[i] = g.buffer(256, 'dummy' + i)); };
const bw = (b, off = 0, size = null) => ({ buffer: b, offset: off, size: Math.floor((size == null ? b._pf - off : size) / 4) * 4 });

// ------------------------------------------------------------------ row stats: ST[(off+r)*2] = (mean, rstd)   (mode 0 LN, 1 RMS)
const ROWSTATS = `
struct P { cols: u32, ld: u32, c0: u32, mode: u32, eps: f32, off: u32, r0: u32, pad: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> ST: array<f32>;
var<workgroup> red: array<f32, 256>;
fn wsum(v: f32, t: u32) -> f32 {
  red[t] = v; workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) { if (t < s) { red[t] = red[t] + red[t + s]; } workgroupBarrier(); }
  let r = red[0]; workgroupBarrier(); return r;
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let row = wid.x + wid.y * 65535u + p.r0; let base = row * p.ld + p.c0;
  var mean = 0.0;
  if (p.mode == 0u) { var s = 0.0; for (var i = t; i < p.cols; i = i + 256u) { s = s + X[base + i]; } mean = wsum(s, t) / f32(p.cols); }
  var q = 0.0; for (var i = t; i < p.cols; i = i + 256u) { let d = X[base + i] - mean; q = q + d * d; }
  let var_ = wsum(q, t) / f32(p.cols);
  if (t == 0u) { ST[(p.off + row) * 2u] = mean; ST[(p.off + row) * 2u + 1u] = inverseSqrt(var_ + p.eps); }
}`;
/** x:{buf,off(bytes),ld(floats),c0}, rows, cols, mode 'ln'|'rms', eps, st:{buf, off(rows)} */
export function rowstats(gpu, a) {
  const pipe = gpu.pipeline('rowstats', () => ROWSTATS);
  for (let r0 = 0; r0 < a.rows; r0 += 65535) {
    const n = Math.min(65535, a.rows - r0);
    const pr = P().u32(a.cols).u32(a.x.ld || a.cols).u32(a.x.c0 || 0).u32(a.mode === 'rms' ? 1 : 0).f32(a.eps).u32(a.st.off || 0).u32(r0).u32(0);
    gpu.runner.dispatch(pipe, [bw(a.x.buf, a.x.off || 0), bw(a.st.buf)], pr.ab, [n, 1], (a.cols * n * 8) / gpu.rate.mem);
  }
}

// ------------------------------------------------------------------ GroupNorm statistics over an HWC f16 tensor [pix][C], 32 groups
// Exact, single pass over the tensor: each workgroup owns 64 pixels and computes per group its count, mean and M2 with an exact
// two-pass over its OWN chunk; the finaliser merges the partials with Chan's parallel formula (stable in f32).  Partials of separate
// calls merge (wgOff), so a banded / recomputed tensor gets exactly the statistics of the whole tensor (VAE band recompute at 720p).
const GN_PART = `
enable f16;
struct P { pix: u32, C: u32, wgOff: u32, pad: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f16>;
@group(0) @binding(2) var<storage, read_write> PT: array<vec4<f32>>;  // [wg][32] = (n, mean, M2, 0)
var<workgroup> acc: array<f32, 512>;
var<workgroup> mg: array<f32, 32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let wg = wid.x + wid.y * 65535u; let px0 = wg * 64u; let npx = min(64u, p.pix - min(px0, p.pix));
  let n = npx * p.C; let base = px0 * p.C; let cpg = p.C / 32u;
  var s0 = 0.0; var s1 = 0.0;
  for (var e = t; e < n; e = e + 256u) { let v = f32(X[base + e]); if (((e / 256u) & 1u) == 0u || p.C <= 256u) { s0 = s0 + v; } else { s1 = s1 + v; } }
  acc[t] = s0; acc[t + 256u] = s1; workgroupBarrier();
  let cnt = f32(npx * cpg);
  if (t < 32u) { var gs = 0.0; for (var j = 0u; j < 512u; j = j + 1u) { if ((j % p.C) / cpg == t) { gs = gs + acc[j]; } } mg[t] = gs / max(cnt, 1.0); }
  workgroupBarrier();
  s0 = 0.0; s1 = 0.0;
  for (var e = t; e < n; e = e + 256u) { let d = f32(X[base + e]) - mg[(e % p.C) / cpg]; if (((e / 256u) & 1u) == 0u || p.C <= 256u) { s0 = s0 + d * d; } else { s1 = s1 + d * d; } }
  workgroupBarrier();
  acc[t] = s0; acc[t + 256u] = s1; workgroupBarrier();
  if (t < 32u) { var m2 = 0.0; for (var j = 0u; j < 512u; j = j + 1u) { if ((j % p.C) / cpg == t) { m2 = m2 + acc[j]; } }
    PT[(p.wgOff + wg) * 32u + t] = vec4<f32>(cnt, mg[t], m2, 0.0); }
}`;
const GN_RED = `
struct P { nwg: u32, C: u32, eps: f32, off: u32, gOff: u32, bOff: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> PT: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> GB: array<f32>;         // gamma at gOff, beta at bOff (floats)
@group(0) @binding(3) var<storage, read_write> AB: array<f32>;   // out: a[C], b[C] at off
@compute @workgroup_size(32)
fn main(@builtin(local_invocation_index) t: u32) {
  var n = 0.0; var mean = 0.0; var m2 = 0.0;
  for (var w = 0u; w < p.nwg; w = w + 1u) {
    let q = PT[w * 32u + t];
    if (q.x > 0.0) { let nn = n + q.x; let d = q.y - mean; mean = mean + d * (q.x / nn); m2 = m2 + q.z + d * d * (n * q.x / nn); n = nn; }
  }
  let rs = inverseSqrt(m2 / max(n, 1.0) + p.eps);
  let cpg = p.C / 32u;
  for (var c = t * cpg; c < (t + 1u) * cpg; c = c + 1u) { let a = GB[p.gOff + c] * rs; AB[p.off + c] = a; AB[p.off + p.C + c] = GB[p.bOff + c] - mean * a; }
}`;
export const gnPartBytes = (pix) => Math.ceil(pix / 64) * 512;
/** partial statistics of x (HWC f16, pix x C; C a power of two 32..512) into part rows [wgOff, wgOff + ceil(pix/64)) */
export function gnPartial(gpu, a) {
  const nwg = Math.ceil(a.pix / 64), wx = Math.min(nwg, 65535), wy = Math.ceil(nwg / 65535);
  gpu.runner.dispatch(gpu.pipeline('gn_part2', () => GN_PART), [bw(a.x.buf, a.x.off || 0, a.x.size), bw(a.part)], P().u32(a.pix).u32(a.C).u32(a.wgOff || 0).u32(0).ab,
    [wx, wy], a.pix * a.C * 4 / gpu.rate.mem);
  return nwg;
}
/** merge nwg partial rows -> ab = {buf, off(floats)}: a[C], b[C] with GN(x) = a*x + b per channel; gb: {buf, gOff, bOff} (bytes) */
export function gnFinalize(gpu, a) {
  gpu.runner.dispatch(gpu.pipeline('gn_red2', () => GN_RED), [bw(a.part), bw(a.gb.buf), bw(a.ab.buf)],
    P().u32(a.nwg).u32(a.C).f32(a.eps || 1e-6).u32(a.ab.off || 0).u32(a.gb.gOff / 4).u32(a.gb.bOff / 4).u32(0).u32(0).ab, [1, 1], a.nwg * 64 / gpu.rate.mem + 1e-3);
}
/** Exact GroupNorm(32) statistics of x -> ab.  part: scratch >= gnPartBytes(pix).  (a.ms is accepted and unused.) */
export function groupnormStats(gpu, a) {
  const nwg = gnPartial(gpu, { x: a.x, pix: a.pix, C: a.C, part: a.part });
  gnFinalize(gpu, { part: a.part, nwg, C: a.C, gb: a.gb, ab: a.ab, eps: a.eps });
}

// ------------------------------------------------------------------ elementwise
const EW = `
enable f16;
struct P { n: u32, cols: u32, op: u32, aOff: u32, bOff: u32, vOff: u32, v2Off: u32, alpha: f32, ldo: u32, cOff: u32, half: u32, gate: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> A: array<f32>;
@group(0) @binding(2) var<storage, read_write> B: array<f16>;
@group(0) @binding(3) var<storage, read> V: array<f32>;
fn silu(x: f32) -> f32 { return x / (1.0 + exp(-x)); }
fn gelu(x: f32) -> f32 { let u = 0.7978845608 * (x + 0.044715 * x * x * x); return 0.5 * x * (1.0 + tanh(clamp(u, -15.0, 15.0))); }
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  for (var i = gid.x; i < p.n; i = i + nw.x * 256u) {
    let col = i % p.cols;
    switch p.op {
      case 0u: { var g = 1.0; if (p.gate == 1u) { g = V[p.vOff + col]; } A[p.aOff + i] = A[p.aOff + i] + g * f32(B[p.bOff + i]); }      // fp32 residual += gate * x16
      case 1u: { A[p.aOff + i] = f32(B[p.bOff + i]); }                                                                             // cast 16 -> 32
      case 2u: { B[p.bOff + i] = f16(A[p.aOff + i]); }                                                                             // cast 32 -> 16
      case 3u: { B[p.bOff + i] = f16(silu(f32(B[p.bOff + i]))); }                                                                  // SiLU in place
      case 4u: { B[p.bOff + i] = f16(gelu(f32(B[p.bOff + i]))); }                                                                  // GELU (tanh) in place
      case 5u: { B[p.bOff + i] = f16(A[p.aOff + i] * (1.0 + V[p.vOff + col]) + V[p.v2Off + col]); }                                 // adaLN modulate
      case 6u: { A[p.aOff + i] = A[p.aOff + i] + p.alpha * V[p.vOff + i]; }                                                        // axpy (Euler step)
      case 7u: { let r = i / p.cols; B[p.cOff + r * p.ldo + col] = f16(silu(f32(B[p.bOff + r * 2u * p.half + col])) * f32(B[p.bOff + r * 2u * p.half + p.half + col])); } // SwiGLU rows [gate|up] -> out (cols = half)
      default: {}
    }
  }
}`;
const OPS = { resadd: 0, cast16to32: 1, cast32to16: 2, silu: 3, gelu: 4, modulate: 5, axpy: 6, swiglu: 7 };
/** a = {op, n, cols, A:{buf,off(floats)}, B:{buf,off(halves)}, V:{buf}, vOff, v2Off, alpha, gate, swiglu: {half, ldo, cOff}} */
export function ew(gpu, a) {
  const pipe = gpu.pipeline('ew', () => EW);
  const A = a.A ? a.A.buf : dummy(gpu, 0), B = a.B ? a.B.buf : dummy(gpu, 1), V = a.V ? a.V.buf : dummy(gpu, 2);
  const s = a.swiglu || {};
  const pr = P().u32(a.n).u32(a.cols || a.n).u32(OPS[a.op]).u32(a.A ? a.A.off || 0 : 0).u32(a.B ? a.B.off || 0 : 0).u32(a.vOff || 0).u32(a.v2Off || 0)
    .f32(a.alpha || 0).u32(s.ldo || 0).u32(s.cOff || 0).u32(s.half || 0).u32(a.gate ? 1 : 0);
  gpu.runner.dispatch(pipe, [bw(A), bw(B), bw(V)], pr.ab, [Math.min(65535, Math.ceil(a.n / 256)), 1], a.n * 8 / gpu.rate.mem);
}

// ------------------------------------------------------------------ row softmax: S f32 [rows][ld] (n valid columns, scaled) -> P f16 [rows][ld]
const SOFTMAX = `
enable f16;
struct P { n: u32, scale: f32, r0: u32, ld: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> S: array<f32>;
@group(0) @binding(2) var<storage, read_write> O: array<f16>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let base = (wid.x + p.r0) * p.ld;
  var m = -3.0e38; for (var i = t; i < p.n; i = i + 256u) { m = max(m, S[base + i] * p.scale); }
  red[t] = m; workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) { if (t < s) { red[t] = max(red[t], red[t + s]); } workgroupBarrier(); }
  m = red[0]; workgroupBarrier();
  var z = 0.0; for (var i = t; i < p.n; i = i + 256u) { z = z + exp(S[base + i] * p.scale - m); }
  red[t] = z; workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) { if (t < s) { red[t] = red[t] + red[t + s]; } workgroupBarrier(); }
  let inv = 1.0 / red[0];
  for (var i = t; i < p.n; i = i + 256u) { O[base + i] = f16(exp(S[base + i] * p.scale - m) * inv); }
}`;
export function softmax(gpu, a) {
  const pipe = gpu.pipeline('softmax2', () => SOFTMAX), ld = a.ld || a.n;
  for (let r0 = 0; r0 < a.rows; r0 += 65535) {
    const n = Math.min(65535, a.rows - r0);
    gpu.runner.dispatch(pipe, [bw(a.s.buf, a.s.off || 0), bw(a.o.buf, a.o.off || 0)], P().u32(a.n).f32(a.scale || 1).u32(r0).u32(ld).ab, [n, 1], n * a.n * 12 / gpu.rate.mem);
  }
}

// ------------------------------------------------------------------ transpose f16 [R][C] -> [C][R]
const TRANSPOSE = `
enable f16;
struct P { R: u32, C: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f16>;
@group(0) @binding(2) var<storage, read_write> Y: array<f16>;
var<workgroup> tl: array<f16, 272>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let r = wid.y * 16u + l.y; let c = wid.x * 16u + l.x;
  if (r < p.R && c < p.C) { tl[l.y * 17u + l.x] = X[r * p.C + c]; }
  workgroupBarrier();
  let r2 = wid.y * 16u + l.x; let c2 = wid.x * 16u + l.y;
  if (r2 < p.R && c2 < p.C) { Y[c2 * p.R + r2] = tl[l.x * 17u + l.y]; }
}`;
export function transpose16(gpu, a) {
  const pipe = gpu.pipeline('transpose16', () => TRANSPOSE);
  gpu.runner.dispatch(pipe, [bw(a.x.buf, a.x.off || 0), bw(a.y.buf, a.y.off || 0)], P().u32(a.R).u32(a.C).u32(0).u32(0).ab,
    [Math.ceil(a.C / 16), Math.ceil(a.R / 16)], a.R * a.C * 4 / gpu.rate.mem);
}

// ------------------------------------------------------------------ direct conv (tiny Cin or Cout): HWC f16 in, OHWI f16 weights, f32 bias
const CONVD = `
enable f16;
struct P { Hi: u32, Wi: u32, Ci: u32, Ho: u32, Wo: u32, Co: u32, ks: u32, stride: u32, padT: u32, padL: u32, gn: u32, out32: u32, wOff: u32, bOff: u32, abOff: u32, i0: u32, n: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f16>;
@group(0) @binding(2) var<storage, read> W: array<f16>;
@group(0) @binding(3) var<storage, read> F: array<f32>;           // bias at bOff (floats)
@group(0) @binding(4) var<storage, read> G: array<f32>;           // GN a/b at abOff (floats)
@group(0) @binding(5) var<storage, read_write> Y: array<f16>;
@group(0) @binding(6) var<storage, read_write> Y32: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = p.i0 + gid.x + gid.y * 65535u * 64u; if (idx >= p.n) { return; }
  let co = idx % p.Co; let px = idx / p.Co; let oy = px / p.Wo; let ox = px % p.Wo;
  var s = F[p.bOff + co];
  for (var ky = 0u; ky < p.ks; ky = ky + 1u) { for (var kx = 0u; kx < p.ks; kx = kx + 1u) {
    let iy = i32(oy * p.stride + ky) - i32(p.padT); let ix = i32(ox * p.stride + kx) - i32(p.padL);
    if (iy < 0 || ix < 0 || iy >= i32(p.Hi) || ix >= i32(p.Wi)) { continue; }
    let xb = (u32(iy) * p.Wi + u32(ix)) * p.Ci; let wb = p.wOff + ((co * p.ks + ky) * p.ks + kx) * p.Ci;
    for (var c = 0u; c < p.Ci; c = c + 1u) {
      var v = f32(X[xb + c]);
      if (p.gn == 1u) { let z = v * G[p.abOff + c] + G[p.abOff + p.Ci + c]; v = z / (1.0 + exp(-z)); }
      s = s + v * f32(W[wb + c]);
    } } }
  if (p.out32 == 1u) { Y32[idx] = s; } else { Y[idx] = f16(s); }
}`;
/** conv for Cin=3 or Cout=3.  w:{buf,off(bytes)} OHWI f16; f:{buf} f32 holding the bias (bOff floats); g:{buf} GN a,b (abOff floats).
 *  Split into dispatches under the time cap (this kernel runs at roughly 10 % of the GEMM rate). */
export function convDirect(gpu, a) {
  const pipe = gpu.pipeline('convd2', () => CONVD);
  const n = a.Ho * a.Wo * a.Co, per = 2 * a.ks * a.ks * a.Ci, rate = (gpu.rate.mm32 || gpu.rate.mm) * 0.1;
  const chunk = Math.max(64 * 64, Math.floor(rate * gpu.opts.maxDispatchSeconds / per / 64) * 64);
  const yb = a.out32 ? dummy(gpu, 1) : a.y.buf, y32 = a.out32 ? a.y.buf : dummy(gpu, 2);
  const binds = [bw(a.x.buf, a.x.off || 0), bw(a.w.buf), bw(a.f.buf), bw(a.g ? a.g.buf : dummy(gpu, 0)), bw(yb, a.out32 ? 0 : (a.y.off || 0)), bw(y32, a.out32 ? (a.y.off || 0) : 0)];
  for (let i0 = 0; i0 < n; i0 += chunk) {
    const m = Math.min(chunk, n - i0), wgs = Math.ceil(m / 64);
    const pr = P().u32(a.Hi).u32(a.Wi).u32(a.Ci).u32(a.Ho).u32(a.Wo).u32(a.Co).u32(a.ks).u32(a.stride || 1).u32(a.padT || 0).u32(a.padL || 0)
      .u32(a.gn ? 1 : 0).u32(a.out32 ? 1 : 0).u32((a.w.off || 0) / 2).u32(a.bOff || 0).u32(a.abOff || 0).u32(i0).u32(i0 + m);
    gpu.runner.dispatch(pipe, binds, pr.ab, [Math.min(wgs, 65535), Math.ceil(wgs / 65535)], m * per / rate);
  }
}

// ------------------------------------------------------------------ small layout kernels (text encoder taps, VAE in/out, RGBA8)
const LAYOUT = `
enable f16;
struct P { op: u32, n: u32, a: u32, b: u32, c: u32, d: u32, e: u32, f: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> XF: array<f32>;
@group(0) @binding(2) var<storage, read> XH: array<f16>;
@group(0) @binding(3) var<storage, read> V: array<f32>;
@group(0) @binding(4) var<storage, read_write> YF: array<f32>;
@group(0) @binding(5) var<storage, read_write> YH: array<f16>;
@group(0) @binding(6) var<storage, read_write> YU: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  for (var i = gid.x; i < p.n; i = i + nw.x * 256u) {
    switch p.op {
      case 0u: {   // cast2d: YH[r*ldy + y0 + c] = f16(XF[r*ldx + x0 + c])      a=cols b=ldx c=x0 d=ldy e=y0
        let r = i / p.a; let c = i % p.a; YH[r * p.d + p.e + c] = f16(XF[r * p.b + p.c + c]); }
      case 1u: {   // chanAffine (GroupNorm apply, no activation): YH[i] = f16(XH[i] * V[f + ch] + V[f + C + ch])   a=C f=abOff
        let ch = i % p.a; YH[i] = f16(f32(XH[i]) * V[p.f + ch] + V[p.f + p.a + ch]); }
      case 2u: {   // vae_in: tokens f32 [h*w][128] -> BN^-1 -> unpatchify -> HWC f16 [(2h)(2w)][32]     a=w c=meanOff d=varOff
        let ch32 = i % 32u; let px = i / 32u; let W2 = 2u * p.a; let Y = px / W2; let X = px % W2;
        let ch = ch32 * 4u + (Y & 1u) * 2u + (X & 1u); let t = (Y >> 1u) * p.a + (X >> 1u);
        YH[i] = f16(XF[t * 128u + ch] * sqrt(V[p.d + ch] + 1.0e-4) + V[p.c + ch]); }
      case 3u: {   // vae_out: HWC f16 [(2h)(2w)][ldc] (channels 0..31) -> patchify -> BN -> tokens f32 [h*w][128]   a=w b=ldc c=meanOff d=varOff
        let ch = i % 128u; let t = i / 128u; let ii = t / p.a; let jj = t % p.a; let c = ch / 4u; let pi = (ch / 2u) & 1u; let pj = ch & 1u;
        let src = ((2u * ii + pi) * (2u * p.a) + 2u * jj + pj) * p.b + c;
        YF[i] = (f32(XH[src]) - V[p.c + ch]) / sqrt(V[p.d + ch] + 1.0e-4); }
      case 4u: {   // rgba8: f32 [pix][3] -> u32 RGBA, the reference implementation's image-save rule floor(255 * clamp((x + 1) / 2, 0, 1))
        let r = u32(floor(255.0 * clamp((XF[i * 3u] + 1.0) / 2.0, 0.0, 1.0)));
        let g = u32(floor(255.0 * clamp((XF[i * 3u + 1u] + 1.0) / 2.0, 0.0, 1.0)));
        let b = u32(floor(255.0 * clamp((XF[i * 3u + 2u] + 1.0) / 2.0, 0.0, 1.0)));
        YU[i] = r | (g << 8u) | (b << 16u) | (255u << 24u); }
      case 5u: {   // f32 -> f16 copy (img_in input tokens)
        YH[i] = f16(XF[i]); }
      default: {}
    }
  }
}`;
const LOPS = { cast2d: 0, chanAffine: 1, vaeIn: 2, vaeOut: 3, rgba8: 4, cast16: 5 };
/** a = {op, n, xf, xh, v, yf, yh, yu: {buf, off(bytes), size?}, a..f: u32 params} */
export function layout(gpu, a) {
  const pipe = gpu.pipeline('layout', () => LAYOUT);
  const b = (t, i) => (t ? bw(t.buf, t.off || 0, t.size) : bw(dummy(gpu, i)));
  gpu.runner.dispatch(pipe, [b(a.xf, 0), b(a.xh, 1), b(a.v, 2), b(a.yf, 3), b(a.yh, 4), b(a.yu, 5)],
    P().u32(LOPS[a.op]).u32(a.n).u32(a.a || 0).u32(a.b || 0).u32(a.c || 0).u32(a.d || 0).u32(a.e || 0).u32(a.f || 0).ab,
    [Math.min(Math.ceil(a.n / 256), 8192), 1], a.n * 8 / gpu.rate.mem);
}
