// SPDX-License-Identifier: MPL-2.0
// Copyright 2026 Joshua Hughes
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// Small kernels for the video stage (SANA-Video 2.0 DiT, Gemma-2 text encoder, LTX-2.3 decoder output), on top of ops/video.js.
//   rmshead   : per-row RMSNorm over D (optional weight), optional rotate-half RoPE (Gemma), written head-major [H][Ltot][hd] or token-major
//   normadd   : Gemma post-norm residual: x_f32 += t * rsqrt(mean(t^2) + eps) * w   (w stored as 1 + w)
//   vops      : elementwise: add32 (y += a, or y = a), add16 (y16 = a + b), mulsig (o16 *= sigmoid(g16)), addvec (f32 = f32 + f16 broadcast),
//               dmd (x = c1 (x - s v) + c2 noise), ltxrgba (LTX decoder conv_out 4x4 unpatchify -> RGBA8 frames, tile core crop)
// Stores are u32-packed f16 pairs (no single-component writes of shared vectors anywhere).
'use strict';
import { P } from '../core/gpu.js';

const bw = (b, off = 0, size = null) => ({ buffer: b, offset: off, size: Math.floor((size == null ? b._pf - off : size) / 16) * 16 });
const dummy = (g, i) => { g._vdummies = g._vdummies || []; return g._vdummies[i] || (g._vdummies[i] = g.buffer(256, 'vdummy' + i)); };

// ------------------------------------------------------------------ rmshead
const RMSHEAD = `
enable f16;
struct P { M: u32, D: u32, hd: u32, ldx: u32, x0: u32, norm: u32, eps: f32, wOff: u32, rope: u32, out: u32, Ltot: u32, rowOff: u32, ldy: u32, y0: u32, r0: u32, pad: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<u32>;      // f16 pairs, row-major [M][ldx]
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read> RP: array<f32>;     // rope [row][hd/2][cos, sin] (rotate-half pairs (d, d + hd/2))
@group(0) @binding(4) var<storage, read_write> O: array<u32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let m = wid.x + wid.y * 65535u + p.r0; if (m >= p.M) { return; }
  let xb = (m * p.ldx + p.x0) / 2u; let D2 = p.D / 2u;
  var rs = 1.0;
  if (p.norm == 1u) {
    var s = 0.0; for (var i = t; i < D2; i = i + 256u) { let v = unpack2x16float(X[xb + i]); s = s + dot(v, v); }
    red[t] = s; workgroupBarrier();
    for (var k = 128u; k > 0u; k = k >> 1u) { if (t < k) { red[t] = red[t] + red[t + k]; } workgroupBarrier(); }
    rs = inverseSqrt(red[0] / f32(p.D) + p.eps);
  }
  let half = p.hd / 2u;
  if (p.rope == 1u) {   // thread owns pairs (d, d+1) and their rotate-half partners (d + half, d + half + 1)
    let nq = p.D / 4u;
    for (var i = t; i < nq; i = i + 256u) {
      let h = i / (half / 2u); let d = (i % (half / 2u)) * 2u; let c = h * p.hd + d;
      var a = unpack2x16float(X[xb + c / 2u]); var b = unpack2x16float(X[xb + (c + half) / 2u]);
      if (p.norm == 1u) { a = a * rs * vec2<f32>(W[p.wOff + c], W[p.wOff + c + 1u]); b = b * rs * vec2<f32>(W[p.wOff + c + half], W[p.wOff + c + half + 1u]); }
      let rb = ((p.rowOff + m) * half + d) * 2u;
      let cs = vec2<f32>(RP[rb], RP[rb + 2u]); let sn = vec2<f32>(RP[rb + 1u], RP[rb + 3u]);
      let lo = a * cs - b * sn; let hi = b * cs + a * sn;
      let ob = ((h * p.Ltot + p.rowOff + m) * p.hd + d) / 2u;
      O[ob] = pack2x16float(lo); O[ob + half / 2u] = pack2x16float(hi);
    }
  } else {
    for (var i = t; i < D2; i = i + 256u) {
      let c = 2u * i; var v = unpack2x16float(X[xb + i]);
      if (p.norm == 1u) { v = v * rs * vec2<f32>(W[p.wOff + c], W[p.wOff + c + 1u]); }
      if (p.out == 1u) { O[(m * p.ldy + p.y0) / 2u + i] = pack2x16float(v); }
      else { let h = c / p.hd; let d = c % p.hd; O[((h * p.Ltot + p.rowOff + m) * p.hd + d) / 2u] = pack2x16float(v); }
    }
  }
}`;
/** x:{buf, off?, ld, c0 (column)} f16; D columns; hd head dim; w:{buf, off(floats)}|null; rope:{buf}|null (rotate-half, rows rowOff+m);
 *  out:{buf, off?} head-major [H][Ltot][hd] at rows rowOff.. (or tok: {ld, c0} token-major) */
export function rmshead(gpu, a) {
  const pipe = gpu.pipeline('rmshead', () => RMSHEAD), tok = !!a.tok;
  for (let r0 = 0; r0 < a.M; r0 += 65535 * 32) {
    const n = Math.min(65535 * 32, a.M - r0);
    gpu.runner.dispatch(pipe, [bw(a.x.buf, a.x.off || 0), bw(a.w ? a.w.buf : dummy(gpu, 0)), bw(a.rope ? a.rope.buf : dummy(gpu, 1)), bw(a.out.buf, a.out.off || 0)],
      P().u32(a.M).u32(a.D).u32(a.hd || 128).u32(a.x.ld).u32(a.x.c0 || 0).u32(a.w ? 1 : 0).f32(a.eps || 1e-6).u32(a.w ? a.w.off || 0 : 0).u32(a.rope ? 1 : 0)
        .u32(tok ? 1 : 0).u32(a.Ltot || a.M).u32(a.rowOff || 0).u32(tok ? a.tok.ld : 0).u32(tok ? a.tok.c0 || 0 : 0).u32(r0).u32(0).ab,
      [Math.min(n, 65535), Math.ceil(n / 65535)], n * a.D * 6 / gpu.rate.mem);
  }
}

// ------------------------------------------------------------------ normadd (Gemma-2 post-attention / post-feedforward norms)
const NORMADD = `
struct P { M: u32, D: u32, eps: f32, wOff: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> T: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> X: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let m = wid.x; let b = m * p.D;
  var s = 0.0; for (var i = t; i < p.D; i = i + 256u) { let v = T[b + i]; s = s + v * v; }
  red[t] = s; workgroupBarrier();
  for (var k = 128u; k > 0u; k = k >> 1u) { if (t < k) { red[t] = red[t] + red[t + k]; } workgroupBarrier(); }
  let rs = inverseSqrt(red[0] / f32(p.D) + p.eps);
  for (var i = t; i < p.D; i = i + 256u) { X[b + i] = X[b + i] + T[b + i] * rs * W[p.wOff + i]; }
}`;
export function normadd(gpu, a) {
  gpu.runner.dispatch(gpu.pipeline('normadd', () => NORMADD), [bw(a.t.buf), bw(a.w.buf), bw(a.x.buf, a.x.off || 0)],
    P().u32(a.M).u32(a.D).f32(a.eps || 1e-6).u32(a.w.off || 0).ab, [a.M, 1], a.M * a.D * 12 / gpu.rate.mem);
}

// ------------------------------------------------------------------ vops
const VOPS = `
enable f16;
struct P { op: u32, n: u32, a: u32, b: u32, c: u32, d: u32, e: u32, f: u32, s0: f32, s1: f32, s2: f32, g: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> AF: array<f32>;
@group(0) @binding(2) var<storage, read> BF: array<f32>;
@group(0) @binding(3) var<storage, read> AH: array<f16>;
@group(0) @binding(4) var<storage, read_write> YF: array<f32>;
@group(0) @binding(5) var<storage, read_write> YU: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  for (var i = gid.x; i < p.n; i = i + nw.x * 256u) {
    switch p.op {
      case 0u: { if (p.a == 1u) { YF[i] = AF[i]; } else { YF[i] = YF[i] + AF[i]; } }                 // add32 (a=1: store)
      case 1u: { let j = 2u * i; YU[i] = pack2x16float(vec2<f32>(AF[j] + BF[j], AF[j + 1u] + BF[j + 1u])); }   // add16: y16 = a + b (n = pairs)
      case 2u: { let j = 2u * i; let col = j % p.a; let row = j / p.a;                                 // mulsig: o16 *= sigmoid(g16[row*ldg + col])
        let o = unpack2x16float(YU[i]); let g = vec2<f32>(f32(AH[row * p.b + col]), f32(AH[row * p.b + col + 1u]));
        YU[i] = pack2x16float(o / (vec2<f32>(1.0) + exp(-g))); }
      case 3u: { YF[i] = AF[p.a + i] + f32(AH[p.b + i % p.c]); }                                         // addvec: y = a[aOff + i] + h[hOff + i % mod]
      case 4u: { YF[i] = p.s1 * (YF[i] - p.s0 * AF[i]) + p.s2 * BF[i]; }                               // dmd: x = (1-s') (x - s v) + s' noise
      case 5u: {   // ltxrgba: conv_out [F][h4][w4][48] f32 -> RGBA8 frames [F][H][W] (tile: core rows/cols in tile coords, written at the frame origin)
        let tw = p.c - p.b; let px = i % tw; let rest = i / tw; let th = p.e - p.d; let py = rest % th; let f = rest / th;   // b..c = core x range, d..e = core y range (tile pixels)
        let x = p.b + px; let y = p.d + py; let w4 = p.f >> 16u; let h4 = p.f & 65535u;
        let vox = (f * h4 + y / 4u) * w4 + x / 4u; let q = (x % 4u) * 4u + (y % 4u);
        var u = vec3<u32>(0u);
        for (var cc = 0u; cc < 3u; cc = cc + 1u) { u[cc] = u32(clamp(127.5 * AF[vox * 48u + cc * 16u + q] + 127.5, 0.0, 255.0)); }
        let W = p.g >> 16u; let H = p.g & 65535u; let fx = p.a >> 16u; let fy = p.a & 65535u;
        YU[(f * H + fy + py) * W + fx + px] = u.x | (u.y << 8u) | (u.z << 16u) | (255u << 24u); }
      case 6u: {   // ltxin: latent tokens f32 [T][h][w][128] tile (x0, y0, tw, th) -> z * std + mean -> f16 THWC pairs   a=w b=h c=x0|y0<<16 d=meanOff e=stdOff f=tw|th<<16
        let tw = p.f & 65535u; let th = p.f >> 16u; let c = (i % 64u) * 2u; let vox = i / 64u; let j = vox % tw; let ii = (vox / tw) % th; let t = vox / (tw * th);
        let src = ((t * p.b + (p.c >> 16u) + ii) * p.a + (p.c & 65535u) + j) * 128u + c;
        YU[i] = pack2x16float(vec2<f32>(AF[src] * BF[p.e + c] + BF[p.d + c], AF[src + 1u] * BF[p.e + c + 1u] + BF[p.d + c + 1u])); }
      case 7u: {   // crop: f16 THWC [T][H][W][C] region (x0, y0, w, h) -> [T][h][w][C]   a=C/2 b=W c=H d=x0|y0<<16 e=w|h<<16
        let w = p.e & 65535u; let h = p.e >> 16u; let c2 = i % p.a; let vox = i / p.a; let j = vox % w; let ii = (vox / w) % h; let t = vox / (w * h);
        let src = 2u * (((t * p.c + (p.d >> 16u) + ii) * p.b + (p.d & 65535u) + j) * p.a + c2);
        YU[i] = pack2x16float(vec2<f32>(f32(AH[src]), f32(AH[src + 1u]))); }
      case 8u: {   // ltxrgba16: conv_out f16 [F][h4][w4][48] -> RGBA8 [nf][ch][cw] for frames a.., core x b..c / y d..e (tile pixels); f = w4<<16 | h4
        let cw = p.c - p.b; let chh = p.e - p.d; let px = i % cw; let rest = i / cw; let py = rest % chh; let fr = p.a + rest / chh;
        let x = p.b + px; let y = p.d + py; let w4 = p.f >> 16u; let h4 = p.f & 65535u;
        let src = ((fr * h4 + y / 4u) * w4 + x / 4u) * 48u + (x % 4u) * 4u + (y % 4u);
        let r = u32(clamp(127.5 * f32(AH[src]) + 127.5, 0.0, 255.0)); let g = u32(clamp(127.5 * f32(AH[src + 16u]) + 127.5, 0.0, 255.0));
        let b = u32(clamp(127.5 * f32(AH[src + 32u]) + 127.5, 0.0, 255.0));
        YU[i] = r | (g << 8u) | (b << 16u) | (255u << 24u); }
      default: {}
    }
  }
}`;
const VOP = { add32: 0, add16: 1, mulsig: 2, addvec: 3, dmd: 4, ltxrgba: 5, ltxin: 6, crop: 7, ltxrgba16: 8 };
/** a = {op, n, af, bf, ah, yf, yu: {buf, off}, a..g u32, s0..s2 f32} */
export function vop(gpu, a) {
  const pipe = gpu.pipeline('vops', () => VOPS), b = (t, i) => (t ? bw(t.buf, t.off || 0, t.size) : bw(dummy(gpu, i)));
  gpu.runner.dispatch(pipe, [b(a.af, 2), b(a.bf, 3), b(a.ah, 4), b(a.yf, 5), b(a.yu, 6)],
    P().u32(VOP[a.op]).u32(a.n).u32(a.a || 0).u32(a.b || 0).u32(a.c || 0).u32(a.d || 0).u32(a.e || 0).u32(a.f || 0).f32(a.s0 || 0).f32(a.s1 || 0).f32(a.s2 || 0).u32(a.g || 0).ab,
    [Math.min(Math.ceil(a.n / 256), 16384), 1], a.n * 12 / gpu.rate.mem);
}

// ------------------------------------------------------------------ RoPE tables (JS, float64 angles -> f32)
/** Wan 3-D RoPE for SANA: tokens (f, i, j) row-major; pairs split t/h/w = pt/ph/pw (head 128: 22/21/21, head 256: 44/42/42); [N][pairs][cos, sin] */
export function ropeWan(T, H, W, pairs) {
  const [pt, ph, pw] = pairs, np = pt + ph + pw, out = new Float32Array(T * H * W * np * 2);
  const fr = (n) => Array.from({ length: n }, (_, k) => Math.pow(10000, -2 * k / (2 * n)));
  const ft = fr(pt), fh = fr(ph), fw = fr(pw);
  let o = 0;
  for (let f = 0; f < T; f++) for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
    for (let k = 0; k < pt; k++) { const a = f * ft[k]; out[o++] = Math.cos(a); out[o++] = Math.sin(a); }
    for (let k = 0; k < ph; k++) { const a = i * fh[k]; out[o++] = Math.cos(a); out[o++] = Math.sin(a); }
    for (let k = 0; k < pw; k++) { const a = j * fw[k]; out[o++] = Math.cos(a); out[o++] = Math.sin(a); }
  }
  return out;
}
/** Gemma-2 RoPE (rotate-half, theta 1e4, head 256): [L][128][cos, sin]; inv_freq in float32 like HF */
export function ropeGemma(L) {
  const out = new Float32Array(L * 256), f = Math.fround;
  const inv = Array.from({ length: 128 }, (_, d) => f(1 / f(Math.pow(10000, f((2 * d) / 256)))));
  for (let t = 0; t < L; t++) for (let d = 0; d < 128; d++) { const a = f(t * inv[d]); out[(t * 128 + d) * 2] = Math.cos(a); out[(t * 128 + d) * 2 + 1] = Math.sin(a); }
  return out;
}

// ------------------------------------------------------------------ rmsrow32: f32 rows -> RMSNorm * w -> f16 rows (token-major, row offset)
const RMSROW32 = `
struct P { M: u32, D: u32, eps: f32, wOff: u32, ldy: u32, y0: u32, rowOut: u32, pad: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<u32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let m = wid.x; let D2 = p.D / 2u; let b = m * D2;
  var s = 0.0; for (var i = t; i < D2; i = i + 256u) { let v = X[b + i]; s = s + dot(v, v); }
  red[t] = s; workgroupBarrier();
  for (var k = 128u; k > 0u; k = k >> 1u) { if (t < k) { red[t] = red[t] + red[t + k]; } workgroupBarrier(); }
  let rs = inverseSqrt(red[0] / f32(p.D) + p.eps);
  for (var i = t; i < D2; i = i + 256u) { O[((p.rowOut + m) * p.ldy + p.y0) / 2u + i] = pack2x16float(X[b + i] * rs * vec2<f32>(W[p.wOff + 2u * i], W[p.wOff + 2u * i + 1u])); }
}`;
/** x:{buf, off} f32 [M][D] -> out f16 [rowOut + m][ld] at column c0; w {buf, off(floats)} */
export function rmsrow32(gpu, a) {
  gpu.runner.dispatch(gpu.pipeline('rmsrow32', () => RMSROW32), [bw(a.x.buf, a.x.off || 0), bw(a.w.buf), bw(a.out.buf)],
    P().u32(a.M).u32(a.D).f32(a.eps || 1e-6).u32(a.w.off || 0).u32(a.out.ld || a.D).u32(a.out.c0 || 0).u32(a.out.row || 0).u32(0).ab, [a.M, 1], a.M * a.D * 8 / gpu.rate.mem);
}
