// SPDX-License-Identifier: MPL-2.0
// Copyright 2026 Joshua Hughes
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// Video-model kernels (SANA-Video 2.0 DiT, Gemma-2 TE, LTX-2.3 VAE) on top of the core runtime (core/gpu.js).
//   laprep    : gated-linear-attention prep: q/k RMSNorm over the FULL hidden dim (2560), weights, RoPE (interleaved pairs),
//               k *= sigmoid(beta), head-major Q/K/V write (u32-packed stores only).
//   kvstate   : per head KV = sum_n V[n] (x) K'[n]  (128 x 128, fp32 accumulators), token range split across workgroups
//   kvreduce  : sum of the splits * (1/N) -> fp16 KV (the o_norm after the read-out makes the 1/N scale exact-neutral)
//   attn256   : flash attention for head_dim 256 (SANA anchors 10x256, Gemma-2 8x256 GQA), masks none / causal+valid /
//               valid-length, optional logit soft-cap (Gemma-2: 50), fp32 scores + softmax stats
//   attnres   : SANA "Attention Residual" depth aggregation: softmax over sources of q . rms(src), weighted sum (fp32 out)
//   pixstats  : LTX PixelNorm rstd per voxel over channels (fp16 THWC in)
//   d2s       : LTX depth-to-space (c p1 p2 p3) -> (t p1, h p2, w p3), first p1-1 frames dropped; u32 (2-channel) stores
// Rule (Metal): never store a single component of a shared-memory vector; transpose in registers.
'use strict';
import { P } from '../core/gpu.js';
import { attnF, attnFFits } from './attention.js';   // p0pc0rn v028
const bw = (b, off = 0, size = null) => ({ buffer: b, offset: off, size: Math.floor((size == null ? b._pf - off : size) / 16) * 16 });

// ------------------------------------------------------------------ laprep
const LAPREP = `
enable f16;
struct P { N: u32, D: u32, H: u32, ldqkv: u32, ldg: u32, gOff: u32, eps: f32, r0: u32, hp: u32, beta: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> QKV: array<u32>;        // [N][ldqkv] f16 (q | k | v), read as u32 pairs
@group(0) @binding(2) var<storage, read> G: array<f16>;          // gate/beta GEMM output [N][ldg]; beta logits at gOff + h
@group(0) @binding(3) var<storage, read> W: array<f32>;          // q_norm weight [D] | k_norm weight [D]
@group(0) @binding(4) var<storage, read> RP: array<f32>;         // rope [N][hp][cos, sin]  (hp = head_dim / 2 pairs per head)
@group(0) @binding(5) var<storage, read_write> O: array<u32>;    // Q | K | V head-major [3][H][N][128] f16 (u32 pairs)
var<workgroup> rq: array<f32, 256>; var<workgroup> rk: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let n = wid.x + wid.y * 65535u + p.r0; if (n >= p.N) { return; }
  let pairs = p.D / 2u; let rb = n * (p.ldqkv / 2u);
  var sq = 0.0; var sk = 0.0;
  for (var i = t; i < pairs; i = i + 256u) { let a = unpack2x16float(QKV[rb + i]); let b = unpack2x16float(QKV[rb + pairs + i]); sq = sq + dot(a, a); sk = sk + dot(b, b); }
  rq[t] = sq; rk[t] = sk; workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) { if (t < s) { rq[t] = rq[t] + rq[t + s]; rk[t] = rk[t] + rk[t + s]; } workgroupBarrier(); }
  let iq = inverseSqrt(rq[0] / f32(p.D) + p.eps); let ik = inverseSqrt(rk[0] / f32(p.D) + p.eps);
  let sec = p.H * p.N * p.hp;                                     // u32 words per section
  for (var i = t; i < pairs; i = i + 256u) {
    let h = i / p.hp; let pp = i % p.hp; let c = 2u * i;
    let cs = RP[(n * p.hp + pp) * 2u]; let sn = RP[(n * p.hp + pp) * 2u + 1u];
    let q = unpack2x16float(QKV[rb + i]) * iq * vec2<f32>(W[c], W[c + 1u]);
    let k = unpack2x16float(QKV[rb + pairs + i]) * ik * vec2<f32>(W[p.D + c], W[p.D + c + 1u]);
    var beta = 1.0; if (p.beta == 1u) { beta = 1.0 / (1.0 + exp(-f32(G[n * p.ldg + p.gOff + h]))); }
    let o = (h * p.N + n) * p.hp + pp;
    O[o] = pack2x16float(vec2<f32>(q.x * cs - q.y * sn, q.x * sn + q.y * cs));
    O[sec + o] = pack2x16float(vec2<f32>(k.x * cs - k.y * sn, k.x * sn + k.y * cs) * beta);
    O[2u * sec + o] = QKV[rb + 2u * pairs + i];
  }
}`;
/** qkv:{buf} f16 [N][3D]; g:{buf, ld, off} gate/beta GEMM output (f16) with beta logits at column off+h; w: f32 [2D] (q_norm|k_norm);
 *  rope f32 [N][64][2]; out: head-major Q|K|V [3][H][N][128] f16 */
function laprep(gpu, a) {
  const pipe = gpu.pipeline('laprep', () => LAPREP);
  for (let r0 = 0; r0 < a.N; r0 += 65535 * 64) {
    const n = Math.min(65535 * 64, a.N - r0), wx = Math.min(n, 65535), wy = Math.ceil(n / 65535);
    gpu.runner.dispatch(pipe, [bw(a.qkv.buf), bw(a.g ? a.g.buf : a.qkv.buf), bw(a.w.buf, a.w.off || 0), bw(a.rope.buf), bw(a.out.buf)],
      P().u32(a.N).u32(a.D).u32(a.D / (a.hd || 128)).u32(a.qkv.ld || 3 * a.D).u32(a.g ? a.g.ld : 0).u32(a.g ? a.g.off : 0).f32(a.eps || 1e-5).u32(r0)
        .u32((a.hd || 128) / 2).u32(a.g ? 1 : 0).u32(0).u32(0).ab, [wx, wy], n * a.D * 12 / gpu.rate.mem);
  }
}

// ------------------------------------------------------------------ kvstate + kvreduce
const KVSTATE = `
enable f16;
struct P { N: u32, H: u32, per: u32, S: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> QKV: array<vec4<f16>>;    // head-major sections; K at section 1, V at section 2
@group(0) @binding(2) var<storage, read_write> PT: array<vec4<f32>>; // partial [S][H][128 dv][128 dk]
var<workgroup> ks: array<vec4<f16>, 512>; var<workgroup> vs: array<vec4<f16>, 512>;   // [16 tokens][32 vec4]
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let s = wid.x; let h = wid.y; let tx = li % 16u; let ty = li / 16u;
  let sec = p.H * p.N * 32u; let kb = sec + h * p.N * 32u; let vb = 2u * sec + h * p.N * 32u;
  let n0 = s * p.per; let n1 = min(p.N, n0 + p.per);
${[0, 1, 2, 3, 4, 5, 6, 7].map((r) => `  var a${r}0 = vec4<f32>(0.0); var a${r}1 = vec4<f32>(0.0);`).join('\n')}
  for (var c = n0; c < n1; c = c + 16u) {
    for (var i = li; i < 512u; i = i + 256u) { let tk = c + i / 32u; let ok = tk < n1; let src = min(tk, p.N - 1u) * 32u + i % 32u;
      ks[i] = select(vec4<f16>(0.0), QKV[kb + src], ok); vs[i] = select(vec4<f16>(0.0), QKV[vb + src], ok); }
    workgroupBarrier();
    for (var j = 0u; j < 16u; j = j + 1u) {
      let v0 = vec4<f32>(vs[j * 32u + ty * 2u]); let v1 = vec4<f32>(vs[j * 32u + ty * 2u + 1u]);
      let k0 = vec4<f32>(ks[j * 32u + tx * 2u]); let k1 = vec4<f32>(ks[j * 32u + tx * 2u + 1u]);
${[0, 1, 2, 3, 4, 5, 6, 7].map((r) => `      a${r}0 = fma(vec4<f32>(v${r >> 2}[${r & 3}]), k0, a${r}0); a${r}1 = fma(vec4<f32>(v${r >> 2}[${r & 3}]), k1, a${r}1);`).join('\n')}
    }
    workgroupBarrier();
  }
  let ob = ((s * p.H + h) * 128u + ty * 8u) * 32u + tx * 2u;
${[0, 1, 2, 3, 4, 5, 6, 7].map((r) => `  PT[ob + ${r * 32}u] = a${r}0; PT[ob + ${r * 32 + 1}u] = a${r}1;`).join('\n')}
}`;
const KVRED = `
enable f16;
struct P { H: u32, S: u32, scale: f32, pad: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> PT: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> KV: array<vec2<u32>>;  // f16 [H][128][128] as 4-half packs
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; let n = p.H * 4096u; if (i >= n) { return; }
  var a = vec4<f32>(0.0);
  for (var s = 0u; s < p.S; s = s + 1u) { a = a + PT[s * n + i]; }
  a = a * p.scale;
  KV[i] = vec2<u32>(pack2x16float(a.xy), pack2x16float(a.zw));
}`;
/** qkvh: head-major Q|K|V buffer (laprep output); kv: f16 [H][128 dv][128 dk] out; part: scratch >= S*H*65536 B */
function kvstate(gpu, a) {
  const S = a.splits || Math.max(1, Math.min(64, Math.ceil(a.N / 512))), per = Math.ceil(a.N / S / 16) * 16;
  const p1 = gpu.pipeline('kvstate', () => KVSTATE), p2 = gpu.pipeline('kvreduce', () => KVRED);
  gpu.runner.dispatch(p1, [bw(a.qkvh.buf), bw(a.part)], P().u32(a.N).u32(a.H).u32(per).u32(S).ab, [S, a.H], 2 * a.N * 16384 * a.H / gpu.rate.mm);
  gpu.runner.dispatch(p2, [bw(a.part), bw(a.kv.buf)], P().u32(a.H).u32(S).f32(1 / a.N).u32(0).ab, [Math.ceil(a.H * 4096 / 256), 1], a.H * 65536 * S / gpu.rate.mem);
  return 2 * a.N * 16384 * a.H;
}

// ------------------------------------------------------------------ attn256: flash attention, head_dim 256 (portable, no subgroups)
function attn256Code() {
  const L = [];
  L.push('enable f16;');
  L.push('struct P { Lq: u32, Lk: u32, ldo: u32, Hq: u32, Hkv: u32, scale: f32, mode: u32, nvalid: u32, qt0: u32, h0: u32, cap: f32, p1: u32 };');
  L.push('@group(0) @binding(0) var<uniform> p: P;');
  L.push('@group(0) @binding(1) var<storage, read> Qb: array<vec4<f16>>;');
  L.push('@group(0) @binding(2) var<storage, read> Kb: array<vec4<f16>>;');
  L.push('@group(0) @binding(3) var<storage, read> Vb: array<vec4<f16>>;');
  L.push('@group(0) @binding(4) var<storage, read_write> Ob: array<vec4<f16>>;');
  L.push('var<workgroup> qT: array<vec4<f16>, 1024>;');    // [256 d][4]  (16 queries)
  L.push('var<workgroup> kT: array<vec4<f16>, 1024>;');    // [256 d][4]  (16 keys)
  L.push('var<workgroup> vS: array<vec4<f16>, 1024>;');    // [16 keys][64]
  L.push('var<workgroup> pS: array<f32, 256>;');           // [16 keys][16 q]
  L.push('var<workgroup> mS: array<f32, 16>; var<workgroup> lS: array<f32, 16>; var<workgroup> aS: array<f32, 16>;');
  L.push('@compute @workgroup_size(64)');
  L.push('fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {');
  L.push('  let h = wid.y + p.h0; let hk = h / (p.Hq / p.Hkv); let q0 = (wid.x + p.qt0) * 16u;');
  L.push('  let qb = h * p.Lq * 64u; let kb = hk * p.Lk * 64u; let ty = li / 16u; let tx = li % 16u;');
  L.push('  for (var t = li; t < 256u; t = t + 64u) { let qq4 = t / 64u; let d4 = t % 64u; let qa = q0 + qq4 * 4u; let lq = p.Lq - 1u;');
  L.push('    let a = Qb[qb + min(qa, lq) * 64u + d4]; let b = Qb[qb + min(qa + 1u, lq) * 64u + d4]; let c = Qb[qb + min(qa + 2u, lq) * 64u + d4]; let e = Qb[qb + min(qa + 3u, lq) * 64u + d4];');
  ['x', 'y', 'z', 'w'].forEach((cc, j) => L.push(`    qT[(d4 * 4u + ${j}u) * 4u + qq4] = vec4<f16>(a.${cc}, b.${cc}, c.${cc}, e.${cc});`));
  L.push('  }');
  L.push('  var kEnd = p.Lk; if (p.mode == 1u) { kEnd = min(p.Lk, min(q0 + 15u, p.nvalid - 1u) + 1u); } if (p.mode == 2u) { kEnd = min(p.Lk, p.nvalid); }');
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) L.push(`  var o${r}${c} = vec4<f32>(0.0);`);
  L.push('  if (li < 16u) { mS[li] = -1.0e30; lS[li] = 0.0; }');
  L.push('  for (var k0 = 0u; k0 < kEnd; k0 = k0 + 16u) {');
  L.push('    workgroupBarrier();');
  L.push('    for (var t = li; t < 256u; t = t + 64u) { let kk4 = t / 64u; let d4 = t % 64u; let ka = k0 + kk4 * 4u; let lk = p.Lk - 1u;');
  L.push('      let a = Kb[kb + min(ka, lk) * 64u + d4]; let b = Kb[kb + min(ka + 1u, lk) * 64u + d4]; let c = Kb[kb + min(ka + 2u, lk) * 64u + d4]; let e = Kb[kb + min(ka + 3u, lk) * 64u + d4];');
  ['x', 'y', 'z', 'w'].forEach((cc, j) => L.push(`      kT[(d4 * 4u + ${j}u) * 4u + kk4] = vec4<f16>(a.${cc}, b.${cc}, c.${cc}, e.${cc});`));
  L.push('    }');
  L.push('    for (var t = li; t < 1024u; t = t + 64u) { vS[t] = Vb[kb + min(k0 + t / 64u, p.Lk - 1u) * 64u + t % 64u]; }');
  L.push('    workgroupBarrier();');
  L.push('    var s = vec4<f32>(0.0); let kq = tx / 4u; let kc = tx % 4u;');
  L.push('    for (var d = 0u; d < 256u; d = d + 1u) { s = fma(vec4<f32>(qT[d * 4u + ty]), vec4<f32>(f32(kT[d * 4u + kq][kc])), s); }');
  L.push('    s = s * p.scale; if (p.cap > 0.0) { s = p.cap * tanh(clamp(s / p.cap, vec4<f32>(-15.0), vec4<f32>(15.0))); }');
  L.push('    let kj = k0 + tx; let qi = q0 + ty * 4u + vec4<u32>(0u, 1u, 2u, 3u);');
  L.push('    var ok = vec4<bool>(kj < p.Lk);');
  L.push('    if (p.mode == 1u) { ok = ok & (vec4<u32>(kj) <= min(qi, vec4<u32>(p.nvalid - 1u))); } if (p.mode == 2u) { ok = ok & vec4<bool>(kj < p.nvalid); }');
  L.push('    s = select(vec4<f32>(-1.0e30), s, ok);');
  for (let r = 0; r < 4; r++) L.push(`    pS[tx * 16u + ty * 4u + ${r}u] = s[${r}];`);
  L.push('    workgroupBarrier();');
  L.push('    if (li < 16u) { let mo = mS[li]; var mx = mo; for (var j = 0u; j < 16u; j = j + 1u) { mx = max(mx, pS[j * 16u + li]); }');
  L.push('      var sm = 0.0; for (var j = 0u; j < 16u; j = j + 1u) { let v = pS[j * 16u + li]; let e = select(exp(v - mx), 0.0, v <= -1.0e29); pS[j * 16u + li] = e; sm = sm + e; }');
  L.push('      let al = exp(mo - mx); aS[li] = al; lS[li] = lS[li] * al + sm; mS[li] = mx; }');
  L.push('    workgroupBarrier();');
  for (let r = 0; r < 4; r++) { L.push(`    { let al = aS[ty * 4u + ${r}u];`); for (let c = 0; c < 4; c++) L.push(`      o${r}${c} = o${r}${c} * al;`); L.push('    }'); }
  L.push('    for (var j = 0u; j < 16u; j = j + 1u) { let pb = j * 16u + ty * 4u; let pv = vec4<f32>(pS[pb], pS[pb + 1u], pS[pb + 2u], pS[pb + 3u]); let vb = j * 64u + tx * 4u;');
  for (let c = 0; c < 4; c++) L.push(`      let v${c} = vec4<f32>(vS[vb + ${c}u]);`);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) L.push(`      o${r}${c} = fma(vec4<f32>(pv[${r}]), v${c}, o${r}${c});`);
  L.push('    }');
  L.push('  }');
  L.push('  workgroupBarrier();');
  for (let r = 0; r < 4; r++) {
    L.push(`  { let qi = q0 + ty * 4u + ${r}u; let inv = 1.0 / lS[ty * 4u + ${r}u]; if (qi < p.Lq) { let ob = (qi * p.ldo + h * 256u) / 4u + tx * 4u;`);
    for (let c = 0; c < 4; c++) L.push(`    Ob[ob + ${c}u] = vec4<f16>(o${r}${c} * inv);`);
    L.push('  } }');
  }
  L.push('}');
  return L.join('\n');
}
/** p0pc0rn v026.9: attn256 in 9.2 KB of workgroup memory (16 KB devices): Q and K one 128-dim half at a time, then V in the same tile */
function attn256SplitCode() {
  const L = [];
  L.push('enable f16;');
  L.push('struct P { Lq: u32, Lk: u32, ldo: u32, Hq: u32, Hkv: u32, scale: f32, mode: u32, nvalid: u32, qt0: u32, h0: u32, cap: f32, p1: u32 };');
  L.push('@group(0) @binding(0) var<uniform> p: P;');
  L.push('@group(0) @binding(1) var<storage, read> Qb: array<vec4<f16>>;');
  L.push('@group(0) @binding(2) var<storage, read> Kb: array<vec4<f16>>;');
  L.push('@group(0) @binding(3) var<storage, read> Vb: array<vec4<f16>>;');
  L.push('@group(0) @binding(4) var<storage, read_write> Ob: array<vec4<f16>>;');
  L.push('var<workgroup> sh: array<vec4<f16>, 1024>;');   // [0, 512): Q half [128 d][4] ; [512, 1024): K half [128 d][4] ; then V [16 keys][64]
  L.push('var<workgroup> pS: array<f32, 256>;');
  L.push('var<workgroup> mS: array<f32, 16>; var<workgroup> lS: array<f32, 16>; var<workgroup> aS: array<f32, 16>;');
  L.push('@compute @workgroup_size(64)');
  L.push('fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {');
  L.push('  let h = wid.y + p.h0; let hk = h / (p.Hq / p.Hkv); let q0 = (wid.x + p.qt0) * 16u;');
  L.push('  let qb = h * p.Lq * 64u; let kb = hk * p.Lk * 64u; let ty = li / 16u; let tx = li % 16u;');
  L.push('  var kEnd = p.Lk; if (p.mode == 1u) { kEnd = min(p.Lk, min(q0 + 15u, p.nvalid - 1u) + 1u); } if (p.mode == 2u) { kEnd = min(p.Lk, p.nvalid); }');
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) L.push(`  var o${r}${c} = vec4<f32>(0.0);`);
  L.push('  if (li < 16u) { mS[li] = -1.0e30; lS[li] = 0.0; }');
  L.push('  for (var k0 = 0u; k0 < kEnd; k0 = k0 + 16u) {');
  L.push('    var s = vec4<f32>(0.0); let kq = tx / 4u; let kc = tx % 4u;');
  L.push('    for (var hf = 0u; hf < 2u; hf = hf + 1u) {');
  L.push('      workgroupBarrier();');
  L.push('      for (var t = li; t < 128u; t = t + 64u) { let qq4 = t / 32u; let d4 = hf * 32u + t % 32u; let qa = q0 + qq4 * 4u; let lq = p.Lq - 1u;');
  L.push('        let a = Qb[qb + min(qa, lq) * 64u + d4]; let b = Qb[qb + min(qa + 1u, lq) * 64u + d4]; let c = Qb[qb + min(qa + 2u, lq) * 64u + d4]; let e = Qb[qb + min(qa + 3u, lq) * 64u + d4];');
  L.push('        let dl = d4 - hf * 32u;');
  ['x', 'y', 'z', 'w'].forEach((cc, j) => L.push(`        sh[(dl * 4u + ${j}u) * 4u + qq4] = vec4<f16>(a.${cc}, b.${cc}, c.${cc}, e.${cc});`));
  L.push('      }');
  L.push('      for (var t = li; t < 128u; t = t + 64u) { let kk4 = t / 32u; let d4 = hf * 32u + t % 32u; let ka = k0 + kk4 * 4u; let lk = p.Lk - 1u;');
  L.push('        let a = Kb[kb + min(ka, lk) * 64u + d4]; let b = Kb[kb + min(ka + 1u, lk) * 64u + d4]; let c = Kb[kb + min(ka + 2u, lk) * 64u + d4]; let e = Kb[kb + min(ka + 3u, lk) * 64u + d4];');
  L.push('        let dl = d4 - hf * 32u;');
  ['x', 'y', 'z', 'w'].forEach((cc, j) => L.push(`        sh[512u + (dl * 4u + ${j}u) * 4u + kk4] = vec4<f16>(a.${cc}, b.${cc}, c.${cc}, e.${cc});`));
  L.push('      }');
  L.push('      workgroupBarrier();');
  L.push('      for (var d = 0u; d < 128u; d = d + 1u) { s = fma(vec4<f32>(sh[d * 4u + ty]), vec4<f32>(f32(sh[512u + d * 4u + kq][kc])), s); }');
  L.push('    }');
  L.push('    s = s * p.scale; if (p.cap > 0.0) { s = p.cap * tanh(clamp(s / p.cap, vec4<f32>(-15.0), vec4<f32>(15.0))); }');
  L.push('    let kj = k0 + tx; let qi = q0 + ty * 4u + vec4<u32>(0u, 1u, 2u, 3u);');
  L.push('    var ok = vec4<bool>(kj < p.Lk);');
  L.push('    if (p.mode == 1u) { ok = ok & (vec4<u32>(kj) <= min(qi, vec4<u32>(p.nvalid - 1u))); } if (p.mode == 2u) { ok = ok & vec4<bool>(kj < p.nvalid); }');
  L.push('    s = select(vec4<f32>(-1.0e30), s, ok);');
  for (let r = 0; r < 4; r++) L.push(`    pS[tx * 16u + ty * 4u + ${r}u] = s[${r}];`);
  L.push('    workgroupBarrier();');
  L.push('    for (var t = li; t < 1024u; t = t + 64u) { sh[t] = Vb[kb + min(k0 + t / 64u, p.Lk - 1u) * 64u + t % 64u]; }');   // V into the same tile (scores are done)
  L.push('    if (li < 16u) { let mo = mS[li]; var mx = mo; for (var j = 0u; j < 16u; j = j + 1u) { mx = max(mx, pS[j * 16u + li]); }');
  L.push('      var sm = 0.0; for (var j = 0u; j < 16u; j = j + 1u) { let v = pS[j * 16u + li]; let e = select(exp(v - mx), 0.0, v <= -1.0e29); pS[j * 16u + li] = e; sm = sm + e; }');
  L.push('      let al = exp(mo - mx); aS[li] = al; lS[li] = lS[li] * al + sm; mS[li] = mx; }');
  L.push('    workgroupBarrier();');
  for (let r = 0; r < 4; r++) { L.push(`    { let al = aS[ty * 4u + ${r}u];`); for (let c = 0; c < 4; c++) L.push(`      o${r}${c} = o${r}${c} * al;`); L.push('    }'); }
  L.push('    for (var j = 0u; j < 16u; j = j + 1u) { let pb = j * 16u + ty * 4u; let pv = vec4<f32>(pS[pb], pS[pb + 1u], pS[pb + 2u], pS[pb + 3u]); let vb = j * 64u + tx * 4u;');
  for (let c = 0; c < 4; c++) L.push(`      let v${c} = vec4<f32>(sh[vb + ${c}u]);`);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) L.push(`      o${r}${c} = fma(vec4<f32>(pv[${r}]), v${c}, o${r}${c});`);
  L.push('    }');
  L.push('  }');
  L.push('  workgroupBarrier();');
  for (let r = 0; r < 4; r++) {
    L.push(`  { let qi = q0 + ty * 4u + ${r}u; let inv = 1.0 / lS[ty * 4u + ${r}u]; if (qi < p.Lq) { let ob = (qi * p.ldo + h * 256u) / 4u + tx * 4u;`);
    for (let c = 0; c < 4; c++) L.push(`    Ob[ob + ${c}u] = vec4<f16>(o${r}${c} * inv);`);
    L.push('  } }');
  }
  L.push('}');
  return L.join('\n');
}
/** q:[Hq][Lq][256], k/v:[Hkv][Lk][256] f16 head-major; o: [Lq][ldo] token-major at h*256.  mode 0 none, 1 causal+valid, 2 valid-length; cap: soft-cap */
function attn256(gpu, a) {
  const F = gpu.tune && gpu.tune.attnF256;   // p0pc0rn v028: the phone-tuned kernel (ops/attention.js attnF), every mode
  if (F && attnFFits(gpu, Object.assign({ D: 256 }, F))) { attnF(gpu, a, Object.assign({ D: 256, flavour: 'v', rate: gpu.rate.attn }, F)); return 4 * a.Lq * (a.Lk || a.Lq) * 256 * a.Hq * (a.mode === 1 ? 0.5 : 1); }
  const small = ((gpu.limits && gpu.limits.maxComputeWorkgroupStorageSize) || 16384) < 25792;   // v026.9: 25.2 KB -> the 9.2 KB variant
  const pipe = small ? gpu.pipeline('attn256s', attn256SplitCode) : gpu.pipeline('attn256', attn256Code);
  const Hq = a.Hq, Hkv = a.Hkv || Hq, Lq = a.Lq, Lk = a.Lk || Lq, ldo = a.o.ld || Hq * 256, qt = Math.ceil(Lq / 16);
  const r256 = gpu.rate.attn / 3, fWG = 4 * 16 * Lk * 256, cap = r256 * gpu.opts.maxDispatchSeconds;   // v028: this kernel runs at ~1/3 of the 128-d attention rate; full-size dispatches (0.6 s) reset the iPhone's GPU at 720p
  let qPer = qt, hPer = Math.max(1, Math.floor(cap / (fWG * qt)));
  if (fWG * qt > cap) { hPer = 1; qPer = Math.max(1, Math.floor(cap / fWG)); }
  const binds = [bw(a.q.buf, a.q.off || 0, Hq * Lq * 512), bw(a.k.buf, a.k.off || 0, Hkv * Lk * 512), bw(a.v.buf, a.v.off || 0, Hkv * Lk * 512), bw(a.o.buf, a.o.off || 0)];
  for (let h0 = 0; h0 < Hq; h0 += hPer) for (let q0 = 0; q0 < qt; q0 += qPer) {
    const nh = Math.min(hPer, Hq - h0), nq = Math.min(qPer, qt - q0);
    gpu.runner.dispatch(pipe, binds, P().u32(Lq).u32(Lk).u32(ldo).u32(Hq).u32(Hkv).f32(a.scale != null ? a.scale : 1 / 16).u32(a.mode || 0).u32(a.nvalid || Lk).u32(q0).u32(h0).f32(a.cap || 0).u32(0).ab,
      [nq, nh], fWG * nq * nh / r256);
  }
  return 4 * Lq * Lk * 256 * Hq * (a.mode === 1 ? 0.5 : 1);
}

// ------------------------------------------------------------------ attnres: depth aggregation over <= 8 sources
const ATTNRES = `
enable f16;
struct P { N: u32, D: u32, ns: u32, stride: u32, eps: f32, qOff: u32, r0: u32, pad: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> S: array<vec2<f16>>;     // sources [ns][N][D] (stride = N*D halves)
@group(0) @binding(2) var<storage, read> Q: array<f32>;          // projection weight [D] at qOff
@group(0) @binding(3) var<storage, read_write> O: array<vec2<f32>>; // hidden [N][D] fp32
var<workgroup> r1: array<f32, 256>; var<workgroup> r2: array<f32, 256>; var<workgroup> lg: array<f32, 8>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let n = wid.x + wid.y * 65535u + p.r0; if (n >= p.N) { return; }
  let D2 = p.D / 2u;
  for (var i = 0u; i < p.ns; i = i + 1u) {
    let b = (i * p.stride + n * p.D) / 2u; var ss = 0.0; var dq = 0.0;
    for (var c = t; c < D2; c = c + 256u) { let v = vec2<f32>(S[b + c]); ss = ss + dot(v, v); dq = dq + dot(v, vec2<f32>(Q[p.qOff + 2u * c], Q[p.qOff + 2u * c + 1u])); }
    r1[t] = ss; r2[t] = dq; workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) { if (t < s) { r1[t] = r1[t] + r1[t + s]; r2[t] = r2[t] + r2[t + s]; } workgroupBarrier(); }
    if (t == 0u) { lg[i] = r2[0] * inverseSqrt(r1[0] / f32(p.D) + p.eps); }
    workgroupBarrier();
  }
  var mx = -1.0e30; for (var i = 0u; i < p.ns; i = i + 1u) { mx = max(mx, lg[i]); }
  var z = 0.0; for (var i = 0u; i < p.ns; i = i + 1u) { z = z + exp(lg[i] - mx); }
  for (var c = t; c < D2; c = c + 256u) {
    var acc = vec2<f32>(0.0);
    for (var i = 0u; i < p.ns; i = i + 1u) { acc = acc + exp(lg[i] - mx) / z * vec2<f32>(S[(i * p.stride + n * p.D) / 2u + c]); }
    O[n * D2 + c] = acc;
  }
}`;
function attnres(gpu, a) {
  const pipe = gpu.pipeline('attnres', () => ATTNRES);
  for (let r0 = 0; r0 < a.N; r0 += 65535 * 64) {
    const n = Math.min(65535 * 64, a.N - r0);
    gpu.runner.dispatch(pipe, [bw(a.src.buf), bw(a.q.buf), bw(a.out.buf)], P().u32(a.N).u32(a.D).u32(a.ns).u32(a.N * a.D).f32(a.eps || 1e-6).u32(a.q.off || 0).u32(r0).u32(0).ab,
      [Math.min(n, 65535), Math.ceil(n / 65535)], n * a.D * a.ns * 4 / gpu.rate.mem);
  }
}

// ------------------------------------------------------------------ pixstats: rstd per voxel over C channels (LTX PixelNorm, eps 1e-8)
const PIXSTATS = `
enable f16;
struct P { V: u32, C: u32, eps: f32, ro: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> R: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let v = g.x + g.y * nw.x * 64u; if (v >= p.V) { return; }
  let c4 = p.C / 4u; var s = 0.0;
  for (var c = 0u; c < c4; c = c + 1u) { let x = vec4<f32>(X[v * c4 + c]); s = s + dot(x, x); }
  R[v + p.ro] = inverseSqrt(s / f32(p.C) + p.eps);
}`;
function pixstats(gpu, a) {
  const pipe = gpu.pipeline('pixstats', () => PIXSTATS), wgs = Math.ceil(a.V / 64);
  gpu.runner.dispatch(pipe, [bw(a.x.buf, a.x.off || 0), bw(a.r.buf)], P().u32(a.V).u32(a.C).f32(a.eps || 1e-8).u32(a.ro || 0).ab,
    [Math.min(wgs, 65535), Math.ceil(wgs / 65535)], a.V * a.C * 2 / gpu.rate.mem);
}

// ------------------------------------------------------------------ d2s: [T][H][W][C*p1*p2*p3] -> [T*p1-(p1-1)][H*p2][W*p3][C]
const D2S = `
enable f16;
struct P { T: u32, H: u32, W: u32, C: u32, p1: u32, p2: u32, p3: u32, To: u32, sk: u32, q0: u32, q1: u32, q2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f16>;
@group(0) @binding(2) var<storage, read_write> Y: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let i = g.x + g.y * nw.x * 256u; let C2 = p.C / 2u; let Ho = p.H * p.p2; let Wo = p.W * p.p3;
  if (i >= p.To * Ho * Wo * C2) { return; }
  let c = (i % C2) * 2u; let vox = i / C2; let ox = vox % Wo; let oy = (vox / Wo) % Ho; let ot = vox / (Wo * Ho) + p.sk;
  let it = ot / p.p1; let i1 = ot % p.p1; let iy = oy / p.p2; let i2 = oy % p.p2; let ix = ox / p.p3; let i3 = ox % p.p3;
  let PP = p.p1 * p.p2 * p.p3; let r = (i1 * p.p2 + i2) * p.p3 + i3;
  let src = ((it * p.H + iy) * p.W + ix) * p.C * PP;
  Y[i] = pack2x16float(vec2<f32>(f32(X[src + c * PP + r]), f32(X[src + (c + 1u) * PP + r])));
}`;
function d2s(gpu, a) {
  // p0pc0rn v029 time chunks: sk = leading (t p1) frames of this input to skip (p1 - 1 for the clip's first chunk, else 0), To = frames out
  const sk = a.sk != null ? a.sk : a.p1 - 1, To = a.To != null ? a.To : a.T * a.p1 - sk;
  const pipe = gpu.pipeline('d2s', () => D2S), n = To * a.H * a.p2 * a.W * a.p3 * a.C / 2, wgs = Math.ceil(n / 256);
  gpu.runner.dispatch(pipe, [bw(a.x.buf, a.x.off || 0), bw(a.y.buf, a.y.off || 0)], P().u32(a.T).u32(a.H).u32(a.W).u32(a.C).u32(a.p1).u32(a.p2).u32(a.p3).u32(To).u32(sk).u32(0).u32(0).u32(0).ab,
    [Math.min(wgs, 65535), Math.ceil(wgs / 65535)], n * 8 / gpu.rate.mem);
  return To;
}

export { laprep, kvstate, attn256, attnres, pixstats, d2s };
