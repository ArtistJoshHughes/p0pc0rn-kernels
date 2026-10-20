// SPDX-License-Identifier: MPL-2.0
// Copyright 2026 Joshua Hughes
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// Flash attention, head_dim 128, fp16 tiles in workgroup memory, fp32 scores / online-softmax statistics / accumulators.
// The L x L score matrix never leaves the workgroup.  Q [Hq][Lq][128] f16, K/V [Hkv][Lk][128] f16 (head-major, as written by
// the GEMM 'heads' epilogue), O [Lq][ldo] f16 token-major at column h*128 (the input layout of the projection GEMM).
// mask: 0 = none (FLUX DiT), 1 = causal + valid length (Qwen TE: key j visible to query i iff j <= min(i, nvalid-1)),
//       2 = valid length only (cross-attention to padded text keys).
// GQA: query head h reads kv head h / (Hq/Hkv).
// Variants: {BQ:32} (row-owner softmax, portable); {BQ:64} (scores stay in registers, row max/sum via a 2 KB shared reduce,
// P stored fp16; no subgroups needed - Safari has none); {sg:true} uses subgroupShuffleXor for the row reduce instead;
// pv16 = fp16 P*V partial sums per 32-key tile; s16 = fp16 QK partials per 32-dim chunk (scores/softmax stats stay fp32).
// RULE: never store a single component of a shared vec4 (Metal read-modify-writes the vector and races); transpose in registers.
'use strict';
import { P } from '../core/gpu.js';

export function attnCode(c) {
  const BQ = c.BQ, NT = (BQ / 4) * 8, Q4 = BQ / 4, sg = !!c.sg;
  const kvShared = BQ === 64 || !!c.kvs;   // v026.9: kvs = V reuses the K tile (the 16 KB variant)
  const reg = sg || BQ === 64;   // BQ64 keeps scores in registers (P stored f16): subgroup shuffles, or a small shared reduce
  const pT = BQ === 64 ? 'f16' : 'f32';
  const L = [];
  L.push('enable f16;'); if (sg) L.push('enable subgroups;');
  L.push('struct P { Lq: u32, Lk: u32, ldo: u32, Hq: u32, Hkv: u32, scale: f32, mode: u32, nvalid: u32, qt0: u32, h0: u32, p0: u32, p1: u32 };');
  L.push('@group(0) @binding(0) var<uniform> p: P;');
  L.push('@group(0) @binding(1) var<storage, read> Qb: array<vec4<f16>>;');
  L.push('@group(0) @binding(2) var<storage, read> Kb: array<vec4<f16>>;');
  L.push('@group(0) @binding(3) var<storage, read> Vb: array<vec4<f16>>;');
  L.push('@group(0) @binding(4) var<storage, read_write> Ob: array<vec4<f16>>;');
  L.push(`var<workgroup> qT: array<vec4<f16>, ${128 * Q4}>;`);          // [d][q/4]
  L.push(`var<workgroup> kT: array<vec4<f16>, 1024>;`);                   // [d][key/4]  (32 keys)
  if (!kvShared) L.push(`var<workgroup> vS: array<vec4<f16>, 1024>;`);    // [key][d/4]
  L.push(`var<workgroup> pS: array<${pT}, ${32 * BQ}>;`);                  // [key][q]
  if (!reg) L.push(`var<workgroup> mS: array<f32, ${BQ}>; var<workgroup> lS: array<f32, ${BQ}>; var<workgroup> aS: array<f32, ${BQ}>;`);
  if (reg && !sg) L.push(`var<workgroup> rS: array<f32, ${BQ * 8}>;`);
  const VS = kvShared ? 'kT' : 'vS';
  L.push(`@compute @workgroup_size(${NT})`);
  L.push('fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {');
  L.push(`  let h = wid.y + p.h0; let hk = h / (p.Hq / p.Hkv); let q0 = (wid.x + p.qt0) * ${BQ}u;`);
  L.push('  let qb = h * p.Lq * 32u; let kb = hk * p.Lk * 32u;');
  L.push('  let ty = li / 8u; let tx = li % 8u;');
  // Q tile, transposed through registers (4 queries x 4 dims per task) with WHOLE vec4 stores: a single-component store
  // into a shared vec4 is a read-modify-write of the vector on Metal and races with the neighbouring lanes.
  L.push(`  for (var t = li; t < ${BQ * 8}u; t = t + ${NT}u) { let qq4 = t / 32u; let d4 = t % 32u; let qa = q0 + qq4 * 4u; let lq = p.Lq - 1u;`);
  L.push('    let a = Qb[qb + min(qa, lq) * 32u + d4]; let b = Qb[qb + min(qa + 1u, lq) * 32u + d4]; let c = Qb[qb + min(qa + 2u, lq) * 32u + d4]; let e = Qb[qb + min(qa + 3u, lq) * 32u + d4];');
  L.push(`    let o = d4 * 4u * ${Q4}u + qq4;`);
  ['x', 'y', 'z', 'w'].forEach((cc, j) => L.push(`    qT[o + ${j * Q4}u] = vec4<f16>(a.${cc}, b.${cc}, c.${cc}, e.${cc});`));
  L.push('  }');
  L.push(`  var kEnd = p.Lk; if (p.mode == 1u) { kEnd = min(p.Lk, min(q0 + ${BQ - 1}u, p.nvalid - 1u) + 1u); } if (p.mode == 2u) { kEnd = min(p.Lk, p.nvalid); }`);
  for (let r = 0; r < 4; r++) for (let cI = 0; cI < 4; cI++) L.push(`  var o${r}${cI} = vec4<f32>(0.0);`);
  if (reg) for (let r = 0; r < 4; r++) L.push(`  var m${r} = -1.0e30; var l${r} = 0.0;`);
  else L.push(`  if (li < ${BQ}u) { mS[li] = -1.0e30; lS[li] = 0.0; }`);
  L.push('  for (var k0 = 0u; k0 < kEnd; k0 = k0 + 32u) {');
  L.push('    workgroupBarrier();');
  L.push(`    for (var t = li; t < 256u; t = t + ${NT}u) { let kk4 = t / 32u; let d4 = t % 32u; let ka = k0 + kk4 * 4u; let lk = p.Lk - 1u;`);
  L.push('      let a = Kb[kb + min(ka, lk) * 32u + d4]; let b = Kb[kb + min(ka + 1u, lk) * 32u + d4]; let c = Kb[kb + min(ka + 2u, lk) * 32u + d4]; let e = Kb[kb + min(ka + 3u, lk) * 32u + d4];');
  L.push('      let o = d4 * 32u + kk4;');
  ['x', 'y', 'z', 'w'].forEach((cc, j) => L.push(`      kT[o + ${j * 8}u] = vec4<f16>(a.${cc}, b.${cc}, c.${cc}, e.${cc});`));
  L.push('    }');
  if (!kvShared) L.push(`    for (var t = li; t < 1024u; t = t + ${NT}u) { vS[t] = Vb[kb + min(k0 + t / 32u, p.Lk - 1u) * 32u + t % 32u]; }`);
  L.push('    workgroupBarrier();');
  // scores for rows ty*4..+3, keys tx*4..+3
  for (let r = 0; r < 4; r++) L.push(`    var s${r} = vec4<f32>(0.0);`);
  if (c.s16) {   // fp16 partial dot products over 32-dim chunks, flushed to fp32 (scores and softmax stats stay fp32)
    L.push('    for (var dc = 0u; dc < 128u; dc = dc + 32u) {');
    for (let r = 0; r < 4; r++) L.push(`      var h${r} = vec4<f16>(0.0);`);
    L.push(`      for (var d = dc; d < dc + 32u; d = d + 1u) { let qv = qT[d * ${Q4}u + ty]; let kv = kT[d * 8u + tx];`);
    for (let r = 0; r < 4; r++) L.push(`        h${r} = fma(vec4<f16>(qv[${r}]), kv, h${r});`);
    L.push('      }');
    for (let r = 0; r < 4; r++) L.push(`      s${r} = s${r} + vec4<f32>(h${r});`);
    L.push('    }');
  } else {
  L.push(`    for (var d = 0u; d < 128u; d = d + 1u) { let qv = vec4<f32>(qT[d * ${Q4}u + ty]); let kv = vec4<f32>(kT[d * 8u + tx]);`);
  for (let r = 0; r < 4; r++) L.push(`      s${r} = fma(vec4<f32>(qv[${r}]), kv, s${r});`);
  L.push('    }');
  }
  L.push('    let kj = k0 + tx * 4u + vec4<u32>(0u, 1u, 2u, 3u);');
  for (let r = 0; r < 4; r++) {
    L.push(`    { let qi = q0 + ty * 4u + ${r}u; var ok = kj < vec4<u32>(p.Lk);`);
    L.push(`      if (p.mode == 1u) { ok = ok & (kj <= vec4<u32>(min(qi, p.nvalid - 1u))); } if (p.mode == 2u) { ok = ok & (kj < vec4<u32>(p.nvalid)); }`);
    L.push(`      s${r} = select(vec4<f32>(-1.0e30), s${r} * p.scale, ok); }`);
  }
  if (reg) {   // m, l in registers; row max / sum over the 8 lanes of a row block via subgroup shuffles, or a 2 KB shared array
    const red = (v, op) => sg
      ? [1, 2, 4].map((k) => `${v} = ${op}(${v}, subgroupShuffleXor(${v}, ${k}u));`).join(' ')
      : null;
    for (let r = 0; r < 4; r++) L.push(`    var mx${r} = max(max(s${r}.x, s${r}.y), max(s${r}.z, s${r}.w));`);
    if (sg) for (let r = 0; r < 4; r++) L.push('    ' + red(`mx${r}`, 'max'));
    else {
      for (let r = 0; r < 4; r++) L.push(`    rS[(ty * 4u + ${r}u) * 8u + tx] = mx${r};`);
      L.push('    workgroupBarrier();');
      for (let r = 0; r < 4; r++) L.push(`    for (var i = 0u; i < 8u; i = i + 1u) { mx${r} = max(mx${r}, rS[(ty * 4u + ${r}u) * 8u + i]); }`);
      L.push('    workgroupBarrier();');
    }
    for (let r = 0; r < 4; r++) {
      L.push(`    let mn${r} = max(m${r}, mx${r}); let pe${r} = select(exp2(s${r} - vec4<f32>(mn${r})), vec4<f32>(0.0), s${r} <= vec4<f32>(-1.0e29));`);
      L.push(`    var sm${r} = pe${r}.x + pe${r}.y + pe${r}.z + pe${r}.w;`);
    }
    if (sg) for (let r = 0; r < 4; r++) L.push(`    sm${r} = sm${r} + subgroupShuffleXor(sm${r}, 1u); sm${r} = sm${r} + subgroupShuffleXor(sm${r}, 2u); sm${r} = sm${r} + subgroupShuffleXor(sm${r}, 4u);`);
    else {
      for (let r = 0; r < 4; r++) L.push(`    rS[(ty * 4u + ${r}u) * 8u + tx] = sm${r};`);
      L.push('    workgroupBarrier();');
      for (let r = 0; r < 4; r++) L.push(`    sm${r} = 0.0; for (var i = 0u; i < 8u; i = i + 1u) { sm${r} = sm${r} + rS[(ty * 4u + ${r}u) * 8u + i]; }`);
    }
    for (let r = 0; r < 4; r++) {
      L.push(`    { let al = exp2(m${r} - mn${r}); l${r} = l${r} * al + sm${r}; m${r} = mn${r};`);
      for (let cI = 0; cI < 4; cI++) L.push(`      o${r}${cI} = o${r}${cI} * al;`);
      for (let j = 0; j < 4; j++) L.push(`      pS[(tx * 4u + ${j}u) * ${BQ}u + ty * 4u + ${r}u] = ${pT}(pe${r}[${j}]);`);
      L.push('    }');
    }
    if (kvShared) {
      L.push('    workgroupBarrier();');
      L.push(`    for (var t = li; t < 1024u; t = t + ${NT}u) { kT[t] = Vb[kb + min(k0 + t / 32u, p.Lk - 1u) * 32u + t % 32u]; }`);
    }
    L.push('    workgroupBarrier();');
  } else {
    for (let r = 0; r < 4; r++) for (let j = 0; j < 4; j++) L.push(`    pS[(tx * 4u + ${j}u) * ${BQ}u + ty * 4u + ${r}u] = s${r}[${j}];`);
    L.push('    workgroupBarrier();');
    L.push(`    if (li < ${BQ}u) { let mo = mS[li]; var mx = mo;`);
    L.push(`      for (var j = 0u; j < 32u; j = j + 1u) { mx = max(mx, pS[j * ${BQ}u + li]); }`);
    L.push('      var sm = 0.0;');
    L.push(`      for (var j = 0u; j < 32u; j = j + 1u) { let s = pS[j * ${BQ}u + li]; let e = select(exp2(s - mx), 0.0, s <= -1.0e29); pS[j * ${BQ}u + li] = e; sm = sm + e; }`);
    L.push('      let al = exp2(mo - mx); aS[li] = al; lS[li] = lS[li] * al + sm; mS[li] = mx; }');
    L.push('    workgroupBarrier();');
    for (let r = 0; r < 4; r++) { L.push(`    { let al = aS[ty * 4u + ${r}u];`); for (let cI = 0; cI < 4; cI++) L.push(`      o${r}${cI} = o${r}${cI} * al;`); L.push('    }'); }
    if (kvShared) {   // v026.9: every lane finished reading K (the scores) before the first barrier: V goes into the same tile
      L.push(`    for (var t = li; t < 1024u; t = t + ${NT}u) { kT[t] = Vb[kb + min(k0 + t / 32u, p.Lk - 1u) * 32u + t % 32u]; }`);
      L.push('    workgroupBarrier();');
    }
  }
  // O += P V   (rows ty*4..+3, dims tx*16..+15)
  if (c.pv16) {   // fp16 partial sums over one 32-key tile (P in [0,1]), flushed into the fp32 accumulators per tile
    for (let r = 0; r < 4; r++) for (let cI = 0; cI < 4; cI++) L.push(`    var h${r}${cI} = vec4<f16>(0.0);`);
    L.push('    for (var j = 0u; j < 32u; j = j + 1u) {');
    L.push(`      let pb = j * ${BQ}u + ty * 4u; let pv = vec4<f16>(f16(pS[pb]), f16(pS[pb + 1u]), f16(pS[pb + 2u]), f16(pS[pb + 3u]));`);
    L.push(`      let vb = j * 32u + tx * 4u;`);
    for (let cI = 0; cI < 4; cI++) L.push(`      let v${cI} = ${VS}[vb + ${cI}u];`);
    for (let r = 0; r < 4; r++) for (let cI = 0; cI < 4; cI++) L.push(`      h${r}${cI} = fma(vec4<f16>(pv[${r}]), v${cI}, h${r}${cI});`);
    L.push('    }');
    for (let r = 0; r < 4; r++) for (let cI = 0; cI < 4; cI++) L.push(`    o${r}${cI} = o${r}${cI} + vec4<f32>(h${r}${cI});`);
  } else {
  L.push('    for (var j = 0u; j < 32u; j = j + 1u) {');
  L.push(`      let pb = j * ${BQ}u + ty * 4u; let pv = vec4<f32>(f32(pS[pb]), f32(pS[pb + 1u]), f32(pS[pb + 2u]), f32(pS[pb + 3u]));`);
  L.push(`      let vb = j * 32u + tx * 4u;`);
  for (let cI = 0; cI < 4; cI++) L.push(`      let v${cI} = vec4<f32>(${VS}[vb + ${cI}u]);`);
  for (let r = 0; r < 4; r++) for (let cI = 0; cI < 4; cI++) L.push(`      o${r}${cI} = fma(vec4<f32>(pv[${r}]), v${cI}, o${r}${cI});`);
  L.push('    }');
  }
  L.push('  }');
  if (!reg) L.push('  workgroupBarrier();');
  for (let r = 0; r < 4; r++) {
    L.push(`  { let qi = q0 + ty * 4u + ${r}u; let inv = 1.0 / ${reg ? `l${r}` : `lS[ty * 4u + ${r}u]`};`);
    L.push(`    if (qi < p.Lq) { let ob = (qi * p.ldo + h * 128u) / 4u + tx * 4u;`);
    for (let cI = 0; cI < 4; cI++) L.push(`      Ob[ob + ${cI}u] = vec4<f16>(o${r}${cI} * inv);`);
    L.push('    } }');
  }
  L.push('}');
  return L.join('\n');
}

/** a = {q:{buf,off}, k:{buf,off}, v:{buf,off}, o:{buf,off,ld}, Lq, Lk, Hq, Hkv, mode=0, nvalid, cfg:{BQ, sg}} */
/** v026.9: the workgroup memory (bytes) a variant declares */
export function attnBytes(c) {
  const BQ = c.BQ, kvs = BQ === 64 || !!c.kvs, reg = !!c.sg || BQ === 64;
  return 128 * (BQ / 4) * 8 + 1024 * 8 + (kvs ? 0 : 1024 * 8) + 32 * BQ * (BQ === 64 ? 2 : 4) + (reg ? 0 : 3 * BQ * 4) + (reg && !c.sg ? BQ * 8 * 4 : 0);
}
export function attention(gpu, a) {
  const F = a.cfgF !== undefined ? a.cfgF : gpu.tune && gpu.tune.attnF;   // p0pc0rn v028: the phone-tuned kernel (attnF below)
  if (F && attnFFits(gpu, Object.assign({ D: 128 }, F))) { attnF(gpu, a, Object.assign({ D: 128, flavour: 'p', rate: gpu.rate.attn }, F)); return 4 * a.Lq * (a.Lk || a.Lq) * 128 * a.Hq; }
  let cfg = a.cfg || gpu.tune?.attn || { BQ: gpu.subgroups && gpu.sgMin >= 8 ? 64 : 32, sg: gpu.subgroups && gpu.sgMin >= 8 };
  const wgMax = (gpu.limits && gpu.limits.maxComputeWorkgroupStorageSize) || 16384;
  if (attnBytes(cfg) > wgMax) cfg = { BQ: 16, kvs: true };   // v026.9: Android-like 16 KB of workgroup memory (14.2 KB)
  const key = `attn.${cfg.BQ}.${cfg.sg ? 'sg' : ''}.${cfg.pv16 ? 'pv16' : ''}.${cfg.s16 ? 's16' : ''}${cfg.kvs ? '.kvs' : ''}`;
  const pipe = gpu.pipeline(key, () => attnCode(cfg));
  const Hq = a.Hq, Hkv = a.Hkv || a.Hq, Lq = a.Lq, Lk = a.Lk || a.Lq, ldo = a.o.ld || Hq * 128;
  const qt = Math.ceil(Lq / cfg.BQ);
  const fWG = 4 * cfg.BQ * Lk * 128, cap = gpu.cap('attn');
  let qPer = qt, hPer = Math.max(1, Math.floor(cap / (fWG * qt)));
  if (fWG * qt > cap) { hPer = 1; qPer = Math.max(1, Math.floor(cap / fWG)); }
  const scale = (a.scale != null ? a.scale : 1 / Math.sqrt(128)) * Math.LOG2E;
  const vw = (t, n) => ({ buffer: t.buf, offset: t.off || 0, size: t.size || n });
  const qs = Hq * Lq * 256, ks = Hkv * Lk * 256;
  const binds = [vw(a.q, qs), vw(a.k, ks), vw(a.v, ks), { buffer: a.o.buf, offset: a.o.off || 0, size: a.o.size || Math.ceil(Lq * ldo * 2 / 16) * 16 }];
  for (let h0 = 0; h0 < Hq; h0 += hPer) {
    for (let q0 = 0; q0 < qt; q0 += qPer) {
      const nh = Math.min(hPer, Hq - h0), nq = Math.min(qPer, qt - q0);
      const pr = P().u32(Lq).u32(Lk).u32(ldo).u32(Hq).u32(Hkv).f32(scale).u32(a.mode || 0).u32(a.nvalid || Lk).u32(q0).u32(h0).u32(0).u32(0);
      gpu.runner.dispatch(pipe, binds, pr.ab, [nq, nh], (fWG * nq * nh) / gpu.rate.attn);
    }
  }
  return 4 * Lq * Lk * 128 * Hq;
}

// ------------------------------------------------------------------ attnF: the phone-tuned flash attention (head_dim 128 and 256)

// p0pc0rn v028 (phone benchmark, Oct 2026): register-tiled flash attention, tuned on an iPhone 16 Pro Max (Safari 26.6.1).
// Q resident in workgroup memory, fp16 QK partials per 32-d chunk + fp16 PV partials flushed to fp32, exact online softmax, padded / interleaved
// tiles. Picture attention 464 -> 749 GFLOP/s (1.61x), video anchors 145 -> 564 GFLOP/s (3.89x); every mode (causal + valid, valid length,
// GQA, the Gemma-2 soft-cap) matches the old kernels to <= 0.15% (rel L2). Devices without 32 KB of workgroup memory keep the old kernels.
export const ATTNF_P = { BQ: 64, BK: 64, TK: 4, PQ: 4, PD: 8, pv16: true, s16: true, il: true };
export const ATTNF_V = { BQ: 32, BK: 128, TK: 8, PQ: 8, PD: 8, pv16: true, s16: true, il: true };
export function attnFFits(gpu, c) {
  try { return attnFCfg(c).bytes <= ((gpu.limits && gpu.limits.maxComputeWorkgroupStorageSize) || 16384); } catch (e) { return false; }
}
export function attnFCfg(c) {
  const D = c.D, BQ = c.BQ, BK = c.BK, TK = c.TK || 4, PQ = c.PQ || 4, PD = c.PD || 16, pv16 = !!c.pv16, kpad = c.kpad || 0, ppad = c.ppad || 0, il = !!c.il;
  const NT = (BQ / 4) * (BK / TK), NTv = (BQ / PQ) * (D / PD);
  if (NT !== NTv) throw new Error(`attnF: QK threads ${NT} != PV threads ${NTv}`);
  if (NT > 256 || TK % 4 || PD % 4) throw new Error('attnF: bad shape');
  const qV = D * BQ / 4, kChunk = 32 * (BK / 4 + kpad), pV = BK * (BQ / 4 + ppad), red = Math.max((BQ / 4) * (BK / TK) * 2, BQ);   // vec4<f16> counts (+ alpha/l rows)
  const kvBudget = 4096 - qV - pV;   // 32 KB = 4096 vec4<f16>; the KV region holds a K chunk, then the reduce, then V chunks
  let VK = c.VK || 0;
  if (!VK) { VK = 4; while (VK * 2 <= BK && VK * 2 * D / 4 <= Math.max(kvBudget, kChunk)) VK *= 2; }
  const kv = Math.max(kChunk, VK * D / 4, red);
  const bytes = (qV + kv + pV) * 8;
  if (bytes > 32768) throw new Error(`attnF: ${bytes} bytes of workgroup memory`);
  if (BK % VK) throw new Error('attnF: VK must divide BK');
  return { D, BQ, BK, TK, PQ, PD, pv16, NT, VK, qV, kv, pV, bytes, kpad, ppad, il, flavour: c.flavour || (D === 256 ? 'v' : 'p') };
}

export function attnFCode(c0) {
  const c = attnFCfg(c0), { D, BQ, BK, TK, PQ, PD, pv16, NT, VK } = c, D4 = D / 4, Q4 = BQ / 4, K4 = BK / 4, TX = BK / TK, e2 = c.flavour === 'p';
  const KR = K4 + c.kpad, QR = Q4 + c.ppad;   // padded row strides (vec4) of the K chunk and of P
  const L = [], ex = (x) => e2 ? `exp2(${x})` : `exp(${x})`;
  L.push('enable f16;');
  L.push(e2 ? 'struct P { Lq: u32, Lk: u32, ldo: u32, Hq: u32, Hkv: u32, scale: f32, mode: u32, nvalid: u32, qt0: u32, h0: u32, p0: u32, p1: u32 };'
    : 'struct P { Lq: u32, Lk: u32, ldo: u32, Hq: u32, Hkv: u32, scale: f32, mode: u32, nvalid: u32, qt0: u32, h0: u32, cap: f32, p1: u32 };');
  L.push('@group(0) @binding(0) var<uniform> p: P;');
  L.push('@group(0) @binding(1) var<storage, read> Qb: array<vec4<f16>>;');
  L.push('@group(0) @binding(2) var<storage, read> Kb: array<vec4<f16>>;');
  L.push('@group(0) @binding(3) var<storage, read> Vb: array<vec4<f16>>;');
  L.push('@group(0) @binding(4) var<storage, read_write> Ob: array<vec4<f16>>;');
  L.push(`var<workgroup> qT: array<vec4<f16>, ${c.qV}>;`);
  L.push(`var<workgroup> kv: array<vec4<f16>, ${c.kv}>;`);
  L.push(`var<workgroup> pS: array<vec4<f16>, ${c.pV}>;`);
  L.push('fn rd4(i: u32) -> vec4<f32> { return vec4<f32>(bitcast<vec2<f32>>(kv[i]), bitcast<vec2<f32>>(kv[i + 1u])); }');
  L.push('fn wr4(i: u32, v: vec4<f32>) { kv[i] = bitcast<vec4<f16>>(v.xy); kv[i + 1u] = bitcast<vec4<f16>>(v.zw); }');
  L.push(`@compute @workgroup_size(${NT})`);
  L.push('fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {');
  L.push(`  let h = wid.y + p.h0; let hk = h / (p.Hq / p.Hkv); let q0 = (wid.x + p.qt0) * ${BQ}u;`);
  L.push(`  let qb = h * p.Lq * ${D4}u; let kb = hk * p.Lk * ${D4}u; let ty = li / ${TX}u; let tx = li % ${TX}u;`);
  // Q tile -> qT[d][q4], transposed through registers
  L.push(`  for (var t = li; t < ${Q4 * D4}u; t = t + ${NT}u) { let qq4 = t / ${D4}u; let d4 = t % ${D4}u; let qa = q0 + qq4 * 4u; let lq = p.Lq - 1u;`);
  L.push(`    let a = Qb[qb + min(qa, lq) * ${D4}u + d4]; let b = Qb[qb + min(qa + 1u, lq) * ${D4}u + d4]; let c = Qb[qb + min(qa + 2u, lq) * ${D4}u + d4]; let e = Qb[qb + min(qa + 3u, lq) * ${D4}u + d4];`);
  L.push(`    let o = d4 * ${4 * Q4}u + qq4;`);
  ['x', 'y', 'z', 'w'].forEach((cc, j) => L.push(`    qT[o + ${j * Q4}u] = vec4<f16>(a.${cc}, b.${cc}, c.${cc}, e.${cc});`));
  L.push('  }');
  L.push(`  var kEnd = p.Lk; if (p.mode == 1u) { kEnd = min(p.Lk, min(q0 + ${BQ - 1}u, p.nvalid - 1u) + 1u); } if (p.mode == 2u) { kEnd = min(p.Lk, p.nvalid); }`);
  // PV mapping
  const PG = BQ / PQ;   // query groups in PV
  L.push(`  let py = li / ${D / PD}u; let pxd = li % ${D / PD}u;   // PV: rows py*${PQ}.., dims pxd*${PD}..`);
  const PC = PD / 4;   // vec4 columns per PV thread
  for (let r = 0; r < PQ; r++) for (let cI = 0; cI < PC; cI++) L.push(`  var o${r}_${cI} = vec4<f32>(0.0);`);
  L.push('  var m = vec4<f32>(-1.0e30); var l = vec4<f32>(0.0);');   // QK rows ty*4..+3
  if (PQ !== 4) { for (let r = 0; r < PQ; r++) L.push(`  var lp${r} = 0.0;`); }
  L.push('  let qi = q0 + ty * 4u + vec4<u32>(0u, 1u, 2u, 3u);');
  L.push(`  for (var k0 = 0u; k0 < kEnd; k0 = k0 + ${BK}u) {`);
  const TKV = TK / 4;   // score vec4s per row
  for (let r = 0; r < 4; r++) for (let j = 0; j < TKV; j++) L.push(`    var s${r}_${j} = vec4<f32>(0.0);`);
  L.push(`    for (var dc = 0u; dc < ${D}u; dc = dc + 32u) {`);
  if (c0.s16) for (let r = 0; r < 4; r++) for (let j = 0; j < TKV; j++) L.push(`      var g${r}_${j} = vec4<f16>(0.0);`);
  L.push('      workgroupBarrier();');
  // K chunk [32 d][K4]: tasks = K4 key-quads x 8 d-quads
  L.push(`      for (var t = li; t < ${K4 * 8}u; t = t + ${NT}u) { let kk4 = t / 8u; let dq = t % 8u; let d4 = dc / 4u + dq; let ka = k0 + kk4 * 4u; let lk = p.Lk - 1u;`);
  L.push(`        let a = Kb[kb + min(ka, lk) * ${D4}u + d4]; let b = Kb[kb + min(ka + 1u, lk) * ${D4}u + d4]; let c = Kb[kb + min(ka + 2u, lk) * ${D4}u + d4]; let e = Kb[kb + min(ka + 3u, lk) * ${D4}u + d4];`);
  L.push(`        let o = dq * ${4 * KR}u + kk4;`);
  ['x', 'y', 'z', 'w'].forEach((cc, j) => L.push(`        kv[o + ${j * KR}u] = vec4<f16>(a.${cc}, b.${cc}, c.${cc}, e.${cc});`));
  L.push('      }');
  L.push('      workgroupBarrier();');
  L.push('      for (var d = 0u; d < 32u; d = d + 1u) {');
  if (c0.s16) {   // fp16 partial dot products over the 32-d chunk (the fp16-everywhere design)
    L.push(`        let qv = qT[(dc + d) * ${Q4}u + ty];`);
    for (let j = 0; j < TKV; j++) L.push(`        let k${j} = kv[d * ${KR}u + tx * ${TKV}u + ${j}u];`);
    for (let r = 0; r < 4; r++) for (let j = 0; j < TKV; j++) L.push(`        g${r}_${j} = fma(vec4<f16>(qv.${'xyzw'[r]}), k${j}, g${r}_${j});`);
    L.push('      }');
    for (let r = 0; r < 4; r++) for (let j = 0; j < TKV; j++) L.push(`      s${r}_${j} = s${r}_${j} + vec4<f32>(g${r}_${j});`);
  } else {
  L.push(`        let qv = vec4<f32>(qT[(dc + d) * ${Q4}u + ty]);`);
  for (let j = 0; j < TKV; j++) L.push(`        let k${j} = vec4<f32>(kv[d * ${KR}u + tx * ${TKV}u + ${j}u]);`);
  for (let r = 0; r < 4; r++) for (let j = 0; j < TKV; j++) L.push(`        s${r}_${j} = fma(vec4<f32>(qv.${'xyzw'[r]}), k${j}, s${r}_${j});`);
  L.push('      }');
  }
  L.push('    }');
  // scale / cap / mask
  for (let r = 0; r < 4; r++) for (let j = 0; j < TKV; j++) L.push(`    s${r}_${j} = s${r}_${j} * p.scale;`);
  if (!e2) { L.push('    if (p.cap > 0.0) { let cp = vec4<f32>(p.cap);'); for (let r = 0; r < 4; r++) for (let j = 0; j < TKV; j++) L.push(`      s${r}_${j} = cp * tanh(clamp(s${r}_${j} / cp, vec4<f32>(-15.0), vec4<f32>(15.0)));`); L.push('    }'); }
  for (let j = 0; j < TKV; j++) {
    L.push(`    { let kj = k0 + tx * ${TK}u + ${j * 4}u + vec4<u32>(0u, 1u, 2u, 3u); var ok = kj < vec4<u32>(p.Lk); if (p.mode == 2u) { ok = ok & (kj < vec4<u32>(p.nvalid)); }`);
    for (let r = 0; r < 4; r++) L.push(`      { var mk = ok; if (p.mode == 1u) { mk = mk & (kj <= vec4<u32>(min(qi.${'xyzw'[r]}, p.nvalid - 1u))); } s${r}_${j} = select(vec4<f32>(-1.0e30), s${r}_${j}, mk); }`);
    L.push('    }');
  }
  const rowmax = (r) => { let e = `s${r}_0`; for (let j = 1; j < TKV; j++) e = `max(${e}, s${r}_${j})`; return `max(max((${e}).x, (${e}).y), max((${e}).z, (${e}).w))`; };
  L.push(`    let rb = ty * ${TX * 2}u;`);
  L.push('    workgroupBarrier();');
  L.push(`    wr4(rb + tx * 2u, vec4<f32>(${[0, 1, 2, 3].map(rowmax).join(', ')}));`);
  L.push('    workgroupBarrier();');
  L.push(`    var rm = rd4(rb); for (var j = 1u; j < ${TX}u; j = j + 1u) { rm = max(rm, rd4(rb + j * 2u)); }`);
  L.push(`    let mn = max(m, rm); let al = ${ex('m - mn')};`);
  for (let r = 0; r < 4; r++) for (let j = 0; j < TKV; j++) L.push(`    let e${r}_${j} = select(${ex(`s${r}_${j} - vec4<f32>(mn.${'xyzw'[r]})`)}, vec4<f32>(0.0), s${r}_${j} <= vec4<f32>(-1.0e29));`);
  for (let j = 0; j < TKV; j++) for (let k = 0; k < 4; k++) {
    const cc = 'xyzw'[k];
    L.push(`    pS[(tx * ${TK}u + ${j * 4 + k}u) * ${QR}u + ty] = vec4<f16>(vec4<f32>(e0_${j}.${cc}, e1_${j}.${cc}, e2_${j}.${cc}, e3_${j}.${cc}));`);
  }
  const rowsum = (r) => { let e = `e${r}_0`; for (let j = 1; j < TKV; j++) e += ` + e${r}_${j}`; return `dot(${e}, vec4<f32>(1.0))`; };
  L.push(`    let ls = vec4<f32>(${[0, 1, 2, 3].map(rowsum).join(', ')});`);
  L.push('    workgroupBarrier();');
  L.push('    wr4(rb + tx * 2u, ls);');
  L.push('    workgroupBarrier();');
  L.push(`    var rs = rd4(rb); for (var j = 1u; j < ${TX}u; j = j + 1u) { rs = rs + rd4(rb + j * 2u); }`);
  L.push('    l = l * al + rs; m = mn;');
  if (PQ === 4 && D / PD === TX) {   // the PV rows are the QK rows: alpha in registers
    for (let r = 0; r < 4; r++) for (let cI = 0; cI < PC; cI++) L.push(`    o${r}_${cI} = o${r}_${cI} * al.${'xyzw'[r]};`);
  } else {   // PV rows differ: alpha + l per row through shared (after the sums are read)
    L.push('    workgroupBarrier();');
    L.push(`    if (tx == 0u) { wr4(ty * 2u, al); wr4(${BQ / 4 * 2}u + ty * 2u, l); }`);
    L.push('    workgroupBarrier();');
    for (let r = 0; r < PQ; r++) {
      const row = `(py * ${PQ}u + ${r}u)`;
      L.push(`    { let ra = rd4((${row} / 4u) * 2u)[${row} % 4u]; lp${r} = rd4(${BQ / 4 * 2}u + (${row} / 4u) * 2u)[${row} % 4u];`);
      for (let cI = 0; cI < PC; cI++) L.push(`      o${r}_${cI} = o${r}_${cI} * ra;`);
      L.push('    }');
    }
  }
  L.push(`    for (var vc = 0u; vc < ${BK}u; vc = vc + ${VK}u) {`);
  L.push('      workgroupBarrier();');
  L.push(`      for (var t = li; t < ${VK * D4}u; t = t + ${NT}u) { kv[t] = Vb[kb + min(k0 + vc + t / ${D4}u, p.Lk - 1u) * ${D4}u + t % ${D4}u]; }`);
  L.push('      workgroupBarrier();');
  const PQ4 = PQ / 4;   // p vec4s per PV thread
  if (pv16) for (let r = 0; r < PQ; r++) for (let cI = 0; cI < PC; cI++) L.push(`      var h${r}_${cI} = vec4<f16>(0.0);`);
  L.push(`      for (var j = 0u; j < ${VK}u; j = j + 1u) {`);
  L.push(`        let vb = j * ${D4}u + ${c.il ? 'pxd' : `pxd * ${PC}u`};`);
  for (let q = 0; q < PQ4; q++) L.push(`        let pv${q} = ${pv16 ? '' : 'vec4<f32>('}pS[(vc + j) * ${QR}u + py * ${PQ4}u + ${q}u]${pv16 ? '' : ')'};`);
  for (let cI = 0; cI < PC; cI++) L.push(`        let v${cI} = ${pv16 ? '' : 'vec4<f32>('}kv[vb + ${c.il ? cI * (D / PD) : cI}u]${pv16 ? '' : ')'};`);
  for (let r = 0; r < PQ; r++) for (let cI = 0; cI < PC; cI++) {
    const pr = `pv${r >> 2}.${'xyzw'[r & 3]}`;
    if (pv16) L.push(`        h${r}_${cI} = fma(vec4<f16>(${pr}), v${cI}, h${r}_${cI});`);
    else L.push(`        o${r}_${cI} = fma(vec4<f32>(${pr}), v${cI}, o${r}_${cI});`);
  }
  L.push('      }');
  if (pv16) for (let r = 0; r < PQ; r++) for (let cI = 0; cI < PC; cI++) L.push(`      o${r}_${cI} = o${r}_${cI} + vec4<f32>(h${r}_${cI});`);
  L.push('    }');
  L.push('  }');
  for (let r = 0; r < PQ; r++) {
    const qrow = `(q0 + py * ${PQ}u + ${r}u)`, lv = (PQ === 4 && D / PD === TX) ? `l.${'xyzw'[r]}` : `lp${r}`;
    L.push(`  { let qr = ${qrow}; if (qr < p.Lq) { let inv = 1.0 / ${lv}; let ob = (qr * p.ldo + h * ${D}u) / 4u + ${c.il ? 'pxd' : `pxd * ${PC}u`};`);
    for (let cI = 0; cI < PC; cI++) L.push(`    Ob[ob + ${c.il ? cI * (D / PD) : cI}u] = vec4<f16>(o${r}_${cI} * inv);`);
    L.push('  } }');
  }
  L.push('}');
  return L.join('\n');
}

/** dispatch like the engine (head-major q/k/v, token-major o); flavour 'v' = attn256 params (cap, natural exp), 'p' = attention() (exp2) */
export function attnF(gpu, a, c) {
  const cfg = attnFCfg(c), BQ = cfg.BQ, D = cfg.D, key = 'attnF.' + JSON.stringify(c);
  const pipe = gpu.pipeline(key, () => attnFCode(c));
  const Hq = a.Hq, Hkv = a.Hkv || Hq, Lq = a.Lq, Lk = a.Lk || Lq, ldo = a.o.ld || Hq * D, qt = Math.ceil(Lq / BQ);
  const rate = c.rate || 0.6e12, fWG = 4 * BQ * Lk * D, cap = rate * 0.2;
  let qPer = qt, hPer = Math.max(1, Math.floor(cap / (fWG * qt)));
  if (fWG * qt > cap) { hPer = 1; qPer = Math.max(1, Math.floor(cap / fWG)); }
  const vw = (t, n) => ({ buffer: t.buf, offset: t.off || 0, size: Math.floor(n / 16) * 16 });
  const binds = [vw(a.q, Hq * Lq * D * 2), vw(a.k, Hkv * Lk * D * 2), vw(a.v, Hkv * Lk * D * 2), { buffer: a.o.buf, offset: a.o.off || 0, size: Math.ceil(Lq * ldo * 2 / 16) * 16 }];
  const scale = cfg.flavour === 'p' ? (a.scale != null ? a.scale : 1 / Math.sqrt(D)) * Math.LOG2E : (a.scale != null ? a.scale : 1 / Math.sqrt(D));
  for (let h0 = 0; h0 < Hq; h0 += hPer) for (let q0 = 0; q0 < qt; q0 += qPer) {
    const nh = Math.min(hPer, Hq - h0), nq = Math.min(qPer, qt - q0);
    const pr = P().u32(Lq).u32(Lk).u32(ldo).u32(Hq).u32(Hkv).f32(scale).u32(a.mode || 0).u32(a.nvalid || Lk).u32(q0).u32(h0);
    if (cfg.flavour === 'p') pr.u32(0).u32(0); else pr.f32(a.cap || 0).u32(0);
    gpu.runner.dispatch(pipe, binds, pr.ab, [nq, nh], fWG * nq * nh / rate);
  }
}
