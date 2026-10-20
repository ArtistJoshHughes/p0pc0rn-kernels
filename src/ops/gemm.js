// SPDX-License-Identifier: MPL-2.0
// Copyright 2026 Joshua Hughes
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// GEMM family:  Y[m, n] = sum_k X'[m, k] * W[n, k]   (y = x W^T), weights in the encodings of docs/FORMAT.md,
// dequantised INSIDE the kernel (codes -> fp16 tile in workgroup memory, never expanded in device memory).
// fp16 tiles, fp32 accumulation (acc:'f16' variant = fp16 partial sums per 32-k step, for autotune only).
//
// Prologues (x):  'f16'   X fp16 row-major [M, ldx]
//                 'lnmod' X fp32 residual; X' = (x - mean[m]) * rstd[m] * (1 + scale[k]) + shift[k]    (LayerNorm + adaLN modulation fused)
//                 'rms'   X fp32 residual; X' = x * rstd[m] * w[k]                                     (RMSNorm fused)
//                 'conv'  implicit-GEMM conv over an HWC fp16 image (3x3 / 1x1, stride 1/2, fused nearest-2x upsample,
//                         optional fused GroupNorm-apply + SiLU: X' = silu(a[c] * x + b[c]), zero padding applied AFTER it)
// Epilogues (epi): 'f16' | 'f32' store;  'resadd' R_f32 += gate[n] * acc (fp32 residual, gate optional);
//                 'swiglu' (weight rows [gate | up]) Y = silu(gate) * up;  'heads' QK-RMSNorm + RoPE + head-major Q/K/V write;
//                 'bias' fp16 store of acc + bias[n] (+ residual Y in place when res=true)  (VAE convs / linears)
//                 resadd / swiglu take an optional bias vector bv {buf, off, up} (SANA linears have biases); resadd store:true writes
//                 Y = gate * (acc + bias) instead of adding; swiglu act:'gelu' = GeGLU (Gemma-2, tanh GELU)
'use strict';
import { P } from '../core/gpu.js';

export const DEFAULT_TILE = { BM: 64, BN: 128 };

function keyOf(c) {
  return ['gemm', c.w, c.x, c.epi, c.BM, c.BN, c.acc || 'f32', c.rope || '-', c.sg ? 'sg' : '', c.gate ? 'g' : '', c.res ? 'r' : '',
    c.gn ? 'gn' : '', c.up2 ? 'up' : '', c.pn ? 'pn' : '', c.bias ? 'b' : '', c.store ? 'st' : '', c.act === 'gelu' ? 'gelu' : ''].join('.');
}

export function gemmCode(c) {
  const BM = c.BM, BN = c.BN, TM = 8, TN = 8, TX = BN / TN, TY = BM / TM, NT = TX * TY;
  if (NT > 256 || BM % 32 || BN % 64) throw new Error('bad tile ' + BM + 'x' + BN);
  const split = c.epi === 'heads' && c.rope === 'half';
  const xf32 = c.x === 'lnmod' || c.x === 'rms';
  const B = [];                       // storage bindings in order (binding i+1)
  const bind = (name, decl) => { B.push(name); return `@group(0) @binding(${B.length}) var<storage, ${decl}`; };
  const L = [];
  L.push('enable f16;');
  if (c.sg) L.push('enable subgroups;');
  L.push(`struct P { M: u32, N: u32, K: u32, ldx: u32, kx0: u32, wR0: u32, wR1: u32, wG: u32, wk0: u32, ldy: u32, ny0: u32, mt0: u32, nt0: u32,
  rowOff: u32, Ltot: u32, Hq: u32, Hkv: u32, eps: f32, pvOff: u32, pvStride: u32, evOff: u32, statOff: u32,
  Hi: u32, Wi: u32, Ci: u32, Ho: u32, Wo: u32, stride: u32, padT: u32, padL: u32, ks: u32, Nw: u32, bvOff: u32, bvUp: u32 };`);
  L.push('@group(0) @binding(0) var<uniform> p: P;');
  L.push(bind('X', `read> X: array<${xf32 ? 'vec4<f32>' : 'vec4<f16>'}>;`));
  L.push(bind('W', 'read> W: array<u32>;'));
  const yT = c.epi === 'f32' || c.epi === 'resadd' ? 'vec4<f32>' : 'vec4<f16>';
  L.push(bind('Y', `${c.epi === 'resadd' || (c.epi === 'bias' && c.res) ? 'read_write' : 'read_write'}> Y: array<${yT}>;`));
  if (xf32 || (c.x === 'conv3' && c.pn)) L.push(bind('ST', 'read> ST: array<f32>;'));
  if (xf32 || (c.x === 'conv' && c.gn)) L.push(bind('PV', 'read> PV: array<vec4<f32>>;'));
  const red128 = c.epi === 'heads' || c.epi === 'onormgate';      // epilogues that reduce over one 128-wide head
  const needEV = (c.epi === 'resadd' && c.gate) || red128 || c.epi === 'bias';
  if (needEV) L.push(bind('EV', 'read> EV: array<f32>;'));
  if (c.epi === 'onormgate') L.push(bind('GT', 'read> GT: array<vec4<f16>>;'));
  if (c.bias) L.push(bind('BV', 'read> BV: array<f32>;'));   // bias vector for 'resadd' (+ store) and 'swiglu' (gate at bvOff, up at bvOff + bvUp)
  if (c.epi === 'heads') L.push(bind('RP', 'read> RP: array<f32>;'));
  L.push(`var<workgroup> xs: array<vec4<f16>, ${32 * BM / 4}>;`);
  L.push(`var<workgroup> ws: array<vec4<f16>, ${32 * BN / 4}>;`);
  if (red128 && !c.sg) L.push(`var<workgroup> red: array<f32, ${BM * TX}>;`);

  // ---- one prologue-applied X vector: row `row`, k = k0 + 4*kq .. +3
  L.push(`fn xload(row: u32, k0: u32, kq: u32) -> vec4<f16> {`);
  if (c.x === 'f16') {
    L.push(`      let v = X[(row * p.ldx + p.kx0 + k0) / 4u + kq];`);
  } else if (xf32) {
    L.push(`      let xv = X[(row * p.ldx + p.kx0 + k0) / 4u + kq];`);
    L.push(`      let sr = (p.statOff + row) * 2u; let kk = (p.kx0 + k0) / 4u + kq;`);
    if (c.x === 'lnmod') L.push(`      let v = vec4<f16>((xv - vec4<f32>(ST[sr])) * ST[sr + 1u] * (vec4<f32>(1.0) + PV[p.pvOff / 4u + kk]) + PV[(p.pvOff + p.pvStride) / 4u + kk]);`);
    else L.push(`      let v = vec4<f16>(xv * ST[sr + 1u] * PV[p.pvOff / 4u + kk]);`);
  } else if (c.x === 'conv3') {   // 3x3x3 conv over THWC, stride 1; time padding = replicate (LTX causal=False), space = zeros; p.stride = Ti
    L.push(`      let HW = p.Ho * p.Wo; let ot = row / HW; let rem = row % HW; let oy = rem / p.Wo; let ox = rem % p.Wo;`);
    L.push(`      let tap = k0 / p.Ci; let c0 = k0 % p.Ci; let kt = tap / 9u; let ky = (tap / 3u) % 3u; let kx = tap % 3u;`);
    // p0pc0rn v029 time chunks: output frame = row frame + padT; the input frame is t - padL (a buffer that starts at frame padL) or
    // t % ks (a ring of ks frames, ks > 1).  padT = padL = 0, ks = 1 is the whole clip, as before.
    L.push(`      let itg = u32(clamp(i32(ot + p.padT + kt) - 1, 0, i32(p.stride) - 1)); let it = select(itg - p.padL, itg % p.ks, p.ks > 1u); let iy = i32(oy + ky) - 1; let ix = i32(ox + kx) - 1;`);
    L.push(`      let ok = iy >= 0 && ix >= 0 && iy < i32(p.Hi) && ix < i32(p.Wi);`);
    L.push(`      let vox = (it * p.Hi + u32(clamp(iy, 0, i32(p.Hi) - 1))) * p.Wi + u32(clamp(ix, 0, i32(p.Wi) - 1));`);
    L.push(`      let xv = X[vox * (p.Ci / 4u) + c0 / 4u + kq];`);
    if (c.pn) {   // LTX PixelNorm (per-voxel rstd over channels) + SiLU, zero padding after the transform
      L.push(`      let z = vec4<f32>(xv) * ST[vox];`);
      L.push(`      let v = select(vec4<f16>(0.0), vec4<f16>(z / (vec4<f32>(1.0) + exp(-z))), ok);`);
    } else L.push(`      let v = select(vec4<f16>(0.0), xv, ok);`);
  } else {   // conv
    L.push(`      let oy = row / p.Wo; let ox = row % p.Wo; let tap = k0 / p.Ci; let c0 = k0 % p.Ci;`);
    L.push(`      let iy = i32(oy * p.stride + tap / p.ks) - i32(p.padT); let ix = i32(ox * p.stride + tap % p.ks) - i32(p.padL);`);
    const HH = c.up2 ? 'i32(p.Hi * 2u)' : 'i32(p.Hi)', WW = c.up2 ? 'i32(p.Wi * 2u)' : 'i32(p.Wi)';
    L.push(`      let ok = iy >= 0 && ix >= 0 && iy < ${HH} && ix < ${WW};`);
    L.push(`      let sy = u32(max(iy, 0))${c.up2 ? ' >> 1u' : ''}; let sx = u32(max(ix, 0))${c.up2 ? ' >> 1u' : ''};`);
    L.push(`      let xv = X[(min(sy, p.Hi - 1u) * p.Wi + min(sx, p.Wi - 1u)) * (p.Ci / 4u) + c0 / 4u + kq];`);
    if (c.gn) {
      L.push(`      let cq = c0 / 4u + kq; let z = vec4<f32>(xv) * PV[p.pvOff / 4u + cq] + PV[(p.pvOff + p.pvStride) / 4u + cq];`);
      L.push(`      let v = select(vec4<f16>(0.0), vec4<f16>(z / (vec4<f32>(1.0) + exp(-z))), ok);`);
    } else L.push(`      let v = select(vec4<f16>(0.0), xv, ok);`);
  }

  L.push('  return v;');
  L.push('}');

  // ---- weight row for ws column cc (0..BN-1) of this workgroup
  L.push(`fn wrow(nb: u32, cc: u32) -> u32 {`);
  if (c.epi === 'swiglu') L.push(`  let o = (nb + cc) >> 1u; return select(p.wR1, p.wR0, (cc & 1u) == 0u) + o;`);
  else L.push(`  return min(p.wR0 + nb + cc, p.Nw - 1u);`);
  L.push('}');

  L.push(`@compute @workgroup_size(${NT})`);
  L.push(`fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32${c.sg ? ', @builtin(subgroup_invocation_id) sgi: u32' : ''}) {`);
  L.push(`  let mB = (wid.y + p.mt0) * ${BM}u; let nB = (wid.x + p.nt0) * ${BN}u;`);
  L.push(`  let tx = li % ${TX}u; let ty = li / ${TX}u;`);
  for (let r = 0; r < TM; r++) for (let j = 0; j < 2; j++) L.push(`  var a${r}_${j} = vec4<f32>(0.0);`);
  // p0pc0rn v028: conv3 hoists its voxel indices (rows once per thread, the tap once per 32-k step): bit-identical, iPhone 1.09-1.15x
  const CV3P = c.x === 'conv3' && BM * 2 <= NT;
  if (CV3P) {
    L.push('  let lmP = p.M - 1u; let HWP = p.Ho * p.Wo; let rbP = mB + (li >> 3u) * 4u;');
    L.push('  let r4 = min(vec4<u32>(rbP) + vec4<u32>(0u, 1u, 2u, 3u), vec4<u32>(lmP));');
    L.push('  let cot = r4 / vec4<u32>(HWP); let rem4 = r4 - cot * HWP; let coy = vec4<i32>(rem4 / vec4<u32>(p.Wo)); let cox = vec4<i32>(rem4 % vec4<u32>(p.Wo));');
  }
  L.push(`  for (var k0 = 0u; k0 < p.K; k0 = k0 + 32u) {`);
  // ---- X tile -> xs[k][m/4] (transposed through registers: 4 rows x 4 k per task, WHOLE vec4 stores only.
  //      Never write single components of a shared vec4: on Metal that is a read-modify-write of the vector and races.)
  if (CV3P) {
    L.push(`    if (li < ${BM * 2}u) { let t = li;`);
    L.push('      let mq = t >> 3u; let kq = t & 7u;');
    L.push('      let tap = k0 / p.Ci; let c0 = k0 - tap * p.Ci; let kt = tap / 9u; let r9 = tap - kt * 9u; let ky = r9 / 3u; let kx = r9 - ky * 3u;');
    L.push('      let itg4 = vec4<u32>(clamp(vec4<i32>(cot + vec4<u32>(kt + p.padT)) - 1, vec4<i32>(0), vec4<i32>(i32(p.stride) - 1)));');
    L.push('      let it4 = select(itg4 - vec4<u32>(p.padL), itg4 % vec4<u32>(p.ks), p.ks > 1u);');
    L.push('      let iy4 = coy + i32(ky) - 1; let ix4 = cox + i32(kx) - 1;');
    L.push('      let ok4 = (iy4 >= vec4<i32>(0)) & (ix4 >= vec4<i32>(0)) & (iy4 < vec4<i32>(i32(p.Hi))) & (ix4 < vec4<i32>(i32(p.Wi)));');
    L.push('      let vox4 = (it4 * p.Hi + vec4<u32>(clamp(iy4, vec4<i32>(0), vec4<i32>(i32(p.Hi) - 1)))) * p.Wi + vec4<u32>(clamp(ix4, vec4<i32>(0), vec4<i32>(i32(p.Wi) - 1)));');
    L.push('      let cb = p.Ci / 4u; let co = c0 / 4u + kq;');
    for (let i = 0; i < 4; i++) {
      const cc = 'xyzw'[i];
      L.push(`      let xv${i} = X[vox4.${cc} * cb + co];`);
      if (c.pn) L.push(`      let z${i} = vec4<f32>(xv${i}) * ST[vox4.${cc}]; let v${i} = select(vec4<f16>(0.0), vec4<f16>(z${i} / (vec4<f32>(1.0) + exp(-z${i}))), ok4.${cc});`);
      else L.push(`      let v${i} = select(vec4<f16>(0.0), xv${i}, ok4.${cc});`);
    }
  } else {
  L.push(`    for (var t = li; t < ${BM * 2}u; t = t + ${NT}u) {`);
  L.push(`      let mq = t >> 3u; let kq = t & 7u; let r0 = mB + mq * 4u; let lm = p.M - 1u;`);
  L.push(`      let v0 = xload(min(r0, lm), k0, kq); let v1 = xload(min(r0 + 1u, lm), k0, kq); let v2 = xload(min(r0 + 2u, lm), k0, kq); let v3 = xload(min(r0 + 3u, lm), k0, kq);`);
  }
  L.push(`      let xo = kq * 4u * ${BM / 4}u + mq;`);
  ['x', 'y', 'z', 'w'].forEach((c, j) => L.push(`      xs[xo + ${j * BM / 4}u] = vec4<f16>(v0.${c}, v1.${c}, v2.${c}, v3.${c});`));
  L.push('    }');
  // ---- W tile -> ws[k][n] (dequantised to fp16)
  if (c.w === 'q4') {
    L.push(`    let gi = (p.wk0 + k0) / 32u;`);
    L.push(`    for (var t = li; t < ${BN}u; t = t + ${NT}u) {`);
    L.push(`      let cq = t % ${BN / 4}u; let w = t / ${BN / 4}u;`);
    L.push(`      var q: vec4<u32>; var sc: vec4<f32>; var mn: vec4<f32>;`);
    L.push(`      for (var j = 0u; j < 4u; j = j + 1u) { let r = wrow(nB, cq * 4u + j); let rec = ((r >> 6u) * p.wG + gi) * 320u; let rr = r & 63u;`);
    L.push(`        q[j] = W[rec + rr * 4u + w]; let sm = unpack2x16float(W[rec + 256u + rr]); sc[j] = sm.x; mn[j] = sm.y; }`);
    L.push(`      for (var i = 0u; i < 8u; i = i + 1u) {`);
    L.push(`        let cv = vec4<f32>((q >> vec4<u32>(4u * i)) & vec4<u32>(15u));`);
    L.push(`        ws[(w * 8u + i) * ${BN / 4}u + cq] = vec4<f16>(cv * sc + mn); }`);
    L.push('    }');
  } else if (c.w === 'q8') {
    L.push(`    let gi = (p.wk0 + k0) / 64u; let hf = ((p.wk0 + k0) / 32u) & 1u;`);
    L.push(`    for (var t = li; t < ${BN * 2}u; t = t + ${NT}u) {`);
    L.push(`      let cq = t % ${BN / 4}u; let w = t / ${BN / 4}u;`);
    L.push(`      var q: vec4<u32>; var sc: vec4<f32>;`);
    L.push(`      for (var j = 0u; j < 4u; j = j + 1u) { let r = wrow(nB, cq * 4u + j); let rec = ((r >> 6u) * p.wG + gi) * 1056u; let rr = r & 63u;`);
    L.push(`        q[j] = W[rec + rr * 16u + hf * 8u + w]; let s2 = unpack2x16float(W[rec + 1024u + (rr >> 1u)]); sc[j] = select(s2.x, s2.y, (rr & 1u) == 1u); }`);
    L.push(`      for (var i = 0u; i < 4u; i = i + 1u) {`);
    L.push(`        let cv = vec4<f32>((vec4<i32>(q << vec4<u32>(24u - 8u * i))) >> vec4<u32>(24u));`);
    L.push(`        ws[(w * 4u + i) * ${BN / 4}u + cq] = vec4<f16>(cv * sc); }`);
    L.push('    }');
  } else {   // f16 row-major, wG = row stride in halves
    L.push(`    for (var t = li; t < ${BN * 4}u; t = t + ${NT}u) {`);
    L.push(`      let cq = t % ${BN / 4}u; let w = t / ${BN / 4}u;`);
    L.push(`      var lo: vec4<f16>; var hi: vec4<f16>;`);
    L.push(`      for (var j = 0u; j < 4u; j = j + 1u) { let r = wrow(nB, cq * 4u + j); let h2 = bitcast<vec2<f16>>(W[(r * p.wG + p.wk0 + k0) / 2u + w]); lo[j] = h2.x; hi[j] = h2.y; }`);
    L.push(`      ws[(w * 2u) * ${BN / 4}u + cq] = lo; ws[(w * 2u + 1u) * ${BN / 4}u + cq] = hi;`);
    L.push('    }');
  }
  L.push('    workgroupBarrier();');
  // ---- register-tiled outer product: 8 rows x 8 cols per thread
  const w0 = split ? `ws[kb + tx]` : `ws[kb + tx * 2u]`, w1 = split ? `ws[kb + ${BN / 8}u + tx]` : `ws[kb + tx * 2u + 1u]`;
  if (c.acc === 'f16') for (let r = 0; r < TM; r++) for (let j = 0; j < 2; j++) L.push(`    var h${r}_${j} = vec4<f16>(0.0);`);
  L.push(`    for (var k = 0u; k < 32u; k = k + 1u) {`);
  L.push(`      let xb = k * ${BM / 4}u + ty * 2u; let kb = k * ${BN / 4}u;`);
  if (c.acc === 'f16') {
    L.push(`      let x0 = xs[xb]; let x1 = xs[xb + 1u]; let u0 = ${w0}; let u1 = ${w1};`);
    for (let r = 0; r < TM; r++) { const xv = `x${r >> 2}[${r & 3}]`; L.push(`      h${r}_0 = fma(vec4<f16>(${xv}), u0, h${r}_0); h${r}_1 = fma(vec4<f16>(${xv}), u1, h${r}_1);`); }
  } else {
    L.push(`      let x0 = vec4<f32>(xs[xb]); let x1 = vec4<f32>(xs[xb + 1u]); let u0 = vec4<f32>(${w0}); let u1 = vec4<f32>(${w1});`);
    for (let r = 0; r < TM; r++) { const xv = `x${r >> 2}[${r & 3}]`; L.push(`      a${r}_0 = fma(vec4<f32>(${xv}), u0, a${r}_0); a${r}_1 = fma(vec4<f32>(${xv}), u1, a${r}_1);`); }
  }
  L.push('    }');
  if (c.acc === 'f16') for (let r = 0; r < TM; r++) for (let j = 0; j < 2; j++) L.push(`    a${r}_${j} = a${r}_${j} + vec4<f32>(h${r}_${j});`);
  L.push('    workgroupBarrier();');
  L.push('  }');

  // ---- epilogues
  const rows = (fn) => { for (let r = 0; r < TM; r++) L.push(`  { let m = mB + ty * 8u + ${r}u; if (m < p.M) { ${fn(r)} } }`); };
  const col = split ? null : `nB + tx * 8u`;
  if (c.epi === 'f16' || c.epi === 'f32') {
    const T = c.epi === 'f16' ? 'vec4<f16>' : 'vec4<f32>';
    rows((r) => `let b = (m * p.ldy + p.ny0 + ${col}) / 4u; if (${col} < p.N) { Y[b] = ${T}(a${r}_0); } if (${col} + 4u < p.N) { Y[b + 1u] = ${T}(a${r}_1); }`);
  } else if (c.epi === 'resadd') {
    if (c.gate) L.push(`  let gb = p.evOff + ${col}; let g0 = vec4<f32>(EV[gb], EV[gb + 1u], EV[gb + 2u], EV[gb + 3u]); let g1 = vec4<f32>(EV[gb + 4u], EV[gb + 5u], EV[gb + 6u], EV[gb + 7u]);`);
    else L.push('  let g0 = vec4<f32>(1.0); let g1 = g0;');
    if (c.bias) L.push(`  let vb = p.bvOff + ${col}; let c0 = vec4<f32>(BV[vb], BV[vb + 1u], BV[vb + 2u], BV[vb + 3u]); let c1 = vec4<f32>(BV[vb + 4u], BV[vb + 5u], BV[vb + 6u], BV[vb + 7u]);`);
    else L.push('  let c0 = vec4<f32>(0.0); let c1 = c0;');
    const prev = (i) => (c.store ? '' : `Y[${i}] + `);
    rows((r) => `let b = (m * p.ldy + p.ny0 + ${col}) / 4u; if (${col} < p.N) { Y[b] = ${prev('b')}g0 * (a${r}_0 + c0); } if (${col} + 4u < p.N) { Y[b + 1u] = ${prev('b + 1u')}g1 * (a${r}_1 + c1); }`);
  } else if (c.epi === 'swiglu') {
    // ws columns interleave (gate n, up n): a_0 = (g0,u0,g1,u1), a_1 = (g2,u2,g3,u3)
    if (c.bias) L.push(`  let ob = p.bvOff + (nB >> 1u) + tx * 4u; let gB = vec4<f32>(BV[ob], BV[ob + 1u], BV[ob + 2u], BV[ob + 3u]); let uB = vec4<f32>(BV[ob + p.bvUp], BV[ob + p.bvUp + 1u], BV[ob + p.bvUp + 2u], BV[ob + p.bvUp + 3u]);`);
    else L.push('  let gB = vec4<f32>(0.0); let uB = gB;');
    const act = c.act === 'gelu' ? 'g * 0.5 * (vec4<f32>(1.0) + tanh(clamp(0.7978845608 * (g + 0.044715 * g * g * g), vec4<f32>(-15.0), vec4<f32>(15.0))))' : 'g / (vec4<f32>(1.0) + exp(-g))';
    rows((r) => `let g = vec4<f32>(a${r}_0.x, a${r}_0.z, a${r}_1.x, a${r}_1.z) + gB; let u = vec4<f32>(a${r}_0.y, a${r}_0.w, a${r}_1.y, a${r}_1.w) + uB;
      Y[(m * p.ldy + p.ny0 + (nB >> 1u) + tx * 4u) / 4u] = vec4<f16>(${act} * u);`);
  } else if (c.epi === 'bias') {
    L.push(`  let bb = p.evOff + ${col}; let b0 = vec4<f32>(EV[bb], EV[bb + 1u], EV[bb + 2u], EV[bb + 3u]); let b1 = vec4<f32>(EV[bb + 4u], EV[bb + 5u], EV[bb + 6u], EV[bb + 7u]);`);
    rows((r) => `let b = (m * p.ldy + p.ny0 + ${col}) / 4u;
      if (${col} < p.N) { Y[b] = vec4<f16>(a${r}_0 + b0${c.res ? ' + vec4<f32>(Y[b])' : ''}); }
      if (${col} + 4u < p.N) { Y[b + 1u] = vec4<f16>(a${r}_1 + b1${c.res ? ' + vec4<f32>(Y[b + 1u])' : ''}); }`);
  } else if (red128) {
    if (BN !== 128) throw new Error(c.epi + ' epilogue needs BN = 128 (one head per workgroup column)');
    // per-row sum of squares over the 128 head dims (16 threads x 8 cols)
    for (let r = 0; r < TM; r++) L.push(`  var s${r} = dot(a${r}_0, a${r}_0) + dot(a${r}_1, a${r}_1);`);
    if (c.sg) {
      for (let r = 0; r < TM; r++) L.push(`  s${r} = s${r} + subgroupShuffleXor(s${r}, 1u); s${r} = s${r} + subgroupShuffleXor(s${r}, 2u); s${r} = s${r} + subgroupShuffleXor(s${r}, 4u); s${r} = s${r} + subgroupShuffleXor(s${r}, 8u);`);
    } else {
      for (let r = 0; r < TM; r++) L.push(`  red[(ty * 8u + ${r}u) * 16u + tx] = s${r};`);
      L.push('  workgroupBarrier();');
      for (let r = 0; r < TM; r++) L.push(`  { var t = 0.0; for (var i = 0u; i < 16u; i = i + 1u) { t = t + red[(ty * 8u + ${r}u) * 16u + i]; } s${r} = t; }`);
    }
    if (c.epi === 'onormgate') {   // SANA linear attention read-out: o_norm (RMS over the head) * w, * sigmoid(gate)
      L.push(`  let c0 = tx * 8u; let n0 = vec4<f32>(EV[p.evOff + c0], EV[p.evOff + c0 + 1u], EV[p.evOff + c0 + 2u], EV[p.evOff + c0 + 3u]); let n1 = vec4<f32>(EV[p.evOff + c0 + 4u], EV[p.evOff + c0 + 5u], EV[p.evOff + c0 + 6u], EV[p.evOff + c0 + 7u]);`);
      rows((r) => `let inv = inverseSqrt(s${r} / 128.0 + p.eps); let gi = (m * p.Ltot + p.ny0 + nB + c0) / 4u;
        let g0 = vec4<f32>(GT[gi]); let g1 = vec4<f32>(GT[gi + 1u]);
        let b = (m * p.ldy + p.ny0 + nB + c0) / 4u;
        Y[b] = vec4<f16>(a${r}_0 * inv * n0 / (vec4<f32>(1.0) + exp(-g0))); Y[b + 1u] = vec4<f16>(a${r}_1 * inv * n1 / (vec4<f32>(1.0) + exp(-g1)));`);
    } else {
    L.push(`  let hd = nB / 128u; let sec = select(select(2u, 1u, hd < p.Hq + p.Hkv), 0u, hd < p.Hq);`);
    L.push(`  let c0 = ${split ? 'tx * 4u' : 'tx * 8u'}; let nwo = p.evOff + select(128u, 0u, sec == 0u);`);
    if (split) L.push(`  let n0 = vec4<f32>(EV[nwo + c0], EV[nwo + c0 + 1u], EV[nwo + c0 + 2u], EV[nwo + c0 + 3u]); let n1 = vec4<f32>(EV[nwo + c0 + 64u], EV[nwo + c0 + 65u], EV[nwo + c0 + 66u], EV[nwo + c0 + 67u]);`);
    else L.push(`  let n0 = vec4<f32>(EV[nwo + c0], EV[nwo + c0 + 1u], EV[nwo + c0 + 2u], EV[nwo + c0 + 3u]); let n1 = vec4<f32>(EV[nwo + c0 + 4u], EV[nwo + c0 + 5u], EV[nwo + c0 + 6u], EV[nwo + c0 + 7u]);`);
    rows((r) => {
      let s = `let row = p.rowOff + m; var v0 = a${r}_0; var v1 = a${r}_1;
      if (sec < 2u) { let inv = inverseSqrt(s${r} / 128.0 + p.eps); v0 = v0 * inv * n0; v1 = v1 * inv * n1;`;
      if (split) {   // Qwen rotate_half: pair (d, d+64), angle index d
        s += ` let rb = row * 128u + c0 * 2u; let cs = vec4<f32>(RP[rb], RP[rb + 2u], RP[rb + 4u], RP[rb + 6u]); let sn = vec4<f32>(RP[rb + 1u], RP[rb + 3u], RP[rb + 5u], RP[rb + 7u]);
        let lo = v0 * cs - v1 * sn; let hi = v1 * cs + v0 * sn; v0 = lo; v1 = hi; }
        let ob = (hd * p.Ltot + row) * 32u + tx; Y[ob] = vec4<f16>(v0); Y[ob + 16u] = vec4<f16>(v1);`;
      } else {       // FLUX interleaved pairs (2p, 2p+1), pair index p = col/2
        s += ` let rb = row * 128u + c0; let c4 = vec4<f32>(RP[rb], RP[rb + 2u], RP[rb + 4u], RP[rb + 6u]); let s4 = vec4<f32>(RP[rb + 1u], RP[rb + 3u], RP[rb + 5u], RP[rb + 7u]);
        v0 = vec4<f32>(v0.x * c4.x - v0.y * s4.x, v0.x * s4.x + v0.y * c4.x, v0.z * c4.y - v0.w * s4.y, v0.z * s4.y + v0.w * c4.y);
        v1 = vec4<f32>(v1.x * c4.z - v1.y * s4.z, v1.x * s4.z + v1.y * c4.z, v1.z * c4.w - v1.w * s4.w, v1.z * s4.w + v1.w * c4.w); }
        let ob = (hd * p.Ltot + row) * 32u + tx * 2u; Y[ob] = vec4<f16>(v0); Y[ob + 1u] = vec4<f16>(v1);`;
      }
      return s;
    });
  }
    }
  L.push('}');
  return { code: L.join('\n'), bindings: B };
}

/**
 * Run a GEMM.  a = {
 *   w: {buf, off, enc:'q4g32a.t64'|'q8g64s.t64'|'f16', rows (Nw), cols (Ktot)}   weight tensor (whole tensor; sub-ranges via r0/r1/k0)
 *   x: {buf, off, ld, k0=0, dtype}  activations;  M, N (output/ws columns of this call), K (reduction length)
 *   wr0=0 (first weight row), wr1 (swiglu: first 'up' row), wk0=0 (first k column of the weight)
 *   epi, y: {buf, off, ld, n0=0}, gate/bias/norm vectors: ev {buf, off(floats)}, stats {buf, off(rows)}, pv {buf, off(floats), stride(floats)}
 *   heads: {Ltot, Hq, Hkv, rowOff, eps, rope: {buf, off}}; conv: {Hi, Wi, Ci, Ho, Wo, stride, padT, padL, ks, up2, gn}
 *   tile: {BM, BN}, acc }
 */
export function gemm(gpu, a) {
  const enc = a.w.enc;
  const w = enc === 'q4g32a.t64' ? 'q4' : enc === 'q8g64s.t64' ? 'q8' : enc === 'f16' ? 'f16' : null;
  if (!w) throw new Error('gemm: unsupported weight encoding ' + enc);
  const cv = a.conv || null;
  // tile: the call's, else the tune for this convolution kind ('conv' / 'conv3'), else for this weight encoding, else the default
  const tile = a.tile || (cv && gpu.tune?.gemm?.[cv.kind === 'conv3' ? 'conv3' : 'conv']) || gpu.tune?.gemm?.[w] || DEFAULT_TILE;
  const cfg = { w, x: cv ? (cv.kind === 'conv3' ? 'conv3' : 'conv') : (a.x.mode || 'f16'), epi: a.epi || 'f16', BM: tile.BM, BN: (a.epi === 'heads' || a.epi === 'onormgate') ? 128 : tile.BN,
    acc: a.acc || (gpu.tune && gpu.tune.acc) || 'f32', rope: a.heads ? a.heads.ropeMode : undefined, sg: (a.epi === 'heads' || a.epi === 'onormgate') && gpu.sgOK16, pn: !!(cv && cv.pn),
    gate: !!a.ev && a.epi === 'resadd', res: !!a.res, gn: !!(cv && cv.gn), up2: !!(cv && cv.up2), bias: !!a.bv, store: !!a.store, act: a.act || 'silu' };
  const key = keyOf(cfg);
  let meta = gpu._gemmMeta || (gpu._gemmMeta = new Map());
  const pipe = gpu.pipeline(key, () => { const g = gemmCode(cfg); meta.set(key, g.bindings); return g.code; });
  const bnames = meta.get(key);
  const M = a.M, N = a.N, K = a.K;
  if (K % 32) throw new Error('gemm: K must be a multiple of 32');
  const Ktot = a.w.cols, wG = w === 'q4' ? Ktot / 32 : w === 'q8' ? Ktot / 64 : (a.w.ld || Ktot);
  const BMt = cfg.BM, BNt = cfg.BN;
  const ntiles = Math.ceil(N / BNt), mtiles = Math.ceil(M / BMt);
  const flopsTile = 2 * BMt * BNt * K;
  const rmm = (cfg.acc === 'f16' ? gpu.rate.mm16 : gpu.rate.mm32) || gpu.rate.mm;   // each accumulate mode has its own measured rate
  const cap = rmm * gpu.opts.maxDispatchSeconds;
  // split: whole N per dispatch if possible, rows chunked; else split N as well
  let nPer = ntiles, mPer = Math.max(1, Math.floor(cap / (flopsTile * ntiles)));
  if (mPer < 1 || flopsTile * ntiles > cap) { mPer = 1; nPer = Math.max(1, Math.floor(cap / flopsTile)); }
  const h = a.heads || {}, c = cv || {};
  // v026.9: a vector inside a buffer larger than one binding (a bias in a 139 MB shard) is bound from the 256-byte boundary before it
  const lim = (gpu.limits && gpu.limits.maxStorageBufferBindingSize) || Infinity;
  const win = (t) => {
    if (!t) return [null, 0];
    const sz = t.buf._pf || t.buf.size, off = t.off || 0;
    if (sz <= lim) return [{ buffer: t.buf }, off];
    const byte = off * 4, base = Math.floor(byte / 256) * 256, n = Math.min(lim, sz - base);
    return [{ buffer: t.buf, offset: base, size: n - (n % 16) }, (byte - base) / 4];
  };
  const [vST, oST] = win(a.stats), [vPV, oPV] = win(a.pv), [vEV, oEV] = win(a.ev), [vBV, oBV] = win(a.bv);
  const views = {
    X: { buffer: a.x.buf, offset: a.x.off || 0, size: (a.x.size != null ? a.x.size : a.x.buf._pf - (a.x.off || 0)) },
    W: { buffer: a.w.buf, offset: a.w.off || 0, size: a.w.bytes },
    Y: { buffer: a.y.buf, offset: a.y.off || 0, size: (a.y.size != null ? a.y.size : a.y.buf._pf - (a.y.off || 0)) },
    ST: vST, PV: vPV, EV: vEV,
    RP: h.rope && { buffer: h.rope.buf }, GT: a.gt && { buffer: a.gt.buf }, BV: vBV,
  };
  for (const k in views) if (views[k] && views[k].size == null) views[k].size = views[k].buffer._pf;
  for (const k in views) if (views[k]) views[k].size = Math.floor(views[k].size / 16) * 16 || 16;
  const binds = bnames.map((n) => { if (!views[n]) throw new Error('gemm: missing binding ' + n + ' for ' + key); return views[n]; });
  for (let m0 = 0; m0 < mtiles; m0 += mPer) {
    for (let n0 = 0; n0 < ntiles; n0 += nPer) {
      const mt = Math.min(mPer, mtiles - m0), nt = Math.min(nPer, ntiles - n0);
      const pr = P().u32(M).u32(N).u32(K).u32(a.x.ld || K).u32(a.x.k0 || 0).u32(a.wr0 || 0).u32(a.wr1 || 0).u32(wG).u32(a.wk0 || 0)
        .u32(a.y.ld || N).u32(a.y.n0 || 0).u32(m0).u32(n0)
        .u32(h.rowOff || 0).u32(h.Ltot || 0).u32(h.Hq || 0).u32(h.Hkv || 0).f32(h.eps || 1e-6)
        .u32(a.pv ? oPV : 0).u32(a.pv ? a.pv.stride || 0 : 0).u32(a.ev ? oEV : 0).u32(a.stats ? oST : 0)
        .u32(c.Hi || 0).u32(c.Wi || 0).u32(c.Ci || 0).u32(c.Ho || 0).u32(c.Wo || 0).u32(c.stride || 1).u32(c.padT || 0).u32(c.padL || 0).u32(c.ks || 1)
        .u32(a.w.rows).u32(a.bv ? oBV : 0).u32(a.bv ? a.bv.up || 0 : 0);
      gpu.runner.dispatch(pipe, binds, pr.ab, [nt, mt], (flopsTile * mt * nt) / rmm);
    }
  }
  return 2 * M * N * K;
}
