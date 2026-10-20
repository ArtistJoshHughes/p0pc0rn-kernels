// SPDX-License-Identifier: MPL-2.0
// Copyright 2026 Joshua Hughes
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// Weight conversion on the CPU (browsers and Node): reading safetensors files (F32, F16, BF16, F8_E4M3, F8_E5M2), the encodings of
// docs/FORMAT.md (q4g32a.t64, q8g64s.t64, f16, f32), convolution weights reordered for the implicit-GEMM kernels, and the encoding
// rule the converter applies to each tensor. The quantisers follow FORMAT.md exactly: fp32 maths, scales and minimums rounded to fp16,
// codes rounded half to even.
import { f2h, h2f } from '../core/gpu.js';

const F = Math.fround;
const rf16 = (x) => h2f(f2h(x));
const rhe = (x) => { const f = Math.floor(x), d = x - f; return d > 0.5 ? f + 1 : d < 0.5 ? f : (f % 2 === 0 ? f : f + 1); };   // round half to even

// ------------------------------------------------------------------ safetensors
/** parse the header of a safetensors file from its first bytes (at least 8 + header length) -> {header, dataStart} */
export function safetensorsHeader(bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes), dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const n = Number(dv.getBigUint64(0, true));
  if (n > 100e6 || 8 + n > u8.byteLength) throw new Error('safetensors: bad or truncated header');
  const header = JSON.parse(new TextDecoder().decode(u8.subarray(8, 8 + n)));
  return { header, dataStart: 8 + n };
}
/** the tensors of a parsed header, in file order: [{name, dtype, shape, begin, end}] (begin/end relative to the data start) */
export function safetensorsList(header) {
  return Object.entries(header).filter(([k]) => k !== '__metadata__')
    .map(([name, t]) => ({ name, dtype: t.dtype, shape: t.shape, begin: t.data_offsets[0], end: t.data_offsets[1] }))
    .sort((a, b) => a.begin - b.begin);
}
const BYTES = { F32: 4, F16: 2, BF16: 2, F8_E4M3: 1, F8_E5M2: 1 };
export const dtypeBytes = (dt) => { if (!BYTES[dt]) throw new Error('unsupported dtype ' + dt + ' (F32, F16, BF16, F8_E4M3, F8_E5M2)'); return BYTES[dt]; };
function f8Table(e4m3) {
  const t = new Float32Array(256);
  for (let i = 0; i < 256; i++) {
    const s = i & 0x80 ? -1 : 1, e = e4m3 ? (i >> 3) & 15 : (i >> 2) & 31, m = e4m3 ? i & 7 : i & 3, mb = e4m3 ? 3 : 2, bias = e4m3 ? 7 : 15;
    if (e4m3 && e === 15 && m === 7) t[i] = NaN;
    else if (!e4m3 && e === 31) t[i] = m ? NaN : s * Infinity;
    else t[i] = s * (e === 0 ? (m / (1 << mb)) * 2 ** (1 - bias) : (1 + m / (1 << mb)) * 2 ** (e - bias));
  }
  return t;
}
let E4M3 = null, E5M2 = null;
/** raw tensor bytes (any alignment) of a dtype -> Float32Array */
export function toFloat32(dtype, bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes), n = u8.byteLength / dtypeBytes(dtype), out = new Float32Array(n);
  if (dtype === 'F32') { new Uint8Array(out.buffer).set(u8); return out; }
  if (dtype === 'F8_E4M3' || dtype === 'F8_E5M2') { const t = dtype === 'F8_E4M3' ? (E4M3 = E4M3 || f8Table(true)) : (E5M2 = E5M2 || f8Table(false)); for (let i = 0; i < n; i++) out[i] = t[u8[i]]; return out; }
  const h = new Uint16Array(n); new Uint8Array(h.buffer).set(u8);
  if (dtype === 'BF16') { const u = new Uint32Array(out.buffer); for (let i = 0; i < n; i++) u[i] = h[i] << 16; return out; }
  for (let i = 0; i < n; i++) out[i] = h2f(h[i]);
  return out;
}

// ------------------------------------------------------------------ encodings (rows: a multiple of 64 for q4 / q8)
/** q4g32a.t64 records for `rows` rows of K values (row-major fp32) -> Uint8Array; deq (optional) receives the dequantised weights */
export function packQ4(W, rows, K, deq = null) {
  if (rows % 64 || K % 32) throw new Error('q4g32a.t64 needs rows % 64 == 0 and K % 32 == 0');
  const NT = rows / 64, G = K / 32, out = new Uint8Array(NT * G * 1280), dv = new DataView(out.buffer);
  for (let t = 0; t < NT; t++) for (let g = 0; g < G; g++) {
    const rec = (t * G + g) * 1280;
    for (let r = 0; r < 64; r++) {
      const n = t * 64 + r, base = n * K + g * 32; let mnv = Infinity, mxv = -Infinity;
      for (let i = 0; i < 32; i++) { const v = W[base + i]; if (v < mnv) mnv = v; if (v > mxv) mxv = v; }
      const mn = rf16(mnv); let s = rf16(F(F(mxv - mn) / 15)); if (s === 0) s = 1;
      for (let j = 0; j < 16; j++) {
        const q0 = Math.min(15, Math.max(0, rhe(F(F(W[base + 2 * j] - mn) / s)))), q1 = Math.min(15, Math.max(0, rhe(F(F(W[base + 2 * j + 1] - mn) / s))));
        out[rec + r * 16 + j] = q0 | (q1 << 4);
        if (deq) { deq[base + 2 * j] = F(q0 * s + mn); deq[base + 2 * j + 1] = F(q1 * s + mn); }
      }
      dv.setUint32(rec + 1024 + r * 4, (f2h(s) | (f2h(mn) << 16)) >>> 0, true);
    }
  }
  return out;
}
/** q8g64s.t64 records for `rows` rows of K values -> Uint8Array */
export function packQ8(W, rows, K, deq = null) {
  if (rows % 64 || K % 64) throw new Error('q8g64s.t64 needs rows % 64 == 0 and K % 64 == 0');
  const NT = rows / 64, G = K / 64, out = new Uint8Array(NT * G * 4224), dv = new DataView(out.buffer);
  for (let t = 0; t < NT; t++) for (let g = 0; g < G; g++) {
    const rec = (t * G + g) * 4224;
    for (let r = 0; r < 64; r++) {
      const n = t * 64 + r, base = n * K + g * 64; let am = 0;
      for (let i = 0; i < 64; i++) am = Math.max(am, Math.abs(W[base + i]));
      let s = rf16(F(am / 127)); if (s === 0) s = 1;
      for (let i = 0; i < 64; i++) { const q = Math.min(127, Math.max(-128, rhe(F(W[base + i] / s)))); out[rec + r * 64 + i] = q & 255; if (deq) deq[base + i] = F(q * s); }
      dv.setUint16(rec + 4096 + r * 2, f2h(s), true);
    }
  }
  return out;
}
/** fp16, round to nearest even -> Uint8Array */
export function packF16(W) {
  if (typeof Float16Array !== 'undefined') return new Uint8Array(Float16Array.from(W).buffer);
  const h = new Uint16Array(W.length); for (let i = 0; i < W.length; i++) h[i] = f2h(W[i]); return new Uint8Array(h.buffer);
}
export const packF32 = (W) => new Uint8Array(Float32Array.from(W).buffer);
/** bytes per 64-row tile of K columns, per encoding */
export const tileBytes = (enc, K) => enc === 'q4g32a.t64' ? (K / 32) * 1280 : enc === 'q8g64s.t64' ? (K / 64) * 4224 : enc === 'f16' ? 64 * K * 2 : 64 * K * 4;
/** encode rows x K fp32 values (rows a multiple of 64 for q4 / q8) */
export function encodeRows(enc, W, rows, K) {
  if (enc === 'q4g32a.t64') return packQ4(W, rows, K);
  if (enc === 'q8g64s.t64') return packQ8(W, rows, K);
  if (enc === 'f16') return packF16(W.subarray(0, rows * K));
  if (enc === 'f32') return packF32(W.subarray(0, rows * K));
  throw new Error('unknown encoding ' + enc);
}

// ------------------------------------------------------------------ layout and the encoding rule
/** the matrix view of a tensor: 2-D as is; convolution weights [O, I, kh, kw] -> OHWI and [O, I, kt, kh, kw] -> OTHWI, so each output
 *  channel's row is the implicit GEMM's K vector (FORMAT.md); others flatten their leading dimensions into rows */
export function matrixOf(shape) {
  if (shape.length === 4) return { rows: shape[0], cols: shape[1] * shape[2] * shape[3], layout: 'OHWI' };
  if (shape.length === 5) return { rows: shape[0], cols: shape[1] * shape[2] * shape[3] * shape[4], layout: 'OTHWI' };
  if (shape.length <= 1) return { rows: 1, cols: shape.length ? shape[0] : 1, layout: 'vector' };
  return { rows: shape.slice(0, -1).reduce((a, b) => a * b, 1), cols: shape[shape.length - 1], layout: 'row-major' };
}
/** reorder rows of a convolution weight in place-free fashion: OIHW / OITHW rows (I * taps) -> (taps * I) */
export function reorderConvRows(W, rows, shape) {
  const I = shape[1], taps = shape.slice(2).reduce((a, b) => a * b, 1), K = I * taps, out = new Float32Array(rows * K);
  for (let r = 0; r < rows; r++) for (let i = 0; i < I; i++) for (let t = 0; t < taps; t++) out[r * K + t * I + i] = W[r * K + i * taps + t];
  return out;
}
/** which encoding a tensor gets: vectors -> f32; names matching keep -> f16; matrices that fit the quantised tiles -> q4 / q8 (bits);
 *  everything else -> f16. opts: {bits: 4 | 8 | 16, keep: RegExp, f32: RegExp} */
export function chooseEncoding(name, shape, opts = {}) {
  const bits = opts.bits || 4, m = matrixOf(shape);
  if (opts.f32 && opts.f32.test(name)) return 'f32';
  if (m.layout === 'vector') return 'f32';
  if ((opts.keep && opts.keep.test(name)) || bits === 16) return 'f16';
  if (bits === 4 && m.rows % 64 === 0 && m.cols % 32 === 0) return 'q4g32a.t64';
  if ((bits === 4 || bits === 8) && m.rows % 64 === 0 && m.cols % 64 === 0) return 'q8g64s.t64';
  return 'f16';
}
export const MANIFEST_FORMAT = 'p0pc0rn-kernels.weights/1';
