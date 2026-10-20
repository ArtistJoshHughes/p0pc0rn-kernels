// SPDX-License-Identifier: MPL-2.0
// Copyright 2026 Joshua Hughes
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// Converter core: safetensors sources -> weight shards in the kernels' encodings + a manifest (docs/FORMAT.md, "Converted models").
// It streams each tensor in blocks of rows, so a model never has to fit in memory. tools/convert.mjs is the command line (Node);
// convertSafetensors() converts one file held in memory (browsers too).
import { safetensorsHeader, safetensorsList, dtypeBytes, toFloat32, chooseEncoding, matrixOf, reorderConvRows, encodeRows, tileBytes, MANIFEST_FORMAT } from './encode.js';

export const ALIGN = 256;   // tensor offsets inside a shard: WebGPU's minStorageBufferOffsetAlignment
export const encodedBytes = (enc, rows, cols) => enc === 'q4g32a.t64' || enc === 'q8g64s.t64' ? (rows / 64) * tileBytes(enc, cols) : rows * cols * (enc === 'f16' ? 2 : 4);

/** sources: [{file, header, dataStart, read(offset, length) -> Uint8Array | Promise}]; out: {begin(shardName), write(bytes), end()}
 *  opts: {bits: 4 | 8 | 16, keep: RegExp, f32: RegExp, shardBytes = 512 MB, onTensor(entry)} -> manifest */
export async function convert(sources, out, opts = {}) {
  const shardBytes = opts.shardBytes || 512 * 2 ** 20, tensors = {}, shards = [], skipped = {};
  let cur = null;
  const open = async () => { if (cur) await close(); cur = { file: `weights-${String(shards.length + 1).padStart(5, '0')}.bin`, bytes: 0 }; await out.begin(cur.file); };
  const close = async () => { if (!cur) return; await out.end(); shards.push({ file: cur.file, bytes: cur.bytes }); cur = null; };
  const put = async (u8) => { await out.write(u8); cur.bytes += u8.byteLength; };
  for (const src of sources) {
    for (const t of safetensorsList(src.header)) {
      if (tensors[t.name]) throw new Error('tensor ' + t.name + ' appears twice');
      if (!SUPPORTED.has(t.dtype)) { skipped[t.name] = t.dtype; continue; }   // integer / boolean tensors (ids, masks) are not weights
      const m = matrixOf(t.shape), enc = chooseEncoding(t.name, t.shape, opts), size = encodedBytes(enc, m.rows, m.cols), eb = dtypeBytes(t.dtype);
      if ((t.end - t.begin) !== m.rows * m.cols * eb) throw new Error(`tensor ${t.name}: ${t.end - t.begin} bytes for shape [${t.shape}] ${t.dtype}`);
      if (!cur || (cur.bytes > 0 && Math.ceil(cur.bytes / ALIGN) * ALIGN + size > shardBytes)) await open();
      const pad = (ALIGN - (cur.bytes % ALIGN)) % ALIGN; if (pad) await put(new Uint8Array(pad));
      const entry = { enc, dtype: t.dtype, shape: t.shape, rows: m.rows, cols: m.cols, layout: m.layout, file: cur.file, offset: cur.bytes, bytes: size };
      const conv = m.layout === 'OHWI' || m.layout === 'OTHWI';
      const block = Math.max(64, Math.floor((4 * 2 ** 20) / Math.max(1, m.cols) / 64) * 64);   // ~4M values per block, whole 64-row tiles
      for (let r0 = 0; r0 < m.rows; r0 += block) {
        const nr = Math.min(block, m.rows - r0), raw = await src.read(src.dataStart + t.begin + r0 * m.cols * eb, nr * m.cols * eb);
        let W = toFloat32(t.dtype, raw);
        if (conv) W = reorderConvRows(W, nr, t.shape);
        await put(encodeRows(enc, W, nr, m.cols));
      }
      if (cur.bytes - entry.offset !== size) throw new Error(`tensor ${t.name}: wrote ${cur.bytes - entry.offset} of ${size} bytes`);
      tensors[t.name] = entry; if (opts.onTensor) opts.onTensor(t.name, entry);
    }
  }
  await close();
  return { format: MANIFEST_FORMAT, sources: sources.map((s) => s.file), bits: opts.bits || 4, tensors, shards, skipped };
}
const SUPPORTED = new Set(['F32', 'F16', 'BF16', 'F8_E4M3', 'F8_E5M2']);

/** one safetensors file in memory (ArrayBuffer / Uint8Array) -> {manifest, shards: {name: Uint8Array}} */
export async function convertSafetensors(buffer, opts = {}, file = 'model.safetensors') {
  const u8 = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer), { header, dataStart } = safetensorsHeader(u8);
  const files = {}; let parts = null, name = null;
  const out = { begin: (n) => { name = n; parts = []; }, write: (b) => { parts.push(b); }, end: () => { const n = parts.reduce((a, p) => a + p.byteLength, 0), o = new Uint8Array(n); let k = 0; for (const p of parts) { o.set(p, k); k += p.byteLength; } files[name] = o; } };
  const manifest = await convert([{ file, header, dataStart, read: (off, len) => u8.subarray(off, off + len) }], out, opts);
  return { manifest, shards: files };
}

/** build a safetensors file in memory from {name: {dtype, shape, data: Uint8Array}} (tests, small tools) */
export function writeSafetensors(tensors, metadata = null) {
  const header = {}; let off = 0;
  if (metadata) header.__metadata__ = metadata;
  for (const [name, t] of Object.entries(tensors)) { header[name] = { dtype: t.dtype, shape: t.shape, data_offsets: [off, off + t.data.byteLength] }; off += t.data.byteLength; }
  const json = JSON.stringify(header), n0 = new TextEncoder().encode(json).byteLength;
  const hb = new TextEncoder().encode(json + ' '.repeat((8 - ((8 + n0) % 8)) % 8)), out = new Uint8Array(8 + hb.byteLength + off), dv = new DataView(out.buffer);
  dv.setBigUint64(0, BigInt(hb.byteLength), true); out.set(hb, 8);
  let k = 8 + hb.byteLength; for (const t of Object.values(tensors)) { out.set(t.data, k); k += t.data.byteLength; }
  return out;
}
