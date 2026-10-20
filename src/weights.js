// SPDX-License-Identifier: MPL-2.0
// Copyright 2026 Joshua Hughes
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// Load a converted model (tools/convert.mjs output: manifest.json + weight shards) onto the GPU. Each shard becomes one GPU buffer, and
// each tensor a view the ops take directly: gemm(gpu, { w: model.tensor('blocks.0.attn.qkv.weight'), ... }).
import { MANIFEST_FORMAT } from './convert/encode.js';

/** base: the folder URL holding manifest.json (ending in '/'); opts: {fetch, onProgress(doneBytes, totalBytes)} */
export async function loadWeights(gpu, base, opts = {}) {
  const get = opts.fetch || ((u) => fetch(u)), url = (f) => new URL(f, base).href;
  const res = await get(url('manifest.json')); if (!res.ok) throw new Error('manifest.json: HTTP ' + res.status);
  const manifest = await res.json();
  if (manifest.format !== MANIFEST_FORMAT) throw new Error('not a converted model (format ' + manifest.format + ')');
  const total = manifest.shards.reduce((a, s) => a + s.bytes, 0), bufs = {}; let done = 0;
  try {
    for (const s of manifest.shards) {
      const r = await get(url(s.file)); if (!r.ok) throw new Error(s.file + ': HTTP ' + r.status);
      const data = new Uint8Array(await r.arrayBuffer());
      if (data.byteLength !== s.bytes) throw new Error(`${s.file}: ${data.byteLength} bytes, the manifest says ${s.bytes}`);
      bufs[s.file] = gpu.upload(data, s.file); done += s.bytes; if (opts.onProgress) opts.onProgress(done, total);
    }
  } catch (e) { for (const b of Object.values(bufs)) gpu.free(b); throw e; }
  return modelOf(manifest, bufs, gpu);
}

/** the same from shards already in memory ({file: Uint8Array}, e.g. convertSafetensors' output) */
export function uploadWeights(gpu, manifest, shards) {
  const bufs = {}; for (const s of manifest.shards) bufs[s.file] = gpu.upload(shards[s.file], s.file);
  return modelOf(manifest, bufs, gpu);
}

function modelOf(manifest, bufs, gpu) {
  return {
    manifest,
    names: () => Object.keys(manifest.tensors),
    /** a weight for gemm's `w`: {enc, buf, off, bytes, rows, cols} (+ shape, layout) */
    tensor(name) {
      const t = manifest.tensors[name]; if (!t) throw new Error('no tensor ' + name);
      return { enc: t.enc, buf: bufs[t.file], off: t.offset, bytes: t.bytes, rows: t.rows, cols: t.cols, shape: t.shape, layout: t.layout };
    },
    /** an fp32 vector (norm weight, bias) as the ops take it: {buf, off} with off counted in floats */
    vector(name) {
      const t = manifest.tensors[name]; if (!t) throw new Error('no tensor ' + name);
      if (t.enc !== 'f32') throw new Error(name + ' is ' + t.enc + ', not f32');
      return { buf: bufs[t.file], off: t.offset / 4 };
    },
    free() { for (const b of Object.values(bufs)) gpu.free(b); },
  };
}
