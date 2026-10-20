#!/usr/bin/env node
// SPDX-License-Identifier: MPL-2.0
// Copyright 2026 Joshua Hughes
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// Convert safetensors model files into the kernels' weight encodings: shards of at most --shard-mb plus manifest.json (docs/FORMAT.md).
// Node 22+, no dependencies; tensors stream in blocks, so models larger than memory convert.
//
//   node tools/convert.mjs <model.safetensors | model folder | files...> --out <folder> [--bits 4|8|16] [--keep <regex>] [--f32 <regex>]
//                          [--shard-mb 512] [--dry-run]
//   --bits 4   matrices that fit 64-row tiles -> q4g32a.t64 (8 -> q8g64s.t64, 16 -> fp16); everything else -> fp16; vectors -> fp32
//   --keep     tensors whose names match stay fp16 (default "embed": embeddings are looked up by row, not multiplied)
//   --f32      tensors whose names match are stored fp32
import fs from 'node:fs';
import path from 'node:path';
import { convert, encodedBytes } from '../src/convert/convert.js';
import { safetensorsHeader, safetensorsList, chooseEncoding, matrixOf } from '../src/convert/encode.js';

const argv = process.argv.slice(2), flag = (k, d) => { const i = argv.indexOf(k); if (i < 0) return d; const v = argv[i + 1]; argv.splice(i, 2); return v; };
const OUT = flag('--out', null), BITS = +flag('--bits', 4), KEEP = flag('--keep', 'embed'), F32 = flag('--f32', null), SHARD = +flag('--shard-mb', 512);
const DRY = argv.includes('--dry-run'); if (DRY) argv.splice(argv.indexOf('--dry-run'), 1);
if (!argv.length || (!OUT && !DRY) || ![4, 8, 16].includes(BITS)) {
  console.error('usage: node tools/convert.mjs <model.safetensors | folder | files...> --out <folder> [--bits 4|8|16] [--keep <regex>] [--f32 <regex>] [--shard-mb 512] [--dry-run]');
  process.exit(2);
}
const opts = { bits: BITS, keep: KEEP ? new RegExp(KEEP) : null, f32: F32 ? new RegExp(F32) : null, shardBytes: SHARD * 2 ** 20 };
const files = argv.flatMap((p) => fs.statSync(p).isDirectory() ? fs.readdirSync(p).filter((f) => f.endsWith('.safetensors')).sort().map((f) => path.join(p, f)) : [p]);
if (!files.length) { console.error('no .safetensors files found'); process.exit(2); }

function readFully(fd, len, pos) {
  const b = Buffer.allocUnsafe(len); let k = 0;
  while (k < len) { const n = fs.readSync(fd, b, k, len - k, pos + k); if (!n) throw new Error('unexpected end of file'); k += n; }
  return new Uint8Array(b.buffer, b.byteOffset, len);
}
const sources = files.map((f) => {
  const fd = fs.openSync(f, 'r'), n = Number(Buffer.from(readFully(fd, 8, 0)).readBigUInt64LE(0));
  const { header, dataStart } = safetensorsHeader(readFully(fd, 8 + n, 0));
  return { file: path.basename(f), header, dataStart, fd, read: (off, len) => readFully(fd, len, off) };
});

const MB = (b) => (b / 1e6).toFixed(1) + ' MB';
if (DRY) {
  const sum = {};
  for (const s of sources) for (const t of safetensorsList(s.header)) {
    const enc = chooseEncoding(t.name, t.shape, opts), m = matrixOf(t.shape), e = sum[enc] || (sum[enc] = { n: 0, bytes: 0 });
    e.n++; e.bytes += encodedBytes(enc, m.rows, m.cols);
  }
  for (const [enc, e] of Object.entries(sum)) console.log(`${enc.padEnd(12)} ${String(e.n).padStart(5)} tensors  ${MB(e.bytes)}`);
  process.exit(0);
}

fs.mkdirSync(OUT, { recursive: true });
let fd = null, n = 0;
const out = { begin: (name) => { fd = fs.openSync(path.join(OUT, name), 'w'); },
  write: (b) => { let k = 0; while (k < b.byteLength) k += fs.writeSync(fd, b, k, b.byteLength - k); }, end: () => { fs.closeSync(fd); fd = null; } };
const t0 = Date.now();
const manifest = await convert(sources, out, Object.assign({}, opts, { onTensor: (name, e) => { if (++n % 50 === 0) console.log(`${n} tensors, ${((Date.now() - t0) / 1000).toFixed(0)} s`); } }));
for (const s of sources) fs.closeSync(s.fd);
manifest.created = new Date().toISOString();
fs.writeFileSync(path.join(OUT, 'manifest.json'), JSON.stringify(manifest, null, 1));
const total = manifest.shards.reduce((a, s) => a + s.bytes, 0), by = {};
for (const e of Object.values(manifest.tensors)) by[e.enc] = (by[e.enc] || 0) + 1;
console.log(`converted ${Object.keys(manifest.tensors).length} tensors in ${((Date.now() - t0) / 1000).toFixed(0)} s: ${MB(total)} in ${manifest.shards.length} shard(s); ` +
  Object.entries(by).map(([k, v]) => `${v} ${k}`).join(', ') + (Object.keys(manifest.skipped).length ? `; skipped ${Object.keys(manifest.skipped).length} non-float tensors` : ''));
