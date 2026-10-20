// SPDX-License-Identifier: MPL-2.0
// Copyright 2026 Joshua Hughes
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// Compile-only check of the declarations (npm run check:types): the README's quick start and the main calls, typed.
import { createGpu, defaultTune, calibrate, gemm, attention, attnF, ATTNF_P, toF16, fromF16, convertSafetensors, writeSafetensors, uploadWeights, loadWeights } from './index';
import type { Gpu, GemmArgs, AttentionArgs, Model } from './index';

export async function quickStart(): Promise<Float32Array> {
  const gpu: Gpu = await createGpu();
  defaultTune(gpu);
  await calibrate(gpu);
  const M = 64, N = 256, K = 512;
  const x = gpu.upload(toF16(new Float32Array(M * K).fill(0.5)));
  const w = gpu.upload(toF16(new Float32Array(N * K).fill(0.01)));
  const y = gpu.buffer(M * N * 2);
  const args: GemmArgs = { w: { enc: 'f16', buf: w, off: 0, bytes: N * K * 2, rows: N, cols: K }, x: { buf: x }, y: { buf: y }, M, N, K };
  const flops: number = gemm(gpu, args);
  void flops;
  return fromF16(new Uint16Array(await gpu.read(y, 0, M * N * 2)));
}

export function attend(gpu: Gpu, q: GPUBuffer, k: GPUBuffer, v: GPUBuffer, o: GPUBuffer): void {
  const a: AttentionArgs = { q: { buf: q }, k: { buf: k }, v: { buf: v }, o: { buf: o, ld: 8 * 128 }, Lq: 4096, Hq: 8, mode: 1, nvalid: 4000 };
  attention(gpu, a);
  attnF(gpu, a, Object.assign({ D: 128 as const, flavour: 'p' as const }, ATTNF_P));
}

export async function converted(gpu: Gpu): Promise<Model> {
  const file = writeSafetensors({ 'layer.weight': { dtype: 'F32', shape: [64, 64], data: new Uint8Array(64 * 64 * 4) } });
  const { manifest, shards } = await convertSafetensors(file, { bits: 4, keep: /embed/ });
  const model = uploadWeights(gpu, manifest, shards);
  const w = model.tensor('layer.weight');
  gemm(gpu, { w, x: { buf: gpu.buffer(64 * 64 * 2) }, y: { buf: gpu.buffer(64 * 64 * 2) }, M: 64, N: 64, K: 64 });
  return loadWeights(gpu, 'https://example.com/model/', { onProgress: (d: number, t: number) => void (d / t) });
}
