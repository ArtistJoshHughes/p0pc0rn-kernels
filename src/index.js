// SPDX-License-Identifier: MPL-2.0
// Copyright 2026 Joshua Hughes
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// p0pc0rn kernels: WebGPU compute kernels (WGSL generated from JavaScript) for running diffusion and language models in a browser,
// tuned on phone GPUs. Every op takes the context from createGpu() first: op(gpu, args). See README.md and docs/.
export { MB, GpuError, createGpu, Gpu, Runner, P, view, f2h, h2f, toF16, fromF16 } from './core/gpu.js';
export { calibrate, defaultTune, tuneEntry, TUNE_TABLE } from './core/tune.js';
export { gemm, gemmCode, DEFAULT_TILE } from './ops/gemm.js';
export { attention, attnCode, attnBytes, attnF, attnFCode, attnFCfg, attnFFits, ATTNF_P, ATTNF_V } from './ops/attention.js';
export { rowstats, gnPartBytes, gnPartial, gnFinalize, groupnormStats, ew, softmax, transpose16, convDirect, layout } from './ops/misc.js';
export { laprep, kvstate, attn256, attnres, pixstats, d2s } from './ops/video.js';
export { rmshead, normadd, vop, ropeWan, ropeGemma, rmsrow32 } from './ops/video2.js';
export { loadWeights, uploadWeights } from './weights.js';
export { convert, convertSafetensors, writeSafetensors, encodedBytes } from './convert/convert.js';
export { safetensorsHeader, safetensorsList, toFloat32, packQ4, packQ8, packF16, packF32, encodeRows, matrixOf, reorderConvRows, chooseEncoding, MANIFEST_FORMAT } from './convert/encode.js';
