# Changelog

All notable changes to this project are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-20

First public release: the kernels the p0pc0rn app runs, under the Mozilla Public License 2.0.

### Added

- **Runtime:** WebGPU device set-up with `shader-f16`, a buffer pool, dispatch batching, speed calibration (`calibrate`) and default kernel choices (`defaultTune`).
- **GEMM family:** in-kernel dequantisation of 4-bit (`q4g32a.t64`), 8-bit (`q8g64s.t64`) and fp16 weights; fused LayerNorm / RMSNorm / convolution prologues; fused residual, SwiGLU/GeGLU, QK-norm + RoPE and bias epilogues; fp32 or fp16-partial accumulation.
- **3×3×3 convolution** with hoisted voxel indices (bit-identical, 9–15% faster on iPhone), plus time offsets and frame rings for decoding a clip a few frames at a time.
- **Flash attention:** heads of 128 and 256, all mask modes, grouped-query attention and the soft cap, with variants for 16 KB of workgroup memory.
- **`attnF`:** register-tiled flash attention tuned on the iPhone (×1.6–1.9 for head 128, ×3.8–3.9 for head 256 there).
- **Small kernels** for normalisation, softmax, elementwise ops, layout, linear-attention state and the video decoder.
- **Converter:** safetensors (F32, F16, BF16, FP8) to the kernels' encodings, streaming, with convolution weights reordered; `loadWeights()` puts the result on the GPU.
- **Benchmark and tuning page:** per-device settings sweep with correctness checks, results shareable to the repository's Discussions, and a per-GPU tuning table (`TUNE_TABLE`).
- **TypeScript declarations** for the whole public API.
- **Tooling:** a test suite against CPU references with a headless runner, the converter's tests, and a GPU-free check for CI.
- **Docs:** design notes, the weight encodings, measured results.
