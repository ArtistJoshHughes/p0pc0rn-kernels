# Measured results

Measured in October 2026 on two devices:

- **Phone:** iPhone 16 Pro Max (Apple A18 Pro GPU), Safari 26.6.1.
- **Desktop:** AMD Radeon integrated graphics, Windows, Microsoft Edge.

Every comparison ran interleaved: the old and new kernel alternate on the same inputs, so heat affects both alike. Every new kernel was checked against the kernel it replaced before it was timed.

## The phone's GPU

| Probe | Result |
|---|---|
| Fused multiply-add throughput, fp32 or fp16 | 2.2 TFLOP/s |
| fp16 and fp32 FMAs mixed (dual issue) | 3.0 TFLOP/s |
| Workgroup-memory loads, `vec4<f16>` | ~692 GB/s |
| Device-memory copy | ~46 GB/s |
| GEMM inner loop alone, 8×8 tiles, fp16 partial sums | 1.61 TFLOP/s |
| GEMM inner loop alone, 8×8 tiles, fp32 sums | 1.24 TFLOP/s |
| Features | `shader-f16` and `timestamp-query`; **no `subgroups`**; 32 KB of workgroup memory |

The GEMM in this repository runs at about 1.17 TFLOP/s on a cool phone, about 72% of its inner loop's 1.61 TFLOP/s.

**Heat matters more than any kernel.** A FLUX.2 [klein] 4B denoising step at 1280×720 took 26.7 s on a cool phone and 42–52 s on a hot one. Over a long run, a fixed GEMM "canary" slowed down by a factor of 1.64.

## Kernels on the phone (new vs the kernel it replaced)

| Kernel | Old | New | Speed-up | Agreement with the old kernel |
|---|---|---|---|---|
| Attention, head 128 (pictures, `attnF` 64×64) | 464 GFLOP/s | 749 GFLOP/s | ×1.61–1.85 | relative error 1.4·10⁻³ |
| Attention, head 256 (video anchors, `attnF` 32×128) | 145 GFLOP/s | 564 GFLOP/s | ×3.77–3.89 | relative error 1.2·10⁻³ |
| 3×3×3 convolution, hoisted voxel indices | | | ×1.09–1.15 | bit-identical |
| Picture decoder, fp16 partial sums | 18.7 / 16.9 s | 16.0 / 15.6 s per decode | −11% time | 62–64 dB PSNR (desktop) |

## Kernels on the desktop's AMD iGPU

Measured with this repository's benchmark (`bench/index.html`, medians of 3 interleaved rounds) on AMD Radeon integrated graphics (RDNA 2) in Microsoft Edge:

| Kernel | Standard kernel | Phone-tuned kernel | Speed-up |
|---|---|---|---|
| Attention, head 128 (4112 tokens × 8 heads) | 171 GFLOP/s | 290 GFLOP/s | ×1.70 |
| Attention, head 256 (4096 tokens × 10 heads) | 55 GFLOP/s | 223 GFLOP/s | ×4.05 |

| Kernel | Rate |
|---|---|
| GEMM 4112 × 3072 × 3072, 4-bit weights | 482 GFLOP/s |
| GEMM 4112 × 9216 × 3072, 4-bit weights | 486 GFLOP/s |
| GEMM 4112 × 3072 × 3072, fp16 weights | 175 GFLOP/s |
| 3×3×3 convolution, 512 → 512 channels, 4 × 32 × 32 | 197 GFLOP/s |

The 4-bit GEMM runs 2.8× faster than the fp16 one here, most likely because this integrated GPU is limited by memory bandwidth: 4-bit weights move under a third of the bytes (5 bits per weight including scales, against 16).

No GEMM candidate beat the existing GEMM on either device.

### First tuning result: the 3D convolution's tile shape (AMD)

The benchmark's **Tune for this device** sweep tried five tile shapes for the 3×3×3 convolution at the LTX-2.3 decoder's three channel widths. Every tile gives **bit-identical** output: each output value is summed in the same order whatever the tile. So this speed-up changes nothing in the result.

| Channels, size | Default 64×128 | Fastest: 128×64 | Speed-up |
|---|---|---|---|
| 512 → 512, 4 × 32 × 32 | 194 GFLOP/s | 374 GFLOP/s | ×1.93 |
| 256 → 256, 4 × 64 × 64 | 194 GFLOP/s | 374 GFLOP/s | ×1.93 |
| 128 → 128, 4 × 128 × 128 | 337 GFLOP/s | 389 GFLOP/s | ×1.16 |

This is the first entry in `TUNE_TABLE`: RDNA 2 GPUs now get 128×64 convolution tiles. The GEMM and attention defaults were already the fastest settings on this GPU. The iPhone has not been swept with these shapes yet.

## In the p0pc0rn app (iPhone 16 Pro Max)

These runs compare the app's released version with a build that uses these kernels. They used the same prompts and seeds, alternating in A-B-B-A order with cool-down rests. The clip numbers also include the app's time-chunked clip decoder, which uses the 3D convolution's time offsets to decode a clip in fewer, larger tiles.

| Render | Released app | With these kernels | Time | Speed |
|---|---|---|---|---|
| Picture 1280×720, 4 steps (2 runs each) | 174.7 s | 156.2 s | −10.6% | +11.8% |
| Clip 832×480, 49 frames (2 runs each) | 213.6 s | 174.7 s | −18.2% | +22.3% |
| Clip 1280×736, 49 frames (1 run each) | 689.4 s | 473.7 s | −31.3% | +45.5% |

A render gains about half of what a kernel gains on its own. The phone is limited by power and heat, so it throttles back part of a faster kernel's saving. For example, attention ran ×1.8 faster in isolation, which should have saved about 6.4 s per picture step; the render saved about 3.4 s.

## What did not work on the phone

Each of these was measured interleaved against the existing kernel:

- **Padding workgroup-memory rows** to avoid bank conflicts: 11% slower on Apple (7% faster on AMD).
- **64-wide k-steps with register prefetch:** slower.
- **8×16 register tiles with fp32 accumulators:** registers spill.
- **Pure fp16 accumulation in the GEMM:** ×1.13–1.15 faster, but a relative error of 0.8–1.4%, too large to use.
- **A second GEMM design with register staging arrays:** 13–15% slower than the existing GEMM on Apple.
- **Splitting a GEMM or attention across the fp16 and fp32 pipes:** slower, despite the 3.0 TFLOP/s the probes found for mixed FMAs.
