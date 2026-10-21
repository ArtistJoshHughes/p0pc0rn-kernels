# p0pc0rn kernels

WebGPU compute kernels for running image, video and language models in a web browser, tuned on an iPhone. They are the kernels inside [p0pc0rn](https://artistjoshhughes.github.io/p0pc0rn/), an app that makes pictures and clips entirely on the phone.

- Plain JavaScript ES modules that generate WGSL. No build step, no dependencies.
- Weights stay quantised in GPU memory (4-bit and 8-bit) and are dequantised inside the kernels.
- Every kernel is checked against a CPU reference by the test suite, and the benchmark runs each new kernel next to the kernel it replaced.

Run the benchmark and the tests on your own device: [artistjoshhughes.github.io/p0pc0rn-kernels](https://artistjoshhughes.github.io/p0pc0rn-kernels/).

## What's inside

| Kernel | What it does |
|---|---|
| `gemm` | Matrix multiply with in-kernel dequantisation of 4-bit, 8-bit or fp16 weights. Fused prologues: LayerNorm + adaLN modulation, RMSNorm, implicit-GEMM 2D convolution, 3×3×3 convolution. Fused epilogues: residual add, SwiGLU/GeGLU, QK-norm + RoPE head split, bias. |
| `conv3` (a `gemm` mode) | 3×3×3 convolution over a clip. It can run over a few frames at a time, from a buffer offset or a ring of frames, with output bit-identical to one whole-clip call. |
| `attention`, `attn256` | Flash attention for heads of 128 and 256: causal / valid-length masks, grouped-query attention, an optional soft cap. Variants exist for 16 KB of workgroup memory. |
| `attnF` | Register-tiled flash attention for heads of 128 and 256, tuned on the iPhone's GPU: 1.6–1.9× and 3.8–3.9× faster than the standard kernels there. |
| Small kernels | Row statistics, GroupNorm, softmax, elementwise ops, transpose, small direct convolutions, RoPE tables, linear-attention state, depth-to-space. |

## Results on an iPhone 16 Pro Max

| | Before | After |
|---|---|---|
| Attention, head 128 | 464 GFLOP/s | 749 GFLOP/s |
| Attention, head 256 | 145 GFLOP/s | 564 GFLOP/s |
| 3×3×3 convolution | | 9–15% faster, bit-identical |
| p0pc0rn picture, 1280×720 | 174.7 s | 156.2 s (−10.6%) |
| p0pc0rn clip, 1280×736, 49 frames | 689.4 s | 473.7 s (−31.3%) |

The clip timings also include the app's time-chunked decoder, which uses `conv3`. Method, more devices and what didn't work: [docs/RESULTS.md](docs/RESULTS.md).

## Requirements

- A browser with WebGPU and the `shader-f16` feature: Safari 26 on iPhone, iPad and Mac, Chrome or Edge on desktop and Android.
- The phone-tuned attention needs 32 KB of workgroup memory. On devices with less, `attention()` and `attn256()` keep using the standard kernels.

## Use it

There is nothing to install. The kernels are plain ES modules with no dependencies, so a web page can import them straight from this repository's site:

```js
import { createGpu, gemm } from 'https://artistjoshhughes.github.io/p0pc0rn-kernels/src/index.js';
```

The site serves the main branch, so it changes whenever the repository does. To keep a fixed version, download the repository (**Code → Download ZIP**) and copy `src/` into your project. TypeScript declarations are in the repository's `types/index.d.ts`; they use WebGPU's types from [`@webgpu/types`](https://www.npmjs.com/package/@webgpu/types).

## Quick start

```js
import { createGpu, defaultTune, calibrate, gemm, toF16, fromF16 } from 'https://artistjoshhughes.github.io/p0pc0rn-kernels/src/index.js';

const gpu = await createGpu();      // WebGPU device with shader-f16
defaultTune(gpu);                    // fp16 partial sums, phone-tuned attention where it fits
await calibrate(gpu);                // measures speed, so no dispatch runs longer than 0.2 s

const M = 64, N = 256, K = 512;
const x = gpu.upload(toF16(new Float32Array(M * K).fill(0.5)));   // activations [M][K], fp16
const w = gpu.upload(toF16(new Float32Array(N * K).fill(0.01)));  // weights [N][K], fp16
const y = gpu.buffer(M * N * 2);                                  // output [M][N], fp16

gemm(gpu, { w: { enc: 'f16', buf: w, off: 0, bytes: N * K * 2, rows: N, cols: K }, x: { buf: x }, y: { buf: y }, M, N, K });
const out = fromF16(new Uint16Array(await gpu.read(y, 0, M * N * 2)));   // ≈ 2.56 everywhere
```

Every op is called as `op(gpu, args)`. The argument objects are typed in `types/index.d.ts` and documented above each function in `src/ops/`, and [docs/DESIGN.md](docs/DESIGN.md) explains the kernels. The weight encodings are specified byte by byte in [docs/FORMAT.md](docs/FORMAT.md). The tests in `test/kernels.test.js` show complete calls for GEMM, the 3D convolution and each attention kernel.

## Converting a model

The converter turns safetensors files (F32, F16, BF16 or FP8) into the kernels' encodings. A model larger than memory converts too, because tensors stream in blocks. It runs with Node 22 or later in a copy of this repository:

```sh
node tools/convert.mjs path/to/model --out model-q4 --bits 4
```

- Matrices that fit 64-row tiles become 4-bit (`--bits 8` gives 8-bit, `--bits 16` fp16), and everything else becomes fp16.
- Vectors (norms, biases) become fp32. Embeddings stay fp16 (`--keep` takes a regular expression).
- Convolution weights are reordered from PyTorch's layout to the one the convolution kernels read.
- The output is a `manifest.json` plus shards of at most 512 MB. Add `--dry-run` to see the plan first.

In the page, load the result and pass tensors straight to the ops:

```js
import { loadWeights, gemm } from 'https://artistjoshhughes.github.io/p0pc0rn-kernels/src/index.js';
const model = await loadWeights(gpu, 'https://example.com/model-q4/');
gemm(gpu, { w: model.tensor('model.layers.0.mlp.up_proj.weight'), x, y, M, N, K });
```

`convertSafetensors()` does the same for one file in memory, in a browser too.

## Tests

```sh
npm run check                  # no GPU: syntax, licence headers, shader generation, the converter's checks
npm test                       # the GPU tests in headless Chrome or Edge (Node 22+; --browser <path> to choose)
node test/run.mjs --low-power  # the integrated GPU, on a machine with two
```

Or serve the repository with any static file server and open `test/index.html`, which also works on a phone.

## Benchmark and tuning

Open `bench/index.html` through a static server (the repository's GitHub Pages site serves it), then:

- **Benchmark** times each new kernel next to the one it replaced. It uses interleaved rounds, reports medians, and runs a canary before and after so that a hot device shows up.
- **Tune for this device** tries the settings of each kernel (tile shapes, accumulation, attention precision), checks each against the default's output, and reports the fastest correct one.
- **Share results** opens a pre-filled post in this repository's **Benchmarks** discussions. It shows exactly what will be posted: the GPU's type and limits, the operating system, the browser's name and version, the date and the timings. Nothing is published unless the person posts it; opening GitHub passes the text to GitHub in the link.

Shared tuning results become entries in `TUNE_TABLE` (`src/core/tune.js`), which `defaultTune()` applies per GPU family. Headless runs: `npm run bench`, or `node test/run.mjs --page "bench/index.html?auto=1&sweep=1"` for the sweep.

## Repository layout

```
src/core/     gpu.js (device, buffer pool, dispatch batching), tune.js (calibrate, defaultTune, TUNE_TABLE)
src/ops/      gemm.js, attention.js, misc.js, video.js, video2.js
src/convert/  encode.js (safetensors, the encodings), convert.js (streaming converter)
src/          index.js, weights.js (loadWeights)
tools/        convert.mjs (command line)
types/        index.d.ts
test/         kernels.test.js (GPU, CPU references), convert.test.mjs, index.html, run.mjs (headless runner)
bench/        bench.js, sweep.js, share.js, index.html
docs/         DESIGN.md, FORMAT.md, RESULTS.md
site/         index.html (the GitHub Pages front page)
```

## Licence

Mozilla Public License 2.0: see [LICENSE](LICENSE).

- You can use the kernels in any project, open or closed, commercial or not.
- If you change these files and distribute the result, publish your changed files under MPL-2.0.
- Each file's copyright and licence notice must stay in place.
- The licence grants no rights to the p0pc0rn name or logo.

The p0pc0rn app is a separate project under its own licence.
