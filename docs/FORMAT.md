# Weight encodings

The GEMM family reads weights in one of three encodings and dequantises them inside the kernel: codes are expanded to an fp16 tile in workgroup memory and never to a full tensor in device memory. The test suite's `packQ4` and `packQ8` (in `test/kernels.test.js`) are reference encoders for the two quantised formats.

All multi-byte values are little-endian. A "u32 word" is 4 bytes read as a little-endian u32, which is what WGSL `array<u32>` sees. `N` is the number of rows (output features), `K` the number of columns (input features); the weight is `W[n, k]` and the GEMM computes `y = x·Wᵀ`.

## `q4g32a.t64`: 4-bit, groups of 32 along K, asymmetric, 64-row tile records

Requires N % 64 = 0 and K % 32 = 0.

**Quantiser.** For each row n and group g, the 32 values are `v = W[n, 32g .. 32g+31]` in fp32:

```
mn = f16(min(v))                                   # rounded to fp16 first
s  = f16((max(v) - mn) / 15)                       # fp32 maths, then rounded to fp16; if s == 0 then s = 1.0
q  = clamp(round_half_even((v - mn) / s), 0, 15)   # fp32 maths
dequantised: w = q * s + mn                        # fp32
```

**Layout.** NT = N/64 tiles, G = K/32 groups. The tensor is NT·G **records of 1280 bytes**, `record(t, g)` at byte `(t·G + g)·1280`. The k-groups of one tile are consecutive, so a workgroup walking K for a fixed tile reads sequentially.

| Bytes in the record | Content |
|---|---|
| 0..1023 | Codes: 64 rows × 16 bytes, row r (output n = 64t + r) at byte 16r. Byte j of a row is `q[k=2j] \| (q[k=2j+1] << 4)`. Equivalently, u32 word w (0..3) of row r holds k = 8w..8w+7, nibble i (bits 4i..4i+3) = k 8w+i. |
| 1024..1279 | 64 u32 words; word r = `bits(f16 scale) \| bits(f16 min) << 16`. In WGSL: `let sm = unpack2x16float(word); scale = sm.x; min = sm.y`. |

Record and tensor offsets are 16-byte aligned (1280 = 80·16), so `vec4<u32>` loads are legal. Size: N·K·5/8 bytes (5.0 bits per weight).

## `q8g64s.t64`: 8-bit, groups of 64 along K, symmetric, 64-row tile records

Requires N % 64 = 0 and K % 64 = 0.

**Quantiser:**

```
s = f16(max|v| / 127)                         # if s == 0 then s = 1.0
q = clamp(round_half_even(v / s), -128, 127)
dequantised: w = q * s
```

**Layout.** G = K/64. Records of **4224 bytes** at `(t·G + g)·4224`:

- Bytes 0..4095: codes, 64 rows × 64 bytes, row r at 64r; byte i is the int8 (two's complement) code for k = 64g + i. So u32 word w of a row holds k = 4w..4w+3, byte i = k 4w+i.
- Bytes 4096..4223: 64 × f16 scale; row r at 4096 + 2r, so u32 word r/2 holds rows 2⌊r/2⌋ (low half) and 2⌊r/2⌋+1 (high half).

Size: N·K·1.03125 bytes (8.25 bits per weight).

## `f16`

IEEE binary16, rounded to nearest even from the fp32 source, row-major `[N][K]` (the row stride may be larger than K: pass `w.ld`).

Convolution weights are stored so that each output channel's row is the contiguous K vector of the implicit GEMM:

- 2D convolutions (`conv`): OHWI, K = kh·kw·Cin in the order (ky, kx, ci).
- 3D convolutions (`conv3`, 3×3×3): K = 27·Cin in the order (kt, ky, kx, ci), that is `k = (kt·9 + ky·3 + kx)·Cin + ci`.

## `f32`

IEEE binary32, little-endian. The converter stores vectors this way (norm weights, biases): the ops read them as fp32 arrays, with offsets counted in floats.

## Converted models

`tools/convert.mjs` (or `convertSafetensors()`) writes one folder: `manifest.json` and shards named `weights-00001.bin`, `weights-00002.bin` and so on. Each shard is at most `--shard-mb` (512 MB by default; one bigger tensor gets a shard of its own), so a phone can upload each shard as one buffer. Every tensor starts at a 256-byte boundary inside its shard (WebGPU's storage-buffer offset alignment), and the padding is zeros.

```json
{
  "format": "p0pc0rn-kernels.weights/1",
  "sources": ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
  "bits": 4,
  "tensors": {
    "model.layers.0.mlp.up_proj.weight": { "enc": "q4g32a.t64", "dtype": "BF16", "shape": [9216, 3072], "rows": 9216, "cols": 3072,
      "layout": "row-major", "file": "weights-00001.bin", "offset": 0, "bytes": 17694720 }
  },
  "shards": [{ "file": "weights-00001.bin", "bytes": 536870656 }],
  "skipped": { "position_ids": "I64" },
  "created": "2026-10-20T08:30:00.000Z"
}
```

- `enc` is one of the encodings above. `dtype` and `shape` describe the source tensor.
- `rows` × `cols` is the matrix the kernels see. `layout` is `row-major`, `vector`, `OHWI` (2D convolution, reordered from PyTorch's OIHW) or `OTHWI` (3D convolution, from OITHW).
- Tensors that are not floating point (ids, masks) are listed in `skipped` and not stored.

## Activations

- Activations are fp16, row-major `[M][ld]`. LayerNorm and RMSNorm prologues read an fp32 residual instead.
- Images are HWC fp16 (`conv`) and clips are THWC fp16 (`conv3`).
- Attention takes Q `[Hq][Lq][D]` and K, V `[Hkv][Lk][D]` (head-major, as the GEMM's `heads` epilogue writes them), and writes O token-major, `[Lq][ld]` at column `h·D`.
