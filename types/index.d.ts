// SPDX-License-Identifier: MPL-2.0
// Copyright 2026 Joshua Hughes
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// Type declarations for p0pc0rn-kernels. WebGPU's own types come from @webgpu/types.
/// <reference types="@webgpu/types" />

// ------------------------------------------------------------------ runtime
export declare const MB: number;
export declare class GpuError extends Error {}

export interface GpuOptions {
  powerPreference?: GPUPowerPreference;
  /** use the `subgroups` feature where the adapter has it (default true) */
  wantSubgroups?: boolean;
  wantF16?: boolean;
  /** the longest a single dispatch may run, in seconds (default 0.2); phones and Windows reset the GPU on long dispatches */
  maxDispatchSeconds?: number;
  timestamps?: boolean;
  onLost?: (info: GPUDeviceLostInfo) => void;
}
/** measured throughput (FLOP/s, bytes/s) that sizes every dispatch; set by calibrate() */
export interface GpuRates { mm: number; mm32?: number; mm16?: number; attn: number; mem: number }
export interface Calibration { gemmF32Gflops: number; gemmF16Gflops: number; attnGflops: number }
export interface Tile { BM: number; BN: number }
/** kernel choices the ops read from gpu.tune; arguments can override them per call */
export interface GpuTune {
  acc?: 'f16' | 'f32';
  vaeAcc?: 'f16' | 'f32';
  attn?: AttentionConfig;
  attnF?: AttnFConfig | null;
  attnF256?: AttnFConfig | null;
  /** GEMM tiles per weight encoding, or per convolution kind (which wins over the encoding for convolutions) */
  gemm?: Partial<Record<'q4' | 'q8' | 'f16' | 'conv' | 'conv3', Tile>>;
}
/** a buffer the runtime allocated: `_pf` is its size in bytes */
export type Buffer = GPUBuffer & { _pf: number };

export declare class Gpu {
  constructor(opts?: GpuOptions);
  opts: GpuOptions & { maxDispatchSeconds: number };
  adapter: GPUAdapter;
  device: GPUDevice;
  limits: GPUSupportedLimits;
  info: { vendor: string; architecture: string; device: string; description: string };
  f16: boolean;
  subgroups: boolean;
  sgMin: number;
  sgMax: number;
  lost: boolean;
  rate: GpuRates;
  tune?: GpuTune;
  calib?: Calibration;
  runner: Runner;
  /** live and peak bytes of buffers the runtime holds */
  alive: number;
  peak: number;
  init(): Promise<void>;
  describe(): Record<string, string | number | boolean>;
  /** a storage buffer of `size` bytes (zeroed); reuses a pooled one of the same size and usage */
  buffer(size: number, label?: string, usage?: number): Buffer;
  /** a new buffer holding `data` (or write it into `dst` at `dstOff`) */
  upload(data: ArrayBuffer | ArrayBufferView, label?: string, dst?: GPUBuffer | null, dstOff?: number): Buffer;
  /** read [off, off + size) back; flushes queued work first */
  read(buf: GPUBuffer, off?: number, size?: number | null): Promise<ArrayBuffer>;
  /** return a buffer to the pool (or destroy it) */
  free(buf: GPUBuffer): void;
  freeAll(): void;
  drainPool(): void;
  poolStats(): Record<string, number>;
  resetPeak(): void;
  /** submit everything queued and wait until the GPU has finished it */
  done(): Promise<void>;
  /** the compute pipeline for `key`, compiled from codeFn() the first time */
  pipeline(key: string, codeFn: () => string): GPUComputePipeline;
  /** the most FLOPs (or bytes) one dispatch of this kind may do */
  cap(kind?: keyof GpuRates): number;
}
export declare class Runner {
  constructor(gpu: Gpu);
  /** queue one dispatch; params: up to 256 bytes of uniforms (see P()); sec: its estimated GPU time */
  dispatch(pipe: GPUComputePipeline, bindings: Array<GPUBuffer | GPUBufferBinding>, params: ArrayBuffer, workgroups: [number, number?, number?], sec?: number): void;
  copy(src: GPUBuffer, srcOff: number, dst: GPUBuffer, dstOff: number, size: number): void;
  flush(): void;
}
export interface ParamWriter { u32(v: number): ParamWriter; i32(v: number): ParamWriter; f32(v: number): ParamWriter; readonly ab: ArrayBuffer }
/** uniform parameters: P().u32(a).f32(b).ab */
export declare function P(): ParamWriter;
export declare function view(buffer: GPUBuffer, offset?: number, size?: number | null): GPUBufferBinding;
export declare function createGpu(opts?: GpuOptions): Promise<Gpu>;
export declare function f2h(v: number): number;
export declare function h2f(h: number): number;
export declare function toF16(a: ArrayLike<number>): Uint16Array;
export declare function fromF16(u: Uint16Array): Float32Array;

export interface TuneEntry { match: Partial<Record<'vendor' | 'architecture' | 'device' | 'description', string>>; tune: GpuTune; source?: string }
export declare const TUNE_TABLE: TuneEntry[];
export declare function tuneEntry(gpu: Gpu, table?: TuneEntry[]): TuneEntry | null;
/** set gpu.tune: the TUNE_TABLE entry for this GPU over the defaults, then `o` over both */
export declare function defaultTune(gpu: Gpu, o?: GpuTune & { table?: TuneEntry[] }): GpuTune;
/** measure GEMM and attention throughput (sets gpu.rate and gpu.calib) */
export declare function calibrate(gpu: Gpu): Promise<Calibration>;

// ------------------------------------------------------------------ tensors as the ops take them
/** a range of a buffer; `off` is in bytes unless an op says floats */
export interface View { buf: GPUBuffer; off?: number; size?: number }
export type Encoding = 'q4g32a.t64' | 'q8g64s.t64' | 'f16' | 'f32';
/** a weight matrix [rows][cols] in one of the encodings of docs/FORMAT.md */
export interface Weight { enc: Exclude<Encoding, 'f32'>; buf: GPUBuffer; off: number; bytes: number; rows: number; cols: number; ld?: number }

// ------------------------------------------------------------------ GEMM family
export interface ConvArgs {
  /** 'conv': 2D over HWC; 'conv3': 3x3x3 over THWC (stride carries the clip's frame count) */
  kind?: 'conv' | 'conv3';
  Hi: number; Wi: number; Ci: number; Ho: number; Wo: number;
  stride?: number;
  /** conv: padding; conv3: padT = first output frame of this call, padL = first frame of the input buffer */
  padT?: number; padL?: number;
  /** conv: kernel size; conv3: ring of ks input frames when > 1 */
  ks?: number;
  /** fused nearest-2x upsampling of the input (conv) */
  up2?: boolean;
  /** fused GroupNorm-apply + SiLU (conv; scale / shift from pv) */
  gn?: boolean;
  /** fused PixelNorm + SiLU (conv3; per-voxel scale from stats) */
  pn?: boolean;
}
export interface GemmArgs {
  w: Weight;
  /** activations [M][ld] fp16; mode 'lnmod' / 'rms' read an fp32 residual and apply the fused norm */
  x: View & { ld?: number; k0?: number; mode?: 'f16' | 'lnmod' | 'rms' };
  y: View & { ld?: number; n0?: number };
  M: number; N: number; K: number;
  epi?: 'f16' | 'f32' | 'resadd' | 'swiglu' | 'heads' | 'bias' | 'onormgate';
  acc?: 'f16' | 'f32';
  tile?: Tile;
  /** first weight row, first 'up' row (swiglu), first weight column */
  wr0?: number; wr1?: number; wk0?: number;
  /** gate / norm vectors (fp32, off in floats) */
  ev?: { buf: GPUBuffer; off?: number };
  stats?: { buf: GPUBuffer; off?: number };
  pv?: { buf: GPUBuffer; off?: number; stride?: number };
  bv?: { buf: GPUBuffer; off?: number; up?: number };
  gt?: { buf: GPUBuffer };
  res?: boolean;
  store?: boolean;
  act?: 'silu' | 'gelu';
  heads?: { Ltot: number; Hq: number; Hkv: number; rowOff?: number; eps?: number; ropeMode?: string; rope?: { buf: GPUBuffer } };
  conv?: ConvArgs;
}
export declare const DEFAULT_TILE: Tile;
/** Y = X'·Wᵀ with in-kernel dequantisation; returns the FLOPs */
export declare function gemm(gpu: Gpu, args: GemmArgs): number;
export declare function gemmCode(cfg: Record<string, unknown>): { code: string; bindings: string[] };

// ------------------------------------------------------------------ attention
export interface AttentionConfig { BQ: 16 | 32 | 64; sg?: boolean; pv16?: boolean; s16?: boolean; kvs?: boolean }
export interface AttnFConfig {
  D?: 128 | 256;
  /** 'p': exp2 softmax (attention()), 'v': natural exp with an optional soft cap (attn256()) */
  flavour?: 'p' | 'v';
  BQ: number; BK: number; TK?: number; PQ?: number; PD?: number; VK?: number;
  pv16?: boolean; s16?: boolean; il?: boolean; kpad?: number; ppad?: number;
  /** throughput used to size dispatches (FLOP/s) */
  rate?: number;
}
export interface AttentionArgs {
  /** Q [Hq][Lq][D], K / V [Hkv][Lk][D] fp16, head-major; O [Lq][ld] fp16, token-major at column h·D */
  q: View; k: View; v: View; o: View & { ld?: number };
  Lq: number; Lk?: number; Hq: number; Hkv?: number;
  /** 0 none, 1 causal + valid length, 2 valid length */
  mode?: 0 | 1 | 2;
  nvalid?: number;
  scale?: number;
  /** soft cap of the scores (head 256 / attnF 'v') */
  cap?: number;
  cfg?: AttentionConfig;
  /** override gpu.tune.attnF for this call (null = the standard kernel) */
  cfgF?: AttnFConfig | null;
}
export declare const ATTNF_P: AttnFConfig;
export declare const ATTNF_V: AttnFConfig;
/** head 128; uses attnF when gpu.tune.attnF is set and fits; returns the FLOPs */
export declare function attention(gpu: Gpu, args: AttentionArgs): number;
/** head 256; uses attnF when gpu.tune.attnF256 is set and fits; returns the FLOPs */
export declare function attn256(gpu: Gpu, args: AttentionArgs): number;
/** the register-tiled attention for heads of 128 or 256 (needs 32 KB of workgroup memory) */
export declare function attnF(gpu: Gpu, args: AttentionArgs, cfg: AttnFConfig): void;
export declare function attnFFits(gpu: Gpu, cfg: AttnFConfig): boolean;
export declare function attnFCfg(cfg: AttnFConfig): Required<Pick<AttnFConfig, 'BQ' | 'BK' | 'TK' | 'PQ' | 'PD' | 'VK'>> & { bytes: number; NT: number } & AttnFConfig;
export declare function attnFCode(cfg: AttnFConfig): string;
export declare function attnCode(cfg: AttentionConfig): string;
export declare function attnBytes(cfg: AttentionConfig): number;

// ------------------------------------------------------------------ small kernels (argument objects are documented above each function in src/ops/)
export type OpArgs = Record<string, any>;
export declare function rowstats(gpu: Gpu, args: OpArgs): void;
export declare function gnPartBytes(pixels: number): number;
export declare function gnPartial(gpu: Gpu, args: OpArgs): void;
export declare function gnFinalize(gpu: Gpu, args: OpArgs): void;
export declare function groupnormStats(gpu: Gpu, args: OpArgs): void;
export declare function ew(gpu: Gpu, args: OpArgs): void;
export declare function softmax(gpu: Gpu, args: OpArgs): void;
export declare function transpose16(gpu: Gpu, args: OpArgs): void;
export declare function convDirect(gpu: Gpu, args: OpArgs): void;
export declare function layout(gpu: Gpu, args: OpArgs): void;
export declare function laprep(gpu: Gpu, args: OpArgs): void;
export declare function kvstate(gpu: Gpu, args: OpArgs): void;
export declare function attnres(gpu: Gpu, args: OpArgs): void;
export declare function pixstats(gpu: Gpu, args: OpArgs): void;
export declare function d2s(gpu: Gpu, args: OpArgs): number;
export declare function rmshead(gpu: Gpu, args: OpArgs): void;
export declare function normadd(gpu: Gpu, args: OpArgs): void;
export declare function vop(gpu: Gpu, args: OpArgs): void;
export declare function rmsrow32(gpu: Gpu, args: OpArgs): void;
export declare function ropeWan(T: number, H: number, W: number, pairs: [number, number, number]): Float32Array;
export declare function ropeGemma(L: number): Float32Array;

// ------------------------------------------------------------------ converted models
export type Dtype = 'F32' | 'F16' | 'BF16' | 'F8_E4M3' | 'F8_E5M2';
export interface ManifestTensor { enc: Encoding; dtype: Dtype; shape: number[]; rows: number; cols: number; layout: 'row-major' | 'vector' | 'OHWI' | 'OTHWI'; file: string; offset: number; bytes: number }
export interface Manifest { format: string; sources: string[]; bits: number; tensors: Record<string, ManifestTensor>; shards: Array<{ file: string; bytes: number }>; skipped: Record<string, string>; created?: string }
export interface ConvertOptions {
  /** 4 -> q4g32a.t64, 8 -> q8g64s.t64, 16 -> fp16 (matrices that fit 64-row tiles; others fp16, vectors fp32) */
  bits?: 4 | 8 | 16;
  /** names that stay fp16 */
  keep?: RegExp | null;
  /** names stored fp32 */
  f32?: RegExp | null;
  shardBytes?: number;
  onTensor?: (name: string, entry: ManifestTensor) => void;
}
export interface Source { file: string; header: Record<string, any>; dataStart: number; read(offset: number, length: number): Uint8Array | Promise<Uint8Array> }
export interface Sink { begin(shard: string): void | Promise<void>; write(bytes: Uint8Array): void | Promise<void>; end(): void | Promise<void> }
export declare const MANIFEST_FORMAT: string;
export declare function convert(sources: Source[], out: Sink, opts?: ConvertOptions): Promise<Manifest>;
export declare function convertSafetensors(buffer: ArrayBuffer | Uint8Array, opts?: ConvertOptions, file?: string): Promise<{ manifest: Manifest; shards: Record<string, Uint8Array> }>;
export declare function writeSafetensors(tensors: Record<string, { dtype: Dtype | string; shape: number[]; data: Uint8Array }>, metadata?: Record<string, string> | null): Uint8Array;
export declare function encodedBytes(enc: Encoding, rows: number, cols: number): number;
export declare function safetensorsHeader(bytes: ArrayBuffer | Uint8Array): { header: Record<string, any>; dataStart: number };
export declare function safetensorsList(header: Record<string, any>): Array<{ name: string; dtype: string; shape: number[]; begin: number; end: number }>;
export declare function toFloat32(dtype: Dtype, bytes: ArrayBuffer | Uint8Array): Float32Array;
export declare function packQ4(W: Float32Array, rows: number, K: number, deq?: Float32Array | null): Uint8Array;
export declare function packQ8(W: Float32Array, rows: number, K: number, deq?: Float32Array | null): Uint8Array;
export declare function packF16(W: ArrayLike<number>): Uint8Array;
export declare function packF32(W: ArrayLike<number>): Uint8Array;
export declare function encodeRows(enc: Encoding, W: Float32Array, rows: number, K: number): Uint8Array;
export declare function matrixOf(shape: number[]): { rows: number; cols: number; layout: ManifestTensor['layout'] };
export declare function reorderConvRows(W: Float32Array, rows: number, shape: number[]): Float32Array;
export declare function chooseEncoding(name: string, shape: number[], opts?: ConvertOptions): Encoding;

export interface Model {
  manifest: Manifest;
  names(): string[];
  /** a weight ready for gemm's `w` */
  tensor(name: string): Weight & { shape: number[]; layout: ManifestTensor['layout'] };
  /** an fp32 vector as the ops take it (off in floats) */
  vector(name: string): { buf: GPUBuffer; off: number };
  free(): void;
}
/** fetch manifest.json + shards from `base` (a folder URL ending in '/') and upload them */
export declare function loadWeights(gpu: Gpu, base: string | URL, opts?: { fetch?: (url: string) => Promise<Response>; onProgress?: (done: number, total: number) => void }): Promise<Model>;
export declare function uploadWeights(gpu: Gpu, manifest: Manifest, shards: Record<string, Uint8Array>): Model;
