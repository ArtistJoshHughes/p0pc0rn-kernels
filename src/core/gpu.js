// SPDX-License-Identifier: MPL-2.0
// Copyright 2026 Joshua Hughes
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// Core runtime: device, buffer accounting, pipeline cache, batched dispatch runner.
// Pure JS (ES module), no dependencies, no wasm.  See docs/DESIGN.md.
'use strict';

const U = () => GPUBufferUsage;
const MB = 1e6;

class GpuError extends Error {}

/** Create the GPU context.  opts: {powerPreference, wantSubgroups=true, wantF16=true, maxDispatchSeconds=0.2, onLost(info)} */
async function createGpu(opts = {}) {
  if (!globalThis.navigator || !navigator.gpu) throw new GpuError('WebGPU is not available (navigator.gpu missing)');
  const g = new Gpu(opts);
  await g.init();
  return g;
}

/** v026.9: workgroup bytes a WGSL module declares (arrays of scalars / vectors; unknown element types count 4 bytes) */
function wgBytesOf(code) {
  const SZ = { f16: 2, f32: 4, i32: 4, u32: 4, 'vec2<f16>': 4, 'vec4<f16>': 8, 'vec2<f32>': 8, 'vec2<u32>': 8, 'vec4<f32>': 16, 'vec4<u32>': 16, 'vec3<f32>': 16, 'atomic<u32>': 4, 'atomic<i32>': 4 };
  const re = /var<workgroup>\s+\w+\s*:\s*(array<\s*([a-z0-9<>]+)\s*,\s*(\d+)\s*>|[a-z0-9<>]+)\s*;/g; let m, n = 0;
  while ((m = re.exec(code))) n += m[2] ? (SZ[m[2]] || 4) * Number(m[3]) : (SZ[m[1]] || 4);
  return n;
}
class Gpu {
  constructor(opts) {
    this.opts = Object.assign({ powerPreference: 'high-performance', wantSubgroups: true, wantF16: true, maxDispatchSeconds: 0.2 }, opts);
    this.alive = 0; this.peak = 0; this.bufs = new Set();
    // p0pc0rn v026.2: the reuse pool (see buffer / free); hw = the most live bytes this device ever held (the pool's cap with the live ones)
    this.pool = new Map(); this.pooled = 0; this.pooledN = 0; this.hw = 0; this.maxBuf = 0; this.ps = { creates: 0, reuses: 0, evictions: 0 }; this.poolSeq = 0;
    this.pipes = new Map();
    this.lostInfo = null;
    // FLOP/s assumed for dispatch sizing until calibrate() measures it (conservative = the slow desktop iGPU)
    this.rate = { mm: 0.2e12, attn: 0.08e12, mem: 20e9 };
  }

  async init() {
    const ad = await navigator.gpu.requestAdapter({ powerPreference: this.opts.powerPreference });
    if (!ad) throw new GpuError('no WebGPU adapter');
    this.adapter = ad;
    const info = ad.info || {};
    this.info = { vendor: info.vendor || '', architecture: info.architecture || '', device: info.device || '', description: info.description || '' };
    const feats = [];
    this.f16 = ad.features.has('shader-f16');
    if (!this.f16) throw new GpuError('shader-f16 is required (all target devices have it)');
    feats.push('shader-f16');
    this.subgroups = !!(this.opts.wantSubgroups && ad.features.has('subgroups'));
    if (this.subgroups) feats.push('subgroups');
    if (ad.features.has('timestamp-query') && this.opts.timestamps) feats.push('timestamp-query');
    const L = ad.limits;
    const want = {
      maxBufferSize: L.maxBufferSize,
      maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
      maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: L.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: L.maxComputeWorkgroupSizeX,
      maxStorageBuffersPerShaderStage: L.maxStorageBuffersPerShaderStage,
    };
    this.device = await ad.requestDevice({ requiredFeatures: feats, requiredLimits: want });
    this.limits = this.device.limits;
    // subgroup size range (Chrome exposes adapter.info.subgroupMinSize / MaxSize); kernels that rely on lane groups of 16 check this
    this.sgMin = info.subgroupMinSize || 0; this.sgMax = info.subgroupMaxSize || 0;
    this.sgOK16 = this.subgroups && this.sgMin >= 16;
    this.device.lost.then((li) => {
      this.lostInfo = { reason: li && li.reason, message: li && li.message, t: Date.now() };
      if (this.opts.onLost) try { this.opts.onLost(this.lostInfo); } catch (e) { /* ignore */ }
    });
    this.device.addEventListener('uncapturederror', (ev) => {
      this.lastError = (ev.error && ev.error.message) || String(ev.error);
      if (this.opts.onError) this.opts.onError(this.lastError); else console.error('[gpu] ' + this.lastError);
    });
    this.runner = new Runner(this);
  }

  get lost() { return !!this.lostInfo; }
  describe() {
    return Object.assign({}, this.info, { f16: this.f16, subgroups: this.subgroups, sgMin: this.sgMin, sgMax: this.sgMax,
      maxBufferMB: Math.round(this.limits.maxBufferSize / 1048576), maxBindMB: Math.round(this.limits.maxStorageBufferBindingSize / 1048576),
      wgMem: this.limits.maxComputeWorkgroupStorageSize, wgInv: this.limits.maxComputeInvocationsPerWorkgroup });
  }

  // ------------------------------------------------------------------ buffers (every live byte is counted; iOS kills the page ~1.5 GB)
  buffer(size, label = '', usage = 0) {
    size = Math.max(16, Math.ceil(size / 16) * 16);
    if (size > this.limits.maxBufferSize) throw new GpuError(`buffer ${label} ${size} B exceeds maxBufferSize`);
    const us = usage || (U().STORAGE | U().COPY_DST | U().COPY_SRC), key = us + ':' + size;
    // p0pc0rn v026.2: a pooled buffer of exactly this size and usage: cleared (zeroed, like a new one) before anything later uses it
    const L = this.pool.get(key);
    if (L && L.length) {
      const b = L.pop(); if (!L.length) this.pool.delete(key);
      this.pooled -= size; this.pooledN--; this.ps.reuses++;
      this.runner.flush();
      const enc = this.device.createCommandEncoder(); enc.clearBuffer(b); this.device.queue.submit([enc.finish()]);
      this.bufs.add(b); this.alive += size;
      if (this.alive > this.peak) this.peak = this.alive;
      if (this.alive > this.hw) this.hw = this.alive;
      return b;
    }
    // a new buffer: pooled + live never above the cap (destroy the oldest pooled ones to make room)
    this._evict(Math.max(this.poolCap(), this.alive + size) - this.alive - size);
    const b = this.device.createBuffer({ size, label, usage: us });
    b._pf = size; b._key = key; this.bufs.add(b); this.alive += size; this.ps.creates++; if (size > this.maxBuf) this.maxBuf = size;
    if (this.alive > this.peak) this.peak = this.alive;
    if (this.alive > this.hw) this.hw = this.alive;
    return b;
  }
  /** p0pc0rn v026.2: a freed buffer goes to the pool if pooled + live stays within the high-water mark; else it is destroyed */
  free(b) {
    if (!b || !this.bufs.has(b)) return;
    this.bufs.delete(b); this.alive -= b._pf;
    if (b._key && !this.lost && this.alive + this.pooled + b._pf <= this.poolCap()) {
      const L = this.pool.get(b._key) || []; L.push(b); this.pool.set(b._key, L); b._seq = ++this.poolSeq;
      this.pooled += b._pf; this.pooledN++;
      return;
    }
    b.destroy();
  }
  /** destroy pooled buffers, oldest first, until at most `room` bytes are pooled */
  _evict(room) {
    if (this.pooled <= room) return;
    const all = []; for (const L of this.pool.values()) for (const b of L) all.push(b);
    all.sort((x, y) => x._seq - y._seq);
    for (const b of all) {
      if (this.pooled <= room) break;
      const L = this.pool.get(b._key); L.splice(L.indexOf(b), 1); if (!L.length) this.pool.delete(b._key);
      this.pooled -= b._pf; this.pooledN--; this.ps.evictions++;
      b.destroy();
    }
  }
  drainPool() { this._evict(0); }
  /** v026.3: the most live bytes this device held + its largest buffer: room for the alternate weight shard of a streamed loop */
  poolCap() { return this.hw + this.maxBuf; }   // (not cap(): that is the dispatch-size cap below)
  poolStats() { return { aliveMB: Math.round(this.alive / 1e6), pooledMB: Math.round(this.pooled / 1e6), bufs: this.bufs.size, pooledN: this.pooledN, creates: this.ps.creates, reuses: this.ps.reuses, evictions: this.ps.evictions, hwMB: Math.round(this.hw / 1e6), capMB: Math.round(this.poolCap() / 1e6) }; }
  freeAll() { for (const b of Array.from(this.bufs)) this.free(b); this.drainPool(); }
  resetPeak() { this.peak = this.alive; }
  /** upload a TypedArray / ArrayBuffer into a NEW buffer (or into dst at dstOff) */
  upload(data, label = '', dst = null, dstOff = 0) {
    const ab = data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    const b = dst || this.buffer(ab.byteLength, label);
    const n = ab.byteLength - (ab.byteLength % 4);
    if (n) this.device.queue.writeBuffer(b, dstOff, ab.buffer, ab.byteOffset, n);
    if (n < ab.byteLength) { const t = new Uint8Array(4); t.set(ab.subarray(n)); this.device.queue.writeBuffer(b, dstOff + n, t); }
    return b;
  }
  /** read back [off, off+size) of a buffer as an ArrayBuffer (flushes the runner first) */
  async read(buf, off = 0, size = null) {
    this.runner.flush();
    size = size == null ? buf._pf - off : size;
    const n = Math.ceil(size / 4) * 4;
    // p0pc0rn v026.2: ONE reusable staging buffer (grows by powers of two up to 64 MB; bigger reads get their own); reads serialised
    const go = async () => {
      let rb = null, own = false;
      if (n <= 64e6) {
        if (!this._rb || this._rb.size < n) { if (this._rb) this._rb.destroy(); let s2 = 1 << 20; while (s2 < n) s2 *= 2; this._rb = this.device.createBuffer({ size: s2, usage: U().MAP_READ | U().COPY_DST, label: 'readback' }); }
        rb = this._rb;
      } else { rb = this.device.createBuffer({ size: n, usage: U().MAP_READ | U().COPY_DST }); own = true; }
      const enc = this.device.createCommandEncoder();
      enc.copyBufferToBuffer(buf, off, rb, 0, n);
      this.device.queue.submit([enc.finish()]);
      await rb.mapAsync(GPUMapMode.READ, 0, n);
      try { return rb.getMappedRange(0, n).slice(0, size); } finally { rb.unmap(); if (own) rb.destroy(); }
    };
    const p = (this._rq || Promise.resolve()).then(go, go);
    this._rq = p.catch(() => {});
    return p;
  }
  async done() {
    this.runner.flush();
    if (this.lost) throw new GpuError('GPU device lost: ' + JSON.stringify(this.lostInfo));
    await this.device.queue.onSubmittedWorkDone();
    if (this.lost) throw new GpuError('GPU device lost: ' + JSON.stringify(this.lostInfo));
  }

  // ------------------------------------------------------------------ pipelines
  pipeline(key, codeFn) {
    let p = this.pipes.get(key);
    if (p) return p;
    const code = codeFn();
    const wl = this.limits && this.limits.maxComputeWorkgroupStorageSize, wb = wl ? wgBytesOf(code) : 0;   // v026.9: fail loudly, never draw garbage
    if (wl && wb > wl) throw new Error(`This GPU allows ${wl} bytes of workgroup memory; the kernel ${key} needs ${wb}`);
    const mod = this.device.createShaderModule({ code, label: key });
    p = this.device.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'main' }, label: key });
    p._key = key; p._code = code; p._mod = mod;
    this.pipes.set(key, p);
    return p;
  }
  async checkPipeline(p) {   // surface WGSL compile errors with line numbers (tests call this)
    const ci = await p._mod.getCompilationInfo();
    const errs = ci.messages.filter((m) => m.type === 'error');
    if (errs.length) throw new GpuError(`WGSL ${p._key}: ` + errs.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join(' | '));
  }

  /** seconds a dispatch of `flops` FLOPs of kind may take -> max FLOPs per dispatch (~0.2 s, the OS GPU watchdog guard) */
  cap(kind = 'mm') { return this.rate[kind] * this.opts.maxDispatchSeconds; }
}

/** Batches dispatches into command buffers.  Uniform params of a batch live in one uniform buffer written right before submit.
 *  A batch is submitted when its estimated GPU time reaches `batchSeconds` (keeps the GPU fed without long stalls) or it is full. */
class Runner {
  constructor(gpu) {
    this.g = gpu; this.SLOT = 256; this.CAP = 512; this.batchSeconds = 0.08;
    this.pool = []; this.cur = null; this.submits = 0; this.dispatches = 0;
  }
  _begin() {
    const d = this.g.device;
    let ub = this.pool.pop();
    if (!ub) ub = d.createBuffer({ size: this.SLOT * this.CAP, usage: U().UNIFORM | U().COPY_DST, label: 'uniform-batch' });
    this.cur = { enc: d.createCommandEncoder(), pass: null, ub, data: new ArrayBuffer(this.SLOT * this.CAP), n: 0, sec: 0 };
  }
  /** pipeline, list of storage bindings ({buffer, offset, size} | GPUBuffer), params (ArrayBuffer <= 256 B), workgroups [x,y,z], est seconds */
  /** v026.9: a storage binding never larger than maxStorageBufferBindingSize (an open-ended arena view is cut to it) */
  _fit(b) {
    const lim = this.g.limits && this.g.limits.maxStorageBufferBindingSize; if (!lim) return b;
    const buf = b instanceof GPUBuffer ? b : b.buffer, off = b instanceof GPUBuffer ? 0 : (b.offset || 0);
    const size = b instanceof GPUBuffer || b.size == null ? buf.size - off : b.size;
    if (size <= lim) return b;
    this.cuts = (this.cuts || 0) + 1; if (!this.cutLabels) this.cutLabels = new Set(); if (this.cutLabels.size < 32) this.cutLabels.add(String(buf.label || '?'));
    return { buffer: buf, offset: off, size: Math.floor(lim / 256) * 256 };
  }
  dispatch(pipe, bindings, params, wg, sec = 0) {
    bindings = bindings.map((b) => this._fit(b));   // v026.9
    if (!this.cur) this._begin();
    const c = this.cur;
    if (!c.pass) c.pass = c.enc.beginComputePass();
    const slot = c.n++;
    new Uint8Array(c.data, slot * this.SLOT, params.byteLength).set(new Uint8Array(params));
    const entries = [{ binding: 0, resource: { buffer: c.ub, offset: slot * this.SLOT, size: this.SLOT } }];
    bindings.forEach((b, i) => entries.push({ binding: i + 1, resource: b instanceof GPUBuffer ? { buffer: b } : b }));
    const bg = this.g.device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    c.pass.setPipeline(pipe); c.pass.setBindGroup(0, bg);
    c.pass.dispatchWorkgroups(wg[0], wg[1] || 1, wg[2] || 1);
    c.sec += sec; this.dispatches++;
    if (c.n >= this.CAP || c.sec >= this.batchSeconds) this.flush();
  }
  /** copy between buffers inside the current batch (ends the compute pass) */
  copy(src, srcOff, dst, dstOff, size) {
    if (!this.cur) this._begin();
    const c = this.cur;
    if (c.pass) { c.pass.end(); c.pass = null; }
    c.enc.copyBufferToBuffer(src, srcOff, dst, dstOff, size);
  }
  flush() {
    const c = this.cur;
    if (!c) return;
    this.cur = null;
    if (c.pass) c.pass.end();
    const q = this.g.device.queue;
    if (c.n) q.writeBuffer(c.ub, 0, c.data, 0, c.n * this.SLOT);
    q.submit([c.enc.finish()]);
    this.submits++;
    q.onSubmittedWorkDone().then(() => { if (this.pool.length < 64) this.pool.push(c.ub); else c.ub.destroy(); });   // p0pc0rn v026.2: keep up to 64 (was 8)
  }
}

/** Uniform params helper: P().u32(a).f32(b)... .ab  (all fields 4 B, std140-free because every struct member is a scalar) */
function P() {
  const ab = new ArrayBuffer(256), u = new Uint32Array(ab), f = new Float32Array(ab);
  let i = 0;
  const o = { u32(v) { u[i++] = v >>> 0; return o; }, i32(v) { u[i++] = v | 0; return o; }, f32(v) { f[i++] = v; return o; },
    get ab() { return ab.slice(0, Math.max(16, Math.ceil(i / 4) * 16)); }, get n() { return i; } };
  return o;
}

/** binding helper: a tensor view {buffer, offset, size} with 4-byte multiple size */
function view(buffer, offset = 0, size = null) {
  size = size == null ? buffer._pf - offset : size;
  return { buffer, offset, size: Math.ceil(size / 4) * 4 };
}

// ------------------------------------------------------------------ fp16 helpers (CPU side)
const _f = new Float32Array(1), _u = new Uint32Array(_f.buffer);
function f2h(v) {
  _f[0] = v; const x = _u[0];
  const sign = (x >>> 16) & 0x8000, e = (x >>> 23) & 0xff; let m = x & 0x7fffff;
  if (e === 0xff) return sign | 0x7c00 | (m ? 0x200 : 0);
  const ne = e - 112;
  if (ne >= 0x1f) return sign | 0x7c00;
  if (ne <= 0) {
    if (ne < -10) return sign;
    m |= 0x800000; const sh = 14 - ne; let hm = m >>> sh; const rem = m & ((1 << sh) - 1), half = 1 << (sh - 1);
    if (rem > half || (rem === half && (hm & 1))) hm++;
    return sign | hm;
  }
  let r = sign | (ne << 10) | (m >>> 13); const rem2 = m & 0x1fff;
  if (rem2 > 0x1000 || (rem2 === 0x1000 && (r & 1))) r++;
  return r;
}
function h2f(h) {
  const s = (h & 0x8000) ? -1 : 1, e = (h >>> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 5.960464477539063e-8;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * Math.pow(2, e - 15);
}
function toF16(f32arr) { const o = new Uint16Array(f32arr.length); for (let i = 0; i < f32arr.length; i++) o[i] = f2h(f32arr[i]); return o; }
function fromF16(u16) { const o = new Float32Array(u16.length); for (let i = 0; i < u16.length; i++) o[i] = h2f(u16[i]); return o; }

export { MB, GpuError, createGpu, Gpu, Runner, P, view, f2h, h2f, toF16, fromF16 };
