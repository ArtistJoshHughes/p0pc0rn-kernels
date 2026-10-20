// SPDX-License-Identifier: MPL-2.0
// Copyright 2026 Joshua Hughes
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// Sharing results: nothing leaves the device by itself. report() keeps only what tuning needs (the GPU's type and limits, the operating
// system, the browser's name and version, the date, the timings); post() opens a pre-filled GitHub Discussion in the Benchmarks category,
// which passes that text to GitHub in the link; the person reviews it and posts (or not) from their own GitHub account.
export const REPO = 'ArtistJoshHughes/p0pc0rn-kernels';   // owner/name of the repository whose Discussions collect results
export const VERSION = '0.1.0';

/** the operating system and the browser's name and version from the user agent, never the full string (it can identify a device) */
export function platform(ua = navigator.userAgent) {
  const os = /iPhone/.test(ua) ? 'iPhone' : /iPad/.test(ua) ? 'iPad' : /Android/.test(ua) ? 'Android' : /Mac OS X/.test(ua) ? 'Mac' : /Windows/.test(ua) ? 'Windows' : /Linux/.test(ua) ? 'Linux' : 'other';
  const m = ua.match(/Edg\/(\d+)/) || ua.match(/Chrome\/(\d+)/) || ua.match(/Firefox\/(\d+)/) || ua.match(/Version\/(\d+(\.\d+)?).*Safari/);
  const name = /Edg\//.test(ua) ? 'Edge' : /Chrome\//.test(ua) ? 'Chrome' : /Firefox\//.test(ua) ? 'Firefox' : /Safari\//.test(ua) ? 'Safari' : 'other';
  return { os, browser: name + (m ? ' ' + m[1] : '') };
}

/** what a shared result holds */
export function report(gpu, { bench = null, sweep = null, canary = null } = {}) {
  const info = gpu.info || {}, L = gpu.limits || {};
  return {
    v: 1, kernels: VERSION, date: new Date().toISOString().slice(0, 10), ...platform(),
    gpu: { vendor: info.vendor || '', architecture: info.architecture || '', description: info.description || '', subgroups: !!gpu.subgroups,
      workgroupKB: Math.round((L.maxComputeWorkgroupStorageSize || 0) / 1024), maxBindingMB: Math.round((L.maxStorageBufferBindingSize || 0) / 2 ** 20) },
    calib: gpu.calib || null, canary,
    bench: bench && bench.map((r) => ({ name: r.name, ms: r.ms, gflops: r.gflops })),
    sweep: sweep && sweep.map((f) => ({ key: f.key, best: f.best, settings: f.settings.map((s) => ({ label: s.label, ok: s.ok, ms: s.ms, gflops: s.gflops })) })),
  };
}

/** the Discussion's title and body: a readable summary, then the data as JSON */
export function discussion(r) {
  const title = `${r.os} · ${r.browser} · ${r.gpu.vendor || 'GPU'} ${r.gpu.architecture || ''}`.replace(/\s+/g, ' ').trim();
  const lines = [`**Device:** ${r.os}, ${r.browser}; GPU ${[r.gpu.vendor, r.gpu.architecture, r.gpu.description].filter(Boolean).join(' / ') || 'unknown'};` +
    ` ${r.gpu.workgroupKB} KB workgroup memory; subgroups ${r.gpu.subgroups ? 'yes' : 'no'}`, ''];
  if (r.bench) { lines.push('| Kernel | ms | GFLOP/s |', '|---|---|---|'); for (const b of r.bench) lines.push(`| ${b.name} | ${b.ms} | ${b.gflops} |`); lines.push(''); }
  if (r.sweep) { lines.push('**Fastest settings here:**'); for (const f of r.sweep) lines.push(`- ${f.key}: ${f.best ? `${f.best.label} (×${f.best.speedup} vs default)` : 'none passed'}`); lines.push(''); }
  if (r.canary) lines.push(`Canary: ${r.canary.startGflops} → ${r.canary.endGflops} GFLOP/s`, '');
  let json = JSON.stringify(r);
  const body = (j) => lines.join('\n') + '\n<details><summary>Data</summary>\n\n```json\n' + j + '\n```\n</details>\n';
  if (encodeURIComponent(body(json)).length > 6000) json = JSON.stringify(Object.assign({}, r, { sweep: r.sweep && r.sweep.map((f) => ({ key: f.key, best: f.best })) }));
  return { title, body: body(json) };
}
export const discussionUrl = ({ title, body }) =>
  `https://github.com/${REPO}/discussions/new?category=benchmarks&title=${encodeURIComponent(title)}&body=${encodeURIComponent(body)}`;
