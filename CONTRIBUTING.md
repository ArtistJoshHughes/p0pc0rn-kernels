# Contributing

Bug reports, measurements from more devices, and kernel improvements are welcome.

## Before you open a pull request

1. Run `npm run check` (no GPU needed). It checks that every script parses, that every file has the licence header, and that the shaders generate.
2. Run `npm test` on a machine with a WebGPU browser, and include its output. Say which device and browser you ran it on.
3. For a speed change, run the benchmark before and after on the same device in one session, and include both results. Phones lose 40–50% of their speed when hot, so numbers from different sessions can't be compared.

## Kernel changes

- A new kernel or variant needs a test in `test/kernels.test.js` against a CPU reference, or a bit-identical check against the kernel it replaces.
- Keep the old kernel available until the new one has been measured on the devices it targets. Kernels that only some devices can run need a fallback, as `attnF` has.
- Never store a single component of a `vec4` in workgroup memory: on Apple GPUs it races. Transpose in registers and store whole vectors.
- Keep each dispatch under `gpu.opts.maxDispatchSeconds` by sizing the grid from `gpu.rate`. Long dispatches reset phone GPUs.
- Match the surrounding code's style: generated WGSL in JavaScript, comments that say why.

## Licence

By contributing, you agree that your contribution is licensed under the Mozilla Public License 2.0, the repository's licence. New source files carry the same header as the existing ones: the SPDX line, a copyright line and the licence notice.
