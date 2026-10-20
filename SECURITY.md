# Security

Please report a security problem privately, through this repository's **Security** tab ("Report a vulnerability"), rather than in a public issue.

These kernels run inside the browser's WebGPU sandbox and make no network requests. The most likely problems are reads or writes outside a buffer's intended range, or shaders that hang the GPU. Reports of either are useful even when the browser contains them.
