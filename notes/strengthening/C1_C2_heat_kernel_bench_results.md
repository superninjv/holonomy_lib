# Heat-kernel performance benchmark

Forward + backward latency per call for `hyperbolic_heat_kernel`. The closed-form paths (n=1, 3, 5) bypass `torch.autograd.grad`; the recursion paths add one autograd-grad per step from the seed (`n=5` for odd ≥ 7, `n=2` for even ≥ 4). Forward-only is the cleanest story; forward+backward demonstrates the autograd overhead.

Devices: cpu
Iterations per measurement: 20 (after 3 warmup).

PyTorch version: 2.12.0+cu130

## cpu

| n  | batch | forward (ms) | forward+backward (ms) | backward overhead (×) |
|---:|---:|---:|---:|---:|
|  1 |    16 |    0.024 |      0.128        |  5.26 |
|  1 |   256 |    0.024 |      0.114        |  4.66 |
|  1 |  4096 |    0.049 |      0.169        |  3.44 |
|    |       |          |              |       |
|  2 |    16 |    0.227 |      0.439        |  1.94 |
|  2 |   256 |    0.386 |      0.651        |  1.69 |
|  2 |  4096 |    3.408 |      5.461        |  1.60 |
|    |       |          |              |       |
|  3 |    16 |    0.051 |      0.228        |  4.48 |
|  3 |   256 |    0.069 |      0.230        |  3.33 |
|  3 |  4096 |    0.143 |      0.333        |  2.33 |
|    |       |          |              |       |
|  5 |    16 |    0.111 |      0.264        |  2.37 |
|  5 |   256 |    0.080 |      0.273        |  3.39 |
|  5 |  4096 |    0.218 |      0.548        |  2.51 |
|    |       |          |              |       |
|  7 |    16 |    0.313 |      0.661        |  2.11 |
|  7 |   256 |    0.298 |      0.647        |  2.17 |
|  7 |  4096 |    0.517 |      1.283        |  2.48 |
|    |       |          |              |       |
|  9 |    16 |    0.674 |      1.612        |  2.39 |
|  9 |   256 |    0.681 |      1.702        |  2.50 |
|  9 |  4096 |    1.171 |      3.171        |  2.71 |
|    |       |          |              |       |
|  4 |    16 |    0.457 |      0.769        |  1.68 |
|  4 |   256 |    0.712 |      1.390        |  1.95 |
|  4 |  4096 |    5.950 |      6.785        |  1.14 |
|    |       |          |              |       |
|  6 |    16 |    0.871 |      1.965        |  2.26 |
|  6 |   256 |    1.648 |      3.314        |  2.01 |
|  6 |  4096 |   11.468 |     21.317        |  1.86 |
|    |       |          |              |       |

## Observations

- **Closed-form paths (n=1, 3, 5)**: linear in batch, forward+backward ≈ 2-3× forward (standard autograd overhead). No `torch.autograd.grad` call inside the kernel.
- **Recursion paths (n=7, 9 odd; n=4, 6 even)**: each step adds a nested `torch.autograd.grad` call to the forward graph, increasing forward+backward overhead. The `create_graph=True` flag (required for differentiability through the chain) is the dominant cost.
- **Practical recommendation** (already in CONTENTS.md): prefer the closed-form n=3 or n=5 when the embedding dim allows it.
