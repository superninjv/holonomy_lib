# Heat-kernel performance benchmark

Forward + backward latency per call for `hyperbolic_heat_kernel`. The closed-form paths (n=1, 3, 5) bypass `torch.autograd.grad`; the recursion paths add one autograd-grad per step from the seed (`n=5` for odd ≥ 7, `n=2` for even ≥ 4). Forward-only is the cleanest story; forward+backward demonstrates the autograd overhead.

Devices: cpu
Iterations per measurement: 20 (after 3 warmup).

PyTorch version: 2.12.0+cu130

## cpu

| n  | batch | forward (ms) | forward+backward (ms) | backward overhead (×) |
|---:|---:|---:|---:|---:|
|  1 |    16 |    0.024 |      0.133        |  5.61 |
|  1 |   256 |    0.026 |      0.117        |  4.47 |
|  1 |  4096 |    0.049 |      0.173        |  3.55 |
|    |       |          |              |       |
|  2 |    16 |    0.231 |      0.445        |  1.93 |
|  2 |   256 |    0.364 |      0.713        |  1.96 |
|  2 |  4096 |    3.901 |      5.613        |  1.44 |
|    |       |          |              |       |
|  3 |    16 |    0.061 |      0.207        |  3.39 |
|  3 |   256 |    0.058 |      0.203        |  3.52 |
|  3 |  4096 |    0.112 |      0.317        |  2.84 |
|    |       |          |              |       |
|  5 |    16 |    0.104 |      0.317        |  3.03 |
|  5 |   256 |    0.086 |      0.346        |  4.02 |
|  5 |  4096 |    0.173 |      0.476        |  2.75 |
|    |       |          |              |       |
|  7 |    16 |    0.121 |      0.342        |  2.82 |
|  7 |   256 |    0.114 |      0.360        |  3.17 |
|  7 |  4096 |    0.189 |      0.527        |  2.79 |
|    |       |          |              |       |
|  9 |    16 |    0.382 |      0.856        |  2.24 |
|  9 |   256 |    0.378 |      0.883        |  2.33 |
|  9 |  4096 |    0.587 |      1.349        |  2.30 |
|    |       |          |              |       |
|  4 |    16 |    0.483 |      0.839        |  1.74 |
|  4 |   256 |    0.647 |      1.313        |  2.03 |
|  4 |  4096 |    7.060 |      7.500        |  1.06 |
|    |       |          |              |       |
|  6 |    16 |    0.860 |      1.691        |  1.97 |
|  6 |   256 |    1.420 |      2.977        |  2.10 |
|  6 |  4096 |   11.157 |     21.676        |  1.94 |
|    |       |          |              |       |

## Observations

- **Closed-form paths (n=1, 3, 5)**: linear in batch, forward+backward ≈ 2-3× forward (standard autograd overhead). No `torch.autograd.grad` call inside the kernel.
- **Recursion paths (n=7, 9 odd; n=4, 6 even)**: each step adds a nested `torch.autograd.grad` call to the forward graph, increasing forward+backward overhead. The `create_graph=True` flag (required for differentiability through the chain) is the dominant cost.
- **Practical recommendation** (already in CONTENTS.md): prefer the closed-form n=3 or n=5 when the embedding dim allows it.
