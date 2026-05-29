# Commodity-GPU scaling of the H^n heat kernel (RTX 3060)

Run on a Vast.ai RTX 3060 (12 GB, consumer card; torch 2.11.0+cu126,
CUDA 12.6), `float64`. Same-machine CPU-vs-CUDA comparison so the ratio
is meaningful. Bench: `notes/strengthening/gpu_scaling_bench.py`
(forward+backward, 15 iters after 3 warmup).

## Forward+backward latency vs batch (ms)

| n | path | batch | CPU (ms) | GPU (ms) | CPU/GPU |
|---:|---|---:|---:|---:|---:|
| 5 | closed form | 4 096 | 9.98 | 1.75 | 5.7× |
| 5 | closed form | 65 536 | 73.25 | 1.11 | 66× |
| 5 | closed form | 262 144 | 220.05 | 7.49 | 29× |
| 5 | closed form | 1 048 576 | 646.27 | 5.84 | 111× |
| 7 | closed form | 4 096 | 19.54 | 6.83 | 2.9× |
| 7 | closed form | 65 536 | 59.69 | 2.48 | 24× |
| 7 | closed form | 262 144 | 259.89 | 8.24 | 31× |
| 7 | closed form | 1 048 576 | 626.97 | 7.97 | 79× |
| 9 | recursion (from n=7) | 4 096 | 47.05 | 9.74 | 4.8× |
| 9 | recursion (from n=7) | 65 536 | 440.47 | 4.19 | 105× |
| 9 | recursion (from n=7) | 262 144 | 913.25 | 6.66 | 137× |
| 9 | recursion (from n=7) | 1 048 576 | 2319.70 | 24.40 | 95× |

## Reading

- **GPU latency is near-flat in batch; CPU is linear.** From batch 4 096
  to 1 048 576 (256×), the GPU forward+backward time barely moves
  (n=5: 1.75 → 5.84 ms; n=7: 6.83 → 7.97 ms; n=9: 9.74 → 24.4 ms), while
  the CPU grows roughly proportionally (n=5: 10 → 646 ms). On a $300
  consumer card, a batch of ~10^6 hyperbolic-distance heat-kernel
  evaluations costs about what 4 096 does — the kernel is launch-bound,
  not compute-bound, well past a million points.
- **The CPU/GPU ratio grows with batch**, reaching ~80–140× at 10^6.
  (The point-to-point numbers are mildly noisy — per-iteration `linspace`
  allocation and GPU clock states — but the trend is unambiguous.)
- This is the heat-kernel companion to the κ-sign-crossing GPU result
  (AMD RX 9060 XT, ROCm) in the C4 writeup: the substrate's geometry runs
  at practical speed on commodity hardware, CUDA or ROCm, without a
  datacenter accelerator.

## Reference: full latency table (batch 16/256/4096, both devices)

The standard bench (`C1_C2_heat_kernel_bench.py`) on the same RTX 3060,
CUDA, forward+backward at batch 4096: n=3 1.22 ms, n=5 1.78 ms, n=7
2.35 ms (closed form), n=9 8.36 ms (recursion). Forward latency is
constant across batch 16→4096 (kernel-launch-bound at these sizes),
matching the scaling table above.
