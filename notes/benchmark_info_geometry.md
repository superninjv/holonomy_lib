# Benchmark results

_torch 2.12.0+cpu, CUDA available: False_


## `info_geometry.bregman_divergence_squared_euclidean`

_Bregman with F(x) = (1/2)||x||^2; sum-of-squares cost._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 d=16 | cpu | torch.float64 | 0.024 | 0.023 | 8 |
| B=1 d=256 | cpu | torch.float64 | 0.025 | 0.025 | 8 |
| B=1 d=4096 | cpu | torch.float64 | 0.030 | 0.030 | 8 |
| B=256 d=64 | cpu | torch.float64 | 0.062 | 0.053 | 8 |

## `info_geometry.kl_divergence_categorical`

_Discrete KL with the torch.where guard for the q=0 case._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 k=16 | cpu | torch.float64 | 0.064 | 0.052 | 8 |
| B=1 k=256 | cpu | torch.float64 | 0.059 | 0.050 | 8 |
| B=1024 k=64 | cpu | torch.float64 | 1.156 | 1.105 | 8 |

## `info_geometry.kl_divergence_gaussian`

_Gaussian KL with Cholesky-based solves; dominated by O(B*d^3)._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 d=4 | cpu | torch.float64 | 0.128 | 0.124 | 8 |
| B=1 d=16 | cpu | torch.float64 | 0.156 | 0.149 | 8 |
| B=1 d=64 | cpu | torch.float64 | 0.210 | 0.189 | 8 |
| B=16 d=16 | cpu | torch.float64 | 0.151 | 0.147 | 8 |
| B=1 d=256 | cpu | torch.float64 | 1.012 | 0.938 | 8 |