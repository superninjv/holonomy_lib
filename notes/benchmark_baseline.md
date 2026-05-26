# Benchmark results

_torch 2.12.0+cpu, CUDA available: False_


## `algebra.truncated_svd_exact`

_Full SVD then top-r slice (Eckart-Young exact)._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 m=64 n=64 r=8 | cpu | torch.float64 | 0.655 | 0.550 | 15 |
| B=1 m=64 n=64 r=8 | cpu | torch.float32 | 0.474 | 0.414 | 15 |
| B=1 m=256 n=256 r=16 | cpu | torch.float64 | 12.356 | 9.789 | 15 |
| B=1 m=256 n=256 r=16 | cpu | torch.float32 | 9.463 | 6.219 | 15 |
| B=1 m=1024 n=1024 r=32 | cpu | torch.float64 | 373.280 | 267.748 | 15 |
| B=1 m=1024 n=1024 r=32 | cpu | torch.float32 | 199.990 | 168.052 | 15 |
| B=1 m=2048 n=64 r=8 | cpu | torch.float64 | 3.394 | 2.201 | 15 |
| B=1 m=2048 n=64 r=8 | cpu | torch.float32 | 1.836 | 1.455 | 15 |
| B=16 m=128 n=128 r=16 | cpu | torch.float64 | 36.560 | 33.383 | 15 |
| B=16 m=128 n=128 r=16 | cpu | torch.float32 | 23.401 | 22.364 | 15 |

## `algebra.truncated_svd_randomized`

_Halko-Martinsson-Tropp randomized SVD with oversample=5, n_iter=2._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 m=64 n=64 r=8 | cpu | torch.float64 | 0.400 | 0.330 | 15 |
| B=1 m=64 n=64 r=8 | cpu | torch.float32 | 0.339 | 0.303 | 15 |
| B=1 m=256 n=256 r=16 | cpu | torch.float64 | 1.537 | 1.391 | 15 |
| B=1 m=256 n=256 r=16 | cpu | torch.float32 | 1.051 | 0.976 | 15 |
| B=1 m=1024 n=1024 r=32 | cpu | torch.float64 | 9.907 | 8.714 | 15 |
| B=1 m=1024 n=1024 r=32 | cpu | torch.float32 | 11.747 | 10.472 | 15 |
| B=1 m=2048 n=64 r=8 | cpu | torch.float64 | 1.642 | 1.239 | 15 |
| B=1 m=2048 n=64 r=8 | cpu | torch.float32 | 1.328 | 0.874 | 15 |
| B=16 m=128 n=128 r=16 | cpu | torch.float64 | 21.652 | 12.273 | 15 |
| B=16 m=128 n=128 r=16 | cpu | torch.float32 | 5.294 | 4.762 | 15 |

## `discrete_geometry.discrete_ricci_flow`

_Edge-weight evolution: w *= (1 - dt*κ) per step._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 n_steps=5 n_iter=50 | cpu | torch.float64 | 92.085 | 88.562 | 15 |
| B=1 n=16 n_steps=5 n_iter=50 | cpu | torch.float32 | 101.433 | 92.197 | 15 |
| B=1 n=32 n_steps=5 n_iter=50 | cpu | torch.float64 | 884.461 | 804.051 | 15 |
| B=1 n=32 n_steps=5 n_iter=50 | cpu | torch.float32 | 390.227 | 334.495 | 15 |

## `discrete_geometry.ollivier_ricci_curvature`

_Sinkhorn-based pairwise W_1 with shortest-path metric._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 n_iter=100 | cpu | torch.float64 | 33.975 | 27.528 | 15 |
| B=1 n=16 n_iter=100 | cpu | torch.float32 | 29.801 | 25.159 | 15 |
| B=1 n=32 n_iter=100 | cpu | torch.float64 | 273.433 | 214.953 | 15 |
| B=1 n=32 n_iter=100 | cpu | torch.float32 | 129.560 | 122.544 | 15 |
| B=1 n=64 n_iter=100 | cpu | torch.float64 | 22599.899 | 22180.103 | 15 |
| B=1 n=64 n_iter=100 | cpu | torch.float32 | 11775.811 | 10755.330 | 15 |
| B=4 n=16 n_iter=100 | cpu | torch.float64 | 77.581 | 64.984 | 15 |
| B=4 n=16 n_iter=100 | cpu | torch.float32 | 94.317 | 73.848 | 15 |

## `discrete_geometry.ricci_flow_with_surgery`

_Flow + periodic threshold-based edge removal._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 n_steps=10 n_iter=50 | cpu | torch.float64 | 197.902 | 162.702 | 15 |
| B=1 n=16 n_steps=10 n_iter=50 | cpu | torch.float32 | 197.382 | 178.874 | 15 |
| B=1 n=32 n_steps=10 n_iter=50 | cpu | torch.float64 | 1480.287 | 1051.322 | 15 |
| B=1 n=32 n_steps=10 n_iter=50 | cpu | torch.float32 | 836.093 | 727.892 | 15 |

## `manifolds.FixedRank.inner`

_Frobenius inner product over ambient tangents._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 m=64 n=64 r=8 | cpu | torch.float64 | 0.013 | 0.007 | 15 |
| B=1 m=64 n=64 r=8 | cpu | torch.float32 | 0.012 | 0.008 | 15 |
| B=1 m=256 n=256 r=16 | cpu | torch.float64 | 0.032 | 0.031 | 15 |
| B=1 m=256 n=256 r=16 | cpu | torch.float32 | 0.029 | 0.021 | 15 |
| B=16 m=64 n=64 r=8 | cpu | torch.float64 | 0.035 | 0.028 | 15 |
| B=16 m=64 n=64 r=8 | cpu | torch.float32 | 0.026 | 0.024 | 15 |
| B=1 m=1024 n=1024 r=32 | cpu | torch.float64 | 0.436 | 0.224 | 15 |
| B=1 m=1024 n=1024 r=32 | cpu | torch.float32 | 0.119 | 0.101 | 15 |

## `manifolds.FixedRank.projection`

_Ambient → tangent projection at a point (5 bmms)._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 m=64 n=64 r=8 | cpu | torch.float64 | 0.050 | 0.049 | 15 |
| B=1 m=64 n=64 r=8 | cpu | torch.float32 | 0.044 | 0.043 | 15 |
| B=1 m=256 n=256 r=16 | cpu | torch.float64 | 0.194 | 0.186 | 15 |
| B=1 m=256 n=256 r=16 | cpu | torch.float32 | 0.120 | 0.111 | 15 |
| B=16 m=64 n=64 r=8 | cpu | torch.float64 | 0.159 | 0.151 | 15 |
| B=16 m=64 n=64 r=8 | cpu | torch.float32 | 0.091 | 0.090 | 15 |
| B=1 m=1024 n=1024 r=32 | cpu | torch.float64 | 12.862 | 10.771 | 15 |
| B=1 m=1024 n=1024 r=32 | cpu | torch.float32 | 3.105 | 2.111 | 15 |

## `manifolds.FixedRank.retraction`

_Move-and-project-back retraction via full SVD truncation._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 m=64 n=64 r=8 | cpu | torch.float64 | 0.405 | 0.358 | 15 |
| B=1 m=64 n=64 r=8 | cpu | torch.float32 | 0.339 | 0.301 | 15 |
| B=1 m=256 n=256 r=16 | cpu | torch.float64 | 7.398 | 6.620 | 15 |
| B=1 m=256 n=256 r=16 | cpu | torch.float32 | 7.092 | 4.530 | 15 |
| B=16 m=64 n=64 r=8 | cpu | torch.float64 | 13.366 | 6.138 | 15 |
| B=16 m=64 n=64 r=8 | cpu | torch.float32 | 5.324 | 4.632 | 15 |
| B=1 m=1024 n=1024 r=32 | cpu | torch.float64 | 193.002 | 164.091 | 15 |
| B=1 m=1024 n=1024 r=32 | cpu | torch.float32 | 141.889 | 106.126 | 15 |

## `manifolds.SPD.distance`

_Affine-invariant geodesic distance; two eighs._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.195 | 0.178 | 15 |
| B=1 n=16 | cpu | torch.float32 | 0.159 | 0.143 | 15 |
| B=1 n=64 | cpu | torch.float64 | 0.503 | 0.469 | 15 |
| B=1 n=64 | cpu | torch.float32 | 0.406 | 0.388 | 15 |
| B=16 n=16 | cpu | torch.float64 | 0.547 | 0.535 | 15 |
| B=16 n=16 | cpu | torch.float32 | 0.458 | 0.450 | 15 |
| B=16 n=64 | cpu | torch.float64 | 6.293 | 5.696 | 15 |
| B=16 n=64 | cpu | torch.float32 | 4.495 | 4.034 | 15 |
| B=1 n=256 | cpu | torch.float64 | 9.793 | 7.107 | 15 |
| B=1 n=256 | cpu | torch.float32 | 3.741 | 3.679 | 15 |

## `manifolds.SPD.exp`

_exp_S(V) = S^{1/2} expm(S^{-1/2} V S^{-1/2}) S^{1/2}; one eigh + matrix_exp._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.374 | 0.301 | 15 |
| B=1 n=16 | cpu | torch.float32 | 0.411 | 0.359 | 15 |
| B=1 n=64 | cpu | torch.float64 | 0.820 | 0.692 | 15 |
| B=1 n=64 | cpu | torch.float32 | 0.839 | 0.618 | 15 |
| B=16 n=16 | cpu | torch.float64 | 1.071 | 0.904 | 15 |
| B=16 n=16 | cpu | torch.float32 | 0.947 | 0.776 | 15 |
| B=16 n=64 | cpu | torch.float64 | 8.425 | 6.333 | 15 |
| B=16 n=64 | cpu | torch.float32 | 6.061 | 4.432 | 15 |
| B=1 n=256 | cpu | torch.float64 | 8.426 | 6.651 | 15 |
| B=1 n=256 | cpu | torch.float32 | 4.889 | 4.362 | 15 |

## `manifolds.SPD.inner`

_Affine-invariant inner; two linear solves._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.056 | 0.055 | 15 |
| B=1 n=16 | cpu | torch.float32 | 0.055 | 0.055 | 15 |
| B=1 n=64 | cpu | torch.float64 | 0.158 | 0.156 | 15 |
| B=1 n=64 | cpu | torch.float32 | 0.164 | 0.162 | 15 |
| B=16 n=16 | cpu | torch.float64 | 0.126 | 0.125 | 15 |
| B=16 n=16 | cpu | torch.float32 | 0.143 | 0.143 | 15 |
| B=16 n=64 | cpu | torch.float64 | 1.027 | 1.011 | 15 |
| B=16 n=64 | cpu | torch.float32 | 1.170 | 1.151 | 15 |
| B=1 n=256 | cpu | torch.float64 | 1.725 | 1.669 | 15 |
| B=1 n=256 | cpu | torch.float32 | 1.550 | 1.519 | 15 |

## `manifolds.SPD.log`

_log_S(T); one eigh of S + one eigh of the inner whitened matrix._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.226 | 0.179 | 15 |
| B=1 n=16 | cpu | torch.float32 | 0.294 | 0.227 | 15 |
| B=1 n=64 | cpu | torch.float64 | 0.900 | 0.745 | 15 |
| B=1 n=64 | cpu | torch.float32 | 0.747 | 0.577 | 15 |
| B=16 n=16 | cpu | torch.float64 | 1.136 | 0.978 | 15 |
| B=16 n=16 | cpu | torch.float32 | 0.996 | 0.831 | 15 |
| B=16 n=64 | cpu | torch.float64 | 12.330 | 10.565 | 15 |
| B=16 n=64 | cpu | torch.float32 | 11.771 | 6.988 | 15 |
| B=1 n=256 | cpu | torch.float64 | 13.076 | 9.612 | 15 |
| B=1 n=256 | cpu | torch.float32 | 5.925 | 4.897 | 15 |

## `provenance.chained_inside_record`

_Two-op chain — exercises tensor-id → hex lookup._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.067 | 0.065 | 15 |
| B=1 n=16 | cpu | torch.float32 | 0.064 | 0.064 | 15 |
| B=1 n=128 | cpu | torch.float64 | 0.171 | 0.107 | 15 |
| B=1 n=128 | cpu | torch.float32 | 0.088 | 0.086 | 15 |
| B=16 n=16 | cpu | torch.float64 | 0.083 | 0.077 | 15 |
| B=16 n=16 | cpu | torch.float32 | 0.107 | 0.075 | 15 |

## `provenance.decorator_inside_record`

_Full provenance overhead per call: bind args, hash, register._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.036 | 0.033 | 15 |
| B=1 n=16 | cpu | torch.float32 | 0.034 | 0.032 | 15 |
| B=1 n=128 | cpu | torch.float64 | 0.034 | 0.033 | 15 |
| B=1 n=128 | cpu | torch.float32 | 0.034 | 0.026 | 15 |
| B=16 n=16 | cpu | torch.float64 | 0.028 | 0.027 | 15 |
| B=16 n=16 | cpu | torch.float32 | 0.026 | 0.025 | 15 |

## `provenance.decorator_outside_record`

_Decorator no-op path — should be one attribute lookup._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.002 | 0.001 | 15 |
| B=1 n=16 | cpu | torch.float32 | 0.002 | 0.001 | 15 |
| B=1 n=128 | cpu | torch.float64 | 0.002 | 0.001 | 15 |
| B=1 n=128 | cpu | torch.float32 | 0.002 | 0.001 | 15 |
| B=16 n=16 | cpu | torch.float64 | 0.002 | 0.001 | 15 |
| B=16 n=16 | cpu | torch.float32 | 0.002 | 0.001 | 15 |

## `spectral.laplacian.combinatorial`

_L = D - A; diag_embed + subtract._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.014 | 0.013 | 15 |
| B=1 n=16 | cpu | torch.float32 | 0.013 | 0.013 | 15 |
| B=1 n=64 | cpu | torch.float64 | 0.016 | 0.016 | 15 |
| B=1 n=64 | cpu | torch.float32 | 0.015 | 0.015 | 15 |
| B=1 n=256 | cpu | torch.float64 | 0.046 | 0.041 | 15 |
| B=1 n=256 | cpu | torch.float32 | 0.028 | 0.028 | 15 |
| B=16 n=64 | cpu | torch.float64 | 0.044 | 0.039 | 15 |
| B=16 n=64 | cpu | torch.float32 | 0.029 | 0.029 | 15 |
| B=1 n=1024 | cpu | torch.float64 | 0.268 | 0.233 | 15 |
| B=1 n=1024 | cpu | torch.float32 | 0.107 | 0.105 | 15 |

## `spectral.laplacian.random_walk`

_L_rw = I - D^{-1} A; one broadcast._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.046 | 0.045 | 15 |
| B=1 n=16 | cpu | torch.float32 | 0.036 | 0.035 | 15 |
| B=1 n=64 | cpu | torch.float64 | 0.078 | 0.043 | 15 |
| B=1 n=64 | cpu | torch.float32 | 0.041 | 0.038 | 15 |
| B=1 n=256 | cpu | torch.float64 | 0.086 | 0.080 | 15 |
| B=1 n=256 | cpu | torch.float32 | 0.069 | 0.067 | 15 |
| B=16 n=64 | cpu | torch.float64 | 0.082 | 0.078 | 15 |
| B=16 n=64 | cpu | torch.float32 | 0.082 | 0.068 | 15 |
| B=1 n=1024 | cpu | torch.float64 | 1.895 | 1.133 | 15 |
| B=1 n=1024 | cpu | torch.float32 | 0.242 | 0.202 | 15 |

## `spectral.laplacian.signed`

_L^σ = D^{|σ|} - A; |.|-sum + diag_embed._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.018 | 0.018 | 15 |
| B=1 n=16 | cpu | torch.float32 | 0.017 | 0.017 | 15 |
| B=1 n=64 | cpu | torch.float64 | 0.022 | 0.022 | 15 |
| B=1 n=64 | cpu | torch.float32 | 0.020 | 0.020 | 15 |
| B=1 n=256 | cpu | torch.float64 | 0.070 | 0.065 | 15 |
| B=1 n=256 | cpu | torch.float32 | 0.049 | 0.048 | 15 |
| B=16 n=64 | cpu | torch.float64 | 0.066 | 0.065 | 15 |
| B=16 n=64 | cpu | torch.float32 | 0.049 | 0.048 | 15 |
| B=1 n=1024 | cpu | torch.float64 | 0.336 | 0.297 | 15 |
| B=1 n=1024 | cpu | torch.float32 | 0.156 | 0.147 | 15 |

## `spectral.laplacian.symmetric_normalized`

_L_sym = I - D^{-1/2} A D^{-1/2}; two broadcasts._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.056 | 0.055 | 15 |
| B=1 n=16 | cpu | torch.float32 | 0.053 | 0.053 | 15 |
| B=1 n=64 | cpu | torch.float64 | 0.059 | 0.059 | 15 |
| B=1 n=64 | cpu | torch.float32 | 0.056 | 0.055 | 15 |
| B=1 n=256 | cpu | torch.float64 | 0.097 | 0.093 | 15 |
| B=1 n=256 | cpu | torch.float32 | 0.081 | 0.077 | 15 |
| B=16 n=64 | cpu | torch.float64 | 0.094 | 0.092 | 15 |
| B=16 n=64 | cpu | torch.float32 | 0.078 | 0.077 | 15 |
| B=1 n=1024 | cpu | torch.float64 | 2.112 | 1.858 | 15 |
| B=1 n=1024 | cpu | torch.float32 | 0.444 | 0.242 | 15 |

## `spectral.laplacian_eigenmaps`

_Bottom-k eigenpairs via dense eigh of L_sym._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=64 k=8 | cpu | torch.float64 | 0.368 | 0.314 | 15 |
| B=1 n=64 k=8 | cpu | torch.float32 | 0.287 | 0.259 | 15 |
| B=1 n=256 k=16 | cpu | torch.float64 | 3.890 | 3.356 | 15 |
| B=1 n=256 k=16 | cpu | torch.float32 | 2.266 | 1.843 | 15 |
| B=16 n=64 k=8 | cpu | torch.float64 | 5.105 | 3.787 | 15 |
| B=16 n=64 k=8 | cpu | torch.float32 | 3.548 | 2.676 | 15 |
| B=1 n=1024 k=32 | cpu | torch.float64 | 112.752 | 89.668 | 15 |
| B=1 n=1024 k=32 | cpu | torch.float32 | 60.113 | 51.411 | 15 |

## `tensor_calculus.hosvd`

_Truncated higher-order SVD; d serial SVDs._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 dims=(16, 16, 16) ranks=(4, 4, 4) | cpu | torch.float64 | 0.861 | 0.782 | 15 |
| B=1 dims=(16, 16, 16) ranks=(4, 4, 4) | cpu | torch.float32 | 0.820 | 0.758 | 15 |
| B=1 dims=(32, 32, 32) ranks=(8, 8, 8) | cpu | torch.float64 | 6.816 | 4.165 | 15 |
| B=1 dims=(32, 32, 32) ranks=(8, 8, 8) | cpu | torch.float32 | 4.241 | 3.838 | 15 |
| B=1 dims=(64, 64, 64) ranks=(16, 16, 16) | cpu | torch.float64 | 28.084 | 26.136 | 15 |
| B=1 dims=(64, 64, 64) ranks=(16, 16, 16) | cpu | torch.float32 | 25.103 | 21.385 | 15 |
| B=1 dims=(16, 16, 16, 16) ranks=(4, 4, 4, 4) | cpu | torch.float64 | 8.772 | 7.515 | 15 |
| B=1 dims=(16, 16, 16, 16) ranks=(4, 4, 4, 4) | cpu | torch.float32 | 7.569 | 7.173 | 15 |
| B=4 dims=(32, 32, 32) ranks=(8, 8, 8) | cpu | torch.float64 | 17.552 | 15.574 | 15 |
| B=4 dims=(32, 32, 32) ranks=(8, 8, 8) | cpu | torch.float32 | 15.727 | 14.016 | 15 |

## `tensor_calculus.mode_product`

_n-mode product = movedim + bmm + reshape._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 dims=(16, 16, 16) axis=1 j=8 | cpu | torch.float64 | 0.026 | 0.026 | 15 |
| B=1 dims=(16, 16, 16) axis=1 j=8 | cpu | torch.float32 | 0.027 | 0.027 | 15 |
| B=1 dims=(64, 64, 64) axis=2 j=16 | cpu | torch.float64 | 0.221 | 0.182 | 15 |
| B=1 dims=(64, 64, 64) axis=2 j=16 | cpu | torch.float32 | 0.128 | 0.115 | 15 |
| B=1 dims=(32, 32, 32, 32) axis=2 j=16 | cpu | torch.float64 | 3.272 | 1.133 | 15 |
| B=1 dims=(32, 32, 32, 32) axis=2 j=16 | cpu | torch.float32 | 1.058 | 0.868 | 15 |

## `tensor_calculus.mode_unfolding`

_movedim + reshape; ideally zero copies._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 dims=(16, 16, 16) axis=1 | cpu | torch.float64 | 0.007 | 0.007 | 15 |
| B=1 dims=(16, 16, 16) axis=1 | cpu | torch.float32 | 0.007 | 0.006 | 15 |
| B=1 dims=(32, 32, 32) axis=2 | cpu | torch.float64 | 0.020 | 0.019 | 15 |
| B=1 dims=(32, 32, 32) axis=2 | cpu | torch.float32 | 0.015 | 0.015 | 15 |
| B=1 dims=(64, 64, 64) axis=3 | cpu | torch.float64 | 0.005 | 0.005 | 15 |
| B=1 dims=(64, 64, 64) axis=3 | cpu | torch.float32 | 0.005 | 0.005 | 15 |
| B=8 dims=(32, 32, 32) axis=1 | cpu | torch.float64 | 0.005 | 0.004 | 15 |
| B=8 dims=(32, 32, 32) axis=1 | cpu | torch.float32 | 0.004 | 0.004 | 15 |