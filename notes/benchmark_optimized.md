# Benchmark results

_torch 2.12.0+cpu, CUDA available: False_


## `algebra.truncated_svd_exact`

_Full SVD then top-r slice (Eckart-Young exact)._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 m=64 n=64 r=8 | cpu | torch.float64 | 0.474 | 0.448 | 5 |
| B=1 m=256 n=256 r=16 | cpu | torch.float64 | 7.366 | 7.161 | 5 |
| B=1 m=1024 n=1024 r=32 | cpu | torch.float64 | 207.177 | 192.929 | 5 |
| B=1 m=2048 n=64 r=8 | cpu | torch.float64 | 2.034 | 1.900 | 5 |
| B=16 m=128 n=128 r=16 | cpu | torch.float64 | 25.354 | 25.170 | 5 |

## `algebra.truncated_svd_randomized`

_Halko-Martinsson-Tropp randomized SVD with oversample=5, n_iter=2._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 m=64 n=64 r=8 | cpu | torch.float64 | 0.275 | 0.265 | 5 |
| B=1 m=256 n=256 r=16 | cpu | torch.float64 | 1.197 | 1.162 | 5 |
| B=1 m=1024 n=1024 r=32 | cpu | torch.float64 | 6.798 | 6.567 | 5 |
| B=1 m=2048 n=64 r=8 | cpu | torch.float64 | 0.992 | 0.942 | 5 |
| B=16 m=128 n=128 r=16 | cpu | torch.float64 | 5.854 | 5.746 | 5 |

## `discrete_geometry.discrete_ricci_flow`

_Edge-weight evolution: w *= (1 - dt*κ) per step._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 n_steps=5 n_iter=50 | cpu | torch.float64 | 47.366 | 46.372 | 5 |
| B=1 n=32 n_steps=5 n_iter=50 | cpu | torch.float64 | 329.756 | 326.610 | 5 |

## `discrete_geometry.ollivier_ricci_curvature`

_Sinkhorn-based pairwise W_1 with shortest-path metric._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 n_iter=100 | cpu | torch.float64 | 17.955 | 17.857 | 5 |
| B=1 n=32 n_iter=100 | cpu | torch.float64 | 132.660 | 130.711 | 5 |
| B=1 n=64 n_iter=100 | cpu | torch.float64 | 1733.006 | 1709.596 | 5 |
| B=4 n=16 n_iter=100 | cpu | torch.float64 | 40.215 | 37.892 | 5 |

## `discrete_geometry.ricci_flow_with_surgery`

_Flow + periodic threshold-based edge removal._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 n_steps=10 n_iter=50 | cpu | torch.float64 | 96.189 | 93.817 | 5 |
| B=1 n=32 n_steps=10 n_iter=50 | cpu | torch.float64 | 675.355 | 664.901 | 5 |

## `manifolds.FixedRank.inner`

_Frobenius inner product over ambient tangents._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 m=64 n=64 r=8 | cpu | torch.float64 | 0.006 | 0.005 | 5 |
| B=1 m=256 n=256 r=16 | cpu | torch.float64 | 0.024 | 0.023 | 5 |
| B=16 m=64 n=64 r=8 | cpu | torch.float64 | 0.025 | 0.023 | 5 |
| B=1 m=1024 n=1024 r=32 | cpu | torch.float64 | 0.196 | 0.149 | 5 |

## `manifolds.FixedRank.projection`

_Ambient → tangent projection at a point (5 bmms)._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 m=64 n=64 r=8 | cpu | torch.float64 | 0.034 | 0.033 | 5 |
| B=1 m=256 n=256 r=16 | cpu | torch.float64 | 0.164 | 0.157 | 5 |
| B=16 m=64 n=64 r=8 | cpu | torch.float64 | 0.127 | 0.107 | 5 |
| B=1 m=1024 n=1024 r=32 | cpu | torch.float64 | 6.738 | 6.379 | 5 |

## `manifolds.FixedRank.retraction`

_Move-and-project-back retraction via full SVD truncation._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 m=64 n=64 r=8 | cpu | torch.float64 | 0.303 | 0.294 | 5 |
| B=1 m=256 n=256 r=16 | cpu | torch.float64 | 1.274 | 1.246 | 5 |
| B=16 m=64 n=64 r=8 | cpu | torch.float64 | 1.670 | 1.612 | 5 |
| B=1 m=1024 n=1024 r=32 | cpu | torch.float64 | 7.613 | 7.437 | 5 |

## `manifolds.SPD.distance`

_Affine-invariant geodesic distance; two eighs._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.119 | 0.113 | 5 |
| B=1 n=64 | cpu | torch.float64 | 0.377 | 0.366 | 5 |
| B=16 n=16 | cpu | torch.float64 | 0.430 | 0.426 | 5 |
| B=16 n=64 | cpu | torch.float64 | 4.337 | 4.247 | 5 |
| B=1 n=256 | cpu | torch.float64 | 4.915 | 4.836 | 5 |

## `manifolds.SPD.exp`

_exp_S(V) = S^{1/2} expm(S^{-1/2} V S^{-1/2}) S^{1/2}; one eigh + matrix_exp._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.305 | 0.301 | 5 |
| B=1 n=64 | cpu | torch.float64 | 0.679 | 0.645 | 5 |
| B=16 n=16 | cpu | torch.float64 | 0.856 | 0.839 | 5 |
| B=16 n=64 | cpu | torch.float64 | 4.341 | 3.844 | 5 |
| B=1 n=256 | cpu | torch.float64 | 4.951 | 4.731 | 5 |

## `manifolds.SPD.inner`

_Affine-invariant inner; two linear solves._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.044 | 0.041 | 5 |
| B=1 n=64 | cpu | torch.float64 | 0.118 | 0.115 | 5 |
| B=16 n=16 | cpu | torch.float64 | 0.091 | 0.090 | 5 |
| B=16 n=64 | cpu | torch.float64 | 0.728 | 0.669 | 5 |
| B=1 n=256 | cpu | torch.float64 | 1.199 | 1.161 | 5 |

## `manifolds.SPD.log`

_log_S(T); one eigh of S + one eigh of the inner whitened matrix._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.147 | 0.145 | 5 |
| B=1 n=64 | cpu | torch.float64 | 0.560 | 0.520 | 5 |
| B=16 n=16 | cpu | torch.float64 | 0.596 | 0.591 | 5 |
| B=16 n=64 | cpu | torch.float64 | 5.755 | 5.655 | 5 |
| B=1 n=256 | cpu | torch.float64 | 6.361 | 6.287 | 5 |

## `provenance.chained_inside_record`

_Two-op chain — exercises tensor-id → hex lookup._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.073 | 0.068 | 5 |
| B=1 n=128 | cpu | torch.float64 | 0.120 | 0.104 | 5 |
| B=16 n=16 | cpu | torch.float64 | 0.079 | 0.077 | 5 |

## `provenance.decorator_inside_record`

_Full provenance overhead per call: bind args, hash, register._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.032 | 0.030 | 5 |
| B=1 n=128 | cpu | torch.float64 | 0.027 | 0.027 | 5 |
| B=16 n=16 | cpu | torch.float64 | 0.029 | 0.028 | 5 |

## `provenance.decorator_outside_record`

_Decorator no-op path — should be one attribute lookup._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.001 | 0.001 | 5 |
| B=1 n=128 | cpu | torch.float64 | 0.001 | 0.001 | 5 |
| B=16 n=16 | cpu | torch.float64 | 0.001 | 0.001 | 5 |

## `spectral.laplacian.combinatorial`

_L = D - A; diag_embed + subtract._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.011 | 0.010 | 5 |
| B=1 n=64 | cpu | torch.float64 | 0.013 | 0.012 | 5 |
| B=1 n=256 | cpu | torch.float64 | 0.040 | 0.039 | 5 |
| B=16 n=64 | cpu | torch.float64 | 0.044 | 0.042 | 5 |
| B=1 n=1024 | cpu | torch.float64 | 0.240 | 0.190 | 5 |

## `spectral.laplacian.random_walk`

_L_rw = I - D^{-1} A; one broadcast._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.055 | 0.041 | 5 |
| B=1 n=64 | cpu | torch.float64 | 0.046 | 0.040 | 5 |
| B=1 n=256 | cpu | torch.float64 | 0.081 | 0.080 | 5 |
| B=16 n=64 | cpu | torch.float64 | 0.069 | 0.069 | 5 |
| B=1 n=1024 | cpu | torch.float64 | 0.918 | 0.815 | 5 |

## `spectral.laplacian.signed`

_L^σ = D^{|σ|} - A; |.|-sum + diag_embed._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.012 | 0.012 | 5 |
| B=1 n=64 | cpu | torch.float64 | 0.023 | 0.016 | 5 |
| B=1 n=256 | cpu | torch.float64 | 0.054 | 0.054 | 5 |
| B=16 n=64 | cpu | torch.float64 | 0.059 | 0.054 | 5 |
| B=1 n=1024 | cpu | torch.float64 | 0.304 | 0.280 | 5 |

## `spectral.laplacian.symmetric_normalized`

_L_sym = I - D^{-1/2} A D^{-1/2}; two broadcasts._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.042 | 0.041 | 5 |
| B=1 n=64 | cpu | torch.float64 | 0.048 | 0.047 | 5 |
| B=1 n=256 | cpu | torch.float64 | 0.085 | 0.085 | 5 |
| B=16 n=64 | cpu | torch.float64 | 0.085 | 0.084 | 5 |
| B=1 n=1024 | cpu | torch.float64 | 1.817 | 1.738 | 5 |

## `spectral.laplacian_eigenmaps`

_Bottom-k eigenpairs via dense eigh of L_sym._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=64 k=8 | cpu | torch.float64 | 0.343 | 0.319 | 5 |
| B=1 n=256 k=16 | cpu | torch.float64 | 2.697 | 2.575 | 5 |
| B=16 n=64 k=8 | cpu | torch.float64 | 2.718 | 2.592 | 5 |
| B=1 n=1024 k=32 | cpu | torch.float64 | 61.122 | 57.041 | 5 |

## `tensor_calculus.hosvd`

_Truncated higher-order SVD; d serial SVDs._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 dims=(16, 16, 16) ranks=(4, 4, 4) | cpu | torch.float64 | 0.859 | 0.763 | 5 |
| B=1 dims=(32, 32, 32) ranks=(8, 8, 8) | cpu | torch.float64 | 3.140 | 3.070 | 5 |
| B=1 dims=(64, 64, 64) ranks=(16, 16, 16) | cpu | torch.float64 | 18.379 | 17.565 | 5 |
| B=1 dims=(16, 16, 16, 16) ranks=(4, 4, 4, 4) | cpu | torch.float64 | 5.118 | 5.097 | 5 |
| B=4 dims=(32, 32, 32) ranks=(8, 8, 8) | cpu | torch.float64 | 12.030 | 11.674 | 5 |

## `tensor_calculus.mode_product`

_n-mode product = movedim + bmm + reshape._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 dims=(16, 16, 16) axis=1 j=8 | cpu | torch.float64 | 0.060 | 0.033 | 5 |
| B=1 dims=(64, 64, 64) axis=2 j=16 | cpu | torch.float64 | 0.206 | 0.197 | 5 |
| B=1 dims=(32, 32, 32, 32) axis=2 j=16 | cpu | torch.float64 | 1.166 | 1.054 | 5 |

## `tensor_calculus.mode_unfolding`

_movedim + reshape; ideally zero copies._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 dims=(16, 16, 16) axis=1 | cpu | torch.float64 | 0.005 | 0.005 | 5 |
| B=1 dims=(32, 32, 32) axis=2 | cpu | torch.float64 | 0.020 | 0.020 | 5 |
| B=1 dims=(64, 64, 64) axis=3 | cpu | torch.float64 | 0.008 | 0.007 | 5 |
| B=8 dims=(32, 32, 32) axis=1 | cpu | torch.float64 | 0.006 | 0.006 | 5 |