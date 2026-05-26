# Benchmark results

_torch 2.12.0+cu130, CUDA available: False_


## `algebra.dense_eigh_reference`

_torch.linalg.eigh on the SAME symmetric inputs; the cost Lanczos should beat at large n._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=64 | cpu | torch.float64 | 0.233 | 0.212 | 10 |
| B=1 n=256 | cpu | torch.float64 | 3.337 | 2.906 | 10 |
| B=1 n=1024 | cpu | torch.float64 | 80.930 | 70.542 | 10 |
| B=8 n=256 | cpu | torch.float64 | 26.947 | 25.688 | 10 |

## `algebra.lanczos_eigsh`

_Symmetric Lanczos with full reorthogonalization; winning regime is k << n with modest n_iter._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=64 k=1 n_iter=30 | cpu | torch.float64 | 2.826 | 2.286 | 10 |
| B=1 n=256 k=8 n_iter=40 | cpu | torch.float64 | 4.895 | 4.865 | 10 |
| B=1 n=1024 k=16 n_iter=60 | cpu | torch.float64 | 24.469 | 24.345 | 10 |
| B=8 n=256 k=8 n_iter=40 | cpu | torch.float64 | 15.076 | 13.200 | 10 |

## `algebra.lanczos_eigsh_SA`

_Shift-and-invert Lanczos for smallest eigenvalues. LU-factor (A-σI) once, then per-iter lu_solve. Compare cost vs `lanczos_eigsh` (LA) and dense `eigvalsh`._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=64 k=1 n_iter=20 sigma=0.0 | cpu | torch.float64 | 2.563 | 2.542 | 10 |
| B=1 n=256 k=8 n_iter=30 sigma=0.0 | cpu | torch.float64 | 4.771 | 4.707 | 10 |
| B=1 n=1024 k=16 n_iter=40 sigma=0.0 | cpu | torch.float64 | 18.902 | 17.916 | 10 |

## `algebra.truncated_svd_exact`

_Full SVD then top-r slice (Eckart-Young exact)._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 m=64 n=64 r=8 | cpu | torch.float64 | 0.565 | 0.543 | 10 |
| B=1 m=256 n=256 r=16 | cpu | torch.float64 | 8.895 | 8.478 | 10 |
| B=1 m=1024 n=1024 r=32 | cpu | torch.float64 | 249.809 | 229.980 | 10 |
| B=1 m=2048 n=64 r=8 | cpu | torch.float64 | 2.258 | 2.181 | 10 |
| B=16 m=128 n=128 r=16 | cpu | torch.float64 | 32.291 | 29.100 | 10 |

## `algebra.truncated_svd_randomized`

_Halko-Martinsson-Tropp randomized SVD with oversample=5, n_iter=2._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 m=64 n=64 r=8 | cpu | torch.float64 | 0.350 | 0.337 | 10 |
| B=1 m=256 n=256 r=16 | cpu | torch.float64 | 1.425 | 1.403 | 10 |
| B=1 m=1024 n=1024 r=32 | cpu | torch.float64 | 9.153 | 8.095 | 10 |
| B=1 m=2048 n=64 r=8 | cpu | torch.float64 | 1.171 | 1.037 | 10 |
| B=16 m=128 n=128 r=16 | cpu | torch.float64 | 7.829 | 7.306 | 10 |# Benchmark results

_torch 2.12.0+cu130, CUDA available: False_


## `spectral.diffusion_map`

_Coifman-Lafon embedding; dominated by the eigh in laplacian_eigenmaps._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=64 k=8 t=1.0 | cpu | torch.float64 | 0.478 | 0.469 | 10 |
| B=1 n=256 k=16 t=1.0 | cpu | torch.float64 | 3.716 | 3.606 | 10 |
| B=1 n=1024 k=32 t=1.0 | cpu | torch.float64 | 91.450 | 75.453 | 10 |

## `spectral.effective_resistance`

_Pairwise R via Moore-Penrose pseudoinverse; dense eigh + outer product._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=32 | cpu | torch.float64 | 0.255 | 0.249 | 10 |
| B=1 n=128 | cpu | torch.float64 | 1.046 | 1.020 | 10 |
| B=1 n=512 | cpu | torch.float64 | 17.001 | 16.478 | 10 |
| B=8 n=64 | cpu | torch.float64 | 2.029 | 2.006 | 10 |

## `spectral.heat_kernel_chebyshev_dense`

_exp(-t·L) as a dense (B, n, n) tensor via Chebyshev._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=32 t=1.0 K=30 | cpu | torch.float64 | 0.755 | 0.744 | 10 |
| B=1 n=128 t=1.0 K=30 | cpu | torch.float64 | 2.393 | 2.328 | 10 |
| B=1 n=512 t=1.0 K=30 | cpu | torch.float64 | 65.600 | 60.222 | 10 |
| B=1 n=128 t=1.0 K=60 | cpu | torch.float64 | 4.922 | 4.812 | 10 |

## `spectral.heat_kernel_chebyshev_signal`

_exp(-t·L) @ signal via the same Chebyshev recurrence; should beat the dense path when k_signal << n._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=128 k_signal=4 t=1.0 K=30 | cpu | torch.float64 | 0.852 | 0.834 | 10 |
| B=1 n=512 k_signal=4 t=1.0 K=30 | cpu | torch.float64 | 2.968 | 2.930 | 10 |
| B=1 n=512 k_signal=64 t=1.0 K=30 | cpu | torch.float64 | 11.316 | 11.175 | 10 |

## `spectral.laplacian.combinatorial`

_L = D - A; diag_embed + subtract._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.055 | 0.054 | 10 |
| B=1 n=64 | cpu | torch.float64 | 0.073 | 0.072 | 10 |
| B=1 n=256 | cpu | torch.float64 | 0.245 | 0.228 | 10 |
| B=16 n=64 | cpu | torch.float64 | 0.219 | 0.208 | 10 |
| B=1 n=1024 | cpu | torch.float64 | 14.286 | 11.382 | 10 |

## `spectral.laplacian.combinatorial_sparse`

_Sparse-COO combinatorial Laplacian; expected to win vs dense at n ≥ ~1000._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| n=256 density=0.05 | cpu | torch.float64 | 0.208 | 0.202 | 10 |
| n=1024 density=0.01 | cpu | torch.float64 | 0.225 | 0.222 | 10 |
| n=4096 density=0.003 | cpu | torch.float64 | 0.303 | 0.292 | 10 |

## `spectral.laplacian.random_walk`

_L_rw = I - D^{-1} A; one broadcast._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.154 | 0.120 | 10 |
| B=1 n=64 | cpu | torch.float64 | 0.193 | 0.187 | 10 |
| B=1 n=256 | cpu | torch.float64 | 0.433 | 0.365 | 10 |
| B=16 n=64 | cpu | torch.float64 | 0.414 | 0.389 | 10 |
| B=1 n=1024 | cpu | torch.float64 | 8.173 | 6.146 | 10 |

## `spectral.laplacian.signed`

_L^σ = D^{|σ|} - A; |.|-sum + diag_embed._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.107 | 0.089 | 10 |
| B=1 n=64 | cpu | torch.float64 | 0.121 | 0.083 | 10 |
| B=1 n=256 | cpu | torch.float64 | 0.248 | 0.235 | 10 |
| B=16 n=64 | cpu | torch.float64 | 0.238 | 0.223 | 10 |
| B=1 n=1024 | cpu | torch.float64 | 3.403 | 2.615 | 10 |

## `spectral.laplacian.symmetric_normalized`

_L_sym = I - D^{-1/2} A D^{-1/2}; two broadcasts._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.120 | 0.106 | 10 |
| B=1 n=64 | cpu | torch.float64 | 0.207 | 0.197 | 10 |
| B=1 n=256 | cpu | torch.float64 | 0.462 | 0.413 | 10 |
| B=16 n=64 | cpu | torch.float64 | 0.434 | 0.358 | 10 |
| B=1 n=1024 | cpu | torch.float64 | 9.297 | 7.137 | 10 |

## `spectral.laplacian.symmetric_normalized_sparse`

_Sparse-COO L_sym; same crossover pattern._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| n=256 density=0.05 | cpu | torch.float64 | 0.223 | 0.213 | 10 |
| n=1024 density=0.01 | cpu | torch.float64 | 0.241 | 0.231 | 10 |
| n=4096 density=0.003 | cpu | torch.float64 | 0.443 | 0.434 | 10 |

## `spectral.laplacian_eigenmaps`

_Bottom-k eigenpairs via dense eigh of L_sym._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=64 k=8 | cpu | torch.float64 | 0.402 | 0.366 | 10 |
| B=1 n=256 k=16 | cpu | torch.float64 | 4.590 | 3.878 | 10 |
| B=16 n=64 k=8 | cpu | torch.float64 | 4.980 | 3.699 | 10 |
| B=1 n=1024 k=32 | cpu | torch.float64 | 106.554 | 87.722 | 10 |

## `spectral.magnetic.combinatorial`

_Hermitian magnetic Laplacian for directed graphs; complex output._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 q=0.25 | cpu | torch.float64 | 0.087 | 0.084 | 10 |
| B=1 n=64 q=0.25 | cpu | torch.float64 | 0.127 | 0.125 | 10 |
| B=1 n=256 q=0.25 | cpu | torch.float64 | 0.490 | 0.473 | 10 |
| B=16 n=64 q=0.25 | cpu | torch.float64 | 0.430 | 0.418 | 10 |
| B=1 n=256 q=0.0 | cpu | torch.float64 | 0.235 | 0.230 | 10 |

## `spectral.magnetic.sign_magnetic_combinatorial`

_Signed-directed Hermitian Laplacian; one extra abs() vs plain magnetic._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 q=0.25 | cpu | torch.float64 | 0.098 | 0.095 | 10 |
| B=1 n=64 q=0.25 | cpu | torch.float64 | 0.141 | 0.140 | 10 |
| B=1 n=256 q=0.25 | cpu | torch.float64 | 0.578 | 0.530 | 10 |
| B=16 n=64 q=0.25 | cpu | torch.float64 | 0.484 | 0.474 | 10 |
| B=1 n=256 q=0.0 | cpu | torch.float64 | 0.307 | 0.298 | 10 |

## `spectral.magnetic.sign_magnetic_symmetric_normalized`

_Normalized signed-directed form._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 q=0.25 | cpu | torch.float64 | 0.125 | 0.123 | 10 |
| B=1 n=64 q=0.25 | cpu | torch.float64 | 0.171 | 0.169 | 10 |
| B=1 n=256 q=0.25 | cpu | torch.float64 | 0.626 | 0.605 | 10 |
| B=16 n=64 q=0.25 | cpu | torch.float64 | 0.575 | 0.530 | 10 |
| B=1 n=256 q=0.0 | cpu | torch.float64 | 0.386 | 0.363 | 10 |

## `spectral.magnetic.symmetric_normalized`

_Normalized magnetic Laplacian; spectrum in [0, 2]._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 q=0.25 | cpu | torch.float64 | 0.115 | 0.113 | 10 |
| B=1 n=64 q=0.25 | cpu | torch.float64 | 0.158 | 0.155 | 10 |
| B=1 n=256 q=0.25 | cpu | torch.float64 | 0.536 | 0.518 | 10 |
| B=16 n=64 q=0.25 | cpu | torch.float64 | 0.487 | 0.468 | 10 |
| B=1 n=256 q=0.0 | cpu | torch.float64 | 0.299 | 0.292 | 10 |# Benchmark results

_torch 2.12.0+cu130, CUDA available: False_


## `info_geometry.bregman_divergence_squared_euclidean`

_Bregman with F(x) = (1/2)||x||^2; sum-of-squares cost._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 d=16 | cpu | torch.float64 | 0.022 | 0.020 | 10 |
| B=1 d=256 | cpu | torch.float64 | 0.021 | 0.020 | 10 |
| B=1 d=4096 | cpu | torch.float64 | 0.025 | 0.025 | 10 |
| B=256 d=64 | cpu | torch.float64 | 0.046 | 0.045 | 10 |

## `info_geometry.fisher_information_categorical`

_Diagonal Fisher on the simplex; clamp + reciprocal + diag_embed._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 k=16 | cpu | torch.float64 | 0.011 | 0.011 | 10 |
| B=1 k=256 | cpu | torch.float64 | 0.023 | 0.023 | 10 |
| B=1024 k=64 | cpu | torch.float64 | 6.846 | 5.994 | 10 |

## `info_geometry.fisher_information_gaussian_mean`

_Sigma^{-1} via Cholesky; same path as kl_divergence_gaussian._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 d=16 | cpu | torch.float64 | 0.052 | 0.050 | 10 |
| B=1 d=64 | cpu | torch.float64 | 0.088 | 0.086 | 10 |
| B=1 d=256 | cpu | torch.float64 | 0.844 | 0.796 | 10 |
| B=16 d=64 | cpu | torch.float64 | 0.600 | 0.569 | 10 |

## `info_geometry.kl_divergence_categorical`

_Discrete KL with the torch.where guard for the q=0 case._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 k=16 | cpu | torch.float64 | 0.041 | 0.040 | 10 |
| B=1 k=256 | cpu | torch.float64 | 0.044 | 0.044 | 10 |
| B=1024 k=64 | cpu | torch.float64 | 1.312 | 1.189 | 10 |

## `info_geometry.kl_divergence_gaussian`

_Gaussian KL with Cholesky-based solves; dominated by O(B*d^3)._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 d=4 | cpu | torch.float64 | 0.124 | 0.120 | 10 |
| B=1 d=16 | cpu | torch.float64 | 0.129 | 0.126 | 10 |
| B=1 d=64 | cpu | torch.float64 | 0.203 | 0.185 | 10 |
| B=16 d=16 | cpu | torch.float64 | 0.187 | 0.185 | 10 |
| B=1 d=256 | cpu | torch.float64 | 1.128 | 1.097 | 10 |

## `info_geometry.natural_gradient`

_F^{-1} grad via torch.linalg.solve (no explicit inverse)._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 d=16 | cpu | torch.float64 | 0.035 | 0.033 | 10 |
| B=1 d=64 | cpu | torch.float64 | 0.064 | 0.064 | 10 |
| B=1 d=256 | cpu | torch.float64 | 0.336 | 0.322 | 10 |
| B=16 d=64 | cpu | torch.float64 | 0.224 | 0.206 | 10 |# Benchmark results

_torch 2.12.0+cu130, CUDA available: False_


## `topology.persistence_diagrams_python`

_VR + Z/2 reduction with CPython set columns. Reference for the torch-backend crossover._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| n=20 max_dim=1 max_radius=1.5 | cpu | torch.float64 | 5.520 | 4.954 | 5 |
| n=40 max_dim=1 max_radius=1.0 | cpu | torch.float64 | 15.101 | 13.151 | 5 |
| n=80 max_dim=1 max_radius=0.7 | cpu | torch.float64 | 43.011 | 31.734 | 5 |

## `topology.persistence_diagrams_torch`

_VR + Z/2 reduction with torch.LongTensor columns. Device-agnostic; not yet a CUDA kernel — same algorithm, expect slower than CPython sets for small inputs._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| n=20 max_dim=1 max_radius=1.5 | cpu | torch.float64 | 38.993 | 35.793 | 5 |
| n=40 max_dim=1 max_radius=1.0 | cpu | torch.float64 | 137.148 | 134.943 | 5 |
| n=80 max_dim=1 max_radius=0.7 | cpu | torch.float64 | 903.335 | 653.160 | 5 |