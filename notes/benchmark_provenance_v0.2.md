# Benchmark results

_torch 2.12.0+cu130, CUDA available: False_


## `provenance.chained_inside_record`

_Two-op chain — exercises tensor-id → hex lookup._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.118 | 0.108 | 30 |
| B=1 n=128 | cpu | torch.float64 | 0.202 | 0.196 | 30 |
| B=16 n=16 | cpu | torch.float64 | 0.141 | 0.134 | 30 |

## `provenance.decorator_inside_record`

_Full provenance overhead per call: bind args, hash, register._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.026 | 0.024 | 30 |
| B=1 n=128 | cpu | torch.float64 | 0.025 | 0.024 | 30 |
| B=16 n=16 | cpu | torch.float64 | 0.026 | 0.025 | 30 |

## `provenance.decorator_outside_record`

_Decorator no-op path — should be one attribute lookup._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.001 | 0.001 | 30 |
| B=1 n=128 | cpu | torch.float64 | 0.001 | 0.001 | 30 |
| B=16 n=16 | cpu | torch.float64 | 0.002 | 0.001 | 30 |

## `provenance.hashing_big_tensor`

_Fresh-tensor identity, isolates _tensor_content_hex byte cost. Phase 1b sketch-hashing should flatten this curve._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.036 | 0.034 | 30 |
| B=1 n=256 | cpu | torch.float64 | 0.166 | 0.162 | 30 |
| B=1 n=1024 | cpu | torch.float64 | 6.806 | 6.085 | 30 |

## `provenance.recording_with_cache`

_laplacian → identity chain with cache_tensors=True. Baseline for phase 1c disk-cache comparison._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.134 | 0.110 | 30 |
| B=1 n=256 | cpu | torch.float64 | 0.418 | 0.404 | 30 |
| B=1 n=1024 | cpu | torch.float64 | 9.333 | 7.789 | 30 |

## `provenance.replay_baseline`

_Direct call to the affected op with the substituted tensor. The math floor that replay overhead sits on._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.008 | 0.007 | 30 |
| B=1 n=256 | cpu | torch.float64 | 0.022 | 0.021 | 30 |
| B=1 n=1024 | cpu | torch.float64 | 0.086 | 0.067 | 30 |

## `provenance.replay_overhead`

_replay() of a 2-op chain after substituting the middle node — DAG walk + topo-sort + single-op dispatch._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.013 | 0.012 | 30 |
| B=1 n=256 | cpu | torch.float64 | 0.032 | 0.024 | 30 |
| B=1 n=1024 | cpu | torch.float64 | 0.116 | 0.085 | 30 |