# Benchmark results

_torch 2.12.0+cu130, CUDA available: False_


## `provenance.chained_inside_record`

_Two-op chain — exercises tensor-id → hex lookup._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.108 | 0.106 | 30 |
| B=1 n=128 | cpu | torch.float64 | 0.184 | 0.182 | 30 |
| B=16 n=16 | cpu | torch.float64 | 0.127 | 0.126 | 30 |

## `provenance.decorator_inside_record`

_Full provenance overhead per call: bind args, hash, register._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.025 | 0.023 | 30 |
| B=1 n=128 | cpu | torch.float64 | 0.025 | 0.024 | 30 |
| B=16 n=16 | cpu | torch.float64 | 0.026 | 0.026 | 30 |

## `provenance.decorator_outside_record`

_Decorator no-op path — should be one attribute lookup._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.001 | 0.001 | 30 |
| B=1 n=128 | cpu | torch.float64 | 0.001 | 0.001 | 30 |
| B=16 n=16 | cpu | torch.float64 | 0.001 | 0.001 | 30 |

## `provenance.hashing_big_tensor`

_Fresh-tensor identity, isolates _tensor_content_hex byte cost. Phase 1b sketch-hashing should flatten this curve._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.034 | 0.033 | 30 |
| B=1 n=256 | cpu | torch.float64 | 0.161 | 0.156 | 30 |
| B=1 n=1024 | cpu | torch.float64 | 6.272 | 5.271 | 30 |

## `provenance.hashing_big_tensor_sketch`

_Sketch-mode hashing — compare against hashing_big_tensor (full mode) at matching sizes; crossover ~n=256._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.099 | 0.094 | 30 |
| B=1 n=256 | cpu | torch.float64 | 0.164 | 0.148 | 30 |
| B=1 n=1024 | cpu | torch.float64 | 0.350 | 0.337 | 30 |

## `provenance.recording_with_cache`

_laplacian → identity chain with cache_tensors=True. Baseline for phase 1c disk-cache comparison._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.113 | 0.106 | 30 |
| B=1 n=256 | cpu | torch.float64 | 0.461 | 0.392 | 30 |
| B=1 n=1024 | cpu | torch.float64 | 8.445 | 6.876 | 30 |

## `provenance.recording_with_disk_cache`

_Same chain as recording_with_cache but with cache_to_disk set — measures torch.save overhead per output._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.580 | 0.526 | 30 |
| B=1 n=256 | cpu | torch.float64 | 1.023 | 0.970 | 30 |
| B=1 n=1024 | cpu | torch.float64 | 19.139 | 17.811 | 30 |

## `provenance.replay_baseline`

_Direct call to the affected op with the substituted tensor. The math floor that replay overhead sits on._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.005 | 0.005 | 30 |
| B=1 n=256 | cpu | torch.float64 | 0.016 | 0.015 | 30 |
| B=1 n=1024 | cpu | torch.float64 | 0.068 | 0.063 | 30 |

## `provenance.replay_overhead`

_replay() of a 2-op chain after substituting the middle node — DAG walk + topo-sort + single-op dispatch._

| size | device | dtype | median (ms) | min (ms) | iters |
|---|---|---|---:|---:|---:|
| B=1 n=16 | cpu | torch.float64 | 0.012 | 0.012 | 30 |
| B=1 n=256 | cpu | torch.float64 | 0.028 | 0.024 | 30 |
| B=1 n=1024 | cpu | torch.float64 | 0.094 | 0.071 | 30 |