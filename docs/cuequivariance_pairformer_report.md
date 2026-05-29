# cuEquivariance PairFormer Integration Report

## Summary

This change keeps the NVIDIA cuEquivariance PairFormer integration behind
explicit feature flags and leaves the default AlphaFold3 PairFormer path
unchanged.

The current implementation is now correctness-first rather than a direct call
to the cuEquivariance high-level triangle multiplication update. In particular,
`CueTriangleMultiplication` has been rewritten to match AlphaFold3's legacy
`TriangleMultiplication` stage by stage under non-zero `final_init='linear'`
coverage. The benchmark does not yet justify enabling either cuEquivariance path
by default.

Update: after installing `triton==3.7.0`, the integration now separates strict
and performance modes. Strict mode is the default and forces the cuEquivariance
gated GEMM helpers through their fallback path for numerical equivalence. Perf
mode can be enabled with `ALPHAFOLD3_CUE_GEMM_MODE=perf` or
`devtools/benchmark_pairformer_cue.py --accuracy-mode perf`; it permits
tile-aligned shapes to use cuEquivariance/Triton gated GEMM kernels and uses a
wider benchmark tolerance.

## Public Configuration

The public PairFormer flags are unchanged and still default to `false`:

```python
PairFormerIteration.Config(
    ...,
    use_cue_triangle_multiplication=False,
    use_cue_triangle_attention=False,
)
```

These defaults are intentional. They should remain disabled until correctness
and representative performance both support enabling them.

## Modified Components

- `src/alphafold3/model/network/modules.py`
  - Keeps lazy imports for optional cuEquivariance JAX dependencies.
  - Adds `_require_cue_triangle_gemm_ops` for cuEquivariance lower-level gated
    GEMM helpers.
  - Adds `_cue_triangle_gemm_fallback` so the helper can use its JAX fallback
    when strict mode is active, `triton` is unavailable, or channel dimensions
    are not compatible with the cuEquivariance/Triton tile requirements.
  - Keeps `CueGridSelfAttention` for optional triangle attention replacement.
  - Rewrites `CueTriangleMultiplication` as a stage-equivalent implementation.
  - Keeps `use_cue_triangle_attention` and
    `use_cue_triangle_multiplication` feature gates.

- `src/alphafold3/model/network/modules_test.py`
  - Adds fake cuEquivariance attention and gated GEMM helpers for deterministic
    CPU tests.
  - Tests `CueTriangleMultiplication` against the legacy implementation for
    outgoing and incoming equations.
  - Tests `float32`, `bfloat16`, full mask, partial mask, forward output, and
    input gradients.
  - Uses `final_init='linear'` in correctness tests so zero initialization
    cannot hide projection or gating errors.
  - Adds non-zero-final-init coverage for `CueGridSelfAttention`.
  - Adds PairFormer-iteration-level comparison with both cuEquivariance flags.
  - Extends the real GPU smoke test for both triangle multiplication equations.

- `devtools/benchmark_pairformer_cue.py`
  - Adds a tracked benchmark for one PairFormer iteration.
  - Compares `legacy`, `attention_only`, `multiplication_only`, and `both`.
  - Reports shape, dtype, flags, warmup, iterations, mean, median, minimum,
    max/mean absolute difference, and speedup.

- `docs/cuequivariance_pairformer_report.md`
  - Records the current correctness and performance status.

## Replacement Status

### PairFormer Triangle Attention

`CueGridSelfAttention` calls `cuequivariance_jax.triangle_attention` for the
attention core and converts between AlphaFold3 and cuEquivariance layouts.

Current status:

- GPU execution: validated.
- Numerical agreement: close to legacy in unit tests and benchmark probes.
  The observed benchmark error is approximately `max_abs <= 0.0033` and
  `mean_abs <= 0.00039` for the tested float32 cases.
- Performance: mixed. It is faster at `num_res=256` in the latest short
  benchmark, but slightly slower at `num_res=64` and `num_res=128`.
- Recommendation: keep the flag but do not enable by default.

### PairFormer Triangle Multiplication

`CueTriangleMultiplication` no longer calls
`cuequivariance_jax.triangle_multiplicative_update` as a whole operation. That
high-level function was not numerically equivalent to AlphaFold3's
`TriangleMultiplication` under non-zero output initialization.

The replacement now follows the legacy stages:

1. Input layer norm: legacy `hm.LayerNorm`.
2. Input projection and input gate: cuEquivariance
   `sigmoid_gated_dual_gemm` when `use_glu_kernel=True`.
3. Triangle update: legacy-equivalent JAX `einsum` with the original outgoing
   and incoming equations.
4. Center layer norm: legacy `hm.LayerNorm(axis=0, param_axis=0)`.
5. Output projection and output gate: cuEquivariance
   `sigmoid_gated_dual_gemm_dual_x` when `use_glu_kernel=True`.

Fallbacks and limitations:

- If `use_glu_kernel=False`, the implementation falls back to the existing
  JAX/Haiku projection and gating logic.
- In strict mode, the cuEquivariance gated GEMM helper is invoked with
  `fallback=True`. This is the default and is intended for correctness.
- In perf mode, tile-aligned channel dimensions can use the Triton
  implementation. Non-aligned dimensions still fall back.
- The triangle `einsum` and layer norms remain legacy/JAX because those stages
  need exact semantic alignment and do not currently have a validated
  lower-level cuEquivariance replacement in this integration.

Current status:

- GPU execution: validated.
- Numerical agreement: validated against legacy for forward output and input
  gradients under `final_init='linear'` in unit tests.
- Performance: workload-dependent. Perf mode can accelerate selected
  module-level triangle multiplication cases, but PairFormer-level speedup is
  not stable.
- Recommendation: keep the flag for continued development, but do not enable by
  default.

## Validation Performed

### Static and Dependency Checks

```bash
git diff --check
conda run -n alphafold3 --no-capture-output python -m pip check
```

Observed result for this revision:

- `git diff --check`: passed.
- `pip check`: `No broken requirements found.`

### Config Golden Check

```bash
conda run -n alphafold3 --no-capture-output python -c "import json, pathlib, run_alphafold; actual=json.dumps(run_alphafold.make_model_config().as_dict(), sort_keys=True, indent=2); expected=pathlib.Path('src/alphafold3/test_data/model_config.json').read_text(); assert actual == expected, 'model_config.json differs'; print('model_config golden matches')"
```

Observed result:

- `model_config golden matches`

### CPU / Default Unit Tests

```bash
conda run -n alphafold3 --no-capture-output python -m pytest src/alphafold3/model/network/modules_test.py
```

Observed result:

- `21 passed, 2 skipped`

The skipped tests are real cuEquivariance GPU tests when the process
environment does not expose the required CUDA library path.

### GPU Unit Tests

The GPU test requires the Python environment's CUDA libraries to take
precedence over system CUDA libraries. Without this, `cuequivariance_jax` can
fail to import because system CUDA libraries may not expose the symbols
required by the cuEquivariance ops package.

Validated command:

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=/home/wangzeli/miniconda3/envs/alphafold3/lib/python3.12/site-packages/nvidia/cublas/lib:/home/wangzeli/miniconda3/envs/alphafold3/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/wangzeli/miniconda3/envs/alphafold3/lib/python3.12/site-packages/nvidia/cusparse/lib:/home/wangzeli/miniconda3/envs/alphafold3/lib/python3.12/site-packages/nvidia/cusolver/lib:/home/wangzeli/miniconda3/envs/alphafold3/lib/python3.12/site-packages/nvidia/nvjitlink/lib \
conda run -n alphafold3 --no-capture-output python -m pytest src/alphafold3/model/network/modules_test.py
```

Observed result before strict/perf mode:

- `23 passed, 2 warnings`

Observed result after strict/perf mode:

- `24 passed, 2 warnings`

GPU used:

- NVIDIA GeForce RTX 4090
- Driver `580.95.05`
- CUDA runtime reported by `nvidia-smi`: `13.0`

## Benchmark

Tracked benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=/home/wangzeli/miniconda3/envs/alphafold3/lib/python3.12/site-packages/nvidia/cublas/lib:/home/wangzeli/miniconda3/envs/alphafold3/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/wangzeli/miniconda3/envs/alphafold3/lib/python3.12/site-packages/nvidia/cusparse/lib:/home/wangzeli/miniconda3/envs/alphafold3/lib/python3.12/site-packages/nvidia/cusolver/lib:/home/wangzeli/miniconda3/envs/alphafold3/lib/python3.12/site-packages/nvidia/nvjitlink/lib \
conda run -n alphafold3 --no-capture-output python devtools/benchmark_pairformer_cue.py --sizes 64,128,256 --warmup 3 --iters 10 --assert-close
```

Benchmark configuration:

- Backend: GPU
- Device: NVIDIA GeForce RTX 4090
- dtype: `float32`
- channels: `128`
- warmup: `3`
- iterations: `10`
- tolerance: `rtol=0.005`, `atol=0.005`
- final init: `linear`

| num_res | Case | Mean ms | Median ms | Min ms | Speedup vs legacy | Max abs diff | Mean abs diff |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | legacy | 4.663 | 4.918 | 2.957 | 1.000x | 0 | 0 |
| 64 | attention_only | 5.012 | 5.222 | 3.540 | 0.930x | 0.00309622 | 0.000382581 |
| 64 | multiplication_only | 5.141 | 5.131 | 4.707 | 0.907x | 0 | 0 |
| 64 | both | 4.860 | 5.186 | 2.874 | 0.959x | 0.00309622 | 0.000382581 |
| 128 | legacy | 4.453 | 4.644 | 3.064 | 1.000x | 0 | 0 |
| 128 | attention_only | 4.634 | 4.886 | 3.055 | 0.961x | 0.003196 | 0.000349217 |
| 128 | multiplication_only | 5.396 | 5.547 | 3.862 | 0.825x | 0 | 0 |
| 128 | both | 4.911 | 5.408 | 3.440 | 0.907x | 0.003196 | 0.000349217 |
| 256 | legacy | 12.426 | 13.535 | 9.035 | 1.000x | 0 | 0 |
| 256 | attention_only | 10.217 | 11.296 | 5.647 | 1.216x | 0.0032208 | 0.000319337 |
| 256 | multiplication_only | 17.842 | 18.633 | 13.280 | 0.696x | 0 | 0 |
| 256 | both | 18.068 | 18.913 | 12.059 | 0.688x | 0.0032208 | 0.000319337 |

Conclusion:

- The benchmark is reproducible and now covers all four flag combinations.
- Triangle attention shows a speedup in this short run only at `num_res=256`.
- Triangle multiplication is numerically exact in this benchmark but slower.
- `attention+multiplication` is slower than legacy because multiplication
  dominates the runtime.
- There is still no stable evidence to enable cuEquivariance by default.

### Strict / Perf Mode Benchmark Update

`devtools/benchmark_pairformer_cue.py` now supports:

```bash
--accuracy-mode strict
--accuracy-mode perf
--module-bench
```

Strict mode uses `rtol=0.005`, `atol=0.005` and keeps the gated GEMM helpers on
the fallback path. Perf mode uses `rtol=0.02`, `atol=0.02` and allows
tile-aligned shapes to use cuEquivariance/Triton gated GEMM kernels.

Latest perf-mode PairFormer-level benchmark, `channels=128`, `warmup=3`,
`iters=10`:

| num_res | Case | Speedup vs legacy | Max abs diff | Mean abs diff |
| ---: | --- | ---: | ---: | ---: |
| 64 | attention_only | 1.111x | 0.00309622 | 0.000382581 |
| 64 | multiplication_only | 2.168x | 0.0118027 | 0.00166148 |
| 64 | both | 0.726x | 0.0117807 | 0.00165845 |
| 128 | attention_only | 0.532x | 0.003196 | 0.000349217 |
| 128 | multiplication_only | 0.544x | 0.0116069 | 0.00164307 |
| 128 | both | 0.352x | 0.0122645 | 0.00164135 |
| 256 | attention_only | 0.717x | 0.0032208 | 0.000319337 |
| 256 | multiplication_only | 0.702x | 0.0136712 | 0.00162794 |
| 256 | both | 0.826x | 0.0133629 | 0.00162695 |

Latest module-level benchmark shows selected local wins but not a stable
PairFormer-level win:

- `triangle_multiplication` incoming direction: about `1.08x` at `num_res=128`
  and `1.12x` at `num_res=256`.
- `grid_self_attention transpose=False`: about `1.50x` at `num_res=128` and
  `1.08x` at `num_res=256`.
- `grid_self_attention transpose=True` remains mixed.

## CUDA Library Path Requirement

Use the NVIDIA libraries from the `alphafold3` conda environment before system
CUDA libraries:

```bash
export LD_LIBRARY_PATH=/home/wangzeli/miniconda3/envs/alphafold3/lib/python3.12/site-packages/nvidia/cublas/lib:/home/wangzeli/miniconda3/envs/alphafold3/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/wangzeli/miniconda3/envs/alphafold3/lib/python3.12/site-packages/nvidia/cusparse/lib:/home/wangzeli/miniconda3/envs/alphafold3/lib/python3.12/site-packages/nvidia/cusolver/lib:/home/wangzeli/miniconda3/envs/alphafold3/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH}
```

For GPU-only tests and benchmarks:

```bash
export CUDA_VISIBLE_DEVICES=0
```

## Known Issues

1. Triangle multiplication is correct in strict mode but only selectively faster
   in perf mode.

   Perf mode introduces larger numerical differences, currently around
   `1e-2` max absolute difference at PairFormer level in the tested shapes.

2. Triangle attention performance is workload-dependent.

   The latest short benchmark shows a win at `num_res=256`, but not at
   `num_res=64` or `num_res=128`. More representative protein sizes, batch
   shapes, and end-to-end inference profiling are still needed.

3. Layer norm and triangle einsum are still legacy/JAX in
   `CueTriangleMultiplication`.

   This is deliberate for correctness. Replacing these stages requires a
   validated lower-level cuEquivariance primitive or an equivalent custom
   kernel.

4. The default model behavior is unchanged.

   Both flags default to `false`, and this should remain true until benchmark
   evidence is stronger.

## Has This Implemented Replacement of PairFormer Modules?

Partially.

The code now supports optional cuEquivariance-backed PairFormer triangle
attention and triangle multiplication paths. Triangle multiplication no longer
uses the non-equivalent high-level cuEquivariance update and is now numerically
aligned with legacy behavior in the tested forward and input-gradient cases.

It is not yet a full production replacement:

- Triangle attention has mixed performance.
- Triangle multiplication uses cuEquivariance only for projection/gating helper
  stages and currently falls back without `triton`.
- The computationally central triangle `einsum` remains legacy/JAX.
- The default PairFormer implementation remains active unless experimental
  flags are explicitly enabled.

## Recommended Next Steps

1. Install and validate a compatible Python `triton` package.

   Re-run the same tests and benchmark without cuEquivariance gated GEMM
   fallback mode. This is the next required step before judging multiplication
   acceleration.

2. Profile kernel timelines.

   Use Nsight Systems or XLA profiling to separate triangle attention,
   projection/gating, layer norm, and triangle `einsum` costs.

3. Evaluate replacing the triangle `einsum`.

   If cuEquivariance exposes a lower-level primitive that matches AlphaFold3's
   outgoing/incoming equations exactly, add stage-level tests before using it.

4. Expand benchmarks.

   Include representative `num_res`, `channels`, dtype, mask sparsity, and full
   model inference settings. Keep compile time excluded and report absolute
   numerical error against legacy.

5. Keep both flags disabled by default.

   Default enablement should require validated correctness, stable speedup, and
   documented CUDA environment setup.
