# cuEquivariance PairFormer Changes Summary

## Overview

This change adds optional NVIDIA cuEquivariance-backed execution paths for
selected AlphaFold3 PairFormer triangle operations while keeping the default
model behavior unchanged.

The implementation is intentionally feature-gated. The original PairFormer path
is still used unless these flags are explicitly enabled:

```python
PairFormerIteration.Config(
    use_cue_triangle_multiplication=True,
    use_cue_triangle_attention=True,
)
```

Both flags remain `False` by default.

## Modified Files

- `src/alphafold3/model/network/modules.py`
  - Adds lazy cuEquivariance imports and clearer dependency errors.
  - Adds `CueGridSelfAttention` as an optional triangle attention adapter.
  - Rewrites `CueTriangleMultiplication` as a stage-equivalent adapter instead
    of calling cuEquivariance's high-level triangle multiplication update.
  - Adds shape-aware and mode-aware fallback for cuEquivariance gated GEMM
    kernels.
  - Adds `use_cue_triangle_multiplication` and
    `use_cue_triangle_attention` feature flags to `PairFormerIteration.Config`.

- `src/alphafold3/model/network/modules_test.py`
  - Adds deterministic fake cuEquivariance helpers for CPU-compatible tests.
  - Adds numerical comparisons against legacy TriangleMultiplication and
    GridSelfAttention.
  - Covers outgoing/incoming triangle multiplication, `float32`, `bfloat16`,
    full/partial masks, non-zero `final_init='linear'`, forward outputs, and
    input gradients.
  - Adds GPU smoke coverage for real cuEquivariance triangle multiplication.
  - Adds tests for the strict/perf gated GEMM fallback policy.

- `devtools/benchmark_pairformer_cue.py`
  - Adds a tracked GPU benchmark for one PairFormer iteration.
  - Compares `legacy`, `attention_only`, `multiplication_only`, and `both`.
  - Adds optional module-level benchmarks for triangle multiplication and grid
    self-attention.
  - Reports mean, median, minimum, speedup, max absolute difference, and mean
    absolute difference.
  - Supports `--accuracy-mode strict` and `--accuracy-mode perf`.

- `docs/cuequivariance_pairformer_report.md`
  - Records the implementation status, validation commands, CUDA library path
    requirements, benchmark results, limitations, and recommended next steps.

- `src/alphafold3/.gitignore`
  - Ignores Python bytecode/cache artifacts.
  - Ignores generated converter pickle payloads without ignoring converter
    source scripts.

- `src/alphafold3/constants/converters/`
  - Adds converter helper scripts for generating CCD-related pickle payloads.
  - Generated pickle outputs remain ignored.

## Triangle Attention Acceleration

The optional attention path replaces the attention core inside
`GridSelfAttention` with:

```python
cuequivariance_jax.triangle_attention(...)
```

The adapter keeps the surrounding AlphaFold3 logic intact:

1. AlphaFold3 still creates Q/K/V, bias, gating, and output projections.
2. `_cue_triangle_attention_inputs` converts AlphaFold3 tensor layouts to the
   cuEquivariance triangle attention layout.
3. `cuequivariance_jax.triangle_attention` computes the attention core.
4. `_cue_triangle_attention_output` converts the output back to AlphaFold3's
   layout.
5. The original gate and output projection logic continues to run.

This means the cuEquivariance replacement is scoped to the attention core, not
the entire attention block.

## Triangle Multiplication Acceleration

The implementation does not use cuEquivariance's high-level
`triangle_multiplicative_update` because that path was not numerically
equivalent to AlphaFold3's legacy implementation under non-zero
`final_init='linear'`.

Instead, `CueTriangleMultiplication` follows the legacy computation stage by
stage:

1. Input layer norm uses AlphaFold3 `hm.LayerNorm`.
2. Input projection and input gate use cuEquivariance lower-level
   `sigmoid_gated_dual_gemm` when enabled by the fallback policy.
3. The triangle update itself remains the legacy-equivalent JAX `einsum`.
4. Center layer norm uses AlphaFold3 `hm.LayerNorm(axis=0, param_axis=0)`.
5. Output projection and output gate use cuEquivariance lower-level
   `sigmoid_gated_dual_gemm_dual_x` when enabled by the fallback policy.

The current cuEquivariance acceleration is therefore focused on the gated GEMM
projection stages. The central triangle `einsum` is still JAX/legacy.

## Strict and Perf Modes

The gated GEMM path is controlled by `ALPHAFOLD3_CUE_GEMM_MODE`:

- `strict` is the default. It forces fallback for cuEquivariance gated GEMM
  helpers and prioritizes numerical equivalence.
- `perf` allows tile-aligned shapes to use cuEquivariance/Triton gated GEMM
  kernels.

Even in `perf` mode, the implementation falls back if the channel dimensions do
not satisfy the cuEquivariance Triton tile requirements:

```python
input_channels % 32 == 0
output_channels % 32 == 0
```

The benchmark script maps `--accuracy-mode strict` and `--accuracy-mode perf`
to this environment variable.

## Validation Summary

Validated GPU command shape:

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=/home/wangzeli/miniconda3/envs/alphafold3/lib/python3.12/site-packages/nvidia/cublas/lib:/home/wangzeli/miniconda3/envs/alphafold3/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/wangzeli/miniconda3/envs/alphafold3/lib/python3.12/site-packages/nvidia/cusparse/lib:/home/wangzeli/miniconda3/envs/alphafold3/lib/python3.12/site-packages/nvidia/cusolver/lib:/home/wangzeli/miniconda3/envs/alphafold3/lib/python3.12/site-packages/nvidia/nvjitlink/lib \
conda run -n alphafold3 --no-capture-output python -m pytest src/alphafold3/model/network/modules_test.py
```

Latest observed GPU unit test result:

- `24 passed, 2 warnings`

Strict benchmark uses `rtol=0.005`, `atol=0.005`.
Perf benchmark uses `rtol=0.02`, `atol=0.02`.

## Current Performance Interpretation

The implementation is usable for controlled experiments, but it is not ready to
be enabled by default.

Observed tendencies:

- Triangle attention can accelerate some shapes, especially selected
  `transpose=False` module-level cases.
- Triangle multiplication incoming direction shows some module-level speedup in
  selected shapes.
- PairFormer-level performance remains inconsistent.
- Perf mode introduces larger numerical differences than strict mode.

The current recommendation is to keep both feature flags disabled by default and
use the benchmark script to evaluate shape-specific selective enablement.

## Remaining Work

1. Stabilize benchmarks with longer warmup and iteration counts.
2. Separate compile/autotune time from steady-state runtime.
3. Profile individual stages with Nsight Systems or XLA profiling.
4. Investigate a validated cuEquivariance or custom-kernel replacement for the
   central triangle `einsum`.
5. Add representative `bfloat16`, larger `num_res`, and full inference tests.
6. Keep default PairFormer behavior unchanged until correctness and performance
   are both proven.
