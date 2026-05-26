# cuEquivariance PairFormer Integration Report

## Summary

This change adds an optional NVIDIA cuEquivariance-backed execution path for
selected AlphaFold3 PairFormer triangle operations. The implementation is
feature-gated and leaves the default PairFormer path unchanged.

The current implementation should be treated as an experimental integration,
not as a complete production replacement for PairFormer. GPU smoke tests pass
when the correct CUDA library path is used, but performance testing does not
show a stable PairFormer-level speedup, and the triangle multiplication
replacement still has correctness gaps under non-zero output initialization.

## Modified Components

- `pyproject.toml`
  - Adds `cuequivariance-jax==0.10.0`.
  - Adds `cuequivariance-ops-jax-cu12==0.10.0`.

- `uv.lock`
  - Locks cuEquivariance and related NVIDIA CUDA Python wheel dependencies.

- `src/alphafold3/model/network/modules.py`
  - Adds lazy loading helper `_require_cuequivariance_jax`.
  - Adds `CueGridSelfAttention`.
  - Adds `CueTriangleMultiplication`.
  - Adds `PairFormerIteration.Config.use_cue_triangle_attention`.
  - Adds `PairFormerIteration.Config.use_cue_triangle_multiplication`.
  - Switches PairFormer attention/multiplication module classes based on the
    new flags.

- `src/alphafold3/test_data/model_config.json`
  - Adds the new config fields with default value `false`.

- `src/alphafold3/model/network/modules_test.py`
  - Adds tests for default flag behavior.
  - Adds tests that the default path does not require cuEquivariance.
  - Adds layout tests using a fake cuEquivariance module.
  - Adds dependency error-message coverage.
  - Adds a real GPU smoke test for cuEquivariance triangle multiplication when
    CUDA ops are available.

## Replacement Status

### PairFormer Attention

`GridSelfAttention` can be optionally replaced with `CueGridSelfAttention` by
setting:

```python
PairFormerIteration.Config(
    ...,
    use_cue_triangle_attention=True,
)
```

This path calls `cuequivariance_jax.triangle_attention` for the attention core.
The adapter handles layout conversion between AlphaFold3's existing attention
layout and cuEquivariance's triangle attention layout.

Current status:

- GPU execution: validated.
- Numerical agreement: close to the existing implementation in smoke tests and
  benchmark probes.
- PairFormer-level speedup: not demonstrated consistently.

### PairFormer Triangle Multiplication

`TriangleMultiplication` can be optionally replaced with
`CueTriangleMultiplication` by setting:

```python
PairFormerIteration.Config(
    ...,
    use_cue_triangle_multiplication=True,
)
```

This path calls `cuequivariance_jax.triangle_multiplicative_update`.

Current status:

- GPU execution: validated for the smoke-test case.
- Numerical agreement: incomplete. The current unit smoke test passes under the
  test configuration, but PairFormer-level probes with non-zero
  `final_init='linear'` show large output differences against the existing
  AlphaFold3 implementation.
- PairFormer-level speedup: not valid to claim while correctness remains
  unresolved.

## Validation Performed

### Static and Dependency Checks

```bash
git diff --check
conda run -n alphafold3 --no-capture-output python -m pip check
```

Results:

- `git diff --check`: passed.
- `pip check`: `No broken requirements found.`

### Config Golden Check

```bash
conda run -n alphafold3 --no-capture-output python -c "import json, pathlib, run_alphafold; actual=json.dumps(run_alphafold.make_model_config().as_dict(), sort_keys=True, indent=2); expected=pathlib.Path('src/alphafold3/test_data/model_config.json').read_text(); assert actual == expected, 'model_config.json differs'; print('model_config golden matches')"
```

Result:

- `model_config golden matches`

### CPU / Default Environment Unit Tests

```bash
conda run -n alphafold3 --no-capture-output python -m pytest src/alphafold3/model/network/modules_test.py
```

Result:

- `10 passed, 1 skipped`

The skipped test is the real cuEquivariance GPU smoke test when the normal
process environment does not expose the required CUDA library path.

### GPU Unit Tests

The GPU test requires the Python environment's CUDA libraries to take
precedence over system CUDA libraries. Without this, `cuequivariance_jax` fails
to import because system `libcublas.so.12` does not provide
`cublasGemmGroupedBatchedEx`.

Validated command:

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=/home/wangzeli/miniconda3/envs/alphafold3/lib/python3.12/site-packages/nvidia/cublas/lib:/home/wangzeli/miniconda3/envs/alphafold3/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/wangzeli/miniconda3/envs/alphafold3/lib/python3.12/site-packages/nvidia/cusparse/lib:/home/wangzeli/miniconda3/envs/alphafold3/lib/python3.12/site-packages/nvidia/cusolver/lib:/home/wangzeli/miniconda3/envs/alphafold3/lib/python3.12/site-packages/nvidia/nvjitlink/lib \
conda run -n alphafold3 --no-capture-output python -m pytest src/alphafold3/model/network/modules_test.py
```

Result:

- `11 passed, 1 warning`

GPU used:

- NVIDIA GeForce RTX 4090
- Driver `580.95.05`
- CUDA runtime reported by `nvidia-smi`: `13.0`

## Performance Findings

The benchmark used one JIT-compiled PairFormer iteration, float32 activations,
`channels=128`, GPU 0, and `block_until_ready()` timing after warmup.

Only `use_cue_triangle_attention=True` was benchmarked as numerically acceptable.
`use_cue_triangle_multiplication=True` was excluded from speedup claims because
it showed large correctness differences.

| num_res | Legacy Mean | cuEq Attention Mean | Mean Speedup |
| ---: | ---: | ---: | ---: |
| 64 | 0.467 ms | 0.444 ms | 1.052x |
| 128 | 0.999 ms | 1.101 ms | 0.908x |
| 256 | 4.589 ms | 4.862 ms | 0.944x |

Conclusion:

- A small speedup appears at `num_res=64`.
- Larger tested sizes are slower than the existing PairFormer attention path.
- There is no evidence yet of a robust PairFormer-level acceleration.

## Known Issues

1. cuEquivariance import requires explicit CUDA library precedence.

   The system cuBLAS library is older than the cuEquivariance ops package
   expects. The conda environment contains a compatible
   `nvidia-cublas-cu12==12.9.2.10`, but it must appear before system CUDA
   libraries on `LD_LIBRARY_PATH`.

2. `CueTriangleMultiplication` is not yet numerically equivalent in realistic
   non-zero-output tests.

   Existing smoke coverage was not sufficient because zero final initialization
   can hide output-projection and gating differences. With `final_init='linear'`,
   PairFormer-level probes show large output differences.

3. PairFormer-level acceleration has not been demonstrated.

   The attention-only cuEquivariance path is close numerically but does not
   improve performance for the larger tested sizes.

4. The default model behavior is unchanged.

   Both new flags default to `false`. This is intentional and should remain so
   until correctness and performance are proven.

## Has This Implemented Replacement of PairFormer Modules?

Partially.

The code now supports optional replacement hooks for PairFormer triangle
attention and triangle multiplication through NVIDIA cuEquivariance. However,
the implementation is not yet a complete validated replacement:

- Attention replacement exists and runs on GPU, but does not yet provide stable
  acceleration.
- Triangle multiplication replacement exists and can run in a smoke test, but
  it has unresolved numerical correctness issues.
- The default PairFormer implementation remains active unless the experimental
  flags are explicitly enabled.

## Recommended Next Steps

1. Fix `CueTriangleMultiplication` numerical equivalence.

   Add tests with `final_init='linear'` and non-zero output projections so the
   test suite catches the current mismatch. Compare each internal stage against
   the legacy implementation:

   - input layer norm
   - input projection and gate
   - outgoing/incoming triangle einsum direction
   - center layer norm axis/layout
   - output projection
   - output gate

2. Decide whether cuEquivariance triangle attention is worth keeping.

   It currently does not show clear speedup at `num_res=128` or `num_res=256`.
   Profile kernel timelines before expanding usage.

3. Add a reproducible benchmark target.

   Put the PairFormer benchmark into a tracked script or pytest benchmark marked
   as GPU-only. It should report:

   - shape
   - dtype
   - flags enabled
   - compile time excluded
   - warmup count
   - iteration count
   - mean, median, min
   - max/mean absolute error vs legacy

4. Normalize CUDA library setup.

   Add documentation or environment activation hooks so cuEquivariance loads the
   Python environment's NVIDIA CUDA libraries before system libraries.

5. Expand tests before enabling either flag by default.

   Required coverage:

   - `float32` and `bfloat16`
   - incoming and outgoing triangle multiplication
   - masked and partially masked inputs
   - non-zero final initialization
   - forward numerical comparison
   - input gradient comparison
   - representative PairFormer iteration comparison

6. Keep the new flags defaulted to `false`.

   Do not enable cuEquivariance by default until correctness and performance are
   both demonstrated on representative AlphaFold3 workloads.
