"""Benchmarks PairFormer cuEquivariance adapters against the legacy path."""

from collections.abc import Sequence
import argparse
import os
import statistics
import sys
import time

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from alphafold3.model import model_config
from alphafold3.model.network import modules


_CASES = {
    'legacy': dict(
        use_cue_triangle_multiplication=False,
        use_cue_triangle_attention=False,
    ),
    'attention_only': dict(
        use_cue_triangle_multiplication=False,
        use_cue_triangle_attention=True,
    ),
    'multiplication_only': dict(
        use_cue_triangle_multiplication=True,
        use_cue_triangle_attention=False,
    ),
    'both': dict(
        use_cue_triangle_multiplication=True,
        use_cue_triangle_attention=True,
    ),
}


def _parse_sizes(value: str) -> list[int]:
  return [int(item) for item in value.split(',') if item]


def _make_pairformer(*, use_cue_triangle_multiplication: bool,
                     use_cue_triangle_attention: bool):
  config = modules.PairFormerIteration.Config(
      num_layer=1,
      use_cue_triangle_multiplication=use_cue_triangle_multiplication,
      use_cue_triangle_attention=use_cue_triangle_attention,
  )
  global_config = model_config.GlobalConfig(
      bfloat16='none',
      final_init='linear',
      flash_attention_implementation='triton',
  )

  def forward(act, mask):
    return modules.PairFormerIteration(
        config, global_config, name='pairformer'
    )(act, mask)

  return hk.transform(forward)


def _make_triangle_multiplication(*, use_cue: bool, equation: str):
  config = modules.TriangleMultiplication.Config(equation=equation)
  global_config = model_config.GlobalConfig(
      bfloat16='none',
      final_init='linear',
      flash_attention_implementation='triton',
  )
  module_cls = (
      modules.CueTriangleMultiplication
      if use_cue
      else modules.TriangleMultiplication
  )

  def forward(act, mask):
    return module_cls(config, global_config, name='triangle')(act, mask)

  return hk.transform(forward)


def _make_grid_self_attention(*, use_cue: bool, transpose: bool):
  config = modules.GridSelfAttention.Config(num_head=4)
  global_config = model_config.GlobalConfig(
      bfloat16='none',
      final_init='linear',
      flash_attention_implementation='triton',
  )
  module_cls = modules.CueGridSelfAttention if use_cue else modules.GridSelfAttention

  def forward(act, mask):
    return module_cls(
        config, global_config, transpose=transpose, name='attention'
    )(act, mask)

  return hk.transform(forward)


def _time_apply(apply_fn, params, act, mask, *, warmup: int, iters: int):
  for _ in range(warmup):
    jax.block_until_ready(apply_fn(params, act, mask))

  times = []
  for _ in range(iters):
    start = time.perf_counter()
    out = apply_fn(params, act, mask)
    jax.block_until_ready(out)
    times.append(time.perf_counter() - start)
  return times


def _summarize(times: Sequence[float]) -> tuple[float, float, float]:
  return statistics.mean(times), statistics.median(times), min(times)


def _diff(actual, expected) -> tuple[float, float]:
  diff = np.asarray(actual, dtype=np.float32) - np.asarray(
      expected, dtype=np.float32
  )
  return float(np.max(np.abs(diff))), float(np.mean(np.abs(diff)))


def _print_result(name: str, times: Sequence[float], legacy_mean: float,
                  max_abs: float, mean_abs: float):
  mean, median, minimum = _summarize(times)
  speedup = legacy_mean / mean
  print(
      f'{name:>19} '
      f'mean={mean * 1000:8.3f} ms '
      f'median={median * 1000:8.3f} ms '
      f'min={minimum * 1000:8.3f} ms '
      f'speedup={speedup:6.3f}x '
      f'max_abs={max_abs:.6g} '
      f'mean_abs={mean_abs:.6g}'
  )


def _assert_close_if_requested(
    actual, expected, *, assert_close: bool, rtol: float, atol: float
):
  if not assert_close:
    return
  np.testing.assert_allclose(
      np.asarray(actual),
      np.asarray(expected),
      rtol=rtol,
      atol=atol,
  )


def _bench_transforms(
    transforms, *, act, mask, warmup: int, iters: int, assert_close: bool,
    rtol: float, atol: float
):
  params = transforms['legacy'].init(jax.random.PRNGKey(0), act, mask)
  apply_fns = {
      name: jax.jit(lambda p, x, m, transformed=transformed:
                   transformed.apply(p, None, x, m))
      for name, transformed in transforms.items()
  }

  legacy_out = jax.block_until_ready(apply_fns['legacy'](params, act, mask))
  legacy_times = _time_apply(
      apply_fns['legacy'],
      params,
      act,
      mask,
      warmup=warmup,
      iters=iters,
  )
  legacy_mean, _, _ = _summarize(legacy_times)
  _print_result('legacy', legacy_times, legacy_mean, 0.0, 0.0)

  for name in transforms:
    if name == 'legacy':
      continue
    out = jax.block_until_ready(apply_fns[name](params, act, mask))
    max_abs, mean_abs = _diff(out, legacy_out)
    _assert_close_if_requested(
        out,
        legacy_out,
        assert_close=assert_close,
        rtol=rtol,
        atol=atol,
    )
    times = _time_apply(
        apply_fns[name],
        params,
        act,
        mask,
        warmup=warmup,
        iters=iters,
    )
    _print_result(name, times, legacy_mean, max_abs, mean_abs)


def _run_pairformer_benchmark(args, *, rtol: float, atol: float):
  for num_res in args.sizes:
    print(f'\nnum_res={num_res} channels={args.channels} dtype=float32')
    act = jax.random.normal(
        jax.random.PRNGKey(42 + num_res),
        (num_res, num_res, args.channels),
        dtype=jnp.float32,
    )
    mask = jnp.ones((num_res, num_res), dtype=jnp.float32)

    transforms = {
        name: _make_pairformer(**case_kwargs)
        for name, case_kwargs in _CASES.items()
    }
    _bench_transforms(
        transforms,
        act=act,
        mask=mask,
        warmup=args.warmup,
        iters=args.iters,
        assert_close=args.assert_close,
        rtol=rtol,
        atol=atol,
    )


def _run_module_benchmark(args, *, rtol: float, atol: float):
  for num_res in args.sizes:
    act = jax.random.normal(
        jax.random.PRNGKey(1000 + num_res),
        (num_res, num_res, args.channels),
        dtype=jnp.float32,
    )
    mask = jnp.ones((num_res, num_res), dtype=jnp.float32)

    for equation in ('ikc,jkc->ijc', 'kjc,kic->ijc'):
      print(
          f'\nmodule=triangle_multiplication equation={equation} '
          f'num_res={num_res} channels={args.channels} dtype=float32'
      )
      _bench_transforms(
          {
              'legacy': _make_triangle_multiplication(
                  use_cue=False, equation=equation
              ),
              'cue': _make_triangle_multiplication(
                  use_cue=True, equation=equation
              ),
          },
          act=act,
          mask=mask,
          warmup=args.warmup,
          iters=args.iters,
          assert_close=args.assert_close,
          rtol=rtol,
          atol=atol,
      )

    for transpose in (False, True):
      print(
          f'\nmodule=grid_self_attention transpose={transpose} '
          f'num_res={num_res} channels={args.channels} dtype=float32'
      )
      _bench_transforms(
          {
              'legacy': _make_grid_self_attention(
                  use_cue=False, transpose=transpose
              ),
              'cue': _make_grid_self_attention(
                  use_cue=True, transpose=transpose
              ),
          },
          act=act,
          mask=mask,
          warmup=args.warmup,
          iters=args.iters,
          assert_close=args.assert_close,
          rtol=rtol,
          atol=atol,
      )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--sizes', default='64,128,256', type=_parse_sizes)
  parser.add_argument('--channels', default=128, type=int)
  parser.add_argument('--warmup', default=5, type=int)
  parser.add_argument('--iters', default=30, type=int)
  parser.add_argument(
      '--accuracy-mode', choices=('strict', 'perf'), default='strict'
  )
  parser.add_argument('--rtol', default=None, type=float)
  parser.add_argument('--atol', default=None, type=float)
  parser.add_argument('--assert-close', action='store_true')
  parser.add_argument('--module-bench', action='store_true')
  args = parser.parse_args()
  sys.argv = [sys.argv[0]]
  os.environ['ALPHAFOLD3_CUE_GEMM_MODE'] = args.accuracy_mode
  rtol = args.rtol
  atol = args.atol
  if rtol is None:
    rtol = 5e-3 if args.accuracy_mode == 'strict' else 2e-2
  if atol is None:
    atol = 5e-3 if args.accuracy_mode == 'strict' else 2e-2

  print(f'backend={jax.default_backend()}')
  print(f'devices={jax.devices()}')
  print(
      f'warmup={args.warmup} iters={args.iters} channels={args.channels} '
      f'accuracy_mode={args.accuracy_mode} rtol={rtol} atol={atol}'
  )

  _run_pairformer_benchmark(args, rtol=rtol, atol=atol)
  if args.module_bench:
    _run_module_benchmark(args, rtol=rtol, atol=atol)


if __name__ == '__main__':
  main()
